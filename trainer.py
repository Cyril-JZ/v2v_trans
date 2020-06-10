import torch
import torch.nn as nn
import os

from networks import AdaINGen, MsImageDis
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from comp_loss import TemporalLoss
from torch.autograd import Variable
from networks import Vgg16


class Trainer(nn.Module):
    def __init__(self, hps, use_global=False):
        super(Trainer, self).__init__()
        # Initiate the networks
        self.gen_a = AdaINGen(hps.input_dim_a, hps, use_global)  # auto-encoder for domain a
        self.gen_b = AdaINGen(hps.input_dim_b, hps, use_global)  # auto-encoder for domain b
        self.dis_a = MsImageDis(hps.input_dim_a, hps)  # discriminator for domain a
        self.dis_b = MsImageDis(hps.input_dim_b, hps)  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hps.gen_style_dim

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # fix the noise used in sampling
        self.s_a = nn.Parameter(torch.randn(hps.display_size, self.style_dim, 1, 1).to(self.device))
        self.s_b = nn.Parameter(torch.randn(hps.display_size, self.style_dim, 1, 1).to(self.device))

        # Setup the optimizers
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=hps.lr, betas=(hps.beta1, hps.beta2), weight_decay=hps.weight_decay)
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=hps.lr, betas=(hps.beta1, hps.beta2), weight_decay=hps.weight_decay)
        self.dis_scheduler = get_scheduler(self.dis_opt, hps)
        self.gen_scheduler = get_scheduler(self.gen_opt, hps)

        # Network weight initialization
        self.apply(weights_init(hps.init))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if hps.vgg_w > 0:
            self.vgg = Vgg16()
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        if hps.g_comp > 0 or hps.d_comp > 0:
            self.temp_loss = TemporalLoss()

    def recon_criterion(self, inp, target):
        return torch.mean(torch.abs(inp - target))

    def forward(self, x_a, x_b):
        self.eval()
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, self.s_a)
        x_ab = self.gen_b.decode(c_a, self.s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hps):
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hps.recon_x_cyc_w > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hps.recon_x_cyc_w > 0 else None

        if hps.g_comp > 0:
            second_x_a, flow_a = self.temp_loss.GenerateFakeData(x_a)
            second_x_b, flow_b = self.temp_loss.GenerateFakeData(x_b)

            # encode
            second_c_a, second_s_a_prime = self.gen_a.encode(second_x_a)
            second_c_b, second_s_b_prime = self.gen_b.encode(second_x_b)
            # decode (within domain)
            second_x_a_recon = self.gen_a.decode(second_c_a, second_s_a_prime)
            second_x_b_recon = self.gen_b.decode(second_c_b, second_s_b_prime)
            # decode (cross domain)
            second_x_ba = self.gen_a.decode(second_c_b, s_a)
            second_x_ab = self.gen_b.decode(second_c_a, s_b)
            # encode again
            second_c_b_recon, _ = self.gen_a.encode(second_x_ba)
            second_c_a_recon, _ = self.gen_b.encode(second_x_ab)
            # decode again (if needed)
            second_x_aba = self.gen_a.decode(second_c_a_recon, second_s_a_prime)
            second_x_bab = self.gen_b.decode(second_c_b_recon, second_s_b_prime)

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hps.recon_x_cyc_w > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hps.recon_x_cyc_w > 0 else 0

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hps.vgg_w > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hps.vgg_w > 0 else 0

        # G temp loss
        self.loss_gen_temp_a = self.temp_loss(x_a_recon, second_x_a_recon, flow_a) if hps.g_comp > 0 else 0
        self.loss_gen_temp_aba = self.temp_loss(x_aba, second_x_aba, flow_a) if hps.g_comp > 0 else 0
        self.loss_gen_temp_b = self.temp_loss(x_b_recon, second_x_b_recon, flow_b) if hps.g_comp > 0 else 0
        self.loss_gen_temp_bab = self.temp_loss(x_bab, second_x_bab, flow_b) if hps.g_comp > 0 else 0

        # total loss
        self.loss_gen_total = hps.gan_w * self.loss_gen_adv_a + \
                              hps.gan_w * self.loss_gen_adv_b + \
                              hps.recon_x_w * self.loss_gen_recon_x_a + \
                              hps.recon_s_w * self.loss_gen_recon_s_a + \
                              hps.recon_c_w * self.loss_gen_recon_c_a + \
                              hps.recon_x_w * self.loss_gen_recon_x_b + \
                              hps.recon_s_w * self.loss_gen_recon_s_b + \
                              hps.recon_c_w * self.loss_gen_recon_c_b + \
                              hps.recon_x_cyc_w * self.loss_gen_cycrecon_x_a + \
                              hps.recon_x_cyc_w * self.loss_gen_cycrecon_x_b + \
                              hps.vgg_w * self.loss_gen_vgg_a + \
                              hps.vgg_w * self.loss_gen_vgg_b + \
                              hps.g_comp * self.loss_gen_temp_a + \
                              hps.g_comp * self.loss_gen_temp_aba + \
                              hps.g_comp * self.loss_gen_temp_b + \
                              hps.g_comp * self.loss_gen_temp_bab

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()

        if torch.cuda.is_available():
            s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
            s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        else:
            s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1))
            s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1))

        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, self.s_a[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, self.s_b[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, hps):
        self.dis_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))

        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        # D loss
        loss_dis_a_fake, loss_dis_a_real = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        loss_dis_b_fake, loss_dis_b_real = self.dis_a.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_a = loss_dis_a_fake + loss_dis_a_real
        self.loss_dis_b = loss_dis_b_fake + loss_dis_b_real

        if hps.d_comp > 0:
            second_x_a, flow_a = self.temp_loss.GenerateFakeData(x_a)
            second_x_b, flow_b = self.temp_loss.GenerateFakeData(x_b)
            # encode
            second_c_a, _ = self.gen_a.encode(second_x_a)
            second_c_b, _ = self.gen_b.encode(second_x_b)
            # decode (cross domain)
            second_x_ba = self.gen_a.decode(second_c_b, s_a)
            second_x_ab = self.gen_b.decode(second_c_a, s_b)

            second_loss_dis_a_fake, second_loss_dis_a_real = self.dis_a.calc_dis_loss(second_x_ba.detach(), second_x_a)
            second_loss_dis_b_fake, second_loss_dis_b_real = self.dis_a.calc_dis_loss(second_x_ab.detach(), second_x_b)

        self.loss_dis_comp_a = self.recon_criterion(second_loss_dis_a_fake, loss_dis_a_fake) if hps.d_comp > 0 else 0
        self.loss_dis_comp_b = self.recon_criterion(second_loss_dis_b_fake, loss_dis_b_fake) if hps.d_comp > 0 else 0

        self.loss_dis_total = hps.gan_w * self.loss_dis_a + \
                              hps.gan_w * self.loss_dis_b + \
                              hps.d_comp * self.loss_dis_comp_a + \
                              hps.d_comp * self.loss_dis_comp_b

        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hps):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name, map_location=lambda storage, loc: storage)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name, map_location=lambda storage, loc: storage)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'), map_location=lambda storage, loc: storage)
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hps, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hps, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
