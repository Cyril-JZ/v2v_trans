import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import weights_init, get_model_list, get_scheduler
from comp_loss import TemporalLoss
from torch.autograd import Variable
from modules import Conv2dBlock, ResBlocks, MLP


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # down-sampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero',
                 use_global=False):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            if use_global:
                self.model += [nn.Upsample(scale_factor=2),
                               Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='global_ln', activation=activ,
                                           pad_type=pad_type)]
            else:
                self.model += [nn.Upsample(scale_factor=2),
                               Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


# Generator
class Generator(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, hps, use_global=False):
        super(Generator, self).__init__()
        dim = hps.gen_dim
        style_dim = hps.gen_style_dim
        n_downsample = hps.gen_n_downsample
        n_res = hps.gen_n_res
        activ = hps.gen_activ
        pad_type = hps.gen_pad_type
        mlp_dim = hps.gen_mlp_dim

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        if use_global:
            self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'global_in', activ,
                                              pad_type=pad_type)
            self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='global_adain',
                               activ=activ, pad_type=pad_type, use_global=use_global)
        else:
            self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
            self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain',
                               activ=activ, pad_type=pad_type, use_global=use_global)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params


# Discriminator
class Discriminator(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, hps):
        super(Discriminator, self).__init__()
        self.n_layer = hps.dis_n_layer
        self.gan_type = hps.dis_gan_type
        self.dim = hps.dis_dim
        self.norm = hps.dis_norm
        self.activ = hps.dis_activ
        self.num_scales = hps.dis_num_scales
        self.pad_type = hps.dis_pad_type
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss_fake = 0
        loss_real = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss_fake += torch.mean((out0 - 0) ** 2)
                loss_real += torch.mean((out1 - 1) ** 2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss_fake += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0))
                loss_real += torch.mean(F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss_fake, loss_real

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1) ** 2)  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


class V2VModel(nn.Module):
    def __init__(self, hps, use_global=False):
        super(V2VModel, self).__init__()

        # Model config
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.style_dim = hps.gen_style_dim

        # Initialization
        self.gen_a = Generator(hps.input_dim_a, hps, use_global)  # auto-encoder for domain a
        self.gen_b = Generator(hps.input_dim_b, hps, use_global)  # auto-encoder for domain b
        self.dis_a = Discriminator(hps.input_dim_a, hps)  # discriminator for domain a
        self.dis_b = Discriminator(hps.input_dim_b, hps)  # discriminator for domain b

        # Setup the optimizers
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_optimizer = optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=hps.lr, betas=(hps.beta1, hps.beta2), weight_decay=hps.weight_decay)
        self.gen_optimizer = optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=hps.lr, betas=(hps.beta1, hps.beta2), weight_decay=hps.weight_decay)
        self.dis_scheduler = get_scheduler(self.dis_optimizer, hps)
        self.gen_scheduler = get_scheduler(self.gen_optimizer, hps)

        if hps.g_comp > 0:
            self.temp_loss = TemporalLoss()

        # Network weight initialization
        self.apply(weights_init(hps.init))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

    def recons_loss(self, inp, target):
        return torch.mean(torch.abs(inp - target))

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

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # reconstruction loss (x, s, c, cycle)
        self.loss_gen_recon_x_a = self.recons_loss(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recons_loss(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recons_loss(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recons_loss(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recons_loss(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recons_loss(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recons_loss(x_aba, x_a) if hps.recon_x_cyc_w > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recons_loss(x_bab, x_b) if hps.recon_x_cyc_w > 0 else 0

        # G temp loss
        self.loss_gen_temp_a = self.temp_loss(x_a_recon, second_x_a_recon, flow_a) if hps.g_comp > 0 else 0
        self.loss_gen_temp_aba = self.temp_loss(x_aba, second_x_aba, flow_a) if hps.g_comp > 0 else 0
        self.loss_gen_temp_b = self.temp_loss(x_b_recon, second_x_b_recon, flow_b) if hps.g_comp > 0 else 0
        self.loss_gen_temp_bab = self.temp_loss(x_bab, second_x_bab, flow_b) if hps.g_comp > 0 else 0

        # total loss
        self.loss_gen_total = (
                hps.gan_w * (self.loss_gen_adv_a + self.loss_gen_adv_b) +
                hps.recon_x_w * (self.loss_gen_recon_x_a + self.loss_gen_recon_x_b) +
                hps.recon_s_w * (self.loss_gen_recon_s_a + self.loss_gen_recon_s_b) +
                hps.recon_c_w * (self.loss_gen_recon_c_a + self.loss_gen_recon_c_b) +
                hps.recon_x_cyc_w * (self.loss_gen_cycrecon_x_a + self.loss_gen_cycrecon_x_b) +
                hps.g_comp * (
                        self.loss_gen_temp_a + self.loss_gen_temp_aba + self.loss_gen_temp_b + self.loss_gen_temp_bab)
        )

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def dis_update(self, x_a, x_b, hps):
        self.dis_opt.zero_grad()

        # sampled style code
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
        self.loss_dis_total = hps.gan_w * (self.loss_dis_a + self.loss_dis_b)

        self.loss_dis_total.backward()
        self.dis_opt.step()

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
