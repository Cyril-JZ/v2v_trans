import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.utils as vutils
from trainer import Trainer
from config import Config
from data import ImageFolder
from utils import load_inception, get_data_loader_folder
from scipy.stats import entropy
from torch.autograd import Variable


if __name__ == '__main__':
    config = Config().parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load experiment setting
    config.num_style = 1 if config.style != '' else config.num_style
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    input_dim = config.input_dim_a if config.a2b else config.input_dim_b

    # Setup model and data loader
    image_names = ImageFolder(config.input_folder, transform=None, return_paths=True)
    data_loader = get_data_loader_folder(config.input_folder, 1, False, new_size=256, crop=False)

    config.vgg_model_path = config.output_path
    style_dim = config.gen_style_dim
    trainer = Trainer(config).to(device)

    state_dict = torch.load(config.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
    trainer.eval()
    encode = trainer.gen_a.encode if config.a2b else trainer.gen_b.encode  # encode function
    decode = trainer.gen_b.decode if config.a2b else trainer.gen_a.decode  # decode function

    if config.compute_IS:
        IS = []
        all_preds = []
    if config.compute_CIS:
        CIS = []

    # Start testing
    style_fixed = Variable(torch.randn(config.num_style, style_dim, 1, 1).to(device), volatile=True)
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        if config.compute_CIS:
            cur_preds = []
        print(names[1])
        images = Variable(images.to(device), volatile=True)
        content, _ = encode(images)
        style = style_fixed if config.synchronized else Variable(torch.randn(config.num_style, style_dim,
                                                                             1, 1).to(device), volatile=True)
        for j in range(config.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content, s)
            outputs = (outputs + 1) / 2.
            if config.compute_IS or config.compute_CIS:
                pred = f.softmax(inception(inception_up(outputs)),
                                 dim=1).cpu().data.numpy()  # get the predicted class distribution
            if config.compute_IS:
                all_preds.append(pred)
            if config.compute_CIS:
                cur_preds.append(pred)
            # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
            basename = os.path.basename(names[1])
            path = os.path.join(config.output_folder + "_%02d" % j, basename)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        if config.compute_CIS:
            cur_preds = np.concatenate(cur_preds, 0)
            py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
            for j in range(cur_preds.shape[0]):
                pyx = cur_preds[j, :]
                CIS.append(entropy(pyx, py))
        if not config.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(config.output_folder, 'input{:03d}.jpg'.format(i)), padding=0,
                              normalize=True)
    if config.compute_IS:
        all_preds = np.concatenate(all_preds, 0)
        py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
        for j in range(all_preds.shape[0]):
            pyx = all_preds[j, :]
            IS.append(entropy(pyx, py))

    if config.compute_IS:
        print("Inception Score: {}".format(np.exp(np.mean(IS))))
    if config.compute_CIS:
        print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))