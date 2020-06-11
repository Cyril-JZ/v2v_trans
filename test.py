import os
import torch
import torchvision.utils as vutils
from model import V2VModel
from config import Config
from data import ImageFolder
from utils import get_data_loader_folder
from torch.autograd import Variable


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    # load config
    config = Config().parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load experiment setting
    config.num_style = 1 if config.style != '' else config.num_style
    input_dim = config.input_dim_a if config.a2b else config.input_dim_b
    style_dim = config.gen_style_dim

    # Setup model and data loader
    image_names = ImageFolder(config.input_folder, return_paths=True)
    data_loader = get_data_loader_folder(config.input_folder, 1, False)

    model = V2VModel(config).to(device)
    state_dict = torch.load(config.checkpoint)
    model.gen_a.load_state_dict(state_dict['a'])
    model.gen_b.load_state_dict(state_dict['b'])
    model.eval()
    encode = model.gen_a.encode if config.a2b else model.gen_b.encode  # encode function
    decode = model.gen_b.decode if config.a2b else model.gen_a.decode  # decode function

    # Start testing
    style_fixed = Variable(torch.randn(config.num_style, style_dim, 1, 1).to(device), volatile=True)
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        print(names[1])
        images = Variable(images.to(device), volatile=True)
        content, _ = encode(images)
        style = style_fixed if config.synchronized else Variable(torch.randn(config.num_style, style_dim,
                                                                             1, 1).to(device), volatile=True)
        for j in range(config.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content, s)
            outputs = (outputs + 1) / 2.

            basename = os.path.basename(names[1])
            path = os.path.join(config.output_folder + "_%02d" % j, basename)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)

        if not config.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(config.output_folder, 'input{:03d}.jpg'.format(i)), padding=0,
                              normalize=True)
