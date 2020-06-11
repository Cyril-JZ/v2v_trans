import os
import math
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from data import ImageFolder


def get_data_loader_folder(input_folder, batch_size, train, num_workers=4):
    dataset = ImageFolder(input_folder, train=train)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


def get_all_data_loaders(conf):
    train_loader_a = get_data_loader_folder(os.path.join(conf.data_root, 'trainA'), conf.batch_size, True,
                                            conf.num_workers)
    test_loader_a = get_data_loader_folder(os.path.join(conf.data_root, 'testA'), conf.batch_size, False,
                                           conf.num_workers)
    train_loader_b = get_data_loader_folder(os.path.join(conf.data_root, 'trainB'), conf.batch_size, True,
                                            conf.num_workers)
    test_loader_b = get_data_loader_folder(os.path.join(conf.data_root, 'testB'), conf.batch_size, False,
                                           conf.num_workers)

    return train_loader_a, train_loader_b, test_loader_a, test_loader_b


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and (
                       'loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hps, iterations=-1):
    return lr_scheduler.StepLR(optimizer, step_size=hps.step_size, gamma=hps.gamma,
                               last_epoch=iterations)


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun
