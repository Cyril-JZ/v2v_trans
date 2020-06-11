import torch
import os
import sys
from config import Config
from model import V2VModel
from utils import get_all_data_loaders, prepare_sub_folder, write_loss
from torch.utils.tensorboard import SummaryWriter


def train(it_id):
    while True:
        for idx, (image_a, image_b) in enumerate(zip(train_loader_a, train_loader_b)):
            # print(idx)

            # obtain input image pairs
            image_a = image_a.cuda().detach() if torch.cuda.is_available() else image_a.detach()
            image_b = image_b.cuda().detach() if torch.cuda.is_available() else image_b.detach()

            # Main training code
            model.dis_update(image_a, image_b, config)
            model.gen_update(image_a, image_b, config)

            # Updating lr
            model.dis_scheduler.step()
            model.gen_scheduler.step()

            # Dump training stats in log file
            if (it_id + 1) % config.log_iter == 0:
                write_loss(it_id, model, train_writer)

            # Save network weights
            if (it_id + 1) % config.snapshot_save_iter == 0:
                model.save(checkpoint_directory, it_id)

            it_id += 1
            if it_id + 1 >= max_iter:
                sys.exit('Finish training')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Load config
    config = Config().parser.parse_args()
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    torch.manual_seed(config.seed)
    max_iter = config.max_iter
    display_size = config.display_size

    # Achieve data loader
    train_loader_a, train_loader_b, _, _ = get_all_data_loaders(config)
    train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).to(device)
    train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).to(device)

    # Main models
    model = V2VModel(config).to(device)

    # Setup logger and output folders
    model_name = config.model_name
    train_writer = SummaryWriter(config.log_dir)
    output_directory = os.path.join(config.output_path + '/outputs', model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

    # Start training
    iterations = model.resume(checkpoint_dir=config.checkpoint, hyperparameters=config) if config.resume else 0

    print('Start training')
    train(iterations)
