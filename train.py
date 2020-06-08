import torch
import os
import sys
from config import Config
from trainer import Trainer
from utils import get_all_data_loaders, prepare_sub_folder, write_loss, write_2images, write_html
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    config = Config().parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    torch.manual_seed(config.seed)
    max_iter = config.max_iter
    display_size = config.display_size
    config.vgg_model_path = config.output_path

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    trainer = Trainer(config).to(device)
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

    train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).to(device)
    train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).to(device)
    test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).to(device)
    test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).to(device)

    # Setup logger and output folders
    model_name = config.model_name
    train_writer = SummaryWriter(config.log_dir)
    output_directory = os.path.join(config.output_path + '/outputs', model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

    # Start training
    iterations = trainer.resume(checkpoint_dir=config.checkpoint, hyperparameters=config) if config.resume else 0

    print('Start training')
    while True:
        for it_idx, (image_a, image_b) in enumerate(zip(train_loader_a, train_loader_b)):
            print(it_idx)

            if torch.cuda.is_available():
                image_a, image_b = image_a.cuda().detach(), image_b.cuda().detach()
            else:
                image_a, image_b = image_a.detach(), image_b.detach()

            # Main training code
            trainer.dis_update(image_a, image_b, config)
            trainer.gen_update(image_a, image_b, config)
            trainer.update_learning_rate()

            # Dump training stats in log file
            if (iterations + 1) % config.log_iter == 0:
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) % config.image_save_iter == 0:
                with torch.no_grad():
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))

                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

            if (iterations + 1) % config.image_display_iter == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights
            if (iterations + 1) % config.snapshot_save_iter == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')
