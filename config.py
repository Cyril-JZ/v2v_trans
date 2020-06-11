import argparse


class Config:
    parser = argparse.ArgumentParser()

    # logger options
    parser.add_argument('--display_size', default=16, type=int,
                        help='How many images do you want to display each time')
    parser.add_argument('--snapshot_save_iter', default=10000, type=int,
                        help='How often do you want to save trained models')
    parser.add_argument('--log_iter', default=1, type=int,
                        help='How often do you want to log the training stats')

    # optimization options
    parser.add_argument('--max_iter', default=1000000, type=int,
                        help='maximum number of training iterations')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--weight_decay', default=0.0001, type=float,
                        help='weight decay')
    parser.add_argument('--beta1', default=0.5, type=float,
                        help='Adam parameter')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='Adam parameter')
    parser.add_argument('--init', default='kaiming', type=str,
                        help='initialization [gaussian/kaiming/xavier/orthogonal]')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--step_size', default=100000, type=int,
                        help='how often to decay learning rate')
    parser.add_argument('--gamma', default=0.5, type=float,
                        help='how much to decay learning rate')
    parser.add_argument('--gan_w', default=1, type=float,
                        help='weight of adversarial loss')
    parser.add_argument('--recon_x_w', default=10, type=float,
                        help='weight of image reconstruction loss')
    parser.add_argument('--recon_s_w', default=1, type=float,
                        help='weight of style reconstruction loss')
    parser.add_argument('--recon_c_w', default=1, type=float,
                        help='weight of content reconstruction loss')
    parser.add_argument('--recon_x_cyc_w', default=10, type=float,
                        help='weight of explicit style augmented cycle consistency loss')
    parser.add_argument('--g_comp', default=10, type=float,
                        help='weight of comp loss for geneartor')

    # model options
    parser.add_argument('--gen_dim', default=64, type=int,
                        help='number of filters in the bottommost layer')
    parser.add_argument('--gen_mlp_dim', default=256, type=int,
                        help='number of filters in MLP')
    parser.add_argument('--gen_style_dim', default=8, type=int,
                        help='length of style code')
    parser.add_argument('--gen_activ', default='relu', type=str,
                        help='activation function [relu/lrelu/prelu/selu/tanh]')
    parser.add_argument('--gen_n_downsample', default=2, type=int,
                        help='number of downsampling layers in content encoder')
    parser.add_argument('--gen_n_res', default=4, type=int,
                        help='number of residual blocks in content encoder/decoder')
    parser.add_argument('--gen_pad_type', default='reflect', type=str,
                        help='padding type [zero/reflect]')
    parser.add_argument('--dis_dim', default=64, type=int,
                        help='number of filters in the bottommost layer')
    parser.add_argument('--dis_norm', default='none', type=str,
                        help='normalization layer [none/bn/in/ln]')
    parser.add_argument('--dis_activ', default='lrelu', type=str,
                        help='activation function [relu/lrelu/prelu/selu/tanh]')
    parser.add_argument('--dis_n_layer', default=4, type=int,
                        help='number of layers in D')
    parser.add_argument('--dis_gan_type', default='lsgan', type=str,
                        help='GAN loss [lsgan/nsgan]')
    parser.add_argument('--dis_num_scales', default=3, type=int,
                        help='number of scales')
    parser.add_argument('--dis_pad_type', default='reflect', type=str,
                        help='padding type [zero/reflect]')

    # data options
    parser.add_argument('--input_dim_a', default=3, type=int,
                        help='number of image channels [1/3]')
    parser.add_argument('--input_dim_b', default=3, type=int,
                        help='number of image channels [1/3]')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='number of data loading threads')
    parser.add_argument('--data_root', default='./data', type=str,
                        help='dataset folder location')

    # others
    parser.add_argument('--output_path', type=str, default='.',
                        help="path for logs, checkpoints, and VGG model weight")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--checkpoint', type=str,
                        help="checkpoint of autoencoders")
    parser.add_argument('--trainer', default='MUNIT', type=str,
                        help="MUNIT|UNIT")
    parser.add_argument('--log_dir', type=str, default='./munit_oilpainting',
                        help="log path")
    parser.add_argument('--model_name', type=str, default='./munit_oilpainting',
                        help='model name')

    # for test
    parser.add_argument('--input_folder', type=str, help="input image folder")
    parser.add_argument('--output_folder', type=str, help="output image folder")
    parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--num_style', type=int, default=10, help="number of styles to sample")
    parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
    parser.add_argument('--output_only', action='store_true',
                        help="whether only save the output images or also save the input images")
    parser.add_argument('--inception_a', type=str, default='.',
                        help="path to the pretrained inception network for domain A")
    parser.add_argument('--inception_b', type=str, default='.',
                        help="path to the pretrained inception network for domain B")