import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter

import functools


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        self.conv = nn.Sequential(nn.Conv2d(input_nc, 1, kernel_size=kw, stride=2, padding=padw),
                                  nn.Conv2d(1, 1, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                                  nn.Conv2d(1, 1, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                                  nn.Conv2d(1, 1, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                                  nn.Conv2d(1, 1, kernel_size=kw, stride=1, padding=padw))

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.constant_(m.weight.data, 1 / 16.)

        self.apply(init_func)

    def forward(self, inp):
        return self.conv(inp)


def crop_2d(input, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0):
    assert input.dim() == 4, 'only support Input(B, C, W, H)'
    B, C, W, H = input.size()
    return input[:, :,
           crop_left:(W - crop_right),
           crop_bottom:(H - crop_top)]


class Crop2d(nn.Module):
    """
    :params torch.Tensor input: Input(B, C, W, H)
    """

    def __init__(self, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0):
        super(Crop2d, self).__init__()
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

    def forward(self, inp):
        return crop_2d(inp,
                       self.crop_left,
                       self.crop_right,
                       self.crop_top,
                       self.crop_bottom)


class NewInstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(NewInstanceNorm, self).__init__()
        self.epsilon = epsilon
        self.saved_mean = None
        self.saved_std = None
        self.x_max = None
        self.x_min = None
        self.have_expand = False

    def forward(self, inp):

        if not self.have_expand:
            size = inp.size()
            self.saved_mean = self.saved_mean.expand(size)
            self.saved_std = self.saved_std.expand(size)
            self.x_min = self.x_min.expand(size)
            self.x_max = self.x_max.expand(size)
            self.have_expand = True

        x = inp - self.saved_mean
        x = x * self.saved_std
        x = torch.max(self.x_min, x)
        x = torch.min(self.x_max, x)

        return x

    def baseline_forward(self, x):
        mean = torch.mean(x, (2, 3), True)
        x = x - mean
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp, mean, tmp

    def compute_norm(self, inp):
        # mean and var
        self.saved_mean = torch.mean(inp, (0, 2, 3), True)
        x = inp - self.saved_mean
        tmp = torch.mul(x, x)
        self.saved_std = torch.rsqrt(torch.mean(tmp, (0, 2, 3), True) + self.epsilon)
        x = x * self.saved_std

        # max and min
        tmp_max, _ = torch.max(x, 2, True)
        tmp_max, _ = torch.max(tmp_max, 0, True)
        self.x_max, _ = torch.max(tmp_max, 3, True)

        tmp_min, _ = torch.min(x, 2, True)
        tmp_min, _ = torch.min(tmp_min, 0, True)
        self.x_min, _ = torch.min(tmp_min, 3, True)

        self.have_expand = False
        return x


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)  # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(BatchNorm2d, self).__init__()
        self.IN = NewInstanceNorm(epsilon=eps)
        self.ori_IN = InstanceNorm(epsilon=eps)

        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.test = False

    def reshape_weight(self):
        if len(self.weight.shape) == 1:
            self.weight = Parameter(self.weight.data.unsqueeze(1).unsqueeze(2).unsqueeze(3))

    def forward(self, X):
        B, C, H, W = X.shape

        if not self.test:
            X = self.IN.compute_norm(X)
            self.test = True
        else:
            X = self.IN(X)

        if self.affine:
            X = F.conv2d(X, self.weight, self.bias, groups=C)
        return X


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(InstanceNorm2d, self).__init__()
        self.IN = NewInstanceNorm(epsilon=eps)
        self.test = False
        self.max = None
        self.min = None

    def reset(self):
        self.test = False
        self.max = None
        self.min = None

    def forward(self, X):

        if not self.test:
            X = self.IN.compute_norm(X)
            self.max = X.max()
            self.min = X.min()
            self.test = True
        else:
            assert self.max is not None and self.min is not None
            X = self.IN(X)
            X = torch.clamp(X, self.min, self.max)

        return X
