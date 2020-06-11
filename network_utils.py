import torch
import torch.nn as nn


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
