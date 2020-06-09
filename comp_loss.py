import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(x, flo, padding_mode='border', mode='nearest'):
    B, C, H, W = x.size()

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid - flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, padding_mode=padding_mode, mode=mode)
    return output


class TemporalLoss(nn.Module):
    def __init__(self, data_sigma=True, data_w=True, noise_level=0.001):
        super(TemporalLoss, self).__init__()
        self.MSE = torch.nn.MSELoss()

        self.data_sigma = data_sigma
        self.data_w = data_w
        self.noise_level = noise_level

    """
        Flow should have most values in the range of [-1, 1]. 
        For example, values x = -1, y = -1 is the left-top pixel of input, 
        and values  x = 1, y = 1 is the right-bottom pixel of input.
        Flow should be from pre_frame to cur_frame
    """

    def GaussianNoise(self, ins, mean=0, stddev=0.001):
        stddev = stddev + random.random() * stddev
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        if ins.is_cuda:
            noise = noise.cuda()
        return ins + noise

    def GenerateFakeFlow(self, height, width):
        ''' height: img.shape[0], width: img.shape[1]. '''

        flow = np.random.normal(0, scale=5, size=[height // 100, width // 100, 2])
        flow = cv2.resize(flow, (width, height))
        flow[:, :, 0] += random.randint(-10, 10)
        flow[:, :, 1] += random.randint(-10, 10)
        flow = cv2.blur(flow, (100, 100))
        return torch.from_numpy(flow.transpose((2, 0, 1))).float()

    def GenerateFakeData(self, first_frame):
        ''' Input should be a (H,W,3) numpy, with value range [0,1]. '''

        if self.data_w:
            forward_flow = self.GenerateFakeFlow(first_frame.shape[2], first_frame.shape[3])
            if first_frame.is_cuda:
                forward_flow = forward_flow.cuda()
            forward_flow = forward_flow.expand(first_frame.shape[0], 2, first_frame.shape[2], first_frame.shape[3])
            second_frame = warp(first_frame, forward_flow)
        else:
            second_frame = first_frame.clone()
            forward_flow = None

        if self.data_sigma:
            second_frame = self.GaussianNoise(second_frame, stddev=self.noise_level)

        return second_frame, forward_flow

    def forward(self, first_frame, second_frame, forward_flow):
        if self.data_w:
            first_frame = warp(first_frame, forward_flow)
        temporalloss = torch.mean(torch.abs(first_frame - second_frame))

        # L2 Sqrt
        # self.MSE(fake_pre_frame, pre_frame_) ** 0.5

        # # L1
        # temporalloss = torch.mean(torch.abs(fake_pre_frame - pre_frame_))

        # # L2
        # temporalloss = self.MSE(fake_pre_frame, pre_frame_)
        return temporalloss


if __name__ == '__main__':
    TemporalLoss = TemporalLoss()
