import torch
import torch.utils.data as data
import cv2
import random
from glob import glob


class ImageFolder(data.Dataset):
    def __init__(self, root, return_paths=False, train=False):
        self.imgs = glob(root + '/*.*')
        self.return_paths = return_paths
        if train:
            self.loadSize = 288
            self.Flip = True
        else:
            self.loadSize = 256
            self.Flip = False
        self.fineSize = 256

    def ProcessImg(self, img):
        ''' Given an image with channel [BGR] which values [0,255]
            The output values [-1,1]
        '''

        x1 = random.randint(0, self.loadSize - self.fineSize)
        y1 = random.randint(0, self.loadSize - self.fineSize)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.loadSize, self.loadSize))
        img = img[x1:x1 + self.fineSize, y1:y1 + self.fineSize, :]

        if self.Flip and random.random() <= 0.5:
            img = cv2.flip(img, 1)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        img = img.div_(255.0)
        img = (img - 0.5) * 2
        return img

    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index])
        img = self.ProcessImg(img)

        if self.return_paths:
            return img, self.imgs[index]
        else:
            return img

    def __len__(self):
        return len(self.imgs)
