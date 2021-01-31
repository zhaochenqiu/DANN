from loaddata import loadFiles_plus
import numpy as np

import random as rd

# import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

import imageio

import matplotlib.pyplot as plt


np.set_printoptions(threshold=np.inf)



def randTempSample(vector, left, right, num):

    vec = vector[left:right]

    rd.shuffle(vec)
    re_vec = []

    length = len(vec)

#    python 格式化输出
#    print('len = %d' % len(vec))

    for i in range(num):
        idx = i%length
        re_vec.append(vec[idx])

    return re_vec


class EZNet(nn.Module):
    def __init__(self):
        super(EZNet, self).__init__()


        self.conv1 = nn.Conv2d(3, 1024, 15)
        self.fc1 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 1024)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


def getImgBlock(vec):

    imgblock = []

    for fs in vec:
        img = imageio.imread(fs)

        imgblock.append(img)

    imgblock = np.asarray(imgblock)
    return imgblock


def getDataBlock(vec):
    imgblock = []

    for fs in vec:
        img = imageio.imread(fs)

        imgblock.append(img)


    imgblock = np.asarray(imgblock).transpose(1,2,0,3)
    row, column, frames, byte = imgblock.shape

    imgblock = np.reshape(imgblock, (row*column, frames, byte)  )


    return imgblock




pa_im = '/home/cqzhao/dataset/dataset2014/dataset/baseline/highway/input/'
ft_im = '.jpg'

fs, fullfs = loadFiles_plus(pa_im, ft_im)


left = 0
right = 1149

# type(fullfs) = <class 'list'>
vec = randTempSample(fullfs, left, right, 100)
print(vec)

imgblock = getDataBlock(vec)

print("size = ", imgblock.shape)

# im = np.squeeze( imgblock[1, :, :] )
im = imgblock[10:500, :, :]

print(im.shape)

plt.figure()
plt.imshow(im)
plt.show()
