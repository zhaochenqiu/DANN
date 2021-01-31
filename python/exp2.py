from loaddata import loadFiles_plus
import numpy as np

import random as rd

import torch
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

    imgblock = np.asarray(imgblock, dtype=np.float32)
    return imgblock


def getDataBlock(vec):
    imgblock = []

    for fs in vec:
        img = imageio.imread(fs)

        imgblock.append(img)


    imgblock = np.asarray(imgblock, dtype=np.float32).transpose(1,2,0,3)
    row, column, frames, byte = imgblock.shape

    imgblock = np.reshape(imgblock, (row*column, frames, byte)  )


    return imgblock




pa_im = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/input/'
ft_im = '.jpg'

pa_gt = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth/'
ft_gt = '.png'

fs_im, fullfs_im = loadFiles_plus(pa_im, ft_im)
fs_gt, fullfs_gt = loadFiles_plus(pa_gt, ft_gt)


left = 0
right = 1147

curidx = 1148

radius = 10

# type(fullfs) = <class 'list'>
vec = randTempSample(fullfs_im, left, right, radius**2)

# print(fullfs_im[right])
# print(fullfs_gt[right])

curimg = np.asarray( imageio.imread(fullfs_im[curidx]), dtype=np.float32)
labimg = np.asarray( imageio.imread(fullfs_gt[curidx]), dtype=np.float32)



imgblock = getImgBlock(vec)

frames, row, column, byte = imgblock.shape

for i in range(frames):
    imgblock[i, :, :, :] = np.abs(imgblock[i, :, :, :] - curimg)/255.0



imgblock = imgblock.transpose(1, 2, 0 ,3)

imgblock = np.reshape(imgblock, (row*column, frames, byte)  )
imglabel = np.reshape(labimg,   (row*column, 1))

print(imgblock.shape)
print(imglabel.shape)


layer1 = nn.Conv1d(100, 1024, 3)
layer2 = nn.Linear(1024, 2)


x1 = torch.tensor(imgblock)

x2 = layer1(x1)
x3 = F.relu(x2)
x4 = x3.view(-1, 1024)
x5 = layer2(x4)

print('x1.shape = ', x1.shape)
print('x2.shape = ', x2.shape)
print('x3.shape = ', x3.shape)
print('x4.shape = ', x4.shape)
print('x5.shape = ', x5.shape)
# m = nn.Conv1d(100, 1, 3)

#
# m = nn.Conv1d(16, 50, 3)
# data = torch.randn(20, 16, 3)
# output = m(data)
#
# print(data.shape)
# print(output.shape)



#
# idx = np.where(imglabel == 255)[0]
#
#
# idx_l = 86040
# idx_r = idx_l + 50
#
# # im = imgblock[idx_l:idx_r, :, :]
# # lb = imglabel[idx_l:idx_r, :]
# #
#
#
#
# im = imgblock[idx[0:40], :, :]
# lb = imglabel[idx[0:40], :]
#
#
# print(idx)
#
# # print(im)
#
# print('im.shape = ', im.shape)
#
# print(imgblock.shape)
# print(imglabel.shape)
# print(lb.shape)
# print(lb)
#
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(im)
# plt.subplot(1, 3, 2)
# plt.imshow(labimg, cmap='gray')
# plt.subplot(1, 3, 3)
# plt.imshow(lb, cmap='gray')
# plt.show()
#
#
#
#
# # left = 0
# # right = 1149
# #
# # vec = randTempSample(fullfs, left, right, 100)
# # print(vec)
# #
# # imgblock = getDataBlock(vec)
# #
# # print("size = ", imgblock.shape)
# #
# # # im = np.squeeze( imgblock[1, :, :] )
# # im = imgblock[10:500, :, :]
# #
# # print(im.shape)
# #
# # plt.figure()
# # plt.imshow(im)
# # plt.show()
