import sys
sys.path.append("/home/cqzhao/projects/matrix/")
sys.path.append("../../")


import os


import matplotlib.pyplot as plt


import time


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt


import numpy as np

from common_py.dataIO import loadImgs_pytorch
from common_py.evaluationBS import evaluation_numpy
from common_py.evaluationBS import evaluation_numpy_entry
from common_py.dataIO import saveImg

from common_py.dataIO import readImg_byFilesIdx
from common_py.dataIO import loadFiles_plus

from function.prodis import getEmptyCF
from function.prodis import getHist_plus
from function.prodis import getEmptyCF_plus
from function.prodis import ProductDis_plus
from function.prodis import ProductDis_multi
from function.prodis import ProductDis_multiW
from function.prodis import ProductDis_fast
from function.prodis import getRandCF
from function.prodis import getRandCF_plus
from function.prodis import ProductDis_test


from function.prodis import DifferentiateDis_multi


import imageio


from torch.autograd import Variable

np.set_printoptions(threshold=np.inf)



# print("Hello World")
#
#
# showHello()
#


# def video2tpixels(imgs, curidx):
#
#     return None

def getDisData():
    pa_im = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/input'
#    pa_im = 'D:/dataset/dataset2014/dataset/dynamicBackground/fountain01/input'
    ft_im = 'jpg'

    pa_gt = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth'
#    pa_gt = 'D:/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth'
    ft_gt = 'png'


    imgs = loadImgs_pytorch(pa_im, ft_im)
    labs = loadImgs_pytorch(pa_gt, ft_gt)


    frame, row, column, byte = imgs.shape

    curidx = 1140

    im = imgs[curidx]
    lb = labs[curidx]


    c_L, f_L = getEmptyCF(-255, 255, 1)
    len_hist = torch.numel(f_L)

    hist_data = torch.empty([row*column, len_hist , byte])
    labs_data = torch.empty(row*column)

    cnt = 0



    for r in range(row):
        for c in range(column):
            vec = imgs[:, r, c, :].squeeze()
            val = im[r, c, :]

            sub = vec - val

            for b in range(byte):
#                c_I, f_I = getHist_plus(sub[:, b], 1, -255, 255)
                hist_data[cnt, :, b] = torch.histc(sub[:, b], 511, -255, 255)

            labs_data[cnt] = lb[r, c]


            cnt = cnt + 1


        print("r = ", r, " ", row)



    # randomly permutate the input

    idx = torch.randperm(row*column)

    hist_data = hist_data[idx]
    labs_data = labs_data[idx]




    idx_fg = labs_data == 255
    idx_bk = labs_data == 0


    hist_fg = hist_data[idx_fg]
    hist_bk = hist_data[idx_bk]

    labs_fg = labs_data[idx_fg]
    labs_bk = labs_data[idx_bk]





    data_fg = hist_fg[0:100, :, 0].squeeze()
    data_bk = hist_bk[0:100, :, 0].squeeze()



    data = torch.cat((data_fg, data_bk), dim = 0)
    labs = torch.zeros(200)
    labs[0:100] = labs[0:100] + 1


    num_labs = torch.numel(labs)

    idx = torch.randperm(torch.numel(labs))

    labs = labs[idx]
    data = data[idx]



    return data, labs




# def getNormalData
def getNormalData(imgs, labs, curidx):


    frame, row, column, byte = imgs.shape


    im = imgs[curidx]
    lb = labs[curidx]


    # 使用这种方式可以保证和getNormalData_old 完全一样
#     im = im/255.0
#     imgs = imgs/255.0


    imgs_vec = imgs - im

    imgs_vec = imgs_vec/255.0
    imgs_vec = imgs_vec.reshape(frame, row*column, byte)
    imgs_vec = imgs_vec.permute(1, 0, 2)


    c_L, f_L = getEmptyCF(-1, 1, 0.01)
    len_hist = torch.numel(f_L)

    hist_data = torch.empty([row*column, len_hist , byte])
    labs_data = lb.reshape(row*column)


    for i in range(row*column):
        for b in range(byte):
            hist_data[i, :, b] = torch.histc(imgs_vec[i, :, b], 201, -1, 1)/(frame*1.0)



    return hist_data, labs_data







# def getNormalData
def getNormalData_old(imgs, labs, curidx):

#    imgs = loadImgs_pytorch(pa_im, ft_im)
#    labs = loadImgs_pytorch(pa_gt, ft_gt)


    frame, row, column, byte = imgs.shape

#    curidx = 1140

    im = imgs[curidx]
    lb = labs[curidx]


    c_L, f_L = getEmptyCF(-1, 1, 0.01)
    len_hist = torch.numel(f_L)

    hist_data = torch.empty([row*column, len_hist , byte])
    labs_data = torch.empty(row*column)

    cnt = 0





    for r in range(row):
        for c in range(column):
            vec = imgs[:, r, c, :].squeeze()/255.0
            val = im[r, c, :]/255.0


            sub = vec - val

            for b in range(byte):
#                c_I, f_I = getHist_plus(sub[:, b], 1, -255, 255)
                hist_data[cnt, :, b] = torch.histc(sub[:, b], 201, -1, 1)/(frame * 1.0)

            labs_data[cnt] = lb[r, c]

            cnt = cnt + 1



        print("r = ", r, " ", row)


    return hist_data, labs_data



# def getTempData(pa_im, ft_im, pa_gt, ft_gt):
#     imgs = loadImgs_pytorch(pa_im, ft_im)
#     labs = loadImgs_pytorch(pa_gt, ft_gt)
#
#
#     frame, row, column, byte = imgs.shape
#
# #    curidx = 1140
#
#
#     c_L, f_L = getEmptyCF(-1, 1, 0.01)
#     len_hist = torch.numel(f_L)
#
#     hist_data = torch.empty([row*column, len_hist , byte])
#     labs_data = torch.empty(row*column)
#
#     cnt = 0
#
#
#
#     for r in range(row):
#         for c in range(column):
#             vec = imgs[:, r, c, :].squeeze()/255.0
#
#
#
#             for b in range(byte):
# #                c_I, f_I = getHist_plus(sub[:, b], 1, -255, 255)
#                 hist_data[cnt, :, b] = torch.histc(vec[:, b], 201, -1, 1)/(frame * 1.0)
#
#             cnt = cnt + 1
#
#
#         print("r = ", r, " ", row)
#
#
#     return hist_data, imgs, labs





def getHistData(pa_im, ft_im, pa_gt, ft_gt, curidx):

    imgs = loadImgs_pytorch(pa_im, ft_im)
    labs = loadImgs_pytorch(pa_gt, ft_gt)


    frame, row, column, byte = imgs.shape

#    curidx = 1140

    im = imgs[curidx]
    lb = labs[curidx]


    c_L, f_L = getEmptyCF(-255, 255, 1)
    len_hist = torch.numel(f_L)

    hist_data = torch.empty([row*column, len_hist , byte])
    labs_data = torch.empty(row*column)

    cnt = 0



    for r in range(row):
        for c in range(column):
            vec = imgs[:, r, c, :].squeeze()
            val = im[r, c, :]

            sub = vec - val

            for b in range(byte):
#                c_I, f_I = getHist_plus(sub[:, b], 1, -255, 255)
                hist_data[cnt, :, b] = torch.histc(sub[:, b], 511, -255, 255)

            labs_data[cnt] = lb[r, c]

            cnt = cnt + 1


        print("r = ", r, " ", row)


    return hist_data, labs_data





def getVideoData():
    pa_im = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/input'
    pa_im = 'D:/dataset/dataset2014/dataset/dynamicBackground/fountain01/input'
    ft_im = 'jpg'

    pa_gt = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth'
    pa_gt = 'D:/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth'
    ft_gt = 'png'


    imgs = loadImgs_pytorch(pa_im, ft_im)
    labs = loadImgs_pytorch(pa_gt, ft_gt)


    frame, row, column, byte = imgs.shape

    curidx = 1140

    im = imgs[curidx]
    lb = labs[curidx]


    c_L, f_L = getEmptyCF(-255, 255, 1)
    len_hist = torch.numel(f_L)

    hist_data = torch.empty([row*column, len_hist , byte])
    labs_data = torch.empty(row*column)

    cnt = 0



    for r in range(row):
        for c in range(column):
            vec = imgs[:, r, c, :].squeeze()
            val = im[r, c, :]

            sub = vec - val

            for b in range(byte):
#                c_I, f_I = getHist_plus(sub[:, b], 1, -255, 255)
                hist_data[cnt, :, b] = torch.histc(sub[:, b], 511, -255, 255)

            labs_data[cnt] = lb[r, c]

            cnt = cnt + 1


        print("r = ", r, " ", row)


    return hist_data, labs_data




def balanceData_plus(data, labs):

    idx_fg = labs == 255
    idx_bk = labs == 0


    data_fg = data[idx_fg]
    data_bk = data[idx_bk]

    labs_fg = labs[idx_fg]
    labs_bk = labs[idx_bk]


    num_fg = torch.numel(labs_fg)
    num_bk = torch.numel(labs_bk)

    value = round(num_bk/num_fg)


#    print(data_fg.shape)
    data_fg = data_fg.repeat(value, 1, 1)
    labs_fg = labs_fg.repeat(value)


    re_data = torch.cat( (data_fg, data_bk), dim = 0)
    re_labs = torch.cat( (labs_fg, labs_bk), dim = 0)

    return re_data, re_labs






def balanceData(data, labs):

    idx_fg = labs == 255
    idx_bk = labs == 0


    data_fg = data[idx_fg]
    data_bk = data[idx_bk]

    labs_fg = labs[idx_fg]
    labs_bk = labs[idx_bk]


    num_fg, len_fg = data_fg.shape
    num_bk, len_bk = data_bk.shape



#     print("borderline ----------------")
#     print(data_fg.shape)
#     print(data_bk.shape)
#     print("borderline ----------------")


    # 一切求快，之后再改
    value = round(num_bk/num_fg)

    data_fg = data_fg.repeat(value, 1)
    labs_fg = labs_fg.repeat(value)


#     print("data_fg.shape:", data_fg.shape)
#     print("data_bk.shape:", data_bk.shape)
#
#     print("labs_fg.shape:", labs_fg.shape)
#     print("labs_bk.shape:", labs_bk.shape)


    re_data = torch.cat( (data_fg, data_bk), dim = 0)
    re_labs = torch.cat( (labs_fg, labs_bk), dim = 0)





    return re_data, re_labs



def getTensorOptim(var, learning_rate):

    var = Variable(var, requires_grad = True)

    return optim.Adam([var], lr = learning_rate, amsgrad = True)



class ClassNet(nn.Module):

    def __init__(self, input_size=1000, hidden_size=2000):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)


        self.hidden_size = hidden_size


    def forward(self, input):
        x = F.relu(self.fc1(input))

        x = x.view(-1, self.hidden_size)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)



class ClassifyNetwork(nn.Module):
    def __init__(self, dis_num):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 20, (dis_num, 1) )
        self.conv2 = nn.Conv2d(20, 2000, (1, 201))

        self.fc1 = nn.Linear(2000, 2)


    def forward(self, input):

        x = F.relu( self.conv2(self.conv1(input)) )
        x = x.view(-1, 2000)
        x = self.fc1(x)


        return F.log_softmax(x, dim = 1)




# def train(data_vid, labs_vid, batchsize, device, network, loss_func, prodis_mul,
#           c_data, c_W, f_W_R, f_W_G, f_W_B, c_Z, delta, params,
#           optim_W_R, optim_W_G, optim_W_B, optim_net, net_pa, num_epoch):
#

def train(data_vid, labs_vid, batchsize, device, network, loss_func, prodis_mul,
          c_data, c_W, f_W_R, f_W_G, f_W_B, c_B, f_B_R, f_B_G, f_B_B, c_Z, delta, params,
          optim_W_R, optim_W_G, optim_W_B, optim_B_R, optim_B_G, optim_B_B,
          optim_net, net_pa, num_epoch):


    LEN_DATA = torch.numel(labs_vid)

    value = round(LEN_DATA/batchsize)


    for epoch in range(num_epoch):

        idx = torch.randperm(LEN_DATA)

        data_vid = data_vid[idx]
        labs_vid = labs_vid[idx]


        for i in range(value):
            left = i*batchsize
            right = (i + 1)*batchsize


            data = data_vid[left:right]
            labs = labs_vid[left:right]

            data = data.to(device)
            labs = labs.to(device, dtype = torch.int64)



            c_Z_R, f_Z_W_R = prodis_mul(c_data, data[:, :, 0], c_W, f_W_R, c_Z, delta, params)
            c_Z_G, f_Z_W_G = prodis_mul(c_data, data[:, :, 1], c_W, f_W_G, c_Z, delta, params)
            c_Z_B, f_Z_W_B = prodis_mul(c_data, data[:, :, 2], c_W, f_W_B, c_Z, delta, params)


            c_Z_R, f_Z_B_R = prodis_mul(c_data, data[:, :, 0], c_B, f_B_R, c_Z, delta, params)
            c_Z_G, f_Z_B_G = prodis_mul(c_data, data[:, :, 1], c_B, f_B_G, c_Z, delta, params)
            c_Z_B, f_Z_B_B = prodis_mul(c_data, data[:, :, 2], c_B, f_B_B, c_Z, delta, params)





            f_F = torch.cat(( f_Z_W_R.unsqueeze(-1), f_Z_W_G.unsqueeze(-1), f_Z_W_B.unsqueeze(-1)),
                            dim = 3 ) + torch.cat(( f_Z_B_R.unsqueeze(-1), f_Z_B_G.unsqueeze(-1), f_Z_B_B.unsqueeze(-1)), dim = 3 )

            f_F = f_F.permute(0, 3, 1, 2)

        #    print("f_F.shape:", f_F.shape)
            output = network(f_F)


            loss = loss_func(output, labs)

            print("epoch = ", epoch,  "  ", i, "\\" , value,   " loss = ", loss.item())



            optim_W_R.zero_grad()
            optim_W_G.zero_grad()
            optim_W_B.zero_grad()

            optim_B_R.zero_grad()
            optim_B_G.zero_grad()
            optim_B_B.zero_grad()

            optim_net.zero_grad()

            loss.backward(retain_graph = True)

            optim_W_R.step()
            optim_W_G.step()
            optim_W_B.step()

            optim_B_R.step()
            optim_B_G.step()
            optim_B_B.step()

            optim_net.step()


        left = i*batchsize
        right = LEN_DATA



        data = data_vid[left:right]
        labs = labs_vid[left:right]

        data = data.to(device)
        labs = labs.to(device, dtype = torch.int64)



        c_Z_R, f_Z_W_R = prodis_mul(c_data, data[:, :, 0], c_W, f_W_R, c_Z, delta, params)
        c_Z_G, f_Z_W_G = prodis_mul(c_data, data[:, :, 1], c_W, f_W_G, c_Z, delta, params)
        c_Z_B, f_Z_W_B = prodis_mul(c_data, data[:, :, 2], c_W, f_W_B, c_Z, delta, params)


        c_Z_R, f_Z_B_R = prodis_mul(c_data, data[:, :, 0], c_B, f_B_R, c_Z, delta, params)
        c_Z_G, f_Z_B_G = prodis_mul(c_data, data[:, :, 1], c_B, f_B_G, c_Z, delta, params)
        c_Z_B, f_Z_B_B = prodis_mul(c_data, data[:, :, 2], c_B, f_B_B, c_Z, delta, params)





        f_F = torch.cat(( f_Z_W_R.unsqueeze(-1), f_Z_W_G.unsqueeze(-1), f_Z_W_B.unsqueeze(-1)),
                        dim = 3 ) + torch.cat(( f_Z_B_R.unsqueeze(-1), f_Z_B_G.unsqueeze(-1), f_Z_B_B.unsqueeze(-1)), dim = 3 )

        f_F = f_F.permute(0, 3, 1, 2)

    #    print("f_F.shape:", f_F.shape)
        output = network(f_F)


        loss = loss_func(output, labs)

        print("epoch = ", epoch,  "  ", i, "\\" , value,   " loss = ", loss.item())



        optim_W_R.zero_grad()
        optim_W_G.zero_grad()
        optim_W_B.zero_grad()

        optim_B_R.zero_grad()
        optim_B_G.zero_grad()
        optim_B_B.zero_grad()

        optim_net.zero_grad()

        loss.backward(retain_graph = True)

        optim_W_R.step()
        optim_W_G.step()
        optim_W_B.step()

        optim_B_R.step()
        optim_B_G.step()
        optim_B_B.step()

        optim_net.step()




        if epoch % 1 == 0:
            name_net = net_pa + "network_dis_" + str(epoch).zfill(4) + ".pt"
            name_f_W_R = net_pa + "f_W_R_" + str(epoch).zfill(4) + ".pt"
            name_f_W_G = net_pa + "f_W_G_" + str(epoch).zfill(4) + ".pt"
            name_f_W_B = net_pa + "f_W_B_" + str(epoch).zfill(4) + ".pt"

            name_f_B_R = net_pa + "f_B_R_" + str(epoch).zfill(4) + ".pt"
            name_f_B_G = net_pa + "f_B_G_" + str(epoch).zfill(4) + ".pt"
            name_f_B_B = net_pa + "f_B_B_" + str(epoch).zfill(4) + ".pt"


            torch.save(f_W_R, name_f_W_R)
            torch.save(f_W_G, name_f_W_G)
            torch.save(f_W_B, name_f_W_B)


            torch.save(f_B_R, name_f_B_R)
            torch.save(f_B_G, name_f_B_G)
            torch.save(f_B_B, name_f_B_B)


            torch.save(network.state_dict(), name_net)

            print("\n\n save model completed")


#def test(data_vid, batchsize, device, prodis_mul, c_data, network,
#         c_W, f_W_R, f_W_G, f_W_B, c_Z, delta, params):
def test(data_vid, batchsize, device, prodis_mul, c_data, network,
            c_W, f_W_R, f_W_G, f_W_B,
         c_B, f_B_R, f_B_G, f_B_B, c_Z, delta, params):



    LEN_DATA, dis_num, byte = data_vid.shape

    re_labs = np.zeros(LEN_DATA)

    value = round(LEN_DATA/batchsize)



    for i in range(value):
        left = i*batchsize
        right = (i + 1)*batchsize


        data = data_vid[left:right]
#        labs = labs_vid[left:right]

        data = data.to(device)
#        labs = labs.to(device, dtype = torch.int64)



        c_Z_R, f_Z_W_R = prodis_mul(c_data, data[:, :, 0], c_W, f_W_R, c_Z, delta, params)
        c_Z_G, f_Z_W_G = prodis_mul(c_data, data[:, :, 1], c_W, f_W_G, c_Z, delta, params)
        c_Z_B, f_Z_W_B = prodis_mul(c_data, data[:, :, 2], c_W, f_W_B, c_Z, delta, params)

        c_Z_R, f_Z_B_R = prodis_mul(c_data, data[:, :, 0], c_B, f_B_R, c_Z, delta, params)
        c_Z_G, f_Z_B_G = prodis_mul(c_data, data[:, :, 1], c_B, f_B_G, c_Z, delta, params)
        c_Z_B, f_Z_B_B = prodis_mul(c_data, data[:, :, 2], c_B, f_B_B, c_Z, delta, params)





        f_F = torch.cat(( f_Z_W_R.unsqueeze(-1), f_Z_W_G.unsqueeze(-1), f_Z_W_B.unsqueeze(-1)),
                        dim = 3 ) + torch.cat(( f_Z_B_R.unsqueeze(-1), f_Z_B_G.unsqueeze(-1), f_Z_B_B.unsqueeze(-1)), dim = 3 )

#        f_F = torch.cat(( f_Z_R.unsqueeze(-1), f_Z_G.unsqueeze(-1), f_Z_B.unsqueeze(-1)), dim = 3 )

        f_F = f_F.permute(0, 3, 1, 2)

    #    print("f_F.shape:", f_F.shape)
        output = network(f_F)


        re_labs[left:right] = output.argmax(dim = 1, keepdim = True).cpu().detach().squeeze()

#        print( i, ":", value )



    left = i*batchsize
    right = LEN_DATA



    data = data_vid[left:right]
#    labs = labs_vid[left:right]

    data = data.to(device)
#    labs = labs.to(device, dtype = torch.int64)



#     c_Z_R, f_Z_R = prodis_mul(c_data, data[:, :, 0], c_W, f_W_R, c_Z, delta, params)
#     c_Z_G, f_Z_G = prodis_mul(c_data, data[:, :, 1], c_W, f_W_G, c_Z, delta, params)
#     c_Z_B, f_Z_B = prodis_mul(c_data, data[:, :, 2], c_W, f_W_B, c_Z, delta, params)
#

    c_Z_R, f_Z_W_R = prodis_mul(c_data, data[:, :, 0], c_W, f_W_R, c_Z, delta, params)
    c_Z_G, f_Z_W_G = prodis_mul(c_data, data[:, :, 1], c_W, f_W_G, c_Z, delta, params)
    c_Z_B, f_Z_W_B = prodis_mul(c_data, data[:, :, 2], c_W, f_W_B, c_Z, delta, params)

    c_Z_R, f_Z_B_R = prodis_mul(c_data, data[:, :, 0], c_B, f_B_R, c_Z, delta, params)
    c_Z_G, f_Z_B_G = prodis_mul(c_data, data[:, :, 1], c_B, f_B_G, c_Z, delta, params)
    c_Z_B, f_Z_B_B = prodis_mul(c_data, data[:, :, 2], c_B, f_B_B, c_Z, delta, params)





    f_F = torch.cat(( f_Z_W_R.unsqueeze(-1), f_Z_W_G.unsqueeze(-1), f_Z_W_B.unsqueeze(-1)),
                    dim = 3 ) + torch.cat(( f_Z_B_R.unsqueeze(-1), f_Z_B_G.unsqueeze(-1), f_Z_B_B.unsqueeze(-1)), dim = 3 )




#    f_F = torch.cat(( f_Z_R.unsqueeze(-1), f_Z_G.unsqueeze(-1), f_Z_B.unsqueeze(-1)), dim = 3 )

    f_F = f_F.permute(0, 3, 1, 2)

#    print("f_F.shape:", f_F.shape)
    output = network(f_F)

    re_labs[left:right] = output.argmax(dim = 1, keepdim = True).cpu().detach().squeeze()


    return re_labs



def getListNormalData(imgs, labs, curidx_list, mode):


    curidx = curidx_list[0]

    hist_data, hist_labs = getNormalData(imgs, labs, curidx)



    NUM = len(curidx_list)

    for curidx in curidx_list[1:NUM]:

        temp_data, temp_labs = getNormalData(imgs, labs, curidx)

        hist_data = torch.cat((hist_data, temp_data), dim = 0)
        hist_labs = torch.cat((hist_labs, temp_labs), dim = 0)



    if mode == "train":
        idx = (hist_labs == 255) | (hist_labs == 0)

        hist_data = hist_data[idx]
        hist_labs = hist_labs[idx]


        LEN_NUM = torch.numel(hist_labs)
        idx = torch.randperm(LEN_NUM)

        hist_data = hist_data[idx]
        hist_labs = hist_labs[idx]


    return hist_data, hist_labs



def main():

    pa_im = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/input'
#    pa_im = 'D:/dataset/dataset2014/dataset/dynamicBackground/fountain01/input'
    ft_im = 'jpg'

    pa_gt = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth'
#    pa_gt = 'D:/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth'
    ft_gt = 'png'

    net_pa = '../../data/'
    sa_pa = '../../data/results/'

    use_cuda = torch.cuda.is_available()


    print("------------")
    print(use_cuda)
    print("------------")

    torch.manual_seed(0)

    device = torch.device("cuda:1" if use_cuda else "cpu")

    prodis_mul = ProductDis_multi.apply

    params = {'zero_swap': True, 'zero_approx': True, 'normal': False}



    imgs = loadImgs_pytorch(pa_im, ft_im)
    labs = loadImgs_pytorch(pa_gt, ft_gt)




    frames, row_im, column_im, byte_im = imgs.shape

#    curidx = [600, 720, 1000, 1140, 1160]
    curidx = [1140]

    print("generating trainning data")
    data_vid, labs_vid = getListNormalData(imgs, labs, curidx, "train")
    print("completed")

    labs_vid = torch.round(labs_vid/255)




    left = -1
    right = 1
    delta = 0.01
    num_dis = 16


    c_W, f_W_R = getRandCF_plus(left, right, delta, num_dis)
    c_W, f_W_G = getRandCF_plus(left, right, delta, num_dis)
    c_W, f_W_B = getRandCF_plus(left, right, delta, num_dis)


    c_B, f_B_R = getRandCF_plus(left, right, delta, num_dis)
    c_B, f_B_G = getRandCF_plus(left, right, delta, num_dis)
    c_B, f_B_B = getRandCF_plus(left, right, delta, num_dis)



    c_Z, f_Z       = getEmptyCF_plus(left, right, delta, 1)
    c_data, f_data = getEmptyCF_plus(left, right, delta, 1)


    # randomly permutate the data
    LEN_DATA = torch.numel(labs_vid)
    batchsize = 100

    num_epoch = 20



    c_W = c_W.to(device)
    c_B = c_B.to(device)
    c_Z = c_Z.to(device)

    f_W_R = f_W_R.to(device)
    f_W_G = f_W_G.to(device)
    f_W_B = f_W_B.to(device)


    f_B_R = f_B_R.to(device)
    f_B_G = f_B_G.to(device)
    f_B_B = f_B_B.to(device)


    c_data = c_data.to(device)


    network = ClassifyNetwork(num_dis).to(device)


    optim_net = optim.Adam(network.parameters(), lr = 0.001)


#     optim_W_R = getTensorOptim(f_W_R, 0.01)
#     optim_W_G = getTensorOptim(f_W_G, 0.01)
#     optim_W_B = getTensorOptim(f_W_B, 0.01)


    f_W_R = Variable(f_W_R, requires_grad = True)
    optim_W_R = optim.Adam([f_W_R], lr = 0.01, amsgrad = True)

    f_W_G = Variable(f_W_G, requires_grad = True)
    optim_W_G = optim.Adam([f_W_G], lr = 0.01, amsgrad = True)

    f_W_B = Variable(f_W_B, requires_grad = True)
    optim_W_B = optim.Adam([f_W_B], lr = 0.01, amsgrad = True)


    f_B_R = Variable(f_B_R, requires_grad = True)
    optim_B_R = optim.Adam([f_B_R], lr = 0.01, amsgrad = True)

    f_B_G = Variable(f_B_G, requires_grad = True)
    optim_B_G = optim.Adam([f_B_G], lr = 0.01, amsgrad = True)

    f_B_B = Variable(f_B_B, requires_grad = True)
    optim_B_B = optim.Adam([f_B_B], lr = 0.01, amsgrad = True)



    class_weights = torch.FloatTensor([0.5, 0.5]).to(device)
    loss_func = torch.nn.NLLLoss(weight=class_weights, reduction='sum').to(device)



    train(data_vid, labs_vid, batchsize, device, network, loss_func, prodis_mul,
          c_data, c_W, f_W_R, f_W_G, f_W_B, c_B, f_B_R, f_B_G, f_B_B, c_Z, delta, params,
          optim_W_R, optim_W_G, optim_W_B, optim_B_R, optim_B_G, optim_B_B,
          optim_net, net_pa, num_epoch)






    fs, fullfs = loadFiles_plus(pa_gt, ft_gt)

    for frame_idx in curidx:

        data_vid, labs_vid = getNormalData(imgs, labs, frame_idx)

        re_labs = test(data_vid, batchsize, device, prodis_mul, c_data, network,
            c_W, f_W_R, f_W_G, f_W_B,
            c_B, f_B_R, f_B_G, f_B_B, c_Z, delta, params)



        labs_tru = readImg_byFilesIdx(frame_idx, pa_gt, ft_gt)


    #        gt_fg = np.round(labs[curidx].detach().squeeze().numpy())
        gt_fg = np.round(labs_tru)
        im_fg = np.round(np.reshape(re_labs, (288, 432))*255)



        Re, Pr, Fm = evaluation_numpy(im_fg, gt_fg)

        filename = sa_pa + fs[frame_idx] + '.png'

        saveImg(filename, im_fg.astype(np.uint8))


        print("eva Re:", Re)
        print("eva Pr:", Pr)
        print("eva Fm:", Fm)



    TP_sum = 0
    FP_sum = 0
    TN_sum = 0
    FN_sum = 0


    for frame_idx in range(1184):

        data_vid, labs_vid = getNormalData(imgs, labs, frame_idx)

#        re_labs = test(data_vid, batchsize, device, prodis_mul, c_data, network,
#            c_W, f_W_R, f_W_G, f_W_B, c_Z, delta, params)
        re_labs = test(data_vid, batchsize, device, prodis_mul, c_data, network,
            c_W, f_W_R, f_W_G, f_W_B,
            c_B, f_B_R, f_B_G, f_B_B, c_Z, delta, params)



        labs_tru = readImg_byFilesIdx(frame_idx, pa_gt, ft_gt)


    #        gt_fg = np.round(labs[curidx].detach().squeeze().numpy())
        gt_fg = np.round(labs_tru)
        im_fg = np.round(np.reshape(re_labs, (288, 432))*255)



        TP, FP, TN, FN = evaluation_numpy_entry(im_fg, gt_fg)


        Re = TP/max((TP + FN), 1)
        Pr = TP/max((TP + FP), 1)

        Fm = (2*Pr*Re)/max((Pr + Re), 0.0001)



        TP_sum = TP_sum + TP
        FP_sum = FP_sum + FP
        TN_sum = TN_sum + TN
        FN_sum = FN_sum + FN

        Re_acc = TP_sum/max((TP_sum + FN_sum), 1)
        Pr_acc = TP_sum/max((TP_sum + FP_sum), 1)

        Fm_acc = (2*Pr_acc*Re_acc)/max((Pr_acc + Re_acc), 0.0001)


        filename = sa_pa + fs[frame_idx] + '.png'

        print("")

        print("borderline *-------------------------------------------*")

        print("current files:", fs[frame_idx])

        saveImg(filename, im_fg.astype(np.uint8))


        print("current Re:", Re)
        print("current Pr:", Pr)
        print("current Fm:", Fm)
        print("accumulated Re:", Re_acc)
        print("accumulated Pr:", Pr_acc)
        print("accumulated Fm:", Fm_acc)

        print("borderline *-------------------------------------------*")

        print("")

















if __name__ == '__main__':
    main()
