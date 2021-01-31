import sys
sys.path.append("/home/cqzhao/projects/matrix/")
sys.path.append("../../")


import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt


import numpy as np

from common_py.dataIO import loadImgs_pytorch
from common_py.evaluationBS import evaluation_numpy

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
        self.conv2 = nn.Conv2d(20, 1000, (1, 511))

        self.fc1 = nn.Linear(1000, 2)


    def forward(self, input):

        x = F.relu( self.conv2(self.conv1(input)) )
        x = x.view(-1, 1000)
        x = self.fc1(x)


        return F.log_softmax(x, dim = 1)





def main():



    data_vid = torch.load("../../data/data_fountain01.pt")
    labs_vid = torch.load("../../data/labs_fountain01.pt")

    net_pa = '../../data/'


#     data_vid, labs_vid = balanceData_plus(data_vid, labs_vid)
#
#
    labs_vid = torch.round(labs_vid/255)


    # initialize the network
    use_cuda = torch.cuda.is_available()


    print("------------")
    print(use_cuda)
    print("------------")

    torch.manual_seed(0)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    prodis_mul = ProductDis_multi.apply



    params = {'zero_swap': True, 'zero_approx': True, 'normal': False}


    left = -255
    right = 255
    delta = 1
    num_dis = 12


    c_W, f_W_R = getRandCF_plus(left, right, delta, num_dis)
    c_W, f_W_G = getRandCF_plus(left, right, delta, num_dis)
    c_W, f_W_B = getRandCF_plus(left, right, delta, num_dis)


    c_Z, f_Z       = getEmptyCF_plus(left, right, delta, 1)
    c_data, f_data = getEmptyCF_plus(left, right, delta, 1)


    # randomly permutate the data
    LEN_DATA = torch.numel(labs_vid)
    batchsize = 100



    c_W = c_W.to(device)
    c_Z = c_Z.to(device)

    f_W_R = f_W_R.to(device)
    f_W_G = f_W_G.to(device)
    f_W_B = f_W_B.to(device)


    c_data = c_data.to(device)

    network = ClassifyNetwork(num_dis).to(device)
    network.load_state_dict(torch.load('../../data/network_dis_0040.pt'))



    f_W_R = torch.load('../../data/layer_dis_R_0040.pt')
    f_W_G = torch.load('../../data/layer_dis_G_0040.pt')
    f_W_B = torch.load('../../data/layer_dis_B_0040.pt')





    re_labs = np.zeros(LEN_DATA)

    value = round(LEN_DATA/batchsize)



    for i in range(value):
        left = i*batchsize
        right = (i + 1)*batchsize


        data = data_vid[left:right]
        labs = labs_vid[left:right]

        data = data.to(device)
        labs = labs.to(device, dtype = torch.int64)



        c_Z_R, f_Z_R = prodis_mul(c_data, data[:, :, 0], c_W, f_W_R, c_Z, delta, params)
        c_Z_G, f_Z_G = prodis_mul(c_data, data[:, :, 1], c_W, f_W_G, c_Z, delta, params)
        c_Z_B, f_Z_B = prodis_mul(c_data, data[:, :, 2], c_W, f_W_B, c_Z, delta, params)



        f_F = torch.cat(( f_Z_R.unsqueeze(-1), f_Z_G.unsqueeze(-1), f_Z_B.unsqueeze(-1)), dim = 3 )

        f_F = f_F.permute(0, 3, 1, 2)

    #    print("f_F.shape:", f_F.shape)
        output = network(f_F)


        re_labs[left:right] = output.argmax(dim = 1, keepdim = True).cpu().detach().squeeze()

        print( i, ":", value )



    left = i*batchsize
    right = LEN_DATA



    data = data_vid[left:right]
    labs = labs_vid[left:right]

    data = data.to(device)
    labs = labs.to(device, dtype = torch.int64)



    c_Z_R, f_Z_R = prodis_mul(c_data, data[:, :, 0], c_W, f_W_R, c_Z, delta, params)
    c_Z_G, f_Z_G = prodis_mul(c_data, data[:, :, 1], c_W, f_W_G, c_Z, delta, params)
    c_Z_B, f_Z_B = prodis_mul(c_data, data[:, :, 2], c_W, f_W_B, c_Z, delta, params)



    f_F = torch.cat(( f_Z_R.unsqueeze(-1), f_Z_G.unsqueeze(-1), f_Z_B.unsqueeze(-1)), dim = 3 )

    f_F = f_F.permute(0, 3, 1, 2)

#    print("f_F.shape:", f_F.shape)
    output = network(f_F)

    re_labs[left:right] = output.argmax(dim = 1, keepdim = True).cpu().detach().squeeze()
    labs_vid = labs_vid.cpu().detach().squeeze().numpy()



    pa_gt = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth'
    pa_gt = 'D:/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth'
    ft_gt = 'png'


    labs = loadImgs_pytorch(pa_gt, ft_gt)

    curidx = 1140

    gt_fg = np.round(labs[curidx].detach().squeeze().numpy())
    im_fg = np.round(np.reshape(re_labs, (288, 432))*255)
#    gt_fg = np.round(np.reshape(labs_vid, (288, 432))*255)

#    print(gt_fg)


    Re, Pr, Fm = evaluation_numpy(im_fg, gt_fg)

    print("Re:", Re)
    print("Pr:", Pr)
    print("Fm:", Fm)


    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_fg)
    plt.subplot(1, 2, 2)
    plt.imshow(gt_fg)

    plt.show()


#     im = np.reshape(re_labs, (288, 432))*255
#
#     plt.figure()
#     plt.imshow(im)
#     plt.show()





#
#
#     num_dis = 4
#
#     c_W, f_W = getRandCF_plus(left - 5, right + 5, delta, num_dis)
#
#     c_Z, f_Z = getEmptyCF_plus(left - 10, right + 10, delta, num_dis)
#
#
#     c_data, f_temp = getEmptyCF_plus(left, right, delta, 1)
#
#
#
#     c_W = c_W.to(device)
#     f_W = f_W.to(device)
#
#     c_Z = c_Z.to(device)
#     f_Z = f_Z.to(device)
#
#     c_data = c_data.to(device)
#
#
# #    NUM_X, LEN_X = f_X.shape
#     NUM_W, LEN_W = f_W.shape
#
#     LEN_Z = torch.numel(c_Z)
#
#
#     f_W = torch.load("f_W.pt")
#     classnet = ClassNet(LEN_Z*num_dis,2000).to(device)
#     classnet.load_state_dict(torch.load('./network.pt'))
#
#
#
#     value = round(length/batchsize)
#
#
#     re_labs = np.zeros(length)
#
#
#
#     for i in range(value):
#         left = i*batchsize
#         right = (i + 1)*batchsize
#
#
#         data = data_all[left:right]
#         labs = labs_all[left:right]
#
#
#         data = data.to(device)
#         labs = labs.to(device, dtype = torch.int64)
#
#
#
#         c_Z1, f_Z1 = prodis_mul(c_data, data, c_W, f_W, c_Z, border, params)
#
#
#
#         num, row, column = f_Z1.shape
#
#         f_F = f_Z1.reshape(num, row*column)
#
#
#         output = classnet(f_F)
#
#         re_labs[left:right] = output.argmax(dim = 1, keepdim = True).cpu().detach().squeeze()
#
#         print("output.shape:", output.shape)
#
#
#
#
#
#
#
#     left = i*batchsize
#     right = length
#
#
#     data = data_all[left:right]
#     labs = labs_all[left:right]
#
#     data = data.to(device)
#     labs = labs.to(device, dtype = torch.int64)
#
#     c_Z1, f_Z1 = prodis_mul(c_data, data, c_W, f_W, c_Z, border, params)
#
#
#
#     num, row, column = f_Z1.shape
#
#     f_F = f_Z1.reshape(num, row*column)
#
#
#     output = classnet(f_F)
#
#     re_labs[left:right] = output.argmax(dim = 1, keepdim = True).cpu().detach().squeeze()
#
#
#     im = np.reshape(re_labs, (288, 432))*255
#
#
#
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.imshow(im)
#     plt.show()
#





if __name__ == '__main__':
    main()
