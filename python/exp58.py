import sys
sys.path.append("/home/cqzhao/projects/matrix/")
sys.path.append("../../")


import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from common_py.dataIO import loadImgs_pytorch


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



    # randomly permutate the input
    idx = torch.randperm(row*column)

    hist_data = hist_data[idx]
    labs_data = labs_data[idx]


    return hist_data, labs_data


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




def main():

    data_all, labs_all = getVideoData()
    data_all = data_all[:, :, 0].squeeze()


    data_all, labs_all = balanceData(data_all, labs_all)



#     torch.save(data_all, "data_all.pt")
#
#     data_test = torch.load("data_all.pt")
#
#
#
#     print(torch.sum(torch.abs(data_all - data_test)))







#     labs_all = torch.round(labs_all/255.0)
#
#     length = torch.numel(labs_all)
#
#
#     batchsize = 100
#
#
#
#
#
#
#
#     # initialize the network
#     use_cuda = torch.cuda.is_available()
#
#
#     print("------------")
#     print(use_cuda)
#     print("------------")
#
#     torch.manual_seed(0)
#
#     device = torch.device("cuda:1" if use_cuda else "cpu")
#
#     prodis_mul = ProductDis_multi.apply
#
#
#
#     border = 1
#     params = {'zero_swap': True, 'zero_approx': True, 'normal': False}
#
#
#     left = -255
#     right = 255
#     delta = 1
#
#
#     num_dis = 16
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
#     f_W = Variable(f_W, requires_grad = True)
#     optim_W = optim.Adam([f_W], lr=0.01, amsgrad=True)
#
#
#     classnet = ClassNet(LEN_Z*num_dis,2000).to(device)
#     optim_net = optim.Adam(classnet.parameters(), lr = 0.001)
#
#
#     class_weights = torch.FloatTensor([0.5, 0.5]).to(device)
#     loss_func = torch.nn.NLLLoss(weight=class_weights, reduction='sum').to(device)
#
#
#
#
#
#
#     for epoch in range(1000):
#
#
#
#         idx = torch.randperm(length)
#
#         data_all = data_all[idx]
#         labs_all = labs_all[idx]
#
#
#         value = round(length/batchsize)
#
#
#
#
#
#
#         for i in range(value):
#             left = i*batchsize
#             right = (i + 1)*batchsize
#
#
#             data = data_all[left:right]
#             labs = labs_all[left:right]
#
#
#             data = data.to(device)
#             labs = labs.to(device, dtype = torch.int64)
#
#
#
# #             print("c_data.shape:", c_data.shape)
# #             print("data.shape:", data.shape)
#
#
#
#             c_Z1, f_Z1 = prodis_mul(c_data, data, c_W, f_W, c_Z, border, params)
#
#
#
#             num, row, column = f_Z1.shape
#
#             f_F = f_Z1.reshape(num, row*column)
#
#
#             output = classnet(f_F)
#
#
#             loss = loss_func(output, labs)
#
# #            print("before print")
#             print("epoch = ", epoch,  "  ", i, "\\" , value,   " loss = ", loss.item())
#
#
#             optim_W.zero_grad()
#             optim_net.zero_grad()
#
#             loss.backward(retain_graph=True)
#
#             optim_W.step()
#             optim_net.step()
#
#
#
#
#
#
#         left = i*batchsize
#         right = length
#
#
#         data = data_all[left:right]
#         labs = labs_all[left:right]
#
#         data = data.to(device)
#         labs = labs.to(device, dtype = torch.int64)
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
#
#         loss = loss_func(output, labs)
#
# #        print("loss = ", loss.item())
#         print("epoch = ", epoch, "loss = ", loss.item())
#
#
#         optim_W.zero_grad()
#         optim_net.zero_grad()
#
#         loss.backward(retain_graph=True)
#
#         optim_W.step()
#         optim_net.step()









if __name__ == '__main__':
    main()
