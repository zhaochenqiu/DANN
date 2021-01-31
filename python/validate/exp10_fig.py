import sys
sys.path.append("/home/cqzhao/projects/matrix/")
sys.path.append("../../../")
sys.path.append("../")

import matplotlib.pyplot as plt

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from common_py.dataIO import loadImgs_pytorch


from function.prodis import getEmptyCF
from function.prodis import getEmptyCF_plus
from function.prodis import getHist_plus
from function.prodis import ProductDis_plus
from function.prodis import ProductDis_multi
from function.prodis import ProductDis_multiW
from function.prodis import ProductDis_fast
from function.prodis import getRandCF
from function.prodis import getRandCF_plus
from function.prodis import ProductDis_test

from function.prodis import ProDisFunction as ProF

from torch.autograd import Variable

from function.prodis import DifferentiateDis_multi



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


class ReplaceNet(nn.Module):
    def __init__(self, input_size=1000, output_size=2000):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)


    def forward(self ,input):

        x = self.fc(input)

        return x


def main():

    use_cuda = torch.cuda.is_available()


    print("------------")
    print(use_cuda)
    print("------------")

    torch.manual_seed(0)

    device = torch.device("cuda:0" if use_cuda else "cpu")

#    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


#    prodis_lay = ProductDis.apply
#     prodis_lay = ProductDis_plus.apply
#
#     prodis_mw   = ProductDis_fast.apply
#
#     prodis_fast = ProductDis_fast.apply
#
#     prodis_multiW = ProductDis_multiW.apply
#
#     prodis_test = ProductDis_test.apply



    prodis_mul = ProductDis_multi.apply
#    prodis_mul = ProductDis_plus.apply
    difdis_mul = DifferentiateDis_multi.apply





    left = 0
    right = 50
    delta = 0.1



    with torch.no_grad():
        num = 10000000
        border = delta
        params = {'zero_swap': False, 'zero_approx': False, 'normal': False}

        # preparing the training data
        X1 = torch.empty(num).normal_(mean = 4, std = 1)
        #+ torch.empty(num).normal_(mean = 1, std = 0.1)
        X2 = torch.empty(num).normal_(mean = 10, std = 1)


        Z = X1.mul(X2)


        c_X1, f_X1 = getHist_plus(X1, border, left, right)
        c_X2, f_X2 = getHist_plus(X2, border, left, right)


        c_Z, f_Z = getHist_plus(Z, border, left, right + 200)

#     num_dis = 100
    num_dis = 1

    c_W, f_W = getRandCF_plus(left, right, delta, num_dis)
    f_W = f_W - f_W
#
    print(f_W.shape)
#
#
#     c_W, f_W = getHist_plus(X2, border, left, right)
#     f_W = f_W.unsqueeze(0)
#     print(f_W.shape)



    c_B, f_B = getRandCF_plus(left, right, delta, num_dis)






    c_X1 = c_X1.to(device)
    f_X1 = f_X1.to(device)

    c_X2 = c_X2.to(device)
    f_X2 = f_X2.to(device)


    c_W = c_W.to(device)
    f_W = f_W.to(device)


    c_B = c_B.to(device)
    f_B = f_B.to(device)


    c_Z = c_Z.to(device)
    f_Z = f_Z.to(device)



    f_W = Variable(f_W, requires_grad = True)
    optim_W = optim.Adam([f_W], lr=0.00001, amsgrad=True)
#    optim_W = optim.SGD([f_W], lr=0.000001, momentum=0.9)




    f_B = Variable(f_B, requires_grad = True)
    optim_B = optim.Adam([f_B], lr=0.00001, amsgrad=True)

#    optim_B = optim.SGD([f_B], lr=0.01, momentum=0.9)


    plt.figure(figsize=(16,8))

    f_X1 = f_X1.unsqueeze(0)

    for i in range(5000):


        c_Z_W, f_Z_W = prodis_mul(c_X1, f_X1, c_W, f_W, c_Z, border, params)


        loss = (f_Z_W - f_Z).abs().sum()


        print("loss = ", loss.item())


        optim_W.zero_grad()

        loss.backward(retain_graph=True)

        optim_W.step()


    plt.clf()


    plt.subplot(3, 2, 1)
    plt.bar(c_X1.detach().squeeze().cpu().numpy(), f_X1.detach().squeeze().cpu().numpy(),
            width = 0.1,
            color = (0.1, 0.1, 0.1, 0.5))         # R G B alpha
    plt.title("X: Z = XW")

    plt.subplot(3, 2, 2)
    plt.bar(c_W.detach().squeeze().cpu().numpy(), f_W.detach().squeeze().cpu().numpy(),
            width = 0.1,
            color = (0.1, 0.1, 0.1, 0.5))         # R G B alpha
    plt.title("W: Z = XW")

    plt.subplot(3, 2, 3)
    plt.bar(c_Z_W.detach().squeeze().cpu().numpy(), f_Z_W.detach().squeeze().cpu().numpy(),
            width = 0.1,
            color = (0.1, 0.1, 0.1, 0.5))         # R G B alpha
    plt.title("Z: Z = XW")


    plt.subplot(3, 2, 4)
    plt.bar(c_X1.detach().squeeze().cpu().numpy(), f_X1.detach().squeeze().cpu().numpy(),
            width = 0.1,
            color = (0.1, 0.1, 0.1, 0.5))         # R G B alpha
    plt.title("X: Z_T = XW_T")

    plt.subplot(3, 2, 5)
    plt.bar(c_X2.detach().squeeze().cpu().numpy(), f_X2.detach().squeeze().cpu().numpy(),
            width = 0.1,
            color = (0.1, 0.1, 0.1, 0.5))         # R G B alpha
    plt.title("W_T: Z_T = XW_T")

    plt.subplot(3, 2, 6)
    plt.bar(c_Z.detach().squeeze().cpu().numpy(), f_Z.detach().squeeze().cpu().numpy(),
            width = 0.1,
            color = (0.1, 0.1, 0.1, 0.5))         # R G B alpha
    plt.title("Z_T: Z_T = XW_T")






    plt.draw()
    plt.pause(0.001)


#    plt.ylim((0.0, 1.0))
    plt.show()







#         plt.figure()
#         plt.subplot(1, 3, 1)
#         plt.bar(c_X, f_X,
#                 width = 0.1,
#                 color = (0.1, 0.1, 0.1, 0.5))         # R G B alpha
#
#         plt.subplot(1, 3, 2)
#         plt.bar(c_W, f_W,
#                 width = 0.1,
#                 color = (0.1, 0.1, 0.1, 0.5))         # R G B alpha
#
#         plt.subplot(1, 3, 3)
#         plt.bar(c_Z, f_Z,
#                 width = 0.1,
#                 color = (0.1, 0.1, 0.1, 0.5))         # R G B alpha
#
#         plt.show()

if __name__ == '__main__':
    main()
