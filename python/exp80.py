import sys
sys.path.append("/home/cqzhao/projects/matrix/")
sys.path.append("../../")


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

from function.prodis import DifferentiateDis_multi
from function.prodis import DifferentiateDis_bias

from function.prodis import ProDisFunction as ProF

from torch.autograd import Variable



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



#     prodis_mul = ProductDis_multi.apply
#     diffdis_mul = DifferentiateDis_multi.apply

#    prodis_mul = DifferentiateDis_multi.apply

    prodis_mul = ProductDis_multi.apply
    difdis_bia = DifferentiateDis_bias.apply




    left = -10
    right = 10
    delta = 0.1



    with torch.no_grad():
        num = 10000000
        border = 0.1
        params = {'zero_swap': True, 'zero_approx': True, 'normal': False}

        # preparing the training data
        X1 = torch.empty(num).normal_(mean = 0, std = 1)
        X2 = torch.empty(num).normal_(mean = 0, std = 1.2)


        c_X1, f_X1 = getHist_plus(X1, border, left, right)
        c_X2, f_X2 = getHist_plus(X2, border, left, right)


        f_X = torch.cat((f_X1.unsqueeze(0), f_X2.unsqueeze(0)), dim = 0)
        c_X = c_X1.clone()




#     c_W, f_W1 = getRandCF(left, right, delta)
#     c_W, f_W2 = getRandCF(left, right, delta)
#
#     f_W = torch.cat((f_W1.unsqueeze(0), f_W2.unsqueeze(0)), dim = 0)
#     print("f_W.shape = ", f_W.shape)








#     num_dis = 100
    num_dis = 20

    c_W, f_W = getRandCF_plus(left - 5, right + 5, delta, num_dis)
    c_B, f_B = getRandCF_plus(left - 3, right + 3, delta, num_dis)




    c_Z, f_Z = getEmptyCF_plus(left - 10, right + 10, delta, num_dis)
    c_ZB, f_ZB = getEmptyCF_plus(left - 15, right + 15, delta, num_dis)



    c_X = c_X.to(device)
    f_X = f_X.to(device)


    c_X1 = c_X1.to(device)
    f_X1 = f_X1.to(device)

    c_X2 = c_X2.to(device)
    f_X2 = f_X2.to(device)


    c_W = c_W.to(device)
    f_W = f_W.to(device)

    c_B = c_B.to(device)
    f_B = f_B.to(device)

    c_ZB = c_ZB.to(device)
    f_ZB = f_ZB.to(device)


    c_Z = c_Z.to(device)
    f_Z = f_Z.to(device)



    NUM_X, LEN_X = f_X.shape
    NUM_W, LEN_W = f_W.shape

    LEN_Z = torch.numel(c_Z)
    LEN_ZB = torch.numel(c_ZB)

#     print("NUM_X:", NUM_X)
#     print("NUM_W:", NUM_W)
#
#     print("LEN_X:", LEN_X)
#     print("LEN_W:", LEN_W)
#     print("LEN_Z:", LEN_Z)




    f_W = Variable(f_W, requires_grad = True)
    optim_W = optim.Adam([f_W], lr=0.01, amsgrad=True)


    f_B = Variable(f_B, requires_grad = True)
    optim_B = optim.Adam([f_B], lr=0.01, amsgrad=True)


    classnet = ClassNet(LEN_ZB*num_dis,2000).to(device)
    optim_net = optim.Adam(classnet.parameters(), lr = 0.001)


    class_weights = torch.FloatTensor([0.5, 0.5]).to(device)
    loss_func = torch.nn.NLLLoss(weight=class_weights, reduction='sum').to(device)


    target0 = 0
    target1 = 1

    target0 = torch.tensor([target0])
    target1 = torch.tensor([target1])

    target0 = target0.to(device, dtype = torch.int64)
    target1 = target1.to(device, dtype = torch.int64)

    target = torch.cat((target0, target1), dim=0)



#     print("f_X.shape:", f_X.shape)
#     print("f_W.shape:", f_W.shape)
#     print("c_Z.shape:", c_Z.shape)


    for i in range(1000):


        c_Z1, f_Z1 = prodis_mul(c_X, f_X, c_W, f_W, c_Z, border, params)

        c_Z2, f_Z2 = difdis_bia(c_Z1, f_Z1, c_B, f_B, c_ZB, border, params)

    #    print("f_Z1.shape:", f_Z1.shape)

    #    print("f_Z2.shape:", f_Z2.shape)



    #    num, row, column = f_Z1.shape

        num, row, column = f_Z2.shape

        f_F = f_Z2.reshape(num, row*column)

    #    print("f_F.shape:", f_F.shape)


        output = classnet(f_F)


        loss = loss_func(output, target)

        print("loss = ", loss.item())


        optim_B.zero_grad()
        optim_W.zero_grad()
        optim_net.zero_grad()

        loss.backward(retain_graph=True)

        optim_B.step()
        optim_W.step()
        optim_net.step()






#     for i in range(50):
#
#         c_Z1, f_Z1 = prodis_mul(c_X, f_X, c_W, f_W, c_Z, border, params)
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
#         loss = loss_func(output, target)
#
#         print("loss = ", loss.item())
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
