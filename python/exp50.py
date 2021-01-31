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
    prodis_lay = ProductDis_plus.apply

    prodis_mw   = ProductDis_fast.apply

    prodis_fast = ProductDis_fast.apply

    prodis_mul = ProductDis_multi.apply

    prodis_multiW = ProductDis_multiW.apply

    prodis_test = ProductDis_test.apply



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



    print("f_X.shape:", f_X.shape)
    print("c_X.shape:", c_X.shape)


    print("c_X1.shape:", c_X1.shape)
    print("c_X2.shape:", c_X2.shape)





    num_dis = 100

    c_W, f_W = getRandCF_plus(left - 5, right + 5, delta, num_dis)

    c_Z, f_Z = getEmptyCF_plus(left - 10, right + 10, delta, num_dis)


    c_X = c_X.to(device)
    f_X = f_X.to(device)


    c_X1 = c_X1.to(device)
    f_X1 = f_X1.to(device)

    c_X2 = c_X2.to(device)
    f_X2 = f_X2.to(device)


    c_W = c_W.to(device)
    f_W = f_W.to(device)


    c_Z = c_Z.to(device)
    f_Z = f_Z.to(device)





    f_W = Variable(f_W, requires_grad = True)
    optim_W = optim.Adam([f_W], lr=0.01, amsgrad=True)


    classnet = ClassNet(20100,2000).to(device)
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




    starttime = time.time()

    c_Z1, f_Z1 = prodis_mul(c_X, f_X, c_W, f_W, c_Z, border, params)

    endtime = time.time()




    idx_x = 1
    idx_w = 43




    c_Z2, f_Z2 = prodis_fast(c_X, f_X[idx_x,:], c_W, f_W[idx_w,:], c_Z, border, params)

    print("total time:", endtime - starttime)

    print("f_Z1.shape:", f_Z1.shape)

    print(f_X1.shape)



    print("borderline -------------------------------------")




    checkf1 = f_Z1[idx_x,idx_w,:]
    checkf2 = f_Z2

    print(torch.sum(checkf1))
    print(torch.sum(checkf2))


    print(torch.sum(torch.abs(checkf1 - checkf2)))








#     for i in range(50):
#
#         starttime = time.time()
#         c_Z1, f_Z1 = ProF.productDis_plus(c_X1, f_X1, c_W, f_W, c_Z, border, params, prodis_mw)
#         c_Z2, f_Z2 = ProF.productDis_plus(c_X2, f_X2, c_W, f_W, c_Z, border, params, prodis_mw)
#         endtime = time.time()
#
#         print("product time:", endtime - starttime)
#
#
#         row, column = f_Z1.shape
#
#         f_Z1 = f_Z1.reshape(1, row*column)
#         f_Z2 = f_Z2.reshape(1, row*column)
#
#
#
#         f_F = torch.cat((f_Z1, f_Z2), dim = 0)
#
#         print("f_F.shape:", f_F.shape)
#
#
#         starttime = time.time()
#         output = classnet(f_F)
#         endtime = time.time()
#
#         print("network time:", endtime - starttime)
#
#
#
#         loss = loss_func(output, target)
#
#
#         print("loss = ", loss)
#
#
#         optim_W.zero_grad()
#         optim_net.zero_grad()
#
#         loss.backward(retain_graph=True)
#
#         optim_W.step()
#         optim_net.step()







#     for i in range(50):
#
# #        c_Z1, f_Z1 = prodis_lay(c_X1, f_X1, c_W, f_W[0], c_Z, f_Z, border, params)
# #        c_Z2, f_Z2 = prodis_lay(c_X1, f_X1, c_W, f_W[1], c_Z, f_Z, border, params)
#
#         c_Z1, f_Z1 = ProF.productDis_plus(c_X1, f_X1, c_W, f_W, c_Z, f_Z, border, params, prodis_lay)
#         c_Z2, f_Z2 = ProF.productDis_plus(c_X2, f_X2, c_W, f_W, c_Z, f_Z, border, params, prodis_lay)
#
#
#
#
#
#
# #     c_Z, f_Z = ProF.productDis_plus(c_X1, f_X1, c_W, f_W, c_Z, f_Z, border, params, prodis_lay)
# #
# #     print("f_Z.shape = ", f_Z.shape)
#
#
#
#
#
#
#         c_Z3, f_Z3 = prodis_lay(c_X2, f_X2, c_W, f_W[0], c_Z, f_Z, border, params)
#         c_Z4, f_Z4 = prodis_lay(c_X2, f_X2, c_W, f_W[1], c_Z, f_Z, border, params)
#
#
#         f_F1 = torch.cat((f_Z1, f_Z2), dim = 0)
#         f_F2 = torch.cat((f_Z3, f_Z4), dim = 0)
#
#
#         f_F = torch.cat((f_F1.unsqueeze(0), f_F2.unsqueeze(0)), dim = 0)
#
#
#
#         output = classnet(f_F)
#
#
#         loss = loss_func(output, target)
#
#
#         print("loss = ", loss)
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
