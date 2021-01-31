import sys
sys.path.append('../')
sys.path.append('../../../')



import argparse

import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt
import time


from torch.autograd import Function


from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F


from function.prodis import ProductDis

from function.prodis import ProductDis_multi
from function.prodis import DifferentiateDis_multi



# np.set_printoptions(threshold=np.inf)


def getHist(X, delta = 0.1):
    with torch.no_grad():

        num = torch.numel(X)



        min_X = torch.round(torch.min(X).mul(1.0/delta)).mul(delta)
        max_X = torch.round(torch.max(X).mul(1.0/delta)).mul(delta) + delta


        bins = ((max_X - min_X)/delta).detach().int()


        f_X = torch.histc(X, bins, min_X, max_X)/num
        c_X = torch.linspace(min_X, max_X - delta, bins)

        return c_X, f_X


def getHist_plus(X, delta = 0.1, min_X = None, max_X = None):
    with torch.no_grad():

        num = torch.numel(X)

        flag_min = 0
        flag_max = 0


        if min_X == None:
            min_X = torch.round(torch.min(X).mul(1.0/delta)).mul(delta)
            flag_min = 1

        if max_X == None:
            max_X = torch.round(torch.max(X).mul(1.0/delta)).mul(delta)
            flag_max = 1

        if flag_min == 0:
            min_X = torch.tensor(min_X)

        if flag_max == 0:
            max_X = torch.tensor(max_X)


        max_X = max_X + delta


        bins = ((max_X - min_X)/delta).clone().detach().int()

        f_X = torch.histc(X, bins, min_X, max_X)/num
        c_X = torch.linspace(min_X, max_X - delta, bins)

        c_X = torch.round(c_X/delta)*delta


        return c_X, f_X


def getEmptyCF(left, right, delta):

    bins = round((right - left)/delta) + 1

    c_X = torch.linspace(left, right, bins)

    f_X = c_X.clone() - c_X


    return c_X, f_X



def getRandCF(left, right, delta):

    bins = round((right - left)/delta) + 1

    c_X = torch.linspace(left, right, bins)

    f_X = torch.abs(torch.randn(bins))
    f_X = f_X/torch.sum(f_X)


    return c_X, f_X




def corDisVal(c_X, f_X, c_Y, f_Y):

    with torch.no_grad():

        left  = torch.max(torch.min(c_X), torch.min(c_Y))
        right = torch.min(torch.max(c_X), torch.max(c_Y))

        border = c_X[1] - c_X[0]


        pos_l = (torch.abs(c_X - left)  < 0.5*border).nonzero()
        pos_r = (torch.abs(c_X - right) < 0.5*border).nonzero()

        f_X = f_X[pos_l:pos_r + 1]


        pos_l = (torch.abs(c_Y - left)  < 0.5*border).nonzero()
        pos_r = (torch.abs(c_Y - right) < 0.5*border).nonzero()

        f_Y = f_Y[pos_l:pos_r + 1]

        return torch.sum(f_X.mul(f_Y))/(f_X.mul(f_X).sum().sqrt() * f_Y.mul(f_Y).sum().sqrt() )








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


# data_vid.shape =  torch.Size([236877, 201, 3])
# labs_vid.shape =  torch.Size([236877])

def main():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    print("------------")
    print(use_cuda)
    print("------------")

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    prodis_lay = ProductDis.apply

    prodis_mul = ProductDis_multi.apply
    difdis_mul = DifferentiateDis_multi.apply





    with torch.no_grad():
        num = 10000000
        border = 0.1
        params = {'zero_swap': True, 'zero_approx': True, 'normal': False}

        # preparing the training data
        X1 = torch.empty(num).normal_(mean = 0, std = 1)
        X2 = torch.empty(num).normal_(mean = 0, std = 1.2)


        c_X1, f_X1 = getHist_plus(X1, border)
        c_X2, f_X2 = getHist_plus(X2, border)


    left = -10
    right = 10
    delta = 0.1

    c_W, f_W = getRandCF(left, right, delta)
    c_T, f_T = getRandCF(left, right, delta)



    c_Z, f_Z = getEmptyCF(-10, 10, border)



    c_X1 = c_X1.to(device)
    f_X1 = f_X1.to(device)


    c_X2 = c_X2.to(device)
    f_X2 = f_X2.to(device)


    c_W = c_W.to(device)
    f_W = f_W.to(device)

    c_T = c_T.to(device)
    f_T = f_T.to(device)


    c_Z = c_Z.to(device)
    f_Z = f_Z.to(device)






    f_W = Variable(f_W, requires_grad = True)
    optim_W = optim.Adam([f_W], lr=0.01, amsgrad=True)


    f_T = Variable(f_T, requires_grad = True)
    optim_T = optim.Adam([f_T], lr=0.01, amsgrad=True)


    classnet = ClassNet(201,2000).to(device)
    optim_net = optim.Adam(classnet.parameters(), lr = 0.001)


    class_weights = torch.FloatTensor([0.5, 0.5]).to(device)
    loss_func = torch.nn.NLLLoss(weight=class_weights, reduction='sum').to(device)

    target0 = 0
    target1 = 1

    target0 = torch.tensor([target0])
    target1 = torch.tensor([target1])

    target0 = target0.to(device, dtype = torch.int64)
    target1 = target1.to(device, dtype = torch.int64)





    flag = 0

    for i in range(500):

        if flag == 0:
            c_Z1, f_Z1 = prodis_lay(c_X1, f_X1, c_W, f_W, c_Z, f_Z, border, params)
            c_Z2, f_Z2 = prodis_lay(c_X1, f_X1, c_T, f_T, c_Z, f_Z, border, params)


#            f_F1 = torch.cat((f_Z1, f_Z2), dim = 0)

            f_F1 = f_Z1 + f_Z2


            output = classnet(f_F1)

            loss = loss_func(output, target0)


            print("loss = ", loss)


            optim_W.zero_grad()
            optim_T.zero_grad()
            optim_net.zero_grad()

            loss.backward(retain_graph=True)

            optim_W.step()
            optim_T.step()
            optim_net.step()


            flag = 1

        else:
            c_Z1, f_Z1 = prodis_lay(c_X2, f_X2, c_W, f_W, c_Z, f_Z, border, params)
            c_Z2, f_Z2 = prodis_lay(c_X2, f_X2, c_T, f_T, c_Z, f_Z, border, params)


#            f_F2 = torch.cat((f_Z1, f_Z2), dim = 0)
            f_F2 = f_Z1 + f_Z2



            output = classnet(f_F2)

            loss = loss_func(output, target1)


            print("loss = ", loss)


            optim_W.zero_grad()
            optim_T.zero_grad()
            optim_net.zero_grad()

            loss.backward(retain_graph=True)

            optim_W.step()
            optim_T.step()
            optim_net.step()


            flag = 0


    plt.figure()
    plt.subplot(1, 2, 1)
    plt.bar(c_W.detach().cpu().numpy(),
            f_W.detach().cpu().numpy(),
            width = 0.1,
            color = (0.1, 0.1, 0.1, 0.5))

    plt.subplot(1, 2, 2)
    plt.bar(c_T.detach().cpu().numpy(),
            f_T.detach().cpu().numpy(),
            width = 0.1,
            color = (0.1, 0.1, 0.1, 0.5))

    plt.show()



#     plt.figure()
#
#     plt.subplot(1, 4, 1)
#     plt.bar(c_W.detach().cpu().numpy(),
#             f_W.detach().cpu().numpy(),
#             width = 0.1,
#             color = (0.1, 0.1, 0.1, 0.5))
#
#     plt.subplot(1, 4, 2)
#     plt.bar(c_T.detach().cpu().numpy(),
#             f_T.detach().cpu().numpy(),
#             width = 0.1,
#             color = (0.1, 0.1, 0.1, 0.5))
#
#     plt.subplot(1, 4, 3)
#     plt.bar(c_F1.detach().cpu().numpy(),
#             f_F1.detach().cpu().numpy(),
#             width = 0.1,
#             color = (0.1, 0.1, 0.1, 0.5))
#
#     plt.subplot(1, 4, 4)
#     plt.bar(c_F2.detach().cpu().numpy(),
#             f_F2.detach().cpu().numpy(),
#             width = 0.1,
#             color = (0.1, 0.1, 0.1, 0.5))
#
#     plt.show()





#         if i%10 == 0:
#
#             if i > 2:
#                 plt.subplot(1, 3, 1)
#
#                 plt.bar(c_W.detach().cpu().numpy(),
#                         -f_W.detach().cpu().numpy(),
#                         width = 0.1,
#                         color = (0.1, 0.1, 0.1, 0.5))
#
#                 plt.subplot(1, 3, 2)
#                 plt.bar(c_Z1.detach().cpu().numpy(),
#                         -f_Z1.detach().cpu().numpy(),
#                         width = 0.1,
#                         color = (0.1, 0.1, 0.1, 0.5))
#
#                 plt.subplot(1, 3, 3)
#                 plt.bar(c_Z2.detach().cpu().numpy(),
#                     -f_Z2.detach().cpu().numpy(),
#                         width = 0.1,
#                         color = (0.1, 0.1, 0.1, 0.5))
#
#
#
#
#
#
#                 plt.pause(0.01)

#        plt.show()







#     for i in range(1000):
#
#         c_Z1, f_Z1 = prodis_lay(c_Data, f_Data, c_W, f_W, c_Z, f_Z, border, params)
#
#
#         loss = (f_Z1 - f_Z_gt).pow(2).sum()
#
#
#         corval = corDisVal(c_Z1, f_Z1, c_Z_gt, f_Z_gt)
#
#         print("loss:", loss, "      corval:", corval)
#
#         optimizer.zero_grad()
#
#         loss.backward(retain_graph=True)
#
#         optimizer.step()




#     for i in range(10000):
#
#         f_T.requires_grad = True
#
#         c_Z1, f_Z1 = prodis_lay(c_X, f_X, c_T, f_T, c_Z, f_Z, border, params)
#
#
#
#
#         loss = (f_Z1 - f_gt).pow(2).sum()
#
#
#         corval = corDisVal(c_Z1, f_Z1, c_gt, f_gt)
#
#         print("loss:", loss, "      corval:", corval)
#
#         optimizer.zero_grad()
#
#         loss.backward(retain_graph=True)
#
#         optimizer.step()
#
# #        with torch.no_grad():
# #            f_T = f_T - learning_rate * f_T.grad
#
#




#     f_T.requires_grad = True
#
#
#     c_Z1, f_Z1 = prodis_lay(c_X, f_X, c_T, f_T, c_Z, f_Z, border, params)
#
#     loss = (f_Z1 - f_gt).pow(2).sum()
#
#     print("loss:", loss)
#
#
#     loss.backward(retain_graph=True)
#
#
#
#
#     print("borderline -----------------------------")
#
#     with torch.no_grad():
#         print("f_T.grad:", f_T.grad)
#         f_T = f_T - learning_rate * f_T.grad
#         print("f_T.grad:", f_T.grad)
#
#
#
#
#     print("borderline -----------------------------")








#
#
#
#         Z = X.mul(W)
#
#
#         c_X, f_X = getHist_plus(X, border)
#         c_W, f_W = getHist_plus(W, border)
#
#         c_Z, f_Z = getHist_plus(Z, border)
#
# #        c_Z_ = c_Z.clone()
# #        f_Z_ = torch.abs(f_Z - f_Z)
#
#         c_Z_, f_Z_ = getEmptyCF(-50, 50, border)
#
#
#
#         c_Z = c_Z.to(device)
#         f_Z = f_Z.to(device)
#
#
#         c_X = c_X.to(device)
#         f_X = f_X.to(device)
#
#         c_W = c_W.to(device)
#         f_W = f_W.to(device)
#
#         c_Z_ = c_Z_.to(device)
#         f_Z_ = f_Z_.to(device)
#
#
#
#         starttime = time.time()
#         c_Z_, f_Z_ = prodis_lay(c_X, f_X, c_W, f_W, c_Z_, f_Z_, border, params)
#         endtime = time.time()
#
#
#
#
#         value = corDisVal(c_Z, f_Z, c_Z_, f_Z_)
#
# #        checkcor = corDisVal(c_Z_, f_Z_, c_Z2, f_Z2)
#         print("total time:", endtime - starttime)
#
#         print("sum distribution:", torch.sum(f_Z_))
#
#         print("correlation:", value)
#
# #        print("c_Z_.shape:", c_Z_.shape)
#
#
# #         c_T, f_T = getEmptyCF(-10, 10, 0.01)
# #
# #         print("c_T = ", c_T)
# #         print("f_T = ", f_T)
# #
# #         print("c_T.shape:", c_T.shape)
# #         print("f_T.shape:", f_T.shape)
#
#
#
#
#
# # getEmptyCF(left, right, delta):
#
#
#
# #        print("check cor value:", checkcor)
#
#
#
#         plt.figure()
#         plt.subplot(1, 3, 1)
#         plt.bar(c_X.detach().cpu().numpy(), f_X.detach().cpu().numpy(),
#                 width = 0.1,
#                 color = (0.1, 0.1, 0.1, 0.5))         # R G B alpha
#
#         plt.subplot(1, 3, 2)
#         plt.bar(c_Z.detach().cpu().numpy(), f_Z.detach().cpu().numpy(),
#                 width = 0.1,
#                 color = (0.1, 0.1, 0.1, 0.5))         # R G B alpha
#
#         plt.subplot(1, 3, 3)
#         plt.bar(c_Z_.detach().cpu().numpy(), f_Z_.detach().cpu().numpy(),
#                 width = 0.1,
#                 color = (0.1, 0.1, 0.1, 0.5))         # R G B alpha
#
#         plt.show()
#









if __name__ == '__main__':
    main()
