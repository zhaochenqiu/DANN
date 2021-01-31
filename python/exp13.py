import argparse

import torch

import numpy as np


import matplotlib.pyplot as plt


import time


from torch.autograd import Function


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



class ProductDistribution(Function):

    @staticmethod
    def forward(ctx, C):

        return C

    @staticmethod
    def backward(ctx, grad_output):


        return grad_output



class Test(Function):

    @staticmethod
    def forward(ctx, input, weight):


        print("forward")


        ctx.save_for_backward(input)

#        print(input[0])

        return input



    @staticmethod
    def backward(ctx, grad_output):

        print("backward")

        input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#
#         grad_input[input < 0] = 0

        return grad_output, None






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

#     with torch.no_grad():
#         num = 1000000
#         border = 0.1
#
#
#         X = torch.empty(num).normal_(mean = 0, std = 1)
#         W = torch.empty(num).normal_(mean = 0, std = 1)
#
#         c_X, f_X = getHist(X, 0.1)
#         c_W, f_W = getHist(W, 0.1)
#
#     f_X.requires_grad = True



    test_layer = Test.apply

    N, D_in, D_out = 64, 100, 10

    with torch.no_grad():
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)

#     w = torch.randn(D_in, D_out, requires_grad=True)
#     w = torch.randn(D_in, D_out, requires_grad=True)
        w = torch.randn(D_in, D_out)

    w.requires_grad = True



    t = torch.randn(N, D_in)

#     w = test_layer(w, 'test1', 'test2')

    y_pre = test_layer( x.mm(w),  torch.FloatTensor([1]))

    loss = (y_pre - y).pow(2).sum()

    loss.backward()




#    y = test_layer(x, 'test1', 'test2')

#    y.backward()



#     with torch.no_grad():
# #         data = torch.randn(100)
# #         data = data.detach().numpy()
# #
# #         print(data)
#
# #        num = 1000000
#         num = 10000000
#
#         X = torch.empty(num).normal_(mean = 0, std = 4)
#
#         c_X, f_X = getHist(X, 0.1)
#         print(c_X)
#         print(f_X)
#
#
#         plt.figure()
#         plt.bar(c_X, f_X,
#                 width = 0.1,
#                 color = (0.1, 0.1, 0.1, 0.5))         # R G B alpha
#
#
#         plt.show()






#
#         X = torch.round(X) + 3
#
#         print(X.shape)
#
#         f_Z = torch.histc(X, bins = 4, min = 1, max = 5)
#
#         print(X)
#         print(f_Z)
#
#         getHist(X)
#
#         getHist(X, 20)





#         W = torch.empty(num).normal_(mean = 0, std = 1)
#
#
#         Z = torch.mul(X, W)
#
#         Z = torch.round(Z)
#         Z = Z - torch.min(Z)
#
#
#         f_Z = torch.histc(Z, bins=10, min=0, max=10)
#
#
#         print(f_Z)


# #        Z = Z.detach().numpy()
#
#         f_Z = torch.zeros(10, 1)
#
#
#         starttime = time.time()
#         for i in range(10):
#             print("i = ", i)
#             f_Z[i] = torch.sum(Z == i)
#
#         endtime = time.time()
#
#         print(f_Z)
#
#         print(torch.sum(f_Z))
#
#         print("total time:", endtime - starttime)




#        f_Z = torch.zeros(1, 10)
#
#
#        f_Z[Z] = 1
#
#
#
#
#         print(X)
#         print(W)
#         print(Z)
#
#         print(f_Z)










if __name__ == '__main__':
    main()
