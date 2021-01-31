import argparse

import torch

import numpy as np

import time

import matplotlib.pyplot as plt



from function.prodis import getHist_plus


from function.prodis import DifferentiateDis
from function.prodis import DifferentiateDis_multi

from function.prodis import corDisVal

# np.set_printoptions(threshold=np.inf)



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


    diffdis_mul = DifferentiateDis.apply


    with torch.no_grad():

        num = 1000000

        X = torch.empty(num).normal_(mean = 1,  std = 4)
        Y = torch.empty(num).normal_(mean = 2, std = 2)

        Z = X + Y


        c_Z, f_Z = getHist_plus(Z, 0.1, -30, 30)
        c_X, f_X = getHist_plus(X, 0.1, -15, 15)
        c_Y, f_Y = getHist_plus(Y, 0.1, -15, 15)


        c_Z_pre, f_Z_pre = diffdis_mul(c_X, f_X, c_Y, f_Y, c_Z)


        corvalue = corDisVal(c_Z, f_Z, c_Z_pre, f_Z_pre)

        print("corvalue:", corvalue)


        plt.figure()
        plt.subplot(2, 2, 1)
        plt.bar(c_X, f_X,
                width = 0.1,
                color = (0.1, 0.1, 0.1, 0.5))
        plt.subplot(2, 2, 2)
        plt.bar(c_Y, f_Y,
                width = 0.1,
                color = (0.1, 0.1, 0.1, 0.5))
        plt.subplot(2, 2, 3)
        plt.bar(c_Z, f_Z,
                width = 0.1,
                color = (0.1, 0.1, 0.1, 0.5))
        plt.subplot(2, 2, 4)
        plt.bar(c_Z_pre, f_Z_pre,
                width = 0.1,
                color = (0.1, 0.1, 0.1, 0.5))

        plt.show()



if __name__ == '__main__':
    main()
