import argparse

import torch

import numpy as np



import time

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


    with torch.no_grad():
#         data = torch.randn(100)
#         data = data.detach().numpy()
#
#         print(data)

        num = 1000000

        X = torch.empty(num).normal_(mean = 0, std = 1)
        W = torch.empty(num).normal_(mean = 0, std = 1)


        Z = torch.mul(X, W)

        Z = torch.round(Z)
        Z = Z - torch.min(Z)


#        Z = Z.detach().numpy()

        f_Z = torch.zeros(10, 1)


        starttime = time.time()
        for i in range(10):
            print("i = ", i)
            f_Z[i] = torch.sum(Z == i)

        endtime = time.time()

        print(f_Z)

        print(torch.sum(f_Z))

        print("total time:", endtime - starttime)




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
