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



def corDisVal(c_X, f_X, c_Y, f_Y):
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

'''
pytorch矩阵操作
var = torch.Tensor()  返回一个Tensor

tensor1 = torch.Tensor(3, 3)
tensor2 = torch.Tensor(3, 3)
 var2 = torch.add(tensor1, tensor2)     # 矩阵加
 var3 = torch.sub(tensor1, tensor2)     # 减
 var4 = torch.mul(tensor1, tensor2)     # 乘
 var5 = torch.div(tensor1, tensor2)     # 矩阵点除
 var6 = torch.mm(tensor1, tensor2)      # 矩阵乘

'''

class ProductDis(Function):
    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, c_Z, f_Z, border = 0.1):

        with torch.no_grad():
            N_X = torch.numel(c_X)
            N_W = torch.numel(c_W)
            N_Z = torch.numel(c_Z)


            cc_X = c_X.expand(N_Z, N_X).t()

            W = c_Z / cc_X
            pos = torch.round( W/border ) - torch.round( torch.min(c_W/border) )


            ff_W = torch.cat((f_W, f_W[-2:-1]), dim=0)
            ff_W[N_W] = 0.0


            pos[ ~( (pos > -1) & (pos < torch.numel(c_W) ) ) ] = N_W
            pos = pos.long()

            mf_W = ff_W[pos]


            p_X = f_X.clone()
            p_X[c_X == 0] = 0

            mp_X = p_X.expand(N_Z, N_X).t()


            cc_X = c_X.clone()
            cc_X[c_X == 0] = 1
            mc_X = cc_X.expand(N_Z, N_X).t()
            mc_X = torch.abs(1/mc_X)


            f_Z = torch.sum( mp_X.mul(mf_W).mul(mc_X), dim = 0)


            # when x equals to 0
            deltax = 2.0*border
            x_l = 0 - 1.0*border
            x_r = 0 + 1.0*border

            w_l = c_Z/x_l
            w_r = c_Z/x_r

            idx_l = torch.round(w_l/border) - torch.round( torch.min(c_W/border) )
            idx_r = torch.round(w_r/border) - torch.round( torch.min(c_W/border) )


            idx_l[ ~( (idx_l > -1) & (idx_l < torch.numel(c_W) ) ) ] = N_W
            idx_r[ ~( (idx_r > -1) & (idx_r < torch.numel(c_W) ) ) ] = N_W

            idx_l = idx_l.long()
            idx_r = idx_r.long()


            zp_W = (ff_W[idx_l] + ff_W[idx_r])/deltax
            zp_W = zp_W*1.13


            zf_X = f_X[torch.abs(c_X) < 0.5*border]
            if torch.numel(zf_X) != 1:
                zf_X = 0

            value = zp_W*zf_X

            f_Z = f_Z + value

        f_Z = f_Z/torch.sum(f_Z)

        return c_Z, f_Z

    @staticmethod
    def backward(ctx, grad_output, grad_output2):

        return grad_output, None, None, None, None, None, None



class ProductDis_backup(Function):
    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, c_Z, f_Z, border = 0.1):

        with torch.no_grad():

            N_X = torch.numel(c_X)
            N_Z = torch.numel(c_Z)

            cc_X = c_X.expand(N_Z, N_X).t()

            W = c_Z / cc_X
            W = torch.round( W*(1/border) )


            offset_w = torch.round( torch.min(c_W/border) )
            pos = W - offset_w


            min_pos = -1
            max_pos = torch.numel(c_W)


            N_W = torch.numel(c_W)


            ff_W = torch.cat((f_W, f_W[-2:-1]), dim=0)

            ff_W[N_W] = 0


            idx_nonexist = ~( (pos > min_pos) & (pos < max_pos) )

            pos[idx_nonexist] = N_W
            pos = pos.long()


            mf_W = ff_W[pos]


            p_x = f_X.clone()
            p_x[c_X == 0] = 0


            mp_x = p_x.expand(N_Z, N_X).t()

            cc_X = c_X.clone()
            cc_X[c_X == 0] = 1
            x = cc_X.expand(N_Z, N_X).t()
            mx = torch.abs(1/x)


            mf_Z = mp_x.mul(mf_W).mul(mx)


            f_Z = torch.sum(mf_Z, dim=0)



        f_Z = f_Z/torch.sum(f_Z)

        return c_Z, f_Z

    @staticmethod
    def backward(ctx, grad_output, grad_output2):

        return grad_output, None, None, None, None, None, None






class ProductDis_Old(Function):
    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, c_Z, f_Z, border = 0.1):

        with torch.no_grad():
            for i in range(torch.numel(c_Z)):
                z = c_Z[i]

                idx_nonzero = c_X != 0

                w = torch.round( (z/c_X[idx_nonzero])*(1/border) )

                offset_w = torch.round( torch.min(c_W/border) )
                pos = w - offset_w

                min_pos = -1
                max_pos = torch.numel(c_W)

                idx_exist = (pos > min_pos) & (pos < max_pos)
                pos = pos[idx_exist]

                pos = pos.long()


                x = c_X[idx_nonzero]
                x = x[idx_exist]


                p_x = f_X[idx_nonzero]
                p_x = p_x[idx_exist]

                f_Z[i] = torch.sum( p_x.mul(f_W[pos]).mul(torch.abs(1/x)) )

#                 deltax = 2.0*border
#                 x_l = 0 - 1.0*border
#                 x_r = 0 + 1.0*border
#
#                 w_l = z/x_l
#                 w_r = z/x_r
#
#                 w_l = torch.round(w_l/border)*border
#                 w_r = torch.round(w_r/border)*border
#
#                 idx_l = c_W == w_l
#                 idx_r = c_W == w_r
#
#                 p_W = (f_W[idx_l] + f_W[idx_r])/deltax
#
#                 # 这个值还需要讨论,公式推导的是大于1.13
#                 p_W = p_W*2.26
#
#
#                 if torch.numel(p_W) != 0:
#                     idx_0 = c_X == 0
#                     value = f_X[idx_0] * p_W
#
#                     if torch.numel(value) != 0:
#                         f_Z[i] = f_Z[i] + value


#        f_Z = f_Z/torch.sum(f_Z)

        return c_Z, f_Z

    @staticmethod
    def backward(ctx, grad_output, grad_output2):

        return grad_output, None, None, None, None, None, None





class ProductDistribution(Function):


    # XXX pytorch 计算有精度问题，所以传入一个border用于控制精度
    @staticmethod
    def forward(ctx, c_X, f_X, c_W, f_W, border = 0.1):

        print("forward")

        with torch.no_grad():
#             print("test")

            left = torch.min(c_X)
            left = torch.min(left, torch.min(c_W)*torch.min(c_X))
            left = torch.min(left, torch.min(c_W)*torch.max(c_X))
            left = torch.min(left, torch.max(c_W)*torch.min(c_X))
            left = torch.min(left, torch.max(c_W)*torch.max(c_X))

            left = torch.round(left/border - 0.5)*border


            right = torch.max(c_X)
            right = torch.max(right, torch.min(c_W)*torch.min(c_X))
            right = torch.max(right, torch.min(c_W)*torch.max(c_X))
            right = torch.max(right, torch.max(c_W)*torch.min(c_X))
            right = torch.max(right, torch.max(c_W)*torch.max(c_X))

            right = torch.round(right/border + 0.5)*border


            bins = (right - left)/border + 1
            bins = bins.detach().int()

            c_Z = torch.linspace(left, right, bins)
            f_Z = torch.abs(c_Z - c_Z)


            for i in range(torch.numel(c_Z)):
                z = c_Z[i]

                idx_nonzero = c_X != 0


                w = torch.round( (z/c_X[idx_nonzero])*(1/border) )

                offset_w = torch.round( torch.min(c_W/border) )
                pos = w - offset_w + 1

                min_pos = -1
                max_pos = torch.numel(c_W)

                idx_exist = (pos > min_pos) & (pos < max_pos)
                pos = pos[idx_exist]

                pos = pos.long()


                x = c_X[idx_nonzero]
                x = x[idx_exist]


                p_x = f_X[idx_nonzero]
                p_x = p_x[idx_exist]

                f_Z[i] = torch.sum( p_x.mul(f_W[pos]).mul(torch.abs(1/x)) )



                deltax = 2.0*border
                x_l = 0 - 1.0*border
                x_r = 0 + 1.0*border

                w_l = z/x_l
                w_r = z/x_r

                w_l = torch.round(w_l/border)*border
                w_r = torch.round(w_r/border)*border

                idx_l = c_W == w_l
                idx_r = c_W == w_r

                p_W = (f_W[idx_l] + f_W[idx_r])/deltax

                # 这个值还需要讨论,公式推导的是大于1.13
                p_W = p_W*2.26


                if torch.numel(p_W) == 0:
                    p_W = torch.FloatTensor([0.0])

                idx_0 = c_X == 0

                value = f_X[idx_0] * p_W

                if torch.numel(value) == 0:
                    value = torch.FloatTensor([0.0])


                f_Z[i] = f_Z[i] + value

            f_Z = f_Z/torch.sum(f_Z)

        return c_Z, f_Z


    @staticmethod
    def backward(ctx, grad_output, grad_output2):

        print("bardward")
        return grad_output + 1, grad_output + 2, grad_output + 3, grad_output, None



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



    prodis_net = ProductDistribution.apply

    prodis_old = ProductDis_Old.apply

    prodis_back = ProductDis_backup.apply

    prodis_test = ProductDis.apply


    with torch.no_grad():
        num = 1000000
        border = 0.01


        X = torch.empty(num).normal_(mean = 0, std = 1)
        W = torch.empty(num).normal_(mean = 10, std = 1)

        Z = X.mul(W)


        c_X, f_X = getHist_plus(X, border)
        c_W, f_W = getHist_plus(W, border)

        c_Z, f_Z = getHist_plus(Z, border)

        c_Z_ = c_Z.clone()
        f_Z_ = torch.abs(f_Z - f_Z)


        c_Z2 = c_Z.clone()
        f_Z2 = torch.abs(f_Z - f_Z)


        c_Z = c_Z.to(device)
        f_Z = f_Z.to(device)


        c_X = c_X.to(device)
        f_X = f_X.to(device)

        c_W = c_W.to(device)
        f_W = f_W.to(device)

        c_Z_ = c_Z_.to(device)
        f_Z_ = f_Z_.to(device)


        starttime = time.time()
        c_Z_, f_Z_ = prodis_test(c_X, f_X, c_W, f_W, c_Z_, f_Z_, border)
        endtime = time.time()




        value = corDisVal(c_Z, f_Z, c_Z_, f_Z_)

#        checkcor = corDisVal(c_Z_, f_Z_, c_Z2, f_Z2)
        print("total time:", endtime - starttime)

        print("sum distribution:", torch.sum(f_Z_))

        print("correlation:", value)

        print("c_Z_.shape:", c_Z_.shape)

#        print("check cor value:", checkcor)



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










if __name__ == '__main__':
    main()
