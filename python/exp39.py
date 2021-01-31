import sys
sys.path.append("/home/cqzhao/projects/matrix/")


import matplotlib.pyplot as plt


import torch


from common_py.dataIO import loadImgs_pytorch


from function.prodis import getEmptyCF
from function.prodis import getHist_plus


# print("Hello World")
#
#
# showHello()
#


# def video2tpixels(imgs, curidx):
#
#     return None



def main():


    pa_im = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/input'
    ft_im = 'jpg'

    pa_gt = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth'
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

#             c_R, f_R = getHist_plus(sub[:, 0], 1, -255, 255)
#
#             print("vec.shape = ", vec.shape)
#             print("val.shape = ", val.shape)
#
#             print("sub.shape = ", sub.shape)
#
#             print("c_R", c_R)
#             print("f_R", f_R)






#    im = imgs[1129]
#    lb = labs[1129]


    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im.detach().numpy().astype(int))

    plt.subplot(1, 2, 2)
    plt.imshow(lb.detach().numpy(), cmap='gray')


    plt.show()





if __name__ == '__main__':
    main()
