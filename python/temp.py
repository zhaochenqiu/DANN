    im = imgs[curidx].reshape(row_im*column_im, byte_im)
    im = (im/255.0)*right_data
    im = torch.round(im/delta)

    c_data, f_data = getEmptyCF_plus(left, right, delta, 1)


    num_hist = round((right_data - left_data)/delta) + 1

    num_right = round(right_data/delta) + 1



    offset_right = round(right_data/delta)

    starttime = time.time()
    re_hist = torch.abs(vid_hist - vid_hist)
    for i in range(num_right):
        for b in range(byte_im):
            idx_r = im[:, b] == i
            re_hist[idx_r, (num_hist - offset_right - i ):(num_hist - i) , b] = vid_hist[idx_r, (num_hist - offset_right):num_hist, b]

    endtime = time.time()
    print("total time:", endtime - starttime)




    for i in range(row_im*column_im):
        for b in range(byte_im):
            val = i
            print(corDisVal(c_data, re_hist[val, :, b], c_data, data_vid[val, :, b]))


    val = 56478

    print(re_hist[val, :, 0])
    print(data_vid[val, :, 0])

    print(corDisVal(c_data, re_hist[val, :, 0], c_data, data_vid[val, :, 0]))


    val = 23461

    print(re_hist[val, :, 0])
    print(data_vid[val, :, 0])

    print(corDisVal(c_data, re_hist[val, :, 0], c_data, data_vid[val, :, 0]))




    val = 97123

    print(re_hist[val, :, 0])
    print(data_vid[val, :, 0])

    print(corDisVal(c_data, re_hist[val, :, 0], c_data, data_vid[val, :, 0]))

########################################################################################
########################################################################################
########################################################################################
########################################################################################


def main():


    pa_im = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/input'
    pa_im = 'D:/dataset/dataset2014/dataset/dynamicBackground/fountain01/input'
    ft_im = 'jpg'

    pa_gt = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth'
    pa_gt = 'D:/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth'
    ft_gt = 'png'

#     net_pa = '../../data/'
#     sa_pa = '../../data/results/'
    net_pa = '../../data/network_fountain01/'
    sa_pa =  '../../data/fgimgs_fountain01/'


    use_cuda = torch.cuda.is_available()


    print("------------")
    print(use_cuda)
    print("------------")

    torch.manual_seed(0)

    device = torch.device("cuda:1" if use_cuda else "cpu")

    prodis_mul = ProductDis_multi.apply

    params = {'zero_swap': True, 'zero_approx': True, 'normal': False}



    imgs = loadImgs_pytorch(pa_im, ft_im)
    labs = loadImgs_pytorch(pa_gt, ft_gt)

    left_data = -255
    right_data = 255

    left = -255
    right = 255
    delta = 1
    num_dis = 16



    frames, row_im, column_im, byte_im = imgs.shape

#     print("frames = ", frames)
#     print("row_im = ", row_im)
#     print("column_im = ", column_im)
#     print("byte_im = ", byte_im)

#    curidx = [700, 710, 720, 730, 740, 1120, 1130, 1140, 1150, 1160]
#    curidx = [710, 720, 740, 1130, 1149]
#    curidx = [720, 922, 1130, 1149, 1184]
    curidx = [1149]

    # the first index starts from 0
    curidx[:] = [i - 1 for i in curidx]

#    curidx = [1140]

    starttime = time.time()
    print("generating trainning data")
    data_vid, labs_vid = getNormalData_plus(imgs, labs, curidx, left_data, right_data, delta)
    print("completed")
    endtime = time.time()
    print("total time:", endtime - starttime)

    labs_vid = torch.round(labs_vid/255)


    starttime = time.time()
    vid_hist = getVidHist_plus(imgs, left_data, right_data, delta)
    endtime = time.time()
    print("total time:", endtime - starttime)


    starttime = time.time()
    data_hst, labs_hst = getNormalData_byHistVid(vid_hist, imgs, labs, curidx, left_data, right_data, delta)
    endtime = time.time()
    print("total time:", endtime - starttime)


    print(torch.sum(torch.abs(data_hst - data_vid)))

    print(data_hst.shape)
    print(data_vid.shape)


    num, num_c, byte = data_hst.shape
    c_data, f_data = getEmptyCF_plus(left, right, delta, 1)


    for i in range(num):
        for j in range(byte):
            print(corDisVal(c_data, data_vid[i, :, j], c_data, data_hst[i, :, j]))



