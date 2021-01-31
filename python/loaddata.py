import os
import numpy as np

# print all the elements
np.set_printoptions(threshold=np.inf)



def loadFiles_plus(path_im, keyword = ""):
    re_fs = []
    re_fullfs = []

    files = os.listdir(path_im)
    files = sorted(files)

    for file in files:
        if file.find(keyword) != -1:
            re_fs.append(file)
            re_fullfs.append(path_im + "/" + file)

    return re_fs, re_fullfs


if __name__=='__main__':

    pa_im = '/home/cqzhao/dataset/dataset2014/dataset/baseline/highway/input/'
    ft_im = '.jpg'

    fs, fullfs = loadFiles_plus(pa_im, ft_im)
    print(fs)
    print(fullfs)
