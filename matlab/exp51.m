clear all
close all
clc


%addpath('~/projects/matrix/common/')
%addpath('./function/')

addpath('D:\projects\matrix\common')
addpath('D:\projects\matrix\bgs_cnn_plus\function\bayesian')


addpath('~/projects/matrix/common')
addpath('~/projects/matrix/bgs_cnn_plus/function/bayesian')

%addpath('./old/');


% gt_pa = '~/dataset/dataset2014/dataset/baseline/highway/groundtruth/';
gt_pa = 'D:\dataset\dataset2014\dataset\dynamicBackground\fountain01\groundtruth'
gt_ft = 'png';


re_pa = '../result_old/';
re_pa = 'D:\projects\lab\fgimgs_fountain01_v15'
re_ft = 'png';


im_pa = 'D:\dataset\dataset2014\dataset\dynamicBackground\fountain01\input'
im_ft = 'jpg';



[files_gt fullfiles_gt] = loadFiles_plus(gt_pa, gt_ft);
[files_re fullfiles_re] = loadFiles_plus(re_pa, re_ft);
[files_im fullfiles_im] = loadFiles_plus(im_pa, im_ft);

frames = max(size(fullfiles_gt));
store = [];

TP_sum = 0;
FP_sum = 0;
TN_sum = 0;
FN_sum = 0;


TP_sum_bay = 0;
FP_sum_bay = 0;
TN_sum_bay = 0;
FN_sum_bay = 0;

for i = 1085:frames
    fgim = double(imread(fullfiles_re{i}));
    gtim = double(imread(fullfiles_gt{i}));
    im   = double(imread(fullfiles_im{i}));


    bayfgim = bayesRefine(im, fgim, 3, 0.6);
    bayfgim = bayesRefine(im, bayfgim, 3, 0.6);
    bayfgim = bayesRefine(im, bayfgim, 3, 0.6);



    [TP FP FN TN] = evalution_entry(fgim,gtim);

    [TP_bay FP_bay FN_bay TN_bay] = evalution_entry(bayfgim, gtim);

    Re = TP/(TP + FN);
    Pr = TP / (TP + FP);
    Fm = (2*Pr*Re)/(Pr + Re);

    TP_sum = TP_sum + TP;
    FP_sum = FP_sum + FP;
    TN_sum = TN_sum + TN;
    FN_sum = FN_sum + FN;

    TP_sum_bay = TP_sum_bay + TP_bay;
    FP_sum_bay = FP_sum_bay + FP_bay;
    TN_sum_bay = TN_sum_bay + TN_bay;
    FN_sum_bay = FN_sum_bay + FN_bay;





    
    Re = TP_sum/(TP_sum + FN_sum);
    Pr = TP_sum / (TP_sum + FP_sum);
    Fm = (2*Pr*Re)/(Pr + Re);


    Re_bay = TP_sum_bay/(TP_sum_bay + FN_sum_bay);
    Pr_bay = TP_sum_bay / (TP_sum_bay + FP_sum_bay);
    Fm_bay = (2*Pr_bay*Re_bay)/(Pr_bay + Re_bay);



    [i Re Pr Fm Re_bay Pr_bay Fm_bay]


    displayMatrixImage(1, 1, 3, im, fgim, bayfgim)
%    Re
%    Pr
%    Fm


    
end

Re = TP_sum/(TP_sum + FN_sum);
Pr = TP_sum / (TP_sum + FP_sum);
Fm = (2*Pr*Re)/(Pr + Re);


Re_bay = TP_sum_bay/(TP_sum_bay + FN_sum_bay);
Pr_bay = TP_sum_bay / (TP_sum_bay + FP_sum_bay);
Fm_bay = (2*Pr_bay*Re_bay)/(Pr_bay + Re_bay);


    
% [Re Pr Fm]

[i Re Pr Fm Re_bay Pr_bay Fm_bay]



