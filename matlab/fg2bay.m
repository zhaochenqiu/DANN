function [] = fg2bay(im_pa, im_ft, re_pa, re_ft, sv_pa, sv_ft, bay_radius, bay_rate, bay_num)


disp(im_pa)
disp(im_ft)
disp(re_pa)
disp(re_ft)
disp(sv_pa)
disp(sv_ft)
disp(bay_radius)
disp(bay_rate)
disp(bay_num)




% clear all
% close all
% clc
% 
% 
% %addpath('~/projects/matrix/common/')
% %addpath('./function/')
% 
% %addpath('D:\projects\matrix\common')
% %addpath('D:\projects\matrix\bgs_cnn_plus\function\bayesian')
% 
% 
% addpath('~/projects/matrix/common')
% addpath('~/projects/matrix/bgs_cnn_plus/function/bayesian')
% 
% %addpath('./old/');
% 
% 
% % gt_pa = '~/dataset/dataset2014/dataset/baseline/highway/groundtruth/';
% % gt_pa = 'D:\dataset\dataset2014\dataset\dynamicBackground\fountain01\groundtruth'
% % gt_ft = 'png';
% 
% gt_pa = '~/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth/';
% gt_ft = 'png';
% 
% 
% 
% 
% %re_pa = '../result_old/';
% %re_pa = 'D:\projects\lab\fgimgs_fountain01_v15'
% 
% %re_pa = '~/projects/matrix/data/fgimgs_fountain01_v20_less/';
% re_pa = '~/projects/matrix/data/fgimgs_fountain01_v34/'
% re_ft = 'png';
% 
% 
% 
% %im_pa = 'D:\dataset\dataset2014\dataset\dynamicBackground\fountain01\input'
% %im_ft = 'jpg';
% 
% im_pa = '~/dataset/dataset2014/dataset/dynamicBackground/fountain01/input';
% im_ft = 'jpg';
% 
% 
% 
% [files_gt fullfiles_gt] = loadFiles_plus(gt_pa, gt_ft);
% [files_re fullfiles_re] = loadFiles_plus(re_pa, re_ft);
% [files_im fullfiles_im] = loadFiles_plus(im_pa, im_ft);
% 
% frames = max(size(fullfiles_gt));
% store = [];
% 
% TP_sum = 0;
% FP_sum = 0;
% TN_sum = 0;
% FN_sum = 0;
% 
% 
% TP_sum_bay = 0;
% FP_sum_bay = 0;
% TN_sum_bay = 0;
% FN_sum_bay = 0;
% 
% for i = 400:frames
%     fgim = double(imread(fullfiles_re{i}));
%     gtim = double(imread(fullfiles_gt{i}));
%     im   = double(imread(fullfiles_im{i}));
% 
%     fullfiles_re{i}
%     fullfiles_gt{i}
% 
%     [TP FP FN TN] = evalution_entry(fgim,gtim);
% 
% 
%     i
%     Re = TP/(TP + FN);
%     Pr = TP / (TP + FP);
%     Fm = (2*Pr*Re)/(Pr + Re);
% 
%     [Re Pr Fm]
% 
% 
%     TP_sum = TP_sum + TP;
%     FP_sum = FP_sum + FP;
%     TN_sum = TN_sum + TN;
%     FN_sum = FN_sum + FN;
% 
%     Re = TP_sum/(TP_sum + FN_sum);
%     Pr = TP_sum / (TP_sum + FP_sum);
%     Fm = (2*Pr*Re)/(Pr + Re);
% 
%     [Re Pr Fm]
% 
% 
% 
% 
%     
% end
% 
% Re = TP_sum/(TP_sum + FN_sum);
% Pr = TP_sum / (TP_sum + FP_sum);
% Fm = (2*Pr*Re)/(Pr + Re);
% 

