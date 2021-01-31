clear all
close all
clc


addpath('./function')

% rng(0)

num = 1000000;


X   = normrnd(0, 1, [num, 1]);
W   = normrnd(0, 1, [num, 1]);


border = 0.1;




Z = X .* W;



[f_X c_X] = getHist_plus(X, border);
[f_W c_W] = getHist_plus(W, border);

[f_Z c_Z] = getHist_plus(Z, border);


tic
[c_Z_ f_Z_] = productDis_plus(c_X, f_X, c_W, f_W);
time = toc
% [c_Z_ f_Z_] = productDis(c_X, f_X, c_W, f_W);






figure
subplot(1, 4, 1)
bar(c_X, f_X)
subplot(1, 4, 2)
bar(c_W, f_W)
subplot(1, 4, 3)
bar(c_Z, f_Z)
% ylim([0 0.14])
% xlim([-10 10])

subplot(1, 4, 4)
bar(c_Z_, f_Z_)
% ylim([0 0.14])
% xlim([-10 10])

corDisVal(c_Z, f_Z, c_Z_, f_Z_)

sum(f_Z_)


% corDisVal(c_Z_old, f_Z_old, c_Z, f_Z)
