clear all
close all
clc

addpath('./function')

rng(0)

num = 10000000;


X   = normrnd(20, 4, [num, 1]);
W   = normrnd(40, 8, [num, 1]);

Z = X .* W;


border = 0.1;


[f_X c_X] = getHist_plus(X, border);
[f_W c_W] = getHist_plus(W, border);
[f_Z c_Z] = getHist_plus(Z, border);


tic
[c_Z_s f_Z_s] = productDis_plus(c_X, f_X, c_W, f_W);
time = toc


figure
subplot(1, 2, 1)
bar(c_Z, f_Z)

subplot(1, 2, 2)
bar(c_Z_s, f_Z_s)



p_value = corDisVal(c_Z, f_Z, c_Z_s, f_Z_s);
p_value
