clear all
close all
clc


addpath('./function')

rng(0)

num = 1000000;



X   = normrnd(0, 4, [num, 1]);
W   = normrnd(0, 8, [num, 1]);

Z = X .* W;


border = 0.1;


[f_Z c_Z] = getHist_plus(Z, border);







X1 = X + 50;
W1 = W + 50;

Z1 = X1 .* W1;
Z1 = Z1 - 50*(X + W);


Z1 = Z1 - 2500;

[f_Z1 c_Z1] = getHist_plus(Z1, border);





figure
subplot(1, 2, 1)
bar(c_Z, f_Z)
subplot(1, 2, 2)
bar(c_Z1, f_Z1)

