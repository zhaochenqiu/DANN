clear all
close all
clc


addpath('./function')

% rng(0)

num = 1000000;


X   = normrnd(0, 1, [num, 1]);



Y   = normrnd(0, 1, [num, 1]);


border = 0.1;


Z = X .* Y;


X = X(randperm(max(size(X))));
Z1 = X .* Y;

Y = sort(Y);
Z2 = X .* Y;



[f_X c_X] = getHist_plus(X, border);
[f_Y c_Y] = getHist_plus(Y, border);

[f_Z c_Z] = getHist_plus(Z, border);

[f_Z1 c_Z1] = getHist_plus(Z1, border);
[f_Z2 c_Z2] = getHist_plus(Z2, border);



figure
subplot(1, 3, 1)
bar(c_Z,f_Z)
subplot(1, 3, 2)
bar(c_Z1, f_Z1)
subplot(1, 3, 3)
bar(c_Z2, f_Z2)
