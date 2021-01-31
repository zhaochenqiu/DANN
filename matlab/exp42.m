clear all
close all
clc


addpath('./function')

% rng(0)

num = 1000000;



X   = normrnd(0, 4, [num, 1]);
W   = normrnd(0, 8, [num, 1]);




X1 = X + 50;
W1 = W + 50;

Z1 = X1 .* W1;
Z1 = Z1 - 2500;


T = (X + W)*50;


% Z1 和 T是不独立的，所以你不能用随机去打乱他们,
% 也不可以用subtractDis 函数，因为这个函数假设了两个随机变量是相互独立的
Z2 = Z1 - T;


value = 10;

v2 = sum( Z2 < value )/num;
v2



idx = randperm(num);
Z2 = Z1(idx) - T;

v2 = sum( Z2 < value )/num;
v2

