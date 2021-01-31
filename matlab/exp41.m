clear all
close all
clc

num = 1000000;

X = normrnd(0, 4, [num, 1]);
Y = normrnd(0, 8, [num, 1]);



Z1 = X - Y;


value = 0.1;
v1 = sum(Z1 < value)/num;
v1


Z2 = X(randperm(num)) - Y;
v2 = sum(Z2 < value)/num;
v2

