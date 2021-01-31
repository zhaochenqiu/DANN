clear all
close all
clc

num = 10000000;

data = normrnd(0, 1, [num, 1]);


figure
hist(data, -5:0.01:5)




figure
hist1 = hist(data, -5:0.01:5)
bar(hist1, 1)
