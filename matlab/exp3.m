clear all
close all
clc

num = 10000;

X = normrnd(1, 1, [num, 1]);
Y = normrnd(4, 1, [num, 1]);


Z = X .* Y;


line = -50:0.1:50;



figure
subplot(1, 3, 1)
hist(X, line)
subplot(1, 3, 2)
hist(Y, line)
subplot(1, 3, 3)
hist(Z, line)

% 
% data = normrnd(0, 1, [num, 1]);
% 
% 
% figure
% hist(data, -5:0.01:5)
% 
% 
% 
% 
% figure
% hist1 = hist(data, -5:0.01:5)
% bar(hist1, 1)
