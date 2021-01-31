clear all
close all
clc

num = 10000;
border = 1;


X = normrnd(4, 1, [num, 1]);
Y = normrnd(10, 1, [num, 1]);


% X = round(X);
% Y = round(Y);



Z = X .* Y;


idx = randperm(num);
Z1 = X(idx) .* Y;


idx = randperm(num);
Z2 = X(idx) .* Y;


Z_all = reshape(X * Y', [num^2, 1]);





figure
subplot(1, 6, 1)
hist(X, min(X):border:max(X))
subplot(1, 6, 2)
hist(Y, min(Y):border:max(Y))
subplot(1, 6, 3)
hist(Z, min(Z):border:max(Z))
subplot(1, 6, 4)
hist(Z1, min(Z1):border:max(Z1))
subplot(1, 6, 5)
hist(Z2, min(Z2):border:max(Z2))
subplot(1, 6, 6)
hist(Z_all, min(Z_all):border:max(Z_all))


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
