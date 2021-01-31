clear all
close all
clc

num = 100000;
border = 0.01;


X = normrnd(4, 1, [num, 1]);
W = rand(num, 1);
% W = randn(num, 1)*100;
% W = normrnd(10, 1, [num, 1]);


% X = round(X);
% Y = round(Y);



Z = X .* W;

figure
subplot(1, 3, 1)
hist(X, min(X):border:max(X))
subplot(1, 3, 2)
hist(W, min(W):border:max(W))
subplot(1, 3, 3)
hist(Z, min(Z):border:max(Z))

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
