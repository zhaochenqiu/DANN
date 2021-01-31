clear all
close all
clc


addpath('./function')

% rng(0)

num = 1000000;


X   = normrnd(0, 4, [num, 1]);
W   = normrnd(0, 8, [num, 1]);

Z = X .* W;

border = 0.1;

[f_X c_X] = getHist_plus(X, border);
[f_W c_W] = getHist_plus(W, border);

[f_Z c_Z] = getHist_plus(Z, border);

f_Z_old = f_Z;
c_Z_old = c_Z;


% figure
% subplot(1, 3, 1)
% bar(c_X, f_X)
% subplot(1, 3, 2)
% bar(c_W, f_W)
% subplot(1, 3, 3)
% bar(c_Z, f_Z)




X   = normrnd(0, 4, [num, 1]);
W   = normrnd(0, 8, [num, 1]);

Z_ = (X + 20) .* W;



% Z = Z(randperm(max(size(Z))));

T = 20*W;


[f_T c_T] = getHist_plus(T, border);
[f_X c_X] = getHist_plus(X, border);
[f_W c_W] = getHist_plus(W, border);

[f_Z_ c_Z_] = getHist_plus(Z_, border);



Z = Z_ - T;

[c_Z_fun f_Z_fun] = subDis_func(c_Z_, f_Z_, c_T, f_T, c_X, f_X, c_Z);




figure
subplot(1, 5, 1)
bar(c_X, f_X)
subplot(1, 5, 2)
bar(c_W, f_W)
subplot(1, 5, 3)
bar(c_Z_, f_Z_)
subplot(1, 5, 4)
bar(c_Z, f_Z)
subplot(1, 5, 5)
bar(c_Z_fun, f_Z_fun)


sum(f_Z_fun)
corDisVal(c_Z_fun, f_Z_fun, c_Z, f_Z)


% 
% 
% corDisVal(c_Z_old, f_Z_old, c_Z, f_Z)
% 相关性，极强，说明应该是可行的

% val = 0.1;
% 
% idx1 = abs(T) < val;
% idx2 = abs(Z_) < val;
% idx3 = abs(Z) < val*val;
% 
% 
% sum(idx1)
% sum(idx2)
% sum(idx3)
% 
% 
% sum(idx1 == idx2) / num
% sum(idx1 == idx3) / num




% border = 0.1;
% 
% [f_Z c_Z] = getHist_plus(Z, border);
% 
% 
% figure
% subplot(1, 4, 1)
% bar(c_X, f_X)
% subplot(1, 4, 2)
% bar(c_W, f_W)
% subplot(1, 4, 3)
% bar(c_Z_, f_Z_)
% subplot(1, 4, 4)
% bar(c_Z, f_Z)
% 
% 
% corDisVal(c_Z_old, f_Z_old, c_Z, f_Z)
