clear all
close all
clc


addpath('./function')

rng(0)

num = 1000000;


X = normrnd(0, 4, [num, 1]);
Y = normrnd(0, 2, [num, 1]);


Z = X .* Y;


border = 0.1;



[f_X c_X] = getHist_plus(X, border);
[f_Y c_Y] = getHist_plus(Y, border);
[f_Z c_Z] = getHist_plus(Z, border);



[c_X_ f_X_] = plusVarCon(c_X, f_X, 50);
[c_Y_ f_Y_] = plusVarCon(c_Y, f_Y, 50);


[c_Z_ f_Z_] = productDis_plus(c_X_, f_X_, c_Y_, f_Y_);

MX = 50*X;
NY = 50*Y;

[f_MX c_MX] = getHist_plus(MX, border);
[f_NY c_NY] = getHist_plus(NY, border);

tic
[c_Z_ f_Z_] = differenceDis_plus(c_Z_, f_Z_, c_MX, f_MX);
[c_Z_ f_Z_] = differenceDis_plus(c_Z_, f_Z_, c_NY, f_NY);
time = toc


figure
subplot(1, 2, 1)
bar(c_Z_, f_Z_)

subplot(1, 2, 2)
bar(c_Z, f_Z)




% 
% 
% Z_t = (X + 50) .* (Y + 50);
% 
% [f_Z_t c_Z_t] = getHist_plus(Z_t, border);
% 
% 
% p_value = corDisVal(c_Z, f_Z, c_Z_t, f_Z_t);
% p_value
% 
% 
% 



% [c_Z f_Z] = plusVarCon(c_Z, f_Z, -2500);
% 
% 
% 
% 
% border = 0.01;
% 
% [f_X c_X] = getHist_plus(X, border);
% [f_Y c_Y] = getHist_plus(Y, border);
% 
% [c_MX f_MX] = prodVarCon(c_X, f_X, 50);
% [c_NY f_NY] = prodVarCon(c_Y, f_Y, 50);
% 
% 
% [c_Z f_Z] = differenceDis_plus(c_Z, f_Z, c_MX, f_MX);
% [c_Z f_Z] = differenceDis_plus(c_Z, f_Z, c_NY, f_NY);
% 
% 
% 
% 
% figure
% bar(c_Z, f_Z)







% 
% W   = normrnd(0, 8, [num, 1]);
% 
% Z = X .* W;
% 
% 
% border = 0.1;
% 
% 
% [f_Z c_Z] = getHist_plus(Z, border);
% 
% 
% 
% 
% 
% 
% 
% X1 = X + 50;
% W1 = W + 50;
% 
% Z1 = X1 .* W1;
% Z1 = Z1 - 2500;
% Z1 = Z1 - 50*(X + W);
% 
% [f_Z1 c_Z1] = getHist_plus(Z1, border);
% 
% 
% 
% 
% 
% figure
% subplot(1, 2, 1)
% bar(c_Z, f_Z)
% subplot(1, 2, 2)
% bar(c_Z1, f_Z1)
% 
