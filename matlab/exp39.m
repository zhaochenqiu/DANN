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
Z1 = Z1 - 2500;


T = (X + W)*50;




[f_T c_T] = getHist_plus(T, border);
[f_Z1 c_Z1] = getHist_plus(Z1, border);





Z2 = Z1 - T;
[f_Z2 c_Z2] = getHist_plus(Z2, border);


tic
[c_Z1_ f_Z1_] = differenceDis_plus(c_Z1, f_Z1, c_T, f_T);
time = toc



figure
subplot(1, 5, 1)
bar(c_Z1, f_Z1)
subplot(1, 5, 2)
bar(c_T, f_T)
subplot(1, 5, 3)
bar(c_Z1_, f_Z1_)
subplot(1, 5, 4)
bar(c_Z, f_Z)
subplot(1, 5, 5)
bar(c_Z2, f_Z2)





% figure
% subplot(1, 3, 1)
% bar(c_Z, f_Z)
% subplot(1, 3, 2)
% bar(c_Z1, f_Z1)
% subplot(1, 3, 3)
% bar(c_T, f_T)

p_value = corDisVal(c_Z, f_Z, c_Z1, f_Z1);
p_value
