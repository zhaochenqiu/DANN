clear all
close all
clc


addpath('./function')

rng(0)

num = 1000000;


X = normrnd(40, 4, [num, 1]);
Y = normrnd(80, 2, [num, 1]);

Z = X - Y;


border = 0.1;

[f_X c_X] = getHist_plus(X, border);
[f_Y c_Y] = getHist_plus(Y, border);
[f_Z c_Z] = getHist_plus(Z, border);






[c_Z_s f_Z_s] = differenceDis_plus(c_X, f_X, c_Y, f_Y);




figure
subplot(1, 2, 1)
bar(c_Z, f_Z);

subplot(1, 2, 2)
bar(c_Z_s, f_Z_s)


p_value = corDisVal(c_Z, f_Z, c_Z_s, f_Z_s);
p_value
