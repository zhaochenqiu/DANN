clear all
close all
clc

figure
z = 2;


for i = -0.5:0.001:0.99
%    x = 1 - i;
    x = i

    p1 = exp(-0.5* (z/x)^2);
    p2 = abs(1/x);
    p1
    p2
    y = p1*p2;

    hold on
    plot(x, y, 'b.')

    input('pause')
end
