clear all
close all
clc

left = 0.000001;

x1 = left:0.00001:1;
y1 = 1 ./ x1;


x2 = left:0.00001:1;
y2 = 2 ./ x2;


x3 = left:0.00001:1;
y3 = 3 ./ x3;

% figure

colors = rand(3, 3);


figure
plot(x1, y1, 'Color', colors(1,:))
hold on
plot(x2, y2, 'Color', colors(2,:))
hold on
plot(x3, y3, 'Color', colors(3,:))

ylim([0, 100])
