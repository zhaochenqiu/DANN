clear all
close all
clc


rng(0)

num = 100000;
border = 1;


X = normrnd(10, 2, [num, 1]);
% W = rand(num, 1);
% W = randn(num, 1)*100;
W = normrnd(20, 4, [num, 1]);



Z = X .* W;


line_X = round(min(X)):border:round(max(X));
f_X = hist(X, line_X);
f_X = f_X/num;

line_W = round(min(W)):border:round(max(W));
f_W = hist(W, line_W);
f_W = f_W/num;

line_Z = round(min(Z)):border*1:round(max(Z));
f_Z = hist(Z, line_Z);
f_Z = f_Z/num;





X = round(X*(1/border))*border;
W = round(W*(1/border))*border;



% assume z = 200
z = 200;
p_all = 0;



% product distribution simulation experiment

line_t = 1:border:2000;
f_t = 1:border:2000;


for j = 1:max(size(line_t))
    z = line_t(j);
    p_all = 0;
    for i = 1:max(size(line_X))
        x = line_X(i);

        p_x = f_X(find(line_X == x));

        w = round( (z/x)*(1/border))*border;
        pos = find(line_W == w);

        p_w = 0;
        if min(size(pos)) > 0
            p_w = p_w + f_W(pos);
        end

        temp = p_x*p_w*(1/abs(x));
        p_all = p_all + temp;
    end

    f_t(j) = p_all;
end





figure
subplot(1, 4, 1)
bar(line_X, f_X, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
subplot(1, 4, 2)
bar(line_W, f_W, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
subplot(1, 4, 3)
bar(line_Z, f_Z, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
ylim([0 0.02])
xlim([0 1000])
subplot(1, 4, 4)
bar(line_t, f_t, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
ylim([0 0.02])
xlim([0 1000])




f_Z_t = abs(f_t - f_t);
f_Z_t( round( line_Z * (1/border) )) = f_Z;


p_value = sum(f_Z_t .* f_t)/(sum(f_Z_t.*f_Z_t)^(0.5) * sum(f_t.*f_t)^(0.5))
