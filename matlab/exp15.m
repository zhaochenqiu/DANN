clear all
close all
clc


rng(0)

num = 10000000;
border = 0.01;


X = normrnd(0, 2, [num, 1]);
% W = rand(num, 1);
% W = randn(num, 1)*100;
W = normrnd(-2, 4, [num, 1]);



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

% product distribution simulation experiment





% line_t = 1:border:2000;
% f_t = 1:border:2000;




line_t = min(line_Z):border:max(line_Z);
f_t = min(line_Z):border:max(line_Z);


for j = 1:max(size(line_t))
    [j max(size(line_t))]

    z = line_t(j);
    p_all = 0;
    for i = 1:max(size(line_X))
        x = line_X(i);

        if x ~= 0
            p_x = f_X(find(line_X == x));

            w = round( (z/x)*(1/border))*border;
            pos = find(line_W == w);

            p_w = 0;
            if min(size(pos)) > 0
                p_w = p_w + f_W(pos);
            end

            temp = p_x*p_w*(1/abs(x));
            p_all = p_all + temp;

            if z == 1
                [z x w pos p_x p_w temp p_all]
            end
        else
            x_l = x - 1;
            x_r = x + 1;

            x = x_l;
            w = round( (z/x)*(1/border))*border;
            pos = find(line_W == w);

            p_w = 0;
            if min(size(pos)) > 0
                p_w = p_w + f_W(pos);
            end

            temp_l = p_x*p_w*(1/abs(x));

            x = x_r;
            w = round( (z/x)*(1/border))*border;
            pos = find(line_W == w);

            p_w = 0;
            if min(size(pos)) > 0
                p_w = p_w + f_W(pos);
            end

            temp_r = p_x*p_w*(1/abs(x));

            p_all = p_all + (temp_l + temp_r)/2;
        end
    end

%     if z == 0
%         p_x_0 = sum(X == 0)/num;
%         p_w_0 = sum(X == 0)/num;
% 
%         p_all = p_all + p_x_0*(1 - p_w_0) + (1 - p_x_0)*p_w_0 + p_x_0*p_w_0;
%     end

    f_t(j) = p_all;
end





figure
subplot(1, 4, 1)
bar(line_X, f_X, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
subplot(1, 4, 2)
bar(line_W, f_W, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
subplot(1, 4, 3)
bar(line_Z, f_Z, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
ylim([0 0.15])
xlim([-50 50])
subplot(1, 4, 4)
bar(line_t, f_t, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
ylim([0 0.15])
xlim([-50 50])




% f_Z_t = abs(f_t - f_t);
% f_Z_t( round( line_Z * (1/border) )) = f_Z;
% 
% 
% p_value = sum(f_Z_t .* f_t)/(sum(f_Z_t.*f_Z_t)^(0.5) * sum(f_t.*f_t)^(0.5))


p_value = sum(f_Z .* f_t)/(sum(f_Z .* f_Z)^(0.5) * sum(f_t .* f_t)^(0.5));
p_value
