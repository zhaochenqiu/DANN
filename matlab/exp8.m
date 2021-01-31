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





X = round(X);
W = round(W);



% assume z = 200
z = 200;
p_all = 0;



% product distribution simulation experiment

line_t = 1:2000;
f_t = 1:2000;


for j = 1:max(size(line_t))
    z = line_t(j);
    p_all = 0;
    for i = 1:max(size(line_X))
        x = line_X(i);

        p_x = f_X(find(line_X == x));

        w = round(z/x);
        pos = find(line_W == w);

        p_w = 0;
        if min(size(pos)) > 0
            p_w = p_w + f_W(pos);
        end

        temp = p_x*p_w*(1/abs(x));
        p_all = p_all + temp;

%        [x p_x w p_w temp]
    end

    f_t(j) = p_all;
%    p_all
end







% 
% 
% threshold = 1;
% 
% for i = 1:num
%     z = line_Z(i);
% 
%     p_z = 0;
%     for j = 1:max(size(line_X))
%         x = line_X(j);
%         p_x = f_X(j);
% 
%         w = z/x;
% 
%         p_w = 0;
%         if min(abs(line_W - w)) < threshold
%             pos = find(min(abs(line_W - w)) == abs(line_W - w));
%             p_w = f_W(pos);
%         end
% 
%         f_Z_(i) = p_x*p_w*( 1/abs(x) );
%     end
% 
%     [i num]
% end








figure
subplot(1, 4, 1)
bar(line_X, f_X, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
subplot(1, 4, 2)
bar(line_W, f_W, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
subplot(1, 4, 3)
bar(line_Z, f_Z, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
ylim([0 0.02])
xlim([0 2000])
subplot(1, 4, 4)
bar(line_t, f_t, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
ylim([0 0.02])
xlim([0 2000])


% 
% figure
% subplot(1, 3, 1)
% hist(X, min(X):border:max(X))
% subplot(1, 3, 2)
% hist(W, min(W):border:max(W))
% subplot(1, 3, 3)
% hist(Z, min(Z):border:max(Z))

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
