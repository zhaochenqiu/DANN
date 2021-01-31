clear all
close all
clc


rng(0)

num = 10000;
border = 0.1;


X = normrnd(10, 1, [num, 1]);
% W = rand(num, 1);
% W = randn(num, 1)*100;
W = normrnd(20, 1, [num, 1]);


% X = round(X);
% Y = round(Y);


Z = X .* W;


line_X = min(X):border:max(X);
f_X = hist(X, line_X);

line_W = min(W):border:max(W);
f_W = hist(W, line_W);

line_Z = min(Z):border:max(Z);
f_Z = hist(Z, line_Z);


% 最后只要长度足够大就没问题
num = max(size(line_Z));
f_Z_ = zeros(1, num);


threshold = 1;

for i = 1:num
    z = line_Z(i);

    p_z = 0;
    for j = 1:max(size(line_X))
        x = line_X(j);
        p_x = f_X(j);

        w = z/x;

        p_w = 0;
        if min(abs(line_W - w)) < threshold
            pos = find(min(abs(line_W - w)) == abs(line_W - w));
            p_w = f_W(pos);
        end

        f_Z_(i) = p_x*p_w*( 1/abs(x) );
    end

    [i num]
end








figure
subplot(1, 4, 1)
bar(f_X)
subplot(1, 4, 2)
bar(f_W)
subplot(1, 4, 3)
bar(f_Z)
subplot(1, 4, 4)
bar(f_Z_)

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
