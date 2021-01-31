clear all
close all
clc

addpath('./function')

rng(0)

num = 10000000;


X   = normrnd(20, 4, [num, 1]);
W   = normrnd(40, 8, [num, 1]);

Z = X .* W;


border = 0.1;


[f_X c_X] = getHist_plus(X, border);
[f_W c_W] = getHist_plus(W, border);
[f_Z c_Z] = getHist_plus(Z, border);

f_Z_t = f_Z;
c_Z_t = c_Z;




left = min(c_X);
left = min([left min(c_W)]);
left = min([left min(c_W)*min(c_X)]);
left = min([left min(c_W)*max(c_X)]);
left = min([left max(c_W)*min(c_X)]);
left = min([left max(c_W)*max(c_X)]);

left = round(left/border - 0.5)*border;



right = max(c_X);
right = max([right max(c_W)]);
right = max([right min(c_W)*min(c_X)]);
right = max([right min(c_W)*max(c_X)]);
right = max([right max(c_W)*min(c_X)]);
right = max([right max(c_W)*max(c_X)]);

right = round(right/border + 0.5)*border;


c_Z = left:border:right;
f_Z = abs(c_Z - c_Z);




for i = 1:max(size(c_Z))
    z = c_Z(i);
    p_all = 0;

    for j = 1:max(size(c_X))
        x = c_X(j);

        if x ~= 0
            p_x = f_X(j);

            y = round( (z/x)*(1/border) )*border;
            pos = find( abs(c_W - y) < border*0.5 );

            if min(size(pos)) > 0
                p_all = p_all + p_x*f_W(pos)*(1/abs(x));
            end
        end
    end

    f_Z(i) = p_all;

    [i max(size(c_Z))]
end


p_value = corDisVal(c_Z, f_Z, c_Z_t, f_Z_t);
p_value


