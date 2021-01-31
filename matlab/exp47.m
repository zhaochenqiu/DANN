clear all
close all
clc


addpath('./function')

% rng(0)

range = 10;



list_cor = [];
list_sum = [];


cnt_sum = 0;
cnt_cor = 0;


record_sum = 0;
record_cor = 0;

figure
for i = 1:1000

    mu1 = rand*range;
    mu2 = rand*range;
    sig1 = rand*range;
    sig2 = rand*range;

    num = 1000000;


    X   = normrnd(mu1, sig1, [num, 1]);
    W   = normrnd(mu2, sig2, [num, 1]);


    border = 0.1;

%     cnt_border = border/2;
% 
%     X = round(X / cnt_border)*cnt_border;
%     W = round(W / cnt_border)*cnt_border;
% 

    Z = X .* W;


    [f_X c_X] = getHist_plus(X, border);
    [f_W c_W] = getHist_plus(W, border);

    [f_Z c_Z] = getHist_plus(Z, border);


    tic
    [c_Z_ f_Z_] = productDis_plus(c_X, f_X, c_W, f_W);
    time = toc


    corval = corDisVal(c_Z, f_Z, c_Z_, f_Z_);

    sumval = sum(f_Z_);


    if sumval < 0.95
        cnt_sum = cnt_sum + 1;

        record_sum = [record_sum  -1 mu1 mu2 sig1 sig2 corval sumval];
    end


    if corval < 0.95
        cnt_cor = cnt_cor + 1;

        record_cor = [record_cor -1 mu1 mu2 sig1 sig2 corval sumval];
    end


    list_cor = [list_cor; corval];
    list_sum = [list_sum; sumval];

    [corval sumval]

    subplot(1, 2, 1)
    plot(1:i, list_cor, '*', 'Color', [1 0 0])

    subplot(1, 2, 2)
    plot(1:i, list_sum, 'o', 'Color', [0 0 1])

    drawnow

end


% corDisVal(c_Z_old, f_Z_old, c_Z, f_Z)
