clear all
close all
clc

addpath('./function')

rng(0)

num = 100000;
border = 1;


X = normrnd(10, 2, [num, 1]);
% W = rand(num, 1);
% W = randn(num, 1)*100;
W_t = normrnd(20, 4, [num, 1]);



Z_t = X .* W_t;

[f_Z_t c_Z_t] = getHist(Z_t, border);

figure
bar(c_Z_t, f_Z_t, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])






% 
% 
% 
% line_X = round(min(X)):border:round(max(X));
% f_X = hist(X, line_X);
% f_X = f_X/num;
% 
% line_W = round(min(W)):border:round(max(W));
% f_W = hist(W, line_W);
% f_W = f_W/num;
% 
% line_Z = round(min(Z)):border*1:round(max(Z));
% f_Z = hist(Z, line_Z);
% f_Z = f_Z/num;
% 
% 
% 
% 
% border_t = border/100;
% 
% line_X_t = round(min(X)):border_t:round(max(X));
% f_X_t = hist(X, line_X_t);
% f_X_t = f_X_t/num;
% 
% line_W_t = round(min(W)):border_t:round(max(W));
% f_W_t = hist(W, line_W_t);
% f_W_t = f_W_t/num;
% 
% line_Z_t = round(min(Z)):border_t:round(max(Z));
% f_Z_t = hist(Z, line_Z_t);
% f_Z_t = f_Z_t/num;
% 
% 
% 
% 
% % X = round(X*(1/border))*border;
% % W = round(W*(1/border))*border;
% % 
% 
% 
% % assume z = 200
% 
% % product distribution simulation experiment
% 
% 
% 
% 
% 
% % line_t = 1:border:2000;
% % f_t = 1:border:2000;
% 
% 
% 
% 
% line_t = min(line_Z):border:max(line_Z);
% f_t = min(line_Z):border:max(line_Z);
% 
% 
% for j = 1:max(size(line_t))
% %    [j max(size(line_t))]
% 
%     z = line_t(j);
%     p_all = 0;
% 
%     for i = 1:max(size(line_X))
%         x = line_X(i);
% 
%         if x ~= 0
%             p_x = f_X(find(line_X == x));
% 
%             w = round( (z/x)*(1/border))*border;
%             pos = find(line_W == w);
% 
%             p_w = 0;
%             if min(size(pos)) > 0
%                 p_w = p_w + f_W(pos);
%             end
% 
%             temp = p_x*p_w*(1/abs(x));
%             p_all = p_all + temp;
% 
%         else
% 
%             x_l = x - 0.5;
%             x_r = x + 0.5;
% 
%  
%             p_cum = 0;
% 
% 
%             var_x = x_l:border_t:x_r;
% 
%             for u = 1:max(size(var_x))
%                 x = var_x(u);
% 
%                 if x ~= 0
%                     p_x = f_X_t(find( abs(line_X_t - x) < border_t*0.5  ));
% 
%                     w = round( (z/x)*(1/border_t) )*border_t;
%                     pos = find( abs(line_W_t - w) < border_t*0.5 );
% 
%                     p_w = 0;
%                     if min(size(pos)) > 0
%                         p_w = p_w + f_W_t(pos);
%                     end
% 
%                     temp = p_x*p_w*(1/abs(x));
%                     p_cum = p_cum + temp;
% 
%                  else
% 
% 
%                     old_x = x;
%                     x = old_x - border_t;
% 
% 
%                     p_x = f_X_t(find( abs(line_X_t - x) < border_t*0.5  ));
% 
%                     w = round( (z/x)*(1/border_t) )*border_t;
%                     pos = find( abs(line_W_t - w) < border_t*0.5 );
% 
%                     p_w = 0;
%                     if min(size(pos)) > 0
%                         p_w = p_w + f_W_t(pos);
%                     end
% 
%                     temp_l = p_x*p_w*(1/abs(x));
% 
%                     x = old_x + border_t;
% 
% 
%                     p_x = f_X_t(find( abs(line_X_t - x) < border_t*0.5  ));
% 
%                     w = round( (z/x)*(1/border_t) )*border_t;
%                     pos = find( abs(line_W_t - w) < border_t*0.5 );
% 
%                     p_w = 0;
%                     if min(size(pos)) > 0
%                         p_w = p_w + f_W_t(pos);
%                     end
% 
%                     temp_r = p_x*p_w*(1/abs(x));
% 
% 
% 
%                     p_cum = p_cum + (temp_l + temp_r)/2;
%                 end
%             end
% 
% 
% 
%             p_all = p_all + p_cum;
%         end
%     end
% 
%     f_t(j) = p_all;
% end
% 
% 
% 
% 
% 
% figure
% subplot(1, 4, 1)
% bar(line_X, f_X, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
% subplot(1, 4, 2)
% bar(line_W, f_W, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
% subplot(1, 4, 3)
% bar(line_Z, f_Z, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
% ylim([0 0.16])
% % ylim([0 0.05])
% % xlim([0 500])
% subplot(1, 4, 4)
% bar(line_t, f_t, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
% ylim([0 0.16])
% % ylim([0 0.05])
% % xlim([0 500])
% 
% 
% 
% 
% % f_Z_t = abs(f_t - f_t);
% % f_Z_t( round( line_Z * (1/border) )) = f_Z;
% % 
% % 
% % p_value = sum(f_Z_t .* f_t)/(sum(f_Z_t.*f_Z_t)^(0.5) * sum(f_t.*f_t)^(0.5))
% 
% 
% p_value = sum(f_Z .* f_t)/(sum(f_Z .* f_Z)^(0.5) * sum(f_t .* f_t)^(0.5));
% p_value
% 
% sum(f_t)
