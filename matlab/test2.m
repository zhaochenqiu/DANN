l1 = [];
l2 = [];


for n = -100:1:100

    value = n;


    v1 = sum((Z1( randperm(max(size(Z1)))) - T) < value)/num;



    idx = c_Z1_ < value;
    v2 = sum(f_Z1_(idx));


    l1 = [l1 v1];
    l2 = [l2 v2];

    n

end


sum(l1 .* l2) /(sum(l1 .* l1)^(0.5) * sum(l2 .* l2)^(0.5))
