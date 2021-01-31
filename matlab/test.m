value = 10


idx = randperm(max(size(Z1)));
ZF = Z1(idx);
v1 = sum((ZF - T) < value)/num;



idx = c_Z2 < value;
v3 = sum(f_Z2(idx));


v1
v3
