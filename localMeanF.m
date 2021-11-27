function [ mean ] = localMeanF( A, i )
%LOCALMEANF 此处显示有关此函数的摘要
%   此处显示详细说明
len = 2 * i + 1;
one = ones(len, len);
one(i + 1, i + 1) = 0;
% A周围填充1，再使用valid类型(只计算卷积核全覆盖的地方)卷积
[m, n] = size(A);
APadding = ones(m + 2 * i, n + 2 * i);
for o = i + 1: i + m
    for p = i + 1: i + n
        APadding(o, p) = A(o - i, p - i);
    end
end

B = conv2(APadding, one, 'valid');
count = 0;
sum = 0;
target = (2 * i + 1) ^ 2 - 1;
for o = 1: m
    for p = 1: n
        if A(o, p) < 1
            sum = sum + A(o, p);
            count = count + 1;
            continue;
        end
        if A(o, p) == 1 && B(o, p) < target
            sum = sum + A(o, p);
            count = count + 1;
        end
    end
end

mean = sum / count;

end

