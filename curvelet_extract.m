% 输入灰度图像，输出绘制plot图像的要素（经过curvelet变换）
function [c,v] = curvelet_extract(imdist, n)
    mc = fdct_usfft(imdist, 1);
    fc = [];
    l = length(mc{5});
    for i = 1:l
        fc = [fc; mc{5}{i}(:)];
    end
    fc = log10(abs(fc));
    [c, v] = hist(fc, n);
    c = c ./ sum(c);
end

