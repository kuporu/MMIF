% ����Ҷ�ͼ���������plotͼ���Ҫ�أ�����curvelet�任��
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

