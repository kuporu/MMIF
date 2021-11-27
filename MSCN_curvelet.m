% 绘制MSCN, curvelet概率密度函数图
clc;
close all;
clear;

% MSCN
%=========================================================================%
window = fspecial('gaussian',7,7/6);
window = window/sum(sum(window));

img_r = imread('nature.png');
medical_r = imread('img1666.jpg');
img = rgb2gray(img_r);
medical = rgb2gray(medical_r);
img = double(img);
medical = double(medical);
[r, c] = size(img);
[rm, cm] = size(medical);

feat = [];
% 计算归一化亮度值
mu            = filter2(window, img, 'same');
mu_sq         = mu.*mu;
sigma         = sqrt(abs(filter2(window, img.*img, 'same') - mu_sq));
structdis     = (img-mu)./(sigma+1);

mum            = filter2(window, medical, 'same');
mu_sqm         = mum.*mum;
sigmam         = sqrt(abs(filter2(window, medical.*medical, 'same') - mu_sqm));
structdism     = (medical-mum)./(sigmam+1);

structdisL = reshape(structdis, [r * c, 1]);
structdisLm = reshape(structdism, [rm * cm, 1]);
[y, x] = hist(structdisL, 50); % y表示统计数量，x表示横坐标值
y = y ./ sum(y);
[ym, xm] = hist(structdisLm, 50);
ym = ym ./ sum(ym);
figure;
p1 = plot(x, y, 's-r', xm, ym, '-ob');
set(gca, 'Fontsize', 14);
legend('Natural image', 'Fused image');
ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.GridLineStyle = ':';
ax.GridAlpha = 0.28;
ax.YLabel.String = 'Probability';
ax.YLabel.Color = [0.3 0.3 0.3];
ax.XLabel.String = 'Normalized luminance';
ax.XLabel.Color = [0.3 0.3 0.3];
saveas(gcf, 'mscn.jpg');
%=========================================================================%

% curvelet
%=========================================================================%
% mc = fdct_usfft(medical, 1);
% fc = [];
% l = length(mc{5});
% for i = 1:l
%     fc = [fc; mc{5}{i}(:)];
% end
% fc = log10(abs(fc));
% [c,v] = hist(fc, 40);
% c = c ./ sum(c);
[c, v] = curvelet_extract(medical, 40);
[c1, v1] = curvelet_extract(img, 40);
figure;
p2 = plot(v1, c1, 's-r', v, c, '-ob');
set(gca, 'Fontsize', 14);
legend('Natural image', 'Fused image', 'Location','northwest');
ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.GridLineStyle = ':';
ax.GridAlpha = 0.28;
ax.YLabel.String = 'Probability';
ax.YLabel.Color = [0.3 0.3 0.3];
ax.XLabel.String = 'Logarithm (base 10) of magnitude of curvelet coefficients';
ax.XLabel.Color = [0.3 0.3 0.3];
saveas(gcf, 'cur.jpg');
