function [ maxI, minI, maxQ, minQ ] = filterMMIF( img )
%FILTERMMIF 此处显示有关此函数的摘要
%   此处显示详细说明
img1 = img(:,:,1);
img2 = img(:,:,2);
img3 = img(:,:,3);

% 色彩淡化
img1 = uint8(img1 * 0.2);
img2 = uint8(img2 * 0.2);
img3 = uint8(img3 * 0.2);
% im = cat(3, img1, img2, img3);

% NTST色彩模型淡化后取值范围
img2N = zeros(size(img));
img2N(:,:,2) = 0.596 * double(img1) - 0.274 * double(img2) - 0.322 * double(img3);
maxQ = max(max(img2N(:,:,2)));
minQ = min(min(img2N(:,:,2)));
img2N(:,:,3) = 0.211 * double(img1) - 0.523 * double(img2) + 0.312 * double(img3);
maxI = max(max(img2N(:,:,3)));
minI = min(min(img2N(:,:,3)));


end

