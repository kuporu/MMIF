clear;
close all;
clc;
img = imread('img1524.jpg');
subplot(1,2,1);imshow(img);

% NTSC色彩模型
imgN = zeros(size(img));
% imgN(:,:,1) = 0.299 * double(img(:,:,1)) + 0.587 * double(img(:,:,2)) + 0.114 * double(img(:,:,3));
imgN(:,:,2) = 0.596 * double(img(:,:,1)) - 0.274 * double(img(:,:,2)) - 0.322 * double(img(:,:,3));
imgN(:,:,3) = 0.211 * double(img(:,:,1)) - 0.523 * double(img(:,:,2)) + 0.312 * double(img(:,:,3));
maxNQ1 = max(max(imgN(:,:,2)));
minNQ1 = min(min(imgN(:,:,2)));
maxNI1 = max(max(imgN(:,:,3)));
minNI1 = min(min(imgN(:,:,3)));

% 高斯色彩模型
imgG = zeros(size(img));
% imgG(:,:,1) = 0.299 * double(img(:,:,1)) + 0.587 * double(img(:,:,2)) + 0.114 * double(img(:,:,3));
imgG(:,:,2) = 0.3 * double(img(:,:,1)) - 0.04 * double(img(:,:,2)) - 0.35 * double(img(:,:,3));
imgG(:,:,3) = 0.34 * double(img(:,:,1)) - 0.6 * double(img(:,:,2)) + 0.17 * double(img(:,:,3));
maxH1 = max(max(imgG(:,:,2)));
minH1 = min(min(imgG(:,:,2)));
maxM1 = max(max(imgG(:,:,3)));
minM1 = min(min(imgG(:,:,3)));

%=======================================================================%

img1 = img(:,:,1);
img2 = img(:,:,2);
img3 = img(:,:,3);

% img2N(:,:,2) = 0.596 * double(img1) - 0.274 * double(img2) - 0.322 * double(img3);
% maxNI1 = max(max(img2N(:,:,2)));
% minNI1 = min(min(img2N(:,:,2)));
% img2N(:,:,3) = 0.211 * double(img1) - 0.523 * double(img2) + 0.312 * double(img3);
% maxNQ1 = max(max(img2N(:,:,3)));
% minNQ1 = min(min(img2N(:,:,3)));

% 色彩淡化
img1 = uint8(img1 * 0.2);
img2 = uint8(img2 * 0.2);
img3 = uint8(img3 * 0.2);
im = cat(3, img1, img2, img3);
subplot(1,2,2); imshow(im);

% NTST色彩模型淡化后取值范围
img2N = zeros(size(img));
img2N(:,:,2) = 0.596 * double(img1) - 0.274 * double(img2) - 0.322 * double(img3);
maxNQ2 = max(max(img2N(:,:,2)));
minNQ2 = min(min(img2N(:,:,2)));
img2N(:,:,3) = 0.211 * double(img1) - 0.523 * double(img2) + 0.312 * double(img3);
maxNI2 = max(max(img2N(:,:,3)));
minNI2 = min(min(img2N(:,:,3)));

% 高斯色彩模型淡化后取值范围
img2G = zeros(size(img));
img2G(:,:,2) = 0.3 * double(img1) - 0.04 * double(img2) - 0.35 * double(img3);
maxH2 = max(max(img2G(:,:,2)));
minH2 = min(min(img2G(:,:,2)));
img2G(:,:,2) = 0.34 * double(img1) - 0.6 * double(img2) + 0.17 * double(img3);
maxM2 = max(max(img2G(:,:,2)));
minM2 = min(min(img2G(:,:,2)));



clearvars -except maxNQ2 minNQ2 maxNI2 minNI2 maxNQ1 minNQ1 maxNI1 minNI1
