clear;
close all;
clc;
fileFolder = fullfile('D:/code/MMIF/img/');
dirOutput=dir(fullfile(fileFolder,'*.jpg'));
fileNames = {dirOutput.name};
[m,n] = size(fileNames);
maxI = zeros(n,1);
minI = zeros(n,1);
maxQ = zeros(n,1);
minQ = zeros(n,1);
for i=1:n
    [maxI(i), minI(i), maxQ(i), minQ(i)] = filterMMIF(imread(fileNames{1,i}));
end
maxi = mean(maxI);
mini = mean(minI);
maxq = mean(maxQ);
minq = mean(minQ);

% NTSCÉ«²ÊÄ£ÐÍ
% imgN = zeros(size(img));
% imgN(:,:,2) = 0.596 * double(img(:,:,1)) - 0.274 * double(img(:,:,2)) - 0.322 * double(img(:,:,3));
% imgN(:,:,3) = 0.211 * double(img(:,:,1)) - 0.523 * double(img(:,:,2)) + 0.312 * double(img(:,:,3));
% maxNQ1 = max(max(imgN(:,:,2)));
% minNQ1 = min(min(imgN(:,:,2)));
% maxNI1 = max(max(imgN(:,:,3)));
% minNI1 = min(min(imgN(:,:,3)));

% [maxI, minI, maxQ, minQ] = filterMMIF(img);
% 
clearvars -except maxi mini maxq minq
