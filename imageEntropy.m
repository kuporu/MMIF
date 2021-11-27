function [ entropy1, entropy2 ] = imageEntropy( im1,im2 )
%IMAGEENTROPY 此处显示有关此函数的摘要
%   此处显示详细说明
[~,~,indrow] = unique(im1(:)); % 浮点数映射为整数
[~,~,indcol] = unique(im2(:)); 

indrow = double(indrow(:)) + 1;
indcol = double(indcol(:)) + 1; % matlab中的索引从0开始

row = accumarray(indrow, 1); % 返回形状和[indrow indcol]一样
row = row / numel(indrow); % 统计**次数**除以总**次数**(图像row*col)得到概率

col = accumarray(indcol, 1); % 返回形状和[indrow indcol]一样
col = col / numel(indcol); % 统计**次数**除以总**次数**(图像row*col)得到概率


indNoZero = row ~= 0;
row = row(indNoZero);

indNoZero = col ~= 0;
col = col(indNoZero);


% 联合熵
entropy1 = -sum(row.*log2(row));
entropy2 = -sum(col.*log2(col));


end

