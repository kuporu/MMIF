function [ entropy1, entropy2 ] = imageEntropy( im1,im2 )
%IMAGEENTROPY �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
[~,~,indrow] = unique(im1(:)); % ������ӳ��Ϊ����
[~,~,indcol] = unique(im2(:)); 

indrow = double(indrow(:)) + 1;
indcol = double(indcol(:)) + 1; % matlab�е�������0��ʼ

row = accumarray(indrow, 1); % ������״��[indrow indcol]һ��
row = row / numel(indrow); % ͳ��**����**������**����**(ͼ��row*col)�õ�����

col = accumarray(indcol, 1); % ������״��[indrow indcol]һ��
col = col / numel(indcol); % ͳ��**����**������**����**(ͼ��row*col)�õ�����


indNoZero = row ~= 0;
row = row(indNoZero);

indNoZero = col ~= 0;
col = col(indNoZero);


% ������
entropy1 = -sum(row.*log2(row));
entropy2 = -sum(col.*log2(col));


end

