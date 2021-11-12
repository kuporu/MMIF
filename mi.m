clear;
close all;
clc;
im1 = [0 1 1;0 3 4; 1 2 4];
im2 = [0 1 1;0 3 5; 1 2 4];

[~,~,indrow] = unique(im1(:)); % ������ӳ��Ϊ����
[~,~,indcol] = unique(im2(:)); 

indrow = double(indrow(:)) + 1;
indcol = double(indcol(:)) + 1; % matlab�е�������0��ʼ
jointHistogram = accumarray([indrow indcol], 1); % ������״��[indrow indcol]һ��
jointProb = jointHistogram / numel(indrow); % �����ܴ���(ͼ��row*col)�õ�����

indNoZero = jointHistogram ~= 0;
jointProb1DNoZero = jointProb(indNoZero);

% ������
jointEntropy = -sum(jointProb1DNoZero.*log2(jointProb1DNoZero));

% ��Ե����
histogramImage1 = sum(jointHistogram, 1);
histogramImage2 = sum(jointHistogram, 2);

indNoZero = histogramImage1 ~= 0;
prob1NoZero = histogramImage1(indNoZero);
prob1NoZero = prob1NoZero / sum(prob1NoZero);
entropy1 = -sum(prob1NoZero.*log2(prob1NoZero));

%// Repeat for the second image
indNoZero = histogramImage2 ~= 0;
prob2NoZero = histogramImage2(indNoZero);
prob2NoZero = prob2NoZero / sum(prob2NoZero);
entropy2 = -sum(prob2NoZero.*log2(prob2NoZero));

%// Now compute mutual information
mutualInformation = entropy1 + entropy2 - jointEntropy;




