% ������ʵ��������ͼ
clear;
close all;
clc;

% ��ȡmos_3�ļ��еķ���
fileID = fopen('mos_3.txt', 'r');
formatSpec = '%d %f %d %d';
sizeA = [4 374];
A = fscanf(fileID,formatSpec,sizeA);

% ����ȡ�ķ���22��Ϊһ�����һ��[17 22]��������
newA = reshape(A(2,:), 22, 17);

% ��������ͼ
boxplot(newA);

% ������
ax = gca;
ax.YLabel.String = 'MOS';
% ax.YLabel.Color = [0.5 0.5 0.5];
ax.FontSize = 10;
ax.YLabel.FontSize = 14;
ax.TickLabelInterpreter = 'tex';
ax.XTickLabel = {'\color{black} CSR', '\color{black} DSIFT', '\color{black} LLF', ...
    '\color{black} FF', '\color{black} NFF', '\color{black} GFF',...
    '\color{black} CNN', '\color{black} CSMCA', '\color{black} MST-SR', '\color{black} NSST-P', ...
    '\color{black} MLEPF', '\color{black} NSCT-SP', '\color{black} MPCNN-K', '\color{black} DL',...
    '\color{black} ReLP', '\color{black} SAF', '\color{black} ST'};
ax.XTickLabelRotation = 45;


% ��������ͼ



