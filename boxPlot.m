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
ax.YLabel.Color = [0.5 0.5 0.5];
ax.FontSize = 10;
ax.YLabel.FontSize = 14;
ax.TickLabelInterpreter = 'tex';
ax.XTickLabel = {'\color{gray} CSR', '\color{gray} DSIFT', '\color{gray} LLF', ...
    '\color{gray} FF', '\color{gray} NFF', '\color{gray} GFF',...
    '\color{gray} CNN', '\color{gray} CSMCA', '\color{gray} MST-SR', '\color{gray} NSST-P', ...
    '\color{gray} MLEPF', '\color{gray} NSCT-SP', '\color{gray} MPCNN-K', '\color{gray} DL',...
    '\color{gray} ReLP', '\color{gray} SAF', '\color{gray} ST'};
ax.XTickLabelRotation = 45;


% ��������ͼ



