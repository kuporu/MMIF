% 绘制真实分数箱型图
clear;
close all;
clc;

% 读取mos_3文件中的分数
fileID = fopen('mos_3.txt', 'r');
formatSpec = '%d %f %d %d';
sizeA = [4 374];
A = fscanf(fileID,formatSpec,sizeA);

% 将读取的分数22个为一组放在一个[17 22]的数组中
newA = reshape(A(2,:), 22, 17);

% 绘制箱型图
boxplot(newA);

% 横纵轴
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


% 保存箱型图



