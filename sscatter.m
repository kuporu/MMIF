clear;
clc;
close;

% proposed metric
s = load('s.txt');
csis = load('csis.txt');
% drawScater(csis, s, 'proposed metric');

% other metrics
index = xlsread('D:\index\result2.xls'); % 其他评价指标数值

% SSIM
% drawScater(index(:,1), s, 'SSIM');

% MI
% drawScater(index(:,2), s, 'MI');

%GS
drawScater(index(:,13), s, 'GS');

%TM
% drawScater(index(:,15), s, 'TM');

% CC
% drawScater(index(:,18), s, 'CC');

% FSIMc
% drawScater(index(:,20), s, 'FSIMc');

% VSI
% drawScater(index(:,17), s, 'VSI');



