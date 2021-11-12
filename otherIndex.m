% ��������ָ������ʵ���������ϵ��
clear;
close all;
clc;
index = xlsread('D:\index\result.xls'); % ��������ָ����ֵ

fid = fopen('mos_3.txt', 'r'); % ��ʵ����
formatSpec = '%d %f %d %d';
sizeA = [4 374];
score = fscanf(fid, formatSpec, sizeA);
score = score';
score = score(:,2);
fclose(fid);

% SROCC = zeros(22,1);
% KROCC = zeros(22,1);
% PLCC = zeros(22,1);
% RMSE = zeros(22,1);
INDEX = zeros(22, 4);

[m,n] = size(index); % 374 * 22
for i = 1:n % ���ۺͿ͹۷��������ϵ��
    scoreT = index(:,i);
%     SROCC(i) = corr(score, scoreT, 'type', 'Spearman');
%     KROCC(i) = corr(score, scoreT,'type','Kendall');
%     PLCC(i) = corr(score, scoreT,'type','Pearson');
%     RMSE(i) = sqrt(mean2((score - scoreT).^2));
    INDEX(i, 1) = corr(score, scoreT, 'type', 'Spearman'); % SROCC
    INDEX(i, 2) = corr(score, scoreT,'type','Kendall'); % KROCC
    INDEX(i, 3) = corr(score, scoreT,'type','Pearson'); % PLCC
    INDEX(i, 4) = sqrt(mean2((score - scoreT).^2)); % RMSE
end
    
% hold on
% plot(scoreT, score, 'or', 'MarkerSize',2, 'MarkerFaceColor', 'r');
% hold off



clearvars -except INDEX