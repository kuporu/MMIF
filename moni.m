clear;
close all;
clc;

fids = fopen('s.txt', 'r');
fidCS = fopen('CS.txt', 'r');
fidIS = fopen('IS.txt', 'r');

s = fscanf(fids, '%f');
cs = fscanf(fidCS, '%f');
is = fscanf(fidIS, '%f');

mn = (1 - 0.3) * ((100 * cs) .^ 1.023) + 0.3 * ((100 * is) .^ 0.96);

SROCC_CI = corr(s, mn, 'type', 'Spearman');
KROCC_CI = corr(s, mn,'type','Kendall');
PLCC_CI = corr(s, mn,'type','Pearson');
RMSE_CI = sqrt(mean2((s - mn).^2));

clearvars -except SROCC_CI KROCC_CI PLCC_CI RMSE_CI mn
