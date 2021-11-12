clear;
close all;
clc;

% ��ȡ���۷�����Ԥ�����
fid = fopen('score.txt');
fid_p = fopen('score_p.txt');
score = fscanf(fid, '%f');
score_p = fscanf(fid_p, '%f');
fclose(fid);
fclose(fid_p);

% ����ɢ��ͼ
hold on
plot(score_p, score, 'or', 'MarkerSize',2, 'MarkerFaceColor', 'r');
% �Զ���ָ����Ϻ���
fx = -189.772302 * (0.5 - 1 ./ (1 - exp(-500 * (score_p + 142.278664)))) + 96.056128 * score_p - 97.014643;
line = plot(score_p, fx);
set(line, 'LineWidth', 1);
hold off