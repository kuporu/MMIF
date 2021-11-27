%   将mos.txt文件加载进workspace
%   pic:    融合后的图像
%   score:  图像的分数
%   ref1:   参考图像1
%   ref2:   参考图像2
%%
clear;
clc;
close all;
pics = cell(374, 1);scoreT = zeros(374, 1);
ref1s = cell(374, 1);
ref2s = cell(374, 1);
score_ci = zeros(374, 1);
score_c = zeros(374, 1);
score_i = zeros(374, 1);
fid = fopen('mos_3.txt'); i = 1;

while ~feof(fid)
    pics{i} = fscanf(fid, '%d', 1);
    scoreT(i) = fscanf(fid, '%f', 1);
    ref1s{i} = fscanf(fid, '%d', 1);
    ref2s{i} = fscanf(fid, '%d\n', 1);
    i = i + 1;
end

for i = 1 : length(pics)
    pics{i} = ['img' num2str(pics{i}) '.jpg'];
    ref1s{i} = ['img' num2str(ref1s{i}) '.jpg'];
    ref2s{i} = ['img' num2str(ref2s{i}) '.jpg'];
end



fclose(fid);


prefix_path = 'E:\hgc\pic\';
prefix_path_ref = 'E:\hgc\ref\';
count = 1;
tic
for idx = 1 : length(pics)
    now_path = [prefix_path pics{idx}];
    imgR = imread(now_path); % 融合图像
    
    now_path_ref1 = [prefix_path_ref ref1s{idx}];
    now_path_ref2 = [prefix_path_ref ref2s{idx}];
    ref1 = imread(now_path_ref1);
    ref2 = imread(now_path_ref2); % 伪彩色参考图像
    
    %=====================================================================%
    img = imgR - ref1; % 减去灰度参考图片
    
    %=====================================================================%
    
        % 融合后的图像由RGB变为NTSC
        imgN = zeros(size(img));
%         imgN(:,:,1) = 0.299 * double(img(:,:,1)) + 0.587 * double(img(:,:,2)) + 0.114 * double(img(:,:,3));
        imgN(:,:,2) = 0.596 * double(img(:,:,1)) - 0.274 * double(img(:,:,2)) - 0.322 * double(img(:,:,3));
        imgN(:,:,3) = 0.211 * double(img(:,:,1)) - 0.523 * double(img(:,:,2)) + 0.312 * double(img(:,:,3));
    
        %  伪彩色参考图像由RGB变为NTSC
        ref2N = zeros(size(ref2));
%         ref2N(:,:,1) = 0.299 * double(ref2(:,:,1)) + 0.587 * double(ref2(:,:,2)) + 0.114 * double(ref2(:,:,3));
        ref2N(:,:,2) = 0.596 * double(ref2(:,:,1)) - 0.274 * double(ref2(:,:,2)) - 0.322 * double(ref2(:,:,3));
        ref2N(:,:,3) = 0.211 * double(ref2(:,:,1)) - 0.523 * double(ref2(:,:,2)) + 0.312 * double(ref2(:,:,3));
    %
    %     imgN(:,:,1) = ref2N(:,:,1);
    %     % 此时伪彩色为0-1范围内（并不是ref2中的0-255）
    %     %     sub2 = ntsc2rgb(imgN); % 伪彩色图(保留融合图像的颜色，亮度由参考图像提供)
    %     sub2 = zeros(size(img)); % 对换亮度后融合图像的RGB图像
    %     sub2(:,:,1) = 1 * imgN(:,:,1) + 0.956 * imgN(:,:,2) + 0.621 * imgN(:,:,3);
    %     sub2(:,:,2) = 1 * imgN(:,:,1) - 0.272 * imgN(:,:,2) - 0.647 * imgN(:,:,3);
    %     sub2(:,:,3) = 1 * imgN(:,:,1) - 1.106 * imgN(:,:,2) + 1.703 * imgN(:,:,3);
    
    %=====================================================================%
    
    
    % 灰度图像
    %     imgTemp(:,:, 2) = 0;
    %     imgTemp(:,:, 3) = 0;
    %     sub1 = ntsc2rgb(imgTemp);
    %
    %     sub2N = rgb2ntsc(sub2);
    %     ref2N = rgb2ntsc(ref2);
    %     figure;
    %     subplot(1,2,1); imshow(ref2N(:,:,3));
    %     subplot(1,2,2); imshow(sub2N(:,:,3));
    
    %=====================================================================%
    %     % 色彩相似度（先使用NTSC再使用高斯模型）
    %     sub2H = 0.3 * double(sub2(:,:,1)) + 0.04 * double(sub2(:,:,2)) + -0.35 * double(sub2(:,:,3));
    %     sub2M = 0.34 * double(sub2(:,:,1)) + -0.6 * double(sub2(:,:,2)) + 0.17 * double(sub2(:,:,3));
    %     ref2H = 0.3 * double(ref2(:,:,1)) + 0.04 * double(ref2(:,:,2)) + -0.35 * double(ref2(:,:,3));
    %     ref2M = 0.34 * double(ref2(:,:,1)) + -0.6 * double(ref2(:,:,2)) + 0.17 * double(ref2(:,:,3));
    %
    %     %     sp = (2 * (sub2H .* ref2H + sub2M .* ref2M) + 0.03 )...
    %     %         ./ (sub2H .* sub2H + ref2H .* ref2H + sub2M .* sub2M + ref2M .* ref2M + 0.03);
    %
    %     sa_g = (2 * sub2H .* ref2H + 0.3) ./ (ref2H .* ref2H + sub2H .* sub2H + 0.3);
    %     sb_g = (2 * sub2M .* ref2M + 0.3) ./ (ref2M .* ref2M + sub2M .* sub2M + 0.3);
    %     sp_g = sa_g .* sb_g;
    %
    %     sCo_ng = mean2(sp_g);% KROCC 0.7711; PLCC 0.9639; SROCC 0.9183
    
    %=====================================================================%
        % 色彩相似度（直接利用NTSC模型）
        % 当伪彩色在融合图像中占比较少时，色彩相似度会显著增加，因为融合图像大calc部分为灰度图像和伪彩色参考图像色彩相似度大
        sub2Q = abs(double(imgN(:,:,2)));
        sub2I = abs(double(imgN(:,:,3)));
        ref2Q = abs(double(ref2N(:,:,2)));
        ref2I = abs(double(ref2N(:,:,3)));
    
        sp_n = (2 * (sub2Q .* ref2Q + sub2I .* ref2I) + 0.01 )...
            ./ (sub2Q .* sub2Q + ref2Q .* ref2Q + sub2I .* sub2I + ref2I .* ref2I + 0.01);
        % 方法一
%         sCo_n = mean2(sp_n); %　KROCC 0.4906 PLCC 0.6738 RMSE 21.9993 SROCC 0.6643
        % 方法二
%         len = sum(sum(sp_n < 1));
%         index = find(sp_n < 1);
%         all = sum(sum(sp_n(index)));
%         sCo_n = all / len;
        % 方法三
%         sp_n = sp_n .^ 2;
%         sCo_n = areaMean(sp_n); 

        % 方法三增加边长
        sCo_n = localMeanF(sp_n, 12); 
        % i = 1: KROCC 0.4617 PLCC 0.7522 RMSE 19.7564 SROCC 0.6368
        % i = 2: KROCC 0.4646 PLCC 0.7565 RMSE 18.7989 SROCC 0.6396
        % i = 3: KROCC 0.4682 PLCC 0.7590 RMSE 18.0843 SROCC 0.6439
        % i = 4: KROCC 0.4717 PLCC 0.7606 RMSE 17.5503 SROCC 0.6475
        % i = 5: KROCC 0.4739 PLCC 0.7617 RMSE 17.1358 SROCC 0.6502
        % i = 6: KROCC 0.4769 PLCC 0.7627 RMSE 16.8112 SROCC 0.6539
        % i = 7: KROCC 0.4795 PLCC 0.7634 RMSE 16.5642 SROCC 0.6570
        % i = 8: KROCC 0.4812 PLCC 0.7638 RMSE 16.3883 SROCC 0.6592
        % i = 9: KROCC 0.4841 PLCC 0.7641 RMSE 16.2709 SROCC 0.6624
        % i = 10:KROCC 0.4864 PLCC 0.7642 RMSE 16.2029 SROCC 0.6652
        % i = 11:KROCC 0.4882 PLCC 0.7642 RMSE 16.1785 SROCC 0.6671
        % i = 12:KROCC 0.4908 PLCC 0.7640 RMSE 16.1922 SROCC 0.6696
    %=====================================================================%
        % 举例说明IS和CS的互补性（绘图）
%         figure;
%         subplot(2, 2, 1); imshow(3 * uint8(sub2Q));
%         subplot(2, 2, 2); imshow(3 * uint8(sub2I));
%         subplot(2, 2, 3); imshow(3 * uint8(ref2Q));
%         subplot(2, 2, 4); imshow(3 * uint8(ref2I));
    %=====================================================================%
    
    
    % 色彩相似度（直接使用高斯模型）
%     sub2HD = 0.3 * double(img(:,:,1)) + 0.04 * double(img(:,:,2)) - 0.35 * double(img(:,:,3));
%     sub2MD = 0.34 * double(img(:,:,1)) - 0.6 * double(img(:,:,2)) + 0.17 * double(img(:,:,3));
%     ref2HD = 0.3 * double(ref2(:,:,1)) + 0.04 * double(ref2(:,:,2)) - 0.35 * double(ref2(:,:,3));
%     ref2MD = 0.34 * double(ref2(:,:,1)) - 0.6 * double(ref2(:,:,2)) + 0.17 * double(ref2(:,:,3));
%     
%     sp_nd = (2 * (sub2HD .* ref2HD + sub2MD .* ref2MD) + 0.03 )...
%         ./ (sub2HD .* sub2HD + ref2HD .* ref2HD + sub2MD .* sub2MD + ref2MD .* ref2MD + 0.03);
%     
%     sCo_g = mean2(sp_nd); 
    
    %=====================================================================%
    % matlab绘制sp_nd统计直方图，解释色彩相似度缺陷
    %     sp_nd_l = reshape(sp_nd, [256 * 256, 1]);
    %     hist(sp_nd_l, 26);
    %     set(gca,'XLim',[-1 1.4])
    %     annotation('textarrow',[243/703 288/703], [238/528 180/528], 'String', 'Normal interval');
    %     annotation('textarrow',[446/703 495/703], [370/528 306/528], 'String', 'Error interval');
    %     title('Histogram of chromaticity similarity map');
    %     xlabel('values');
    %     ylabel('statistics');
    %     saveas(gcf,'histogram.jpg');
    %====================================================================%
    
%     信息相似度（NTSC颜色分量）
        ref2N2 = ref2N(:,:,2); % Q
        imgN2 = imgN(:,:,2);
        ref2N2(ref2N2 < 28 & ref2N2 > -20) = 0; % 实验出来阈值（naturePic），可设置为任意相同值
        imgN2(imgN2 < 28 & imgN2 > -20) = 0;
        [ref2IE, sub2IE] = imageEntropy(ref2N2,imgN2);
    
        ref2N3 = ref2N(:,:,3);
        imgN3 = imgN(:,:,3);
        ref2N3(ref2N3 < 16 & ref2N3 > -15) = 0;
        imgN3(imgN3 < 16 & imgN3 > -15) = 0;
        [ref2QE, sub2QE] = imageEntropy(ref2N3,imgN3);
    
        sa = (2 * ref2IE * sub2IE + 0.03) / (ref2IE * ref2IE + sub2IE * sub2IE + 0.03);
        sb = (2 * ref2QE * sub2QE + 0.03) / (ref2QE * ref2QE + sub2QE * sub2QE + 0.03);
        sEn_n = sa .* sb;
    
    %=====================================================================%
    
%     % 信息相似度（高斯颜色分量）
%     sub2HD(sub2HD > -18 & sub2HD < 15) = 0;
%     ref2HD(ref2HD > -18 & ref2HD < 15) = 0;
%     [ref2HE, sub2HE, mutualInformation] = entropyAndMutualInformation(ref2HD,sub2HD);
%     
%     sub2MD(sub2MD > -14 & sub2MD < 20) = 0;
%     ref2MD(ref2MD > -14 & ref2MD < 20) = 0;
%     [ref2ME, sub2ME, mutualInformation2] = entropyAndMutualInformation(ref2MD,sub2MD);
%     
%     % 目前这种比较方法明显优于混合计算
%     sa_g = (2 * ref2HE * sub2HE + 0.03) / (ref2HE * ref2HE + sub2HE * sub2HE + 0.03);
%     sb_g = (2 * ref2ME * sub2ME + 0.03) / (ref2ME * ref2ME + sub2ME * sub2ME + 0.03);
%     sEn_g = sa_g .* sb_g;
    
    %=====================================================================%
    
%     temp = sCo_n * sEn_n;
%     score_ci(idx) = temp;
    score_c(idx) = sCo_n;
    score_i(idx) = sEn_n;
    
    
    
    % 绘图
%         subplot(2,3,1); imshow(imgR); title(titleStr);
%         subplot(2,3,2); imshow(uint8(sub2HD)); title(titleC);
%         subplot(2,3,3); imshow(uint8(sub2MD)); title(titleI);
%         subplot(2,3,4); imshow(uint8(sp_nd));
%         subplot(2,3,5); imshow(uint8(ref2HD));
%         subplot(2,3,6); imshow(uint8(ref2MD));
%         count = count + 1;
    
%     绘图
%         titleStr = ['score ' num2str(temp) ' MOS ' num2str(scoreT(idx))];
%         titleC = ['color ' num2str(sCo_n)];
%         titleI = ['information ' num2str(sEn_n)];
%         subplot(2,2,1); imshow(imgR); title(titleStr);
%         subplot(2,2,2); imshow(ref1);title(titleC);
%         subplot(2,2,3); imshow(ref2);title(titleI);
%         count = count + 1;
    
    
    
end
toc

%%
% 导出真实分数，IS分数，CS分数
% save('s.txt', 'scoreT','-ascii');
% save('CS.txt', 'score_c','-ascii');
% save('IS.txt', 'score_i','-ascii');
%%

% 计算相关系数（CS and IS）(过时)
% SROCC_CI = corr(score_ci, scoreT, 'type', 'Spearman');
% KROCC_CI = corr(score_ci, scoreT,'type','Kendall');
% PLCC_CI = corr(score_ci, scoreT,'type','Pearson');
% RMSE_CI = sqrt(mean2((score_ci * 100 - scoreT).^2));

% 计算相关系数（CS）
SROCC_C = corr(score_c, scoreT, 'type', 'Spearman');
KROCC_C = corr(score_c, scoreT,'type','Kendall');
PLCC_C = corr(score_c, scoreT,'type','Pearson');
RMSE_C = sqrt(mean2((score_c - scoreT).^2));

% 计算相关系数（IS）
SROCC_I = corr(score_i, scoreT, 'type', 'Spearman');
KROCC_I = corr(score_i, scoreT,'type','Kendall');
PLCC_I = corr(score_i, scoreT,'type','Pearson');
RMSE_I = sqrt(mean2((score_i - scoreT).^2));


clearvars -except SROCC_CI KROCC_CI PLCC_CI RMSE_CI SROCC_C KROCC_C ...
    PLCC_C RMSE_C SROCC_I KROCC_I PLCC_I RMSE_I scoreT score_c score_i