function [  ] = drawScater( x, y, str )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
f=fittype('a*(0.5-(1/(1-exp(b*(x-c)))))+d*x+e','independent','x','coefficients',{'a','b','c','d','e'});
func = fit(x, y, f);

p = plot(func, x, y);
ax = gca;
ax.YLim = [0 100];
ax.YLabel.String = 'MOS';
ax.YLabel.FontSize = 14;
ax.XLabel.String = ['Objective score by ' str];
ax.XLabel.FontSize = 14;
set(p, 'LineWidth', 1);
set(p, 'Color', 'black');
legend('Images in MDB', 'Curve fitted','Location','northwest');

saveas(gcf, ['C:\Users\Administrator\Desktop\photos\' str '.jpg']);
end

