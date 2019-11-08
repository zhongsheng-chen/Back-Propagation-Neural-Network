function figHandle = nnupdatefigure(net, i, figHandle, option, trainPerf, valPerf, testPerf, gradient, fail)
%UPDATEFIGURE Update errors, gradients, validation checks.

%   Date: August 31, 2016
%   Author: Zhongsheng Chen (E-mail:zschen@mail.buct.edu.cn)
%   organization: Beijing University of Chemical Technology

if i == 1
    return;
end

interval = 1;
if 2^(fix(log2(i) + 1))  > 500
    interval = fix(log2(i));
end

if option.validation
    subPlotSize = 3;
else
    subPlotSize = 2;
end

figure(figHandle)
figHandle.Name = 'Neural Network Training Performance';
% Plot training errors.
train = subplot(subPlotSize, 1, 1);
legendStr = {'Training'};
if option.validation
    legendStr = [legendStr, {'Validation'}];
end
if option.testing
    legendStr = [legendStr,{'Testing'}];
end
ind = 1 : interval: i;
xplot = ind';
eplot = trainPerf(ind)';
if option.validation
    xplot = [xplot, ind'];
    eplot = [eplot, valPerf(ind)'];
end
if option.testing
    xplot = [xplot, ind'];
    eplot = [eplot, testPerf(ind)'];
end
line = plot(train, xplot, eplot, 'b-');
numLine = length(line);
line(1).Color = 'b'; line(1).LineStyle = '-'; line(1).Marker = 'none';      % Training error curve
if numLine == 2
    if option.validation
        line(numLine).Color = 'g'; line(numLine).LineStyle = '--'; line(numLine).Marker = 'none'; % validation error curve
    end
    if option.testing
        line(numLine).Color = 'r'; line(numLine).LineStyle = '-.'; line(numLine).Marker = 'none'; % Testing error curve
    end
end
if numLine == 3
    if option.validation
        line(numLine - 1).Color = 'g'; line(numLine - 1).LineStyle = '--'; line(numLine - 1).Marker = 'none'; % validation error curve
    end
    if option.testing
        line(numLine).Color = 'r'; line(numLine).LineStyle = '-.'; line(numLine).Marker = 'none'; % Testing error curve
    end
end
train.Title.String = sprintf('Training errors = %3.6f, at %d epoch', trainPerf(i), i);
train.XLim = [0, i + 10];
train.XLabel.String = 'Epochs';
train.YLabel.String = sprintf('Erros (%s)',lower(net.performFcn));
legend(train, legendStr, 'Location', 'NE');

% Plot gradients.
grad = subplot(subPlotSize, 1, 2);
eplot = gradient(ind)';
plot(grad, xplot, eplot, 'k:');
grad.Title.String = sprintf('Gradient = %3.6f, at %d epoch', gradient(i), i);
grad.XLim = [0, i + 10];
grad.XLabel.String = 'Epochs';
grad.YLabel.String = sprintf('Gradient');

% Plot number of succesive iteration of validaton performance fails to decrease.
if option.validation
    failchk = subplot(subPlotSize, 1, 3);
    ind = 1 : 1: i;
    xplot = ind';
    eplot = fail(ind)';
    scatter(failchk, xplot, eplot, 'MarkerFaceColor', 'r');
    failchk.Title.String = sprintf('Validation Checks = %d, at %d epoch', fail(i), i);
    failchk.XLim = [0, i + 10];
    failchk.XLabel.String = 'Epochs';
    failchk.YLabel.String = sprintf('Fails');
end
drawnow

