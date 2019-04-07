clear all
close all
clc

rng(123456789, 'twister')
x1  = linspace(-pi/2, pi/2, 200);
x2 = linspace(-pi/5, pi/5, 200);
y = x1.* sin(x2);

x = [x1; x2];
% x = -100 : 8 :100;
% y = x .^ 3 - x .^ 2 + x;
inputs =  x;
targets = y;


hiddenLayerSize = [20, 3, 30];
net = nncreate(hiddenLayerSize);

net.normalizationFcn = 'mapminmax';
net.normalizationParam.ymax = 1;
net.normalizationParam.ymin = -1;

net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.5;
net.divideParam.valRatio = 0.25;
net.divideParam.testRatio = 0.25;

net.performFcn = 'mse';
net.performParam.normalization = 'none';
net.transferFcn = 'sigmoid';

Nl = size(hiddenLayerSize, 2) + 1;
net.layer{Nl}.transferFcn = 'purelin';

net.lr = 0.75;
net.mc =  0.90;

net.epoch = 200;
net.goal = 0;
net.mingrad = 1e-5;
net.maxfail = 6;

net.showWindow = false;
net.showCommandLine = true;

[net, tr] = nntrain(net, inputs, targets);

numTrain =  length(tr.trainInd);
trainTarg = y(tr.trainInd)';
trainOut = nnpredict(net, x(:, tr.trainInd))';
ax = 1 : numTrain;
ax = ax';
fig = figure();
plot(ax, trainTarg, 'b--o', ax, trainOut, 'b-.+')
xlabel('Index of training samples')
ylabel('Output')
legend('Actual value', 'Predictive value')
title(['Training perforance = ' num2str(tr.performance.train)])

if ~isempty(tr.valInd)
    numVal =  length(tr.valInd);
    valTarg = y(tr.valInd)';
    valOut =  nnpredict(x(:, tr.valInd))';
    
    ax = 1 : numVal;
    ax = ax';
    fig = figure();
    plot(ax, valTarg, 'r:*', ax, valOut, 'r-.s')
    xlabel('Index of validation samples')
    ylabel('Output')
    legend('Actual value', 'Predictive value')
    title(['Validation perforance = ' num2str(tr.performance.val)])
end

if ~isempty(tr.testInd)
    numTest =  length(tr.testInd);
    testTarg = y(tr.testInd)';
    testOut =  nnpredict(x(:, tr.testInd))';
    
    ax = 1 : numTest;
    ax = ax';
    fig = figure();
    plot(ax, testTarg, 'r:*', ax, testOut, 'r-.s')
    xlabel('Index of testing samples')
    ylabel('Output')
    legend('Actual value', 'Predict value')
    title(['Testing perforance = ' num2str(tr.performance.test)])
end