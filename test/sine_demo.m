%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This program is a demo on sine function which is used to demonstrate
%%% how to tran and test back-propagation neural networks.
%%%
%%% Author: Zhongsheng Chen (zhongsheng.chen@outlook.com)
%%% Date: August 31, 2016
%%% organization: Beijing University of Chemical Technology
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
traindata = load('sin_train');
testdata = load('sin_test');
alldata = [traindata; testdata];

input = alldata(:, 1 : end - 1)';
target = alldata(:, end)';

%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters Setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%
hiddenLayerSize = 6;
net = nncreate(hiddenLayerSize);
net.trainParam.mc = 0.95;
net.trainParam.lr = 0.8;
net.trainParam.lr_inc = 1.05;
net.trainParam.lr_dec = 0.70;
net.trainParam.goal = 1e-6;
net.trainParam.epoch = 200;

net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

net.performFcn = 'mse';
net.adaptFcn = 'none';
net.trainParam.showCommandLine = false;
net.trainParam.showWindow = false;

[net, tr] = nntrain(net, input, target);

trainInd = tr.trainInd;
valInd = tr.valInd;
testInd = tr.testInd;

validation = true;
testing =  true;
if isempty(valInd), validation  = false; end
if isempty(testInd), validation  = false; end

trainInp = input(:, trainInd);
trainTarg = target(:, trainInd);
trainOut = nnpredict(net, trainInp);

if validation
    valInp = input(:, valInd);
    valTarg = target(:, valInd);
    valOut = nnpredict(net, valInp);
end

if testing
    testInp = input(:, testInd);
    testTarg = target(:, testInd);
    testOut = nnpredict(net, testInp);
end

trainPerf = tr.best_perf;
valPerf = tr.best_vperf;
testPerf = tr.best_tperf;

%%%%%%%%%%%%%%%%%%%%%%%%%%%% making Prediction %%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainPerformance = nneval(net, trainInp, trainTarg);

figure
index = 1 : length(trainInd);
plot(index, trainTarg, 'b--o', index, trainOut, 'b-.+')
xlabel('Index of training samples')
ylabel('Output')
legend('Actual value', 'Predictive value')
title(['Training perforance = ' num2str(trainPerf)])

if validation
    figure
    index = 1 : length(valInd);
    plot(index, valTarg, 'r:*', index, valOut, 'r-.s')
    xlabel('Index of validation samples')
    ylabel('Output')
    legend('Actual value', 'Predictive value')
    title(['Validation perforance = ' num2str(valPerf)])
end

if testing
    figure
    index = 1 : length(testInd);
    plot(index, testTarg, 'r:*', index, testOut, 'r-.s')
    xlabel('Index of testing samples')
    ylabel('Output')
    legend('Actual value', 'Predict value')
    title(['Testing perforance = ' num2str(testPerf)])
end

%%%%%%%%%%%%%% Plot orignal ponts and predicted points %%%%%%%%%%%%%%%%%%%%
maximum = max(input(:));
minimum = min(input(:));
X = linspace(minimum, maximum, 60);
Y = sin(X);
figure
plot(X, Y,'g--');
hold on
plot(trainInp, trainTarg, 'bo')
xlabel('Index of  samples')
ylabel('Output')
legendStr = {'The curve of sin(x)', 'Orignal points'};
if validation
    plot(valInp, valTarg, 'yo')
    legendStr = [legendStr, 'Validation points'];
end
if testing
    plot(testInp, testTarg, 'go')
    legendStr = [legendStr, 'Testing points'];
end
legend(legendStr)
hold off