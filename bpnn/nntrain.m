function [net, tr] = nntrain(net, input, target)
%NNTRAIN Train a network using BP algorithm.
%   [net, tr] = NNTRAIN(net, inputs, targets) Train a network using
%           training set and test the trained network using testing set. All
%           parameters, such as learn ratio, momentum, activation function,
%           can be specified.
%   Inputs:
%           net - A neural network with initialized weights and biases.
%           inputs - Inputs of network.
%           targets - Targets assosited with inputs.
%   output:
%           net - The trained neural network.
%           tr - The training records including error, gradient,etc,
%           information.
%   Example:
%
%           [net, tr] = NNTRAIN(net, input, target)
%

%   Date: August 31, 2016
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)
rand('state', 0)
global PS TS

[net, data] = nnconfigure(net, input, target);
[net, tr, data, option] = nnprepare(net, data);

PS = data.PS;
TS = data.TS;
pdW = data.pdW;
pdb = data.pdb;
fig = data.figure;

trainInd = data.trainInd;
valInd = data.valInd;
testInd = data.testInd;

trainP  = data.P(:, trainInd); % Training samples.
trainT = data.T(:, trainInd);
valP    = data.P(:, valInd);   % Validation samples.
valT   = data.T(:, valInd);
testP   = data.P(:, testInd);  % Testing samples.
testT  = data.T(:,testInd);

m = size(trainP, 2); % Number of training samples.
if m > 1
    batch = fix(log2(m));
else
    batch = 1;
end

batchSize = fix(m / batch);
maxit = net.trainParam.epoch;

perf = [];
valPerf = [];
testPerf = [];

n = 1;
for i = 1 : maxit
    tic;
    ind = randperm(m);
    for j = 1 : batch
        P = trainP(:, ind((j - 1) * batchSize + 1 : j * batchSize));
        T = trainT(:, ind((j - 1) * batchSize + 1 : j * batchSize));
        
        [outLayer, E, lo(n)] = nnff(net, P, T);
        [net, grad(n), dW, db] = nnbp (net, P, T, E, outLayer, pdW, pdb);
        
        pdW = dW;
        pdb = db;
        n = n + 1;
    end %  for j = 1 : batch
    
    epoch(i) = i;
    time(i) = toc;
    networks{i} = net;
    loss(i) =  mean(lo((n - batch) : (n - 1)));
    gradient(i) = mean(grad((n - batch) : (n - 1)));
    
    perf(i) = evaluate_network(net, trainP, trainT);
    perfStr = sprintf('; Full-batch training MSE = %3.6f', perf(i));
    if option.validation
        valPerf(i) = evaluate_network(net, valP, valT);
        perfStr = sprintf([perfStr, ', validation MSE = %3.6f'],  valPerf(i));
    end
    if option.testing
        testPerf(i) = evaluate_network(net, testP, testT);
        perfStr = sprintf([perfStr, ', testing MSE = %3.6f'], testPerf(i));
    end
    
    [stop, fail(i)] = nnstopcriteria(net, i, gradient, perf, valPerf);
    
    if net.trainParam.showWindow
        fig = nnupdatefigure(net, i, fig, option, perf, valPerf, testPerf, gradient, fail);
    end
   
    % Display information about time, gradient, performance at each interation.
    if net.trainParam.showCommandLine
        showMessage = ['epoch %d / %d. Took %3.6f seconds. Mini-batch average gradient = %3.6f, Mini-batch average loss = %3.6f' perfStr '\n'];
        fprintf(1, showMessage, i, maxit, time(i), gradient(i), loss(i));
    end
    
    % Trigger stop criteria
    stopcriteria = {'maxiteration', 'goal', 'mingrad', 'maxfail'};
    if any(ismember(stopcriteria, stop))
        break;
    end
end %  for i = 1 : maxit

% Find best model with the minimum error.;
[net, best_epoch]= nnfindbest(networks, option, perf, valPerf, testPerf);

tr.trainInd = trainInd;
tr.valInd = valInd;
tr.testInd = testInd;

tr.epoch = maxit;
tr.num_epochs = i;
tr.val_fail = fail;
tr.best_epoch = best_epoch;
tr.time = time;
tr.gradient = gradient;
tr.stop = stop;


tr.perf = perf;
tr.best_perf = perf(best_epoch);

if option.validation
    
    tr.vpref = valPerf;
    tr.best_vperf = valPerf(best_epoch);
end
if option.testing
    tr.tpref = testPerf;
    tr.best_tperf = testPerf(best_epoch);
end

function perf = evaluate_network(net, input, target)
% Evaluate the performance of networks when normalized input and target is fed into .
%   inputs:
%       net - the trained neural network.
%       target - normalized input.
%       output - normalized target.
%   outputs:
%       perf - network's performance.

global TS

outLayer = nnff(net, input, target);
output = outLayer{end};

% Reversed normalization for target and output.
target = nnpostprocess(net.processFcn, target, TS);
output = nnpostprocess(net.processFcn, output, TS);

performFcn = net.performFcn;
switch lower(performFcn)
    case {'mse'}
        perf = mse(target - output);
    case {'mae'}
        perf = mae(target - output);
    case {'sae'}
        perf = sae(target - output);
    case {'sse'}
        perf = sse(target - output);
end

