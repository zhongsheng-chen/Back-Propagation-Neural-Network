function [net, data] = nnconfigure(net, P, T)
%NNCONFIGURE Configure paramaters of networks (i.e. net.numInputs,
%       net.numOutputs and net.weights).

%   Date: August 31, 2016
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)

if ~isequal(size(P, 2), size(T, 2))
    error('NN:Preprocess:misMatch','the size of Inputs mismatch with Targets.');
end

net.numInput = size(P, 1);
net.numOutput = size(T, 1);
net.layer{end}.size = net.numOutput;

% Initialize weights and biases.
net = nninit(net);

% normalize inputs and targets.
switch lower(net.processFcn)
    case {'mapminmax', 'minmax'}
        [pn, ps] = mapminmax(P, net.processParam);
        [tn, ts] = mapminmax(T, net.processParam);
    case {'mapstd', 'std'}
        [pn, ps] = mapstd(P, net.processParam);
        [tn, ts] = mapstd(T, net.processParam);
end

% data.p = P;
% data.t = T;
data.P = pn;
data.T = tn;
data.PS = ps;
data.TS = ts;

N = size(P, 2);
% Dataset division. Return index of dataset for training, validation and testing.
[trainInd, valInd, testInd] = feval(net.divideFcn, N, net.divideParam);

data.trainInd = trainInd;
data.valInd = valInd;
data.testInd = testInd;
end




