function [net, tr, data, option] = nnprepare(net, data)
%NNPREPARE Prepare varibles pdW, pdB and loss to trainning process.
%   [pdW, pdB] = NNPREPARE(net) create pdW and pdB, whose size are same as
%           weights and biases.              

%   Date: August 31, 2016
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)

% Initilize previous delta of weights and biases.
Nl = net.numLayer;
for i = 1 : Nl
    pdW{i} = zeros(size(net.weight{i}));    % previous dW. 
    pdb{i} = zeros(size(net.bias{i}));      % previous db.
end

% Initial learning ratio.
for i = 1 : Nl
    net.layer{i}.rw = net.trainParam.lr;
    net.layer{i}.rb = net.trainParam.lr;
end

data.pdW = pdW;
data.pdb = pdb;

validation = false;
testing = false;
if ~isempty(data.valInd)
    validation = true;
end
if ~isempty(data.testInd)
    testing = true;
end
option.validation = validation;
option.testing = testing;


fig = []; % Plot for errors, gradient.
if net.trainParam.showWindow
    fig = figure();
end
data.figure = fig;

tr.trainInd = [];
tr.valInd = [];
tr.testInd = [];


tr.perf = [];
tr.vperf = [];
tr.tperf = [];

tr.best_perf = [];
tr.best_vperf = [];
tr.best_tperf = [];
tr.epoch = [];
tr.time = [];
tr.gradient = [];  
tr.stop = [];
tr.best_epoch = [];
tr.num_epochs = [];
tr.val_fail = [];