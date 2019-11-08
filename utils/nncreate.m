function net = nncreate(hiddenLayer)
%NNCREATE Create a neural network with a certain number of weights and 
%       biases. All parametes including learning ratio (lr), momoent 
%       coefficient (mc) , trasnsfer fucntion(transferFcn), 
%       stop criteria (maximal iteration (epoch), expected performance 
%       (goal), minimal gradient (mingrad), number of successive iteration
%       valdation performance fail to decrease (maxfail) ), divide 
%       function (divideFcn) and etc, are set up.    
%   net = NNCREATE(hiddenLayer) Create a network with a specified number 
%   of neurons in each hidden layer. Default topology of net is 0 - 10 - 0.
%   Inputs:
%           hiddenLayer - A array to specify number of neurons in hidden
%   layers.
%   output: 
%           net - The created network with default parameters.
% Main parameters are listed as follows.
%           epoch - Maximum number of epochs to train.
%           goal - Performance goal.
%           lr - Learning ratio.
%           mc - Momentum constant.
%           mingrad - Minimum performance gradient.
%           showWindow - show performance of training phrase at each epoch.
%           showCommandLine - display information about training using 
%   command-line.
%   Example:
%           net = nncreate([10, 5]);
%           net.trainParam.lr = 0.7;
    
%   Date: August 31, 2016
%   Author: Zhongsheng Chen (E-mail:zschen@mail.buct.edu.cn)
%   organization: Beijing University of Chemical Technology


net.numInput = 0;
net.numOutput = 0;

if nargin < 1
    hiddenLayer = 10;
end

hiddenLayerSize = [hiddenLayer, net.numOutput];
Nl = size(hiddenLayer, 2) + 1;
for i = 1 : Nl
    if i == Nl
        net.layer{i}.transferFcn = 'purelin';
    else
        net.layer{i}.transferFcn = 'logsig';
    end
     net.layer{i}.size = hiddenLayerSize(i);
end


net. numLayer = Nl;

% Normalization
net.processFcn = 'mapminmax';
net.processParam.ymax = 1;
net.processParam.ymin = -1;

% Data set division
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;

% Training parameters
net.trainParam.mc = 0.90;
net.trainParam.lr = 0.01;
net.trainParam.lr_inc = 1.05;
net.trainParam.lr_dec = 0.70;
net.trainParam.goal = 1e-6;
net.trainParam.epoch = 200;
net.trainParam.min_grad = 1e-5;
net.trainParam.max_fail = 6;

net.performFcn = 'mse';
net.adaptFcn = 'none';

% Display
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = false;

net.weight = [];           % Weights
net.bias = [];           % Biases







