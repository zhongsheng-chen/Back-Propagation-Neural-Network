function  net = nninit(net)
%NNINIT Return a neural network with random weights (W)
%   and biases (B). weights{1} is weights connecting the first hidden layer
%   to input layer (IW).
    
%   Date: August 31, 2016
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)


lb = -1;
ub =  1;
 
Nl = net.numLayer; % Number of layers.

% initialize weights and biases.
for i =  1 : Nl
    if i == 1
        net.weight{i} = unifrnd(lb, ub, net.layer{1}.size, net.numInput);
    else
        net.weight{i} = unifrnd(lb, ub, net.layer{i}.size, net.layer{i - 1}.size);
    end
    net.bias{i} = unifrnd(lb, ub, net.layer{i}.size, 1);
end
