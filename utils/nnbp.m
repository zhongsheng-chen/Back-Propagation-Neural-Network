function [net, gradient, dW, db] = nnbp(net, input, target, error, outlayer, pdW, pdb)
%NNBP Adjust weights and biases of the net according the
%       BP learning algorithm. Return gradient of output layer if output
%       transferFcn is logsig. Otherwise return gradient of last hidden
%       layer if output transferFcn is purelin.
%   Inputs:
%           net -  A feedforward neural network.
%           input - Input of networks.
%           error - Errors between targets and outputs of network.
%           pdW - Previous delta of weights (W).
%           pdb - Previous delta of biases (b).
%   output:
%           net - The updated network
%           gradient - gradient of the network. When transfer function on
%   output layer is purelin, the average gradient of nerons of last hidden
%   layer is returned. Otherwise, the average gradient of neurons of output
%   layer is returned.
%   Example:
%           [net, gradient, dW, db] = NNBP(net, input, target, error, outlayer, pdW, pdb)


%   Date: August 31, 2016
%   Author: Zhongsheng Chen (E-mail:zschen@mail.buct.edu.cn)
%   organization: Beijing University of Chemical Technology

% the number of batch samples (batch size).
Q = size(input, 2);

Nl =  net.numLayer;
switch net.layer{Nl}.transferFcn
    case {'logsig'}
        gradient = outlayer{Nl} .* (1 - outlayer{Nl});
        d{Nl} =  gradient .* error;
    case {'purelin', 'softmax'}
        gradient = outlayer{Nl - 1} .* (1 - outlayer{Nl - 1});
        d{Nl} = error;
end

for i = Nl - 1 : -1 : 1
    switch net.layer{i}.transferFcn
        case {'logsig'}
            grad = outlayer{i} .* (1 - outlayer{i});
        case {'tansig'}
            grad = 1 - outlayer(i) .^ 2;
        case {'tansigopt'}
            grad = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * outlayer{i} .^ 2);
    end
    d{i} = grad .* (net.weight{i + 1}' * d{i + 1});
end

% Comput dW and db.
X = input;
for i = Nl : -1 : 1
    if i == 1
        dW{i} = (d{i} * X') / Q;
    else
        dW{i} = (d{i} * outlayer{i - 1}') / Q;
    end
    db{i} = d{i} * ones(1, Q)' / Q;
end

% Checek gradients
% net = nncheckgrad(net, input, target, dW, db); % removal for speed up


% update weights and biases.
for i = Nl : -1 : 1
    
    rw = net.layer{i}.rw;
    rb = net.layer{i}.rb;
    dW{i} = rw .* dW{i};
    db{i} = rb .* db{i};
    
    mc = net.trainParam.mc;
    if mc > 0
        dW{i} = mc * pdW{i} + (1 - mc) * dW{i};
        db{i} = mc * pdb{i} + (1 - mc) * db{i};
    end
    
    net.weight{i} = net.weight{i} + dW{i};
    net.bias{i} = net.bias{i} + db{i};
end

pdW = dW;
pdb = db;
% Average gradient.
gradient = sum(sum(gradient)) / (size(gradient, 1) * size(gradient, 2));




