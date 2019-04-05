function net = nncheckgrad(net, input, target, dW, db)
%NNCHECKGRAD Carry out validation check for gradients but does not modify 
%   any parameters of networks.


%   Date: August 31, 2016
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)

epsilon = 1e-6;
tol = 1e-8;
Nl = net. numLayer;
for k = 1 : Nl
    for i = 1 : size(net.weight{k}, 1)
        for j = 1 : size(net.weight{k}, 2)
            PNET = net;
            QNET = net;
            PNET.weight{k}(i, j) = PNET.weight{k}(i, j) - epsilon;
            QNET.weight{k}(i, j) = QNET.weight{k}(i, j) + epsilon;
            
            rand('state',0)
            [~, ~, PLOSS ] = nnff(PNET, input, target);
            rand('state',0)
            [~, ~, QLOSS ] = nnff(QNET, input, target);
            dWkij = (PLOSS - QLOSS) / (2 * epsilon);
            e = abs(dWkij - dW{k}(i, j));
            assert(e < tol, 'Numerical gradient checking failed');
        end
    end
end

for k = 1 : Nl
    for i = 1 : size(net.bias{k}, 1)
        for j = 1 : size(net.bias{k}, 2)
            PNET = net;
            QNET = net;
            
            PNET.bias{k}(i, j) = PNET.bias{k}(i, j) - epsilon;
            QNET.bias{k}(i, j) = QNET.bias{k}(i, j) + epsilon;
            %             rand('state',0)
            [~, ~, PLOSS ] = nnff(PNET, input, target);
            %             rand('state',0)
            [~, ~, QLOSS ] = nnff(QNET, input, target);
            dbkij = (PLOSS - QLOSS) / (2 * epsilon);
            e = abs(dbkij - db{k}(i, j));
            assert(e < tol, 'Numerical gradient checking failed');
        end
    end
end
end

