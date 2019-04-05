function T = nnpostprocess(normalizationFcn, TN, TS)
%NNPOSTPROCESS reverse outputs of the network.
%   outputs = POSTPROCESS(net, outputs, TS) reverse outputs of the network 
%       according to target normalization setting (TS);

%   Date: August 31, 2016
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)

switch lower(normalizationFcn)
    case {'mapminmax', 'minmax'}
        T = mapminmax('reverse', TN, TS);
    case {'mapstd', 'std'}
        T = mapstd('reverse', TN, TS);
end



