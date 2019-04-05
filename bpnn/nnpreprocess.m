function PN = preprocess(normalizationFcn, P, PS)
%NNPREPROCESS Return normalized matrix of a given matrix. Atrributes order
%       in column in the matrix.

%   Date: August 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

switch lower(normalizationFcn)
    case {'mapminmax', 'minmax'}
        PN = mapminmax('apply', P, PS);
    case {'mapstd', 'std'}
        PN = mapstd('apply', P, PS);
end
