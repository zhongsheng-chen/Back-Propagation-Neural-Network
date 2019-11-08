function PN = nnpreprocess(normalizationFcn, P, PS)
%NNPREPROCESS Return normalized matrix of a given matrix. Atrributes order
%       in column in the matrix.

%   Date: August 31, 2016
%   Author: Zhongsheng Chen (E-mail:zschen@mail.buct.edu.cn)
%   organization: Beijing University of Chemical Technology

switch lower(normalizationFcn)
    case {'mapminmax', 'minmax'}
        PN = mapminmax('apply', P, PS);
    case {'mapstd', 'std'}
        PN = mapstd('apply', P, PS);
end
