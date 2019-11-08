function [best, pos]= nnfindbest(model, option, trainPerf, valPerf, testPerf)
%NNFINDBEST Find best model from all models.
%   [best, bestloss, bestpos] = FINDBEST(model, option, trainerr, ...
%           valerr, testerr) Return a model best performance. 

%   Date: August 31, 2016
%   Author: Zhongsheng Chen (E-mail:zschen@mail.buct.edu.cn)
%   organization: Beijing University of Chemical Technology

[~, pos] = min(trainPerf);
if option.validation
    [~, pos] = min(valPerf);
end
if option.testing
    [~, pos] = min(testPerf);
end
best = model{pos};
















