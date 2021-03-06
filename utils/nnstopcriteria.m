function [stop, fail] = nnstopcriteria(net, i, gradient, trainPerf, validPerf)
%NNSTOPCRITERIA Trigger a stop criterion and Return stop category.

%   Date: August 31, 2016
%   Author: Zhongsheng Chen (E-mail:zschen@mail.buct.edu.cn)
%   organization: Beijing University of Chemical Technology

persistent count; % count for validation error violation.
if isempty(count)
    count = 0;
end

stop = 'normal';
if ~isempty(validPerf)
    if i > 1 && validPerf(i) - validPerf(i - 1) > 0
        count = count + 1;
    else
        count = 0;
    end
end

fail = count;

if  i == net.trainParam.epoch
    stop = 'maxiteration';
    return;
end

if  trainPerf(i) < net.trainParam.goal
    stop = 'goal';
    return;
end

if gradient(i) < net.trainParam.min_grad
    stop = 'mingrad';
    return;
end

if  fail == net.trainParam.max_fail
    stop = 'maxfail';
    return;
end


