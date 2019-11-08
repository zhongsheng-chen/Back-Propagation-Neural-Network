function [traindata,...
    testdata,...
    trainInd,...
    testInd] = dividedataset(dataset, numTrain)
% A hepler function dividing dataset into two parts. One trainning set,
%   One testing set. For formation used, see grap_data_from_sine().

%   Date: December 31, 2016
%   Author: Zhongsheng Chen (E-mail:zschen@mail.edu.cn)

numTotal = size(dataset, 1);
numTest = numTotal - numTrain;

ind = randperm(numTotal);
trainInd = sort(ind(1 : numTrain));
testInd = sort(ind(numTrain + (1 : numTest)));

traindata = dataset(trainInd, :);
testdata = dataset(testInd, :);
