function generatesin()


numTrain = 200;
numTest = 150;
numAll = numTrain + numTest;

xmin = -2*pi; xmax = 2*pi;

x = rand(numAll, 1) * (xmax - xmin) + xmin;
y = sin(x);
alldata = [x, y];

[traindata, testdata] = dividedataset(alldata, numTrain);

savedataset('sin_train', traindata, 'sin_test', testdata);

function [traindata, testdata, trainInd, testInd] = dividedataset(dataset, numTrain)
%DIVIDEDATASET Divide dataset into two parts, trainning set and 
%       testing set with data format of x - y.

%   Date: December 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)


numTotal = size(dataset, 1);
numTest = numTotal - numTrain;

ind = randperm(numTotal);
trainInd = sort(ind(1 : numTrain));
testInd = sort(ind(numTrain + (1 : numTest)));

traindata = dataset(trainInd, :);          
testdata = dataset(testInd, :);

function status = savedataset(varargin)
%SAVEDATASET Save datasets into text files and return status of writting
%       operation.
% For example,
%       status = savedataset('dataset.txt', dataset);
%       status = savedataset('traindata.txt', traindata, ...
%                            'testdata.txt', testdata);
%       savedataset('dataset.txt', dataset);
%       savedataset('traindata.txt', traindata, ...
%                            'testdata.txt', testdata);

%   Date: December 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

assert(rem(nargin, 2) == 0, 'Number of input must be even.')

for i = 1 : nargin / 2
    nameprovided{i} = varargin{2 * i - 1};
    requiredvalue{i} = varargin{2 * i}; 
end

for i = 1 : nargin / 2
    filename = [nameprovided{i}];
    fid = fopen(filename, 'w');
    if fid < 0
        error('Can not open file: %s', filename);
        status = false;
        break;
    else
        [row, col] = size(requiredvalue{i});
        format = [];
        for j = 1 : col
            format = [format, '%10.6f\t'];
        end
        format = [format, '\n'];
        
        for j = 1 : row
            fprintf(fid, format, requiredvalue{i}(j, :));
        end
        fclose(fid);
    end
end
status = true;


