function grap_data_from_sine()
% A helper function for graping test data from sine function.
%
%   Poitns are drawn from sine function with an input variable
%   over a specific range. Note that these points are
%   arranged in the format of x-y.
%

%   Date: December 31, 2016
%   Author: Zhongsheng Chen (E-mail:zschen@mail.edu.cn)


numTrain = 200;
numTest = 150;
numAll = numTrain + numTest;

xmin = -2*pi; xmax = 2*pi;

x = rand(numAll, 1) * (xmax - xmin) + xmin;
y = sin(x);
alldata = [x, y];

[traindata, testdata] = dividedataset(alldata, numTrain);

savedataset('../data/sin_train', traindata, '../data/sin_test', testdata);






