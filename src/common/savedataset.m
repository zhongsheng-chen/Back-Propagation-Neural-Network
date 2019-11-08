function status = savedataset(varargin)
%A helper function saving data points to a file.
% Args:
%   varargin: hold one or more pairs of data wanted to save and the
%       corresponding file name.
% Rturns:
%   true if writing data points to the file successfully. Or return false.
% For example,
%       status = savedataset('dataset.txt', dataset);
%       status = savedataset('traindata.txt', traindata, ...
%                            'testdata.txt', testdata);
%       savedataset('dataset.txt', dataset);
%       savedataset('traindata.txt', traindata, ...
%                            'testdata.txt', testdata);

%   Date: December 31, 2016
%   Author: Zhongsheng Chen (E-mail:zschen@mail.edu.cn)

assert(rem(nargin, 2) == 0, 'Number of input must be even.')

status = false;
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