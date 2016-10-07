clear all; clc; close all;

% params
params.maxIter = 100;
params.numData = 45;
params.numClasses = 3;
params.lambda = 1e-4;
params.learningRate = 0.01;

% generate data
train.X{1} = randi([1 100], 2, params.numData);
% 产生2*params.numData的矩阵 元素位于1-100间
train.Y{1} = ones(1, params.numData) * 1;

train.X{2} = randi([200 255], 2, params.numData);
train.X{2}(1, :) = train.X{2}(1, :) - 180;
train.Y{2} = ones(1, params.numData) * 2;

train.X{3} = randi([180 255], 2, params.numData);
train.Y{3} = ones(1, params.numData) * 3;

train.X = cat(2, train.X{:});

% normalize data
train.X = double(train.X)/255;

% add bias so that the 1st row is the bias for each category classifier
% bias is very important in this example, no bias, wrong result
train.X = [ones(1, size(train.X,2)); train.X]; 
train.Y = cat(2, train.Y{:});
% w parameters initialization
w = randn(d, params.numClasses);

% convert real value label to label matrix
% 1, 2, 3 -> 001, 010, 100.
l = bsxfun(@(y, ypos) (y == ypos), train.Y', 1:params.numClasses);
rsp = w' * train.X; 
rsp = bsxfun(@minus, rsp, max(rsp, [], 1));
rsp = exp(rsp);
prob = bsxfun(@rdivide, rsp, sum(rsp));
% compute gradient based on current probability
g = - train.X * (l - prob') / n + params.lambda * w;

% update w
w = w - params.learningRate * g;
% compute cost 
logProb = log(prob);
idx = sub2ind(size(logProb), train.Y, 1:size(logProb, 2));
cost = - sum(logProb(idx)) / n + params.lambda * 0.5 * sum(sum(w.^2));