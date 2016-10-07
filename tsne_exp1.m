load('NB_DE0620.mat')
load('DE_IR0620.mat')
% NB_DE=cat(2,NB_DE,ones(404,1));
% NB_DE=cat(2,NB_DE,zeros(404,1));
% DE_IR=cat(2,DE_IR,zeros(101,1));
% DE_IR=cat(2,DE_IR,ones(101,1));
x=cat(1,NB_DE,DE_IR);

train_X=x;
ind = randperm(size(train_X, 1));
train_X = train_X(ind(1:size(train_X,1)),:);
% train_labels = train_labels(1:5000);
% Set parameters
no_dims = 2;
init_dims = 800;
perplexity = 30;
% Run t?SNE
mappedX = tsne(train_X, [], no_dims, init_dims, perplexity);
% Plot results
gscatter(mappedX(:,1), mappedX(:,2), train_labels, 'o');