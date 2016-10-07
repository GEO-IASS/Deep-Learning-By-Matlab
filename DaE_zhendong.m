
% 2个隐含层，每个隐含层节点个数都是100，
% 即整体网络结构为：800-100-100-10. 
% 调参结果
% 800-300-300-200-2. error rate 0.13 

clear 
close all
clc
load('NB_DE0620.mat')
load('DE_IR0620.mat')
NB_DE=cat(2,NB_DE,ones(404,1));
NB_DE=cat(2,NB_DE,zeros(404,1));
DE_IR=cat(2,DE_IR,zeros(101,1));
DE_IR=cat(2,DE_IR,ones(101,1));
x=cat(1,NB_DE,DE_IR);
%按行打乱
r=randperm( size(x,1) );   %生成关于行数的随机排列行数序列
X=x(r, :);                              %根据这个序列对A进行重新排序
train_x =X(1:400,1:800);
test_x  =X(401:505,1:800);
train_y = X(1:400,801:802);
test_y  =X(401:505,801:802);
%% 采用denoising autoencoder进行预训练
rng(0);
sae = saesetup([800 300 300 200 ]); % //其实这里nn中的W已经被随机初始化过
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.5; %这里的denoise autocoder相当于隐含层的dropout,但它是分层训练的

sae.ae{3}.activation_function       = 'sigm';
sae.ae{3}.learningRate              = 1;
sae.ae{3}.inputZeroMaskedFraction   = 0.5; 

opts.numepochs =   2;%每层训练回合数
opts.batchsize = 20;
sae = saetrain(sae, train_x, opts);% //无监督学习，不需要传入标签值，学习好的权重放在sae中，
                                    %  //并且train_x是最后一个隐含层的输出。由于是分层预训练
                                    %  //的，所以每次训练其实只考虑了一个隐含层，隐含层的输入有
                                    %  //相应的denoise操作
% figure,visualize(sae.ae{1}.W{1}(:,2:end)')
% Use the SDAE to initialize a FFNN
nn = nnsetup([800 300 300 200  2]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
%% add pretrained weights 将sae训练好了的权值赋给nn网络作为初始值，
% 覆盖了前面的随机初始化
nn.W{1} = sae.ae{1}.W{1}; 
nn.W{2} = sae.ae{2}.W{1};
nn.W{3} = sae.ae{3}.W{1};
%% Train the FFNN
opts.numepochs =   1;
opts.batchsize = 20;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
str = sprintf('testing error rate is: %f',er);
disp(str)