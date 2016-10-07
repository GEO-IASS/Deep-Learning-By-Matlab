%% //导入数据
% http://www.cnblogs.com/tornadomeet/p/3261247.html
% 实验一 mnist，训练样本数为60000，测试样本为10000，
% 2个隐含层，每个隐含层节点个数都是100，
% 即整体网络结构为：784-100-100-10. 
% 实验对比了有无使用denoise技术时识别的错误率
% 以及两种情况下学习到了的特征形状，

load mnist_uint8;
train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y  = double(test_y);
% 实验1 自编码
% 实验2 降噪自编码
%% //实验一：采用 autoencoder进行预训练
rng(0);
sae = saesetup([784 100 100]); 
% //其实这里nn中的W已经被随机初始化过

sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.; 
%这里的denoise autocoder相当于隐含层的dropout,但它是分层训练的

opts.numepochs =   1;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);
% //无监督学习，不需要传入标签值，学习好的权重放在sae中，
 %  //并且train_x是最后一个隐含层的输出。由于是分层预训练
%  //的，所以每次训练其实只考虑了一个隐含层，隐含层的输入有
%  //相应的denoise操作
visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([784 100 100 10]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
%add pretrained weights
nn.W{1} = sae.ae{1}.W{1}; % //将sae训练好了的权值赋给nn网络作为初始值，覆盖了前面的随机初始化
nn.W{2} = sae.ae{2}.W{1};
% Train the FFNN
opts.numepochs =   1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
str = sprintf('testing error rate is: %f',er);
disp(str)


%% //实验二：采用denoising autoencoder进行预训练
rng(0);
sae = saesetup([784 100 100]); % //其实这里nn中的W已经被随机初始化过
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.5; %这里的denoise autocoder相当于隐含层的dropout,但它是分层训练的
opts.numepochs =   1;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);% //无监督学习，不需要传入标签值，学习好的权重放在sae中，
                                    %  //并且train_x是最后一个隐含层的输出。由于是分层预训练
                                    %  //的，所以每次训练其实只考虑了一个隐含层，隐含层的输入有
                                    %  //相应的denoise操作
figure,visualize(sae.ae{1}.W{1}(:,2:end)')
% Use the SDAE to initialize a FFNN
nn = nnsetup([784 100 100 10]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
%% add pretrained weights 将sae训练好了的权值赋给nn网络作为初始值，
% 覆盖了前面的随机初始化
nn.W{1} = sae.ae{1}.W{1}; 
nn.W{2} = sae.ae{2}.W{1};
%% Train the FFNN
opts.numepochs =   1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
str = sprintf('testing error rate is: %f',er);
disp(str)