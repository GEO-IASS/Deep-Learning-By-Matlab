
% 2�������㣬ÿ��������ڵ��������100��
% ����������ṹΪ��800-100-100-10. 
% ���ν��
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
%���д���
r=randperm( size(x,1) );   %���ɹ������������������������
X=x(r, :);                              %����������ж�A������������
train_x =X(1:400,1:800);
test_x  =X(401:505,1:800);
train_y = X(1:400,801:802);
test_y  =X(401:505,801:802);
%% ����denoising autoencoder����Ԥѵ��
rng(0);
sae = saesetup([800 300 300 200 ]); % //��ʵ����nn�е�W�Ѿ��������ʼ����
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.5; %�����denoise autocoder�൱���������dropout,�����Ƿֲ�ѵ����

sae.ae{3}.activation_function       = 'sigm';
sae.ae{3}.learningRate              = 1;
sae.ae{3}.inputZeroMaskedFraction   = 0.5; 

opts.numepochs =   2;%ÿ��ѵ���غ���
opts.batchsize = 20;
sae = saetrain(sae, train_x, opts);% //�޼ලѧϰ������Ҫ�����ǩֵ��ѧϰ�õ�Ȩ�ط���sae�У�
                                    %  //����train_x�����һ�������������������Ƿֲ�Ԥѵ��
                                    %  //�ģ�����ÿ��ѵ����ʵֻ������һ�������㣬�������������
                                    %  //��Ӧ��denoise����
% figure,visualize(sae.ae{1}.W{1}(:,2:end)')
% Use the SDAE to initialize a FFNN
nn = nnsetup([800 300 300 200  2]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
%% add pretrained weights ��saeѵ�����˵�Ȩֵ����nn������Ϊ��ʼֵ��
% ������ǰ��������ʼ��
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