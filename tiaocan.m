function [er ] = tiaocan(train_x,train_y,test_x,test_y,s2,s3,s4 )
%% ����denoising autoencoder����Ԥѵ��
rng(0);
sae = saesetup([800 s2 s3 s4 ]); % //��ʵ����nn�е�W�Ѿ��������ʼ����
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.5; %�����denoise autocoder�൱���������dropout,�����Ƿֲ�ѵ����

sae.ae{3}.activation_function       = 'sigm';
sae.ae{3}.learningRate              = 1;
sae.ae{3}.inputZeroMaskedFraction   = 0.5; 

opts.numepochs =   1;%ÿ��ѵ���غ���
opts.batchsize = 20;
sae = saetrain(sae, train_x, opts);% //�޼ලѧϰ������Ҫ�����ǩֵ��ѧϰ�õ�Ȩ�ط���sae�У�
                                    %  //����train_x�����һ�������������������Ƿֲ�Ԥѵ��
                                    %  //�ģ�����ÿ��ѵ����ʵֻ������һ�������㣬�������������
                                    %  //��Ӧ��denoise����
% figure,visualize(sae.ae{1}.W{1}(:,2:end)')
% Use the SDAE to initialize a FFNN
nn = nnsetup([800 s2 s3 s4  2]);
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
end

