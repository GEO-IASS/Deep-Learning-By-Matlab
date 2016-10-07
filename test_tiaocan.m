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
a=[50 100 150 200 250 300];
b=[50 100 150 200 250 300];
z=[50 100 150 200 250 300];
jhy=[];
for i=1:length(a)
for j=1:length(b)
    for k=1:length(z)
       
r=randperm( size(x,1) );   %生成关于行数的随机排列行数序列
X=x(r, :);                              %根据这个序列对A进行重新排序
train_x =X(1:400,1:800);
test_x  =X(401:505,1:800);
train_y = X(1:400,801:802);
test_y  =X(401:505,801:802);



jhy(i,j,k)=tiaocan(train_x,train_y,test_x,test_y,a(i),b(j),z(k) );
    end
end
end