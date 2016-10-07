clc;clear
x=load('ex4x.dat');
y=load('ex4y.dat');
% plot(x,y,'o')
[m,n]=size(x)
x=[ones(m,1),x]
pos=find(y==1)
neg=find(y==0)
figure
plot(x(pos,2),x(pos,3),'o')
hold on 
plot(x(neg,2),x(neg,3),'+')
hold on 
xlabel('ex1 socre')
ylabel('ex2 score')
theta=zeros(size(x,2),1)
sig=inline('1./(1+exp(-z))')
iter_max=7
j=zeros(iter_max,1)
for i =1:iter_max
    z=x*theta
    h=sig(z)
    grad=(1/m).*x'*(h-y)
    H=(1/m).*x'*diag(h)*diag(1-h)*x
    
    J(i)=(1/m)*sum(-y.*log(h)-(1-y).*log(1-h))
    theta = theta - inv(H)*grad;
    
%     x=A/B x*B=A x=A*inv(B)
%     x=A\B A*x=B x=inv(A)*B
end
    
    
