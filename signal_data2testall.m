%% ��������
clc
clear variables
close all
M=8;snapshot=256;
f0=1e6;
fc=1e6;
fs=4*f0;
C=M*(M-1);
%%
MaxItr = 1000;
ErrorThr = 1e-3;     % RVM ��ֹ���

DOA11=[];
DOA22=[];
k1=[5.5,13.5,20.67,50:10:70];
% k1=[5,10];
k=repmat(k1,1,1);
D_start=-60;
D_stop=59;
for l=1:length(k)
    DOA1=-58:1.5:58-k(l);
    DOA2=-58+k(l):1.5:58;
    DOA11=[DOA11,DOA1];
    DOA22=[DOA22,DOA2];
end
DOA_train=[DOA11;DOA22]+0.2*rand(2,length(DOA11))-0.1;
LM=size(DOA_train,2);
theta=D_start:1:D_stop;
L=length(theta);
A=exp(1i*pi*fc*(0:M-1)'*sind(theta)/f0);
H=zeros(M*M,L);
for i=1:M
    fhi=A*diag(exp(-1i*pi*(i-1)*sind(theta)));
    H((i-1)*M+1:i*M,:)=fhi;
end

S_label=zeros(LM,L);
gamma=zeros(LM,L);
gamma_R=zeros(LM,L);
R_est=zeros(LM,C);
S_est=zeros(LM,L,2);
for i=1:LM
    [X1,~]=signal_generate(M,snapshot,DOA_train(1,i),f0,fc,fs,1);
    [X2,~]=signal_generate(M,snapshot,DOA_train(2,i),f0,fc,fs,1);
    temp1=awgn(X1,0,'measured');
    temp2=awgn(X2,0,'measured');
    X= temp1+ temp2;
    [R_est(i,:),Rx]=feature_extract_R(X) ;
    temp=H'*vec(Rx);
    temp=temp/norm(temp);
    S_est(i,:,1)=real(temp);
    S_est(i,:,2)=imag(temp);


    S_label(i,round(DOA_train(1,i))+61)=1;
    S_label(i,round(DOA_train(2,i))+61)=1;
    [gamma_R(i,:)] = SBAC_R(X,H,MaxItr,ErrorThr, S_label(i,:),2);

    [gamma(i,:)] = SBAC(X,A,MaxItr,ErrorThr);

i
end
% % %

close all
plot(theta,S_est(i,:,1))
xlim([-60,60])
hold on
plot(theta,(S_label(i,:)'))
grid on
plot(theta,(gamma(i,:)'))
plot(theta,(gamma_R(i,:)'))
legend('A','true','S','R')


save('data2_testall.mat','R_est','DOA_train',...
    'S_label','S_est','theta','gamma','gamma_R')

