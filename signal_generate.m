function [X,s,a,am_vec]=signal_generate(M,snapshot,DOA,f0,fc,fs,flag)
% % M ��Ԫ��
% % snapshot ������
% % DOA �����
% %fc �ź�Ƶ��
% %f0 ��������Ƶ��
% %�������󣬲�������Ƶ�ź�,�����������
% %fs;  ����Ƶ��
% clc ;clear variables; close all
% M=8;snapshot=256;DOA=[-54] ;f0=2e6;fc=2*10^6;fs=100*f0;flag =1;
% N_delay=sind(DOA)/2/f0*fs; % �ӳٵ���
% %% ������ȷ��ʱ��
% if flag==1
%     am_vec=1*rand(1,5*snapshot);
% elseif flag==2
%     am_vec=randn(1,5*snapshot);% ����խ����)
% elseif flag==3
%     am_vec=ones(1,5*snapshot);
% end
% X=zeros(M,snapshot);
% for i=1:M
%     s=exp(1i*pi*fc*(i-1)*sind(DOA)/f0).*exp(1i*(2*pi*fc/fs*(0:snapshot-1)));%�ź�ʱ����
%     
%     if N_delay>0
%         p=floor((i-1)*N_delay);
%         a=am_vec(p+1:p+snapshot);
%         
%         X(i,:)=a.*s;
%     else
%         q=floor((i-1)*N_delay);
%         a=am_vec(q+M/2/f0*fs+1:q+snapshot+M/2/f0*fs);
%         X(i,:)=a.*s;
%     end
%     
% end
% signal_power=(norm(X,'fro'))^2/snapshot/M;

% %%  ����խ���ź�ģ�Ͳ�����������
% if flag==1
%     am_vec=rand(1,snapshot);
% elseif flag==2
%     am_vec=randn(1,snapshot);
% elseif flag==3
%     am_vec=ones(1,snapshot);
% end
% a=exp(1i*pi*fc*(0:M-1)'*sind(DOA)/f0);%;
% s=am_vec.*exp(1i*(2*pi*fc/fs*(1:snapshot)));
% X=a*s;
% signal_power=sum(abs(s.^2),2)/snapshot;% ÿһ���źŹ���


%%  ��������Դ
if flag==1
    am_vec=2*rand(1,snapshot)-1;
elseif flag==2
    am_vec=randn(1,snapshot);
elseif flag==3
    am_vec=ones(1,snapshot);
end
a=exp(1i*pi*fc*(0:M-1)'*sind(DOA)/f0);%;
s=randn(1,snapshot)+1i*randn(1,snapshot);
X=a*s;

