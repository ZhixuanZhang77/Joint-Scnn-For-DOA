
% �Զ�ά���ݾ�������ȡ2������ƽ����

function gamma = Row2Norm_Fun(mu)

[L,N] = size(mu);
gamma=diag(mu*mu')/N;
end