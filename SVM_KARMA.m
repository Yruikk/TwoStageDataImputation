function [alpha,bias,accTr,accTe] = SVM_KARMA(X_miss,maskInv,ytr,Xte,yte,pms)
%SVM_KARMA 此处显示有关此函数的摘要
%   此处显示详细说明
Ntr = size(ytr,1); Nte = size(yte,1);
missMask = ones(size(maskInv)) - maskInv;
kernel_type = 'KARMA';
s = ones(Ntr,1);
Ktr = kerMatForMiss(X_miss,X_miss,maskInv,kernel_type,pms.sigma);

H = diag(ytr)*Ktr*diag(ytr)+1e-6*eye(size(Ktr));
% Check H PSD.
[V,D] = eig(H);
D = diag(D);
D(D<1e-5) = 1e-6;
D = diag(D);
H = V * D * V';

f = -ones(Ntr,1);
lb = zeros(Ntr,1);
ub = pms.C*ones(Ntr,1);
A = []; b = []; Aeq = ytr'; beq = 0;
alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub);

[~,I] = sort(alpha);
ind = I(round(Ntr/2));
bias = s(ind)*ytr(ind) - (alpha.*ytr)'*Ktr(:,ind);
predTr = sign((alpha.*ytr)'*Ktr+bias)';
accTr = sum((predTr.*ytr)>0)/Ntr*100;

missMaskTe = ones(size(Xte));
% nanElements = isnan(Xte);
% missMaskTe = double(not(nanElements));

Nx = size(ytr,1);
Ny = size(yte,1);
M = zeros(Nx,Ny);
for i=1:Nx
    for j =1:Ny
        ind = 1*(missMask(i,:) & missMaskTe(j,:));
        M(i,j) = sum(1-ind);
    end
end
O = size(X_miss,2)*ones(size(Nx,Ny))-M;
W = pms.sigma * ones(size(O));
W(O~=1) = (O(O~=1).^pms.sigma-1)./(O(O~=1)-1);
Kte = W.*(X_miss*Xte');

predTe = sign((alpha.*ytr./s)'*Kte+bias)';
accTe = sum((predTe.*yte)>0)/Nte*100;
end

