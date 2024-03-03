function [alpha,bias,accTr,accTe] = SVM_GEOM(X_miss,maskInv,ytr,Xte,yte,kernel_type,pms)
%SVM_GEOM 此处显示有关此函数的摘要
%   此处显示详细说明
Ntr = size(ytr,1); Nte = size(yte,1);
missMask = ones(size(maskInv))-maskInv; % 0 for missing, 1 for completed.
s = ones(Ntr,1);
switch kernel_type
    case 'Poly'
        Ktr = kermat(X_miss,X_miss,kernel_type,pms.d);
    case 'Gaussian'
        [Ktr,~] = kerMatForMiss(X_miss,X_miss,maskInv,'Gaussian',pms.sigma);
end
[V,d] = eig(Ktr);
dd = diag(d);
dd(dd<1e-6) = 1e-6;
Ktr = V*diag(dd)*V';

for iter = 1:pms.SP
    H = diag(ytr./s)*Ktr*diag(ytr./s);
    
    f = -ones(Ntr,1);
    lb = zeros(Ntr,1);
    ub = pms.C*ones(Ntr,1);
    A = []; b = []; Aeq = ytr'; beq = 0;
    alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub);
        
    alpha(alpha<1e-6) = 0;
    w = sum(repmat(alpha.*ytr,1,size(X_miss,2)).*X_miss,1)';
    denominator = norm(w,2);
    numerator = zeros(Ntr,1);
    for k = 1:Ntr
        mask_i = missMask(k,:)';
        w_i = w.*mask_i;
        numerator(k) = norm(w_i,2);
    end
    s = numerator/denominator;
    s(numerator==0) = 1;
    s(s<1e-6) = 1e-6;
end
[~,I] = sort(alpha);
ind = I(round(Ntr/2));
bias = s(ind)*ytr(ind) - (alpha.*ytr./s)'*Ktr(:,ind);
predTr = sign((alpha.*ytr./s)'*Ktr+bias)';
accTr = sum((predTr.*ytr)>0)/Ntr*100;

missMask_te = ones(size(Xte));
% nanElements = isnan(Xte);
% missMask_te = double(not(nanElements));

Kte = zeros(Ntr,Nte);
for i=1:Ntr
    for j =1:Nte
        ind = 1*(missMask(i,:) & missMask_te(j,:));
        x = X_miss(i,:).*ind;
        y = Xte(j,:).*ind;
        Kte(i,j) = exp(-norm(x-y,2)^2/(2*(pms.sigma^2)));
    end
end

predTe = sign((alpha.*ytr./s)'*Kte+bias)';
accTe = sum((predTe.*yte)>0)/Nte*100;
end

