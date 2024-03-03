function [alpha,bias,accTr,accTe,Xfill] = SVM_MI(X_miss,maskInv,ytr,Xte,yte,cmd)
%SVM_MI 此处显示有关此函数的摘要
%   此处显示详细说明
Xfill = MeanImp(X_miss,maskInv,Xte);

model = svmtrain(ytr,Xfill,cmd);
[~,accTr,~] = svmpredict(ytr,Xfill,model);
[~,accTe,~] = svmpredict(yte,Xte,model);
accTr = accTr(1,1);
accTe = accTe(1,1);
alpha = model.sv_coef;
bias = -model.rho;

end

