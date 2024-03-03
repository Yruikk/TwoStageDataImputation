function [alpha,accTr,accTe,X_filled,tildeK,K_Delta,E]= SVM_Algo(X_miss,maskInv,ytr,Xte,yte,pms)
%SVM_ALGO 此处显示有关此函数的摘要
%   此处显示详细说明
Ntr = size(ytr,1); Nte = size(yte,1);
[alpha,bias,tildeK,K_Delta,E] = Our_Algo(X_miss,maskInv,ytr,pms);

predTr = sign((alpha.*ytr)'*tildeK+bias)';
accTr = sum((predTr.*ytr)>0)/Ntr*100;

X_filled = fillData(X_miss,maskInv,tildeK,pms.sigma);
Kte = kermat(X_filled,Xte,'Gaussian',pms.sigma);
predTe = sign((alpha.*ytr)'*Kte+bias)';
accTe = sum((predTe.*yte)>0)/Nte*100;