clear;close all;%clc;
rng(0);
%%
repTime = 8;
resSVM = zeros(repTime,2);
resAlgoVal = zeros(repTime,2);
etaListVal = zeros(repTime,1);
etaListTe = zeros(repTime,1);
nalphaList = zeros(repTime,1);
CList = zeros(repTime,1);
SList = zeros(repTime,1);

Algo_list = cell(4,1);
C_list = cell(4,1);
Gamma_list = cell(4,1);
R_list = cell(4,1);

spmd
    codistributed(resAlgoVal);
    codistributed(etaListVal);
    codistributed(etaListTe);
    codistributed(nalphaList);
    codistributed(CList);
    codistributed(SList);
end
data_name = 'australian'
ii = 1;
for miss = 0.2:0.2:0.8

    %% Load Data.
    DATA = cell(repTime,1);
    for iter = 1:repTime
        path = ['./repData4-3-3/',data_name,'/missRatio=',num2str(miss),'/',num2str(iter),'/m=',...
            num2str(miss),'rep=',num2str(iter),'.mat'];
        DATA{iter} = load(path);
    end
    %% Main Experiments.
    for iter = 1:repTime
        X_miss = DATA{iter}.Xtr; ytr = DATA{iter}.ytr;
        Xval = DATA{iter}.Xval; yval = DATA{iter}.yval;
        Xte = DATA{iter}.Xte; yte = DATA{iter}.yte;
        maskInv = DATA{iter}.maskInv; Xgt = DATA{iter}.Xgt;
        %% MI.
        [pms,cmd] = CV_MI(X_miss,maskInv,ytr,Xval,yval);
        pms.missRatio = miss;
        CList(iter,:) = pms.C;
        SList(iter,:) = pms.sigma;
        pms_MI = pms;
        [alpha,~,~,~,~] = SVM_MI(X_miss,maskInv,ytr,Xte,yte,cmd,1);
        n_alpha = norm(alpha);
        nalphaList(iter,:) = n_alpha;
        [pms,Cind,Gind,Rind]= CV_Algo(X_miss,maskInv,ytr,Xval,yval,pms,n_alpha);

        [~,accTr,accTe,~,~,~,~]= SVM_Algo(X_miss,maskInv,ytr,Xte,yte,pms);
        resAlgoVal(iter,:) = [accTr/100, accTe/100];
    end
    %% Final Results.
    AlgoVal = [mean(resAlgoVal,1);std(resAlgoVal,1)]
    ii=ii+1;
end