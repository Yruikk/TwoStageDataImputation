clear;close all;%clc;
rng(0);

for miss = 0.2:0.2:0.8
    repTime = 8;
    resGEOM = zeros(repTime,2);
    spmd
        codistributed(resGEOM);
    end
    %% Load Data.
    data_name = 'australian';
    DATA = cell(repTime,1);
    for iter = 1:repTime
        path = ['./repData4-3-3/',data_name,'/missRatio=',num2str(miss),'/',num2str(iter),'/m=',...
            num2str(miss),'rep=',num2str(iter),'.mat'];
        DATA{iter} = load(path);
    end
    %%
    parfor iter = 1:repTime
        X_miss = DATA{iter}.Xtr; ytr = DATA{iter}.ytr;
        Xval = DATA{iter}.Xval; yval = DATA{iter}.yval;
        Xte = DATA{iter}.Xte; yte = DATA{iter}.yte;
        maskInv = DATA{iter}.maskInv; Xgt = DATA{iter}.Xgt;
        % 计算是否有全部缺失的数据
        allOnes = all(maskInv == 1, 2);
        X_miss(allOnes, :) = []; maskInv(allOnes, :) = []; ytr(allOnes, :) = [];
        %% CV.
        kernel_type = 'Gaussian';
        [pms,cmd] = CV_GEOM(X_miss,maskInv,ytr,Xval,yval,kernel_type);
        %% Main GEOM.
        [~,~,accTr,accTe] = SVM_GEOM(X_miss,maskInv,ytr,Xte,yte,kernel_type,pms);
        resGEOM(iter,:) = [accTr, accTe];
    end
    %% Final Results.
    GEOM = [mean(resGEOM,1)/100 ;std(resGEOM,1)/100]
end
toc