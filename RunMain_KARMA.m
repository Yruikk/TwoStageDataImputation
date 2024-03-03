clear;close all;%clc;
rng(0);
%%
tic
for miss = 0.2:0.2:0.8
    % miss
    repTime = 8;
    resKARMA = zeros(repTime,2);
    spmd
        codistributed(resKARMA);
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
        maskInv = DATA{iter}.maskInv;
        % 计算是否有全部缺失的数据
        allOnes = all(maskInv == 1, 2);
        X_miss(allOnes, :) = []; maskInv(allOnes, :) = []; ytr(allOnes, :) = [];

        X_miss(maskInv==1) = 0; maskInv = zeros(size(X_miss));
        %% CV.
        pms = CV_KARMA(X_miss,maskInv,ytr,Xval,yval);
        %% Main KARMA.
        [~,~,accTr,accTe] = SVM_KARMA(X_miss,maskInv,ytr,Xte,yte,pms);
        resKARMA(iter,:) = [accTr, accTe];
    end
    %% Final Results.
    KARMA = [mean(resKARMA,1)/100; std(resKARMA,1)/100]
end
toc
