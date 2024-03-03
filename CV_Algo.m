function [pms,bestCIter,bestGIter,bestRIter]= CV_Algo(X_miss,maskInv,ytr,Xte,yte,pms,n_alpha)
%CV_Algo 此处显示有关此函数的摘要
%   此处显示详细说明
Ntr = size(ytr,1); Nte = size(yte,1);
bestVal = 0;
bestEta = 0;
bestC = 0;
bestS = 0;
bestR = 0;

C_base = pms.C;
g_base = 1/(2*pms.sigma^2);
for i = -1:1
    pms.C = C_base*2^(i);
    for j = -1:1
        gamma = g_base*2^(j);
        pms.sigma = sqrt(1/gamma/2);
        pms.eta = n_alpha;
        r_iter = 0.2;
        pms.r = Ntr*pms.missRatio*r_iter;
        [alpha,bias,tildeK,~,~] = Our_Algo(X_miss,maskInv,ytr,pms);

        X_filled = fillData(X_miss,maskInv,tildeK,pms.sigma);
        Kval = kermat(X_filled,Xte,'Gaussian',pms.sigma);
        predVal = sign((alpha.*ytr)'*Kval+bias)';
        accVal = sum((predVal.*yte)>0)/Nte;

        if accVal >= bestVal
            bestVal = accVal;
            bestEta = pms.eta;
            bestC = pms.C;
            bestS = pms.sigma;
            bestR = pms.r;
        end
    end
end

pms.eta = bestEta;
pms.C = bestC;
pms.sigma = bestS;
pms.r = bestR;

bestCIter = log2(bestC/C_base);
bestGIter = log2((1/2/bestS/bestS)/g_base);
bestRIter = bestR/Ntr;
end

