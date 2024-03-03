function pms = CV_KARMA(Xtr,maskInv,ytr,Xval,yval)
%CV_KARMA 此处显示有关此函数的摘要
%   此处显示详细说明
bestcv = 0;
bestc = 0;
bestG = 0;
for i = -5:5
    for j = 1:4
        pms.C = 2^i;
        pms.sigma = j;
        [~,~,tr,cv] = SVM_KARMA(Xtr,maskInv,ytr,Xval,yval,pms);
        if cv > bestcv && tr > cv
            bestcv = cv;
            bestc = 2^i;
            bestG = j;
        end
    end
end
pms.C = bestc;
pms.sigma = bestG;
end

