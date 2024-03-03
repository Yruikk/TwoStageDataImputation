function [pms,cmd] = CV_MI(Xtr,maskInv,ytr,Xval,yval)
%CV_MI 此处显示有关此函数的摘要
%   此处显示详细说明
bestcv = 0;
bestc = 0;
bestg = 0;
for i = -5:5
    for j = -5:5
        cmd =['-c ',num2str(2^i),' -g ',num2str(2^j),' -q'];

        pms.C = 2^i;
        pms.sigma = sqrt(1/2^j/2);
        [~,~,accTr,cv,~] = SVM_MI(Xtr,maskInv,ytr,Xval,yval,cmd);
        if cv > bestcv && accTr > cv
            bestcv = cv;
            bestc = 2^i;
            bestg = 2^j;
        end
    end
end

cmd =['-c ',num2str(bestc),' -g ',num2str(bestg),' -q'];
pms.C = bestc;
pms.sigma = sqrt(1/bestg/2);
end

