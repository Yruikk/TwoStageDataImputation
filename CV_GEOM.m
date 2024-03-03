function [pms,cmd] = CV_GEOM(Xtr,maskInv,ytr,Xval,yval,kernel_type)
%CV_GEOM 此处显示有关此函数的摘要
%   此处显示详细说明
bestcv = 0;
bestc = 0;
bestg = 0;
bestSP = 0;
for i = -5:5
    for j = -5:5
        for k = 2:5
            pms.C = 2^i;
            pms.sigma = sqrt(1/2^j/2);
            pms.SP = k;
            [~,~,accTr,cv] = SVM_GEOM(Xtr,maskInv,ytr,Xval,yval,kernel_type,pms);
            if cv > bestcv && accTr > cv
                bestcv = cv;
                bestc = 2^i;
                bestg = 2^j;
                bestSP = k;
            end
        end
    end
end
cmd =['-c ',num2str(bestc),' -g ',num2str(bestg),' -q'];
pms.C = bestc;
pms.sigma = sqrt(1/bestg/2);
pms.SP = bestSP;

end

