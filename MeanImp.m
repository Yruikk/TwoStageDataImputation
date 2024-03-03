function Xfill = MeanImp(Xmiss,missMask,Xte)
%MEANIMP 此处显示有关此函数的摘要
%   missMask 0 for completed 1 for missing.

N = size(Xmiss,1);

neiInd = cell(N,1);
Xfill = Xmiss;
XAll = [Xmiss;Xte];
missAll = [missMask;zeros(size(Xte))];


for i = 1:N
    neiInd{i} = find(missMask(i,:)==1);
    if isempty(neiInd{i})
        continue;
    end
    for m = 1:length(neiInd{i})
        featInd = neiInd{i}(m);
        Xfill(i,featInd) = mean(XAll(missAll(:,featInd)==0,featInd));
    end
end
end


