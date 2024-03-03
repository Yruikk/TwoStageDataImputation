function [M,missNum] = genMissMask(N,d,ratio,mask_type)
%GENMISSMASK 此处显示有关此函数的摘要
%   In M, 1 for complete and 0 for missing.
switch mask_type
    case 'random'
        allNum = N*d;
        missNum = round(allNum*ratio);
        k = ones(allNum-missNum,1);
        m = zeros(missNum,1);
        M0 = [k;m];
        rowRank = randperm(size(M0,1));
        M = M0(rowRank,:);
        M = reshape(M,[N,d]);
    case 'sample'
        missSampleNum = round(N*ratio);
        M = ones(N,1);
        
        ind = randperm(N,missSampleNum);
        ind = sort(ind,'ascend');
        M(ind) = 0;
        M = repmat(M,1,d);
        missNum = (N - missSampleNum)*d;
    case 'feature'
        missFeatNum = round(d*ratio);
        M = ones(1,d);
        
        ind = randperm(d,missFeatNum);
        ind = sort(ind,'ascend');
        M(ind) = 0;
        M = repmat(M,N,1);
        missNum = (d - missFeatNum)*N;
end

end

