function X_filled = fillDataGaussian(X_miss,missMask,y,tildeK,sigma,ff)
%FILLDATAGAUSSIAN 此处显示有关此函数的摘要
%   X_miss should be a N*d matrix. N (sample number) and d (dimension).
%   In missMask, 1 for missing and 0 for completed.
N = size(X_miss,1);
neiInd0 = cell(N,1); % Indicates several dimensions of missing features in the sample. If it is empty, the i-th sample is complete.
neiInd1 = cell(N,1); % Indicates whether other data is missing(1) or complete(0) on the dimension that is missing in the i-th sample.
neiInd2 = cell(N,1); % Indicates which samples provide constraints for the recovery of the i-th sample.
neiFeature = cell(N,1); % Indicates the features of the other samples in the dimension that is missing in the i-th sample.
missFeatNum = zeros(N,1);
neiNum = zeros(N,1);

missMask0 = missMask;
Kk = kerMatForMiss(X_miss,X_miss,missMask,'Gaussian',sigma); % For the linear kernel, the missing feature is substituted by 0, so the missing feature happens not to be counted.
Ku = tildeK./Kk;
KuLog = -2*(sigma^2).*log(Ku);

for i = 1:N
    neiInd0{i} = find(missMask(i,:)==1); % Find out where the i-th sample is missing the feature.
    if isempty(neiInd0{i})
        continue;
    end
    missFeatNum(i,1) = length(neiInd0{i});
    neiInd1{i} = missMask(:,neiInd0{i}); 
    neiInd2{i} = find(sum(missMask,2)==0);
    neiNum(i,1) = length(neiInd2{i});
    neiFeature{i} = X_miss(neiInd2{i},neiInd0{i});
end
missDataInd = (1:N)';
sortMat = [missDataInd neiNum missFeatNum];
compleDataInd = find(missFeatNum==0);
sortMat(compleDataInd,:) = []; %#ok<FNDSB>
sortedMat = sortrows(sortMat,[2 3],{'descend' 'ascend'}); % Priority of data recovery: 1. More constraint equations; 2. Less feature-missing.
if sortedMat(1,2)==0
%     error('Cannot be solved.');
    fprintf('Need prefill X.\n');
    preFlag = 1; 
else
    fprintf('Can be solved.\n');
    preFlag = 0;
end

if ff == 1
    preFlag = 1; 
elseif ff == 0
    preFlag = 0;
end

% Start to fill.
if preFlag == 0
    X_filled = X_miss;
    for i = 1:size(sortedMat,1)
        fillInd = sortedMat(i,1);
        % Optimization.
        A = neiFeature{fillInd};
        A = A';
        b = KuLog(neiInd2{fillInd},fillInd);
        fun = @(x)paramFunGaussian(x,A,b);
        x0 = 0.5*ones(size(A,1),1);
        options = optimoptions('fsolve','Algorithm','levenberg-marquardt','Display','none');
        missValue = fsolve(fun,x0,options);
        
        X_filled(fillInd,neiInd0{fillInd}) = missValue;
        missMask(fillInd,neiInd0{fillInd}) = 0; % The features of the sample are no longer considered missing.
        % After each sample is filled, Kk and Ku change accordingly.
        Kk = kerMatForMiss(X_filled,X_filled,missMask,'Gaussian',sigma);
        Ku = tildeK./Kk;
        KuLog = -2*sigma*sigma*log(Ku);
        for j = i+1:size(sortedMat,1)
            fillInd1 = sortedMat(j,1);
            neiInd1{fillInd1} = missMask(:,neiInd0{fillInd1});
            neiInd2{fillInd1} = find(sum(missMask,2)==0);
            neiNum(fillInd1,1) = length(neiInd2{fillInd1});
            neiFeature{fillInd1} = X_filled(neiInd2{fillInd1},neiInd0{fillInd1});
        end
    end
elseif preFlag == 1
    X_filled = X_miss;
    for iter = 1:2
        nInd = sortedMat(1,1);
        dInd = neiInd0{nInd};
        if iter == 1
            Xprefill = classMeanImp(X_miss,missMask,y);
            X_filled(nInd,dInd) = Xprefill(nInd,dInd);
            nInd0 = nInd;
            dInd0 = dInd;
        else
            missFeatNum = length(dInd0);
            samplePick = sortedMat(1:missFeatNum+1,1);
            allInd = [nInd0;samplePick];
            allInd = sort(allInd,'ascend');
            cutX = X_filled(allInd,:);
            cutK = tildeK(allInd,allInd);
            cutMiss = zeros(size(cutX));
            cutMiss(allInd==nInd0,dInd0) = 1;
            
            cutKk = kerMatForMiss(cutX,cutX,cutMiss,'Gaussian',sigma);
            cutKu = cutK./cutKk;
            cutKuLog = -2*sigma*sigma*log(cutKu);
            
%             b = cutKuLog(allInd~=nInd0,allInd==nInd0);
            
            
            
            A = cutX(allInd~=nInd0,dInd0);
            A = A';
            b = cutKuLog(allInd~=nInd0,allInd==nInd0);
            fun = @(x)paramFunGaussian(x,A,b);
            x0 = 0.5*ones(size(A,1),1);
            options = optimoptions('fsolve','Algorithm','levenberg-marquardt');
            missValue = fsolve(fun,x0,options);
            
            X_filled(nInd,neiInd0{nInd}) = missValue;
        end
        missMask(nInd,dInd) = 0;
        missFeatNum(nInd,1) = 0;
        neiInd1{nInd} = [];
        neiInd2{nInd} = [];
        neiNum(nInd,1) = 0;
        neiFeature{nInd} = [];
        for i = 1:N
            neiInd0{i} = find(missMask(i,:)==1);
            if isempty(neiInd0{i})
                continue;
            end
            missFeatNum(i,1) = length(neiInd0{i});
            neiInd2{i} = find(sum(missMask,2)==0);
            neiNum(i,1) = length(neiInd2{i});
            neiFeature{i} = Xprefill(neiInd2{i},neiInd0{i});
        end
        missDataInd = (1:N)';
        sortMat = [missDataInd neiNum missFeatNum];
        compleDataInd = find(missFeatNum==0);
        sortMat(compleDataInd,:) = []; %#ok<FNDSB>
        sortMat(nInd,:) = [];
        sortedMat = sortrows(sortMat,[2 3],{'descend' 'ascend'});
    end
    for i = 1:size(sortedMat,1)
        fillInd = sortedMat(i,1);
        % Optimization.
        A = neiFeature{fillInd};
        if isempty(A)
            error('11');
        end
        A = A';
        b = KuLog(neiInd2{fillInd},fillInd);
        fun = @(x)paramFunGaussian(x,A,b);
        x0 = 0.5*ones(size(A,1),1);
        options = optimoptions('fsolve','Algorithm','levenberg-marquardt','Display','none');
        missValue = fsolve(fun,x0,options);
        
        X_filled(fillInd,neiInd0{fillInd}) = missValue;
        missMask(fillInd,neiInd0{fillInd}) = 0; % The features of the sample are no longer considered missing.
        % After each sample is filled, Kk and Ku change accordingly.
        Kk = kerMatForMiss(X_filled,X_filled,missMask,'Gaussian',sigma);
        Ku = tildeK./Kk;
        KuLog = -2*sigma*sigma*log(Ku);
        for j = i+1:size(sortedMat,1)
            fillInd1 = sortedMat(j,1);
            neiInd2{fillInd1} = find(sum(missMask,2)==0);
            neiNum(fillInd1,1) = length(neiInd2{fillInd1});
            neiFeature{fillInd1} = X_filled(neiInd2{fillInd1},neiInd0{fillInd1});
        end
    end
%     这是迭代了一次，还要把原来用ClassMeanImp补的地方重补。
    
    
   
end
end








