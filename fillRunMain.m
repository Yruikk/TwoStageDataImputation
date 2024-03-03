% clc;
clear;close all;
%% Data Loading.
[rawY,rawX]=libsvmread('Data/heart_scale.txt'); % 270x13;
[Yall,~] = mapminmax(rawY',-1,1);

Xall = full(rawX);
[Xall,~] = mapminmax(Xall',0,1);
X_all = Xall';
Y_all = Yall'; 
N = size(Y_all,1);
d = size(X_all,2);
%% Missing data.
X_max_list = [];
X_mean_list = [];
K_max_list = [];
K_mean_list = [];
m = [0.1 0.9 0.98];
for i = 1:3
    missRatio = m(i);
    fprintf('MissRatio = %.2f.\n', missRatio);
    [mask,missNum] = genMissMask(N,d,missRatio,'random');
    X_miss = X_all.*mask; % Missing values are replaced by 0.

    maskInv = ones(size(mask))-mask; % 1 for missing

    % Gaussian Kernel.
    gamma = 2^(5);
    sigma = sqrt(1/2/gamma);
    [Kk,M] = kerMatForMiss(X_miss,X_miss,maskInv,'Gaussian',sigma);
    K = kermat(X_all,X_all,'Gaussian',sigma);
    
    DeltaX = X_all-X_filled;
    X_max_list = [X_max_list; max(abs(DeltaX),[],'all')];
    X_mean_list = [X_mean_list; mean(abs(DeltaX),'all')];
    
    K_filled = kermat(X_filled,X_filled,'Gaussian',sigma);
    DeltaK = K-K_filled;
    K_max_list = [K_max_list; max(abs(DeltaK),[],'all')];
    K_mean_list = [K_mean_list; mean(abs(DeltaK),'all')];
    

%     figure(2*i-1);
    figure;
    subplot(1,2,1);
    surf(DeltaX);
    subplot(1,2,2)
    surf(DeltaK);
    
    set(gcf, 'Position', [500 500 1300 500]);
end
FINAL = [X_max_list X_mean_list K_max_list K_mean_list];










