function [K,M] = kerMatForMiss(X,Y,missInv,kernel,sigma)
%KERMATFORMISS
%   X and Y are m-by-n matrices, where m denotes the number of samples and n
%   denotes the dimension of samples.
%   In missInv, 1 for missing, 0 for completed.
%   Define Delta = x-y and Delta_d = x_d-y_d
%   Gaussian Kernel Function: k(Delta) = exp(-norm2(Delta)^2/(2*sigma^2))
%   Laplacian Kernel Function: k(Delta)= exp(-norm1(Delta)/sigma)
%   Cauchy Kernel Function: k(Delta) = exp(\prod_1^D (1/pi)*[1/(1+Delta_d^2/sigma^2)])
if nargin < 1
    error('rbfkernel_NeedData');
elseif nargin < 2
    Y = X;
end

if nargin < 3
    if ~ismatrix(X) || ~ismatrix(Y)
        error('rbfkernel_BadInputs');
    end
    kernel = 'Gaussian';
    sigma = 1;
elseif nargin < 4
    sigma = 1;
end

missMask = ones(size(missInv))-missInv;
Nx = size(X,1);
Ny = size(Y,1);
M = zeros(Nx,Ny);
for i=1:Nx
    for j =1:Ny
        ind = 1*(missMask(i,:) & missMask(j,:));
        M(i,j) = sum(1-ind);
    end
end
if strcmp(kernel,'Linear')
    K = X*Y';
elseif strcmp(kernel,'Poly')
    D = X*Y';
    K = (D+1).^sigma;
elseif strcmp(kernel,'Gaussian')
    K = zeros(Nx,Ny);
    for i=1:Nx
        for j =1:Ny
            ind = 1*(missMask(i,:) & missMask(j,:)); % In missMask, 1=miss 0=observed.
            x = X(i,:).*ind;
            y = Y(j,:).*ind;
            K(i,j) = exp(-norm(x-y,2)^2/(2*(sigma^2)));
        end
    end
elseif strcmp(kernel,'KARMA')
    O = size(X,2)*ones(size(Nx,Ny))-M;
    W = sigma * ones(size(O));
    W(O~=1) = (O(O~=1).^sigma-1)./(O(O~=1)-1);
    K = W.*(X*Y');
elseif strcmp(kernel,'Laplacian') || strcmp(kernel,'Cauchy')
    [mx,nx] = size(X);
    [my,ny] = size(Y);
    X = reshape(X,[mx,1,nx]);
    Y = reshape(Y,[1,my,ny]);
    if strcmp(kernel,'Laplacian')
        D = abs(X-Y);
        D = sum(D,3);
        K = exp(-D/sigma);
    elseif strcmp(kernel,'Cauchy')
        D = 1./(1+((X-Y).^2)/(sigma^2));
        K = prod(D,3);
    end
end
end