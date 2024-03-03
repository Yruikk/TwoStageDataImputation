function K = kermat(X,Y,kernel,sigma)
%KERMAT
% X and Y are m-by-n matrices
% m denotes the number of samples
% n denotes the dimension of samples
% Define Delta = x-y and Delta_d = x_d-y_d
% Gaussian Kernel Function: k(Delta) = exp(-norm2(Delta)^2/(2*sigma^2))
% Laplacian Kernel Function: k(Delta)= exp(-norm1(Delta)/sigma)
% Cauchy Kernel Function: k(Delta) = exp(\prod_1^D (1/pi)*[1/(1+Delta_d^2/sigma^2)])
if nargin < 1
    error('kermat_NeedData');
elseif nargin < 2
    Y = X;
end

if nargin < 3
    if ~ismatrix(X) || ~ismatrix(Y)
        error('kermat_BadInputs');
    end
    kernel = 'Gaussian';
    sigma = 1;
elseif nargin < 4
    sigma = 1;
end

if strcmp(kernel,'Linear')
    K = X*Y';
elseif strcmp(kernel,'Gaussian')
    D = sum(X.*X,2)+sum(Y.*Y,2)'-2*X*Y';
    K = exp(-D/(2*(sigma^2)));
elseif strcmp(kernel,'Poly')
    D = X*Y';
    K = (D+1).^sigma;
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