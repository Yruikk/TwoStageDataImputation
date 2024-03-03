function [alpha,bias,Ktr,K_Delta,E] = Our_Algo(X_miss,maskInv,y,pms)
%OUR_ALGO 此处显示有关此函数的摘要
%   此处显示详细说明
N = size(y,1);
Y = diag(y);
[Kk,~] = kerMatForMiss(X_miss,X_miss,maskInv,'Gaussian',pms.sigma);
% "lowBound" and "upBound" correspond to "Bl" and "Bu" in the paper, respectively.
[lowBound,upBound]=cal_bound(X_miss,X_miss,maskInv,pms.sigma); 

tmax = 10000;
K_Delta = ones(N,N);
E = ones(N,N);

alpha = pms.C/2*ones(N,1);
diff = [];
alphavalue = [];

beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;
m_alpha = zeros(N,1);
v_alpha = zeros(N,1);
m_Kd = zeros(N,N);
v_Kd = zeros(N,N);
m_E = zeros(N,N);
v_E = zeros(N,N);

non_converged = 1;
iter = 0; lr_ini = 0.1;
while (non_converged) && iter <tmax
    lr = lr_ini/sqrt(iter+1);

    % Update K_Delta.
    grad = 2*pms.eta*(K_Delta-ones(N,N))-1/2*diag(alpha'*Y)*(Kk.*E)*diag(alpha'*Y);
    m_Kd = beta1*m_Kd+(1-beta1)*grad;
    v_Kd = beta2*v_Kd+(1-beta2)*(grad.^2);
    m_hat = m_Kd/(1-beta1^(iter+1));
    v_hat = v_Kd/(1-beta2^(iter+1));

    update = lr*m_hat./(sqrt(v_hat)+epsilon);
    K_Delta = K_Delta-update;
    dd = diag(K_Delta);
    K_Delta = K_Delta/mean(dd);
    K_Delta = max(K_Delta, lowBound);
    K_Delta = min(K_Delta, upBound);
    Ktr = check_K_PSD(Kk.*K_Delta.*E,epsilon);
    K_Delta = Ktr./Kk./E;

    % Update E.
    grad = 1/2*diag(alpha'*Y)*(Kk.*K_Delta)*diag(alpha'*Y);
    m_E = beta1*m_E+(1-beta1)*grad;
    v_E = beta2*v_E+(1-beta2)*(grad.^2);
    m_hat = m_E/(1-beta1^(iter+1));
    v_hat = v_E/(1-beta2^(iter+1));

    update = lr*m_hat./(sqrt(v_hat)+epsilon);
    E = E-update;
    norm_E_ones = norm(E-ones(N,N), 'fro');
    if norm_E_ones > pms.r
        E = ones(N,N)+pms.r*(E-ones(N,N))/norm_E_ones;
    end
    Ktr = check_K_PSD(Kk.*K_Delta.*E,epsilon);
    E = Ktr./Kk./K_Delta;

    % Update alpha.
    grad = Y*Ktr*Y*alpha-ones(N,1);
    m_alpha = beta1*m_alpha+(1-beta1)*grad;
    v_alpha = beta2*v_alpha+(1-beta2)*(grad.^2);
    m_hat = m_alpha/(1-beta1^(iter+1));
    v_hat = v_alpha/(1-beta2^(iter+1));

    update = lr*m_hat./(sqrt(v_hat)+epsilon);
    alpha = projectionalgo(alpha-update,y,pms.C*ones(N,1),0);

    alphavalue = [alphavalue,alpha];
    iter = iter+1;
    if iter>2
        diff = [diff, norm(alphavalue(:,end)-alphavalue(:,end-1))];
        if  (diff(end)<1e-4)
            non_converged = 0;
        end
    end 
end
[u,d] = eig(Ktr);
dd = max(diag(d),1e-8);
Ktr = u*diag(dd)*u';
% Ktr(logical(eye(N))) = 1; % Set diagonal elements to 1
H = diag(y)*Ktr*diag(y);
f = -ones(N,1);
lb = zeros(N,1);
ub = pms.C*ones(N,1);
A = []; b = []; Aeq = y'; beq = 0;
alpha_new = quadprog(H,f,A,b,Aeq,beq,lb,ub);
if ~isempty(alpha_new)
    alpha_new(alpha_new<1e-5) = 0;
    alpha = alpha_new;
end

% Compute bias.
ind = find(alpha>0);
preds = (alpha.*y)'*Ktr(:,ind);
bias = mean(y(ind)'-preds);

Ktr = Kk.*K_Delta;
end

function nor_K = check_K_PSD(K,epsilon)
    nor_K = (K+K')/2;
    [V,D] = eig(nor_K);
    D = max(D, 0);
    nor_K = V*D*V';
    dd = diag(nor_K);
    nor_K = nor_K/mean(dd);
    nor_K(logical(eye(size(K)))) = 1; % Set diagonal elements to 1
    nor_K(nor_K<epsilon) = epsilon;
end
% function nor_K = check_K_PSD(K,epsilon)
%     nor_K = (K+K')/2;
%     [U,S,V] = svd(nor_K);
%     S = max(S, 0);
%     nor_K = U*S*V';
%     dd = diag(nor_K);
%     nor_K = nor_K/mean(dd);
%     nor_K(logical(eye(size(K)))) = 1; % Set diagonal elements to 1
%     nor_K(nor_K<epsilon) = epsilon;
% end

function [resb,lambda]=projectionalgo(x,a,c,t)
% Computes the Euclidean projection of x on the set
% 0 <= x_i <= c_i ,  a'*x = t

% Prune out indices for which a_i=0
za=find(abs(a)<1e-10);nza=find(abs(a)>1e-10);
resb(za)=min([c(za)';max([zeros(size(c(za)))';x(za)'])])';
x=x(nza);a=a(nza);c=c(nza);

n=size(x,1);
e=ones(size(x));

anf=[a;+inf];
[veca,inda]=sort([-2*x./a;+inf]);
[vecb,indb]=sort([2*(c-x)./a;+inf]);
asa=anf(inda);asb=anf(indb);

[lastpoint,type]=min([veca(1),vecb(1)]);
grad=t-c(a<0)'*a(a<0);
gslope=0;
if type==1
    ai=asa(1);
    veca(1)=[];asa(1)=[];
    if ai>=0
        gslope=gslope-ai^2/2;
    else
        gslope=gslope+ai^2/2;
    end
else
    ai=asb(1);
    vecb(1)=[];asb(1)=[];
    if ai>=0
        gslope=gslope+ai^2/2;
    else
        gslope=gslope-ai^2/2;
    end
end


while min([veca(1),vecb(1)])<inf
    [point,type]=min([veca(1),vecb(1)]);
    interval=point-lastpoint;lastpoint=point;
    grad=grad+interval*gslope;
    if grad<0 
        break; 
    end
    if type==1
        ai=asa(1);
        veca(1)=[];asa(1)=[];
        if ai>=0
            gslope=gslope-ai^2/2;
        else
            gslope=gslope+ai^2/2;
        end
    else
        ai=asb(1);
        vecb(1)=[];asb(1)=[];
        if ai>=0
            gslope=gslope+ai^2/2;
        else
            gslope=gslope-ai^2/2;
        end
    end
end
lambda=point-grad/gslope;
res=e;
for i=1:n
    res(i)=x(i)+(lambda*a(i))/2;
    if res(i)<0 
        res(i)=0; 
    end
    if res(i)>c(i) 
        res(i)=c(i); 
    end
end
resb(nza)=res;resb=resb';
end

