function X_filled = fillData(X_miss,missMask,tildeK,sigma)
%FILLDATA 此处显示有关此函数的摘要
%   X_miss should be a N*d matrix. N (sample number) and d (dimension).
%   In missMask, 1 for missing and 0 for completed.

tildeK = tildeK-diag(diag(tildeK))+eye(size(tildeK));

missInv = ones(size(missMask))-missMask;
newK = 2*sigma*sigma*log(tildeK);
newK = real(newK);

X_init = X_miss';
[DeltaX, ~] = BCD(X_init, newK, missInv', 20000);
X_filled = X_miss + DeltaX';
end

function [Delta_X, fval] = BCD(Xk, K, missInv, max_iter)
% Xk: D*N的矩阵，为常量矩阵
% K: N*N的矩阵，为常量矩阵
% missInv: D*N的矩阵，其中缺失的元素为0，非缺失的元素为1
% max_iter: 最大迭代次数
% X: D*N的矩阵，为最优解
% fval: 最优解对应的目标函数值

[D, N] = size(Xk);
missMask = ones(size(missInv)) - missInv;
Delta_X = 0.5*missMask;
fval = inf;
iter = 0;
tol = 1e-5;

% 初始化Adam算法的参数
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;
m = zeros(D, N);
v = zeros(D, N);

init_lr = 0.1;
while iter < max_iter
    % 对每个块进行更新
    for i = 1:N
        % 固定其他块的参数，仅在当前块上优化目标函数
        x_i = Delta_X(:, i);
        y_i = Xk(:, i);
        x_j = Delta_X(:, [1:i-1, i+1:N]);
        y_j = Xk(:, [1:i-1, i+1:N]);
        K_i = K(i, [1:i-1, i+1:N]);

        % 计算梯度
        X_j = y_j + x_j;
        X_i = y_i + x_i;
        X_i = repmat(X_i,1,N-1);
        tmp1 = X_i - X_j;
        tmp2 = sum(tmp1.^2,1);
        tmp3 = tmp2 + K_i;
        grad = 4 * sum(repmat(tmp3,D,1).*tmp1,2);

        % 更新Adam算法的参数
        m(:, i) = beta1 * m(:, i) + (1 - beta1) * grad;
        v(:, i) = beta2 * v(:, i) + (1 - beta2) * (grad.^2);
        m_hat = m(:, i) / (1 - beta1^(iter+1));
        v_hat = v(:, i) / (1 - beta2^(iter+1));

        % 计算步长
        alpha = init_lr / sqrt(iter+1);

        % 投影法更新变量
        update = alpha * m_hat ./ (sqrt(v_hat) + epsilon);
        idx = missInv(:, i) == 0;
        x_i_new = x_i - idx .* update;
        x_i_all = x_i_new + y_i;
        x_i_all(x_i_all < 0) = 0;
        x_i_all(x_i_all > 1) = 1;
        x_i_new = x_i_all - y_i;

        % 更新当前块的参数
        Delta_X(:, i) = x_i_new;
    end

    % 计算目标函数值
    X_filled = (Xk + Delta_X)'; % NxD
    Dist = sum(X_filled.*X_filled,2)+sum(X_filled.*X_filled,2)'-2*(X_filled*X_filled');
    T = Dist + K;
    fval_new = norm(T, 'fro');

    % 判断是否收敛
    if abs(fval_new - fval) < tol
        %         fprintf('error = %.3e when iter = %d.\n', abs(fval_new - fval), iter);
        fval = fval_new;
        break;
    end
    fval = fval_new;
    iter = iter + 1;
end
end

