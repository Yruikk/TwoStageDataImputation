function [lowBound,upBound] = cal_bound(X,Y,missInv,sigma)
%CAL_BOUND 此处显示有关此函数的摘要
%   此处显示详细说明
missMask = ones(size(missInv))-missInv;
Nx = size(X,1);
Ny = size(Y,1);
D = size(X,2);
M = zeros(Nx,Ny);
Low = zeros(Nx,Ny);
Up = zeros(Nx,Ny);
for i=1:Nx
    for j =1:Ny
        ind_x = missMask(i,:);
        ind_y = missMask(j,:);
        xl = X(i,:);
        yl = Y(j,:);
        xu = X(i,:);
        yu = Y(j,:);
        for d = 1:D
            if (ind_x(d)==1) && (ind_y(d)==1)
                xl(d) = 1; yl(d) = 0;
                xu(d) = 1; yu(d) = 1;
            elseif (ind_x(d)==0) && (ind_y(d)==1)
                if (xl(d)>=0.5)
                    yl(d) = 0;
                else
                    yl(d) = 1;
                end
                yu(d) = xu(d);
            elseif (ind_x(d)==1) && (ind_y(d)==0)
                if (yl(d)>=0.5)
                    xl(d) = 0;
                else
                    xl(d) = 1;
                end
                xu(d) = yu(d);
            end
        end
        Low(i,j) = norm(xl-yl,2)^2;
        Up(i,j) = norm(xu-yu,2)^2;
    end
end
lowBound = exp(-1/2/sigma/sigma*Low);
upBound = exp(-1/2/sigma/sigma*Up);

end

