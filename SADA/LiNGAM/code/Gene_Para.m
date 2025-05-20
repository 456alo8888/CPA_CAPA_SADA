function [ X,B,S] = Gene_Para( dims,samples,indegree ,distribution)
X=zeros(dims,samples);
B = zeros(dims,dims);
% distribution = 'A';
switch(distribution)
    case 'A'%%%拉普拉斯分布
        fprintf('Laplace distrubution!\n' );
        for i=1:dims
            if ~isinf(indegree),
                if i<=indegree,
                    par = 1:(i-1);
                else
                    par = randperm((i-1));
                    change=round(rand(1))+1;
                    par = par(1:change);
                end
            else
                par = 1:(i-1);
            end
            if i==1
%                 S(i,:)=laprnd(0,1,1,samples);
                S(i,:)= laprnd(0,1,1,samples);%%%mu=0;均值；sigma=1;标准差，方差的开平方，m，n分别表示产生随机矩阵的行数和列数
                X(i,:)= S(i,:);
            else
                w = randn(length(par),1);%%%标准正态分布
                wfull = zeros(i-1,1); wfull(par) = w;
                B(i,par) = w';
                S(i,:)=laprnd(0,1,1,samples);
                X(i,:) = wfull'*X(1:(i-1),:)+S(i,:);
            end
        end
    case 'B'
        fprintf('Hyperbolic secant distribution!' );
        for i=1:dims
            if ~isinf(indegree),
                if i<=indegree,
                    par = 1:(i-1);
                else
                    par = randperm((i-1));
                    change=round(rand(1))+1;
                    par = par(1:change);
                end
            else
                par = 1:(i-1);
            end
            if i==1
                S(i,:)=laprnd(0,1,1,samples);
                X(i,:)= S(i,:);
            else
                w = randn(length(par),1);%%%标准正态分布
                wfull = zeros(i-1,1); wfull(par) = w;
                B(i,par) = w';
                S(i,:)=laprnd(0,1,1,samples);
                X(i,:) = wfull'*X(1:(i-1),:)+S(i,:);
            end
        end
    case 'C'
        fprintf('Logistic distribution!' );
    case 'D'%%%正态分布
        fprintf('Normal distribution!' );
        for i=1:dims
            if ~isinf(indegree),
                if i<=indegree,
                    par = 1:(i-1);
                else
                    par = randperm((i-1));
                    change=round(rand(1))+1;
                    par = par(1:change);
                end
            else
                par = 1:(i-1);
            end
            if i==1
                S(i,:)=normrnd (0,1,1, samples);%normrnd (a,b,m, n),生成均值为a,方差为b，m*n阶矩阵正态分布
                X(i,:)= S(i,:);
            else
                w = randn(length(par),1);%%%标准正态分布
                wfull = zeros(i-1,1); wfull(par) = w;
                B(i,par) = w';
                S(i,:)=normrnd (0,1,1, samples);%normrnd (a,b,m, n),生成均值为a,方差为b，m*n阶矩阵正态分布
                X(i,:) = wfull'*X(1:(i-1),:)+S(i,:);
            end
        end
    case 'E'
        fprintf('Raised cosine distribution!' );
    case 'F'%%%均匀分布
        fprintf('Uniform distribution!' );
        for i=1:dims
            if ~isinf(indegree),
                if i<=indegree,
                    par = 1:(i-1);
                else
                    par = randperm((i-1));
                    change=round(rand(1))+1;
                    par = par(1:change);
                end
            else
                par = 1:(i-1);
            end
            if i==1
                S(i,:)=unifrnd (-1,1,1, samples);%unifrnd (a,b,m, n),m*n阶［a，b］均匀分布U（a，b），由于其方差为（b-a）^2/12,故b=sqrt(3),a=-sqrt(3),方差等于1;
                X(i,:)= S(i,:);
            else
                w = randn(length(par),1);%%%标准正态分布
                wfull = zeros(i-1,1); wfull(par) = w;
                B(i,par) = w';
                S(i,:)=unifrnd (-1,1,1, samples);%unifrnd (a,b,m, n),m*n阶［a，b］均匀分布U（a，b）
                X(i,:) = wfull'*X(1:(i-1),:)+S(i,:);
            end
        end  
    otherwise
        fprintf('Wigner semicircle distribution!' );
        
end

