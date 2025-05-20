clc;clear;addpath('.\alg');addpath('.\dataset');addpath(genpath('.\SADA'));
n = 50; % the number of nodes
pM = [1 1;1 2;1 3;2 1;2 2;2 3;3 3;4 4;5 5]; % different in-degrees
Alg = {@CPA}; % chosen algorithm
nsamples = 100; % different sample size
maxCset = 3;  % max conditional size for PC algorithm
for u = 1:size(pM)
    p = pM(u,:)
    parfor T = 1:10  % generate i.i.d. structure & data for different tests
        stru_GT = zeros(n,n);stru_GT(1,2) = 1;stru_GT(2,3) = 1;stru_GT(3,4) = 1;stru_GT(4,5) = 1;
        for m = 6:n
            r = randperm(m-1);
            if rand > 0.2
                stru_GT(r(1:p(1)),m) = 1;
            else
                stru_GT(r(1:p(2)),m) = 1;
            end
        end
        data = genData(stru_GT,nsamples);  % generate data
        % ---------------------- run PC ----------------------------------
        tic;[~,stru_PC] = PC_part(data,1:size(data,2),maxCset,@PaCoT);
        stru_PC = randori(stru_PC); % randomly orient the remaining undirected edge
        cell_PC{T} = [[getRPF_stru(stru_PC,stru_GT),get_SHD(stru_PC,stru_GT)],toc];
        % ---------------------- run CPA+PC ----------------------------------
        tic;[r_s,~] = PC_part(data,1:size(data,2),2,@PaCoT);trs = toc; % get superstructure
        tic;cell_CPA{T} = [Plus_PC(Alg(1),data,stru_GT,r_s,maxCset),toc+trs]; 
    end
    printS = [get_Mean(cell_PC)',get_Mean(cell_CPA)']
end
