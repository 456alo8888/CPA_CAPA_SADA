clc;clear;addpath('.\alg');addpath('.\dataset');addpath(genpath('.\SADA'));
maxCset = 3;
nsamples = 100;
Alg = {@CPA};
n = 100; % #nodes
%     functionType = 1; % 1 Linear; 2 ANM; 3 PNL
%     noiseType = 1; % 1 Gaussian; 2 UNI; 3 CHI; 4 EXP; 5 BETA
for functionType = 1:3
    for noiseType = 1:5
        setting = [functionType,noiseType]
        parfor T = 1:10
            stru_GT = zeros(n,n);
            for m = 3:n
                r = randperm(m-1);
                if rand > 0.5
                    stru_GT(r(1),m) = 1;
                else
                    stru_GT(r(1:2),m) = 1;
                end
            end
            data = genData_PAMI(stru_GT,nsamples,functionType,noiseType);  % generate data of differnet kinds of distributions
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
end
