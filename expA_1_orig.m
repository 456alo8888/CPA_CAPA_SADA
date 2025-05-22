clc;clear;addpath('.\alg');addpath('.\dataset');addpath(genpath('.\SADA'));
data_url = {andes','diabetes','link', 'munin'}; % four grpahs avaliable at https://www.bnlearn.com/bnrepository/
dataID = 1; % choose graph
maxCset = 3; % max conditional size for PC algorithm
stru_GT = readRnet(['.\dataset\',data_url{dataID},'.net']);stru_GT = sortskeleton(stru_GT); % read and get ground truth structure
samplesize = [100 300 500 1000 3000]; % different sample size
Alg = {@CPA} % chosen algorithm
for u = 1:length(samplesize)
    nsamples = samplesize(u)
    parfor T = 1:10 % the number of tests
        data = genData(stru_GT,nsamples); % generate i.i.d. data for different tests
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