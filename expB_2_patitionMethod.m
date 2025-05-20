clc;clear;addpath('.\alg');addpath('.\dataset');addpath(genpath('.\SADA'));
data_url = {'alarm','andes','diabetes','Link'}; % four grpahs avaliable at https://www.bnlearn.com/bnrepository/
dataID = 1; % choose graph
maxCset = 3; % max conditional size for PC algorithm
stru_GT = readRnet(['.\dataset\',data_url{dataID},'.net']);stru_GT = sortskeleton(stru_GT); % read and get ground truth structure
samplesize = [100 300 500]; % different sample size
Alg = {@CPA,@SADA,@CAPA,@CP,@Rando}  % chosen algorithm
for u = 1:length(samplesize)
    nsamples = samplesize(u)
    parfor T = 1:10
        data = genData(stru_GT,nsamples); % generate i.i.d. data for different tests
        % ---------------------- rough_skeleton----------------------------
        tic;[r_s,~] = PC_part(data,1:size(data,2),2,@PaCoT);trs = toc;
        % ---------------------- run CPA ----------------------------------
        tic;cell_CPA{T} = [Plus_PC(Alg(1),data,stru_GT,r_s,maxCset),toc+trs];
        % ---------------------- run SADA ----------------------------------
        tic;cell_SADA{T} = [Plus_PC(Alg(2),data,stru_GT,r_s,maxCset),toc+trs];
        % ---------------------- run CAPA ----------------------------------
        tic;cell_CAPA{T} = [Plus_PC(Alg(3),data,stru_GT,r_s,maxCset),toc+trs];
        % ---------------------- run CP ----------------------------------
        tic;cell_CP{T} = [Plus_PC(Alg(4),data,stru_GT,r_s,maxCset),toc+trs];
        % ---------------------- run Rando ----------------------------------
        tic;cell_Rando{T} = [Plus_PC(Alg(5),data,stru_GT,r_s,maxCset),toc];
    end
    printResult = [get_Mean(cell_CPA)',get_Mean(cell_SADA)',get_Mean(cell_CAPA)',get_Mean(cell_CP)',get_Mean(cell_Rando)']
end
