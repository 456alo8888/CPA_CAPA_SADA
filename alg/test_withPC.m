clc;clear;addpath('.\dataset');addpath(genpath('.\SADA'));data_url = {'Alarm','andes','diabetes','Link'};
dataID = 1;
skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
nsamples = 500; 
tic
for T = 1
    rng(2)
    data = genData(skeleton,nsamples);
    r_s = ske_learn_part(data,1:size(data,2),2,@PaCoT);
    score1 = getRPF_ske(r_s,skeleton)
    toc
    score2 = Plus_PC(@CPA,data,skeleton,r_s)
    toc
end

function [RPF1_structure] = Plus_PC(cpa,data,skeleton,r_s)
Alg{1} = @CPA;
%---------------------------Run algorithm--------------------------
for i = 1:length(Alg)
    tic
    [cut_set,nodeA,nodeB,~] = Alg{i}(data,r_s);
    time{i} = toc;
    PA = unique([nodeA,cut_set]);
    PB = unique([nodeB,cut_set]);
    cut_ratio = max(length(PA),length(PB))/size(skeleton,1)
    %---------------------------   Run LiNGAM   ---------------------------
    maxCset = 3; % conditional size
    struA = ske_learn_part(data,PA,maxCset,@PaCoT);
    struB = ske_learn_part(data,PB,maxCset,@PaCoT);
    struC = struA.*struB;
%     struC = ones(size(skeleton,1),size(skeleton,1));
    struC(nodeA,nodeB) = 0;
    struC(nodeB,nodeA) = 0;
    score{i} = [getRPF_ske(struC,skeleton)];
end
%----------------------------------results---------------------------------
RPF1_structure = [];
elapsed_time = [];
for k = 1:i
    RPF1_structure = [RPF1_structure;score{k}];
    elapsed_time = [elapsed_time;time{k}];
end
end