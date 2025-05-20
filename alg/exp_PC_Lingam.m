clc;clear;rng('default');
addpath('.\dataset');addpath(genpath('.\SADA'));
data_url = {'Child','mildew','water','Alarm','barley','hepar2','hailfinder','win95ptsbn','andes','Pigs','Link','munin2'};
dataID = 4; 
% 1000: 4 5 7 9
skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
nsamples = 1000;
% data = genData(skeleton,nsamples);
Alg = {@Base,@CPD,@CP,@CAPA,@SADA};
for T = 1:4
    data = genData(skeleton,nsamples);
%     idx = randperm(size(skeleton,1));
%     temp1 = skeleton(:,idx);
%     skeleton = temp1(idx,:);
%     data = data(:,idx);

    r_s = ske_learn_part(data,1:size(data,2),2,@PaCoT);  % rough_skeleton
    % ---------------------- run with PC ----------------------------------
    [RPF1_structure,cut_size,elapsed_time] = Plus_PC(Alg,data,skeleton,r_s);
    RPFcell_PC{T} = RPF1_structure; Sizecell_PC{T} = cut_size;Timecell_PC{T} = elapsed_time;
    % ---------------------- run with Lingam ------------------------------
    [RPF1_structure,cut_size,elapsed_time] = Plus_Lingam(Alg,data,skeleton,r_s);
    RPFcell_Lingam{T} = RPF1_structure; Sizecell_Lingam{T} = cut_size;Timecell_Lingam{T} = elapsed_time;
    %------------------------- results of plus PC & Lingam ----------------
    iter = T
    rpf_shd = [get_Mean(RPFcell_PC)',get_Mean(RPFcell_Lingam)']
    rpf_shd_errorBar = [get_errorBar(RPFcell_PC)',get_errorBar(RPFcell_Lingam)']
    sizeCut = [get_Mean(Sizecell_PC),get_Mean(Sizecell_Lingam)];
    TimeAlg = [get_Mean(Timecell_PC)',get_Mean(Timecell_Lingam)']
end

function [RPF1_structure,cut_size,elapsed_time] = Plus_PC(Alg,data,skeleton,r_s)
%---------------------------Run algorithm--------------------------
for i = 1:length(Alg)
    tic
    [cut_set,nodeA,nodeB,~] = Alg{i}(data,r_s);
    PA = unique([nodeA,cut_set]); PB = unique([nodeB,cut_set]);
    sizeG{i} = [size(skeleton,1),length(PA),length(PB),length(cut_set)];
    ratio{i} = max(length(PA),length(PB))/size(skeleton,1);
    %---------------------------   Run LiNGAM   ---------------------------
    maxCset = 3; % conditional size
    struA = stru_learn_part(data,PA,maxCset,@PaCoT);
    struB = stru_learn_part(data,PB,maxCset,@PaCoT);
    struC = struA.*struB;
    for p = 1:size(data,2)
        for q = 1:size(data,2)
            if struC(p,q) == 1 && struC(q,p) == 1
                struC(q,p) = 0;
            end
        end
    end
    struC(nodeA,nodeB) = 0;struC(nodeB,nodeA) = 0;
    score{i} = [getRPF_stru(struC,skeleton), get_SHD(struC,skeleton)];
    time{i} = toc;
end
%----------------------------------results---------------------------------
RPF1_structure = [];
cut_size = [];
elapsed_time = [];
for k = 1:i
    RPF1_structure = [RPF1_structure;score{k}];
    cut_size = [cut_size;sizeG{k}];
    elapsed_time = [elapsed_time;time{k}];
end
end

function [RPF1_structure,cut_size,elapsed_time] = Plus_Lingam(Alg,data,skeleton,r_s)
%---------------------------Run algorithm--------------------------
for i = 1:length(Alg)
    tic
    [cut_set,nodeA,nodeB,~] = Alg{i}(data,r_s);
    PA = unique([nodeA,cut_set]); PB = unique([nodeB,cut_set]);
    sizeG{i} = [size(skeleton,1),length(PA),length(PB),length(cut_set)];
    ratio{i} = max(length(PA),length(PB))/size(skeleton,1);
    %---------------------------   Run LiNGAM   ---------------------------
    struA = LiNGAM_part(data,PA);
    struB = LiNGAM_part(data,PB);
    struC = struA + struB;
    struC(struC==2)=1;
    for p = 1:size(data,2)
        for q = 1:size(data,2)
            if struC(p,q) == 1 && struC(q,p) == 1 
                struC(q,p) = 0;
            end
        end
    end
    score{i} = [getRPF_stru(struC,skeleton), get_SHD(struC,skeleton)];
    time{i} = toc;
end
%----------------------------------results---------------------------------
RPF1_structure = [];
cut_size = [];
elapsed_time = [];
for k = 1:i
    RPF1_structure = [RPF1_structure;score{k}];
    cut_size = [cut_size;sizeG{k}];
    elapsed_time = [elapsed_time;time{k}];
end
end

function errorMean = get_Mean(d)
n = size(d,2);
e = d{1};
% errorBar = zeros(size(e,1),size(e,2));
errorMean = zeros(size(e,1),size(e,2));
for i = 1:size(e,1)
    for j = 1:size(e,2)
        temp = [];
        for k = 1:n
            s = d{k};
            temp = [temp,s(i,j)];
        end
%         errorBar(i,j) = std(temp);
        errorMean(i,j) = mean(temp);
    end
end
end

function errorBar = get_errorBar(d)
n = size(d,2);
e = d{1};
errorBar = zeros(size(e,1),size(e,2));
% errorMean = zeros(size(e,1),size(e,2));
for i = 1:size(e,1)
    for j = 1:size(e,2)
        temp = [];
        for k = 1:n
            s = d{k};
            temp = [temp,s(i,j)];
        end
        errorBar(i,j) = std(temp);
%         errorMean(i,j) = mean(temp);
    end
end
end
