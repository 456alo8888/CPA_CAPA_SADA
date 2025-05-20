clc;clear;addpath('.\dataset');addpath(genpath('.\SADA'));data_url = {'Alarm','andes','diabetes','Link'};
dataID = 1; % choose graph
% load Alarm_rs_500
% load Andes_rs_500
% load Diabetes_rs_500
% load Link_rs_500
skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
nsamples = 500; % choose sample size
Alg = {@CPA,@CP,@CAPA,@SADArs};
tic
parfor T = 1:10
    T
    rng(T)
    data = genData(skeleton,nsamples);
    r_s = ske_learn_part(data,1:size(data,2),2,@PaCoT);  % rough_skeleton
%     r_s = save_rs{T};  % rough_skeleton
    % ---------------------- run with PC ----------------------------------
    [RPF1_structure,elapsed_time] = Plus_PC(Alg,data,skeleton,r_s);
    RPFcell_PC{T} = RPF1_structure; 
    Timecell_PC{T} = elapsed_time;
    %------------------------- results of plus PC & Lingam ----------------
end
rpf = get_Mean(RPFcell_PC)'
elaTime = get_Mean(Timecell_PC)'
toc

%----------------------------- subAlg -------------------------------------
function [RPF1_structure,elapsed_time] = Plus_PC(Alg,data,skeleton,r_s)
%---------------------------Run algorithm--------------------------
for i = 1:length(Alg)
    tic
    [cut_set,nodeA,nodeB,~] = Alg{i}(data,r_s);
    time{i} = toc;
    PA = unique([nodeA,cut_set]); PB = unique([nodeB,cut_set]);
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
    score{i} = [getRPF_stru(struC,skeleton)];
end
%----------------------------------results---------------------------------
RPF1_structure = [];
elapsed_time = [];
for k = 1:i
    RPF1_structure = [RPF1_structure;score{k}];
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
