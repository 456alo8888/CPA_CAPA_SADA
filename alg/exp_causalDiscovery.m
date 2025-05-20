clc;clear;addpath('.\dataset');addpath(genpath('.\SADA'));
data_url = {'Alarm','andes','Pigs','Link','munin2'};
dataID = 3; % choose graph
maxCset = 5;
skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
nsamples = 1000; % choose sample size
% Alg = {@CPA,@SADA,@CAPA,@CP};
Alg = {@CPA,@CAPA};
% Alg = {@SADA};
tic
parfor T = 1:10
    T
    rng(T)
    data = genData(skeleton,nsamples);
    tic
    r_s = ske_learn_part(data,1:size(data,2),2,@PaCoT);  % rough_skeleton
    time_rs = toc;
    % ---------------------- run with PC ----------------------------------
    [RPF1_structure,elapsed_time] = Plus_PC(Alg,data,skeleton,r_s,maxCset);
    elapsed_time = elapsed_time + time_rs;
    RPFcell_PC{T} = RPF1_structure; Timecell_PC{T} = elapsed_time;
    %------------------------- results of plus PC & Lingam ----------------
end
rpf_ave = get_Mean(RPFcell_PC)'
% errorbar_ave = get_errorBar(RPFcell_PC)'
elatime_ave = get_Mean(Timecell_PC)'
dataID
nsamples
for_tex = [rpf_ave;elatime_ave]
toc

%----------------------------- subAlg -------------------------------------
function [RPF1_structure,elapsed_time] = Plus_PC(Alg,data,skeleton,r_s,maxCset)
%---------------------------Run algorithm--------------------------
for i = 1:length(Alg)
    tic
    [cut_set,nodeA,nodeB,~] = Alg{i}(data,r_s);
    PA = unique([nodeA,cut_set]); 
    PB = unique([nodeB,cut_set]);
    %---------------------------   Run LiNGAM   ---------------------------
%     maxCset = 5; % conditional size
    struA = stru_learn_part(data,PA,maxCset,@PaCoT);
    struB = stru_learn_part(data,PB,maxCset,@PaCoT);
    time{i} = toc;
    for v = 1:size(skeleton,1)-1
        for w = i+1:size(skeleton,1)
            if struA(v,w) == 1 && struA(w,v) == 0 && struB(v,w) == 0 && struB(w,v) == 1
               struA(w,v) = 1;
               struB(v,w) = 1;
            end
        end
    end
    struC = struA.*struB;
    for p = 1:size(data,2)
        for q = 1:size(data,2)
            if struC(p,q) == 1 && struC(q,p) == 1
                if rand > 0.5
                    struC(q,p) = 0;
                else
                    struC(p,q) = 0;
                end
            end
        end
    end
    struC(nodeA,nodeB) = 0;struC(nodeB,nodeA) = 0;
    score{i} = [getRPF_stru(struC,skeleton),get_SHD(struC,skeleton)];
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
