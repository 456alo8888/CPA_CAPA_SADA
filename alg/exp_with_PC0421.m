clc;clear;addpath('.\dataset');addpath(genpath('.\SADA'));
data_url = {'Alarm','andes','Pigs','Link','diabetes'};
dataID = 4; % choose graph
maxCset = 5;
skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
nsamples = 250; % choose sample size
Alg = {@CPA,@CP,@CAPA};
% Alg = {@Rando};
tic
parfor T = 1:6
    T
    rng(T)
    data = genData(skeleton,nsamples);
%     r_s = skeleton + skeleton';
     r_s = ske_learn_part(data,1:size(data,2),2,@PaCoT);  % rough_skeleton
    % ---------------------- run with PC ----------------------------------
    [RPF1_structure,elapsed_time] = Plus_PC(Alg,data,skeleton,r_s,maxCset);
    RPFcell_PC{T} = RPF1_structure; Timecell_PC{T} = elapsed_time;
    %------------------------- results of plus PC & Lingam ----------------
end
rpf = get_Mean(RPFcell_PC)'
elaTime = get_Mean(Timecell_PC)'
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
    score{i} = [getRPF_stru(struC,skeleton),get_SHD(struC,skeleton),getRPF_ske2(struC,skeleton)];
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
