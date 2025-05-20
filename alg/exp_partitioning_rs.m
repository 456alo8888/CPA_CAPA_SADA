clc;clear;addpath('.\dataset');addpath(genpath('.\SADA'));data_url = {'Alarm','andes','diabetes','Link'};
dataID = 2; % choose graph
% load Alarm_rs_500
load Andes_rs_500
% load Diabetes_rs_500
% load Link_rs_500
skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
%----------------------------- ground truth -------------------------------------
[cut_set,nodeA,nodeB,~,~] = CPSC_skeleton(skeleton);
v1 = unique([cut_set,nodeA]);v2 = unique([cut_set,nodeB]);
%----------------------------- ground truth -------------------------------------
nsamples = 1000; % choose sample size
% Alg = {@CPA,@CP,@CAPA,@SADArs,@Rando};
Alg = {@CPA};
parfor T = 1:6
    T
    rng(T)
    data = genData(skeleton,nsamples);
    tic
%     r_s = save_rs{T};  % rough_skeleton
    r_s = ske_learn_part(data,1:size(data,2),2,@PaCoT);  % rough_skeleton
    time_rs = toc;
    [rpf,elapsed_time] = get_partitioning(Alg,data,r_s,v1,v2);
%     elapsed_time = elapsed_time + [time_rs,time_rs,time_rs,0]';
    RPFcell_PC{T} = rpf;
    Timecell_PC{T} = elapsed_time;
    %------------------------- results of plus PC & Lingam ----------------
end
rpf_ave = get_Mean(RPFcell_PC)'
elatime_ave = get_Mean(Timecell_PC)'
%
%
%----------------------------- subAlg -------------------------------------
function [rpf,elapsed_time] = get_partitioning(Alg,data,r_s,v1,v2)
for i = 1:length(Alg)
    tic
    [cut_set,nodeA,nodeB,~] = Alg{i}(data,r_s);
    time{i} = toc;
    A = unique([nodeA,cut_set]);
    B = unique([nodeB,cut_set]);
    %------------------------- get rpf1------------------------------------
    rpf1 = get_rpf_partitiong(v1,v2,A,B);
    rpf2 = get_rpf_partitiong(v2,v1,A,B);
    if rpf1 > rpf2
        rpf_cell{i} = [rpf1,max(length(A),length(B))/size(data,2)];
    else
        rpf_cell{i} = [rpf2,max(length(A),length(B))/size(data,2)];
    end
    elapsed_time = 0;
end
rpf = [];
elapsed_time = [];
for k = 1:i
    rpf = [rpf;rpf_cell{k}];
    elapsed_time = [elapsed_time;time{k}];
end
end

function rpf = get_rpf_partitiong(v1,v2,A,B)
% A: ground truth; B: predicted
r = (length(intersect(v1,A))+length(intersect(v2,B)))/(length(v1)+length(v2));
p = (length(intersect(v1,A))+length(intersect(v2,B)))/(length(A)+length(B));
if r == 0 && p ==0
    f1 = 0;
else
    f1 = 2*r*p/(r+p);
end
rpf = [r,p,f1];
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
