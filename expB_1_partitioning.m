clc;clear;addpath('.\dataset');addpath(genpath('.\SADA'));;addpath(genpath('.\alg'));
data_url = {'alarm','andes','diabetes','Link'}; % four grpahs avaliable at https://www.bnlearn.com/bnrepository/
dataID = 1; % choose graph
stru_GT = readRnet(['.\dataset\',data_url{dataID},'.net']);stru_GT = sortskeleton(stru_GT); % read and get ground truth structure
%----------------------------choose GT
load cutGT_Alarm
% load cutGT_Andes
% load cutGT_Diabetes 
% load cutGT_Link
v1 = unique([cutGT{1},cutGT{2}]);v2 = unique([cutGT{1},cutGT{3}]);
nsamples = 500; % choose sample size
Alg = {@CPA,@SADA,@CAPA,@CP,@Rando}; % chosen algorithm
tic
parfor T = 1:10 % the number of tests
    data = genData(stru_GT,nsamples); % generate i.i.d. data for different tests
    tic;[r_s,~] = PC_part(data,1:size(data,2),2,@PaCoT);time_rs = toc;  % get superstructure
    [rpf,elapsed_time] = get_partitioning(Alg,data,r_s,v1,v2);
    elapsed_time = elapsed_time + time_rs;
    RPFcell_PC{T} = rpf;
    Timecell_PC{T} = elapsed_time;
end
rpfshd_ave = get_Mean(RPFcell_PC)'
errorbar_ave = get_errorBar(RPFcell_PC)'
elatime_ave = get_Mean(Timecell_PC)'
toc
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
    if min(length(A),length(B)) == 0
        R = 1;
    else
        R = length(cut_set)/min(length(A),length(B));
    end
    if rpf1 > rpf2
        rpf_cell{i} = [rpf1,R];
    else
        rpf_cell{i} = [rpf2,R];
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

function errorBar = get_errorBar(d)
n = size(d,2);
e = d{1};
errorBar = zeros(size(e,1),size(e,2));
for i = 1:size(e,1)
    for j = 1:size(e,2)
        temp = [];
        for k = 1:n
            s = d{k};
            temp = [temp,s(i,j)];
        end
        errorBar(i,j) = std(temp)/n^(1/2); % Standard Error
    end
end
end
