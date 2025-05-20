clc;clear;
addpath('.\dataset');addpath(genpath('.\SADA'));
% data_url = {'child','mildew','water','Alarm','barley','hepar2','hailfinder','win95ptsbn','andes','Pigs','Link','munin2'};
data_url = {'Alarm'};
for dt = 1%length(data_url)
    dataID = dt
    skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
    nsamples = 500;
    sens_ratio = {0 0.1 0.2 0.3 0.4 0.5};
    Times = 100
    score_cell = cell(1,Times);
    Time_cell = cell(1,Times);
    tmark = clock; 
    parfor T = 1:Times
        rng(T); % by default
%         skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
        data = genData(skeleton,nsamples);
%         idx = randperm(size(skeleton,1));temp1 = skeleton(:,idx);skeleton = temp1(idx,:);data = data(:,idx);
%         r_s = skeleton + skeleton';  % rough_skeleton
        r_s = get_rough_graph(data,2,@PaCoT) + skeleton + skeleton';  % rough_skeleton
        r_s(r_s==2)=1;
        % ---------------------- run with PC ----------------------------------
        [RPF1_stru1,~,ela_time1] = Plus_PC(@CPD,data,skeleton,r_s,sens_ratio);
        % ---------------------- run with Lingam ------------------------------
        [RPF1_stru2,~,ela_time2] = Plus_Lingam(@CPD,data,skeleton,r_s,sens_ratio);
        % ---------------------- results --------------------------------------
        score_cell{T} = [RPF1_stru1;RPF1_stru2];
        Time_cell{T} = [ela_time1',ela_time2'];
%         score_cell{T} = [RPF1_stru1];
%         Time_cell{T} = [ela_time1'];
    end
    %------------------------- Average results of plus PC & Lingam --------
    score_Ave = vpa([get_Mean(score_cell)]',4)
    errorBar =  vpa([get_errorBar(score_cell)]',4)
    Time_Ave = [get_Mean(Time_cell)]
    elaptime = etime(clock,tmark)
end

function r_s = get_ratio_r_s(r_s,sens_ratio,skeleton)
skeleton = skeleton + skeleton';
n = sum(sum(skeleton));
cut_num = floor(n*sens_ratio);
temp = find(skeleton==1);
idx = randperm(size(find(skeleton==1),1));
r_s(temp(idx(1:cut_num))) = 0;
end

function [RPF1_structure,cut_size,elapsed_time] = Plus_PC(Alg,data,skeleton,r_s,sens_ratio)
%---------------------------Run algorithm--------------------------
for i = 1:length(sens_ratio)
    tmarkpc = clock;
    if sens_ratio{i} == 1
        temprp = randperm(37);
        PA = temprp(1:18);
        PB = temprp(19:37);
    else
        r_s_cut = get_ratio_r_s(r_s,sens_ratio{i},skeleton);
        [cut_set,nodeA,nodeB,~] = Alg(data,r_s_cut);
        PA = unique([nodeA,cut_set]); PB = unique([nodeB,cut_set]);
    end
    sizeG{i} = [size(skeleton,1),length(PA),length(PB),length(cut_set)];
    ratio{i} = max(length(PA),length(PB))/size(skeleton,1);
    %---------------------------   Run LiNGAM   ---------------------------
    maxCset = 5; % conditional size
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
    time{i} = etime(clock,tmarkpc);
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

function [RPF1_structure,cut_size,elapsed_time] = Plus_Lingam(Alg,data,skeleton,r_s,sens_ratio)
%---------------------------Run algorithm--------------------------
for i = 1:length(sens_ratio)
    tmarklingam = clock;
    if sens_ratio{i} == 1
        temprp = randperm(37);
        PA = temprp(1:18);
        PB = temprp(19:37);
    else
        r_s_cut = get_ratio_r_s(r_s,sens_ratio{i},skeleton);
        [cut_set,nodeA,nodeB,~] = Alg(data,r_s_cut);
        PA = unique([nodeA,cut_set]);PB = unique([nodeB,cut_set]);
    end
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
    time{i} = etime(clock,tmarklingam);
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
