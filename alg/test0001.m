clc;clear;%rng('default');%parpool(8)
addpath('.\dataset');addpath(genpath('.\SADA'));
data_url = {'Asia','mildew','water','Alarm','barley','hepar2','hailfinder','win95ptsbn','andes','Pigs','Link','munin2'};
dataID = 4; 
% 1000: 4 5 9
skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
nsamples = 1000;
Alg = {@Base};
% load dataAlarm.mat
data = genData(skeleton,nsamples);

% data = dataAlarm;
% struC = double(py.golem.golem_main(data));
% getRPF_stru(struC,skeleton)
% return




r_s = ske_learn_part(data,1:size(data,2),2,@PaCoT);  % rough_skeleton
[RPF1_stru1,~,ela_time1] = Plus_golem(Alg,data,skeleton,r_s);
RPF1_stru1
ela_time1
function [RPF1_structure,cut_size,elapsed_time] = Plus_golem(Alg,data,skeleton,r_s)
%---------------------------Run algorithm--------------------------
for i = 1:length(Alg)
    tmarklingam = clock;
    [cut_set,nodeA,nodeB,~] = Alg{i}(data,r_s);
    PA = unique([nodeA,cut_set]); PB = unique([nodeB,cut_set]);
    sizeG{i} = [size(skeleton,1),length(PA),length(PB),length(cut_set)];
    ratio{i} = max(length(PA),length(PB))/size(skeleton,1);
    %---------------------------   Run LiNGAM   ---------------------------
    struA = golem_part(data,PA);
    struB = golem_part(data,PB);
    struC = struA + struB;
    struC(struC==2)=1;
    for p = 1:size(data,2)
        for q = 1:size(data,2)
            if struC(p,q) == 1 && struC(q,p) == 1 
                struC(q,p) = 0;
            end
        end
    end
    struC
    skeleton
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
