clc;clear;% parpool local% rand('seed',0)
addpath('.\dataset');addpath(genpath('.\SADA'));
data_url = {'child','mildew','water','Alarm','barley','hepar2','hailfinder','win95ptsbn','andes','Pigs','Link','munin2'};
dataID = 4; 
% 500 : 
% 1000: 4=+ 5++
skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
nsamples = 1000;
for T = 1:1
    data = genData(skeleton,nsamples);
    r_s = ske_learn_part(data,1:size(data,2),2,@PaCoT);  % rough_skeleton
    %---------------------------Run algorithm--------------------------
    Alg = {@Base,@CPA,@CAPA,@SADA,@CP};
%     Alg = {@Base};
    for i = 1:length(Alg)
        tic
        [cut_set,nodeA,nodeB,pruned_node] = Alg{i}(data,r_s);
        PA = unique([nodeA,cut_set]); PB = unique([nodeB,cut_set]);
        sizeG{i} = [size(skeleton,1),length(PA),length(PB),length(cut_set)];
        ratio{i} = max(length(PA),length(PB))/size(skeleton,1);
        %---------------------------   Run LiNGAM   ---------------------------
        maxCset = 3; % conditional size
        struA = stru_learn_part(data,PA,maxCset,@PaCoT);
        struB = stru_learn_part(data,PB,maxCset,@PaCoT);
        struC = struA.*struB;
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
    RPFcell{T} = RPF1_structure;
    Sizecell{T} = cut_size;
    Timecell{T} = elapsed_time;
    iter = T
    rpf_shd = calError(RPFcell)
    sizeCut = calError(Sizecell)
    TimeAlg = calError(Timecell)
end

% subplot(1,2,1)
% plot(digraph(skeleton))
% subplot(1,2,2)
% plot(digraph(struC))


function errorMean = calError(d)
n = size(d,2);
e = d{1};
errorBar = zeros(size(e,1),size(e,2));
errorMean = zeros(size(e,1),size(e,2));
for i = 1:size(e,1)
    for j = 1:size(e,2)
        temp = [];
        for k = 1:n
            s = d{k};
            temp = [temp,s(i,j)];
        end
        errorBar(i,j) = std(temp);
        errorMean(i,j) = mean(temp);
    end
end
end
%----------------------------------------- % plot begin


% subplot(1,2,1)
% w = plot(graph(struC,'upper'));
% highlight(w,cut_set,'NodeColor','b');
% highlight(w,nodeA,'NodeColor','m','Marker','d');
% highlight(w,nodeB,'NodeColor','r','Marker','>');
% highlight(w,pruned_node,'NodeColor','g','Marker','s');
% subplot(1,2,2)
% w = plot(digraph(skeleton));
% highlight(w,cut_set,'NodeColor','b');
% highlight(w,nodeA,'NodeColor','m','Marker','d');
% highlight(w,nodeB,'NodeColor','r','Marker','>');
% highlight(w,pruned_node,'NodeColor','g','Marker','s');

% subplot(2,2,3)
% w = plot(graph(skeletonA,'upper'));
% highlight(w,cut_set,'NodeColor','b');
% highlight(w,nodeA,'NodeColor','m','Marker','d');
% highlight(w,nodeB,'NodeColor','r','Marker','>');
% highlight(w,pruned_node,'NodeColor','g','Marker','s');
% subplot(2,2,4)
% w = plot(graph(skeletonB,'upper'));
% highlight(w,cut_set,'NodeColor','b');
% highlight(w,nodeA,'NodeColor','m','Marker','d');
% highlight(w,nodeB,'NodeColor','r','Marker','>');
% highlight(w,pruned_node,'NodeColor','g','Marker','s');
%------------------------------------------ % plot end