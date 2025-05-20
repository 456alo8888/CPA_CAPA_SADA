clc;clear;addpath('.\dataset');addpath(genpath('.\SADA'));tic
data_url = {'Alarm','andes','Pigs','Link','munin4'};
dataID = 2; % choose graph
skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
temp = [];
for i = 1:size(skeleton,1)
    if sum(skeleton(i,:)) == 0 && sum(skeleton(:,i)) == 0
        temp = [temp,i];
    end
end
skeleton(:,temp) = [];skeleton(temp,:) = [];
%----------------------------- CPA-------------------------------------
% [cut_set,nodeA,nodeB] = CPA_skeleton(skeleton);v1 = unique([cut_set,nodeA]);v2 = unique([cut_set,nodeB]);
% w = plot(graph(skeleton+skeleton','upper'));
% highlight(w,nodeB,'NodeColor','k','Marker','o');
% highlight(w,nodeA,'NodeColor','b','Marker','^');
% highlight(w,cut_set,'NodeColor','r','Marker','d');
%----------------------------- ground truth-------------------------------------
cut_set = edge_cutting(skeleton);
G = graph(skeleton+skeleton','upper');
subplot(1,2,1)
w = plot(G);
highlight(w,cut_set,'EdgeColor','r');
[MN,nodeA,nodeB] = CPA_skeleton(skeleton);
CR = max(length(unique([MN,nodeA])),length(unique([MN,nodeB])));
MN
% return
%
G2 = G;
n = length(cut_set);
tempN = [];
for i = 1:n
    tempN = [tempN,find(skeleton(cut_set(i),:)==1),find(skeleton(:,cut_set(i))==1)'];
end
N = unique(tempN)
%
iterNum = floor(length(MN)/2)
%for k = 1:floor(length(MN)/2)
for k = 10
    k
    M = nchoosek(N,k);
    s = size(M,1)
    for i = 1:s
        s
        %
        G0 = rmnode(G2,M(i,:));
        bins = conncomp(G0);
        if length(unique(bins)) == 1
            continue
        else
            cut_size0 = [];
            for j = 1:length(unique(bins))
                cut_size0 = [cut_size0,length(find(bins==j))];
            end
            if max(cut_size0)/size(skeleton,1) > 4/5
                continue
            end
        end
        %   
        cut_size = [];
        pcset = getPC(M(i,:),skeleton);
        G = rmnode(G2,pcset);
        bins = conncomp(G);
        lb = length(unique(bins));
        if lb > 1
            for j = 1:lb
                cut_size = [cut_size,length(find(bins==j))];
            end
            if length(max(cut_size))+ length(pcset) <= CR
                Mset = M(i,:)
                Cset = pcset
                CRgt = length(max(cut_size))+ length(pcset)
                subplot(1,2,2)
                w = plot(G);
                return
            end
        end
    end
end
count
toc

function [pcset] = getPC(M,skeleton)
n = length(M);
temp = [];
for i = 1:n
    temp = [temp,find(skeleton(M(i),:)==1)];
    temp = [temp,find(skeleton(:,M(i))==1)'];
end
pcset = unique([temp,M]);
end