clc;clear;addpath('.\dataset');addpath(genpath('.\SADA'));tic
data_url = {'Alarm','andes','Pigs','Link','munin4'};
dataID = 4; % choose graph
skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
temp = [];
for i = 1:size(skeleton,1)
    if sum(skeleton(i,:)) == 0 && sum(skeleton(:,i)) == 0
        temp = [temp,i];
    end
end
skeleton(:,temp) = [];skeleton(temp,:) = [];
%----------------------------- CPA-------------------------------------
[MN,nodeA,nodeB] = CPA_skeleton(skeleton);
CRgt = length(MN)/min(length(unique([MN,nodeA])),length(unique([MN,nodeB])))
%----------------------------- ground truth-------------------------------------
cut_set_cell = edge_cutting2(skeleton);
G = graph(skeleton+skeleton','upper');
subplot(1,2,1)
w = plot(G);
highlight(w,nodeA,'NodeColor','b');
highlight(w,nodeB,'NodeColor','k');
highlight(w,MN,'NodeColor','r');
k = length(cut_set_cell);
optimalCR = 0;
optimalC = [];
for i = 1:k
    cut_set = cut_set_cell{i};
    [pcset] = getPC(cut_set,skeleton);
    cut_set2 = unique([cut_set,pcset]);
%     highlight(w,cut_set2,'NodeColor','r');
%     highlight(w,cut_set,'NodeColor','k');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    G2 = G;
    
    G0 = rmnode(G2,cut_set);
    bins = conncomp(G0);
    if length(unique(bins)) == 1
        continue
    end
        
    G3 = rmnode(G2,cut_set2);
    bins = conncomp(G3);
    lb = length(unique(bins));
    cut_size = [];
    if lb > 1
        for j = 1:lb
            cut_size = [cut_size,length(find(bins==j))];
        end
        CR = length(cut_set2)/min(max(cut_size)+ length(cut_set2),size(skeleton,1)-max(cut_size));
        if CR <= CRgt
            if optimalCR < CR
                ground_truth_CR = CRgt
                optimalCR = CR
                optimalC = cut_set2;
                ABsize = [max(cut_size)+ length(cut_set2),size(skeleton,1)-max(cut_size)]
                CPAsize = [length(unique([MN,nodeA])),length(unique([MN,nodeB]))]
                iter = i;
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
optimalC;
iter;
[A,B] = findAB(skeleton,iter);
subplot(1,2,2)
w = plot(G);
highlight(w,A,'NodeColor','b');
highlight(w,B,'NodeColor','k');
highlight(w,optimalC,'NodeColor','r');
cutGT{1}= optimalC;
cutGT{2}= A;
cutGT{3}= B;

function [pcset] = getPC(M,skeleton)
n = length(M);
temp = [];
for i = 1:n
    temp = [temp,find(skeleton(M(i),:)==1)];
    temp = [temp,find(skeleton(:,M(i))==1)'];
end
pcset = unique([temp,M]);
end