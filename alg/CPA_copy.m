function [cut_set_original,nodeA,nodeB,pruning_nodes] = CPA(~,r_s)
%-----------------------------------------
% r_s = PC_sklearn_part(data,1:size(data,2),2,@PaCoT);  % rough_skeleton
%-----------------------------------------
prune_flag = 0; % prune or not 
% if prune_flag
%     [p_s_original,~] = pruning(r_s); % p_s_original: pruning skeleton
%     p_s = p_s_original;
%     p_s(all(p_s==0,2),:)=[];
%     p_s(:,all(p_s==0,1))=[]; % p_s: skeleton without ODS
% else
    p_s = r_s;
% end
%-----------------------------------------
[W,edgeM] = change2dual(p_s); % dual skeleton
D = diag(sum(W));
L = D - W;
[V,~] = eig(L);
v2 = V(:,2); % lambad2 dual
cut_set = unique(findCutset(v2,W)); % cut edge dual
%----------------------------------------- cut nodes
temp = [];
for i = 1:length(cut_set)
    for j = 1:size(edgeM,1)
        if cut_set(i) == edgeM(j,3)
            temp = [temp,edgeM(j,[1,2])];
        end
    end
end
cut_set_original = unique(temp); % cut nodes original
%----------------------------------------- + and - edges
nodeA = find(v2>0);
temp = [];
for i = 1:length(nodeA)
    for j = 1:size(edgeM,1)
        if nodeA(i) == edgeM(j,3)
            temp = [temp,edgeM(j,[1,2])];
        end
    end
end
nodeA = unique(setdiff(temp,cut_set_original)); % nodeA original, remove intera
nodeB = setdiff(1:size(p_s,1),[cut_set_original,nodeA]);
pruning_nodes = [];
if prune_flag == 0 % prune or not 
    return
end
%----------------------------------------- get original ID
id1 = all(p_s_original==0,1);
id2 = find(id1==0);
for i = 1:length(cut_set_original)
    cut_set_original(i) = cut_set_original(i) + sum(id1(1:id2(cut_set_original(i)))); % ID before pruning
end
for i = 1:length(nodeA)
    nodeA(i) = nodeA(i) + sum(id1(1:id2(nodeA(i)))); % ID before pruning nodeA
end
for i = 1:length(nodeB)
    nodeB(i) = nodeB(i) + sum(id1(1:id2(nodeB(i)))); % ID before pruning nodeA
end
pruning_nodes = setdiff(1:size(r_s,1),[cut_set_original,nodeA,nodeB]); % pruning_nodes
end

