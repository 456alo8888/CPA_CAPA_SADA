function [cut_set_cell] = edge_cutting2(skeleton)
W = skeleton + skeleton';
D = diag(sum(W));
L = D - W;
[V,~] = eig(L);
v2 = V(:,2); % lambad2 dual
%
s = sortrows([v2,(1:length(v2))'],1);
idx = s(:,2);
[~,b] = sort(idx);
nega = ones(length(v2),1);
%
for i = 1:length(v2)
    nega(i) = -1;
%     if i == 20
%         q2 = abs(v2).*nega(b)
%         [v2,q2]
%     else
        q2 = abs(v2).*nega(b);
%     end
    cut_set_cell{i} = unique(findCutset2(q2,W)); % cut edge dual
end
%----------------------------------------- cut nodes