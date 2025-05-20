function [cut_set] = edge_cutting(skeleton)
W = skeleton + skeleton';
D = diag(sum(W));
L = D - W;
[V,~] = eig(L);
v2 = V(:,2); % lambad2 dual
cut_set = unique(findCutset(v2,W)); % cut edge dual
%----------------------------------------- cut nodes