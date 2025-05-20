function [A,B] = findAB(skeleton,id)
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
for i = 1:id
    nega(i) = -1;
    q2 = abs(v2).*nega(b);
end
A = find(q2>=0)';
B = find(q2<0)';
%----------------------------------------- cut nodes