function [cut_set,nodeA,nodeB,nodeCD]=Rando(X,r_s)
cut_set =[];
nodeCD =[];
r_s = [];
r = randperm(size(X,2));
nodeA = r(1:floor(size(X,2)/2));
nodeB = r(floor(size(X,2)/2)+1:size(X,2));
end
