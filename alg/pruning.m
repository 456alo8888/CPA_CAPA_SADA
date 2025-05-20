function [PRS,nodes] = pruning(RS)
t = 0;
nodes =[];
while t == 0
    t = 1;
    n = size(RS,1);
    for i = 1:n
        if sum(RS(:,i))==1
            t = 0;
            RS(:,i) = 0;
            RS(i,:) = 0;
            nodes = [nodes,i];
        end
    end
end
PRS = RS;
end