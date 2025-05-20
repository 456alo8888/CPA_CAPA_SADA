function [d_s,edgeM] = change2dual(p_s)
G = graph(p_s,'upper');
m = numedges(G);
n = numnodes(G);
d_s = zeros(m,m);
edgeM = zeros(m,3);
t = 1;
for i = 1:n-1
    for j = i+1:n
        if p_s(i,j) == 1
            edgeM(t,:) = [i,j,t];
            t = t + 1;
        end
    end
end
for p = 1:m-1
    for r = p+1:m
        if length(unique([edgeM(p,[1,2]),edgeM(r,[1,2])]))<4
            d_s(edgeM(p,3),edgeM(r,3)) = 1;
            d_s(edgeM(r,3),edgeM(p,3)) = 1;
        end
    end
end
end

