s = [1 1 1 2 2 3 3 4 5 5 6 7];
t = [2 4 8 3 7 4 6 5 6 8 7 8];
weights = [10 10 1 10 1 10 1 1 12 12 12 12];
names = {'A' 'A' 'C' 'D' 'E' 'F' 'G' 'H'};
G = graph(s,t,weights,names)
plot(G,'EdgeLabel',G.Edges.Weight)