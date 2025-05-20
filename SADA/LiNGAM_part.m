function struture = LiNGAM_part(data,part)
% fprintf('Solving Basic Problem (Dim. %d, Samp. %d) using LiNGAM\n',size(X,2),size(X,1))
if isempty(part)
    struture = zeros(size(data,2),size(data,2));
    return
end

X = data(:,part);
position = [1:length(part);part];
[B stde ci k W] = estimate(X');
nNode=size(X, 2);
if nNode <20 
    [B P] = prune(X', k, 'W', W, 'B',B, 'stde',stde, 'method', 'wald');
    [x,y]=find(abs(B)>0);
    E=[x,y];
    P=P(abs(B)>0); 
    E=[E,P];
else
    B = prune(X', k, 'W', W, 'B',B, 'stde',stde, 'method', 'resampling');
    [x,y]=find(abs(B)>0);
    E=[x,y];
    P=abs(B(abs(B)>0)); 
    E=[E,P];
end
% E
struture = zeros(size(data,2),size(data,2));
for i = 1:size(E,1)
    %     if E(i,3) == 0
    struture(position(2,E(i,2)),position(2,E(i,1))) = 1;
    %     end
end

end