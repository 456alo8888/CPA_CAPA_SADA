function Cset = findCutset2(q2,A)
v1 = find(q2>0);
v2 = find(q2<0);
C1 = [];
C2 = [];
for i = 1:length(v1)
    for j = 1:length(v2)
        if A(v1(i),v2(j))==1
            C1 = [C1,v1(i)];
            C2 = [C2,v2(j)];
        end
    end
end
% C1
Cset = C2;
% Cset = [C1,C2];
end