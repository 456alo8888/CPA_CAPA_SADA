function [RPF1_structure] = Plus_PC(Alg,data,skeleton,r_s,maxCset)
%---------------------------Run algorithm--------------------------
for i = 1:length(Alg)
    [cut_set,nodeA,nodeB,~] = Alg{i}(data,r_s);
    PA = unique([nodeA,cut_set]); 
    PB = unique([nodeB,cut_set]);
    %--------------------------- Run PC ---------------------------
    [~,struA] = PC_part(data,PA,maxCset,@ChiSqCIT);     %change here to ChiSqCIT / my_g2_test for Chi-squared/ G2 test with categorical data
    [~,struB] = PC_part(data,PB,maxCset,@ChiSqCIT);     %change here to ChiSqCIT / my_g2_test for Chi-squared/ G2 test with categorical data
    %--------------------------- Merge ---------------------------
    for v = 1:size(skeleton,1)-1
        for w = v+1:size(skeleton,1)
            if struA(v,w) == 1 && struA(w,v) == 0 && struB(v,w) == 0 && struB(w,v) == 1
               struA(w,v) = 1;
               struB(v,w) = 1;
            end
        end
    end
    struC = struA.*struB;
    struC = randori(struC); % randomly orient the remaining undirected edge
    struC(nodeA,nodeB) = 0;struC(nodeB,nodeA) = 0;
    score{i} = [getRPF_stru(struC,skeleton),get_SHD(struC,skeleton)];
end
%----------------------------------results---------------------------------
RPF1_structure = [];
for k = 1:i
    RPF1_structure = [RPF1_structure;score{k}];
end
end