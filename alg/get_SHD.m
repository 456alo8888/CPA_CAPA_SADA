function[Score]= get_SHD(struC,skeleton)
Score = 0;
for i = 1:size(skeleton,1)
    for j = 1:size(skeleton,1)
        if skeleton(i,j)~= struC(i,j)
            Score = Score + 1;
        end
        if skeleton(i,j) == 1 && skeleton(i,j) == struC(j,i)
            Score = Score - 1;
        end
    end
end
end