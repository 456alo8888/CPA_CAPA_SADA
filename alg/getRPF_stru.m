function[Score]= getRPF_stru(Cskeleton,skeleton)
%------------- -------------  require two structures ------------- 
for i = 1:size(Cskeleton,1)-1
    for j = i:size(Cskeleton,1)
        if Cskeleton(i,j) == Cskeleton(j,i)
            Cskeleton(i,j) = 0;
            Cskeleton(j,i) = 0;
        end
    end
end
%------------------------ -- R P F1------------- ------------- 
R = sum(sum(skeleton.*Cskeleton))/sum(sum(skeleton));
if sum(sum(Cskeleton)) == 0
    P =0;
else
    P = sum(sum(skeleton.*Cskeleton))/sum(sum(Cskeleton));
end
if R+P == 0
    Score = [R,P,0];
else
    Score = [R,P,2*R*P/(R+P)];
end
end