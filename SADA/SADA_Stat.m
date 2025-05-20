function [T,P, TP,FP, WrongDirection, recall, precision]=SADA_Stat(skeleton, E)
T=sum(sum(skeleton));
P=size(E,1);
TP=0;
WrongDirection=0;
for i=1:P
    if skeleton(E(i,1),E(i,2))
        TP=TP+1;
    elseif skeleton(E(i,2),E(i,1))
        WrongDirection=WrongDirection+1;
    end
end
FP=P-TP;
recall=TP/T;
if P == 0
    precision =0;
else
    precision=TP/P;
end