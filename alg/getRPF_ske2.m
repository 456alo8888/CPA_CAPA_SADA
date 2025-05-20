function[Score]= getRPF_ske2(Cskeleton,skeleton)
    Cskeleton = Cskeleton + Cskeleton';
    Cskeleton(find(Cskeleton == 2))=1;
    skeleton = skeleton + skeleton';
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