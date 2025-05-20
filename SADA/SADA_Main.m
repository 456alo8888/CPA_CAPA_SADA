function E = SADA_Main( X, splitTries)
% INPUT:
% X     - Data matrix
%OUTPUT:
% E    - edges in the form [a,b,P] presets a->b, P is Pvalue of the edge
[n,d]=size(X);
if d <= max(floor(n/10),3) %adaptive threshold for basic sovler
    E = SADA_LiNGAM_Wrapper( X ); 
else
    %divide 
    nv=size(X,2);
%     fprintf('Split Begin, Variable size %d \n', nv);
    [idxA,idxB,idxCut]=SADA_Split(X, splitTries);
%     fprintf('Split Done, A size: %d, B Size: %d, Cut Size: %d \n', sum(idxA), sum(idxB),sum(idxCut));
    %conquer
    if sum(idxA)==0 || sum(idxB)==0
        E = SADA_LiNGAM_Wrapper( X );%can not find a split
    else
        idxA=idxA|idxCut;
        idxB=idxB|idxCut;
        EA=SADA_Main( X(:, idxA),splitTries);
        EB=SADA_Main( X(:, idxB),splitTries);
        %transfer the local idx to global idx
        if size(EA, 1)>0 
            tmpA=find(idxA>0);
            EA(:,1)=tmpA(EA(:,1));
            EA(:,2)=tmpA(EA(:,2));
        end
        if size(EB, 1)>0
            tmpB=find(idxB>0);
            EB(:,1)=tmpB(EB(:,1));
            EB(:,2)=tmpB(EB(:,2));
        end
        %merge
        E=SADA_Merge(X,EA,EB);
    end
end






