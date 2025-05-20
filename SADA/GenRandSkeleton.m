function skeleton = GenRandSkeleton(nNode, nFreeNode, nIndgree)
if nIndgree > nFreeNode
    disp('Error! nIndgree large than  nFreeNode!');
    return;
end
skeleton = false(nNode,nNode);
n=0;
for i=nFreeNode+1:nNode
    n=0;
    nParent=floor(nIndgree);
    tmp=nIndgree-nParent;
    if rand < tmp
        nParent=nParent+1;
    end
    while n<nParent
        parent=randi(i-1);
        if ~skeleton(i,parent)
            skeleton(i,parent)=true;
            n=n+1;
        end
    end
end