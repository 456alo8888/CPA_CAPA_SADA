function E=SADA_Merge(data, EA,EB)
% INPUT:
% EA,EB - the solution of subproblems, in the form [a,b,P] presets a->b, P is Pvalue of the edge
%OUTPUT:
% E     - the merge edges 

%merge 
E=[EA;EB];
if ~isempty(E)
    nE=size(E,1);
    flag=true(nE, 1);
    E=sortrows(E, 3);%sort in asending order of p valude
    %remove pair redudancy 
    for i=1:nE
        for j=i+1:nE
            if E(i,1)==E(j,1) & E(i,2)==E(j,2)
                flag(j)=false;
            end
        end
    end
    E=E(flag, :);
    
    %remove path redundancy
    nE=size(E,1);
    if nE>0    
        flag=true(nE, 1);
        nNode=max(max(E(:,[1,2])));
        reachable=logical(eye(nNode));
        GInter=false(nNode, nNode, nNode);

%         for i=1:nE
%             reachable(E(i,1),E(i,2))=true;
%             for j=1:nNode
%                 for k=1:nNode
%                     if ~(j==E(i,1)&& k==E(i,2))
%                         if reachable(j, E(i,1))&&reachable(E(i,2),k)%E(i.1)-> E(i,2) are
%                             GInter(j,k, E(i,1))=true;
%                             GInter(j,k, E(i,2))=true;
%                             GInter(j,k, :)= GInter(j,k, :) | GInter(j,E(i,1), :)|GInter(E(i,2),k, :);
%                         end
%                     end
%                 end
%             endx`
%         end
        for i=1:nE
            GInter(E(i,1),:, E(i,2))=true;
            GInter(:, E(i,2), E(i,1))=true;
        end
        for i=1:nE
            %GInter(E(i,1),E(i,2),:)=true;
            GInter(E(i,1),E(i,2),E(i,1))=false;
            GInter(E(i,1),E(i,2),E(i,2))=false;
            if checkSeperate(data(:, E(i,1)), data(:, E(i,2)), data(:, GInter(E(i,1),E(i,2),:), :), 0.05)
                flag(i)=false;
            end
        end
        E=E(flag, :);
    end
    %remove conflict
    nE=size(E,1);
    if nE>0     
        flag=true(nE, 1);
        nNode=max(max(E(:,[1,2])));
        reachable=logical(eye(nNode));
        for i=1:nE
            Gtmp=reachable;
            Gtmp(E(i,1), E(i,2))=true;% add reachable E(i,1) -> E(i,2) 
            Gtmp(Gtmp(:, E(i,1)),Gtmp(E(i,2),:))=true;%x->E(i.1)-> E(i,2)->Y
            if ~isequal(Gtmp & Gtmp', eye(nNode))
                flag(i)=false;
            else
                reachable=Gtmp;
            end
        end
        E=E(flag, :);
    end
end


    

function ind = checkSeperate(x, y, z, alpha)
%%%check each v in X is independent of each v in Y, given Z
[n, dx]=size(x);
[n, dy]=size(y);
[n, dz]=size(z);
ind=true;

xzInd=false(dx,dz);
for i=1:dx
    for j=1:dz
        xzInd(i,j) = PartialCorrelationCIT(x(:,i), z(:, j), [], alpha);
    end;
end
yzInd=false(dy,dz);
for i=1:dy
    for j=1:dz
        yzInd(i,j) = PartialCorrelationCIT(y(:,i), z(:, j),[], alpha);
    end;
end     
        
for i=1:dx
    for j=1:dy
        %%%%%%%%%% shrink the condition set using independence%%%%%%%%%%
        xyzind=false(dz,1);
        for k=1:dz
            xyzind(k) = xzInd(i,k) | yzInd(j,k);
        end;
        %%%%%%%%%% shrink the condition set using cond independenc%%%%%%%%%%
        for k1=1:dz                
            if xyzind(k1)==false 
                for k2=1:dz
                    if  k1~=k2
                        xyzind(k2) = PartialCorrelationCIT(x(:,i), z(:, k1), z(:, k2), alpha) | PartialCorrelationCIT(y(:,j),z(:, k1), z(:, k2), alpha);
                    end
                end
            end
        end;
        zTmp=z(:, find(xyzind==false));

        if isempty(zTmp)
            ind = PartialCorrelationCIT(x(:,i), y(:,j), [], alpha);
            if ind ==false
                return
            end;
        else
            %%%%%%%%%% enumerate all possible subsets of condSetIdx %%%%%%%%
            ind=false;
            dzTmp=size(zTmp,2);
            maxCITSize=min(size(zTmp,2),log2(n/10));
            for k=0:(2^maxCITSize-1)
                B=dec2bin(k,dzTmp);
                if true==PartialCorrelationCIT(x(:,i), y(:,j), zTmp(:, find(B=='1')), alpha);
                    ind=true;
                    break ;
                end
            end
            if ind == false 
                return;
            end;
        end
    end
end

function cit=PartialCorrelationCIT(x, y, Z, alpha)
%implement according to  http://en.wikipedia.org/wiki/Partial_correlation
if isempty(Z)
    n=length(x);
    ncit=0;
    pcc=corrcoef(x,y);
    pcc=pcc(1,2);
else
    [n,ncit]=size(Z);
    Z=[ones(n,1),Z];
    wx=Z\x;
    rx=x-Z*wx;
    wy=Z\y;
    ry=y-Z*wy;
    pcc=(n*rx'*ry- sum(rx)*sum(ry))/sqrt(n*rx'*rx-sum(rx)^2)/sqrt(n*ry'*ry-sum(ry)^2);
end
zpcc=0.5*log((1+pcc)/(1-pcc));
if sqrt(n-ncit-3)*abs(zpcc) > icdf('normal',1-alpha/2,0,1) 
    cit=false;
else
    cit=true;
end

