function [ske,stru] = PC_fast_part(data,Part,maxCset,Alg)
saveInd = [];
n = size(data,2);
skeleton = ones(n,n);
for c = 1:n
    skeleton(c,c) = 0;
end
sid = 1;
for k = 0:maxCset
    for i = 1:n-1
        if ~ismember(i,Part)
            continue
        end
        for j = i+1:n
            if ~ismember(j,Part)
                continue
            end
            continFlag = 1;
            if skeleton(i,j) == 1 && continFlag == 1
                if k == 0
                    ind = Alg(data(:,i),data(:,j),[]);
                    if ind
                        skeleton(i,j) = 0;
                        skeleton(j,i) = 0;
                    end
                else
                    p1 = find(skeleton(i,:)==1);
                    p2 = find(skeleton(j,:)==1);
                    p3 = find(skeleton(:,i)==1)';
                    p4 = find(skeleton(:,j)==1)';
                    p = unique(setdiff([p1,p2,p3,p4],[i,j]));
                    p = intersect(Part,p);
                    if length(p) >= k
                        csetM = nchoosek(p,k);
                    else
                        break
                    end
                    len_csetM = size(csetM,1);
                    for t = 1:len_csetM
                        ind = Alg(data(:,i),data(:,j),data(:,csetM(t,:)));
                        if ind
                            saveInd{sid} = [i,j,csetM(t,:)];
                            sid = sid + 1;
                            skeleton(i,j) = 0;
                            skeleton(j,i) = 0;
                            continFlag = 0;
                            break
                        end
                    end
                end
            end
        end
    end
end
ske = skeleton;
%------------------V-structure b----------------
for k = 1:length(saveInd)
    i = saveInd{k}(1);
    j = saveInd{k}(2);
    z = setdiff(saveInd{k},[i,j]);
    pcI = [find(skeleton(i,:)==1),find(skeleton(:,i)==1)'];
    pcJ = [find(skeleton(j,:)==1),find(skeleton(:,j)==1)'];
    pcIJ = intersect(pcI,pcJ);
    collider = setdiff(pcIJ,z);
    if ~isempty(collider)
        for s = 1:length(collider)
            if skeleton(i,collider(s)) == 1
                skeleton(collider(s),i) = 0;
            end
            if skeleton(j,collider(s)) == 1
                skeleton(collider(s),j) = 0;
            end
        end
    end
end
stru = skeleton;
%------------------V-structure e----------------