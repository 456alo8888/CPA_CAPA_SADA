function [skeleton] = stru_learn_partB(data,Part,maxCset,Alg,skeleton)
n = size(data,2);
skeleton = skeleton + skeleton';
skeleton(skeleton==2)=1;
% skeleton = ones(n,n);
% for c = 1:n
%     skeleton(c,c) = 0;
% end
for k = 0:maxCset
    for i = 1:n-1
        if ~ismember(i,Part)
            continue
        end
        for j = i+1:n
            if ~ismember(j,Part)
                continue
            end
            if skeleton(i,j) == 1
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
                        continue
                    end
                    len_csetM = size(csetM,1);
                    for t = 1:len_csetM
                        ind = Alg(data(:,i),data(:,j),data(:,csetM(t,:)));
                        if ind
                            skeleton(i,j) = 0;
                            skeleton(j,i) = 0;
                            %------------------V-structure ----------------
                            pcI = [find(skeleton(i,:)==1),find(skeleton(:,i)==1)'];
                            pcJ = [find(skeleton(j,:)==1),find(skeleton(:,j)==1)'];
                            pcIJ = intersect(pcI,pcJ);
                            collider = setdiff(pcIJ,csetM(t,:));
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
                            %------------------V-structure ----------------
                            break
                        end
                    end
                end
            end
        end
    end
end