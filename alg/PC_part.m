function [ske,stru] = PC_part(data,Part,maxCset,Alg)
saveInd = [];
n = size(data,2);
skeleton = ones(n,n);
for c = 1:n
    skeleton(c,c) = 0;
end
sid = 1;
for i = 1:n-1
    if ~ismember(i,Part)
        continue
    end
    for j = i+1:n
        if ~ismember(j,Part)
            continue
        end
        continFlag = 1;
        for k = 0:maxCset
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
%------------------ V-structure ----------------
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
% %------------------ propagation Rule 1~3 ----------------
ske1 = skeleton;
ske2 = [];
for repeatimes = 1:100
    if isempty(setdiff(ske1,ske2))
        break;
    end
    ske1 = ske2;
    for j = 1:n-1
        for k = i+1:n
            %------------------ Rule 1
            if skeleton(j,k) == 1 && skeleton(k,j) == 1
                Iset = setdiff(find(skeleton(:,j)==1)',[j,k]);
                for p = 1:length(Iset)
                    if skeleton(Iset(p),k) == 0 && skeleton(k,Iset(p)) == 0
                        skeleton(k,j) = 0;
                    end
                end
            end
            %------------------ Rule 2
            if skeleton(j,k) == 1 && skeleton(k,j) == 1
                JIset = setdiff(find(skeleton(j,:)==1),[j,k]);
                IKset = setdiff(find(skeleton(:,k)==1)',[j,k]);
                if ~isempty(intersect(JIset,IKset))
                    skeleton(k,j) = 0;
                end
            end
            %------------------ Rule 3
            if skeleton(j,k) == 1 && skeleton(k,j) == 1
                Irset = setdiff(find(skeleton(:,k)==1)',[j,k]);
                temp = [];
                for p = 1:length(Irset)
                    if skeleton(Irset(p),j) == 1 && skeleton(j,Irset(p)) == 1
                        temp = [temp,Irset(p)]
                    end
                end
                if length(temp) >= 2
                    for t = 1:length(temp)-1
                        for w = t+1:length(temp)
                            if skeleton(temp(t),temp(w)) == 0 && skeleton(temp(w),temp(t)) == 0
                                skeleton(k,j) = 0;
                                break;
                            end
                        end
                    end
                end
            end
%             %------------------ Rule 4 is not necessary
%             if skeleton(j,k) == 1 && skeleton(k,j) == 1
%                 Iset = intersect(find(skeleton(j,:)==1),find(skeleton(:,j)==1)');
%                 Rset = intersect(find(skeleton(:,k)==1)',find(skeleton(k,:)==0));
%                 for p = 1:length(Iset)
%                     for q = 1:length(Rset)
%                         if skeleton(Iset(p),Rset(q)) == 1 && skeleton(Iset(q),Rset(p)) == 0 && skeleton(Iset(q),k) == 0 && skeleton(k,Iset(q)) == 0
%                             skeleton(k,j) = 0;
%                         end
%                     end
%                 end
%             end
        end
    end
    ske2 = skeleton;
end
%------------------propagation Rule 1~3 end----------------
stru = skeleton;