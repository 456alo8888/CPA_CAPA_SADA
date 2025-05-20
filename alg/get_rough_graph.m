function stru = get_rough_graph(data,maxCset,Alg)
n = size(data,2);
stru = ones(n,n)-eye(n,n);
for k = 0:maxCset
    for i = 1:n-1
        for j = i+1:n
            if stru(i,j) == 1
                if k == 0
                    ind = Alg(data(:,i),data(:,j),[]);
                    if ind
                        stru(i,j) = 0;
                        stru(j,i) = 0;
                    end
                else
                    p1 = find(stru(i,:)==1);
                    p2 = find(stru(j,:)==1);
                    p3 = find(stru(:,i)==1)';
                    p4 = find(stru(:,j)==1)';
                    p = unique(setdiff([p1,p2,p3,p4],[i,j]));
                    if length(p) >= k
                        csetM = nchoosek(p,k);
                    else
                        continue
                    end
                    len_csetM = size(csetM,1);
                    for t = 1:len_csetM
                        ind = Alg(data(:,i),data(:,j),data(:,csetM(t,:)));
                        if ind
                            stru(i,j) = 0;
                            stru(j,i) = 0;
                            break
                        end
                    end
                end
            end
        end
    end
end
end