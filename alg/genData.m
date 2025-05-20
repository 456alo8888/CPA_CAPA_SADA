function[data]=genData(skeleton,nsamples)
[dim, ~]=size(skeleton);
data = (rand(nsamples, dim)-0.5);
for i=1:dim
    parentidx=find(skeleton(:,i)==true);
    for j=1:length(parentidx)
        if parentidx(j)==i
            parentidx(j)=[];
        end
    end
    if ~isempty(parentidx)
        pasample = 0;
        for w = 1:length(parentidx)
            pasample = pasample + data(:, parentidx(w))*(rand*0.5+0.5);
        end
        n =  (rand(nsamples,1)-0.5)*(rand*0.5+0.5);
        data(:, i)= pasample + n;
        data(:, i) = mapminmax(data(:, i)')';
    end
end
end