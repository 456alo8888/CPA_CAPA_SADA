function[data]=genData_PAMI(stru,nsamples,functionType,noiseType)
[dim, ~]=size(stru);
data = (rand(nsamples, dim)-0.5);
for i=1:dim
    parentidx=find(stru(:,i)==true);
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
        if noiseType == 1
            n =  randn(nsamples,1)*(rand*0.5+0.5);
        elseif noiseType == 2
            n =  (rand(nsamples,1)-0.5)*(rand*0.5+0.5);
        elseif noiseType == 3
            n =  chi2rnd(1,nsamples,1)*(rand*0.5+0.5);
            n = n - mean(n);
        elseif noiseType == 4
            n =  exprnd(0.5,nsamples,1)*(rand*0.5+0.5);
            n = n - mean(n);
        elseif noiseType == 5
            n =  betarnd(0.2,0.8,nsamples,1)*(rand*0.5+0.5);
            n = n - mean(n);
        end
        if functionType == 1
            data(:, i)= pasample + n;
        elseif functionType == 2
            data(:, i)= sin(pasample) + n;
        elseif functionType == 3
            data(:, i)= sin(pasample + n);
        end
        data(:, i) = mapminmax(data(:, i)')';
    end
end
end