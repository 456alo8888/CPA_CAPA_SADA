clc;clear;addpath('.\alg');addpath('.\dataset');addpath(genpath('.\SADA'));
data_url = {'alarm','andes','diabetes','Link'};
dataID = 1; % choose graph
maxCset = 3;
skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
samplesize = [100 300 500 700 900];
Alg = {@CPA}
for u = 1:length(samplesize)
    nsamples = samplesize(u)
    parfor T = 1:6
         T;
        rng(T) % fixed rand
        data = genData(skeleton,nsamples);
        % ---------------------- run PC ----------------------------------
        tic;[skeC,struC] = PC_part(data,1:size(data,2),maxCset,@PaCoT);  % rough_skeleton
        for p = 1:size(data,2)
            for q = 1:size(data,2)
                if struC(p,q) == 1 && struC(q,p) == 1
                    if rand > 0.5
                        struC(q,p) = 0;
                    else
                        struC(p,q) = 0;
                    end
                end
            end
        end
        cell_PC{T} = [[getRPF_stru(struC,skeleton),get_SHD(struC,skeleton)],toc];
        % ---------------------- rough_skeleton----------------------------
        tic;[r_s,~] = PC_part(data,1:size(data,2),2,@PaCoT);trs = toc;
        % ---------------------- run CPA ----------------------------------
        tic;cell_CPA{T} = [Plus_PC(Alg(1),data,skeleton,r_s,maxCset),toc+trs];
    end
    printS = [get_Mean(cell_PC)',get_Mean(cell_CPA)']
end
