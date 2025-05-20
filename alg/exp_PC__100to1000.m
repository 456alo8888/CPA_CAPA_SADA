clc;clear;addpath('.\dataset');addpath(genpath('.\SADA'));data_url =  {'diabetes'}
% data_url =  {'Alarm','Barley','Andes','diabetes','Pigs','Link'}
samples = [500]
rpf_Cell = cell(1,length(samples));
error_Cell = cell(1,length(samples));
for s = 1:length(samples)
    for dt = 1:length(data_url)
        dataID = dt;
        skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
        Alg = {@CPD,@CP};
        Times = 8;
        score_cell = cell(1,Times);
        Time_cell = cell(1,Times);
        r_s_Time_cell = cell(1,Times);
        tmark = clock;
        parfor T = 1:Times
%             T
            rng(T); % by default
            data = genData(skeleton,samples(s)); %!!!!! sample size
            t_r_s = clock;
            r_s = get_rough_graph(data,2,@PaCoT);  % rough_skeleton
            elaptime_r_s = etime(clock,t_r_s);
            r_s_Time_cell{T} = elaptime_r_s;
            % ---------------------- run with PC ----------------------------------
            [RPF1_stru1,~,ela_time1] = Plus_PC(Alg,data,skeleton,r_s);
            % ---------------------- run with Lingam ------------------------------
            %         [RPF1_stru2,~,ela_time2] = Plus_Lingam(Alg,data,skeleton,r_s);
            % ---------------------- results --------------------------------------
            score_cell{T} = [RPF1_stru1];
            Time_cell{T} = [ela_time1'];
        end
        %------------------------- Average results of plus PC & Lingam --------
        score_Ave = [get_Mean(score_cell)]';
        Time_Ave = [get_Mean(Time_cell)];
%         Time_Ave2 = [get_Mean(Time_cell)] + [get_Mean(r_s_Time_cell),get_Mean(r_s_Time_cell),0,0]
        errorBar = [get_errorBar(score_cell)]';
        elaptime = etime(clock,tmark)
    end
    A = vpa(score_Ave,4)
    B = vpa(errorBar,4)
    rpf_Cell{s} = A;
    error_Cell{s} = B;
%     C =vpa([A(:,1),B(:,1),A(:,2),B(:,2),A(:,3),B(:,3),A(:,4),B(:,4)],4)
end
save('rpf_Cell.mat','rpf_Cell'); 
save('error_Cell.mat','error_Cell'); 

function [RPF1_structure,cut_size,elapsed_time] = Plus_PC(Alg,data,skeleton,r_s)
%---------------------------Run algorithm--------------------------
for i = 1:length(Alg)
    tmarkpc = clock;
    [cut_set,nodeA,nodeB,~] = Alg{i}(data,r_s);
    PA = unique([nodeA,cut_set]); PB = unique([nodeB,cut_set]);
    %     sizeG{i} = [size(skeleton,1),length(PA),length(PB),length(cut_set)];
    %     ratio{i} = max(length(PA),length(PB))/size(skeleton,1);
    %---------------------------   Run LiNGAM   ---------------------------
    maxCset = 3; % conditional size =3 to save time
    %     if size(skeleton,1) <= 400
    %         maxCset = 4;
    %     end
    if size(skeleton,1) <= 100
        maxCset = size(skeleton,1)-2;
    end
    struA = stru_learn_part(data,PA,maxCset,@PaCoT);
    struB = stru_learn_part(data,PB,maxCset,@PaCoT);
    struC = struA.*struB;
    struC(nodeA,nodeB) = 0;struC(nodeB,nodeA) = 0;
    for p = 1:size(data,2)
        for q = 1:size(data,2)
            if struC(p,q) == 1 && struC(q,p) == 1
                struC(q,p) = 0;
            end
        end
    end
    score{i} = [getRPF_stru(struC,skeleton), get_SHD(struC,skeleton)];
    time{i} = etime(clock,tmarkpc);
end
%----------------------------------results---------------------------------
RPF1_structure = [];
cut_size = [];
elapsed_time = [];
for k = 1:i
    RPF1_structure = [RPF1_structure;score{k}];
    %     cut_size = [cut_size;sizeG{k}];
    elapsed_time = [elapsed_time;time{k}];
end
end
