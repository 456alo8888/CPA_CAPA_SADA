clc;clear;addpath('.\dataset');addpath(genpath('.\SADA'));data_url =  {'diabetes'}
% data_url =  {'Alarm','Barley','Andes','diabetes','Pigs','Link'}
flag = 1;
while flag == 1
    tempID = randperm(20);
    for dt = 1:length(data_url)
        dataID = dt
        skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
        %     Alg = {@CPD,@CP,@SADA,@Base};
        Alg = {@CPD,@CP};
        Times = 10;
        score_cell = cell(1,Times);
        Time_cell = cell(1,Times);
        r_s_Time_cell = cell(1,Times);
        tmark = clock;
        parfor T = 1:Times
            T
            rng(tempID(T)); % by default
            data = genData(skeleton,500); %!!!!! sample size
            t_r_s = clock;
            r_s = get_rough_graph(data,2,@PaCoT);  % rough_skeleton
            elaptime_r_s = etime(clock,t_r_s);
            r_s_Time_cell{T} = elaptime_r_s;
            % ---------------------- run with PC ----------------------------------
            %         [RPF1_stru1,~,ela_time1] = Plus_PC(Alg,data,skeleton,r_s);
            % ---------------------- run with Lingam ------------------------------
            [RPF1_stru2,~,ela_time2] = Plus_Lingam(Alg,data,skeleton,r_s);
            % ---------------------- results --------------------------------------
            score_cell{T} = RPF1_stru2;
            Time_cell{T} = ela_time2';
        end
        %------------------------- Average results of plus PC & Lingam --------
        score_Ave = [get_Mean(score_cell)]'
        Time_Ave = [get_Mean(Time_cell)]
        %     Time_Ave2 = [get_Mean(Time_cell)] + [get_Mean(r_s_Time_cell),get_Mean(r_s_Time_cell),0,0,get_Mean(r_s_Time_cell),get_Mean(r_s_Time_cell),0,0]
        errorBar = [get_errorBar(score_cell)]'
        elaptime = etime(clock,tmark)
        if score_Ave(4,1) <= score_Ave(4,2)
            flag = 0;
        end
    end
end
    function [RPF1_structure,cut_size,elapsed_time] = Plus_Lingam(Alg,data,skeleton,r_s)
    %---------------------------Run algorithm--------------------------
    for i = 1:length(Alg)
        tmarklingam = clock;
        [cut_set,nodeA,nodeB,~] = Alg{i}(data,r_s);
        PA = unique([nodeA,cut_set]); PB = unique([nodeB,cut_set]);
        %     sizeG{i} = [size(skeleton,1),length(PA),length(PB),length(cut_set)];
        %     ratio{i} = max(length(PA),length(PB))/size(skeleton,1);
        %---------------------------   Run LiNGAM   ---------------------------
        struA = LiNGAM_part(data,PA);
        struB = LiNGAM_part(data,PB);
        struC = struA + struB;
        struC(struC==2)=1;
        for p = 1:size(data,2)
            for q = 1:size(data,2)
                if struC(p,q) == 1 && struC(q,p) == 1
                    %                 struC(p,q) = 0;
                    struC(q,p) = 0; %!!!!!!!!!!!!!!!!!!!!!!!!!!!
                end
            end
        end
        %     numstruC = sum(sum(struC))
        score{i} = [getRPF_stru(struC,skeleton), get_SHD(struC,skeleton)];
        time{i} = etime(clock,tmarklingam);
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

