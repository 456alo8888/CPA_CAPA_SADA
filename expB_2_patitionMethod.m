clc;clear;
addpath('alg');
addpath('dataset');
addpath('data')
addpath(genpath('.\SADA'));
addpath('dataset');
data_url = {'andes','diabetes','link' , 'munin'}; % four graphs avaliable at https://www.bnlearn.com/bnrepository/
dataID = 1; % choose graph
maxCset = 3; % max conditional size for PC algorithm

dataset_name = data_url{dataID};
fprintf('Dataset: %s\n', dataset_name);
% Đọc ground truth
fprintf('Reading ground truth from data/%s/graph.csv\n', dataset_name);
tab = readtable(['data/', dataset_name, '/graph.csv']); % No ReadRowNames
fprintf('Ground truth table size: %s\n', mat2str(size(tab)));
fprintf('Ground truth column names: %s\n', strjoin(tab.Properties.VariableNames, ', '));
stru_GT = table2array(tab);
fprintf('Ground truth matrix size: %s\n', mat2str(size(stru_GT)));



samplesize = [1000]; % different sample size
Alg = {@CPA,@SADA,@CAPA,@CP,@Rando}  % chosen algorithm
for u = 1:length(samplesize)
    nsamples = samplesize(u);
    fprintf('Processing sample size: %d\n', nsamples);
    
    % Đọc dữ liệu với tùy chọn categorical
    fprintf('Reading data from data/%s/data_%d.csv\n', dataset_name, nsamples);
    opts = detectImportOptions(['data/', dataset_name, '/data_', num2str(nsamples), '.csv']);
    numVars = length(opts.VariableNames); % Xác định số cột
    opts.VariableTypes = repmat({'categorical'}, 1, numVars);
    opts.MissingRule = 'error'; % Báo lỗi nếu có giá trị thiếu
    tab1 = readtable(['data/', dataset_name, '/data_', num2str(nsamples), '.csv'], opts); % No ReadRowNames
    fprintf('Data table size: %s\n', mat2str(size(tab1)));
    fprintf('Data column names: %s\n', strjoin(tab1.Properties.VariableNames, ', '));
    
    % Log thông tin về kiểu dữ liệu của các cột
    fprintf('Column types in tab1:\n');
    for var = tab1.Properties.VariableNames
        if iscell(tab1.(var{1}))
            fprintf('  %s: cell\n', var{1});
        elseif iscategorical(tab1.(var{1}))
            fprintf('  %s: categorical\n', var{1});
        else
            fprintf('  %s: %s\n', var{1}, class(tab1.(var{1})));
        end
    end
    
    % Chuyển các cột cell thành categorical và xử lý giá trị thiếu
    for var = tab1.Properties.VariableNames
        if iscell(tab1.(var{1}))
            fprintf('Converting column %s from cell to categorical\n', var{1});
            col_data = tab1.(var{1});
            col_data(cellfun(@isempty, col_data)) = {'missing'};
            tab1.(var{1}) = categorical(col_data);
        elseif iscategorical(tab1.(var{1}))
            fprintf('Filling missing values in categorical column %s\n', var{1});
            tab1.(var{1}) = fillmissing(tab1.(var{1}), 'constant', 'missing');
        end
    end
    
    data = table2array(tab1); % data là categorical array
    fprintf('Data matrix size: %s\n', mat2str(size(data)));
    
    % Kiểm tra giá trị thiếu
    if any(ismissing(data(:)))
        error('Data contains missing values after processing.');
    end
    
    % Kiểm tra kích thước
    if isempty(data) || any(size(data) == 0)
        error('Data is empty or has invalid dimensions.');
    end
    
    % Kiểm tra khớp kích thước giữa data và stru_GT
    if size(data, 2) ~= size(stru_GT, 2)
        warning('Data columns (%d) do not match ground truth columns (%d)', ...
            size(data, 2), size(stru_GT, 2));
    end

    % ---------------------- rough_skeleton----------------------------
    tic;[r_s,~] = PC_part(data,1:size(data,2),2,@my_g2_test);trs = toc;
    % ---------------------- run CPA ----------------------------------
    tic;cell_CPA{u} = [Plus_PC(Alg(1),data,stru_GT,r_s,maxCset),toc+trs];
    % ---------------------- run SADA ----------------------------------
    tic;cell_SADA{u} = [Plus_PC(Alg(2),data,stru_GT,r_s,maxCset),toc+trs];
    % ---------------------- run CAPA ----------------------------------
    tic;cell_CAPA{u} = [Plus_PC(Alg(3),data,stru_GT,r_s,maxCset),toc+trs];
    % ---------------------- run CP ----------------------------------
    tic;cell_CP{u} = [Plus_PC(Alg(4),data,stru_GT,r_s,maxCset),toc+trs];
    % ---------------------- run Rando ----------------------------------
    tic;cell_Rando{u} = [Plus_PC(Alg(5),data,stru_GT,r_s,maxCset),toc];

    printResult = [get_Mean(cell_CPA)',get_Mean(cell_SADA)',get_Mean(cell_CAPA)',get_Mean(cell_CP)',get_Mean(cell_Rando)'];
end
