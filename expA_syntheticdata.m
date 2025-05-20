clc; clear;

% In thông tin khởi đầu
fprintf('Script started at %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

% Sửa addpath để trỏ đến thư mục đúng
addpath('alg'); % Thay bằng đường dẫn thực tế nếu cần
addpath('dataset');

data_url = {'ER-gauss-1000-400-800/0', 'ER-gauss-1000-500-1000/0'};
dataID = 1;
maxCset = 3;
dataset_name = data_url{dataID};
fprintf('Dataset: %s\n', dataset_name);

% Đọc ground truth
fprintf('Reading ground truth from data/%s/groundtruth.csv\n', dataset_name);
tab = readtable(['data/', dataset_name, '/groundtruth.csv']); % No ReadRowNames
fprintf('Ground truth table size: %s\n', mat2str(size(tab)));
fprintf('Ground truth column names: %s\n', strjoin(tab.Properties.VariableNames, ', '));
stru_GT = table2array(tab);
fprintf('Ground truth matrix size: %s\n', mat2str(size(stru_GT)));

samplesize = [1000];
Alg = {@CPA};

for u = 1:length(samplesize)
    nsamples = samplesize(u);
    fprintf('Processing sample size: %d\n', nsamples);
    
    % Đọc dữ liệu với tùy chọn categorical
    fprintf('Reading data from data/%s/data.csv\n', dataset_name);
    opts = detectImportOptions(['data/', dataset_name, '/data.csv']);
    numVars = length(opts.VariableNames); % Xác định số cột
    tab1 = readtable(['data/', dataset_name, '/data.csv'], opts); % No ReadRowNames
    fprintf('Data table size: %s\n', mat2str(size(tab1)));
    fprintf('Data column names: %s\n', strjoin(tab1.Properties.VariableNames, ', '));
    
    
    
    data = table2array(tab1); % data là categorical array
    fprintf('Data matrix size: %s\n', mat2str(size(data)));
    
    
    % Chạy thuật toán PC
    fprintf('Running PC algorithm for sample size %d\n', nsamples);
    tic;
    [~, stru_PC] = PC_part(data, 1:size(data, 2), maxCset, @PaCoT);   
    stru_PC = randori(stru_PC);
    time_PC = toc;
    fprintf('PC structure size: %s\n', mat2str(size(stru_PC)));
    cell_PC{u} = [[getRPF_stru(stru_PC, stru_GT), get_SHD(stru_PC, stru_GT)], time_PC];
    fprintf('PC algorithm completed in %.2f seconds\n', time_PC);
    fprintf('  PC: %s\n', mat2str(cell_PC{u}));
    
    % Chạy CPA+PC
    fprintf('Running CPA+PC algorithm for sample size %d\n', nsamples);
    tic;
    [r_s, ~] = PC_part(data, 1:size(data, 2), 2, @PaCoT);
    trs = toc;
    tic;
    cell_CPA{u} = [Plus_PC(Alg(1), data, stru_GT, r_s, maxCset), toc + trs];
    time_CPA = toc + trs;
    fprintf('CPA+PC algorithm completed in %.2f seconds\n', time_CPA);
    
    % Log kết quả
    printS = [get_Mean(cell_PC)', get_Mean(cell_CPA)'];
    fprintf('Results for sample size %d:\n', nsamples);
    fprintf('  PC: %s\n', mat2str(cell_PC{u}));
    fprintf('  CPA: %s\n', mat2str(cell_CPA{u}));
    fprintf('  Summary: %s\n', mat2str(printS));
end