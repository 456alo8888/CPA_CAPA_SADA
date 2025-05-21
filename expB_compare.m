clc; clear;

% Thêm đường dẫn
addpath('alg');
addpath('dataset');
addpath('data');
addpath(genpath('.\SADA'));

data_url = {'andes', 'diabetes', 'link', 'munin'};
dataID = 1;
maxCset = 3;
dataset_name = data_url{dataID};


% Tạo file kết quả với timestamp
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
result_filename = sprintf('results_%s.txt', dataset_name);
fid = fopen(result_filename, 'w');
fprintf('Results will be saved to %s\n', result_filename);


% Đọc ground truth
fprintf('Reading ground truth from data/%s/graph.csv\n', dataset_name);
tab = readtable(['data/', dataset_name, '/graph.csv']);
stru_GT = table2array(tab);

samplesize = [1000];
Alg = {@CPA, @SADA, @CAPA, @CP, @Rando};

for u = 1:length(samplesize)
    nsamples = samplesize(u);
    
    % Đọc dữ liệu với tùy chọn categorical
    fprintf('Reading data from data/%s/data_%d.csv\n', dataset_name, nsamples);
    opts = detectImportOptions(['data/', dataset_name, '/data_', num2str(nsamples), '.csv']);
    numVars = length(opts.VariableNames);
    opts.VariableTypes = repmat({'categorical'}, 1, numVars);
    tab1 = readtable(['data/', dataset_name, '/data_', num2str(nsamples), '.csv'], opts);
    
    
    % Chuyển table thành array
    data = table2array(tab1);
    fprintf('Data matrix size: %s\n', mat2str(size(data)));

    % Tính số mức (ns) cho categorical
    ns = zeros(1, size(data, 2));
    for i = 1:size(data, 2)
        ns(i) = numel(categories(data(:,i)));
    end

    fprintf('Number of categories per variable: %s\n', mat2str(ns));

    % Kiểm tra độ tin cậy
    hps = 5; % Mẫu tối thiểu trên mỗi tổ hợp
    reliable = nsamples / prod(ns) >= hps;
    if ~reliable
        warning('Data may be unreliable for G^2 test: samples (%d) / levels (%d) < %d', ...
            nsamples, prod(ns), hps);
        fprintf(fid, 'Warning: Data may be unreliable for G^2 test: samples (%d) / levels (%d) < %d\n', ...
            nsamples, prod(ns), hps);
    end


    
    % Kiểm tra kích thước
    if isempty(data) || any(size(data) == 0)
        error('Data is empty or has invalid dimensions.');
    end
    
    % Kiểm tra khớp kích thước giữa data và stru_GT
    if size(data, 2) ~= size(stru_GT, 2)
        warning('Data columns (%d) do not match ground truth columns (%d)', ...
            size(data, 2), size(stru_GT, 2));
        fprintf(fid, 'Warning: Data columns (%d) do not match ground truth columns (%d)\n', ...
            size(data, 2), size(stru_GT, 2));
    end
    
    % Chạy thuật toán
    fprintf('Running algorithms for sample size %d\n', nsamples);
    fprintf(fid, 'Running algorithms for sample size %d\n', nsamples);
    % PC_part (rough skeleton)
    tic;
    [r_s, ~] = PC_part(data, 1:size(data, 2), 2, @my_g2_test);
    time_PC_part = toc;
    fprintf('PC_part completed in %.4f seconds\n', time_PC_part);
    fprintf(fid, 'PC_part completed in %.4f seconds\n', time_PC_part);
    
    % CPA
    fprintf('Running CPA algorithm\n');
    fprintf(fid, 'Running CPA algorithm\n');
    tic;
    cell_CPA{u} = [Plus_PC(Alg{1}, data, stru_GT, r_s, maxCset), toc + time_PC_part];
    time_CPA = toc;
    fprintf('CPA completed in %.4f seconds (total: %.4f seconds)\n', time_CPA, time_CPA + time_PC_part);
    fprintf(fid, 'CPA completed in %.4f seconds (total: %.4f seconds)\n', time_CPA, time_CPA + time_PC_part);
    fprintf('  CPA: %s\n', mat2str(cell_CPA{u}));
    fprintf(fid, '  CPA: %s\n', mat2str(cell_CPA{u}));
    
    % SADA
    fprintf('Running SADA algorithm\n');
    fprintf(fid, 'Running SADA algorithm\n');
    tic;
    cell_SADA{u} = [Plus_PC(Alg{2}, data, stru_GT, r_s, maxCset), toc + time_PC_part];
    time_SADA = toc;
    fprintf('SADA completed in %.4f seconds (total: %.4f seconds)\n', time_SADA, time_SADA + time_PC_part);
    fprintf(fid, 'SADA completed in %.4f seconds (total: %.4f seconds)\n', time_SADA, time_SADA + time_PC_part);
    fprintf('  SADA: %s\n', mat2str(cell_SADA{u}));
    fprintf(fid, '  SADA: %s\n', mat2str(cell_SADA{u}));
    
    % CAPA
    fprintf('Running CAPA algorithm\n');
    fprintf(fid, 'Running CAPA algorithm\n');
    tic;
    cell_CAPA{u} = [Plus_PC(Alg{3}, data, stru_GT, r_s, maxCset), toc + time_PC_part];
    time_CAPA = toc;
    fprintf('CAPA completed in %.4f seconds (total: %.4f seconds)\n', time_CAPA, time_CAPA + time_PC_part);
    fprintf(fid, 'CAPA completed in %.4f seconds (total: %.4f seconds)\n', time_CAPA, time_CAPA + time_PC_part);
    fprintf('  CAPA: %s\n', mat2str(cell_CAPA{u}));
    fprintf(fid, '  CAPA: %s\n', mat2str(cell_CAPA{u}));
    
    % Tóm tắt kết quả
    printResult = [get_Mean(cell_CPA)', get_Mean(cell_SADA)', get_Mean(cell_CAPA)'];
    
    fprintf(fid, 'Summary for sample size %d:\n', nsamples);
    fprintf(fid, '  CPA: %s\n', mat2str(cell_CPA{u}));
    fprintf(fid, '  SADA: %s\n', mat2str(cell_SADA{u}));
    fprintf(fid, '  CAPA: %s\n', mat2str(cell_CAPA{u}));
    fprintf(fid, '  Summary: %s\n', mat2str(printResult));
end

% Đóng file
fclose(fid);
fprintf('Results saved to %s\n', result_filename);