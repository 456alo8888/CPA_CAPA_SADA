clc; clear;

% In thông tin khởi đầu
fprintf('Script started at %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

% Sửa addpath để trỏ đến thư mục đúng
addpath('alg'); % Thay bằng đường dẫn thực tế nếu cần
addpath('dataset');

data_url = {'andes', 'diabetes', 'link', 'munin'};
dataID = 1;
maxCset = 3;
dataset_name = data_url{dataID};
fprintf('Dataset: %s\n', dataset_name);

% Đọc ground truth
fprintf('Reading ground truth from data/%s/graph.csv\n', dataset_name);
tab = readtable(['data/', dataset_name, '/graph.csv']); % No ReadRowNames
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
    
    % Chạy thuật toán PC
    fprintf('Running PC algorithm for sample size %d\n', nsamples);
    tic;
    [~, stru_PC] = PC_part(data, 1:size(data, 2), maxCset, @my_g2_test);
    stru_PC = randori(stru_PC);
    time_PC = toc;
    fprintf('PC structure size: %s\n', mat2str(size(stru_PC)));
    cell_PC{u} = [[getRPF_stru(stru_PC, stru_GT), get_SHD(stru_PC, stru_GT)], time_PC];
    fprintf('PC algorithm completed in %.2f seconds\n', time_PC);
    fprintf('  PC: %s\n', mat2str(cell_PC{u}));
    
    % Chạy CPA+PC
    fprintf('Running CPA+PC algorithm for sample size %d\n', nsamples);
    tic;
    [r_s, ~] = PC_part(data, 1:size(data, 2), 2, @my_g2_test);
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