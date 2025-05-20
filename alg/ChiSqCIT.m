function [cit] = ChiSqCIT(x, y, Z)
% Chi-squared Conditional Independence Test
% Inputs: x, y - column vectors of discrete values (numeric or categorical)
%         Z - matrix of discrete values (numeric or categorical)
% Output: cit - true if independent given Z, false otherwise

alpha = 0.05;

% Kiểm tra giá trị thiếu
if any(ismissing(x(:))) || any(ismissing(y(:))) || (~isempty(Z) && any(ismissing(Z(:))))
    error('Input contains missing values.');
end

% Kiểm tra kích thước
if length(x) ~= length(y) || (~isempty(Z) && size(Z, 1) ~= length(x))
    error('Input dimensions mismatch: x, y, and Z must have the same number of rows.');
end

% Đảm bảo x, y là categorical
if ~iscategorical(x)
    x = categorical(x);
end
if ~iscategorical(y)
    y = categorical(y);
end
if ~isempty(Z) && ~iscategorical(Z)
    Z = categorical(Z);
end

if isempty(Z)
    % Trường hợp không có biến điều kiện Z
    tbl = crosstab(x, y);
    if all(size(tbl) > 1) % Đảm bảo bảng có kích thước hợp lệ
        [chi2_stat, p, df] = chi2_from_table(tbl);
    else
        chi2_stat = 0;
        p = 1;
        df = 0;
    end
else
    % Trường hợp có biến điều kiện Z
    if size(Z, 2) > 1
        % Chuyển Z thành chỉ số nhóm duy nhất
        [~, ~, z_bins] = unique(Z, 'rows', 'stable');
        z_bins = double(z_bins); % Chuyển z_bins thành double
    else
        z_bins = double(Z); % Chuyển Z thành double nếu chỉ có 1 cột
    end
    max_z = length(unique(z_bins)); % Số lượng nhóm duy nhất
    
    chi2_stat = 0;
    df = 0;
    
    for k = 1:max_z
        idx = (z_bins == k); % Bây giờ z_bins là double, so sánh hợp lệ
        if sum(idx) < 2
            continue; % Bỏ qua nếu nhóm có ít hơn 2 mẫu
        end
        tbl = crosstab(x(idx), y(idx));
        if all(size(tbl) > 1) % Đảm bảo bảng có kích thước hợp lệ
            [chi2_k, ~, df_k] = chi2_from_table(tbl);
            chi2_stat = chi2_stat + chi2_k;
            df = df + df_k;
        end
    end
    
    % Kiểm tra xem có nhóm hợp lệ nào không
    if df == 0
        p = 1; % Nếu không có bảng hợp lệ, giả định độc lập
    else
        p = 1 - chi2cdf(chi2_stat, df);
    end
end

% Quyết định
cit = (p >= alpha);
end

function [chi2_stat, p, df] = chi2_from_table(tbl)
% Helper: compute chi2 statistic, p-value, and df from a contingency table
row_sums = sum(tbl, 2);
col_sums = sum(tbl, 1);
total = sum(tbl(:));

% Tính giá trị kỳ vọng
expected = row_sums * col_sums / total;

% Tránh chia cho 0 và kiểm tra expected
mask = expected > 0;
if any(mask(:))
    chi2_stat = sum(((tbl(mask) - expected(mask)).^2) ./ expected(mask), 'all');
else
    chi2_stat = 0;
end

df = (size(tbl, 1) - 1) * (size(tbl, 2) - 1);
if df <= 0
    df = 0;
    p = 1;
else
    p = 1 - chi2cdf(chi2_stat, df);
end
end