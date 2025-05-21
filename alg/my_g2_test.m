% my_g2_test from Causal Learner: A Toolbox for Causal Structure and Markov Blanket Learning
% https://github.com/z-dragonl/Causal-Learner
function [cit] = my_g2_test(X_col, Y_col, S_cols)
    alpha = 0.05;

    % Chuẩn bị dữ liệu
    if isempty(S_cols)
        data_sub = [X_col, Y_col];
    else
        data_sub = [X_col, Y_col, S_cols];
    end

    % Tính số mức rời rạc của từng biến
    ns = zeros(1, size(data_sub, 2));
    for i = 1:length(ns)
        ns(i) = numel(categories(data_sub(:, i)));
    end

    % Tính p-value bằng kiểm định G^2
    [p, stat] = citpvalue(data_sub, ns);

    % Độc lập nếu p > alpha
    cit = (p > alpha);
end

function [p, stat] = citpvalue(Data, ns)
    % Data: categorical array, mỗi cột là một biến

    [nObs, nVars] = size(Data);

    % Đếm số lượng tổ hợp giá trị xuất hiện
    Obs = accumarray(double(Data), 1, ns);

    % Biến 1: X, Biến 2: Y, Biến 3+: Z (điều kiện)
    ObsSum2 = sum(Obs, 2); % Sum theo Y
    Obs_xs = repmat(ObsSum2, [1, ns(2)]);

    ObsSum1 = sum(Obs, 1); % Sum theo X
    Obs_ys = repmat(ObsSum1, [ns(1), 1]);

    Obs_s = sum(Obs(:));
    Obs_s = repmat(Obs_s, ns(1), ns(2));

    % Giá trị kỳ vọng theo độc lập
    Exp = Obs_xs .* Obs_ys ./ Obs_s;

    % Tính số bậc tự do
    if nVars > 2
        j3NLevelsProd = prod(ns(3:end));
    else
        j3NLevelsProd = 1;
    end

    ObsSum1 = reshape(ObsSum1, ns(2), j3NLevelsProd);
    ObsSum2 = reshape(ObsSum2, ns(1), j3NLevelsProd);

    df = 0;
    for iComb = 1:j3NLevelsProd
        df = df + max(ns(1) - 1 - sum(~ObsSum2(:, iComb)), 0) * ...
                  max(ns(2) - 1 - sum(~ObsSum1(:, iComb)), 0);
    end

    if df == 0
        p = 1;
        stat = 0;
        return;
    end

    Obs_vector = Obs(:);
    Exp_vector = Exp(:);

    stat = chi2stat(Obs_vector, Exp_vector);

    % p-value với phân phối Chi-squared
    p = gammainc(stat/2, df/2, 'upper');
end

function stat = chi2stat(obs, exp)
    % G^2 statistic
    terms = obs .* log(obs ./ exp);
    terms(isnan(terms)) = 0;
    stat = 2 * sum(terms);
end
