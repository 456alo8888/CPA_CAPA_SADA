k = 1
for i = 1:100
    if ~isempty(Time_cell{i})
        Time_cell2{k} = Time_cell{i}
        k = k + 1
    end
end
    score_Ave = [get_Mean(score_cell2)]'
    errorBar = [get_errorBar(score_cell2)]'
    Time_Ave = [get_Mean(Time_cell2)]