function errorMean = get_Mean(d)
n = size(d,2);
e = d{1};
% errorBar = zeros(size(e,1),size(e,2));
errorMean = zeros(size(e,1),size(e,2));
for i = 1:size(e,1)
    for j = 1:size(e,2)
        temp = [];
        for k = 1:n
            s = d{k};
            temp = [temp,s(i,j)];
        end
%         errorBar(i,j) = std(temp);
        errorMean(i,j) = mean(temp);
    end
end
end