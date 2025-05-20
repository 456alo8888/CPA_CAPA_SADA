function [stru] = randori(stru)
for p = 1:size(stru,2)
    for q = 1:size(stru,2)
        if stru(p,q) == 1 && stru(q,p) == 1
            if rand > 0.5
                stru(q,p) = 0;
            else
                stru(p,q) = 0;
            end
        end
    end
end
end

