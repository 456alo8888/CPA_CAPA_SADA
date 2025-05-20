function x = laprnd(a,b,m,n)%%%a 是位置参数（location parameter）  b ≥ 0 是尺度参数 （scale parameter），m，n分别表示产生随机矩阵的行数和列数
% Reference: http://en.wikipedia.org/wiki/Laplace_distribution
    u = rand(m,n)-0.5;
    x = a - b*sign(u).*log(1-2*abs(u));
end