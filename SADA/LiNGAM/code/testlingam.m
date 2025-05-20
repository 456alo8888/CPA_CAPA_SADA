function [kkk,kest,Estimate_Error,number_connect_edge,true_connection_identified,true_absence_identified,ntruepruned,nsuperfluous,timeTotal,Srcc]=testlingam(dims,randseed,samples,indegree,Dis )%%%原版本
more off;%%%%不允许分页;more on %%%允许分页
dims=10;
randseed=1;
rand('seed',randseed);
randn('seed',randseed);
samples=2000;
indegree=2;
Dis='A';
% A,拉普拉斯
% D，正太分布
% F,均匀分布
[ X,Bp,S] = Gene_Para( dims,samples,indegree,Dis );
p = randperm(dims);
X = X(p,:);
[aaa,kkk]=sort(p);
 Bp = Bp(p,p);
%%%画图
% Mat = abs(Bp)>0;
% graphViz4Matlab('-adjMat',Mat');
[Best,stde,ci,kest,timeTotal] = lingam(X);
aaa=0;
for i=1:dims
    aaa=(kest(i)-kkk(i)).^2+aaa;
end
Srcc=1-6*aaa/(dims*(dims.^2-1));


% [Best,stde,ci,causalperm] = lingam(X,kkk);
% causalperm=kkk;
%----------------------------------------------------------------------%
Estimate_Error=norm((Bp-Best));
%有多少个连接边
connect_edge = find((Bp(:)~=0));
number_connect_edge=length(connect_edge);%%%实际连接边个数
% Number of true connections identified
true_connection_identified=length(find((Bp(:)~=0) & (Best(:)~=0)));%%%识别正确的边个数
% Number of correctly identified absence of connections
true_absence_identified=length(find((Bp(:)==0) & (Best(:)==0)))-dims;%%%考虑到对角线元素不用
% How many connections were incorrectly thought to be absent
% but which actually were present in the generating model?
% (Note that if very weak connections were pruned this is not
% very significant.)%%%正确的枝被剪掉
truepruned = find((Bp(:)~=0) & (Best(:)==0));
maxtruepruned = max(abs(Bp(truepruned)));
ntruepruned = length(truepruned);%%%正确的枝被剪掉

% if ntruepruned,
%     fprintf('Number of true connections pruned: %d [max was %.3f]\n', ...
% 	    ntruepruned, maxtruepruned );
% else
%     fprintf('Number of true connections pruned: 0\n');
% end
%     
% How many spurious connections were added? (i.e. connections
% absent in the generating model but which the algorithm thought
% were present.) Again, if very weak new connections were added
% this is not very bad, but of course if significant connections
% were spuriously included that is quite bad.%%%加入错误的枝
superfluous = find((Bp(:)==0) & (Best(:)~=0));
maxsuperfluous = max(abs(Best(superfluous)));
nsuperfluous = length(superfluous);%%%加入错误的枝





    
