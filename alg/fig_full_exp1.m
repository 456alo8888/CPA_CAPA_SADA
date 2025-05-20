clc;clear;tic;%rng('default')
fs = 15;
[skeleton,names] = readRnet( '.\dataset\diabetes.net');
% [skeleton,names] = readRnet( '.\dataset\Link.net');
skeleton = sortskeleton(skeleton);
temp = [];
for i = 1:size(skeleton,1)
    if sum(skeleton(i,:)) == 0 && sum(skeleton(:,i)) == 0
        temp = [temp,i];
    end
end
skeleton(:,temp) = [];skeleton(temp,:) = [];
Alg = {@CPA_skeleton,@SADA_skeleton,@CAPA_skeleton,@CP_skeleton};
for i = 1:4
    subplot(2,4,i)
    [cut_set,nodeA,nodeB] = Alg{i}(skeleton);
    %-----------------------------------------
    w = plot(graph(skeleton,'upper'));
    highlight(w,nodeB,'NodeColor','k','Marker','o');
    highlight(w,nodeA,'NodeColor','b','Marker','^');
    highlight(w,cut_set,'NodeColor','r','Marker','d');
    grid minor
    set(gca,'ytick',[-4 -2 0 2 4]);
    set(gca,'xtick',[-2 0 2 4 6]);
    set(gca,'xTickLabel',{'', '', '', '', ''},'Fontname','Times New Roman');
    set(gca,'yTickLabel',{'', '', '', '', ''},'Fontname','Times New Roman');
    if i == 1
        ylabel('Diabetes','Fontname','Times New Roman','FontSize',fs);
    end
end
% [skeleton,names] = readRnet( '.\dataset\Andes.net');
[skeleton,names] = readRnet( '.\dataset\Link.net');
skeleton = sortskeleton(skeleton);
temp = [];
for i = 1:size(skeleton,1)
    if sum(skeleton(i,:)) == 0 && sum(skeleton(:,i)) == 0
        temp = [temp,i];
    end
end
skeleton(:,temp) = [];skeleton(temp,:) = [];
Alg = {@CPA_skeleton,@SADA_skeleton,@CAPA_skeleton,@CP_skeleton};
for i = 1:4
    subplot(2,4,4+i)
    [cut_set,nodeA,nodeB] = Alg{i}(skeleton);
    %-----------------------------------------
    w = plot(graph(skeleton,'upper'));
    highlight(w,nodeB,'NodeColor','k','Marker','o');
    highlight(w,nodeA,'NodeColor','b','Marker','^');
    highlight(w,cut_set,'NodeColor','r','Marker','d');
    grid minor
    set(gca,'ytick',[-4 -2 0 2 4]);
    set(gca,'xtick',[-2 0 2 4 6]);
    set(gca,'xTickLabel',{'', '', '', '', ''},'Fontname','Times New Roman');
    set(gca,'yTickLabel',{'', '', '', '', ''},'Fontname','Times New Roman');
    if i == 1
        ylabel('Link','Fontname','Times New Roman','FontSize',fs);
        xlabel('CPA','Fontname','Times New Roman','FontSize',fs);
    elseif i == 2
        xlabel('SADA','Fontname','Times New Roman','FontSize',fs);
    elseif i == 3
        xlabel('CAPA','Fontname','Times New Roman','FontSize',fs);
    else
        xlabel('Dsep-CP','Fontname','Times New Roman','FontSize',fs);
    end
end
% plot(1,1,'.r','MarkerFaceColor','r','Marker','d')
% hold on 
% plot(1,2,'.b','MarkerFaceColor','b','Marker','>')
% hold on 
% plot(2,1,'.k','MarkerFaceColor','k','Marker','o')
% hold on 
% legend({'Cut-set C','Part A-C','Part B-C'},'NumColumns',3)
toc
%----------------------------------------- 