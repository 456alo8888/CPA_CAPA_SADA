clc;clear;tic;fs = 12;lw = 1.3;
load res_causaldiscovery
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = result(1:15,:);
for i = 1:5
    subplot(4,5,i)
    plot(1:3,d([1,6,11]+i-1,1),'*-','Color',[0.85 0.33 0.1],'LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,2),'o--','Color',[0 0.45 0.74],'LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,3),'s:','LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,4),'^-.','LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,5),'+--','LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,1),'*-','Color',[0.85 0.33 0.1],'LineWidth',lw);hold on
    grid minor
    xlim([0.8,3.2]);
    set(gca,'xtick',[1 2 3]);
    set(gca,'xTickLabel',{'', '', ''},'Fontname','Times New Roman');
    if i == 1
        ylabel('Alarm','Fontname','Times New Roman','FontSize',fs);
         ylim([0.1,0.7]);
    elseif i ==2
         ylim([0.1,0.7]);
    elseif i ==3
         ylim([0.1,0.7]);
    elseif i ==4
         ylim([10,70]);
    elseif i ==5
         ylim([0,4]);
    end
end
legend('CPA','SADA','CAPA','Dsep-CP','Rand','Orientation','horizontal');

d = result(16:30,:);
for i = 1:5
    subplot(4,5,i+5)
    plot(1:3,d([1,6,11]+i-1,1),'*-','Color',[0.85 0.33 0.1],'LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,2),'o--','Color',[0 0.45 0.74],'LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,3),'s:','LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,4),'^-.','LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,5),'+--','LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,1),'*-','Color',[0.85 0.33 0.1],'LineWidth',lw);hold on
    grid minor
    xlim([0.8,3.2]);
    set(gca,'xtick',[1 2 3]);
    set(gca,'xTickLabel',{'', '', ''},'Fontname','Times New Roman');
    if i == 1
        ylabel('Andes','Fontname','Times New Roman','FontSize',fs);
         ylim([0.1,0.7]);
    elseif i ==2
         ylim([0.1,0.7]);
    elseif i ==3
         ylim([0.1,0.7]);
    elseif i ==4
         ylim([100,500]);
    elseif i ==5
         ylim([0,1000]);
    end
end

d = result(31:45,:);
for i = 1:5
    subplot(4,5,i+10)
    plot(1:3,d([1,6,11]+i-1,1),'*-','Color',[0.85 0.33 0.1],'LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,2),'o--','Color',[0 0.45 0.74],'LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,3),'s:','LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,4),'^-.','LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,5),'+--','LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,1),'*-','Color',[0.85 0.33 0.1],'LineWidth',lw);hold on
    grid minor
    xlim([0.8,3.2]);
    set(gca,'xtick',[1 2 3]);
    set(gca,'xTickLabel',{'', '', ''},'Fontname','Times New Roman');
    if i == 1
        ylabel('Diabetes','Fontname','Times New Roman','FontSize',fs);
         ylim([0.1,0.75]);
    elseif i ==2
         ylim([0.1,0.75]);
    elseif i ==3
         ylim([0.1,0.75]);
    elseif i ==4
         ylim([100,1000]);
    elseif i ==5
         ylim([0,5000]);
    end
end

d = result(46:60,:);
for i = 1:5
    subplot(4,5,i+15)
    plot(1:3,d([1,6,11]+i-1,1),'*-','Color',[0.85 0.33 0.1],'LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,2),'o--','Color',[0 0.45 0.74],'LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,3),'s:','LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,4),'^-.','LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,5),'+--','LineWidth',lw);hold on
    plot(1:3,d([1,6,11]+i-1,1),'*-','Color',[0.85 0.33 0.1],'LineWidth',lw);hold on
    grid minor
    xlim([0.8,3.2]);
    set(gca,'xtick',[1 2 3]);
    set(gca,'xTickLabel',{'100', '300', '500'},'Fontname','Times New Roman');
    if i == 1
        ylabel('Link','Fontname','Times New Roman','FontSize',fs);
        xlabel('Recall','Fontname','Times New Roman','FontSize',fs);
         ylim([0.1,0.5]);
    elseif i ==2
        xlabel('Precision','Fontname','Times New Roman','FontSize',fs);
         ylim([0.1,0.5]);
    elseif i ==3
        xlabel('F1','Fontname','Times New Roman','FontSize',fs);
         ylim([0.1,0.5]);
    elseif i ==4
        xlabel('SHD','Fontname','Times New Roman','FontSize',fs);
         ylim([500,2000]);
    elseif i ==5
        xlabel('Elapsed time(s)','Fontname','Times New Roman','FontSize',fs);
         ylim([0,40000]);
    end
end