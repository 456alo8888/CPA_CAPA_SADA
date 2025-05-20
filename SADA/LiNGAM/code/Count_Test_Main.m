function [  ] = Count_Test_Main(  )
% %%
% clear all;clc;
% %%%拉普拉斯分布
% dims=20;
% randseed=10;
% R = zeros(randseed,dims,6);%%%表示randseed个dims*4个矩阵
% B=zeros(dims,6);
% Sam=[1000;2000;5000];
% Dim=[10;20;30;40;50;60;70;80;90;100;110;120;130;140;150;160;170;180;190;200];
% A=cell(1,3);
% for k=1:3
%     for j=1:randseed
%         j
%         a=zeros(dims,6);
%         for i=1:dims
%             i
%             [kkk,kest,estimate_error,number_connect_edge,true_connection_identified,true_absence_identified,ntruepruned,nsuperfluous,timeTotal,Srcc]=testlingam(Dim(i),j,Sam(k),2,'A');
%             if true_connection_identified+ntruepruned==0||true_connection_identified+nsuperfluous==0
%                 a(i,:)=[estimate_error,0,0,0,timeTotal,Srcc]';
%             else
%                 p=true_connection_identified/(true_connection_identified+ntruepruned);
%                 q=true_connection_identified/(true_connection_identified+nsuperfluous);
%                 r=2*(p*q)/(p+q);
%                 a(i,:)=[estimate_error,p,q,r,timeTotal,Srcc]';
%             end
%         end
%         R(j,:,:)=a;
%     end
%     B(:,1)=mean(R(:,:,1)',2);%%%每一行的均值
%     B(:,2)=mean(R(:,:,2)',2);
%     B(:,3)=mean(R(:,:,3)',2);
%     B(:,4)=mean(R(:,:,4)',2);
%     B(:,5)=mean(R(:,:,5)',2);
%     B(:,6)=mean(R(:,:,6)',2);
%     A{k}=B;
% end
% a1=A{1};
% a2=A{2};
% a3=A{3};
% xlswrite('Lap_sparse.xls', a1,'Lap1000');
% xlswrite('Lap_sparse.xls', a2,'Lap2000');
% xlswrite('Lap_sparse.xls', a3,'Lap5000');
% disp('finished');
% 
% 
% clear all;clc;
% %%%拉普拉斯分布
% dims=20;
% randseed=10;
% R = zeros(randseed,dims,6);%%%表示randseed个dims*4个矩阵
% B=zeros(dims,6);
% Sam=[1000;2000;5000];
% Dim=[10;20;30;40;50;60;70;80;90;100;110;120;130;140;150;160;170;180;190;200];
% A=cell(1,3);
% for k=1:3
%     for j=1:randseed
%         j
%         a=zeros(dims,6);
%         for i=1:dims
%             i
%             [kkk,kest,estimate_error,number_connect_edge,true_connection_identified,true_absence_identified,ntruepruned,nsuperfluous,timeTotal,Srcc]=testlingam(Dim(i),j,Sam(k),4,'A');
%             if true_connection_identified+ntruepruned==0||true_connection_identified+nsuperfluous==0
%                 a(i,:)=[estimate_error,0,0,0,timeTotal,Srcc]';
%             else
%                 p=true_connection_identified/(true_connection_identified+ntruepruned);
%                 q=true_connection_identified/(true_connection_identified+nsuperfluous);
%                 r=2*(p*q)/(p+q);
%                 a(i,:)=[estimate_error,p,q,r,timeTotal,Srcc]';
%             end
%         end
%         R(j,:,:)=a;
%     end
%     B(:,1)=mean(R(:,:,1)',2);%%%每一行的均值
%     B(:,2)=mean(R(:,:,2)',2);
%     B(:,3)=mean(R(:,:,3)',2);
%     B(:,4)=mean(R(:,:,4)',2);
%     B(:,5)=mean(R(:,:,5)',2);
%     B(:,6)=mean(R(:,:,6)',2);
%     A{k}=B;
% end
% a1=A{1};
% a2=A{2};
% a3=A{3};
% xlswrite('Lap_full.xls', a1,'Lap1000');
% xlswrite('Lap_full.xls', a2,'Lap2000');
% xlswrite('Lap_full.xls', a3,'Lap5000');
% disp('finished');
% 
% 
% %%
% clear all;clc;
% %%%均匀分布
% dims=20;
% randseed=10;
% R = zeros(randseed,dims,6);%%%表示randseed个dims*4个矩阵
% B=zeros(dims,6);
% Sam=[1000;2000;5000];
% Dim=[10;20;30;40;50;60;70;80;90;100;110;120;130;140;150;160;170;180;190;200];
% A=cell(1,3);
% for k=1:3
%     for j=1:randseed
%         j
%         a=zeros(dims,6);
%         for i=1:dims
%             i
%             [kkk,kest,estimate_error,number_connect_edge,true_connection_identified,true_absence_identified,ntruepruned,nsuperfluous,timeTotal,Srcc]=testlingam(Dim(i),j,Sam(k),2,'F');
%             if true_connection_identified+ntruepruned==0||true_connection_identified+nsuperfluous==0
%                 a(i,:)=[estimate_error,0,0,0,timeTotal,Srcc]';
%             else
%                 p=true_connection_identified/(true_connection_identified+ntruepruned);
%                 q=true_connection_identified/(true_connection_identified+nsuperfluous);
%                 r=2*(p*q)/(p+q);
%                 a(i,:)=[estimate_error,p,q,r,timeTotal,Srcc]';
%             end
%         end
%         R(j,:,:)=a;
%     end
%     B(:,1)=mean(R(:,:,1)',2);%%%每一行的均值
%     B(:,2)=mean(R(:,:,2)',2);
%     B(:,3)=mean(R(:,:,3)',2);
%     B(:,4)=mean(R(:,:,4)',2);
%     B(:,5)=mean(R(:,:,5)',2);
%     B(:,6)=mean(R(:,:,6)',2);
%     A{k}=B;
% end
% a1=A{1};
% a2=A{2};
% a3=A{3};
% xlswrite('Uni_sparse.xls', a1,'Uni1000');
% xlswrite('Uni_sparse.xls', a2,'Uni2000');
% xlswrite('Uni_sparse.xls', a3,'Uni5000');
% disp('finished');
% 
% 
% clear all;clc;
% %%%均匀分布
% dims=20;
% randseed=10;
% R = zeros(randseed,dims,6);%%%表示randseed个dims*4个矩阵
% B=zeros(dims,6);
% Sam=[1000;2000;5000];
% Dim=[10;20;30;40;50;60;70;80;90;100;110;120;130;140;150;160;170;180;190;200];
% A=cell(1,3);
% for k=1:3
%     for j=1:randseed
%         j
%         a=zeros(dims,6);
%         for i=1:dims
%             i
%             [kkk,kest,estimate_error,number_connect_edge,true_connection_identified,true_absence_identified,ntruepruned,nsuperfluous,timeTotal,Srcc]=testlingam(Dim(i),j,Sam(k),4,'F');
%             if true_connection_identified+ntruepruned==0||true_connection_identified+nsuperfluous==0
%                 a(i,:)=[estimate_error,0,0,0,timeTotal,Srcc]';
%             else
%                 p=true_connection_identified/(true_connection_identified+ntruepruned);
%                 q=true_connection_identified/(true_connection_identified+nsuperfluous);
%                 r=2*(p*q)/(p+q);
%                 a(i,:)=[estimate_error,p,q,r,timeTotal,Srcc]';
%             end
%         end
%         R(j,:,:)=a;
%     end
%     B(:,1)=mean(R(:,:,1)',2);%%%每一行的均值
%     B(:,2)=mean(R(:,:,2)',2);
%     B(:,3)=mean(R(:,:,3)',2);
%     B(:,4)=mean(R(:,:,4)',2);
%     B(:,5)=mean(R(:,:,5)',2);
%     B(:,6)=mean(R(:,:,6)',2);
%     A{k}=B;
% end
% a1=A{1};
% a2=A{2};
% a3=A{3};
% xlswrite('Uni_full.xls', a1,'Uni1000');
% xlswrite('Uni_full.xls', a2,'Uni2000');
% xlswrite('Uni_full.xls', a3,'Uni5000');
% disp('finished');
% 
% 
% 
% %%
% clear all;clc;
% %%%高斯分布
% dims=20;
% randseed=10;
% R = zeros(randseed,dims,6);%%%表示randseed个dims*4个矩阵
% B=zeros(dims,6);
% Sam=[1000;2000;5000];
% Dim=[10;20;30;40;50;60;70;80;90;100;110;120;130;140;150;160;170;180;190;200];
% A=cell(1,3);
% for k=1:3
%     for j=1:randseed
%         j
%         a=zeros(dims,6);
%         for i=1:dims
%             i
%             [kkk,kest,estimate_error,number_connect_edge,true_connection_identified,true_absence_identified,ntruepruned,nsuperfluous,timeTotal,Srcc]=testlingam(Dim(i),j,Sam(k),2,'D');
%             if true_connection_identified+ntruepruned==0||true_connection_identified+nsuperfluous==0
%                 a(i,:)=[estimate_error,0,0,0,timeTotal,Srcc]';
%             else
%                 p=true_connection_identified/(true_connection_identified+ntruepruned);
%                 q=true_connection_identified/(true_connection_identified+nsuperfluous);
%                 r=2*(p*q)/(p+q);
%                 a(i,:)=[estimate_error,p,q,r,timeTotal,Srcc]';
%             end
%         end
%         R(j,:,:)=a;
%     end
%     B(:,1)=mean(R(:,:,1)',2);%%%每一行的均值
%     B(:,2)=mean(R(:,:,2)',2);
%     B(:,3)=mean(R(:,:,3)',2);
%     B(:,4)=mean(R(:,:,4)',2);
%     B(:,5)=mean(R(:,:,5)',2);
%     B(:,6)=mean(R(:,:,6)',2);
%     A{k}=B;
% end
% a1=A{1};
% a2=A{2};
% a3=A{3};
% xlswrite('Nor_sparse.xls', a1,'Nor1000');
% xlswrite('Nor_sparse.xls', a2,'Nor2000');
% xlswrite('Nor_sparse.xls', a3,'Nor5000');
% disp('finished');


clear all;clc;
%%%高斯分布
dims=20;
randseed=10;
R = zeros(randseed,dims,6);%%%表示randseed个dims*4个矩阵
B=zeros(dims,6);
Sam=[1000;2000;5000];
Dim=[10;20;30;40;50;60;70;80;90;100;110;120;130;140;150;160;170;180;190;200];
A=cell(1,3);
for k=1:3
    for j=1:randseed
        j
        a=zeros(dims,6);
        for i=1:dims
            i
            [kkk,kest,estimate_error,number_connect_edge,true_connection_identified,true_absence_identified,ntruepruned,nsuperfluous,timeTotal,Srcc]=testlingam(Dim(i),j,Sam(k),4,'D');
            if true_connection_identified+ntruepruned==0||true_connection_identified+nsuperfluous==0
                a(i,:)=[estimate_error,0,0,0,timeTotal,Srcc]';
            else
                p=true_connection_identified/(true_connection_identified+ntruepruned);
                q=true_connection_identified/(true_connection_identified+nsuperfluous);
                r=2*(p*q)/(p+q);
                a(i,:)=[estimate_error,p,q,r,timeTotal,Srcc]';
            end
        end
        R(j,:,:)=a;
    end
    B(:,1)=mean(R(:,:,1)',2);%%%每一行的均值
    B(:,2)=mean(R(:,:,2)',2);
    B(:,3)=mean(R(:,:,3)',2);
    B(:,4)=mean(R(:,:,4)',2);
    B(:,5)=mean(R(:,:,5)',2);
    B(:,6)=mean(R(:,:,6)',2);
    A{k}=B;
end
a1=A{1};
a2=A{2};
a3=A{3};
xlswrite('Nor_full.xls', a1,'Nor1000');
xlswrite('Nor_full.xls', a2,'Nor2000');
xlswrite('Nor_full.xls', a3,'Nor5000');
disp('finished');
end

