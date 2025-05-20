clc;clear;addpath('.\dataset');addpath(genpath('.\SADA'));data_url = {'Alarm','andes','diabetes','Link'};
dataID = 2; % choose graph
% load Alarm_rs_500
load Andes_rs_500
% load Diabetes_rs_500
% load Link_rs_500
r_s = save_rs{3};
skeleton = readRnet(['.\dataset\',data_url{dataID},'.net']);skeleton = sortskeleton(skeleton);
%----------------------------- ground truth -------------------------------------
[C,A,B,~,~] = CPSC_skeleton(skeleton);
subplot(1,2,1)
w = plot(graph(skeleton,'upper'));
highlight(w,B,'NodeColor','k','Marker','o'); 
highlight(w,A,'NodeColor','b','Marker','^');
highlight(w,C,'NodeColor','r','Marker','d');
%------------------------------------------------
[C,A,B,~] = CPA(1,r_s);
subplot(1,2,2)
m = plot(graph(skeleton,'upper'));
highlight(m,B,'NodeColor','k','Marker','o'); 
highlight(m,A,'NodeColor','b','Marker','^');
highlight(m,C,'NodeColor','r','Marker','d');