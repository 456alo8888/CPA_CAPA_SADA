clc
addpath('.\LiNGAM_DataGenerator');
addpath(genpath('.\LiNGAM'));
rand('seed', 0);
noisetype='uniform';
nNode=8;
nIndgree=1.25;
nSample=200;
noiseratio=0.3;
nFreeNode=ceil(nIndgree)+2;

%%runing paramters%
splitTries=1;%used just for test
%splitTries=(2*nIndgree+2)^2 %suggested, especially for the large scale probelm 

%% generate data
skeleton = GenRandSkeleton(nNode, nFreeNode, nIndgree);
data=SEMDataGenerator(skeleton,nSample, noisetype, noiseratio);

%%run SADA
E = SADA_Main( data, splitTries)

%%stat results
[T,P, TP,FP, WrongDirection, recall, precision]=SADA_Stat(skeleton, E)

