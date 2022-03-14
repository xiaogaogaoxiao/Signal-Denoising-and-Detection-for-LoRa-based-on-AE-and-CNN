rng shuffle
clear
clc

M = 128;
SNR = -8;
Lambda = 1; %0.25; 0.7;

fname1 = ['MUse_AE_Opt_Data_SNR =' num2str(SNR) '_Lambda=' num2str(Lambda)  '.mat'];
load(fname1)

perm_vect = randperm(length(target_sym));
X_img_inverted = X_img_inverted(:,:,:,perm_vect);
index = floor(0.9*length(target_sym));

%% Training and validation samples
%%
X_train = X_img_inverted(:,:,:,1:index);
X_val = X_img_inverted(:,:,:,index+1:end);

%% Noiseless samples
%%
load('Noiseless_Train_Data_SNR =10.mat')
X_noiseless = X_noiseless(:,:,:,perm_vect);


target_train = X_noiseless(:,:,:,1:index);
target_val = X_noiseless(:,:,:,index+1:end);

options = trainingOptions('adam',...%sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',30, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',40, ...
    'LearnRateDropFactor',0.1, ...
    'MiniBatchSize',10, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','none', ...%'training-progress', ...
    'ValidationData',{X_val,target_val}, ...
    'ValidationFrequency',50);

ncL = optimizableVariable('ncL',[0,2],'Type','integer'); % # of additional convolutional layers

nf1 = optimizableVariable('nf1',[2,100],'Type','integer');
nf2 = optimizableVariable('nf2',[2,100],'Type','integer');
nf3 = optimizableVariable('nf3',[2,100],'Type','integer');
nf4 = optimizableVariable('nf4',[2,100],'Type','integer');

fz1 = optimizableVariable('fz1',[2,20],'Type','integer');
fz2 = optimizableVariable('fz2',[2,20],'Type','integer');
fz3 = optimizableVariable('fz3',[2,20],'Type','integer');
fz4 = optimizableVariable('fz4',[2,20],'Type','integer');

fun = @(x)f_ObjFct_AE(X_train,target_train,M,options,x.ncL,x.nf1,x.nf2,x.nf3,x.nf4,x.fz1,x.fz2,x.fz3,x.fz4);
results = bayesopt(fun,[ncL,nf1,nf2,nf3,nf4,fz1,fz2,fz3,fz4],'MaxObjectiveEvaluations', 40);%,'UseParallel',true);

