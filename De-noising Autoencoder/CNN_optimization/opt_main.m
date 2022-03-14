rng shuffle
clear
clc

M = 128;
SNR = -8;
Lambda = 1; %0.25; 0.7;

fname1 = ['X_denoised_Opt_SNR =' num2str(SNR) '_Lambda=' num2str(Lambda)  '.mat'];
load(fname1)

index = floor(0.9*length(target_sym_train));

%% Training samples
%%
X_train = X_denoised(:,:,:,1:index);
target_train = target_sym_train(1:index);

%% Validation samples
%%
X_val = X_denoised(:,:,:,index+1:end);
target_val = target_sym_train(index+1:end);

options = trainingOptions('adam',...%sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',20, ...
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
nfL = optimizableVariable('nfL',[0,1],'Type','integer'); % # of additional fully connected layers

nf1 = optimizableVariable('nf1',[2,100],'Type','integer');
nf2 = optimizableVariable('nf2',[2,100],'Type','integer');
nf3 = optimizableVariable('nf3',[2,100],'Type','integer');

fz1 = optimizableVariable('fz1',[2,20],'Type','integer');
fz2 = optimizableVariable('fz2',[2,20],'Type','integer');
fz3 = optimizableVariable('fz3',[2,20],'Type','integer');

fcz1 = optimizableVariable('fcz1',[2,1000],'Type','integer');
fcz2 = optimizableVariable('fcz2',[2,1000],'Type','integer');

fun = @(x)f_ObjFct(X_train,target_train,M,options,x.ncL,x.nfL,x.nf1,x.nf2,x.nf3,x.fz1,x.fz2,x.fz3,x.fcz1,x.fcz2);
results = bayesopt(fun,[ncL,nfL,nf1,nf2,nf3,fz1,fz2,fz3,fcz1,fcz2],'MaxObjectiveEvaluations', 40);%,'UseParallel',true);

