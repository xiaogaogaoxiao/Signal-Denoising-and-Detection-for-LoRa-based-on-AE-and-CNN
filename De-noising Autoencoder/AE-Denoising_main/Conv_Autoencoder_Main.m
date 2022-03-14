%%=======================================================================%%
% This is the main function: LoRa Decoder:CNN based on Spectrogram of the %
% transmitted signal as Input.                                            %
%========================================================================%%
rng shuffle
clear
clc
%--------------------------------------------%
% Parametres de la modulation LoRa           %
%--------------------------------------------%
SF = 7;
M = 2^SF;
vec_snr =-16:2:-2;
Lambda=1; % 1 ; 0.25; 0.7;
%------------------------------------------------------------------------%%
% Autoencoders   
%------------------------------------------------------------------------%%
net = cell(size(vec_snr));
info = cell(size(vec_snr));
ind_snr = 0;
for SNR_dB = vec_snr
    disp(SNR_dB)
    ind_snr = ind_snr + 1;
    
    %% ConvAutoencoder APPRENTISSAGE
    %%
    layers = f_denoising_auto(M);
    
    %% Training and validation set
    %%
   % fname1 = ['MUse_AE_Test_Data_SNR =' num2str(SNR_dB) '_Lambda=' num2str(Lambda) '.mat'];
    fname1 = ['MUse_AE_Train_Data_SNR =' num2str(SNR_dB) '_Lambda=' num2str(Lambda) '.mat'];
    load(fname1)
    %load('MUse_AE_Train_Data_SNR =-6.mat')
    
    index = floor(0.9*length(target_sym));
    perm_vect = randperm(length(target_sym));
    X = X_img_inverted(:,:,:,perm_vect);
    
    X_train = X(:,:,:,1:index);
    X_val = X(:,:,:,index+1:end);
    
    %% Noiseless dataset
    %%
    load('Noiseless_Train_Data_SNR =10.mat')
    
    X_noiseless = X_noiseless(:,:,:,perm_vect);
    
    target_train = X_noiseless(:,:,:,1:index);
    target_val = X_noiseless(:,:,:,index+1:end);
     
    %%---------------------------------------%%
    % set the training options                %
    %%---------------------------------------%%
    opts = trainingOptions('adam',... 'sgdm' 'rmsprop'
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',30,...
        'Plots', 'training-progress', ...
        'MiniBatchSize',1e2, ...
        'ValidationData', {X_val, target_val},...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Shuffle', 'every-epoch');
    %%---------------------------------------%%
    % Train the network                       %
    %%---------------------------------------%%
    [net{ind_snr}, info{ind_snr}] = trainNetwork(X_train, target_train, layers, opts);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load('MUse_AE_Test_Data_SNR =-6.mat')
    fname2 = ['MUse_AE_Test_Data_SNR =' num2str(SNR_dB) '_Lambda=' num2str(Lambda) '.mat'];
%     fname2 = ['MUse_AE_Train_Data_SNR =' num2str(SNR_dB) '_Lambda=' num2str(Lambda) '.mat'];
    load(fname2)
    
    target_sym_train = categorical(target_sym(perm_vect));
    
    X_img_inverted = X_img_inverted(:,:,:,perm_vect);
    
    X_denoised = predict(net{ind_snr}, X_img_inverted);
    
   %fname3 = ['X_denoised_test_SNR =' num2str(SNR_dB) '_Lambda=' num2str(Lambda) '.mat'];
   fname3 = ['X_denoised_train_SNR =' num2str(SNR_dB) '_Lambda=' num2str(Lambda) '.mat'];
   save(fname3,'X_denoised','target_sym_train', '-v7.3')
    
end

%% test
%%
% load('MUse_AE_Test_Data_SNR =-6.mat')
% target_sym_train = categorical(target_sym(perm_vect));
% 
% X_img_inverted = X_img_inverted(:,:,:,perm_vect);

%% Test
%%

% X_denoised = predict(net{ind_snr}, X_img_inverted);


% figure (1)
% imshow(X_img_inverted(:,:,1,5));
% 
% figure (2)
% imshow(X_denoised(:,:,1,5));
% 
% figure (3)
% imshow(X_noiseless(:,:,1,5));






