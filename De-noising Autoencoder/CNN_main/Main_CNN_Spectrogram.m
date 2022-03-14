%%=======================================================================%%
% This is the main function: LoRa Decoder:CNN based on Spectrogram of the %
% transmitted signal as Input.                                            %
%========================================================================%%
rng shuffle
clear
clc
close all

%--------------------------------------------%
% Parametres de la modulation LoRa           %
%--------------------------------------------%
B = 250e3;
SF = 7;
M = 2^SF;
vec_snr = -16:2:-4;
MC= 300;%5*60;
Lambda=1;%0.25; 0.7;
%------------------------------------------------------------------------%%
% Solution CNN:                                                           %
%------------------------------------------------------------------------%%
layers = f_CNN(M);

%% Train
%%
Nlora = M*MC;
net = cell(size(vec_snr));
info = cell(size(vec_snr));

ind_snr = 0;
for SNR = vec_snr
    disp(SNR)
    ind_snr = ind_snr + 1;
    
    fname1 = ['X_denoised_train_SNR =' num2str(SNR) '_Lambda=' num2str(Lambda)  '.mat'];
    load(fname1)
    
    index = floor(0.9*length(target_sym_train));
    X_train = X_denoised(:,:,:,1:index);
    target_train = target_sym_train(1:index);
    
    X_val = X_denoised(:,:,:,index+1:end);
    target_val = target_sym_train(index+1:end);
    
    
    %% CNN APPRENTISSAGE
    %%
    
    %%---------------------------------------%%
    % set the training options                %
    %%---------------------------------------%%
    opts = trainingOptions('adam' , ...
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',25,...
        'Plots', 'training-progress', ...
        'MiniBatchSize',1e2, ...
        'ValidationData', {X_val, target_val}, ...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Shuffle', 'every-epoch');
    %%---------------------------------------%%
    % Train the network                       %
    %%---------------------------------------%%
    [net{ind_snr}, info{ind_snr}] = trainNetwork(X_train, target_train, layers, opts);
    
    
end

%%------------------------------------------------------%%
% Test                                                   %
%%------------------------------------------------------%%

Nlora_test = Nlora;
SER_cnn_AE= zeros(size(vec_snr));
N_err = zeros(size(vec_snr));
ind_snr = 0;
for SNR = vec_snr
    disp(SNR)
    ind_snr = ind_snr + 1;
    
    fname2 = ['X_denoised_test_SNR =' num2str(SNR) '_Lambda=' num2str(Lambda) '.mat'];
    load(fname2)
    
    X_test = X_denoised;
    target_test = target_sym_train;
    
    target_dec = (classify(net{ind_snr}, X_test));%-1;
    SER_cnn_AE(ind_snr) =  nnz(target_dec ~= target_test)/ Nlora_test;
end

figure
semilogy(vec_snr, SER_cnn_AE, 'ko-.');
grid on
legend('ConvAE + CNN-based')
%title('SER, SF=7, B=250kHz')

%fname3 = ['SER_lambda=' num2str(Lambda) '_all_SNR.mat'];
% save(fname3, '-v7.3')



figure
semilogy(vec_snr, SER_classi_multi, 'rv-', 'LineWidth',1.2);
hold on
semilogy(vec_snr, SER_cnn_multi, 'ks-','LineWidth',1.2);
hold on;
semilogy(vec_snr, SER_cnn_AE, 'bo-','LineWidth',1.2);
grid on
legend('Classical LoRa','CNN-based [4]','AE-CNN-based')
xlabel('SNR (dB)')
ylabel('SER')
