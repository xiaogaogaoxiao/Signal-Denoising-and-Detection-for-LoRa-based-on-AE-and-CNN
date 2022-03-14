%%-----------------------------------------------------------------------%%
% This code generates the training samples. The training  samples     %
% are set of images of the spectrogram of the transmit symbols(0,...,M-1) %
% denoted by 'X_img' and the symbols 'target_sym'.                                %
%%-----------------------------------------------------------------------%%
%function [X_cell, target, X]= generate_spectro_multiple(SF,MC,SNR,N_I,Delay_range,ov_f,Frame_length,t)
rng shuffle
clear
clc %close all
%--------------------------------------------%
% Parametres de la modulation LoRa           %
%--------------------------------------------%
B = 250e3;
SF = 7;
%-------------------------------------------%
% param induits                             %
%-------------------------------------------%
ov_f = 1;
T = 1/B/ov_f;
M = 2^SF;
Ts = M/B;
t = -Ts/2:T:Ts/2-T;
N = length(t);
N_frame = 3;
MC = 60; % 60 for training and test, 40 for the optimization

Frame_length = N_frame*N;
Delay_range = (N_frame-1)*N;

vec_ind_centered = (-M/2:M/2-1);
decod = exp(-1i*2*pi*(B/(2*Ts)*(vec_ind_centered*T).^2));

dmin = 2e2; % m
dmax = 1e3; % m
N_use = 1;
Lambda = 0.7;%0.25;1; 
N_I = poissrnd(Lambda);
Nu = N_use + N_I;
Ns_u = 1; %number of symbols of useful user
Ns_i = 1; %number of symbols of interferer user
Np = 0; %number of  preambles

m_Nu_all = 0:1:M-1; %randi([0 M-1], N_use,Ns_u); % symbols of useful user
s = zeros(Nu,Frame_length);

vec_snr =-2;%-16:2:-4;

target_sym = zeros (M*MC,1);
X_img = uint8(zeros(2*M,2*M,1,MC)); % cast for the image representation
ind_snr = 0;

for SNR_dB = vec_snr
    disp(SNR_dB)
    ind_snr = ind_snr + 1;
    for j = 1:M
        disp(j)
        %disp(['Symbol ' int2str(m_Nu_all(j))])
        m_Nu = m_Nu_all(j) ;
        Delay = randi(Delay_range,Nu,1);
        m_Ni = randi([0 M-1], N_I,Ns_i); % symbols of interferer user
        
        %% CSS of useful user
        %%
        s(1,Delay(1)+1:Delay(1)+ N) = codeCSS(m_Nu,B,Ts,t,1,ov_f);
        
        %% CSS of interferer user
        %%
        for i=1:N_I
           s(i+1,Delay(i+1)+1:Delay(i+1)+ Ns_i*N) = codeCSS(m_Ni(i,:),B,Ts,t,Ns_i,ov_f);
        end
        
        Pb_dBm = -174 + 10*log10(B) + 6; %noise factor = 6 dB
        Ps_dBm = SNR_dB + Pb_dBm;
        Pb = 10^(Pb_dBm/10-3);
        Ps = 10^(Ps_dBm/10-3);
        
        h_i = zeros(N_I,1);
        
        %% Received power of the interferer user
        %%
        for ii = 1:N_I
            Pi_dBm = f_interf_power(dmin, dmax);
            Pi = 10^(Pi_dBm/10-3);
            h_i(ii) = sqrt(Pi); % channel of interferer user
        end
        h_u = sqrt(Ps); % channel of useful user
        h = [h_u; h_i]; % combined channel
        
        
        for ii =1:MC
            
            w = sqrt(Pb/2)*(randn(1, Frame_length) + 1i*randn(1, Frame_length));
            r = sum(h*ones(1,Frame_length).*s,1)  + w;
            
            %% De-chirping
            %%
            r = reshape(r(Delay(1)+ N*Np + 1:ov_f:N*Np + Delay(1)+N*Ns_u),Ns_u,N/ov_f);            
            
            spectrogram(r, M/4, M/4-1, M/4, B,'yaxis');
            set(gca,'Units','Normalize','Position',[0 0 1 1]);
            set(gcf,'Position',[0 0 2*M 2*M]);
            colormap(gray);
            set(gca, 'Visible', 'off');
            set(gcf, 'Visible', 'off');
            colorbar('off');
            
            F = getframe(gca);
            
            X_img(:,:,1,(j-1)*MC+ii) = rgb2gray(frame2im(F));
            target_sym((j-1)*MC+ii)  = m_Nu;
            
        end
        
    end
    
    
end

% fname= ['MUse_AE_Opt_Data_SNR =' num2str(SNR_dB) '_Lambda=' num2str(Lambda) '.mat']; % MC = 40
fname= ['MUse_AE_Train_Data_SNR =' num2str(SNR_dB) '_Lambda=' num2str(Lambda) '.mat'];
% fname= ['MUse_AE_Test_Data_SNR =' num2str(SNR_dB) '_Lambda=' num2str(Lambda) '.mat'];
save(fname,'X_img','target_sym', '-v7.3')

% figure
% imshow(X_img(:,:,1,1));
% after that, X needs to be inverted (imcomplement)
