%%-----------------------------------------------------------------------%%
% This code generates the Noiseless training samples. The training  samples %
% are set of images of the spectrogram of the transmit symbols(0,...,M-1) %
% denoted by 'X_img_noiseless' and the symbols 'target_sym_noiseless'.                                %
%%-----------------------------------------------------------------------%%
rng shuffle
B = 250e3; %B
SF = 7; %SF
M = 2^SF;
vec_snr = 10;
%-- param induits
T=1/B;
Ts = M/B;
MC = 40;%60;%50;%100;% MC_nn;

target_sym_noiseless = zeros (M*MC,1);
X_img_noiseless = uint8(zeros(2*M,2*M,1,MC)); % cast for the image representation

up_f = 1;

m =0:1:M-1;
ind_snr=0;

for SNR_dB = vec_snr
    disp(SNR_dB)
    ind_snr = ind_snr + 1;
%m = 84;%0:1:M-1;
for j=1:M
    %disp(m(j)) 
    t0 = (-M/2:M/2-1-m(j))*T;
    t1 = (M/2-1-m(j)+1:M/2-1)*T;
    
    phi0 = 2*pi*(B/(2*Ts)*t0.^2 + (m(j)/Ts)*t0);
    phi1 = 2*pi*(B/(2*Ts)*t1.^2 + (m(j)/Ts-B)*t1);
    
    phi = [phi0, phi1];
    s = exp(1i*phi);
    
    Pb_dBm = -174 + 10*log10(B) + 6; %noise factor = 6 dB
    Ps_dBm = SNR_dB + Pb_dBm;
    Pb = 10^(Pb_dBm/10-3);
    Ps = 10^(Ps_dBm/10-3);
    h = sqrt(Ps);%(randn(1) + 1i*randn(1));   
    
    for i =1:MC
         disp(['Symbol ' int2str(m(j)) ' and MC = ' int2str(i)])
        %w = sqrt(Pb/2)*(randn(1, M) + 1i*randn(1, M));
        r = h*s;% + w;
       
        spectrogram( r,(M*up_f)/4,(M*up_f)/4-1,(M*up_f)/4,B,'yaxis');        
        set(gca,'Units','Normalize','Position',[0 0 1 1]);
        set(gcf,'Position',[0 0 2*M 2*M]);
        colormap(gray);
        set(gca, 'Visible', 'off');
        set(gcf, 'Visible', 'off');
        colorbar('off');
        
        F = getframe(gca);
        
        X_img_noiseless(:,:,1,(j-1)*MC+i) = rgb2gray(frame2im(F));        
        target_sym_noiseless((j-1)*MC+i)  = m(j);
        
        
    end
end


fname= ['Noiseless_Train_Data_SNR =' num2str(SNR_dB) '.mat'];
save(fname,'target_sym_noiseless', 'X_img_noiseless', '-v7.3')

end
    
% figure
% imshow(X_img_noiseless(:,:,1,108));
% after that, X needs to be inverted then binarized (imcomplement, imbinarize)
