%%%%%%%%%%%%%%%%%%%%%%%%%%%% codeCSS2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [X]=codeCSS(D,B,Ts,t,N_tot,ov_f)

N=length(t);
X=zeros(1,N_tot*N);

for idx=1:N_tot
    m=D(idx);
    
    %%Intantaneous Frequency
    
    f=zeros(size(t));
    f(1:N-(m*ov_f))= B/Ts*t(1:N-(m*ov_f))/2+m/Ts;
    f(N-(m*ov_f)+1:N)= B/Ts*t(N-(m*ov_f)+1:N)/2+m/Ts-B;    
    X((idx-1)*N+1:idx*N)=exp(2*1i*pi*f.*t);

end
