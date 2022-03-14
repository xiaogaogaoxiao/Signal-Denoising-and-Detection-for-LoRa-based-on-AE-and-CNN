
function interf_power = f_interf_power(dmin, dmax) %[xy, interf_power] = f_interf_power(dmin, dmax)

tr_power = 14; % transmit power that corresponds to SF=7, in dBm

cf_Mhz = 868;
lambda = 3e8/(cf_Mhz*1e6);
gamma = 4;

d2 = 0;
while d2>=dmax^2 || d2<=dmin^2
    
      xy = 2*dmax*rand(1,2);                
      d2 = (xy(1)-dmax)^2+(xy(2)-dmax)^2;  
      
end

interf_power = tr_power - 20*log10((4*pi*d2^(gamma/4))/lambda);