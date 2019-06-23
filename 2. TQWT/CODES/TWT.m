function[w]=TWT(x)
 
Q = 1; r = 3; J = 8; % TQWT parameters
N = length(x); % Length of test signal
w = tqwt_radix2(x,Q,r,J); % TQWT
for j=1:9
    w{1,j}=meanfreq(w{1,j});
end
% y = itqwt_radix2(w,Q,r,N); % Inverse TQWT
% recon_err = max(abs(x' - y)); % Reconstruction error
% fs = 1;
% figure(1);
% PlotSubbands(x,w,Q,r,1,J+1,fs);
% figure(2);
% PlotSubbands(x,w,Q,r,1,J+1,fs,'E','stem');
% E= sum(x.^2);
