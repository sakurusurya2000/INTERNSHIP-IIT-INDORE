close all;
clc;
clear all;
%% Loading Speech Signal 
wlen = 1024;
hop = wlen/4;
nfft = 4096;
[x, fs] = audioread('10.mp3');
N = length(x);
tn = (0:N-1)/fs;
%% Plot the speech Signal in Time Domain
figure(1)
plot(tn,x, 'b');
title('Trended signal')
%% Detrend the Speech Signal.
xn= detrend(x);
figure(2)
plot(tn, xn ,'r');
title('De-trended signal') 
%% Plot the Power Spectral Density of Speech Signal
[pxx,fx] = pwelch(xn,wlen,hop,nfft,fs);
figure(3)
plot(fx,pxx);
title('Power Spectral Density')
%% Design a Banpass IIR Filter 
N = 7;
F1 = 200; F2=250; 
h = designfilt('bandpassfir','FilterOrder',20,'CutoffFrequency1',F1,'CutoffFrequency2',F2,'SampleRate',fs);
%% Apply the filter to Smooth out the Speech Signal 
xfilter = filter(h,xn);

%% Visualize PSD of the Speech signal before and after Filtering 
[pff,ff] = pwelch(xfilter,wlen,hop,nfft,fs); 
figure(4)
plot(fx,pxx)
hold on;
grid on;
plot(ff,pff);
hold on;
legend({'Before Filtering','After Filtering'});
set(gcf,'NumberTitle','Off', 'Name','Before Filtering vs. After Filtering');
%% Overlay the filtered signal on the original Speech signal. 
% Filtered signal is delayed
figure(5);
plot(tn, xn, 'b');
hold on;
grid on;
plot(tn, xfilter,'r');
hold on;
legend({'Original Signal','Filtered Signal'});
set(gcf,'NumberTitle','Off', 'Name','Filtered Signal vs. Actual Signal');
%% SPECTROGRAM
xfilter = xfilter(:, 1);
xlen = length(xfilter);
a=fft(x);
[S, f, t] = spectrogram(xfilter, wlen, hop, nfft, fs);
K = sum(hamming(wlen,'periodic'));
S = abs(S)/K;
if rem(nfft, 2)                     
    S(2:end, :) = S(2:end, :).*2;
else                                
    S(2:end-1, :) = S(2:end-1, :).*2;
end
S = 20*log10(S + 1e-6);
figure(6)
surf(t, f, S)
shading interp
axis tight
box on
view(0, 90)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
xlabel('Time, s')
ylabel('Frequency, Hz')
title('Amplitude spectrogram of the signal')

handl = colorbar;
set(handl, 'FontName', 'Times New Roman', 'FontSize', 14)
ylabel(handl, 'Magnitude, dB')
audiowrite('10-1.wav',xfilter,fs);
%% Methods-FFT
X=fft(x);
figure(7)
plot(real(X));
%% Methods-STFT
[S1, f1, t1] = stft(x, wlen, hop, nfft, fs);
K = sum(hamming(wlen, 'periodic'))/wlen;
S1 = abs(S1)/wlen/K;

if rem(nfft, 2)                    
    S1(2:end, :) = S1(2:end, :).*2;
else                                
    S1(2:end-1, :) = S1(2:end-1, :).*2;
end

S1 = 20*log10(S1 + 1e-6);

figure(8)
surf(t1, f1, S1)
shading interp
xlabel('Time, s')
ylabel('Frequency, Hz')
zlabel('Magnitude, dB')
title('Amplitude spectrogram of the signal')
hold on;
%% Method-Wavelet
