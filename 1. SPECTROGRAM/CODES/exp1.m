clc;
clear all;
close all;
[x, fs] = audioread('1.mp3');
wlen = 1024;
hop = wlen/4;
nfft = 4096;
x = x(:, 1);
xlen = length(x);
[S, f, t] = spectrogram(x, wlen, hop, nfft, fs);
K = sum(rectwin(wlen));
S = abs(S)/K;

if rem(nfft, 2)                    
    S(2:end, :) = S(2:end, :).*2;
else                                
    S(2:end-1, :) = S(2:end-1, :).*2;
end

S = 20*log10(S);

figure(1)
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