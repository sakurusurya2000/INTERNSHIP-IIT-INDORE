clear all;
clc;
close all;
[x, fs] = audioread('1.mp3');
wlen = 1024;
hop = wlen/4;
nfft = 4096;
x = x(:, 1);
xlen = length(x);
a=fft(x);
[S, f, t] = spectrogram(x, wlen, hop, nfft, fs);

s = pwelch(x);
plot(10*log10(s))
drawnow;
sound(x, Fs);

K = sum(hamming(wlen,'periodic'));
S = abs(S)/K;

if rem(nfft, 2)                     
    S(2:end, :) = S(2:end, :).*2;
else                                
    S(2:end-1, :) = S(2:end-1, :).*2;
end

S = 20*log10(S + 1e-6);

figure(1)
surf(t, f, S)
shading interp
xlabel('Time, s')
ylabel('Frequency, Hz')
zlabel('Magnitude, dB')
title('Amplitude spectrogram of the signal')
hold on;