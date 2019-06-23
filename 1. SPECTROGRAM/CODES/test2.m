clc
clear all
close all
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

K = sum(hamming(wlen,'periodic'));
S = abs(S)/K;

if rem(nfft, 2)                     
    S(2:end, :) = S(2:end, :).*2;
else                                
    S(2:end-1, :) = S(2:end-1, :).*2;
end

S = 20*log10(S + 1e-6);
Fn = fs/2;                                                  % Nyquist Frequency (Hz)
Wp = [futcutlow fcuthigh]/Fn;                               % Passband Frequency (Normalised)
Ws = [futcutlow*0.95 fcuthigh/0.95]/Fn;                     % Stopband Frequency (Normalised)
Rp =   1;                                                   % Passband Ripple (dB)
Rs = 150;                                                   % Stopband Ripple (dB)
[n,Wn] = buttord(Wp,Ws,Rp,Rs);                              % Filter Order
[z,p,k] = butter(n,Wn);                                     % Filter Design
[sosbp,gbp] = zp2sos(z,p,k);                                % Convert To Second-Order-Section For Stability
freqz(sosbp, 2^20, fs)                                      % Filter Bode Plot
filtered_signal = filtfilt(sosbp, gbp, signal);

figure(2)
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