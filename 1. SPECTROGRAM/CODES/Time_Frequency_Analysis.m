clc;
close all;
clear all;

% load a signal
[x, fs] = audioread('track.wav');       % load an audio file
%x = x(:, 1);                       % get the first channel

% determine the signal parameters
xlen = length(x);                   % signal length
t = (0:xlen-1)/fs;                  % time vector

% analysis parameters
wlen = 1024;
hop = wlen/4;
nfft = 4096;                        % hop size

TimeRes = wlen/fs;                  % time resulution of the analysis (i.e., window duration), s
FreqRes = 2*fs/wlen;                % frequency resolution of the analysis (using Hanning window), Hz

% time-frequency grid parameters
TimeResGrid = hop/fs;               % time resolution of the grid, s
FreqResGrid = fs/nfft;              % frequency resolution of the grid, Hz 

% perform STFT
w1 = hamming(wlen, 'periodic');

[fS, tS, PSD] = spectrogram(x, w1, wlen-hop, nfft, fs);
% [~, fS, tS, PSD] = spectrogram(x, w1, (wlen-hop), nfft, fs);
Samp = 20*log10(sqrt(PSD.*enbw(w1, fs))*sqrt(2));

% perform spectral analysis
w2 = hamming(xlen, 'periodic');
[PS, fX] = periodogram(x, w2, nfft, fs, 'power');
Xamp = 20*log10(sqrt(PS)*sqrt(2));

% plot the signal waveform
figure(1)
subplot(3, 3, [2 3])
plot(t, x)
grid on
xlim([0 max(t)])
ylim([-1.1*max(abs(x)) 1.1*max(abs(x))])
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)
xlabel('Time, s')
ylabel('Amplitude')
title('The signal in the time domain')

% plot the spectrum
subplot(3, 3, [4 7])
plot(fX, Xamp)
grid on
xlim([0 max(fX)])
ylim([min(Xamp)-10 max(Xamp)+10])
view(-90, 90)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)
xlabel('Frequency, Hz')
ylabel('Magnitude, dB')
title('Amplitude spectrum of the signal')

% plot the spectrogram
subplot(3, 3, [5 6 8 9])
surf(tS, fS, Samp)
shading interp
axis tight
box on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)
xlabel('Time, s')
ylabel('Frequency, Hz')
title('Amplitude spectrogram of the signal')
view(0, 90)

hcol = colorbar('East');
set(hcol, 'FontName', 'Times New Roman', 'FontSize', 12)
ylabel(hcol, 'Magnitude, dB')

% display some analysis paramaters
disp(['Frequency resolution of the analysis: ' num2str(FreqRes) ' Hz'])
disp(['Time resolution of the analysis: ' num2str(TimeRes) ' s'])
disp(['Resolution of the frequency grid: ' num2str(FreqResGrid) ' Hz'])
disp(['Resolution of the time grid: ' num2str(TimeResGrid) ' s'])

% % mark the dominant frequencies in the spectrogram
% [~, inds] = max(Samp, [], 1);
% fmax = fS(inds);
% hold on
% plot3(tS, fmax, zeros(length(tS)), 'r')