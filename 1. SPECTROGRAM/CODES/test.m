[signal, Fs] = audioread('1.mp3'); % Open demo sound
spectrum = pwelch(signal);
plot(10*log10(spectrum))
drawnow;
sound(signal, Fs);