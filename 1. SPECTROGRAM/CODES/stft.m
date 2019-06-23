function [stft, f, t] = stft(x, wlen, hop, nfft, fs)

x = x(:);

xlen = length(x);

win = hamming(wlen, 'periodic');

rown = ceil((1+nfft)/2);
coln = 1+fix((xlen-wlen)/hop);      
stft = zeros(rown, coln);           
indx = 0;

for col = 1:coln
    xw = x(indx+1:indx+wlen).*win;
    X = fft(xw, nfft);
    stft(:, col) = X(1:rown);
    indx = indx + hop;
end

t = (wlen/2:hop:wlen/2+(coln-1)*hop)/fs;
f = (0:rown-1)*fs/nfft;

end

