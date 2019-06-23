function y = denoiseEm(x);

[thr,sorh,keepapp]=ddencmp( 'den' , 'wv' ,x);  

y=wdencmp( 'gbl' ,x, 'db3' ,2,thr,sorh,keepapp);  
subplot(2,1,1)
plot(x);
subplot(2,1,2)
plot(y);

function y = applySkiSlope(x,g,transitionV,fs);


first = transitionV(1);
second = transitionV(2);
third = transitionV(3);
fourth = transitionV(4);
x_length = length(x);
n = nextpow2(x_length);
N = 2^n;
T = 1/fs;
X = fft(x,N);
gain = zeros(N,1);

% Sets the gain for the first stage of frequencies
firstC = (.3*(g-1))/first;
k=0;
while(k/N <= first/fs)
   gain(k+1) = firstC*k/(N*T) + 1;
   gain(N-k) = gain(k+1);
   k=k+1;
end;

% Sets the gain for the second stage of frequencies
secondC = firstC*first +1;    
secondC2 = (second-first)/5;
while(k/N <= second/fs)
   gain(k+1) = 1 + (secondC-1)*exp(-((k/(N*T))-first)/secondC2);
   gain(N-k) = gain(k+1);
   k=k+1;
end;

% Sets the gain for the third stage of frequencies
thirdC = 1 + (secondC-1)*exp(-second/secondC2);  
thirdC2 = (third-second)/5;
while(k/N <= third/fs)
   gain(k+1) = g + (thirdC-g)*exp(-((k/(N*T)-second))/thirdC2);
   gain(N-k) = gain(k+1);
   k=k+1;
end;

% Sets the gain for the fourth stage of frequencies
while(k/N <= fourth/fs)
   gain(k+1) = g;
   gain(N-k) = gain(k+1);
   k=k+1;
end;

% Sets the gain for the fifth stage of frequencies
fifthC = g;                
fifthC2 = (fs/2-fourth)/5;
while(k/N <= .5)
   gain(k+1) = 1 + (fifthC-1)*exp(-((k/(N*T))-fourth)/fifthC2);
   gain(N-k) = gain(k+1);
   k=k+1;
end;


k_v = (0:N-1)/N;
plot(k_v,gain);%entire filter transfer function

figure;%non-redundant filter transfer function
k_v = k_v*fs;
k_v = k_v(1:N/2+1);
plot(k_v,gain(1:N/2+1));
title('Frequency Shaper Transfer Function');
xlabel('Frequency (Hertz)');
ylabel('Gain');

Y = X+gain;
y = real(ifft(Y,N));

y = y(1:x_length);
t=[0:1/fs:(x_length-1)/fs];
figure;
plot(t,y,'r');
%hold;
figure;
plot(t,x);

function y = powerCompress(input, Psat,Fs);

x=input;
len=Fs*0.1;
iter=floor(length(x)/len);
Plow=0.008;
 
for rg=0:1:iter;
 start=rg*len+1;
 en= rg*len+len;
 if rg*len+len>length(x)
 en=length(x);
end
clear signal X  X_pow Y_pow Y y z X_phase;
signal=x(start:en);
n = nextpow2(len);
N = 2^n;
X = fft(signal,N);
X_phase=angle(X);                  % Save the old phase information
X_pow = abs(X)/N;
Y_pow = X_pow;
Y=zeros(N,1);
for k=0:N/2
   if Y_pow(k+1)<Plow              % Take out noise
      Y_pow(k+1)=0;
      Y_pow(N-k)=0;
   elseif Y_pow(k+1)>Psat          % Clip amplitudes higher than Psat
      Y_pow(k+1)=Psat;
      Y_pow(N-k)=Psat;
   end;
   Y(k+1) = Y_pow(k+1)*(cos(X_phase(k+1))+i*sin(X_phase(k+1)));
	Y(N-k) = Y_pow(N-k)*(cos(X_phase(N-k))+i*sin(X_phase(N-k)));
end;

y = real(ifft(Y,N));

z = y(1:en-start+1);

sig_out(start:en)=z;

end;

y = sig_out*2000;

function y = hearingAidF(input,g,Psat,transitionV,newfile);


[x,fs] = audioread(input);
xc = denoiseEm(x);                             % denoising filter
xf = applySkiSlope(xc,g,transitionV,fs);       % frequency shaping filter
y = powerCompress(xf, Psat,fs);                % amplitude shaping filter
x_length = length(x);
t=[0:1/fs:(x_length-1)/fs];
%sound(y,fs);


% plots for the input and output signals

figure;
subplot(2,1,1);
plot(t,x,'b');
axis tight;
xlabel('Time (sec)');
ylabel('Relative Magnitude');
title('Time Profile for Data in Signal 2');


subplot(2,1,2);
plot(t,y,'r');
axis tight;
xlabel('Time (sec)');
ylabel('Relative Magnitude');
title('Time Profile for Data in Adjusted Signal 2');

figure;
subplot(2,1,1);
specgram(x);
title('Spectrogram of Original Signal 2');

subplot(2,1,2);
specgram(y);
title('Spectrogram of Adjusted Signal 2');


%soundsc(input, fs);
sound(y,fs);
%audiowrite(y,fs,nbits,'linear',newfile);
audiowrite('temp_file.wav',y,fs);

hearingAidF('1.mp3', 6, 90, [3000 4000 5000 9000], 'new_sound')