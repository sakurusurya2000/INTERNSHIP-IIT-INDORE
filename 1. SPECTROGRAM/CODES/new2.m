clc
clear all
close all
s=audioread('1-1.wav')';
%s=wavrecord(3*8000,8000)';
%s=s(500:size(s,2))*1;
nn=.000001*rand(1,size(s,2)).*(rand(1,size(s,2))-.5);
s1=s;
t=0:.01:1000000;
ss=.08*sin(60*t(1:size(s,2)));
s1=s1+ss;
%load destroye;
%s1=s1+.5.*bb(1:size(s1,2));

e_min=100;
e_max=.1;
Hvad=0;Hvad_1=0;tt=1;
for i=1:160:size(s,2)-200
    y=s1(i:i+159);
    [T R]=pitch_est_center_clipped(y);
    C(i:i+199)=max(R);
   fff(tt)= sum(y.^2/160);
    e_full=sqrt(sum(y.^2/160));
    y1=fft(y,256);
    y1(1:64)=0;y1(256-63:256)=0;
    y2=ifft(y1,256);y2=y2(1:160);
    e_high=sqrt(sum(abs(y2).^2/160));
    if e_min> e_full
        e_min=e_full;
    end
    if e_max<e_full
        e_max=e_full;
    end
    landa=(e_max-e_min)/e_max;
    thrr=(1-landa)*e_max+landa*e_min;
    e_r=e_high/(e_full-e_high);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if e_full>thrr  
        v=1;
    elseif e_r>10
        v=1;
    elseif C(i)>.2
            v=1;
    else
        v=0;
    end
    
    
    if (v==0&&Hvad_1==1&&e_full>3*e_min)||v==1
        Hvad=1;
    else
        Hvad=0;
    end
    if e_full<.002
        Hvad=0;
    end
    V(i:i+159)=Hvad;
     E(tt)=C(i);tt=tt+1;
    
    Hvad_1=Hvad;
    
end
plot(s1)
hold on
plot(.25*V,'r')    
%wavplay(s(1:size(V,2)).*V,8000)
