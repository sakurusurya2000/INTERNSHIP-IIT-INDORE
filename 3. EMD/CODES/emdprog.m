function[H]=emdprog(g)
g=load('S001.txt');
g=emd(g);

figure;
subplot(911)
plot(g(1,:))
title('IMF1')
 
subplot(912)
plot(g(2,:))
title('IMF2')
 
subplot(913)
plot(g(3,:))
title('IMF3')
 
subplot(914)
plot(g(4,:))
title('IMF4')
 
subplot(915)
plot(g(5,:))
title('IMF5')
 
subplot(916)
plot(g(6,:))
title('IMF6')
 
subplot(917)
plot(g(7,:))
title('IMF7')
 
subplot(918)
plot(g(8,:))
title('IMF8')
 
subplot(919)
plot(g(9,:))
title('IMF9')
 
f=g(1,:)';
N=length(f);
nb=(1:N);
f=f';
MM=N;
if exist('alfa') == 0
    x=2;
    alfa=zeros(1,MM);
    for i=1:MM
        ex=1;
        while abs(ex)>.00001
            ex=-besselj(0,x)/besselj(1,x);
            x=x-ex;
        end
        alfa(i)=x;
        %fprintf('Root # %g  = %8.5f ex = %9.6f \n',i,x,ex)
        x=x+pi;
    end
end
a=N;
for m1=1:MM
    a3(m1)=(2/(a^2*(besselj(1,alfa(m1))).^2))*sum(nb.*f.*besselj(0,alfa(m1)/a*nb));
end
freq=(alfa)/(2*pi*length(f));
 
for m1=1:MM
E(m1)=((a3(m1)^2)*(MM^2)/2).*((besselj(1,alfa(m1)))^2);
end
fmean=[sum(freq.*E/sum(E))];
fmean=meanfreq(a3);
H=fmean;
r1=[2.0892 2.2598   2.2569  2.1951  2.3391]';
r2=[2.3572 2.1468   2.3026  2.3524  2.3123]';
r=[r1;r2];
g1=repmat({'Seizure'},100,1);
g2=repmat({'Seizure-free'},100,1);
g=[g1;g2];
figure;
title('IMF1')
boxplot(l,g)



 
 
