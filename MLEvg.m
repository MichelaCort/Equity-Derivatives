function L=MLEvg(y,x)
%restituisce il valore della log-likelihood in VG
%y contiene i punti delle possibili realizzazioni
%x vettore contenente i parametri della VG(mu_0,mu,sigma,a)
m0=x(1);
m=x(2);
sigma=x(3);
a=x(4);
A=sum((y-m0)*m/sigma^2);
Z=sqrt(2*sigma^2+m^2)*abs(y-m0)./sigma^2;
B=sum(log(besselk(a-1/2,Z)));
C=sum((a-1/2)*(log(abs(y-m0))-1/2*log(m^2+2*sigma^2)));
T=length(y);
L=T/2*log(2/pi)+A-T*(log(gamma(a))+log(sigma))+B+C;

