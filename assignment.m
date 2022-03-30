clc
clear
opts = spreadsheetImportOptions("NumVariables", 2);
opts.Sheet = "Foglio1";
opts.DataRange = "A2:B507";
opts.VariableNames = ["Date", "LastPrice"];
opts.VariableTypes = ["datetime", "double"];
opts = setvaropts(opts, "Date", "InputFormat", "");
SPX = readtable("SPX.xls", opts, "UseExcel", false);
clear opts
Dati = readtable("Dataset.xls");

%% dati d'input
K = Dati{2:end,1};
r = 0.08/100;   
T = 31/360;
q = 1.27/100;   
Stpoors = SPX{:,2};
St =Stpoors(1);

%% punto 1: calcolo dei prezzi mid, verifica vincoli di Merton, monotonicità e convessità
prezziC = Dati{2:end,[3,4]}./100;
prezziP = Dati{2:end,[10,11]}./100;
midC = mean(prezziC,2);
midP = mean(prezziP,2);
pC = max(St*exp(-q*T)-K.*exp(-r*T),0);
pP = max(K.*exp(-r*T)-St*exp(-q*T),0);
%vincoli di Merton
if (midP >= pP) & (midC >= pC)
    risp="i vincoli di Merton sono soddisfatti";
else
    risp= "i vincoli di Merton non sono soddisfatti";
end
%di seguito si verificano i vincoli di monotonicità e convessità
dk=nan(length(K)-1,1);
dc=nan(length(K)-1,1);
dp=nan(length(K)-1,1);
for i=1:length(K)-1
    dk(i) = K(i+1) - K(i);
    dc(i) = (midC(i+1) - midC(i))/dk(i); 
    dp(i) = (midP(i+1) - midP(i))/dk(i);
end
dc2=nan(length(dc)-1,1);
dp2=nan(length(dp)-1,1);
for i=1:length(dc)-1
	dc2(i) = (dc(i+1) - dc(i))/dk(i); 
    dp2(i) = (dp(i+1) - dp(i))/dk(i);
end
if (dc < 0) & (dp > 0)
    risp2="l'andamento dei prezzi delle call e delle put è monotono";
else
    risp2= "l'andamento dei prezzi delle call e delle put non è monotono";
end
if (dc2 > 0) & (dp2 > 0)
    risp3="l'andamento dei prezzi delle call e delle put è convesso";
else
    risp3= "l'andamento dei prezzi delle call e delle put non è sempre convesso";
end

plot(K,midC,'*'); %prezzi decrescenti rispetto k
legend('prezzo della Call');
xlabel('K');
ylabel('midC');
%saveas(gcf, 'Fig1', 'png');
plot (K,midP,'*'); %prezzi crescenti rispetto k
legend('prezzo della Put');
xlabel('K');
ylabel('midP');
%saveas(gcf, 'Fig2', 'png');
clear prezziP prezziC pC pP dc dc2 dk dp dp2 i

 %% punto 2: interpolazione quadratica sulle volatilità implicite, prezzo Call europea attraverso formula B&S
IVM=Dati{2:end,6};
IVC=IVM.*sqrt(T);
B = regstats(IVC,K,'purequadratic');
A0 = B.beta(1);
A1 = B.beta(2);
A2 = B.beta(3);
%interpolazione con strike discreti
IVdisc = A0 + A1.*K + A2.*K.^2;
% interpolazione IV per strike continui non quotati.
Kc = (K(1):K(end));
IVc = A0 + A1.*Kc + A2.*Kc.^2;
plot(K, IVdisc,'ro', Kc, IVc);
legend('volatilità discreta', 'volatilità continua');
xlabel('K');
ylabel('vol');
%saveas(gcf, 'Fig3', 'png');
k=4698.5;
vol = A0 + A1*k + A2*k^2;
CallBS = blsprice(St,k,r,T,vol,q); %con Dividend Yield

clear IVM IVC B IVdisc Kc IVc k vol

%% punto 3: simulazioni Montecarlo
k=4695;
Nsim=10000;
sigma = 0.02;
Nstep=31;
dt=1;
rng(0); %seed per generazione numeri casuali
y=nan(Nstep,Nsim);
ST=[repmat(St,[1 Nsim]);nan(Nstep,Nsim)];
for t=1:Nstep
    for n=1:Nsim       
        y(t,n)=(r-sigma^2/2)*dt+sigma*sqrt(dt)*randn();
        ST(t+1,n)=ST(t,n).*exp(y(t,n));    
    end
end
plot(ST);
legend('10000 traiettorie S&P500');
xlabel('Date');
ylabel('S&P500');
%saveas(gcf, 'Fig4', 'png');
CallMC=exp(-r*T)*mean(max(ST(end,:)-K,0));
CmedioMC=mean(CallMC); %il prezzo coincide con la media campionaria
SN2=(1/(Nsim-1))*(sum((CallMC-CmedioMC)).^2); 
intervalloC=[CmedioMC-1.96*sqrt(SN2/Nsim),CmedioMC+1.96*sqrt(SN2/Nsim)]; 
vol = A0 + A1*k + A2*k^2;
CallBS2 = blsprice(St,k,r,T,vol); 
% prezzo ottenuto con la formula di B&S con la volatilità interpolata.
clear ST y  n CallMC vol

%% punto 4 variabili antitetiche
Z=randn(Nsim,1);
X1=St*exp((r-sigma^2/2)*T+sigma*sqrt(T)*Z);
X2=St*exp((r-sigma^2/2)*T+sigma*sqrt(T)*(-Z));
Payoff1=max(X1-k,0);
Payoff2=max(X2-k,0);
DiscPayoff=exp(-r*T)*(Payoff1+Payoff2)/2;
CallAT=sum(DiscPayoff)/Nsim; 
SN2=(1/(Nsim-1))*sum((DiscPayoff-CallAT).^2);  

intervalloCAT=[CallAT-1.96*sqrt(SN2/Nsim),CallAT+1.96*sqrt(SN2/Nsim)];
clear Z X1 X2 Payoff1 Payoff2 DiscPayoff 

%% punto 5 VIX
StrikeQuote=[K(K<=St); K(K>St)];%Only OTM contracts
OTM=[midP(K<=St); midC(K>St)];%Only OTM prices
deltaK2=diff(StrikeQuote);
Kappa2=StrikeQuote(1:end-1);  
VIX=sqrt(2/T*sum((deltaK2./Kappa2.^2).*exp(r*T).*OTM(1:end-1,1)))*100; 
clear StrikeQuote OTM deltaK2 Kappa2

%% punto 6: simulazioni indici DJIA e S&P500 e calcolo del prezzo di un'opzione "down-and-out"
Pt0 = 35927;
St0=[Pt0,St];
rho = 0.95;
sigma = [0.015,sigma];
rng(0);
SDJI = zeros(Nstep+1,Nsim);
StSP = zeros(Nstep+1,Nsim);
SDJI(1,:) = St0(1)*ones(1,Nsim);
StSP(1,:) = St0(2)*ones(1,Nsim);
for j = 2:Nstep+1
    % genero valori casuali variabili normali indipendenti
    z = randn(2,Nsim);
   % genero valori casuali da var. normali correlate
    w1 = z(1,:);
    w2 = rho*z(1,:)+ sqrt(1-rho^2)*z(2,:);
    % genero cammini correlati
    SDJI(j,:) = SDJI(j-1,:).*exp((r-sigma(1)^2/2)*dt+sigma(1)*sqrt(dt)*w1);
    StSP(j,:) = StSP(j-1,:).*exp((r-sigma(2)^2/2)*dt+sigma(2)*sqrt(dt)*w2);
end
plot(StSP);
legend('10000 simulazioni S&P500');
xlabel('Date');
ylabel('S&P500');
%saveas(gcf, 'Fig5', 'png');
plot(SDJI);
legend('10000 simulazioni DJI');
xlabel('Date');
ylabel('DJI');
%saveas(gcf, 'Fig6', 'png');
k1 = 4690;
k2 = 35600;
OpzEsotMC = exp(-r*T)*mean(max(StSP(end,:)-k1,0).*(min(SDJI,[],2)>k2));
OpzEsot=mean(OpzEsotMC);
clear Pt0 St0 rho sigma j z w1 w2 k1 k2 OpzEsotMC

%% punto 7: stima e andamento grafico v.c Normale e Variance-Gamma
rendimenti = - diff(Stpoors)./Stpoors(2:end);
p0=[0.004	-0.05	0.01	1.2];
%minimizzo la massima verosimiglianza 
par=fmincon(@(x)-MLEvg(rendimenti,x),p0,[],[],[],[],[-10,-10,0,0]);
m0=par(1);
m=par(2);
sigma=par(3);
a=par(4);
%distribuzione variance gamma per l'intervallo x
x=-0.05:0.001:0.05;
distrVG = nan(1,size(x,2));
for i=1:length(x)
    a_0=exp((x(i)-m0)* m/sigma^2);
    Z=sqrt(2*sigma^2+m^2)*abs(x(i)-m0)./sigma^2;
    k=besselk(a-1/2,Z);
    d=sqrt(2)*(abs(x(i)-m0)/(sqrt(m^2+2*sigma^2)))^(a-0.5);
    distrVG(i)=d*a_0*k/(gamma(a)*sigma*sqrt(pi));
end
[mu,sigma]=normfit(rendimenti);
distrN=normpdf(x,mu,sigma);
histogram(rendimenti,100,'Normalization','pdf');%distr.empirica dei rendimenti
hold on
plot(x,distrVG','r','LineWidth',2);%VG density
plot(x,distrN','g','LineWidth',2);% Normal density
legend('Empirica','VG','Normale');
hold off
xlabel('x');
ylabel('densità di rendimenti');
%saveas(gcf, 'Fig7', 'png');
clear rendimenti p0 par m0 a sigma m x i a_0 Z k d mu 