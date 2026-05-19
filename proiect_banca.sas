
*1. Datele sunt importate din fișierul CSV într-un set de date SAS numit BANK;
proc import datafile='/home/u64488689/Bank_Marketing_Dataset_SMALL.csv'
    out=work.bank
    dbms=csv
    replace;
    guessingrows=max;
run;

proc contents data=work.bank;
run;

proc print data=work.bank(obs=10);
run;

*2. Formate definite de utilizator;
proc format;
value subsfmt
0='Nu a subscris'
1='A subscris';

value agefmt
low-29='Tineri'
30-49='Adulti'
50-high='Seniori';

value valfmt
0='Client obisnuit'
1='Client valoros';
run;


*3. Procesare condițională + variabile noi;
data bank2;
set bank;

length GrupVarsta $10;
length NivelRisc $12;

if Age < 30 then GrupVarsta='Tanar';
else if Age < 50 then GrupVarsta='Adult';
else GrupVarsta='Senior';

if AccountBalance > 5000 then ClientValoros=1;
else ClientValoros=0;

if CreditScore < 580 then NivelRisc='Ridicat';
else if CreditScore < 700 then NivelRisc='Mediu';
else NivelRisc='Scazut';

VenitTotal = sum(AnnualIncome, AccountBalance, InvestmentPortfolioValue);
VarstaRotunjita = round(Age);

format TermDepositSubscribed subsfmt. ClientValoros valfmt.;
run;

proc print data=bank2(obs=20);
var ClientID Age GrupVarsta AccountBalance ClientValoros CreditScore NivelRisc TermDepositSubscribed;
run;


*4. Crearea de subseturi de date;
data clienti_valorosi;
set bank2;
if ClientValoros=1;
run;

data clienti_cu_depozit;
set bank2;
if TermDepositSubscribed=1;
run;

proc print data=clienti_valorosi(obs=20);
title 'Clienti valorosi';
run;

proc print data=clienti_cu_depozit(obs=20);
title 'Clienti care au subscris la depozit';
run;


*5. Utilizarea funcțiilor SAS;
data bank3;
set bank2;

ScorMarketingRotunjit = round(MarketingScore, 0.01);
SoldMediuClient = mean(AccountBalance, InvestmentPortfolioValue);
ActivitateTotala = sum(TotalTransactions, NumOnlineTransactions, NumMobileAppLogins);

run;

proc print data=bank3(obs=20);
var ClientID AccountBalance InvestmentPortfolioValue SoldMediuClient MarketingScore ScorMarketingRotunjit ActivitateTotala;
run;
*Sunt utilizate funcții SAS: SUM si ROUND;


*6. Combinarea seturilor de date prin MERGE
Calculăm soldul mediu pe regiune și îl atașăm înapoi la fiecare client;
proc sort data=bank3;
by Region;
run;

proc means noprint data=bank3;
by Region;
var AccountBalance;
output out=medii_regionale mean=MedieSoldRegiune;
run;

data bank_final;
merge bank3 medii_regionale;
by Region;
run;

proc print data=bank_final(obs=20);
var ClientID Region AccountBalance MedieSoldRegiune;
title 'Clienti cu media soldului pe regiune';
run;
*Se calculează media soldului pentru fiecare regiune și se combină cu datele inițiale;

*7. Combinare prin PROC SQL;
proc sql;
create table raport_regiuni as
select Region,
       count(*) as NrClienti,
       mean(AccountBalance) as SoldMediu,
       mean(TermDepositSubscribed) as RataSubscriere
from bank_final
group by Region;
quit;

proc print data=raport_regiuni;
title 'Raport pe regiuni';
run;


*8. Utilizarea de masive;

data bank_array;
set bank_final;

array valori {*} AnnualIncome AccountBalance CreditScore CreditLimit InvestmentPortfolioValue;

do i = 1 to dim(valori);
    if valori{i} < 0 then valori{i}=0;
end;

drop i;
run;

proc print data=bank_array(obs=20);
var ClientID AnnualIncome AccountBalance CreditScore CreditLimit InvestmentPortfolioValue;
run;


*9. Proceduri pentru raportare;
proc sort data=bank_array;
by Region;
run;

proc print data=bank_array(obs=50) noobs label;
by Region;
var ClientID Age Gender CustomerSegment AccountBalance MarketingScore TermDepositSubscribed;
format TermDepositSubscribed subsfmt.;
title 'Raport clienti banca pe regiuni';
run;


*10. Proceduri statistice;
proc means data=bank_array n mean min max;
var Age AnnualIncome AccountBalance CreditScore MarketingScore ResponsePropensity;
title 'Statistici descriptive pentru clientii bancii';
run;

proc freq data=bank_array;
tables Region CustomerSegment TermDepositSubscribed RiskRating;
format TermDepositSubscribed subsfmt.;
title 'Distributii de frecventa';
run;

proc univariate data=bank_array;
var AccountBalance;
histogram;
title 'Analiza distributiei soldurilor';
run;


*11. Grafice;
proc sgplot data=bank_array;
vbar Region;
title 'Numarul clientilor pe regiuni';
run;

proc sgplot data=bank_array;
vbar CustomerSegment / group=TermDepositSubscribed;
title 'Subscrierea la depozit pe segmente de clienti';
run;

proc sgplot data=bank_array;
histogram AccountBalance;
title 'Distributia soldurilor conturilor';
run;

proc sgplot data=bank_array;
scatter x=MarketingScore y=ResponsePropensity / group=TermDepositSubscribed;
title 'Relatia dintre scorul de marketing si propensitatea de raspuns';
run;

*12. SAS ML – Regresie logistică;
proc logistic data=bank_array descending;
class Gender Region CustomerSegment RiskRating ChannelPreference / param=ref;
model TermDepositSubscribed =
Age
AnnualIncome
AccountBalance
CreditScore
MarketingScore
ResponsePropensity
TotalTransactions
NumMobileAppLogins
LastContactDuration
NumContactsInCampaign;
title 'Model logistic pentru estimarea subscrierii la depozit';
run;
