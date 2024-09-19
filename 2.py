Question 1
Posted on absalon is a press release from the European Securities and Markets Authoriy regarding its decision to prohibit sales of binary options to retail investors. Read
the article (https://www.esma.europa.eu/press-news/esma-news/esma-agrees-prohibit-binary-options-and-restrict-cfds-protect-retail-investors) and answer the following
questions:
1. What is the motivation behind this decision? How does this relate to the models we have seen in class?
2. What effect will this decision have on liquidity in the binary options market?
3. Measures announced in the press release differ between the binary option and CFD markets. How will the effects of the regulation be different across the two
markets?
Be concise and to the point. Please try to keep your answer less than 100 words (but not just one sentense).
1. What is the motivation behind this decision? How does this relate to the models we have seen in class?
The motivation behind it is to: control the risk of investing binary options for retail investors and regulate on high risk f
inancial instruments in the financial markets. Also, control the potential fraudulent behaviour which do harm to the financial ma
rket.
Because the revenue from a binary option is all or nothing, investing in binary option is very similar to gambling, which is
extremely risky. Like the Robinhood case in class, retail traders are exposed to irrational biases like anchoring or confirmation
bias, and unwilling to sell the asset although has low price, thus results in larger potential loss.
2. What effect will this decision have on liquidity in the binary options market?
The action would lower the liquidity provided from retail investors in the binary options market, since the prohibition stop
retail investors to enter into the market. However, it may not stop the liquidity provided by institutional or significant invest
ors in the market.
3. Measures announced in the press release differ between the binary option and CFD markets. How will the effects of the regulatio
n be different across the two markets?
The regulation prohibits the marketing, distribution or sale of binary options to retail investors, in the meanwhile, only a
nouncing some restrictions on the marketing, distribution or sale of CFDs to retail investors. For CFD retail investors, more res
trictions are imposed in order to control potential risk, but they are not prohibited from entering the market and they can still
provide liquidity in CFD market. However, for binary options market, the marketing is prohibited.
Question 2 Trade data
In the following use the trade data in tqBAC.csv. Denote trade prices by  and mid-quotes by  .
Sign each trade based on Lee-ready algorithm
Calcualte Spread, Effective Spread and Realized Spread by EXCHANGES
Examine order correlation
𝑞 𝑡 𝑛 𝑡
Import the relevant modules
In [*]:
### In this project, I will use datatable (quicker) rather pandas to manipulate data. You can see which one is more intuitive for you
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import os
data=pd.read_csv('C:\\Users\\Downloads\\tqBAC.csv')
2/21/24, 6:10 PM Asg - Jupyter Notebook
localhost:8888/notebooks/Asg.ipynb 2/11
Lee and Ready algorithm
In the database reported by exchanges, the buyer and seller identify is not revealed. This is very different from bond database you worked in the previous assignment,
which was reported by dealers and records the trade direction of dealers. Given that every trade involves buyers and sellers, how do we know which side market maker
(liquidity supplier) stands and customers (liquidity demander) stands? The simple answer is given by the Lee and Ready algorithm as follows. The idea is to assign trade
directions based on wheter trades happen around ask or bid prices.
A typical classification is
buyer-initiated if pt > mt
buyer-initiated if pt = mt and pt > pt−1 (downtick)
seller-initiated if pt < mt
seller-initiated if pt = mt and pt < pt−1 (uptick)
In [*]:
Calcualte Spread, Effective Spread and Realized Spread by EXCHANGES
There are multiple ways to measure spreads in realty for different purposes. Moreoever, in order to compare spreads across stocks, it is common to normalized spread
based on prices. For example, the spread of Bitcoin is larger than the spread of AMC, simply because Bitcoin trades at $50000 per unit whereas AMC trades at a few
dollars per unit.
1. Quoted spread:  , where  . This is simple bid-ask spread telling you about the potential cost of trading.
2. Effective spread:  , where  is the trade direction (1 for buyer-initiated and -1 for seller initiaed). In reality, because of high-frequency traders
(remind HFs can cancell orders and post another one before your orders arrive at exhcnages), the actually transaction price can differ from bid and ask prices you
see. This effective spread is the actual transaction cost one pays.
3. Realized spread:  . Imagine that you bought some shares at , then the price moves to a new level  because of realization of information or
other things, then the actual spread paid can be negative if the news are good and larger if the news are bad. This is the measure more relavent to market makers as
this measures how much a market maker for providing liqudity over time to  .
In this exercise, please calcuate different spreads, and check some summary statistics for these spreads. Note that for realized spread, using mid-quote in 10 mins
1. calcate correlation of three spreads
2. plot time series of three spreads by hour
3. calcuate mean spreads at the Exchange level
= 𝑆 𝑡
− 𝑏 𝑡 𝑐 𝑡
𝑛 𝑡
= 𝑛 𝑡
+ 𝑏 𝑡 𝑐 𝑡
2
= ( − ) 𝑆 𝑡 𝑒 𝑡 𝑞 𝑡 𝑛 𝑡 𝑒 𝑡
= ( − ) 𝑆 𝑡 𝑒 𝑡 𝑞 𝑡 𝑛 𝑡+Δ 𝑡 𝑛 𝑡+Δ
𝑡 𝑡 + Δ
In [*]:
correlation:
quo_spr eff_spr rea_spr
quo_spr 1.000000 -0.213157 0.015860
eff_spr -0.213157 1.000000 -0.006176
rea_spr 0.015860 -0.006176 1.000000
data['lag_P']=data["PRICE"].shift()
data['mt']=(data['OFR']+data['BID'])/2
def a(x):
if x['PRICE']>x['mt']:
return 1
elif x['PRICE']==x['mt'] and x['PRICE']>x['lag_P']:
return 1
elif x['PRICE']<x['mt']:
return -1
elif x['PRICE']==x['mt'] and x['PRICE']<x['lag_P']:
return -1
data['dt']=data.apply(a,axis=1)
data['mtplus10']=data['mt'].shift(-1207)
data['quo_spr']=(data['OFR']-data['BID'])/data['mt']
data['eff_spr']=data['dt']*(data['PRICE']-data['mt'])
data['rea_spr']=data['dt']*(data['PRICE']-data['mtplus10'])
data.iloc[:,-3:].corr()
2/21/24, 6:10 PM Asg - Jupyter Notebook
localhost:8888/notebooks/Asg.ipynb 3/11
In [ ]:
Quoted Spread:
Effective Spread:
data['pddate']=[datetime.strptime(d[:-5],'%Y-%m-%dT%H:%M:%S') if len(d)==24 else datetime.strptime(d[:-1],'%Y-%m-%dT%H:%M:%S')for d in dat
fig, ax = plt.subplots()
plt.plot(pd.Series(data['quo_spr'].values,index=data['pddate']))
plt.xlabel("Hour")
plt.ylabel("Quoted Spread")
plt.title('Quoted Spread')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:00"))
plt.savefig('C:\\Users\\Desktop\\UCLATrading\\Quoted Spread.png')
fig, ax = plt.subplots()
plt.plot(pd.Series(data['eff_spr'].values,index=data['pddate']))
plt.xlabel("Hour")
plt.ylabel("Effective Spread")
plt.title('Effective Spread')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:00"))
plt.savefig('C:\\Users\\Desktop\\UCLATrading\\Effective Spread.png')
fig, ax = plt.subplots()
plt.plot(pd.Series(data['rea_spr'].values,index=data['pddate']))
plt.xlabel("Hour")
plt.ylabel("Realized Spread")
plt.title('Realized Spread')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:00"))
plt.savefig('C:\\Users\\Desktop\\UCLATrading\\Realized Spread.png')
plt.show()
2/21/24, 6:10 PM Asg - Jupyter Notebook
localhost:8888/notebooks/Asg.ipynb 4/11
Realized Spread:
In [*]:
mean spreads at the Exchange level:
quo_spr eff_spr rea_spr
EX
A 0.000448 0.004519 0.028804
B 0.000405 0.003934 -0.000572
J 0.000404 0.004762 0.001340
K 0.000460 0.003565 0.004507
M 0.000462 0.004600 0.012133
N 0.000418 0.003416 0.004845
P 0.000434 0.003742 0.003113
T 0.000442 0.003244 0.003896
V 0.000410 0.003484 0.000495
X 0.000420 0.004312 -0.006950
Y 0.000404 0.004503 0.002244
Z 0.000431 0.002710 0.000219
Order Sign Correlation
print(data.groupby('EX')[['quo_spr','eff_spr','rea_spr']].mean())
2/21/24, 6:10 PM Asg - Jupyter Notebook
localhost:8888/notebooks/Asg.ipynb 5/11
As discussed in the lecture, order spliting is common for informed investors to minimize their price impact. How to empirically check this? One possibility is to examine
auto-correlation of orders. With Lee and Ready algorithm, we have a sense how liquidity demanders (informed investors) trade. We start with some simple analysis to
check how signed orders are correlated, and then check how to better fit the data to predict sign of next orders. Intuitively, the market makers have a good model to do
so, they can 1) front-run investors to profit more, 2) adjust bid-ask prices and market depth to avoid being adversly selected by informed investors.
1. autocorrelation plot of order sign
2. re-produce the above figure in log term (both x-axis and y-axis are in log term)
3. fit regressions to check whehter past information can predict future order signs.
autocorrelation plot of order sign：
In [ ]:
re-produce the above figure in log term (both x-axis and y-axis are in log term):
In [ ]:
plot_acf(data['dt'].dropna())
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Plot')
plt.savefig('C:\\Users\\Desktop\\UCLATrading\\Auto.png')
plt.show()
plt.plot(np.log(range(46)),np.log(sm.tsa.acf(data['dt'].dropna(),nlags = 45,fft=False)))
plt.xlabel('logLags')
plt.ylabel('logAutocorrelation')
plt.title('Log Autocorrelation Plot')
plt.savefig('C:\\Users\\Desktop\\UCLATrading\\logAuto.png')
plt.show()
2/21/24, 6:10 PM Asg - Jupyter Notebook
localhost:8888/notebooks/Asg.ipynb 6/11
fit regressions to check whether past information can predict future order signs:
Firstly, regress future order signs( here we use 10 min later order signs same as above) on current order signs:
In [ ]:
Results:
OLS Regression Results
=======================================================================================
Dep. Variable: future_order R-squared (uncentered): 0.000
Model: OLS Adj. R-squared (uncentered): -0.000
Method: Least Squares F-statistic: 0.01227
Date: Wed, 21 Feb 2024 Prob (F-statistic): 0.912
Time: 04:32:56 Log-Likelihood: -26013.
No. Observations: 18333 AIC: 5.203e+04
Df Residuals: 18332 BIC: 5.204e+04
Df Model: 1
Covariance Type: nonrobust
==============================================================================
coef std err t P>|t| [0.025 0.975]
------------------------------------------------------------------------------
dt -0.0008 0.007 -0.111 0.912 -0.015 0.014
==============================================================================
Omnibus: 63168.659 Durbin-Watson: 1.103
Prob(Omnibus): 0.000 Jarque-Bera (JB): 3055.573
Skew: 0.101 Prob(JB): 0.00
Kurtosis: 1.010 Cond. No. 1.00
==============================================================================
From the results we know the parameter of current order sign is not significant, and the R2 is 0, meaning past order sign has
nearly no prediction effect on future order signs.
Secondly, regress future order signs on three different current spread calculated above to see whether the performance is improve
d:
data['future_order']=data['dt'].shift(-1207)
mod = sm.OLS(data['future_order'],data['dt'],missing='drop')
print(mod.fit().summary())
2/21/24, 6:10 PM Asg - Jupyter Notebook
localhost:8888/notebooks/Asg.ipynb 7/11
Results:
OLS Regression Results
=======================================================================================
Dep. Variable: future_order R-squared (uncentered): 0.004
Model: OLS Adj. R-squared (uncentered): 0.003
Method: Least Squares F-statistic: 22.27
Date: Wed, 21 Feb 2024 Prob (F-statistic): 2.17e-14
Time: 04:43:26 Log-Likelihood: -25980.
No. Observations: 18333 AIC: 5.197e+04
Df Residuals: 18330 BIC: 5.199e+04
Df Model: 3
Covariance Type: nonrobust
==============================================================================
coef std err t P>|t| [0.025 0.975]
------------------------------------------------------------------------------
quo_spr -32.3002 36.928 -0.875 0.382 -104.682 40.082
eff_spr -8.9439 3.418 -2.617 0.009 -15.643 -2.245
rea_spr -0.5884 0.155 -3.792 0.000 -0.893 -0.284
==============================================================================
Omnibus: 63325.921 Durbin-Watson: 1.109
Prob(Omnibus): 0.000 Jarque-Bera (JB): 3042.296
Skew: 0.101 Prob(JB): 0.00
Kurtosis: 1.015 Cond. No. 239.
==============================================================================
The coefficients of effective spread and realized spread is significant. Meaning effective spread and realized spread can pro
vide some explanation on future order. However, quoted spread is not significant, and the R2 is only 0.004. Thus the model is not
well fitted.
Regress future order signs on current signs,'SIZE','PRICE' and spreads:
OLS Regression Results
=======================================================================================
Dep. Variable: future_order R-squared (uncentered): 0.004
Model: OLS Adj. R-squared (uncentered): 0.003
Method: Least Squares F-statistic: 11.69
Date: Wed, 21 Feb 2024 Prob (F-statistic): 4.11e-13
Time: 04:52:05 Log-Likelihood: -25978.
No. Observations: 18333 AIC: 5.197e+04
Df Residuals: 18327 BIC: 5.202e+04
Df Model: 6
Covariance Type: nonrobust
==============================================================================
coef std err t P>|t| [0.025 0.975]
------------------------------------------------------------------------------
dt -0.0083 0.008 -1.104 0.270 -0.023 0.006
SIZE -1.682e-06 1.22e-06 -1.380 0.168 -4.07e-06 7.07e-07
quo_spr 61.2444 144.610 0.424 0.672 -222.205 344.694
eff_spr -8.0943 3.807 -2.126 0.033 -15.556 -0.633
rea_spr -0.6308 0.159 -3.967 0.000 -0.943 -0.319
PRICE -0.0016 0.003 -0.622 0.534 -0.007 0.003
==============================================================================
Omnibus: 63344.262 Durbin-Watson: 1.109
Prob(Omnibus): 0.000 Jarque-Bera (JB): 3040.677
Skew: 0.101 Prob(JB): 0.00
Kurtosis: 1.015 Cond. No. 1.24e+08
==============================================================================
When adding in some other current information, the coefficient of effective spread and realized spread are still statisticall
y significant. However, the other variables seems to have no help on future order prediction. The R2 is not changing.
In conclusion, effective spread and realized spread are significant predictors, however, OLS model have nearly no strong pred
iction power.
Question 3 Quote data
As we discussed in the class, not only transacted orders are informative, orders sitting on the books could potential provide some valuation information. In this exercise,
we try to test this idea to see whether inbalance order book can help form some trading signals.
Data BAC_nbbo.csv is order bood data (only the best quotes), with each row one of the price or size at the best bid or ask changes which corresponds to change in the
supply or demand.
Calculate order imbalance OFI (keep only Nasdaq exchanges)
Aggregate OFI to second level (take summation)
2/21/24, 6:10 PM Asg - Jupyter Notebook
localhost:8888/notebooks/Asg.ipynb 8/11
Order Imbalance
Order flow imbalance represents the changes in supply and demand.
Best bid or size at the best bid increase -> increase in demand.
Best bid or size at the best bid decreases -> decrease in demand.
Best ask decreases or size at the best ask increases -> increase in supply.
Best ask increases or size at the best ask decreases -> decrease in supply.
Mathematically we summarise these four effects at from time  to as:
where  is the beset Bid price at time and  is the size at those prices, and I is an indicator function. For exampel,  if  and 0, otherwise.
𝑜 − 1 𝑜
= − − + 𝑓 𝑜 𝐼
≥ 𝐶 𝑜 𝐶 𝑜−1 𝑟 𝑜
𝐼
≤ 𝐶 𝑜 𝐶 𝑜−1 𝑟 𝑜−1
𝐼
≤ 𝐵 𝑜 𝐵 𝑜−1 𝑟 𝑜
𝐼
≥ 𝐵 𝑜 𝐵 𝑜−1 𝑟 𝑜−1
𝐶 𝑜 𝑜 𝑟 𝑜 = 1 𝐼
≥ 𝐶 𝑜 𝐶 𝑜−1
≥ 𝐶 𝑜 𝐶 𝑜−1
In [ ]:
In [1]:
In [2]:
In [3]:
Using OFI to generate trading signal: first do train/test split by selecting the first 70% of the data
In [4]:
# Creat second stamp
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import os
data=pd.read_csv('C:\\Users\\Downloads\\BAC_nbbo.csv')
## only keep trading hours
data['TIME_M']=data['TIME_M'].apply(lambda x:'0'+x if len(x)==17 else x)
data=data[data['TIME_M'].apply(lambda x:True if x>='09:30' and x<='16:00'else False)]
# notice the extreme values in BID and ASK!!
# need to clean data
# first, remove negative spreads
# then outlier quotes
data=data[data["ASK"]>=data['BID']]
data=data[data.apply(lambda x:x['BID']>10 and x['BID']<100 and x['ASK']>10 and x['ASK']<100, axis=1)]
data=data[data['EX']=='N']
# Aggregate by second
data['lag_BID']=data['BID'].shift()
data['lag_ASK']=data['ASK'].shift()
data['lag_BIDSIZ']=data['BIDSIZ'].shift()
data['lag_ASKSIZ']=data['ASKSIZ'].shift()
data['OFI']=(data['BID']>=data['lag_BID'])*data['BIDSIZ']-(data['BID']<=data['lag_BID'])*data['lag_BIDSIZ']-(data['ASK']<=data['lag_ASK'])
data['second']=[datetime.strptime(d[:8],'%H:%M:%S')for d in data['TIME_M']]
OFI=data.groupby('second')['OFI'].sum()
# Construct return as log difference of last mid price and first mid price of each second
data['mid_pri']=(data['ASK']+data['BID'])/2
return_second=data.groupby('second')['mid_pri'].apply(lambda x:np.log(x.iloc[-1])-np.log(x.iloc[0]))
second_data=pd.DataFrame(OFI)
second_data['return']=return_second
# Test whether OFI can explain return variations in train data
train=second_data[:int(0.7*len(second_data))]
test=second_data[int(0.7*len(second_data)):]
print(train.corr())
mod = sm.OLS(train['return'],train['OFI'],missing='drop')
print(mod.fit().summary())
2/21/24, 6:10 PM Asg - Jupyter Notebook
localhost:8888/notebooks/Asg.ipynb 9/11
# explainatry power for test sample
correlation:
OFI return
OFI 1.000000 -0.020689
return -0.020689 1.000000
OLS regression Result:
OLS Regression Results
=======================================================================================
Dep. Variable: return R-squared (uncentered): 0.000
Model: OLS Adj. R-squared (uncentered): 0.000
Method: Least Squares F-statistic: 6.728
Date: Wed, 21 Feb 2024 Prob (F-statistic): 0.00950
Time: 14:09:19 Log-Likelihood: 6117.8
No. Observations: 14252 AIC: -1.223e+04
Df Residuals: 14251 BIC: -1.223e+04
Df Model: 1
Covariance Type: nonrobust
==============================================================================
coef std err t P>|t| [0.025 0.975]
------------------------------------------------------------------------------
OFI -0.0001 4.02e-05 -2.594 0.009 -0.000 -2.55e-05
==============================================================================
Omnibus: 8271.348 Durbin-Watson: 1.379
Prob(Omnibus): 0.000 Jarque-Bera (JB): 47669.229
Skew: -2.951 Prob(JB): 0.00
Kurtosis: 9.740 Cond. No. 1.00
==============================================================================
The correlation between OFI and return is -0.02, showing they have a weak negetive relationships. The regression coefficient
of OFI is also significant, meaning OFI has explainatry power for return. However, the model R2 is 0, indicating OFI may not be a
good or not the only explanation for return.
Construct a Predictive Trading Signal
BUT! The above analysis is in-sample. We want to see out-sample results.
In [9]:
In [*]:
# calculate one-second ahead return
second_data['ahead_return']=second_data['return'].shift(-1)
# Test whether OFI can explain FUTURE return variations
# Split sample to test and train samples again
train=second_data[:int(0.7*len(second_data))]
test=second_data[int(0.7*len(second_data)):]
# Test whether lagged OFI can predict FUTURE return
print(train[['ahead_return','OFI']].corr())
mod = sm.OLS(train['ahead_return'],train['OFI'],missing='drop')
res=mod.fit()
print(res.summary())
2/21/24, 6:10 PM Asg - Jupyter Notebook
localhost:8888/notebooks/Asg.ipynb 10/11
correlation on training data:
ahead_return OFI
ahead_return 1.000000 0.011125
OFI 0.011125 1.000000
Regression Results on training data:
OLS Regression Results
=======================================================================================
Dep. Variable: ahead_return R-squared (uncentered): 0.000
Model: OLS Adj. R-squared (uncentered): 0.000
Method: Least Squares F-statistic: 1.075
Date: Wed, 21 Feb 2024 Prob (F-statistic): 0.300
Time: 14:39:33 Log-Likelihood: 6121.3
No. Observations: 14252 AIC: -1.224e+04
Df Residuals: 14251 BIC: -1.223e+04
Df Model: 1
Covariance Type: nonrobust
==============================================================================
coef std err t P>|t| [0.025 0.975]
------------------------------------------------------------------------------
OFI 4.169e-05 4.02e-05 1.037 0.300 -3.71e-05 0.000
==============================================================================
Omnibus: 8281.146 Durbin-Watson: 1.379
Prob(Omnibus): 0.000 Jarque-Bera (JB): 47831.679
Skew: -2.955 Prob(JB): 0.00
Kurtosis: 9.754 Cond. No. 1.00
==============================================================================
The correlation between future return and OFI is 0.01, has a very weak positive relationship.However, the regression coeffici
ent is not significant, and the R2 is also 0.Lagged OFI can hardly predict FUTURE return.
In [*]:
correlation of predicted return from model and future return on test dataset:
predict_return ahead_return
predict_return 1.000000 0.027027
ahead_return 0.027027 1.000000
In [7]:
# explainatry power for test sample
test['predict_return']=res.predict(test[['OFI']])
test[['predict_return','ahead_return']].corr()
# plots cummulative return of the strategy using signal from past OFI
fig, ax = plt.subplots()
plt.plot((1+test['predict_return']).cumprod())
plt.xlabel("Minutes")
plt.ylabel("Cummulative return")
plt.title('Cummulative return')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.savefig('C:\\Users\\Desktop\\UCLATrading\\13.png')
plt.show()
2/21/24, 6:10 PM Asg - Jupyter Notebook
localhost:8888/notebooks/Asg.ipynb 11/11
