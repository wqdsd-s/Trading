
Q1: Data cleanning and data analysis
This exercise is about fixed income markets. Corporate bonds are largely traded in OTC markets. Academic Corporate Bond TRACE Dataset contains historic
transaction-level data on all eligible corporate bondsinvestment grade, high yield and convertible debt. We use this dataset to understand the bond market during the
COVID-19 Crisis.
bond.csv.zip is the dataset containing TRACE data downloaded from WRDS
VariableList.csv contains the variable description, and more detailed description is in TRACE Variable.pdf
I will not give you instructions to clean the data. You need to underrstand what variables to use and decide your way to handle the data
Data Cleaning
How many different companies and corporate bonds are in the data set?
Plot the histogram of the number of trading days
The data reports the contra-party type.
Calculate spread for each trade as follows. Note that we do not see bid/ask prices at OTC markets, so the calculation of spread is not direct. We follow the
calculation in
where Q is +1 for a customer buy and −1 for a customer sell. For each trade, we calculate its reference price as the volume-weighted average price of trades in
the same bond-day
Plot the histogram of calculated trade spread. Do you notice that 1) lots of spreads are exactly zero, 2) there are entries with very large spreads? Please answer
why those spreads are zero? Give one example to explain outlier spreads (check news and list one example that may lead to large spreads)
𝑡𝑝𝑠𝑓𝑎𝑒 = 2𝑄 ∗
𝑢𝑠𝑎𝑒𝑓𝑒𝑝𝑠𝑖𝑑𝑓 − 𝑠𝑓𝑔𝑓𝑠𝑓𝑛𝑑𝑓𝑝𝑠𝑖𝑑𝑓
𝑠𝑓𝑔𝑓𝑠𝑓𝑛𝑑𝑝𝑠𝑖𝑑𝑓
Analysis
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 3/16
Daily analysis
Spread
Aggregate spread information to the company-day level. What is the reasonable way in your opinions to do this?
Using bond-day level spreads to calculate the average spread for each stock and present the results. What can we learn from the ranking of the spreads?
Plot time-series spread using company-day level data for each company. What patterns do you see, and why is that?
Volume
Calculate company-day trading volume for each company
Plot histogram of the company-day trading volume. What is the distribution?
Analsyis
Does past trading volume predicts future spreads?
Are daily trading volume time-series correlated?
Intraday analysis
Spread
For each company, construct and plot the intraday spread pattern by minutes
Volume
For each company, construct and plot the intraday volume pattern by minutes
Analysis
Does the interday pattern change during market stress periods?
Is intraday volume predictable? (Note that you can also construct interday return information)
Import the relevant modules
In [*]:
How many different companies and corporate bonds are in the data set?
In [ ]:
Answer: companies:5 corporate bonds:156
Plot the histogram of the number of trading days:
In [ ]:
Answer:
### In this project, I will use datatable (quicker) rather pandas to manipulate data. You can see which one is more intuitive for you
from datetime import datetime
import pandas as pd
from regpyhdfe import Regpyhdfe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
data=pd.read_csv(r'C:\\Users\\Downloads\\TRACE\\TRACE\\bond.csv')
print(data['company_symbol'].nunique())
print(data['bond_sym_id'].nunique())
x= [datetime.strptime(str(d), '%Y%m%d').date() for d in data['trd_exctn_dt']]
plt.hist(x,bins=100)
plt.xlabel("Trade Date")
plt.ylabel("Frequency")
plt.show()
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 4/16
Calculate spread for each trade
In [*]:
Plot the histogram of calculated trade spread.
In [ ]:
Answer:
data.loc[data['ASCII_RPTD_VOL_TX'.lower()]=='1MM+','ASCII_RPTD_VOL_TX'.lower()]=1e12
data.loc[data['ASCII_RPTD_VOL_TX'.lower()]=='5MM+','ASCII_RPTD_VOL_TX'.lower()]=5e12
data['ASCII_RPTD_VOL_TX'.lower()]=data['ASCII_RPTD_VOL_TX'.lower()].astype(float)
group=data.groupby(['BOND_SYM_ID'.lower(),'TRD_EXCTN_DT'.lower()]).apply(lambda x:(x['ASCII_RPTD_VOL_TX'.lower()]*x['RPTD_PR'.lower()]).su
group.name='ref_pr'
data=data.merge(group,how='left',left_on=['BOND_SYM_ID'.lower(),'TRD_EXCTN_DT'.lower()],right_on=['BOND_SYM_ID'.lower(),'TRD_EXCTN_DT'.low
data["Q"]=data['side'].apply(lambda x: 1 if x=="B" else -1)
data["spread"]=2*data["Q"]*(data['RPTD_PR'.lower()]-data['ref_pr'])/data['ref_pr']
plt.hist(data["spread"],bins=100)
plt.xlabel("spread")
plt.ylabel("Frequency")
plt.show()
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 5/16
From the plot we know lots of spreads are near or exactly zero.
Why those spreads are zero:
Because when the bond market is efficient and there are lots of trading in bond market, the trade price would converge to an
equilibrium price. Or there would be arbitrage opportunities that will be eliminated quickly by investors.
Next get the maximum and minimum spreads and their information:
In [*]:
max spread: value:1.9999969862588658 bond_id: AMC4507267 company: AMC date: 20201019
min spread: value:-2.4444410958431844 bond_id: AMC4507267 company: AMC date: 20201019
Explanation of the outlier:
In 2020/10/19, AMC Entertainment announced their plan on reopening movie theaters which were thrashed by the pandemic.The reo
pening does not include movie theaters in New York City, but the gesture boosted AMC Entertainment’s stock 17% in early trading M
onday. Cinemark was up around 11% and Marcus Theatres hovered around 2.5% higher.This plan gave a positive expectation to the inv
estor and results in large spread of its bonds.
Aggregate spread information to the company-day level. What is the reasonable way in your opinions to do this?
From the histogram of spread above we know that most spread is near zero. Thus a reasonable way to summarize spread is to che
ck whether there are spread that is 3-sigma or even more far from zero and check its daily information. Also we can check the max
imum and minumum spread and its information.
Using bond-day level spreads to calculate the average spread for each stock and present the results. What can we learn from the ranking of the spreads?
In [ ]:
Below shows the ranked average spread of 156 bonds:
bond_sym_id
AAL3707053 -7.558334e-02
AMC4507267 -4.996530e-02
AMC4267538 -4.838985e-02
AMC4506547 -3.444428e-02
AMC5040765 -2.202850e-02
...
TSLA4265473 1.565588e-16
TSLA4474416 1.483076e-04
AAL4288272 4.020696e-04
AAL3671382 7.102421e-04
TSLA4103351 1.315150e-03
Length: 156, dtype: float64
From the results we know: bond AAL3707053 has the minumum average spread and its negative, followed by AMC bonds; three TSLA and
two AAL bonds has the largest spread. However, all the bonds average spread is close to zero.
Plot time-series spread using company-day level data for each company. What patterns do you see, and why is that?
print(max(data["spread"]),data.loc[data["spread"].idxmax(),'BOND_SYM_ID'.lower()],data.loc[data["spread"].idxmax(),'COMPANY_SYMBOL'.lower(
print(min(data["spread"]),data.loc[data["spread"].idxmin(),'BOND_SYM_ID'.lower()],data.loc[data["spread"].idxmin(),'COMPANY_SYMBOL'.lower(
ave=data.groupby(['bond_sym_id']).apply(lambda x:x['spread'].mean())
ave.sort_values(inplace=True)
print(ave)
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 6/16
In [ ]:
AAL:
AAPL:
AMC:
ave=data.groupby(['COMPANY_SYMBOL'.lower(),'TRD_EXCTN_DT'.lower()]).apply(lambda x:x["spread"].mean())
com_index_ave=np.array([i[0] for i in ave.index])
date_index_ave=np.array([i[1] for i in ave.index])
com_list=np.unique(com_index_ave)
for c in com_list:
temp=ave[pd.Series(com_index_ave==c,index=ave.index)]
temp.index=[datetime.strptime(str(d), '%Y%m%d').date() for d in date_index_ave[com_index_ave==c]]
plt.plot(temp)
plt.xlabel("Trade Date")
plt.ylabel("Spread")
plt.title(c)
plt.savefig(r'C:\\Users\\Desktop\\UCLATrading\\'+c+'1.png')
plt.show()
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 7/16
AMZN:
TSLA:
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 8/16
From the plots of each company's time-series spread we can see:
The daily spread began to oscillate violently after 2020-03 and began to go steady after 2020-11. During this period, the spr
ead has an obvious larger variance than the other periods. And also there were a huge plumb in spread after 2020-03. This is conc
ordant with the pandemic.
Calculate company-day trading volume for each company
In [ ]:
Plot histogram of the company-day trading volume. What is the distribution?
In [ ]:
AAL:
AAPL:
ave=data.groupby(['COMPANY_SYMBOL'.lower(),'TRD_EXCTN_DT'.lower()]).apply(lambda x:x["ASCII_RPTD_VOL_TX".lower()].sum())
com_index_ave=np.array([i[0] for i in ave.index])
date_index_ave=np.array([i[1] for i in ave.index])
com_list=np.unique(com_index_ave)
for c in com_list:
temp=ave[pd.Series(com_index_ave==c,index=ave.index)]
temp.index=[datetime.strptime(str(d), '%Y%m%d').date() for d in date_index_ave[com_index_ave==c]]
plt.hist(temp,bins=100)
plt.xlabel("Volume")
plt.ylabel("Frequency")
plt.title(c)
plt.savefig(r'C:\\Users\\alex\\Desktop\\UCLATrading\\'+c+'2.png')
plt.show()
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 9/16
AMC:
AMZN:
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 10/16
TSLA:
From the plots above we can see: most daily total trading volume is less than 1e14. For AAL and TSLA, most daily trading volu
me is under 0.5*1e14. The higher total traing volume, the lower frequency. It shows a pattern of poisson distribution.
Does past trading volume predicts future spreads?
In [ ]:
Calculated the correlation of bond daily spread and its lagged daily trading volume above.
Result:
lag1: -0.0022131992852637227
lag5: -0.00010917798908216872
lag20: -0.002627849657254112
This means that past trading volume is nearly non-correlated with future spread, which hardly provides any power of prediction.
Are daily trading volume time-series correlated?
In [ ]:
daily_spread=data.groupby(['bond_sym_id','TRD_EXCTN_DT'.lower()]).apply(lambda x:x["spread"].mean())
daily_vol=data.groupby(['bond_sym_id','TRD_EXCTN_DT'.lower()]).apply(lambda x:x["ASCII_RPTD_VOL_TX".lower()].sum())
bond_day=pd.DataFrame()
bond_day['spread']=daily_spread
bond_day['vol']=daily_vol
bond_day['lag1_vol']=bond_day['vol'].shift()
bond_day['lag5_vol']=bond_day['vol'].shift(5)
bond_day['lag20_vol']=bond_day['vol'].shift(20)
print(bond_day['spread'].corr(bond_day['lag1_vol']))
print(bond_day['spread'].corr(bond_day['lag5_vol']))
print(bond_day['spread'].corr(bond_day['lag20_vol']))
vol=data.groupby(['COMPANY_SYMBOL'.lower(),'TRD_EXCTN_DT'.lower()]).apply(lambda x:x["ASCII_RPTD_VOL_TX".lower()].sum())
com_index_vol=np.array([i[0] for i in vol.index])
date_index_vol=np.array([i[1] for i in vol.index])
com_list=np.unique(com_index_vol)
df=pd.DataFrame()
for c in com_list:
temp=vol[pd.Series(com_index_vol==c,index=vol.index)]
temp.index=[datetime.strptime(str(d), '%Y%m%d').date() for d in date_index_vol[com_index_vol==c]]
df[c]=temp
print(df.corr())
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 11/16
The correlation matrix of daily trading volume between companys are:
AAL AAPL AMC AMZN TSLA
AAL 1.000000 -0.023092 -0.147003 0.174184 -0.175688
AAPL -0.023092 1.000000 0.329840 0.191962 -0.002591
AMC -0.147003 0.329840 1.000000 0.161806 0.136779
AMZN 0.174184 0.191962 0.161806 1.000000 -0.064176
TSLA -0.175688 -0.002591 0.136779 -0.064176 1.000000
From the correlaton matrix we know: the correlation between AMC and AAPL is the largest, its 0.33, it has a positive correlation.
The correlation between AMZN and AAPL is also positive(0.19). Correlation between AMZN and AAL is 0.17. Correlation betwwen TSLA
and AAL is negative(-0.18), they are negatively correlated.
For each company, construct and plot the intraday spread pattern by minutes
In [ ]:
The plots below shows each days' minute frequency average spread for each company. Each line represents a single day.
AAL:
AAPL:
data['minute']=data['TRD_EXCTN_TM'.lower()].apply(lambda x:x[:-3])
data['minute']=data['TRD_EXCTN_TM'.lower()].apply(lambda x:datetime.strptime(x[:-3], '%H:%M'))
spread=data.groupby(['COMPANY_SYMBOL'.lower(),'TRD_EXCTN_DT'.lower(),'minute']).apply(lambda x:x["spread"].mean())
com_index_spr=np.array([i[0] for i in spread.index])
com_list=np.unique(com_index_spr)
for c in com_list:
temp=spread[pd.Series(com_index_spr==c,index=spread.index)]
gr=temp.groupby(['TRD_EXCTN_DT'.lower()])
fig, ax = plt.subplots()
for i,j in gr:
minu=[k[2] for k in j.index]
j.index=minu
ax.plot(j)
plt.xlabel("minute")
plt.ylabel("Spread")
plt.title(c)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.savefig(r'C:\\Users\\Desktop\\UCLATrading\\'+c+'3.png')
plt.show()
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 12/16
AMC:
AMZN:
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 13/16
TSLA:
For each company, construct and plot the intraday volume pattern by minutes
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 14/16
In [ ]:
The plots below shows each days' minute frequency trading volume for each company. Each line represents a single day
AAL:
AAPL:
data['minute']=data['TRD_EXCTN_TM'.lower()].apply(lambda x:x[:-3])
data['minute']=data['TRD_EXCTN_TM'.lower()].apply(lambda x:datetime.strptime(x[:-3], '%H:%M'))
vol=data.groupby(['COMPANY_SYMBOL'.lower(),'TRD_EXCTN_DT'.lower(),'minute']).apply(lambda x:x["ASCII_RPTD_VOL_TX".lower()].sum())
com_index_vol=np.array([i[0] for i in vol.index])
com_list=np.unique(com_index_vol)
for c in com_list:
temp=vol[pd.Series(com_index_vol==c,index=vol.index)]
gr=temp.groupby(['TRD_EXCTN_DT'.lower()])
fig, ax = plt.subplots()
for i,j in gr:
minu=[k[2] for k in j.index]
j.index=minu
ax.plot(j)
plt.xlabel("minute")
plt.ylabel("Spread")
plt.title(c)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.savefig(r'C:\\Users\\alex\\Desktop\\UCLATrading\\'+c+'4.png')
plt.show()
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 15/16
AMC:
AMZN:
2/8/24, 5:33 PM Asg2 - Jupyter Notebook
localhost:8888/notebooks/Asg2.ipynb# 16/16
TSLA:
Does the interday pattern change during market stress periods?
From the interday spread plots of each company above we can see:
In most days the spread were oscillating more voilently in market stress periods, indicating a higher variance. Even some hug
e change occurred in stress periods. This means that interday spread could change with market trading periods.
Is intraday volume predictable? (Note that you can also construct interday return information)
From the interday total trading volume plots of each company above we can see:
For most days the trading volume has the same pattern, it peaks after 15:00 or in the morning(around 9:00 AM) and most tradin
g occures during the day. It also shares a similar pattern with interday spread and return.
