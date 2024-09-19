
This exercise is about Bitcoin. The data records all trades at Coinbase in March 2020. Unluckily, we only see market price and quantity traded. But the data gives the
indicator of “BUY” or “SELL” on the taker side. This indicator is not available for most of equity trades dataset.
Questions:
Calculate Kyle’s lambda (market impact) for all the data in March
Calculate Kyle’s lambda (market impact), volume, signed volume, volume-weighted price by day and hour
Plot constructed variables in the previous bullet point
What patterns do you see? Make some comments
Is there momentum in Bitcoin returns?
Can you figure out good predictor variables for returns?
Construct a trading strategy based on your analysis
Plot the performance of the trading strategy
Import the relevant modules
In [*]:
In [ ]:
In [ ]:
Calculate Kyle’s lambda (market impact) for all the data in March
import pandas as pd
pd.options.mode.chained_assignment = None # default='warn'
import numpy as np
import yfinance as yf
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import os
import math
from datetime import datetime
# Firstly automatically decompress the zip and gz file and extract all the csv file into one folder
import zipfile
import gzip
zipf=zipfile.ZipFile(r'C:\\Users\\alex\\Downloads\\BTC_coinbase_trades.zip')
zipf.extractall(r'C:\\Users\\alex\\Downloads\\BTC_coinbase_trades')
zipf.close()
path=r'C:\\Users\\alex\\Downloads\\BTC_coinbase_trades\\BTC_coinbase_trades/'
out_path=r'C:\\Users\\alex\\Downloads\\BTCdata/'
os.mkdir(out_path)
dir_list=[]
for root, dirs, files in os.walk(path):
dir_list.append(dirs)
for dirs in dir_list[0]:
path1=path+dirs
for root, dirs1, files in os.walk(path1):
gzfile=gzip.GzipFile(path1+'/'+files[-1])
open(out_path+dirs+files[-1].replace(".gz",""),"wb+").write(gzfile.read())
gzfile.close()
# Read in all files in the folder (note that some files have difference sizes)
for root, dirs, files in os.walk(out_path):
pass
data=pd.read_csv(out_path+files[0],delimiter=";")
for f in files[1:]:
data=data.append(pd.read_csv(out_path+f,delimiter=";"))
data.index=range(len(data))
3/4/24, 3:10 PM Asg(1) - Jupyter Notebook
localhost:8888/notebooks/Asg(1).ipynb#Price-impact-by-hour 2/7
In [*]:
Kyle's lambda in March: 27.59198642848665
Calculate Kyle’s lambda (market impact), volume, signed volume, volume-weighted price by day and hour
In [*]:
Plot constructed variables in the previous bullet point
In [*]:
Kyle's lambda by hour:
data['order']=data['taker_side'].apply(lambda x:1 if x=="BUY" else -1)
data['rand_demand']=np.random.normal(0, 1, len(data))
data['agg_flow']=data['order']+data['rand_demand']
data['cons']=1
mod = sm.OLS(data['price'],data[['cons','agg_flow']],missing='drop')
res=mod.fit()
Klambda_all=res.params[1]
print(Klambda_all)
data['time']=[datetime.strptime(d[:-8],'%Y-%m-%dT%H:%M:%S') for d in data['time_exchange']]
data['day']=data['time'].apply(lambda x:x.date())
data['hour']=data['time_exchange'].apply(lambda x:datetime.strptime(x[:13],'%Y-%m-%dT%H'))
Klambda_hour=pd.Series()
volume_hour=pd.Series()
sign_volume_hour=pd.Series()
volumewei_price_hour=pd.Series()
for i,j in data.groupby(['hour']):
#calculate Kyle's lambda
mod = sm.OLS(j['price'],j[['cons','agg_flow']],missing='drop')
Klambda_hour[i]=mod.fit().params[1]
#volume
volume_hour[i]=j['base_amount'].sum()
#signed volume
sign_volume_hour[i]=(j['order']*j['base_amount']).sum()
#volume-weighted price
volumewei_price_hour[i]=(j['base_amount']*j['price']).sum()/j['base_amount'].sum()
for i,j in [[Klambda_hour,'Kyle lambda'],[volume_hour,'volume'],[sign_volume_hour,'signed volume'],[volumewei_price_hour,'volume-weighted
fig, ax = plt.subplots()
ax.plot(i)
plt.xlabel("hour")
plt.ylabel(j)
plt.title(j)
plt.xticks(rotation=90.)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.savefig(r'C:\\Users\\alex\\Desktop\\UCLATrading\\'+j+'5.png')
plt.show()
3/4/24, 3:10 PM Asg(1) - Jupyter Notebook
localhost:8888/notebooks/Asg(1).ipynb#Price-impact-by-hour 3/7
volume by hour:
signed volume by hour:
3/4/24, 3:10 PM Asg(1) - Jupyter Notebook
localhost:8888/notebooks/Asg(1).ipynb#Price-impact-by-hour 4/7
volume-weighted price by hour:
What patterns do you see? Make some comments
From the hourly plots shown above we know that:
For Kyle's lambda, its expectation is close to zero, however it seems to have a bigger variance in around 3.13 and there were
extreme values around 3.13.
For trading volume, it also has much bigger variance between 3.13 and 3.25.
For signed volume, it oscillated around zero but also has bigger variance after 3.13.
For volume-weighted price, there was a huge plunge began from 3.7 and reached the bottom between 3.13 and 3.17, after that th
ere were a upgoing trend.
Both Kyle's lambda and volume had reaction to drastic change in bitcion price.
Is there momentum in Bitcoin returns?
3/4/24, 3:10 PM Asg(1) - Jupyter Notebook
localhost:8888/notebooks/Asg(1).ipynb#Price-impact-by-hour 5/7
In [*]:
We calculate for the correlation between current Bitcoin return and lagged Bitcoin return(1,5,20,50,100 hour ago):
hour_return ... hour_return_lag100
hour_return 1.000000 ... -0.084665
hour_return_lag1 0.110304 ... -0.022533
hour_return_lag5 0.069197 ... -0.000719
hour_return_lag20 -0.065786 ... 0.008280
hour_return_lag50 0.018405 ... 0.022100
hour_return_lag100 -0.084665 ... 1.000000
From the correlation matrix we know that for 1 hour lag, the correlation is 0.11, indicating a positive relationship; and whe
n lag become larger, the correlation drops and finally becomes negative.
Thus we know that there is a short term(within 5 hours) momentum effect and long term(more than 20 hours) reversal effect in
Bitcoin returns in March 2020.
Can you figure out good predictor variables for returns?
In [*]:
hour_return=(volumewei_price_hour-volumewei_price_hour.shift())/volumewei_price_hour.shift()
hour_return=pd.DataFrame(hour_return)
hour_return.columns=['hour_return']
hour_return['hour_return_lag1']=hour_return['hour_return'].shift()
hour_return['hour_return_lag5']=hour_return['hour_return'].shift(5)
hour_return['hour_return_lag20']=hour_return['hour_return'].shift(20)
hour_return['hour_return_lag50']=hour_return['hour_return'].shift(50)
hour_return['hour_return_lag100']=hour_return['hour_return'].shift(100)
print(hour_return.corr())
data_hour=pd.DataFrame(Klambda_hour,columns=['Klambda_hour'])
data_hour['volume_hour']=volume_hour
data_hour['sign_volume_hour']=sign_volume_hour
data_hour['volumewei_price_hour']=volumewei_price_hour
data_hour=data_hour.merge(hour_return,how='left',left_index=True, right_index=True)
data_hour['future_return']=data_hour['hour_return'].shift(-1)
data_hour['Klambda_lag1']=data_hour['Klambda_hour'].shift(1)
data_hour['Klambda_lag5']=data_hour['Klambda_hour'].shift(5)
data_hour['Klambda_lag20']=data_hour['Klambda_hour'].shift(20)
data_hour['Klambda_lag50']=data_hour['Klambda_hour'].shift(50)
data_hour['Klambda_lag100']=data_hour['Klambda_hour'].shift(100)
data_hour['sign_volume_lag1']=data_hour['sign_volume_hour'].shift()
data_hour['sign_volume_lag5']=data_hour['sign_volume_hour'].shift(5)
data_hour['sign_volume_lag20']=data_hour['sign_volume_hour'].shift(20)
data_hour['sign_volume_lag50']=data_hour['sign_volume_hour'].shift(50)
data_hour['sign_volume_lag100']=data_hour['sign_volume_hour'].shift(100)
colname=['hour_return']+list(data_hour.columns[:4])+list(data_hour.columns[5:])
data_hour=data_hour[colname]
print(data_hour.corr())
3/4/24, 3:10 PM Asg(1) - Jupyter Notebook
localhost:8888/notebooks/Asg(1).ipynb#Price-impact-by-hour 6/7
We check the IC(correlation) between different lagged variable and Bitcoin return:
hour_return ... sign_volume_lag100
hour_return 1.000000 ... -0.080204
Klambda_hour -0.084187 ... 0.046199
volume_hour -0.144472 ... -0.081659
sign_volume_hour 0.604910 ... -0.025037
volumewei_price_hour 0.016654 ... -0.041510
hour_return_lag1 0.110304 ... 0.041147
hour_return_lag5 0.069197 ... 0.054663
hour_return_lag20 -0.065786 ... -0.006734
hour_return_lag50 0.018405 ... 0.020966
hour_return_lag100 -0.084665 ... 0.602679
future_return 0.110304 ... 0.011734
Klambda_lag1 0.066917 ... -0.086713
Klambda_lag5 0.036654 ... -0.015574
Klambda_lag20 0.045091 ... 0.004792
Klambda_lag50 0.074856 ... 0.017006
Klambda_lag100 0.059275 ... 0.008610
sign_volume_lag1 0.172308 ... -0.004778
sign_volume_lag5 0.109187 ... 0.052213
sign_volume_lag20 -0.020432 ... 0.001884
sign_volume_lag50 0.015226 ... -0.006541
sign_volume_lag100 -0.080204 ... 1.000000
From the correlation matrix we can see: 1 hour lagged return(0.11), 1 and 50 hour lagged Kyle's lambda(0.06 and 0.07), and 1
hour lagged signed volume(0.17) has the best IC. They are all positively related with future returns.
Construct a trading strategy based on your analysis
Regress the bitcoin return on signed trading volume (buy - sell) 1 hour ago and Kyle's lambda in 1 hour ago. (Use no constant
term OLS model). The model assume lagged signed trading volume and Kyle's lambda are the two predictive power of future return. T
hen use current signed trading volume and current Kyle's lambda to estimate for future return.
The model was attained by trying different combinanations of predictive variables between the high IC variables shown above(f
or example: Klambda_lag1,sign_volume_lag1,hour_return_lag1).
In [ ]:
Plot the performance of the trading strategy
In [ ]:
mod = sm.OLS(data_hour['hour_return'],data_hour[['sign_volume_lag1','Klambda_lag1']],missing='drop')
res=mod.fit()
result=(res.predict(data_hour[['sign_volume_hour','Klambda_hour']])+1).cumprod()
fig, ax = plt.subplots()
ax.plot(result)
plt.xlabel("hour")
plt.ylabel('cummulative return')
plt.title('performance')
plt.xticks(rotation=90.)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.savefig(r'C:\\Users\\alex\\Desktop\\UCLATrading\\'+'666.png')
plt.show()
3/4/24, 3:10 PM Asg(1) - Jupyter Notebook
localhost:8888/notebooks/Asg(1).ipynb#Price-impact-by-hour 7/7
