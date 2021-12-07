import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader.data as web
import statsmodels.api as sm

start = datetime(2000, 1, 1)
end = datetime(2014, 9, 26)
DAX = web.DataReader(name='^GDAXI', data_source='yahoo',
                     start=start, end=end)
DAX.info()
DAX.tail()
dat_DAX = DAX.describe()
# Figure:Historical DAX index levels
DAX['Close'].plot(figsize=(8, 5), title='time series plot for DAX')
plt.ylabel('closing index level')
# compute daily log returns and draw
Ret = (DAX['Close'] - DAX['Close'].shift(1)) / DAX['Close'].shift(1)
ret = np.log(DAX['Close'] / DAX['Close'].shift(1))
DAX['Return'] = Ret
DAX['Log Return'] = ret
DAX[['Return', 'Log Return']].plot(subplots=True, style='b',
                                   figsize=(8, 5))
plt.subplot(2, 1, 1)
Ret.plot(figsize=(10, 5), title=' Return for DAX')
plt.subplot(2, 1, 2)
ret.plot(figsize=(10, 5), title='Log return for DAX')
# illustrates two stylized facts of equity returns:
# volatility clustering and leverage effect
# The DAX index and daily log returns
DAX[['Close', 'Log Return']].plot(subplots=True, style='b',
                                  figsize=(8, 5))
# moving average 
DAX['42d'] = DAX['Close'].rolling(42).mean()
a1 = DAX['42d']
DAX['252d'] = DAX['Close'].rolling(252).mean()
DAX[['Close', '42d', '252d']].tail()
# The DAX index and moving averages
DAX[['Close', '42d', '252d']].plot(figsize=(8, 5))
# the moving annual historical volatility
DAX['Mov_Vol'] = DAX['Return'].rolling(252).std() * math.sqrt(252)
DAX[['Close', 'Mov_Vol', 'Log Return']].plot(subplots=True, style='b',
                                             figsize=(8, 7))
# Estimating  betas 估算每年的贝塔值
start = datetime(2010, 1, 1)
end = datetime(2018, 12, 31)
# IBM
dat_IBM = web.DataReader(name='IBM', data_source='yahoo',
                         start=start, end=end)
# S&P500
dat_M = web.DataReader(name='^GSPC', data_source='yahoo',
                       start=start, end=end)
# (2).compute daily returns (log)
Ret_IBM = (dat_IBM['Close'] - dat_IBM['Close'].shift(1)) / dat_IBM['Close'].shift(1)
ret_IBM = np.log(dat_IBM['Close'] / dat_IBM['Close'].shift(1))
Ret_M = (dat_M['Close'] - dat_M['Close'].shift(1)) / dat_M['Close'].shift(1)
ret_M = np.log(dat_M['Close'] / dat_M['Close'].shift(1))
# method 1
y0 = Ret_IBM[1:]
x0 = Ret_M[1:]
d = x0.index.year
dd = set(d)
d1 = list(np.sort(list(dd)))
for i in d1:
    x1 = x0[x0.index.year == i]
    y1 = y0[y0.index.year == i]
    x1 = np.array(x1)
    y1 = np.array(y1)
    x1 = sm.add_constant(x1)
    model = sm.OLS(y1, x1).fit()
    print(i, round(model.params[1], 2))
