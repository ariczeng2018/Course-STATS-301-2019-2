# 实验实训三 历史价格数据基础统计分析
from datetime import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np

start = datetime(2012, 1, 1)
end = datetime(2012, 12, 31)

# 1.input data
dat_IBM = web.DataReader(name='IBM', data_source='yahoo',
                         start=start, end=end)
dat_WMT = web.DataReader(name='WMT', data_source='yahoo',
                         start=start, end=end)
dat_M = web.DataReader(name='^GSPC', data_source='yahoo',
                       start=start, end=end)

# 2.price and volume plot##
# plot for IBM
plt.subplot(2, 1, 1)
plt.plot(dat_IBM['Close'], 'b', label='price for IBM')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('price level')
plt.title('price and volume plot for IBM')
plt.subplot(2, 1, 2)
plt.plot(dat_IBM['Volume'], 'g', lw=1.5, label='volume for IBM')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('volume')
# plot for WMT
plt.subplot(2, 1, 1)
plt.plot(dat_WMT['Close'], 'b', label='price for WMT')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('price level')
plt.title('price and volume plot for WMT')
plt.subplot(2, 1, 2)
plt.plot(dat_WMT['Volume'], 'g', lw=1.5, label='volume for WMT')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('volume')
# plot for SP500
plt.subplot(2, 1, 1)
plt.plot(dat_M['Close'], 'b', label='price for S&P500')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('price level')
plt.title('price and volume plot for S&P500')
plt.subplot(2, 1, 2)
plt.plot(dat_M['Volume'], 'g', lw=1.5, label='volume for S&P500')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('volume')
# 3 returns and log returns
Ret_IBM = (dat_IBM['Close'].shift(1) - dat_IBM['Close']) / dat_IBM['Close']
ret_IBM = np.log(dat_IBM['Close'] / dat_IBM['Close'].shift(1))
Ret_WMT = (dat_WMT['Close'].shift(1) - dat_WMT['Close']) / dat_WMT['Close']
ret_WMT = np.log(dat_WMT['Close'] / dat_WMT['Close'].shift(1))
Ret_M = (dat_M['Close'].shift(1) - dat_M['Close']) / dat_M['Close']
ret_M = np.log(dat_M['Close'] / dat_M['Close'].shift(1))

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.hist(ret_IBM, 100)
plt.title('IBM return distribution')
plt.subplot(132)
plt.hist(ret_WMT, 100)
plt.title('WMT return distribution')
plt.subplot(133)
plt.hist(ret_M, 100)
plt.title('S&P500 return distribution')

plt.figure(figsize=(7, 5))
plt.subplot(211)
plt.plot(Ret_IBM, 'b', label='return for IBM')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('return')

plt.subplot(212)
plt.plot(ret_IBM, 'b', label='log return for IBM')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('log return')

# 4.绘制两只股票的收益率随时间变化的线图
plt.plot(ret_IBM, 'b', label='return for IBM')
plt.plot(ret_WMT, 'r', lw=1.5, label='return for WMT')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('return')

# 5.比较单只股票的收益和市场收益
n = min(len(ret_IBM), len(ret_M))
s = np.ones(n) * 2
t = range(n)
line = np.zeros(n)
plt.plot(t, ret_IBM[0:n], 'go', label='IBM')
plt.plot(t, ret_M[0:n], 'bd', label='Market')
plt.plot(t, line, 'r')

plt.legend(loc=0)
plt.axis('tight')
plt.xlim(1, n)
plt.ylim(-0.04, 0.07)
plt.title("Comparions between stock and market retuns")
plt.xlabel("Day")
plt.ylabel("Returns")
plt.show()
