# 实验实训六 Amihud(2002)模型估算反流动性指标
from datetime import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web

start = datetime(2013, 10, 1)
end = datetime(2013, 10, 31)
# data preparation
dat1 = web.DataReader(name='IBM', data_source='yahoo',
                      start=start, end=end)
dat1.to_csv("../ibmIlq.txt")
dat2 = web.DataReader(name='WMT', data_source='yahoo',
                      start=start, end=end)
dat2.to_csv("../wmtIlq.txt")
dat1 = pd.read_csv("../ibmIlq.txt", index_col=0, parse_dates=True)
p1 = dat1["Close"]
v1 = dat1['Volume']
R1 = (p1 - p1.shift(1)) / p1.shift(1)
ilq1 = np.mean(np.abs(R1) / (p1 * v1))
ticker1 = "IBM"
print("Illiquidity measure for ", ticker1, 'is', round(ilq1 * (10 ** 11), 2), 'e-11')
ticker2 = "WMT"
dat2 = pd.read_csv("../wmtIlq.txt", index_col=0, parse_dates=True)
p2 = dat2["Close"]
v2 = dat2['Volume']
R2 = (p2 - p2.shift(1)) / p2.shift(1)
ilq2 = np.mean(np.abs(R2) / (p2 * v2))
print("Illiquidity measure for ", ticker2, 'is', round(ilq2 * (10 ** 11), 2), 'e-11')
