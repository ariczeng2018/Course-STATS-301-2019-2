import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv(r"../data/ibm2013daily.txt", index_col=0)  # 数据准备
rtn = np.diff(np.log(data['Close']))  # 日对数收益率
plt.plot(rtn)
plt.title('time series plot for IBM daily log return')
plt.show()
rtn = pd.DataFrame(data=rtn, columns=['rtn'])  # 月回报率
rtn.index = data.index[1:]
rtn_m = rtn.groupby(pd.DatetimeIndex(rtn.index).month).sum()
rtn_m.plot(title='time series plot for IBM monthly log return',
                 grid=True, marker='o', linestyle='')
rtn_test = stats.ttest_1samp(rtn, 0)  # 检验日的户数回报率是否等于0
print(rtn_test)
data_ = pd.read_csv('../data/IBM1962end.csv', index_col=0, parse_dates=True)  # 探索问题数据
rtn_ = np.diff(np.log(data_['Close']))
rtn_ = pd.DataFrame(data=rtn_, columns=['rtn'])
rtn_.fillna(0)
rtn_.index = pd.DatetimeIndex(data_.index[1:])
rtn_jan = rtn_[rtn_.index.month == 1]
rtn_others = rtn_[rtn_.index.month != 1]
bartlett_test = stats.bartlett(rtn_jan.values[:, -1], rtn_others.values[:, -1])  # 检验一月现象
print(bartlett_test)
