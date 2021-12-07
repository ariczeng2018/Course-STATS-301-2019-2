# 实训一####	CAPM与线性回归
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

dat_IBM = pd.read_csv("../data/ibm2012daily.txt", index_col=0,
                      parse_dates=True)
dat_M = pd.read_csv("../data/sp2012daily.txt", index_col=0,
                    parse_dates=True)
# (2).compute daily returns (log)
Ret_IBM = (dat_IBM['Close'] - dat_IBM['Close'].shift(1)) / dat_IBM['Close'].shift(1)
ret_IBM = np.log(dat_IBM['Close'] / dat_IBM['Close'].shift(1))
Ret_M = (dat_M['Close'] - dat_M['Close'].shift(1)) / dat_M['Close'].shift(1)
ret_M = np.log(dat_M['Close'] / dat_M['Close'].shift(1))
# (3). linear regression based on CAPM using the (log return) data
results = (100 * Ret_IBM).describe()
y = np.array(Ret_IBM[1:])
x = np.array(Ret_M[1:])
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print("slope: %f    intercept: %f" % (slope, intercept))
# To get coefficient of determination (r_squared)
print("r-squared: %f" % r_value ** 2)
print("P value: %f" % round(p_value, 5))
plt.plot(x, y, 'o', label='original data')
plt.plot(x, intercept + slope * x, 'r', label='fitted line')
plt.legend()
plt.xlabel('market return')
plt.ylabel('stock return')
plt.show()
# 月回报率
# 计算2月对数回报率
temp = ret_IBM[ret_IBM.index.month == 2]
# 计算每个月的对数回报率
groups = ret_IBM.groupby(ret_IBM.index.month)
