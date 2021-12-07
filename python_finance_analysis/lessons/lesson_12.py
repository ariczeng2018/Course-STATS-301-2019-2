import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

dell = pd.read_csv("../data/ibm2012daily.txt", index_col=0, parse_dates=True)
dell['return'] = np.log(dell['Close'] / dell['Close'].shift(1))
dell[['Close', 'return']].plot(subplots=True, style='b',
                               figsize=(8, 5))
daily_v = np.std(dell['return'])
annual_v = daily_v * np.sqrt(len(dell['return']))
o1 = stats.shapiro(dell['return'][1:])
o2 = stats.bartlett(dell['return'][1:126], dell['return'][126:])
SP500 = pd.read_csv("../data/sp2012daily.txt", index_col=0, parse_dates=True)
ret = np.log(SP500['Close'] / SP500['Close'].shift(1))
plt.title('Illustration of volatility clustering (S&P500)')
plt.ylabel('Daily returns')
plt.xlabel('Date')
ret.plot()
# ARCH(1)
sp.random.seed(12345)
n = 1000  # n is the number of observations
n1 = 100  # we need to drop the first several observations
n2 = n + n1  # sum of two numbers
a = (0.1, 0.3)  # ARCH (1) coefficients alpha0 and alpha1, see Equation
errors = sp.random.normal(0, 1, n2)  # assume i.i.d standard normal distributed
t = sp.zeros(n2)
t[0] = sp.random.normal(0, sp.sqrt(a[0] / (1 - a[1])), 1)
for i in range(1, n2 - 1):
    t[i] = errors[i] * sp.sqrt(a[0] + a[1] * t[i - 1] ** 2)
y = t[n1 - 1:-1]  # drop the first n1 observations
plt.title('ARCH (1) process')
x = range(n)
plt.plot(x, y)
# GARCH(1,1)
sp.random.seed(12345)
n = 1000  # n is the number of observations
n1 = 100  # we need to drop the first several observations
n2 = n + n1  # sum of two numbers
alpha = (0.1, 0.3)  # GARCH (1,1) coefficients alpha0 and alpha1
beta = 0.2
errors = sp.random.normal(0, 1, n2)
t = sp.zeros(n2)
t[0] = sp.random.normal(0, sp.sqrt(a[0] / (1 - a[1])), 1)
for i in range(1, n2 - 1):
    t[i] = errors[i] * sp.sqrt(alpha[0] + alpha[1] * errors[i - 1] ** 2 + beta * t[i - 1] ** 2)
y = t[n1 - 1:-1]  # drop the first n1 observations
plt.title('GARCH (1,1) process')
x = range(n)
plt.plot(x, y)
plt.show()
