# Generating random numbers from a standard normal distribution
from math import sqrt, pi, exp
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import scipy as sp


def bsm_call_value(S, X, T, r, sigma):
    from scipy import log, exp, sqrt, stats
    d1 = (log(S / X) + (r + sigma * sigma / 2.) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * stats.norm.cdf(d1) - X * exp(-r * T) * stats.norm.cdf(d2)


x1 = sp.random.standard_normal(size=10)
x2 = sp.random.normal(size=10)
x3 = sp.random.normal(0, 1, 10)
# Histogram for a normal distribution
sp.random.seed(12345)
x = sp.random.normal(0.08, 0.2, 5000)
plt.hist(x, 50, density=True)
xx = np.sort(x)
mu = 0.08
sigma = 0.2
y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(xx - mu) ** 2 / (2 * sigma * sigma))
plt.plot(xx, y)
y1 = sp.random.uniform(5, 6, 10000)
plt.hist(y1, 50, density=True)
np.mean(y1)
np.var(y1)
# Graphical presentation of a lognormal distribution
# 当股票的收益率服从正态分布时，其价格服从对数正态分布
x = np.linspace(0, 3, 200)
mu = 0
sigma0 = [0.25, 0.5, 1]
color = ['blue', 'red', 'green']
target = [(1.2, 1.3), (1.7, 0.4), (0.18, 0.7)]
start = [(1.8, 1.4), (1.9, 0.6), (0.18, 1.6)]
for i in range(len(sigma0)):
    sigma = sigma0[i]
    y = 1 / (x * sigma * np.sqrt(2 * pi)) * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma * sigma))
    plt.annotate('mu=' + str(mu) + ', sigma=' + str(sigma), xy=target[i],
                 xytext=start[i], arrowprops=dict(facecolor=color[i], shrink=0.01))
    plt.plot(x, y, color[i])
plt.title('Lognormal distribution')
plt.xlabel('x')
plt.ylabel('lognormal density distribution')
'''
练习：可视化来自如下分布的随机数
1.标准正态分布
2.均值为100，标准差是20的正态分布
3.自由度是0.5的卡方分布
4.lamba值是1.0的泊松分布
'''
sample_size = 500
rn1 = npr.standard_normal(sample_size)
rn2 = npr.normal(100, 20, sample_size)
rn3 = npr.chisquare(df=0.5, size=sample_size)
rn4 = npr.poisson(lam=1.0, size=sample_size)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                             figsize=(7, 7))
ax1.hist(rn1, bins=25)
ax1.set_title('standard normal')
ax1.set_ylabel('frequency')
ax1.grid(True)
ax2.hist(rn2, bins=25)
ax2.set_title('normal(100, 20)')
ax2.grid(True)
ax3.hist(rn3, bins=25)
ax3.set_title('chi square')
ax3.set_ylabel('frequency')
ax3.grid(True)
ax4.hist(rn4, bins=25)
ax4.set_title('Poisson')
ax4.grid(True)
plt.show()
'''
permutation(): 产生不同排列
import numpy as np
x=range(1,11)
print(list(x))
for i in range(5):
     y=np.random.permutation(x)
     print(y)
模拟到期日股价---随机变量的分布
'''
S0 = 100  # initial value
r = 0.05  # constant short rate
sigma = 0.25  # constant volatility
T = 2.0  # in years
i_ = 10000  # number of random draws
ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T +
                  sigma * np.sqrt(T) * npr.standard_normal(i_))
ST2 = S0 * npr.lognormal((r - 0.5 * sigma ** 2) * T,
                         sigma * np.sqrt(T), size=i_)
plt.subplot(2, 1, 1)
plt.hist(ST1, bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.hist(ST2, bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.grid(True)
plt.show()
'''
模拟股价的路径，随机过程角度(随机微分方程SDE)
股价随着时间变化的样本路径
'''
i_ = 10000
M = 50
S0 = 100  # initial value
r = 0.05  # constant short rate
sigma = 0.25  # constant volatility
T = 2.0  #
dt = T / M
S = np.zeros((M + 1, i_))
S[0] = S0
for t in range(1, M + 1):
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                             sigma * np.sqrt(dt) * npr.standard_normal(i_))
plt.plot(S[:, :10], lw=1.5)
plt.xlabel('time')
plt.ylabel('price level')
plt.grid(True)
plt.show()
'''
用模拟法为看涨期权定价
在已知年波动率sigma、无风险利率r、执行价格X和时间T（年）前提下，
只要知道期权到期日的股票价格，就可以
计算一个看涨期权在到期日的收益
'''
S0 = 40.  # stock price at time zero
X = 40.  # exercise price
T = 0.5  # years
r = 0.05  # risk-free rate
sigma = 0.2  # volatility (annual)
n_steps = 100.  # number of steps
sp.random.seed(12345)  # fix those random numbers
n_simulation = 5000  # number of simulation
dt = T / n_steps
call = sp.zeros([n_simulation], dtype=float)
x = range(0, int(n_steps), 1)
for j in range(0, n_simulation):
    sT = S0
    for i in x[:-1]:
        e = sp.random.normal()
        sT *= exp((r - 0.5 * sigma * sigma) * dt + sigma * e * sqrt(dt))
    call[j] = max(sT - X, 0)
call_price = sp.mean(call) * sp.exp(-r * T)
print('call price = ', round(call_price, 2))
o1 = bsm_call_value(S0, X, T, r, sigma)
