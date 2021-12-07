from lessons.IRR import npv_f
from lessons.IRR import IRR_f
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy as sp
from scipy.interpolate import interp1d
import scipy.optimize as optimize


def my_f(x):
    return 3 + x ** 2


def SSE(beta):
    return ((y - beta * x_) * (y - beta * x_)).sum()


cashflows = [-100, 20, 40, 50, 20, 10]
rate = 0.05
total = npv_f(rate, cashflows)
rate = IRR_f(cashflows)
""" 列表操作 """
ls = [425, "BIT", [10, "CS"], 425]
lt = ls
ls[0] = 44
ls1 = [1, 23, 5]
ls = ls1
ls.clear()
lt[0] = 12
ls = lt
ls[0] = 13
ls.append(89)
ls.insert(2, 909)
ltt = ls.copy()
ls.pop(2)
ls.reverse()
ls.remove(89)
list(range(5))
for e in ls:
    print(e, "\n", "")
aa = list(range(10))
aa_len = len(aa)
aa_sum = 0
for a in aa:
    aa_sum += a
aa_mean = aa_sum / aa_len
""" 编写欧式看涨期权定价公式"""
dir(stats)
help(stats.linregress)
help(stats.norm.cdf)
temp = stats.norm.cdf(np.arange(-1, 1, 0.1))
temp_sim = stats.norm.rvs(0, 0.1, 1000)
plt.hist(temp_sim)
plt.plot(temp_sim)

x = np.array([[1, 2, 3], [3, 4, 6]])  # 2 by 3 matrix
z = np.random.rand(50)
y = np.random.normal(size=100)
r = np.array(range(1, 100), float) / 100

cashflows = [50, 40, 20, 10, 50]
npv = sp.npv(0.1, cashflows)  # estimate NPV
cashflows = [-100, 50, 40, 20, 10, 50]
print(round(npv, 2))
npv = sp.npv(0.1, cashflows[1:]) + cashflows[0]
payment = sp.pmt(0.045 / 12, 30 * 12, 250000)
print(round(npv, 2))
ret = sp.array([0.1, 0.05, -0.02])
a = np.zeros(10)  # array with 10 zeros
b = np.zeros((3, 2), dtype=float)  # 3 by 2 with zeros
c = np.ones((4, 3), float)  # 4 by 3 with all ones
d = np.array(range(10), float)  # 0,1, 2,3 .. up to 9
e1 = np.identity(4)  # identity 4 by 4 matrix
e2 = np.eye(4)  # same as above
e3 = np.eye(4, k=1)  # index of the diagonal
f = np.arange(1, 20, 3, float)  # from 1 to 19, interval 3
g = np.array([[2, 2, 2], [3, 3, 3]])  # 2 by 3
h = np.zeros_like(g)  # all zeros
i = np.ones_like(g)  # all ones
pv = np.array([[100, 10, 10.2], [34, 22, 34]])  # 2 by 3
x_ = pv.flatten()  # matrix becomes a vector
vp2 = np.reshape(x_, [3, 2])
A = np.array([[1, 2, 3], [2, 1, 3]])
B = np.array([[1, 2, 3], [3, 1, 2]])
Bt = np.transpose(B)
np.dot(A, Bt)
A1 = np.array([[1, 2, 3], [2, 1, 3]])
B1 = np.array([[1, 2, 3], [3, 1, 2]])
print(np.matmul(A1, np.transpose(B1)))
dataset = np.array(np.random.normal(size=10))
for data in dataset:
    print(data)
# cumulative standard normal distribution
print(sp.stats.norm.cdf(0))
# Generate some data:
np.random.seed(12345678)
x_ = np.random.random(10)
y = 1.6 * x_ + np.random.random(10)
# Perform the linear regression:
slope, intercept, r_value, p_value, std_err = stats.linregress(x_, y)
print("slope: %f    intercept: %f" % (slope, intercept))
# To get coefficient of determination (r_squared):
print("r-squared: %f" % r_value ** 2)
# Plot the data along with the fitted line:
plt.plot(x_, y, 'o', label='original data')
plt.plot(x_, intercept + slope * x_, 'r', label='fitted line')
plt.legend()
plt.show()
# interpolate
x_ = np.linspace(0, 10, 10)  # generate 10 evenly spaced numbers from (0,10)
y = np.exp(-x_ / 3.0)
f = interp1d(x_, y)
f2 = interp1d(x_, y, kind='cubic')
xnew = np.linspace(0, 10, 40)  # 40values from (0,10)
plt.plot(x_, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()
# optimize
x1 = optimize.fmin(my_f, 5)
x2 = optimize.fmin(SSE, 0)
