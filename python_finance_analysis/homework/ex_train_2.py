# 实验实训二 投资组合与分散风险
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

year = [2009, 2010, 2011, 2012, 2013]
ret_A = np.array([0.102, -0.02, 0.213, 0.12, 0.13])
ret_B = np.array([0.1062, 0.23, 0.045, 0.234, 0.113])
port_EW = (ret_A + ret_B) / 2.
print(np.round(np.mean(ret_A), 2), np.round(np.mean(ret_B), 2), np.round(np.mean(port_EW), 2))
print(np.round(np.std(ret_A), 2), np.round(np.std(ret_B), 2), np.round(np.std(port_EW), 2))
# 均值和标准差
# A B port
# 均值(0.11, 0.15, 0.13)
# 标准差(0.07, 0.07, 0.03)
# plots to 比较个股和投资组合的表现
plt.plot(year, ret_A, lw=2)
plt.plot(year, ret_B, lw=2)
plt.plot(year, port_EW, lw=2)
plt.figtext(0.2, 0.65, "Stock A")
plt.figtext(0.15, 0.4, "Stock B")
plt.xlabel("Year")
plt.ylabel("Returns")
plt.title("Indiviudal stocks vs. an equal-weighted 2-stock portflio")
plt.annotate('Equal-weighted Portfolio', xy=(2010, 0.1), xytext=(2011., 0),
             arrowprops=dict(facecolor='black', shrink=0.05), )
plt.ylim(-0.1, 0.3)
plt.show()
# plt.annotate? 注解
# correlation coefficient计算两只股票的相关系数，是-0.75
np.round(sp.corrcoef(ret_A, ret_B)[0, 1], 2)
# 协方差的计算
sp.cov(ret_A, ret_B)
# 相关系数的计算
sp.corrcoef(ret_A, ret_B)
