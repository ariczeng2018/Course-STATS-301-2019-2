# 实验实训七 在险值VaR计算
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
"""
(1)读取数据文件
(2)基于收盘价，计算日对数收益率；
(3)假设日对数收益率均值为0，计算这一年内持有100股IBM股票在95%置信水平下的VaR；
(4)检验日对数收益率均值是否为0，给出检验结果；
(5)基于你的检验结果，是否需要对（3）的结果进行调整。
(6)正态性检验
"""


def normality_tests(arr):
    """
    Tests for normality distribution of given data set
    :param arr: ndarray
    object to generate statistics on
    :return:
    """
    print("Skew of data set  %14.3f" % stats.skew(arr))
    print("Skew test p-value %14.3f" % stats.skewtest(arr)[1])
    print("Kurt of data set  %14.3f" % stats.kurtosis(arr))
    print("Kurt test p-value %14.3f" % stats.kurtosistest(arr)[1])
    print("Normal test p-value %14.3f" % stats.normaltest(arr)[1])


data = pd.read_csv('../data/ibm2013daily.txt', index_col=0, parse_dates=True)
# 注意数据集里面包含了2012年最后一个交易日数据。
p = data['Close']
p0 = p[1]
Val = p0 * 100

ret = np.log(p / p.shift(1))
(100 * ret).describe()
ret = ret[1:]  # 去掉第一个缺失的收益率值
alpha = 0.01
n_days = len(ret)
z = stats.norm.ppf(1 - alpha)
VaR = Val * z * np.sqrt(n_days) * np.std(ret)

stats.ttest_1samp(ret, 0)
# 正态性检验：图示法--对数收益率的分位数-分位数图
sm.qqplot(ret, line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
plt.show()
"""
尽管图形方法很有吸引力，但是它们通常无法代替更严格的测试过程：
偏斜度测试（skewtest） 
测试样本的偏斜是否“正态”（也就是值足够接近0）
峰度测试（kurtosistest） 
测试样本的峰度是否“正态”（也就是值足够接近0）
正态性测试（normaltest） 
结合其他两种测试方法，检验正态性
"""
normality_tests(ret)
stats.normaltest(ret)
