"""
实验实训八 投资组合
数据文件port.txt是记录了股票MSFT、INTC、KO和WMT在2018年每个交易日的收盘价。
（1）读取数据文件；
（2）基于收盘价，计算每只股票的日收益率；给出单只股票收益率的样本均值和样本标准差。
（3）考虑这四只股票的投资组合，在给定下面不同的权重向量条件下
（按股票顺序MSFT、INTC、KO和WMT），分别计算出投资组合日收益率的均值和标准差
    (a)权重向量是ω=(0.25,0.25,0.25,0.25)
    (b)权重向量是ω=(0.3,0.3,0.2,0.2)
    (c)权重向量是ω=(0.2,0.25,0.25,0.3)
    (d)权重向量是ω=(0.25,0.15,0.4,0.2)
（4）计算这四个组合的夏普比率（忽略无风险利率R_f），你会选哪个组合？
"""
import numpy as np
import pandas as pd

# (1)
data = pd.read_csv('../data/port.txt', index_col=0, parse_dates=True)
print('数据集信息')
data.info()
# (2)
temp = data.values[1:] / data.values[:-1] - 1  # 每只股票的日收益率
temp1 = np.log(data.values[1:] / data.values[:-1])  # 每只股票的日对数收益率
Ret = pd.DataFrame(temp, columns=data.columns, index=data.index[1:])
print('均值(%)')
print(np.round(Ret.mean() * 100, 2))
print('标准差(%)')
print(np.round(Ret.std() * 100, 2))
print('相关系数矩阵')
print(np.round(Ret.corr(), 2))
print('协方差矩阵(%)')
print(np.round((100 * Ret).cov(), 2))
# (3)
W = np.array([[0.25, 0.25, 0.25, 0.25],
              [0.3, 0.3, 0.2, 0.2],
              [0.2, 0.25, 0.25, 0.3],
              [0.25, 0.15, 0.4, 0.2]])
Port = temp.dot(W.T)
Port_mean = 100 * Port.mean(axis=0)
print('投资组合收益率均值(%)')
print(np.round(Port_mean, 2))
Port_std = 100 * Port.std(axis=0)
print('投资组合收益率标准差(%)')
print(np.round(Port_std, 2))
# (4)
Sharpe = Port_mean / Port_std
print('夏普比率(%)')
print(np.round(Sharpe * 100, 2))
