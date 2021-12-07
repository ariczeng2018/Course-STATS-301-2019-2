# 5. Roll模型来估算买卖价差
import numpy as np
import pandas as pd

price = pd.read_csv("../data/ibmRoll.txt", index_col=0, parse_dates=True)["Close"]
rtn = np.diff(price)
meanP = np.mean(price)
cov = np.cov(rtn[:-1], rtn[1:])
cov = round(cov[0, 1], 2)
roll = round(2 * (-cov)**0.5, 2)
spread_pct = round(2 * (-cov)**0.5 / meanP * 100, 2)
if cov < 0:
    print('roll', roll, '\n%spread', spread_pct)
else:
    print('cov', cov)
