# Chapter 8  Financial time series
import numpy as np
import scipy as sp
import pandas as pd

from datetime import datetime

start = datetime(2014, 5, 1)
end = datetime(2014, 6, 30)

import pandas_datareader.data as web

dat_DAX = web.DataReader(name='^GDAXI', data_source='yahoo',
                         start=start, end=end)

# help(pd.DataFrame)
df = pd.DataFrame([10, 20, 30, 40], columns=['numbers'],
                  index=['a', 'b', 'c', 'd'])
df

df.index  # the index values
df.columns  # the column names
df.values

df.loc['c']  # selection via index
df.loc[['a', 'd']]  # selection of multiple indices

df.loc[df.index[1:3]]  # selection via Index object

df['numbers']

df.sum()  # sum per column
df.cumsum()
# ==========


a1 = df['numbers']
b1 = a1.shift(1)
# b1.sum()

R1 = (a1 - a1.shift(1)) / a1.shift(1)
r1 = np.log(a1 / a1.shift(1))
# ==========


np.sqrt(df)

'''
对df的数据使用f函数
'''


def f(x):
    value = (x - 3) ** 2 + 4
    return value


df.apply(f)

df.apply(np.sqrt)

###
df.apply(lambda x: x ** 2)  # square of every element

'''
Enlarging the DataFrame object in both dimensions is possible:
'''
df['floats'] = [1.5, 2.5, 3.5, 4.5]  # new column is generated
df

df['floats']  # selection of column

df['names'] = pd.DataFrame(['Yves', 'Guido', 'Felix', 'Francesc'],
                           index=['d', 'a', 'b', 'c'])
df

'''
通常必须避免的一个副作用-索引被简单的编号替代
'''
df_temp = df.append({'numbers': 100, 'floats': 5.75, 'names': 'Henry'},
                    ignore_index=True)
# temporary object; df not changed

'''
更好的做法：附加一个DateFrame对象以提供正确的索引信息
'''
df = df.append(pd.DataFrame({'numbers': 100, 'floats': 5.75,
                             'names': 'Henry'}, index=['z', ]))
df

# dealing with missing data
df_temp1 = df.join(pd.DataFrame([1, 4, 9, 16, 25],
                                index=['a', 'b', 'c', 'd', 'y'],
                                columns=['squares']))

# lose the value for the index y and have a NaN value
# to preserve both indices
df = df.join(pd.DataFrame([1, 4, 9, 16, 25],
                          index=['a', 'b', 'c', 'd', 'y'],
                          columns=['squares', ]), how='outer')

'''
#how='outer'表示使用两个索引中所有值的并集
'''
df

# df.loc['e']=[1,2,3,4]


# Although missing values, the majority of methods still work
df['numbers'].mean()
df[['numbers', 'squares']].mean()
# column-wise mean
df[['numbers', 'squares']].std()
# column-wise standard deviation
df2 = df['numbers'].fillna(35.35)

'''
fillna()处理缺失值NaN
'''
#########second step######
a = np.random.standard_normal((9, 4))

a.round(6)

df = pd.DataFrame(a)
df
df.columns = ['No1', 'No2', 'No3', 'No4']
df
df['No2'][3]  # value in column No2 at index position 3
####handle time  indices

dates = pd.date_range('12/31/2014', periods=9, freq='M')

df.index = dates

df

##transform the dataframe into array
np.array(df).round(6)

df.sum()
df.mean()
df.cumsum()

df.describe()

np.sqrt(df)
np.sqrt(df).sum()

df.cumsum().plot(lw=2.0)

'''
help(pd.to_timedelta)
help(pd.Series)
'''
##Series object
aa = df['No1']
type(df['No1'])

import matplotlib.pyplot as plt

df['No1'].cumsum().plot(style='r', lw=2.)
plt.xlabel('date')
plt.ylabel('value')

tsdat = pd.Series(np.random.random(len(dates)), index=dates)
tsdat.cumsum().plot(style='r', lw=2.)

# GroupBy操作
df['Quarter'] = ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2',
                 'Q3', 'Q3', 'Q3']
df

groups = df.groupby('Quarter')

groups.mean()
groups.max()
groups.size()
groups.sum()

df['Odd_Even'] = ['Odd', 'Even', 'Odd', 'Even', 'Odd',
                  'Even', 'Odd', 'Even', 'Odd']
groups = df.groupby(['Quarter', 'Odd_Even'])
groups.size()
groups.mean()

'''
#=========数据结构补充=================
'''
##字典 dict
x = {"a": 1, "b": 5, "c": 2}
type(x)
x['a']
x.get('b')  # 使用 get 方法可以实现高效的查询

x.items()
x.values()
x.keys()

x.pop('b')
x.update({'d': [2, 3]})

##集合set
y = {"a", "b", "c"}
type(y)

y.update('d')
y.remove('b')

y1 = [1, 2, 4, 5, 6, 6, 6]
y2 = set(y1)

'''
读取和存取数据
##########Getting Data##############
'''
# 1 Inputting data from the clipboard # pd.read_clipboard()
import pandas as pd

# copy first in your written notebook
data = pd.read_clipboard()

# 2 Retrieving historical price data from Yahoo!Finance
import pandas_datareader.data as web

ibm = web.DataReader(name='IBM', data_source='yahoo',
                     start='2013-1-1', end='2013-12-31')

ibm.to_csv("E:/ibm2013.txt")  # output data

# 3 Inputting data from a text file
p1 = pd.read_csv("E:/ibm2013.txt", index_col=0, parse_dates=True)
# p2=pd.read_table("E:/datasets/ibm2013.txt",index_col=0)


# 4 Inputting data from an Excel file
import xlrd

x1 = pd.read_excel("E:/dataset/test.xlsx", index_col=0, header=None)
x1.columns = [['a', 'b']]
x1.index.name = 'time'

infile = pd.ExcelFile("E:/dataset/datexcel.xlsx")
x2 = infile.parse('Sheet2')

'''
####========关于时间=========datetime module==========#######
'''
from datetime import datetime

datetime.now()
datetime.utcnow()
# 表示2019年10月20日10:20，10秒20微秒
someday = datetime(2019, 10, 20, 10, 20, 10, 20)
someday
someday.year
someday.month
someday.day
someday.date()

someday.isoformat()
someday.isoweekday()

someday.strftime("%Y-%m-%d %H:%M:%S")
someday.strftime("%Y/%m/%d %H:%M:%S")

now = datetime.now()
now.strftime("%Y-%m-%d")

now.strftime("%A, %d. %B %Y %I:%M%p")
print("今天是{0:%Y}年{0:%m}月{0:%d}日".format(now))

S = datetime.strptime('2017/09/30', '%Y/%m/%d')
type(S)
print(S)

S = datetime.strptime('2017年9月30日星期六', '%Y年%m月%d日星期六')
print(S)
S = datetime.strptime('2017年9月30日星期六8时42分24秒', '%Y年%m月%d日星期六%H时%M分%S秒')
print(S)
S = datetime.strptime('9/30/2017', '%m/%d/%Y')
print(S)
S = datetime.strptime('9/30/2017 8:42:50', '%m/%d/%Y %H:%M:%S')
print(S)

help(pd.to_datetime)

tt = datetime.strptime('16:00:00', '%H:%M:%S')
t0 = datetime.strptime('9:30:00', '%H:%M:%S')

Time = (tt - t0).total_seconds()
Time

'''
#==============statsmodels=================
'''
##普通最小二乘模型###
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

x = np.random.standard_normal((20, 4))

beta = np.array([[1], [2], [3], [4]])

e = np.random.standard_normal((20, 1))

y = 1 + np.dot(x, beta) + e

x = pd.DataFrame(x)

x.columns = ['b1', 'b2', 'b3', 'b4']

x = np.array(x)

x = sm.add_constant(x)

model = sm.OLS(y, x)

results = model.fit()

results.summary()

results.params

res = results.resid

sm.qqplot(res)
from scipy import stats

sm.qqplot(res, stats.t, distargs=(4,))

x = np.random.standard_normal((20, 4))
beta = np.array([[1], [2], [3], [4]])
e = np.random.standard_normal((20, 1))
y = 1 + np.dot(x, beta) + e
x = pd.DataFrame(x)

x.columns = ['x1', 'x2', 'x3', 'x4']
data = pd.DataFrame(x)

data['y'] = y

results = smf.ols('y~x1+x2+x3+x4', data=data).fit()

results.summary()
results.predict(data[:5])

# help(smf.ols)
