import pandas_datareader
import pandas as pd
from datetime import datetime
import statsmodels.api as sm

start = datetime(2007, 1, 1)
end = datetime(2018, 12, 31)
IBM = pandas_datareader.data.get_data_yahoo('IBM', start, end)
rtn_ibm = IBM['Close'].values[1:] / IBM['Close'].values[:-1] - 1
rtn_ibm = pd.DataFrame(data=rtn_ibm, columns=['rtn'])
rtn_ibm.index = IBM.index[1:]
SP500 = pandas_datareader.data.get_data_yahoo('^GSPC', start, end)
rtn_sp = SP500['Close'].values[1:] / SP500['Close'].values[:-1] - 1
rtn_sp = pd.DataFrame(data=rtn_sp, columns=['rtn'])
rtn_sp.index = SP500.index[1:]

years = list(range(2007, 2019))
r_f = 0.005
n = len(years)
a = [0]*n
b = [0]*n
for i in range(n):
    rtn_ibm_current = rtn_ibm[rtn_ibm.index.year == years[i]]
    rtn_sp_current = rtn_sp[rtn_sp.index.year == years[i]]
    x = sm.add_constant(rtn_sp_current.values - r_f)
    y = rtn_ibm_current.values - r_f
    model = sm.OLS(y, x)
    a[i], b[i] = model.fit().params
print(b)
