#==================Chapter 10==================#
'''
在B-S-M模型期权定价中，给定一组参数S(当前股价)、X(执行价)、T(有效期)、
r(年连续复利无风险利率)和σ(股票收益率的标准差，也称波动率)，可以计算欧式看涨期权的
定价c。
例如，当S=46,X=42,T=0.5,r=0.015, sigma=0.2时，求c
反过来，如果知道S=46,X=42,T=0.5,r=0.015，期权价格c=5.20,
求解得出的σ值就是隐含波动率，求隐含波动率。
方法1：试错法+收敛条件：
方法2：标准法:
'''


'''
试错法计算隐含波动率
'''
import numpy as np
#Implied volatility function based on a European call
def implied_vol_call(S,X,T,r,c):
     from scipy import log,exp,sqrt,stats
     for i in range(200):
          sigma=0.005*(i+1)
          d1=(log(S/X)+(r+sigma*sigma/2.)*T)/(sigma*sqrt(T))
          d2 = d1-sigma*sqrt(T)
          c_sigma = S*stats.norm.cdf(d1)-X*exp(-r*T)*stats.norm.cdf(d2)
          diff=c-c_sigma
          if abs(diff)<=0.01:
               return i,sigma, diff
           
S=46;X=42;T=0.5;r=0.015;c=5.20
implied_vol_call(S,X,T,r,c)

'''
试错法计算隐含波动率:法一
'''
def implied_vol_call_min(S,X,T,r,c):
     from scipy import log,exp,sqrt,stats
     implied_vol=1.0
     min_value=100.0
     for i in range(1,10000):
          sigma=0.0001*(i+1)
          d1=(log(S/X)+(r+sigma*sigma/2.)*T)/(sigma*sqrt(T))
          d2 = d1-sigma*sqrt(T)
          call=S*stats.norm.cdf(d1)-X*exp(-r*T)*stats.norm.cdf(d2)
          abs_diff=abs(call-c)
          if abs_diff<min_value:
               min_value=abs_diff
               implied_vol=sigma
               k=i
               call_out=c
     #print ('k,implied_vol, call_out,abs_diff')
     return k,implied_vol, call_out,min_value

temp=implied_vol_call_min(S,X,T,r,c)
temp[1]

#####Example 3-1. Black-Scholes-Merton (1973) functions####
#
# Valuation of European call options in Black-Scholes-Merton model
# incl. Vega function and implied volatility estimation
# bsm_functions.py
#
# Analytical Black-Scholes-Merton (BSM) Formula

'''
标准法计算隐含波动率：法二
'''

def bsm_call_value(S0, K, T, r, sigma):
    from scipy import stats,log, sqrt, exp
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    value = (S0 * stats.norm.cdf(d1, 0.0, 1.0)
              - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    return value

# Vega function
def bsm_vega(S0, K, T, r, sigma):
    from scipy import stats,log, sqrt
    d1 = (log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
    vega = S0*stats.norm.cdf(d1,0.0,1.0)*sqrt(T)
    return vega




# Implied volatility function
def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it=100):
    for i in range(it):
        sigma_est -= ((bsm_call_value(S0, K, T, r, sigma_est) - C0)
                         / bsm_vega(S0, K, T, r, sigma_est))
    return sigma_est

bsm_call_imp_vol(S, X, T, r, c, sigma_est=0.5, it=100)




'''
cc = np.array([4.5,5.2,5.5])
sigma_imp0=[]
for c in list(cc):
    temp=implied_vol_call(S,X,T,r,c)
    sigma_imp0.append(temp[1])
sigma_imp0
'''



cc= np.array([4.5,5.3,5.7,5.8,6.0])

#cc= np.array([4.5,5.2,5.5,5.8,6.0,6.3,6.5])

sigma_imp1=[]
for c in list(cc):
    temp=implied_vol_call_min(S,X,T,r,c)
    sigma_imp1.append(temp[1])
np.round(sigma_imp1,2)


sigma_imp2=bsm_call_imp_vol(S, X, T, r, cc, sigma_est=0.5, it=100)
np.round(sigma_imp2,2)






















          


