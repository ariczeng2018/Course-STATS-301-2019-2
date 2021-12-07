######Chapter 9##########
'''
欧式看涨期权的收益payoff
'''
def payoff_call(sT,x):
     return (sT-x+abs(sT-x))/2

def payoff_call0(sT,x):
     return max(sT-x,0)
 
import numpy as np
x=20
sT=np.arange(10,50,10)
payoff_call(sT,x)

'''
欧式看涨期权：执行价给定，收益与股价关系绘图
'''
import numpy as np
import matplotlib.pyplot as plt
s = np.arange(10,80,5)
x=30
payoff=(abs(s-x)+s-x)/2
plt.ylim(-10,50)
plt.plot(s,payoff)
plt.xlabel("$S_T$")
plt.ylabel("payoff")

'''
欧式看涨期权：期权费、执行价给定，损益与股价关系
'''
import numpy as np
import matplotlib.pyplot as plt
s = np.arange(30,70,5)
x=45;call=2.5
profit=(abs(s-x)+s-x)/2 -call
y2=np.zeros(len(s))
plt.ylim(-30,50)
plt.plot(s,profit)
plt.plot(s,y2,'-.')
plt.plot(s,-profit)
plt.title("Profit/Loss function")
plt.xlabel('Stock price')
plt.ylabel('Profit (loss)')
plt.annotate('Call option buyer', xy=(55,15), xytext=(35,20),
             arrowprops=dict(facecolor='blue',shrink=0.01),)
plt.annotate('Call option seller', xy=(55,-10), xytext=(40,-20),
             arrowprops=dict(facecolor='red',shrink=0.01),)

'''
欧式看跌期权的损益函数
'''
import numpy as np
import matplotlib.pyplot as plt
s = np.arange(30,70,5)
x=45;p=2
y=p-(abs(x-s)+x-s)/2
y2=np.zeros(len(s))
x3=[x, x]
y3=[-30,10]
plt.ylim(-30,50)
plt.plot(s,y)
plt.plot(s,y2,'-.')
plt.plot(s,-y)
plt.plot(x3,y3)
plt.title("Profit/Loss function for a put option")
plt.xlabel('Stock price')
plt.ylabel('Profit (loss)')
plt.annotate('Put option buyer', xy=(35,12), xytext=(35,45),
         arrowprops=dict(facecolor='red',shrink=0.01),)
plt.annotate('Put option seller', xy=(35,-10), xytext=(35,-25),
         arrowprops=dict(facecolor='blue',shrink=0.01),)
plt.annotate('Exercise price', xy=(45,-30), xytext=(50,-20),
         arrowprops=dict(facecolor='black',shrink=0.01),)




'''
引入欧式看涨期权和看跌期权公式：可以向量化
'''
def bsm_call_value(S,X,T,r,sigma):
    from scipy import log,exp,sqrt,stats
    d1=(log(S/X)+(r+sigma*sigma/2.)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    return S*stats.norm.cdf(d1)-X*exp(-r*T)*stats.norm.cdf(d2)


def bsm_put_value(S,X,T,r,sigma):
    from scipy import log,exp,sqrt,stats
    d1=(log(S/X)+(r+sigma*sigma/2.)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    return X*exp(-r*T)*stats.norm.cdf(-d2)-S*stats.norm.cdf(-d1)

c=bsm_call_value(46,42,0.5,0.015,0.2)
c
'''
分红股票的期权价格
期权到期日是T,股票在T1年发放红利d
'''
from  math import exp
s0=40
d=1.5
r=0.015
T1=6/12
T=1
s=s0-exp(-r*T1)*d
x=42
sigma=0.2
round(bsm_call_value(s,x,T,r,sigma),2)

###Relationship between input values and option values
'''
期权的价格和波动率sigma和到期日T的关系
'''
import numpy as np
import matplotlib.pyplot as plt
s0=30;T0=0.5;sigma0=0.2;r0=0.05;x0=30
sigma=np.arange(0.05,0.8,0.05)
T=np.arange(0.5,2.0,0.5)
call_0=bsm_call_value(s0,x0,T0,r0,sigma0)
call_sigma=bsm_call_value(s0,x0,T0,r0,sigma)
call_T=bsm_call_value(s0,x0,T,r0,sigma0)

#plt.subplot(2,1,1)
plt.plot(sigma,call_sigma,'b')
plt.title("Relationship between volatility and option price")
plt.xlabel("$\sigma$")
plt.ylabel("Call Price")
#plt.subplot(2,1,2)
plt.plot(T,call_T,'g')
plt.title("Relationship between the maturity date T and option price")
plt.xlabel("T")
plt.ylabel("Call Price")












     






