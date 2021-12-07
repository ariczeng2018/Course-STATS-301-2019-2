import numpy as np
from scipy.stats import norm


def _bsm_value(s, x, t, r, sigma):
    d1 = (np.log(s / x) + (r + sigma ** 2 / 2) * t) / (sigma * t ** 0.5)
    d2 = d1 - sigma * t ** .5
    return d1, d2


def bsm_call_value(s, x, t, r, sigma):
    d1, d2 = _bsm_value(s, x, t, r, sigma)
    return s * norm.cdf(d1) - x * np.exp(-r * t) * norm.cdf(d2)


def bsm_put_value(s, x, t, r, sigma):
    d1, d2 = _bsm_value(s, x, t, r, sigma)
    return x * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)


def implied_volatility(s, x, t, r, c, n=50):
    m = len(c)
    n = int(n)
    sigma = np.ones((m, n)) * .2
    for i in range(n - 1):
        for k in range(m):
            d1, _ = _bsm_value(s, x, t, r, sigma[k, i])
            vega = s * norm.cdf(d1) * t ** .5
            sigma[k, i + 1] = sigma[k, i] - (bsm_call_value(s, x, t, r, sigma[k, i]) - c[k]) / vega
    return sigma[:, -1]


if __name__ == '__main__':
    c1 = bsm_call_value(s=46, x=42, t=.5, r=.015, sigma=.2)
    print('看涨期权定价:', c1)
    c_ = np.array([4.5, 5.2, 5.5, 5.8, 6, 6.3, 6.5])
    volatility_standard = implied_volatility(s=46, x=42, t=.5, r=.015, c=c_)
    print(volatility_standard)
