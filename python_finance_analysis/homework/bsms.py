import numpy as np
from scipy import stats


def bsm_call(s0, k, t, r, sig):
    """
    欧式看涨期权定价
    s0: 股价
    k: 执行价格
    t: 到期时间
    r: 无风险利率
    sig: 年波动率
    """
    d1 = np.log(s0 / k) + (r + 0.5 * sig ** 2) * t ** 0.5 / sig
    d2 = np.log(s0 / k) + (r - 0.5 * sig ** 2) * t ** 0.5 / sig
    return s0 * stats.norm.cdf(d1, 0, 1) - k * np.exp(-r * t) * stats.norm.cdf(d2, 0, 1)


def bsm_put(s0, k, t, r, sig):
    """
    欧式看跌期权定价
    """
    d1 = np.log(s0 / k) + (r + 0.5 * sig ** 2) * t ** 0.5 / sig
    d2 = np.log(s0 / k) + (r - 0.5 * sig ** 2) * t ** 0.5 / sig
    return k * np.exp(-r * t) * stats.norm.cdf(-d2, 0, 1) - s0 * stats.norm.cdf(-d1, 0, 1)
