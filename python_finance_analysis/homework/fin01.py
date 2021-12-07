from math import *


def pv_f(fv, r, n):  # 计算未来年金的现值
    return fv / (1 + r) ** n


def pv_perpetuity(c, r):  # 计算永久年金的现值
    return c / r


def pv_perpetuity_due(c, r):  # 计算前置型永久年金的现值
    return pv_perpetuity(c, r) * (1 + r)


def pv_growing_perpetuity(c, r, g):  # 计算增长型永久年金的现值
    assert r > g, '错误: 利率小于等于增长率'
    return c / (r - g)


def pv_annuity(c, r, n):  # 计算定期年金的现值
    return c / r * (1 - (1 + r) ** (-n))


def pv_annuity_due(c, r, n):  # 计算前置型定期年金的现值
    return pv_annuity(c, r, n) * (1 + r)


def pv_growing(c, r, n, g):  # 计算增长型年金的现值
    assert r > g, '错误: 利率小于等于增长率'
    return c / (r - g) * (1 - ((1 + g) / (1 + r)) ** n)


def fv_annuity(pmt, r, n):  # 计算定期年金的未来值
    return pmt / r * ((1 + r) ** n - 1)


def fv_annuity_due(c, r, n):  # 计算前置型定期年金的未来值
    return fv_annuity(c, r, n) * (1 + r)


def fv_growing(pmt, r, n, g):  # 计算增长型年金的未来值
    assert r > g, '错误: 利率小于等于增长率'
    return pmt / (r - g) * ((1 + r) ** n - (1 + g) ** n)


def pv_bond(c, r, fv, n):  # 计算债券现值
    return c / r * (1 - (1 + r) ** (-n)) + fv / (1 + r) ** n


def ear_f(apr, m):  # 计算有效年利率
    return (1 + apr / m) ** m - 1


def r_c(apr, m):  # 将年利率转化为连续复利率Rc
    return m * log(1 + apr / m)


def apr_f(rc, m):  # 将年利率转化为连续复利率Rc
    return m * (exp(rc / m) - 1)
