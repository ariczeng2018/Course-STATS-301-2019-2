# Example:Black-Scholes-Merton (1973) functions
# Valuation of European call options in Black-Scholes-Merton model
# Analytical Black-Scholes-Merton (BSM) Formula

# t=0
def bsm_call_value(S0, K, T, r, sigma):
    """
Valuation of European call option in BSM model.
Analytical formula.
Parameters
==========
S0 : float
initial stock/index level
K : float
strike price
T : float
maturity date (in year fractions)
r : float
constant risk-free short rate
sigma : float
volatility factor in diffusion term
Returns
=======
value : float
present value of the European call option
    """
    from math import log, sqrt, exp
    from scipy import stats
    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    value = (S0 * stats.norm.cdf(d1, 0.0, 1.0)
              - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
     # stats.norm.cdf --> cumulative distribution function
     # for normal distribution
    return value





# Example:Black-Scholes-Merton (1973) functions
# Valuation of European put options in Black-Scholes-Merton model
# Analytical Black-Scholes-Merton (BSM) Formula
# t=0
def bsm_put_value(S0, K, T, r, sigma):
    """
Valuation of European put option in BSM model.
Analytical formula.
Parameters
==========
S0 : float
initial stock/index level
K : float
strike price
T : float
maturity date (in year fractions)
r : float
constant risk-free short rate
sigma : float
volatility factor in diffusion term
Returns
=======
value : float
present value of the European put option
    """
    from math import log, sqrt, exp
    from scipy import stats
    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    value = (K * exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0)-
             S0 * stats.norm.cdf(-d1, 0.0, 1.0))
    return value