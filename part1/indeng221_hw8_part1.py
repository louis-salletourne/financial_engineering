import numpy as np
from scipy.stats import norm

# Define the parameters
S0 = 100.0  # initial stock price
K = 100.0  # strike price
T = 1.0  # time to maturity
r = 0.06  # risk-free rate
q = 0.06  # dividend yield
sigma = 0.35  # volatility
n_times_steps = 100  # number of time steps
n_simulated_stock_paths = 4000  # number of simulated stock paths
option_type = "call"  # "call" or "put"
np.random.seed(42)  # make result reproducible


def option_payoff(S, K, option_type):
    if option_type == "call":
        return max(S - K, 0)
    elif option_type == "put":
        return max(K - S, 0)


def monte_carlo_simulation():
    dt = T / n_times_steps
    sqrt_dt = np.sqrt(dt)
    payoff = np.zeros((n_simulated_stock_paths), dtype=float)
    step = range(0, int(n_times_steps), 1)
    for i in range(0, n_simulated_stock_paths):
        ST = S0
        for j in step:
            epsilon = np.random.normal()
            ST *= np.exp((r - q - 0.5 * sigma * sigma) * dt + sigma * epsilon * sqrt_dt)
        payoff[i] = option_payoff(ST, K, option_type)

    option_price = np.mean(payoff) * np.exp(-r * T)
    std_dev = np.std(payoff) / np.sqrt(n_simulated_stock_paths) * np.exp(-r * T)
    return option_price, std_dev


def antithetic_method():
    dt = T / n_times_steps
    sqrt_dt = np.sqrt(dt)
    payoff = np.zeros((n_simulated_stock_paths), dtype=float)
    step = range(0, int(n_times_steps), 1)
    for i in range(0, n_simulated_stock_paths):
        ST_1 = S0
        ST_2 = S0
        for _ in step:
            epsilon = np.random.normal()
            ST_1 *= np.exp(
                (r - q - 0.5 * sigma * sigma) * dt + sigma * epsilon * sqrt_dt
            )
            ST_2 *= np.exp(
                (r - q - 0.5 * sigma * sigma) * dt - sigma * epsilon * sqrt_dt
            )
        payoff[i] = (
            option_payoff(ST_1, K, option_type) + option_payoff(ST_2, K, option_type)
        ) / 2

    option_price = np.mean(payoff) * np.exp(-r * T)
    std_dev = np.std(payoff) / np.sqrt(n_simulated_stock_paths) * np.exp(-r * T)
    return option_price, std_dev


def control_variate_method():
    expected_ST = S0 * np.exp((r - q) * T)
    dt = T / n_times_steps
    sqrt_dt = np.sqrt(dt)
    payoffs = []
    ST_values = []

    for _ in range(n_simulated_stock_paths):
        ST = S0  # terminal stock price for this path
        for _ in range(n_times_steps):
            epsilon = np.random.normal()
            ST *= np.exp((r - q - 0.5 * sigma**2) * dt + sigma * epsilon * sqrt_dt)

        payoffs.append(option_payoff(ST, K, option_type))
        ST_values.append(ST)

    payoffs = np.array(payoffs)
    ST_values = np.array(ST_values)

    # Estimate beta
    cov_XY = np.cov(payoffs, ST_values)[0, 1]
    var_Y = np.var(ST_values)
    beta = cov_XY / var_Y

    # Control variate method with beta adjustment
    adjusted_payoffs = payoffs + beta * (expected_ST - ST_values)

    # Discounted price with control variate adjustment
    option_price = np.mean(adjusted_payoffs) * np.exp(-r * T)
    std_dev = (
        np.std(adjusted_payoffs) / np.sqrt(n_simulated_stock_paths) * np.exp(-r * T)
    )
    return option_price, std_dev


def bsm_call_price():
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


# Gather results
mc_price, mc_std = monte_carlo_simulation()
antithetic_price, antithetic_std = antithetic_method()
control_variate_price, control_variate_std = control_variate_method()
bsm_price = bsm_call_price()

results = np.array(
    [
        ["Monte Carlo Simulation", round(mc_price, 4), round(mc_std, 4)],
        ["Antithetic Method", round(antithetic_price, 4), round(antithetic_std, 4)],
        [
            "Control Variate Method",
            round(control_variate_price, 4),
            round(control_variate_std, 4),
        ],
        ["BSM Call Option Price", round(bsm_price, 4), "N/A"],
    ]
)

print(results)
