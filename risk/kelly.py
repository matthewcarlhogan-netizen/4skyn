def get_kelly(trade_pnls, current_equity):
    """Bayesian prior Kelly - starts trading immediately (Phase 1)"""
    if len(trade_pnls) < 30:
        # Conservative prior: μ=0.002, σ=0.01 → implied Sharpe ~0.5
        mu_prior = 0.002
        sigma_prior = 0.01
        f = mu_prior / (sigma_prior ** 2)
        c = 0.10  # start at 10% fractional Kelly
        return f, c, 0.0
    # TODO: blend to empirical after 30 trades
    return 0.0, 0.0, 0.0
