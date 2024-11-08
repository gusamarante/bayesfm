from bayesfm import *

# ===== Read Data =====
# Excess return of Fama-French portfolios
ports = get_ff25p()
rf = get_ffrf()
ports = ports.sub(rf, axis=0)
ports = ports.dropna()
ports.columns = [f"FF{(s - 1) * 5 + v}" for s, v in ports.columns]

# Fama-French 5 factors
facts = get_ff5f()


# ===== Bayesian Fama-MacBeth =====
model = BFM(
    assets=ports,
    factors=facts,
    n_draws=10_000,
)

print(model.ci_table_lambda(cred=0.90))
model.plot_lambda(include_fm=True)
