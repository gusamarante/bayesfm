# bayesfm - Bayesian Fama-MacBeth

Implementation of ["Bayesian Fama-MacBeth Regressions"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4989615) from Bryzgalova, Huang and Julliard (2024).
As presented by the authors, this methodology provides reliable risk premia estimates for both tradable and nontradable factors, 
detects those weakly identified, delivers valid credible intervals for all objects of interest, and is intuitive, fast and simple to implement.

## Installation
```bash
pip install bayesfm
```

## Usage
There is a self-contained [example file](https://github.com/gusamarante/bayesfm/blob/main/example.py) using the Fama-French 25 sorted portfolios and their 5 factors.

There are 3 classes available:
- `BFM`: Bayesian Fama-MacBeth
- `BFMGLS`: Bayesian Fama-MacBeth with the GLS precision matrix for the cross-sectional step
- `BFMOMIT`: Bayesian Fama-MacBeth with omitted factors
  - As noted by the authors, the use of this model requires us to include a sufficient number of latent factors in the cross-sectional step, which is chosen with the `p` argument of this class

All three class save the draws of all elements of interest as attributes, and have a method called `plot_lambda`, which plots the posteriors of the risk premia parameters.
This method outputs the chart below, where the blue density are the posterior draws and the orange lines are the canonical Fama-MacBeth two-pass OLS regression estiamtes.

<p align="center">
  <img src="https://github.com/gusamarante/bayesfm/blob/main/images/bfm_lambda_posterior.png?raw=true" alt="Risk Premia Posterior"/>
</p>

