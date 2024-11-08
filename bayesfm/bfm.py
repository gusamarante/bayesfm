"""
Implementation of the Bayesian Fama-MacBeth Regressions from Bryzgalova, Huang
and Julliard (2024).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numpy.linalg import inv, svd
from scipy.stats import invwishart, multivariate_normal


class FM:

    def __init__(self, assets, factors):
        """
        Very simple implementation of the Fama-MacBeth regression using OLS.

        Parameters
        ----------
        assets : pandas.DataFrame
            returns of the test assets

        factors : pandas.DataFrame
            returns of the factor portfolios
        """

        self.beta = pd.DataFrame(
            data=(inv(factors.T @ factors) @ factors.T @ assets).values,
            columns=assets.columns,
            index=factors.columns,
        ).T

        mu = assets.mean()
        self.lambdas = pd.Series(
            data=(inv(self.beta.T @ self.beta) @ self.beta.T @ mu).values,
            index=factors.columns,
            name="Lambdas",
        )


class BFM:
    """
    This class works as both the baseline implementation of the Bayesian
    Fama-MacBeth and the parent class for other variations, which follow the
    same steps, but with different computations.
    """

    def __init__(self, assets, factors, n_draws=1000):
        """
        Implementation of the BFM-OLS

        Parameters
        ----------
        assets : pandas.DataFrame
            returns of the test assets

        factors : pandas.DataFrame
            returns of the factor portfolios

        n_draws : int
            number of draws from the posterior
        """
        self.assets = assets
        self.factors = factors
        self.n_draws = n_draws
        self.y = pd.concat([assets, factors], axis=1).dropna()

        self.t = self.y.shape[0]  # Sample size of the timeseries dimension
        self.n = assets.shape[1]  # Number of test assets
        self.k = factors.shape[1]  # Number of factors

        self.mu_y = self.y.mean()
        self.Sigma_y = self.y.cov()

        self.draws_mu_y, self.draws_Sigma_y = self._draw_mu_sigma()
        self.draws_betas = self._compute_betas()

        self.draws_lambdas = self._compute_lambdas()
        self.draws_r2 = self._compute_r2()

        # Reorganize in Pandas
        self.draws_lambdas = pd.DataFrame(
            data=self.draws_lambdas,
            columns=factors.columns
        )
        self.draws_r2 = pd.Series(
            data=self.draws_r2,
            name="R2",
        )

    def _draw_mu_sigma(self):
        # Draws covariance from inverse wishart
        draws_sigma = invwishart.rvs(
            df=self.t - 1,
            scale=self.t * self.Sigma_y,
            size=self.n_draws,
        )

        # Draws mu from normal, conditional on previous covariances
        draws_mu = np.array(
            [
                multivariate_normal.rvs(
                    mean=self.mu_y,
                    cov=(1 / self.t) * cov,
                )
                for cov in draws_sigma
            ]
        )

        return draws_mu, draws_sigma

    def _compute_betas(self):
        draws_beta = np.array(
            [
                cov[:self.n, -self.k:] @ inv(cov[-self.k:, -self.k:])
                for cov in self.draws_Sigma_y
            ]
        )
        return draws_beta

    def _compute_lambdas(self):
        # campute lambdas from betas and mus
        draws_lambda = np.array(
            [
                inv(b.T @ b) @ b.T @ mu[:self.n]
                for mu, b in zip(self.draws_mu_y, self.draws_betas)
            ]
        )

        return draws_lambda

    def _compute_r2(self):
        mu_r = self.mu_y[:self.n].values
        denom = (mu_r - mu_r.mean()) @ (mu_r - mu_r.mean())
        draws_r2 = [
            1 - ((mu_r - b @ l) @ (mu_r - b @ l)) / denom
            for b, l in zip(self.draws_betas, self.draws_lambdas)
        ]
        return draws_r2

    def plot_lambda(self, include_fm=False):
        """
        Plots the posterior distributions for the risk premia

        Parameters
        ----------
        include_fm : bool
            If true, adds a vertical line indicating the values for the
            canonical OLS Fama-MacBeth regression.
        """
        axes = self.draws_lambdas.hist(
            density=True,
            bins=int(np.sqrt(self.n_draws)),
            sharex=True,
            figsize=(10, 6)
        )

        if include_fm:
            fm = FM(self.assets, self.factors)
            for ax in axes.flatten():
                try:
                    ax.axvline(
                        fm.lambdas[ax.title.get_text()],
                        color='tab:orange',
                        lw=2,
                    )
                except KeyError:
                    continue

        plt.tight_layout()
        plt.show()

    def plot_r2(self):
        """
        Plots the posteurior distribution for the R2
        """
        axes = self.draws_r2.hist(
            density=True,
            bins=int(np.sqrt(self.n_draws)),
            figsize=(10, 6)
        )
        plt.tight_layout()
        plt.show()

    def ci_table_lambda(self, cred=0.95):
        """
        Return a DataFrame with the median and the credible interval with the
        chosen level of credibility for the risk premia

        Parameters
        ----------
        cred : float
            number between 0 and 1 indicating the credibility level of the
            interval
        """
        table = self.draws_lambdas.quantile(
            q=[
                (1 - cred) / 2,
                0.5,
                (1 + cred) / 2,
            ],
        )
        table.index.name = "Quantiles"
        return table

    def ci_table_r2(self, cred=0.95):
        """
        Return a DataFrame with the median and the credible interval with the
        chosen level of credibility for the R2

        Parameters
        ----------
        cred : float
            number between 0 and 1 indicating the credibility level of the
            interval
        """
        table = self.draws_r2.quantile(
            q=[
                (1 - cred) / 2,
                0.5,
                (1 + cred) / 2,
            ],
        )
        table.index.name = "Quantiles"
        return table


class BFMGLS(BFM):
    """
    This class inherits all the methods and constructor from BFM but modifies
    the computation of the risk premia and R2 to use the GLS precision matrix
    """

    def _compute_lambdas(self):
        # compute idiosyncratic error covariance
        draws_sige = np.array(
            [
                cov[:self.n, :self.n] - cov[:self.n, -self.k:] @ inv(cov[-self.k:, -self.k:]) @ cov[:self.n, -self.k:].T
                for cov in self.draws_Sigma_y
            ]
        )

        # campute lambdas from betas and mus
        draws_lambda = np.array(
            [
                inv(b.T @ inv(sige) @ b) @ b.T @ inv(sige) @ mu[:self.n]
                for mu, b, sige in zip(self.draws_mu_y, self.draws_betas, draws_sige)
            ]
        )

        return draws_lambda

    def _compute_r2(self):
        draws_sige = np.array(
            [
                cov[:self.n, :self.n] - cov[:self.n, -self.k:] @ inv(cov[-self.k:, -self.k:]) @ cov[:self.n, -self.k:].T
                for cov in self.draws_Sigma_y
            ]
        )

        mu_r = self.mu_y[:self.n].values
        mu_r_bar = mu_r.mean()

        draws_r2 = [
            1 - ((mu_r - b @ l) @ inv(sige) @ (mu_r - b @ l)) / ((mu_r - mu_r_bar) @ inv(sige) @ (mu_r - mu_r_bar))
            for b, l, sige in zip(self.draws_betas, self.draws_lambdas, draws_sige)
        ]
        return draws_r2


class BFMOMIT(BFM):
    """
    This class inherits all the methods and constructor from BFM but modifies
    the computation of the risk premia and R2 to use the BFM-OMIT methodology
    """

    def __init__(self, assets, factors, n_draws=1000, p=5):
        """
        Implementation of the BFM-OMIT

        Parameters
        ----------
        assets : pandas.DataFrame
            returns of the test assets

        factors : pandas.DataFrame
            returns of the factor portfolios

        n_draws : int
            number of draws from the posterior

        p : int
            number of principal components to use
        """
        self.p = p
        super().__init__(assets, factors, n_draws)

    def _compute_lambdas(self):

        def cov_svd(cov):
            u, s, _ = svd(cov)
            return (u @ np.diag(s))[:, :self.p]

        draws_beta_upsilon = np.array(
            [
                cov_svd(cov[:self.n, :self.n])
                for cov in self.draws_Sigma_y
            ]
        )

        draws_lambda_upsilon = np.array(
            [
                inv(b.T @ b) @ b.T @ mu[:self.n]
                for b, mu in zip(draws_beta_upsilon, self.draws_mu_y)
            ]
        )

        draws_lambda_f = np.array(
            [
                lu.T @ inv(bu.T @ bu) @ bu.T @ cov[:self.n, -self.k:]
                for lu, bu, cov in zip(draws_lambda_upsilon, draws_beta_upsilon, self.draws_Sigma_y)
            ]
        )

        return draws_lambda_f

    def _compute_r2(self):

        def cov_svd(cov):
            u, s, _ = svd(cov)
            return (u @ np.diag(s))[:, :self.p]

        draws_beta_upsilon = np.array(
            [
                cov_svd(cov[:self.n, :self.n])
                for cov in self.draws_Sigma_y
            ]
        )

        draws_lambda_upsilon = np.array(
            [
                inv(b.T @ b) @ b.T @ mu[:self.n]
                for b, mu in zip(draws_beta_upsilon, self.draws_mu_y)
            ]
        )

        mu_r = self.mu_y[:self.n].values
        mu_r_bar = mu_r.mean()
        draws_r2 = np.array(
            [
                1 - ((mu_r - bu @ lu) @ (mu_r - bu @ lu)) / ((mu_r - mu_r_bar) @ (mu_r - mu_r_bar))
                for lu, bu in zip(draws_lambda_upsilon, draws_beta_upsilon)
            ]
        )
        return draws_r2
