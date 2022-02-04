from typing import Optional, Tuple
import numpy as np
import datetime as dt
from scipy.stats import nbinom, beta
from scipy import special
from abc import ABC, abstractmethod


class Optim(ABC):
    @abstractmethod
    def invitation_logic_api(
        self, now: dt.datetime,
        deadline: dt.datetime,
        num_vacancies: int,
        num_remaining_in_pool: int,
        impacted_candidates_data: list
    )-> Tuple[bool, int, Optional[int]]:
        raise NotImplementedError


class NegativeBinomial():
    def __init__(self, prior_beta_mu: int = 1, prior_beta_var: int = 1, nbin_r: int = 5) -> None:

        self.mu = prior_beta_mu
        self.var = prior_beta_var

        self.alpha, self.beta = self.est_prior_beta_params(self.mu, self.var)
        self.alpha_posterior, self.beta_posterior = self.alpha, self.beta

        self.nbin_r = nbin_r
        self.n_samples = 0

    def rvs(self, size: int = 1, random_state: int = None) -> float:
        """Random variates of the beta posterior distribution.
        ---
        params:
            size: number of values
            random_state: scipy random state
        returns:
            beta posterior random values
        """
        return beta(self.alpha_posterior, self.beta_posterior).rvs(
            size=size, random_state=random_state
            )

    def rvs_nbin_mean(self, size: int = 50000) -> float:
        #https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/_discrete_distns.py#L204
        pp = beta.rvs(self.alpha_posterior, self.beta_posterior, size=1)
        return np.mean(nbinom.rvs(self.nbin_r, pp, size=size))

    @staticmethod
    def est_prior_beta_params(mu: float, var: float) -> Tuple[float, float]:
        '''Given mean and varianze, estimate Beta dist param.
        ---
        params:
            mu: mean beta dist.
            var: var beta dist.
        ---
        returns;
            alpha: alpha dist param.
            beta: beta dist param.
        '''
        alpha = ((1.0 - mu) / var - 1.0 / mu) * mu ** 2.0
        beta = alpha * (1 / mu - 1)
        return alpha, beta

    def rvs_mean(self, size: int = 50000) -> float:
        #https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/_discrete_distns.py#L204
        pp = beta.rvs(self.alpha_posterior, self.beta_posterior, size=1)
        return nbinom.rvs(self.nbin_r, pp, size=size)

    def update(self, data: int) -> None:
        """Update posterior parameters with new data samples.
        ---
        params:
            data : data samples from negative binomial dist.
        """
        x = np.asarray(data)
        n = x.size
        self.alpha_posterior += self.nbin_r * n
        self.beta_posterior += np.sum(x)
        self.n_samples += n

    def ppmean(self) -> float:
        """Posterior predictive mean.
        ---
        returns:
            analytically compute de mean posterior.
        """
        a = self.alpha_posterior
        b = self.beta_posterior

        if a > 1:
            return self.nbin_r * b / (a - 1)
        else:
            return np.nan

    def pppdf(self, x):
        """Posterior predictive probability density function.
        ---
        params:
            x : quantile.
        ---
        returns:
            pdf :prob density function evaluated at x.
        """
        a = self.alpha_posterior
        b = self.beta_posterior

        k = np.floor(x)
        pdf = np.zeros(k.shape)
        idx = (k >= 0)
        k = k[idx]

        loggxr = special.gammaln(self.nbin_r + k)
        loggr = special.gammaln(k + 1)
        loggx = special.gammaln(self.nbin_r)

        logcomb = loggxr - loggr - loggx
        logbeta = special.betaln(a + self.nbin_r, b + k) - special.betaln(a, b)
        logpdf = logcomb + logbeta

        pdf[idx] = np.exp(logpdf)

        return pdf
