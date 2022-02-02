from typing import  Optional, Tuple
import numpy as np
from scipy.stats import nbinom, beta

from abc import ABC, abstractmethod


class Optim(ABC):
    @abstractmethod
    def invitation_logic_api(
        self, now, deadline, num_vacancies, num_remaining_in_pool, impacted_candidates_data
    )-> Tuple[bool, int, Optional[int]]:
        raise NotImplementedError


class NegativeBinomial():
    def __init__(self, prior_beta_mu=1, prior_beta_var=1, nbin_r=5) -> None:

        self.mu = prior_beta_mu
        self.var = prior_beta_var

        self.alpha, self.beta = self.est_prior_beta_params(self.mu, self.var)
        self.alpha_posterior, self.beta_posterior = self.alpha, self.beta

        self.nbin_r = nbin_r
        self.n_samples = 0

    def rvs(self, size: int = 1, random_state : int = None) -> float:
        """
        Random variates of the posterior distribution.
        Parameters
        ----------
        size : int (default=1)
            Number of random variates.
        random_state : int or None (default=None)
            The seed used by the random number generator.
        Returns
        -------
        rvs : numpy.ndarray or scalar
            Random variates of given size.
        """
        return beta(self.alpha_posterior, self.beta_posterior).rvs(
            size=size, random_state=random_state)

    def rvs_nbin_mean(self) -> float:
        #https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/_discrete_distns.py#L204
        pp = beta.rvs(self.alpha_posterior, self.beta_posterior, size=1)
        return np.mean(nbinom.rvs(self.nbin_r, pp, size=50000))

    @staticmethod
    def est_prior_beta_params(mu: float, var: float) -> Tuple[float, float]:
        alpha = ((1.0 - mu) / var - 1.0 / mu) * mu ** 2.0
        beta = alpha * (1 / mu - 1)
        return alpha, beta

    def rvs_mean(self) -> int:
        #https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/_discrete_distns.py#L204
        pp = beta.rvs(self.alpha_posterior, self.beta_posterior, size=1)
        return nbinom.rvs(self.nbin_r, pp, size=50000)

    def update(self, data: int) -> None:
        """
        Update posterior parameters with new data samples.
        Parameters
        ----------
        data : array-like, shape = (n_samples)
            Data samples from a negative binomial distribution.
        """
        x = np.asarray(data)
        n = x.size
        self.alpha_posterior += self.nbin_r * n
        self.beta_posterior += np.sum(x)
        self.n_samples += n

    def ppmean(self) -> float:
        r"""
        Posterior predictive mean.
        If :math:`X` follows a negative binomial distribution with parameters
        :math:`r` and :math:`p`, then the posterior predictive expected value
        is given by
        .. math::
            \mathrm{E}[X] = r \frac{\beta}{\alpha - 1},
        where :math:`\alpha` and :math:`\beta` are the posterior values
        of the parameters.
        Returns
        -------
        mean : float
        """
        a = self.alpha_posterior
        b = self.beta_posterior

        if a > 1:
            return self.nbin_r * b / (a - 1)
        else:
            return np.nan