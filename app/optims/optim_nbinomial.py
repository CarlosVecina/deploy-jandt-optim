from scipy.stats import nbinom, beta
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .homework import Optim, NegativeBinomial
from .utils import DataImpactSerializer

PRIOR_BETA_MU = 0.04
PRIOR_BETA_VAR = 0.00014


class OptimNegBinom(Optim, DataImpactSerializer):
    def __init__(self) -> None:
        self.nbin = NegativeBinomial(
            prior_beta_mu=PRIOR_BETA_MU,
            prior_beta_var=PRIOR_BETA_VAR,
            nbin_r=1)

    def __repr__(self):
        return f'Agent Negative Binomial'

    def invitation_logic_api(
        self, now, deadline, num_vacancies, num_remaining_in_pool, impacted_candidates_data
    )-> Tuple[bool, int, Optional[int]]:
        post = super().get_n_first_accepted(impacted_candidates_data)
        if post>0:
            self.nbin.update(post)

        finished = now >= deadline
        fulfilled = (num_vacancies - super().get_total_contract_accepted(impacted_candidates_data)) <=0

        if finished | fulfilled | (num_remaining_in_pool<=0):
            num_candidates_needed, callback_time_minutes = [0,0]
            return True, num_candidates_needed, callback_time_minutes


        num_candidates_needed = round(self.nbin.ppmean())
        callback_time_minutes, _ = super().get_avg_t_response_accepted(impacted_candidates_data, 7)

        return finished, min(num_candidates_needed, num_remaining_in_pool), round(callback_time_minutes)


