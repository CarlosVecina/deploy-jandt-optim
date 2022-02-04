from scipy.stats import nbinom, beta
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import datetime as dt

from .homework import Optim, NegativeBinomial
from .utils import DataImpactSerializer

PRIOR_BETA_MU = 0.04
PRIOR_BETA_VAR = 0.00014


class OptimNegBinom(Optim, DataImpactSerializer):
    def __init__(self):
        self.nbin = NegativeBinomial(
            prior_beta_mu=PRIOR_BETA_MU,
            prior_beta_var=PRIOR_BETA_VAR,
            nbin_r=1)

    def __repr__(self):
        return 'Agent Negative Binomial'

    def invitation_logic_api(
        self, now: dt.datetime,
        deadline: dt.datetime,
        num_vacancies: int,
        num_remaining_in_pool: int,
        impacted_candidates_data: list
    ) -> Tuple[bool, int, Optional[int]]:
        '''Main funct invitation logic API
        ---
        params:
            now: current datetime
            deadline: job request deadline
            num_vacancies:  total num of job request vacancies
            num_remaining_in_pool: num of workers still in the pool
            impacted_candidates_data: list of impact data
        ---
        returns:
            finished: bool if the process is end
            num_candidates_needed: number of candidates to send notification
            callback_time_minutes: minutes untill the next API call
        '''
        post = super().get_n_first_accepted(impacted_candidates_data)
        if post > 0:
            self.nbin.update(post)

        finished = now >= deadline
        fulfilled = (num_vacancies - super().get_total_contract_accepted(impacted_candidates_data)) <= 0

        if finished | fulfilled | (num_remaining_in_pool <= 0):
            num_candidates_needed, callback_time_minutes = [0, 0]
            return True, num_candidates_needed, callback_time_minutes

        num_candidates_needed = round(self.nbin.ppmean())
        callback_time_minutes, _ = super().get_avg_t_response_accepted(impacted_candidates_data, 7)

        return finished, min(num_candidates_needed, num_remaining_in_pool), round(callback_time_minutes)
