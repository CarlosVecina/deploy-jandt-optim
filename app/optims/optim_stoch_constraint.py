from scipy.stats import nbinom, beta
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import pyomo.environ as pyo
import datetime as dt
from distfit import distfit
import random

from .homework import Optim, NegativeBinomial
from .utils import DataImpactSerializer

PROFIT_VACANCY = 2000
COST_SPAM = 20

class OptimStochConstraint(Optim, DataImpactSerializer):
    def __init__(self, beta_mean, beta_var) -> None:
        self.beta_mean=beta_mean
        self.beta_var=beta_var
        self.nbin_model = NegativeBinomial(
            prior_beta_mu=self.beta_mean,
            prior_beta_var=self.beta_var,
            nbin_r=1
        )
        self.l_case_frq = []
        self.l_per_frq = [6]

    def __repr__(self):
        return 'Agent Stochastic Constraint'

    def stoch_optim(self) -> int:
        model_stoch = pyo.ConcreteModel()

        model_stoch.x = pyo.Var([1,2], domain=pyo.NonNegativeIntegers)

        model_stoch.Constraint1 = pyo.Constraint(expr=model_stoch.x[1]<=100)
        model_stoch.Constraint2 = pyo.Constraint(expr=model_stoch.x[1] >=-1)
        model_stoch.Constraint3 = pyo.Constraint(expr=model_stoch.x[2] == model_stoch.x[1] * round(self.nbin_model.rvs()[0],1))
        model_stoch.Constraint4 = pyo.Constraint(expr=model_stoch.x[2] <=10)
        model_stoch.Constraint5 = pyo.Constraint(expr=model_stoch.x[2] >=-1)

        model_stoch.OBJ = pyo.Objective(expr=PROFIT_VACANCY * model_stoch.x[2] - COST_SPAM * model_stoch.x[1], sense=pyo.maximize)

        solver = pyo.SolverFactory("glpk")
        solution = solver.solve(model_stoch)

        model_stoch.solutions.store_to(solution)
        return solution['Solution'].variable['x[1]']['Value']

    def boostrap(self, q: float = 0.2) -> int:
        _l = []
        for _ in range(10):
            _l.append(
                int(self.stoch_optim())
                )
        return min(1,np.quantile(_l, q))

    def forget_info(self, l: list, perc: float) -> int:
        return random.sample(l,int(len(l)*(1-perc)))

    def frequency_exploitation(self, lose_confidence: float = 0) -> int:
        ## try to fit a distribution uner a X certainity. If OK, then persist
        ## censored data not informative (Cox Assuption)

        if lose_confidence > 0:   # exploration mutation
            ## If lose_confidence is != 0, a % of the prior obsservations are forgotten, making the confidence lower and probabily(or not, refit)
            self.l_per_frq = self.forget_info(self.l_per_frq, lose_confidence)

        _temp_l = self.l_per_frq.copy()
        _temp_l.extend(self.l_case_frq)

        if len(_temp_l)>=28:
            dist = distfit()
            s = dist.fit_transform(np.array(_temp_l))
            dict_pred = dist.predict(list(range(np.max(_temp_l))))
            return np.max([1, np.max([i for i, x in enumerate(dict_pred['y_proba'] >= .3) if x])])
        else:
            return int(np.median(_temp_l))

    def persist_list_freq(self) -> None:
        self.l_per_frq.aextendppend(self.l_case_frq)
        self.l_case_frq = []

    def invitation_logic_api(
        self, now, deadline, num_vacancies, num_remaining_in_pool, impacted_candidates_data
    ) -> Tuple[bool, int, Optional[int]]:
        finished = now >= deadline
        fulfilled = (num_vacancies - super().get_total_contract_accepted(impacted_candidates_data)) <=0
        if finished | fulfilled | (num_remaining_in_pool<=0):
            num_candidates_needed, callback_time_minutes = [0, 0]
            self.persist_list_freq()
            return finished, num_candidates_needed, callback_time_minutes

        post = super().get_n_first_accepted(impacted_candidates_data)
        if post>0:
            self.nbin_model.update(post)

        self.l_case_frq = super().get_t_response_accepted(impacted_candidates_data)

        callback_time_minutes = self.frequency_exploitation()

        self.last_call = now
        return finished, min(int(self.boostrap()), num_remaining_in_pool), callback_time_minutes


#print(1)

#my_c = OptimStochConstraint(0.2, 0.001)
#my_c.invitation_logic_api(
#    dt.datetime(2021,1,1),
#    dt.datetime(2021,3,12),
#    10,
#    95,
#    [None]
#)