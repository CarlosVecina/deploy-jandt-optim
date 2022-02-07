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
SOLVER = 'glpk'


class OptimStochConstraint(Optim, DataImpactSerializer):
    def __init__(self, beta_mean: float, beta_var: float, solver: str = SOLVER, lose_confidence: float = 0) -> None:
        self.beta_mean = beta_mean
        self.beta_var = beta_var
        self.nbin_model = NegativeBinomial(
            prior_beta_mu=self.beta_mean,
            prior_beta_var=self.beta_var,
            nbin_r=1
        )
        self.l_case_frq = []
        self.l_per_frq = [16]
        self.dist = None
        self.solver = solver #glpk or ipopt
        self.lose_confidence = lose_confidence
        self.num_remaining_in_pool = None
        self.num_remaining_vacancies = None

    def __repr__(self):
        return 'Agent Stochastic Constraint'

    def stoch_optim(self) -> int:
        '''Stochastic optimizer approximation.
        returns:
            solution: number of estimated invitations that maximize the equation and constraints.
        '''
        model_stoch = pyo.ConcreteModel()

        model_stoch.x = pyo.Var([1, 2], domain=pyo.NonNegativeIntegers)

        model_stoch.Constraint1 = pyo.Constraint(expr=model_stoch.x[1] <= self.num_remaining_in_pool)
        model_stoch.Constraint2 = pyo.Constraint(expr=model_stoch.x[1] >= -1)
        model_stoch.Constraint3 = pyo.Constraint(expr=model_stoch.x[2] == model_stoch.x[1] * self.nbin_model.rvs()[0])
        model_stoch.Constraint4 = pyo.Constraint(expr=model_stoch.x[2] <= self.num_remaining_vacancies)  # TODO: refactor object and constraints
        model_stoch.Constraint5 = pyo.Constraint(expr=model_stoch.x[2] >= -1)

        model_stoch.OBJ = pyo.Objective(expr=PROFIT_VACANCY*model_stoch.x[2] - COST_SPAM*model_stoch.x[1], sense=pyo.maximize)

        solver = pyo.SolverFactory(self.solver)
        solution = solver.solve(model_stoch)

        model_stoch.solutions.store_to(solution)
        return solution['Solution'].variable['x[1]']['Value']

    def boostrap(self, q: float = 0.2) -> int:
        '''Boostraping the invitation stochastic maximization distribution.
        ---
        params:
            q: quantile to extract.
        returns:
            quantile dist value.
        '''
        _l = []
        for _ in range(10):
            _l.append(
                int(self.stoch_optim())
                )
        return max(int(q*10), np.quantile(_l, q))

    def forget_info(self, _l: list, perc: float) -> int:
        '''Force to forget % of pseudo-persisted agent memory.
        ---
        params:
            l: input list with pseudo-memory.
            perc: % of data censored.
        return:
            random sampled input list.
        '''
        return random.sample(_l, int(len(_l) * (1-perc)))

    def frequency_exploitation(self) -> int:
        '''Once some exploration is done, fit distribution parameters from which sample data.
        Censored data not informative (Cox Assuption).
        ---
        return:
            freq: freq sampled from fited dist once exploration is done
        '''
        if self.lose_confidence > 0:   # Exploration mutation
            # If lose_confidence is != 0, a % of the prior obsservations are forgotten, making the confidence lower and probabily(or not, refit)
            self.l_per_frq = self.forget_info(self.l_per_frq, self.lose_confidence)

        _temp_l = self.l_per_frq.copy()
        _temp_l.extend(self.l_case_frq)

        freq = int(np.median(_temp_l))
        if (len(_temp_l) % 10 == 0) & (len(_temp_l) > 28):
            dist = distfit()
            self.dist = dist
            self.dist.fit_transform(np.array(_temp_l))
        if self.dist is not None:   # TODO: evolve the exploration phase
            dict_pred = self.dist.predict(
                list(range(np.max(_temp_l)))
                )
            freq = np.max([1, np.max([i for i, x in enumerate(dict_pred['y_proba'] >= .3) if x])])

        return freq

    def persist_list_freq(self) -> None:
        self.l_per_frq.extend(self.l_case_frq)
        self.l_case_frq = []

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
        self.num_remaining_in_pool = num_remaining_in_pool
        post = super().get_n_first_accepted(impacted_candidates_data)
        if post > 0:
            self.nbin_model.update(post)

        finished = now >= deadline
        self.num_remaining_vacancies = (num_vacancies - super().get_total_contract_accepted(impacted_candidates_data))
        fulfilled = (num_vacancies - super().get_total_contract_accepted(impacted_candidates_data)) <= 0
        if finished | fulfilled | (num_remaining_in_pool <= 0):
            num_candidates_needed, callback_time_minutes = [0, 0]
            self.persist_list_freq()
            return True, num_candidates_needed, callback_time_minutes

        self.l_case_frq = super().get_t_response_accepted(impacted_candidates_data)

        callback_time_minutes = self.frequency_exploitation()

        self.last_call = now
        return finished, min(int(self.boostrap()), num_remaining_in_pool), max(5, callback_time_minutes)
