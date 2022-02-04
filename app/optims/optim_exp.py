from ast import Pass
import logging
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .homework import Optim
from .utils import DataImpactSerializer


class OptimExp(Optim, DataImpactSerializer):
    def __init__(self, is_decay: bool) -> None:
        self.is_decay = is_decay

    def __repr__(self) -> str:
        decay_str = 'Decay' if self.is_decay else 'not Decay'
        return f'Agent Exponential {decay_str}'

    @staticmethod
    def exponential_decay(a: int, b: int, N: int) -> int:
        '''Exponential function decay
        ---
        params:
            a: term 1 mult
            b: term 2
            N: number of samples int
        '''
        # a, b: exponential decay parameter
        # N: number of samples
        return (a * (1-b) ** np.arange(N)).astype(int)

    @staticmethod
    def exponential_increase(a: int, b: int, N: int) -> int:
        '''Exponential function increase
        ---
        params:
            a: term 1
            b: term 2 exp
            N: number of samples int
        ---
        returns:
            gen: exp value
        '''
        gen = np.logspace(np.log(1), np.log(a), int(N), base=np.exp(b)).astype(int)
        gen = list(filter((0).__ne__, gen))
        return gen

    def frequency(self, freq_split: int, now_ts: dt.datetime, deadline_ts: dt.datetime, init_ts: dt.datetime) -> int:
        '''Exponential decay in callback minutes
        ---
        params:
            freq_split: term 1
            now_ts: call timestamp
            deadline_ts: job request deadline
            init_ts: job request estimated init
        ---
        returns:
            callback: minutes till next call
        '''
        t_diff, s = divmod(
            (deadline_ts - init_ts).total_seconds(), 60)
        m_to_dead, s = divmod(
            (deadline_ts - now_ts).total_seconds(), 60)
        m_init, s_init = divmod(
            (now_ts - init_ts).total_seconds(), 60)

        if self.is_decay:
            n_calls = freq_split/(t_diff/t_diff**1.05)
            trial = self.exponential_decay(t_diff*0.15, 0.15, n_calls)
            try:
                callback_time_minutes = trial[sum(np.cumsum(trial[::-1])[::-1] > m_to_dead):][0]
            except:
                callback_time_minutes = int(m_to_dead-1)
        else:
            try:
                n_calls = freq_split/(t_diff/t_diff**1.05)
                trial = self.exponential_increase(t_diff*0.15, 1, n_calls)
                callback_time_minutes = trial[sum(np.cumsum(trial) < m_init)]
            except:
                callback_time_minutes = int(m_to_dead-1)

        return min(m_to_dead-1, callback_time_minutes)

    @staticmethod
    def severity(N, freq: int) -> int:
        '''Severity modeled as the number of impacts
        ---
        params:
            freq: int
        ---
        returns:
            callback: minutes till next call
        '''
        return int(round(N/freq))

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
        finished = now >= deadline
        fulfilled = (num_vacancies - super().get_total_contract_accepted(impacted_candidates_data)) <= 0
        if finished | fulfilled | (num_remaining_in_pool <= 0):
            num_candidates_needed, callback_time_minutes = [0, 0]
            return True, num_candidates_needed, callback_time_minutes

        total_pool = super().get_total_pool(num_remaining_in_pool, impacted_candidates_data)
        init_ts = super().get_init_ts(now, impacted_candidates_data)
        freq_split = 18

        callback_time_minutes = self.frequency(freq_split, now, deadline, init_ts)
        num_candidates_needed = self.severity(total_pool, freq_split)

        return finished, num_candidates_needed, int(callback_time_minutes)
