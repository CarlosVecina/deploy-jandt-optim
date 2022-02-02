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

    @staticmethod
    def exponential_decay(
        a:float, b:float, N:float
    ) -> int:
        # a, b: exponential decay parameter
        # N: number of samples
        return (a * (1-b) ** np.arange(N)).astype(int)

    @staticmethod
    def exponential_increase(a:float, b:float, N:int) -> int:
        gen = np.logspace(np.log(1), np.log(a), int(N), base=np.exp(b)).astype(int)
        l_gen = list(filter((0).__ne__, gen))
        return l_gen

    def frequency(
        self, freq_split:float, now_ts:dt.datetime, deadline_ts:dt.datetime, init_ts: dt.datetime
        ) -> int:
        ## exponential decay in callback time minutes
        t_diff, s = divmod(
            (deadline_ts - init_ts).total_seconds(),
             60)
        m_to_dead, s = divmod(
            (deadline_ts - now_ts).total_seconds(),
             60)
        m_init, s_init = divmod(
            (now_ts - init_ts).total_seconds(),
            60)
        n_calls=freq_split/(t_diff/t_diff**1.05)

        if self.is_decay:
            trial = self.exponential_decay(t_diff*0.15, 0.15, n_calls)
            try:
                callback_time_minutes=trial[sum(np.cumsum(trial[::-1])[::-1] > m_to_dead):][0]
            except:
                callback_time_minutes=int(m_to_dead-1)
        else:
            trial = self.exponential_increase(t_diff*0.15, 1, n_calls)
            try:
                callback_time_minutes = trial[sum(np.cumsum(trial) < m_init)]
            except:
                callback_time_minutes=int(m_to_dead-1)

        return min(m_to_dead-1, callback_time_minutes)

    @staticmethod
    def severity(N, freq: float) -> int:
        return int(round(N/freq))

    def invitation_logic_api(
        self, now, deadline, num_vacancies, num_remaining_in_pool, impacted_candidates_data
    )-> Tuple[bool, int, Optional[int]]:
        finished = now >= deadline
        fulfilled = (num_vacancies - super().get_total_contract_accepted(impacted_candidates_data)) <= 0
        if finished | fulfilled:
            num_candidates_needed, callback_time_minutes = [0,0]
            return finished, num_candidates_needed, callback_time_minutes

        total_pool = super().get_total_pool(num_remaining_in_pool ,impacted_candidates_data)
        init_ts = super().get_init_ts(now, impacted_candidates_data)
        freq_split= 18

        callback_time_minutes = self.frequency(freq_split, now, deadline, init_ts)
        num_candidates_needed = self.severity(total_pool, freq_split)

        return False, num_candidates_needed, int(callback_time_minutes)


#print(1)
#
#tt_true = OptimExp(is_decay=True)
#tt_false = OptimExp(is_decay=False)
#
#tt_false.invitation_logic_api(
#    dt.datetime(2021,1,1),
#    dt.datetime(2021,3,12),
#    10,
#    95,
#    [None]
#)