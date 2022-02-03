import random
import datetime
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from itertools import chain
from numpy.random import choice
from scipy.stats import skellam, weibull_min


##notification prob matrix
NOTIFICATION_STATUS = ["ir_pending", "ir_accepted", "ir_rejected"]

#transition matrix
def build_transition_matrix(offer_acc_prob: float) -> pd.DataFrame:
    transition_matrix = pd.DataFrame([
        ['ir_pending', 'not_in_ft', 1],
        ['ir_accepted', 'offer_accepted', offer_acc_prob],
        ['ir_accepted', 'cancelled', 1-offer_acc_prob],
        ['ir_rejected', 'cancelled', 1]],
        columns=['notification_status', 'offer_status', 'prob'
    ])
    return transition_matrix


class CaseGenerator():
    def __init__(
        self,
        name: str = "1",
        w_acc: float = 0.1,
        w_rej: float = 0.1,
        offer_acc_prob: float = 0.6
        ) -> None:
        self.name = name
        self.remaining_pool = skellam.rvs(200, 50, size=1)[0]
        self.num_vacancies = skellam.rvs(9, 2, size=1)[0]
        self.init_date, self.deadline = self.get_dates()
        self.now = self.init_date
        self.counter = 0
        self.per_impacted_list = []
        self.w_acc = w_acc
        self.w_rej = w_rej
        self.transition_matrix = build_transition_matrix(offer_acc_prob)

    def _int_uniform(self, a: int, b: int) -> int:
        '''Generates a random number from a uniform distribution.
        ---
        params:
            a: lower bound.
            b: upper bound.
        returns:
            integer random number.
        '''
        return int(random.uniform(a, b))

    def init_from_event(self, e_dict: dict) -> None:
        self.now = e_dict['reference_date_time']
        self.deadline = e_dict['deadline']
        self.num_vacancies = e_dict['num_vacancies']
        self.num_remaining_in_pool = e_dict['num_remaining_in_pool']
        self.per_impacted_list = e_dict['impacted_candidates_data']

    def set_name(self, new_name: str) -> None:
        self.name = new_name

    def reset_counter(self) -> None:
        self.counter = 0

    def get_dates(self) -> Tuple[datetime.datetime, datetime.datetime]:
        '''Generates random init date in year 2022 and end date. Simplified last month days logic.
        ---
        '''
        init_date = datetime.datetime(
            2022, self._int_uniform(1, 12), self._int_uniform(1, 28)
        )
        end_date = init_date + random.random() * datetime.timedelta(days=self._int_uniform(1, 20)) #avg diff -5.5 days, var 18
        return init_date, end_date

    def get_impacted_list(self, n_imp: int, mins: int) -> list:
        '''Generates the new impacted workers list of dicts. Cointains notification_status, candidate_status,
        and time_to_respond_ir_minutes.
        ---
        params:
            n_imp: Number of impacts news to create.
            mins: Number of minutes since the last call.
        returns:
            output_lst: List of dicts with impact data.
        '''
        output_lst = []
        for i in range(n_imp):
            draw = choice(
                NOTIFICATION_STATUS,
                1,
                p=[1-(self.w_acc+self.w_rej), self.w_acc, self.w_rej])[0]
            draw_candidate = choice(
                self.transition_matrix[self.transition_matrix.notification_status == draw].offer_status,
                1,
                p=self.transition_matrix[self.transition_matrix.notification_status == draw].prob)[0]

            n = 1       # n samples
            k = 2.4     # shape
            lam = 10.5  # scale
            if draw != 'ir_pending':
                resp_time = weibull_min.rvs(k, loc=0, scale=lam, size=n).astype(int)[0] #not in fitdist test optims
                #TODO: parametrized dist and args
            else:
                resp_time = mins
            output_lst.extend([
                    {
                        "notification_status": draw,
                        "candidate_status": draw_candidate,
                        "time_to_respond_ir_minutes": resp_time
                    },
            ])
            # TODO: it could be implemented and stop for offer_accepted >= num_vacancies or not.
        return output_lst

    def actualize_persisted_impacted_list(self, mins: int) -> None:
        '''Re-evaluates inplace the persisted impact list, switching pending states probabilistically.
        ---
        params:
            mins: Minutes since de last call
        '''
        for impact in self.per_impacted_list:
            if impact['notification_status'] != 'ir_pending':
                continue
            else:
                pass
            roll_notification = choice(
                NOTIFICATION_STATUS, 1,
                p=[1-(self.w_acc+self.w_rej),
                self.w_acc,
                self.w_rej]
                )[0]
            roll_offer = choice(
                self.transition_matrix[self.transition_matrix.notification_status == roll_notification].offer_status,
                1,
                p=self.transition_matrix[self.transition_matrix.notification_status == roll_notification].prob
                )[0]

            if roll_notification == 'ir_pending':
                roll_mins = impact['time_to_respond_ir_minutes'] + mins
            elif roll_notification == 'ir_accepted':
                roll_mins = impact['time_to_respond_ir_minutes'] + min(mins, int(weibull_min.rvs(2.4, loc=0, scale=10, size=1).astype(int)[0]))
            else:
                roll_mins = impact['time_to_respond_ir_minutes'] + min(mins, int(weibull_min.rvs(2.4, loc=0, scale=10, size=1).astype(int)[0]))

            impact['notification_status'] = roll_notification
            impact['candidate_status'] = roll_offer
            impact['time_to_respond_ir_minutes'] = roll_mins

    def persist_impacted_list(self, impact_list: list) -> None:
        '''Aproximate a 'persist' using an object variable.
        ---
        params:
            impact_list: current impact_list
        '''
        self.per_impacted_list.extend(impact_list)

    def generator(self, n_inv, mins):
        '''Main method triggering the generator. Simulates probabilistically an environment.
        ---
        params:
            n_inv: number of notifications sent by the agent
            mins: minnutes callback sent by the agent
        yields:
            dict request body
        '''
        self.now = self.now + datetime.timedelta(minutes=int(mins))

        if self.remaining_pool < n_inv:
            n_inv = self.remaining_pool
        if self.remaining_pool <= 0:
            return None
        if ((self.deadline - self.now).days) < 0 & ((self.deadline - self.now).seconds)<60:
            return
        if (mins<=0):
            return
        # TODO: Vacancies all accepted stop criterion

        self.remaining_pool = max(0,self.remaining_pool-n_inv)
        batch_impacted_list = self.get_impacted_list(n_inv, mins)
        self.actualize_persisted_impacted_list(mins)
        self.persist_impacted_list(batch_impacted_list)

        dict_req={
            "correlation_id": f"Case{self.name}_{self.counter}",
            "reference_date_time": self.now,
            "deadline": self.deadline,
            "num_vacancies": self.num_vacancies,
            "num_remaining_in_pool": self.remaining_pool,
            "impacted_candidates_data": self.per_impacted_list
        }

        self.counter += 1
        yield dict_req

    def online_generator():
        pass