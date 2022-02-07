import random
import datetime
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from itertools import chain
from numpy.random import choice
from scipy.stats import skellam, weibull_min
import copy


##notification prob matrix
NOTIFICATION_STATUS = ["ir_pending", "ir_accepted", "ir_rejected"]

#transition matrix
def build_transition_matrix(offer_acc_prob: float) -> pd.DataFrame:
    '''Crafting the transition prob matrix
    ---
    params:
        offer_acc_prob: probabilities of accept a notification
    ---
    returns:
        transition_matrix: the path trans matrix
    '''
    transition_matrix = pd.DataFrame([
        ['ir_pending', 'not_in_ft', 1],
        ['ir_accepted', 'offer_accepted', offer_acc_prob],
        ['ir_accepted', 'cancelled', 1-offer_acc_prob],
        ['ir_rejected', 'cancelled', 1]],
        columns=['notification_status', 'offer_status', 'prob'
    ])
    return transition_matrix

def est_prior_beta_params(mu: float, var: float) -> Tuple[float, float]:
    '''Given mean and varianze, estimate Beta dist param
    ---
    params:
        mu: mean beta dist
        var: var beta dist
    ---
    returns;
        alpha: alpha dist param
        beta: beta dist param
    '''
    alpha = ((1.0 - mu) / var - 1.0 / mu) * mu ** 2.0
    beta = alpha * (1 / mu - 1)
    return alpha, beta


class CaseGenerator():
    def __init__(
        self,
        name: str = "1",
        w_acc: float = 0.1,
        w_rej: float = 0.1,
        offer_acc_prob: float = 0.6,
        param_pool: int = 400,
        param_vacancies: int = 9
        ) -> None:
        self.name = name
        self.remaining_pool = skellam.rvs(param_pool, int(param_pool*0.2), size=1)[0]
        self.num_vacancies = skellam.rvs(param_vacancies, int(param_vacancies*0.2), size=1)[0]
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
        self.remaining_pool = e_dict['num_remaining_in_pool']
        self.per_impacted_list = e_dict['impacted_candidates_data']

    def set_name(self, new_name: str) -> None:
        '''Set correlation_id first id
        ---
        params:
            new_name str'''
        self.name = new_name

    def reset_counter(self) -> None:
        '''Reset counter'''
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

    def generator(self, n_inv: int, mins: int) -> dict:
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


class ScenarioInitializer():
    def __init__(self, n_cases: int) -> None:
        self.n_cases = n_cases

    def generator(self) -> list:
        '''Initialize n_cases scenarios to share as starting points between agents
        ---
        params:
            n_cases: number of initial states to create
        ---
        yields:
            req: list of initial states requests
        '''
        for c in range(self.n_cases):
            case_suite = CaseGenerator(name=str(c), w_acc=0.1, w_rej=0.1, offer_acc_prob=0.6)
            req = [True]
            callback_time_minutes=1
            num_candidates_needed=0
            req = [tup for tup in case_suite.generator(n_inv=num_candidates_needed,
                                                       mins=callback_time_minutes
                                                      )]
            yield req


class ScenarioSimulator():
    def __init__(self, opt_obj, case_obj) -> None:
        self.opt_obj = opt_obj
        self.case_obj = case_obj

    def generator(self, initial_scenarios: list) -> list:
        '''Create the scenario path given a previous state and agent interaction
        ---
        params:
            initial_scenarios: list of initial states
        ---
        returns:
            req: list of all the results after the interaction of the agent with the environment
        '''
        _l=[]
        counter = 0
        ii = copy.deepcopy(initial_scenarios)
        for c in ii:
            #print(i)
            self.case_obj.set_name(str(counter))
            self.case_obj.reset_counter()
            self.case_obj.init_from_event(c[0])
            #case_suite = CaseGenerator(str(i), w_acc=0.1, w_rej=0.1, offer_acc_prob=0.6)
            req = [True]
            callback_time_minutes=1
            num_candidates_needed=0
            counter += 1
            while len(req) > 0:
                # Optimizer Logic
                req = [tup for tup in self.case_obj.generator(n_inv=num_candidates_needed,
                                                           mins=callback_time_minutes )]
                if len(req) > 0:
                    #finished, num_candidates_needed, callback_time_minutes = opt_nbin.invitation_logic_api(
                    finished, num_candidates_needed, callback_time_minutes = self.opt_obj.invitation_logic_api(
                        now=req[0]["reference_date_time"],
                        deadline=req[0]["deadline"],
                        num_vacancies=req[0]["num_vacancies"],
                        num_remaining_in_pool=req[0]["num_remaining_in_pool"],
                        impacted_candidates_data=req[0]["impacted_candidates_data"]
                    )
                    req[0].update({'finished': finished})
                    req[0].update({'num_candidates_needed': num_candidates_needed})
                    req[0].update({'callback_time_minutes': callback_time_minutes})
                    if len(req[0]["impacted_candidates_data"])>0:
                        req[0].update({'total_accepted': pd.DataFrame(req[0]["impacted_candidates_data"]).candidate_status.str.contains('offer').sum()})
                    else:
                        req[0].update({'total_accepted':0})
                    req[0].update({'optim': self.opt_obj})
                _l.extend(req)
        return _l

    def get_optim_current_state(self):
        '''Return the current state of the agent
        '''
        return self.opt_obj
