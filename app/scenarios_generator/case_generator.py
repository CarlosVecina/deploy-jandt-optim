from scipy.stats import skellam
import random
import datetime
import pandas as pd
from itertools import chain
from numpy.random import choice
from scipy.stats import weibull_min

##notification prob matrix
NOTIFICATION_STATUS = ["ir_pending", "ir_accepted", "ir_rejected"]
WEIGHT_ACCEPTED = .35
WEIGHT_REJECTED = .2
#call transition matrix
OFFER_ACC_P = .6
#transition matrix
TRANSITION_MATRIX = pd.DataFrame([
    ['ir_pending', 'not_in_ft', 1],
    ['ir_accepted', 'offer_accepted', OFFER_ACC_P],
    ['ir_accepted', 'cancelled', 1-OFFER_ACC_P],
    ['ir_rejected', 'cancelled', 1]],
    columns=['notification_status', 'offer_status', 'prob'])


class CaseGenerator():
    def __init__(self, name="1"):
        self.name = name
        self.remaining_pool = skellam.rvs(200, 50, size=1)[0]
        self.num_vacancies = skellam.rvs(9, 2, size=1)[0]
        self.init_date, self.deadline = self.get_dates()
        self.now = self.init_date
        self.counter = 0
        self.per_impacted_list = [{'notification_status': 'ir_accepted', 'candidate_status': 'offer_accepted', 'time_to_respond_ir_minutes': 5}, {'notification_status': 'ir_pending', 'candidate_status': 'not_in_ft', 'time_to_respond_ir_minutes': 14}, {'notification_status': 'ir_pending', 'candidate_status': 'not_in_ft', 'time_to_respond_ir_minutes': 5}, {'notification_status': 'ir_pending', 'candidate_status': 'not_in_ft', 'time_to_respond_ir_minutes': 8}, {'notification_status': 'ir_pending', 'candidate_status': 'not_in_ft', 'time_to_respond_ir_minutes': 23}]

    def _int_uniform(self, a, b):
        return int(random.uniform(a, b))

    def get_dates(self):
        init_date = datetime.datetime(
            2022, self._int_uniform(1, 12), self._int_uniform(1, 28)
        )
        end_date = init_date + random.random() * datetime.timedelta(days=self._int_uniform(1, 20)) #avg diff -5.5 days, 18 var
        return init_date, end_date

    def get_impacted_list(self, n_imp, mins):
        lH = []
        for i in range(n_imp):
            draw = choice(
                NOTIFICATION_STATUS,
                1,
                p=[1-(WEIGHT_ACCEPTED+WEIGHT_REJECTED), WEIGHT_ACCEPTED, WEIGHT_REJECTED])[0]
            draw_candidate = choice(
                TRANSITION_MATRIX[TRANSITION_MATRIX.notification_status == draw].offer_status,
                1,
                p=TRANSITION_MATRIX[TRANSITION_MATRIX.notification_status == draw].prob)[0]

            n = 1     # number of samples
            k = 2.4     # shape
            lam = 10.5   # scale
            if draw != 'ir_pending':
                resp_time = weibull_min.rvs(k, loc=0, scale=lam, size=n).astype(int)[0] #not in fitdist test optims
            else:
                resp_time = mins
            lH.extend([
                    {
                        "notification_status": draw,
                        "candidate_status": draw_candidate,
                        "time_to_respond_ir_minutes": resp_time
                    },
            ])
            #TODO: it could be implemented and stop for offer_accepted >= num_vacancies or not.
        return lH

    def actualize_persisted_impacted_list(self, mins):
        for _, n in enumerate(self.per_impacted_list):
            if n['notification_status'] != 'ir_pending':
                continue
            else:
                pass
            roll_notification = choice(
                NOTIFICATION_STATUS, 1,
                p=[1-(WEIGHT_ACCEPTED+WEIGHT_REJECTED),
                WEIGHT_ACCEPTED,
                WEIGHT_REJECTED]
                )[0]
            roll_offer = choice(
                TRANSITION_MATRIX[TRANSITION_MATRIX.notification_status == roll_notification].offer_status,
                1,
                p=TRANSITION_MATRIX[TRANSITION_MATRIX.notification_status == roll_notification].prob
                )[0]

            if roll_notification == 'ir_pending':
                roll_mins = n['time_to_respond_ir_minutes'] + mins
            elif roll_notification == 'ir_accepted':
                roll_mins = n['time_to_respond_ir_minutes'] + min(mins, int(weibull_min.rvs(2.4, loc=0, scale=10, size=1).astype(int)[0]))
            else:
                roll_mins = n['time_to_respond_ir_minutes'] + min(mins, int(weibull_min.rvs(2.4, loc=0, scale=10, size=1).astype(int)[0]))

            n['notification_status'] = roll_notification
            n['candidate_status'] = roll_offer
            n['time_to_respond_ir_minutes'] = roll_mins


    def persist_impacted_list(self, impact_list):
        self.per_impacted_list.extend(impact_list)


    def batch_process_generator(self, n_inv, mins):
        self.now = self.now + datetime.timedelta(minutes=mins)

        if self.remaining_pool < n_inv:
            n_inv = self.remaining_pool
        if self.remaining_pool <= 0:
            return None
        if ((self.deadline - self.now).days) < 0 & ((self.deadline - self.now).seconds)<60:
            return
        if (mins<=0):
            return None #raise ValueError('N invs is bigger than the remaining pool.')
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
            "impacted_candidates_data": batch_impacted_list
        }

        self.counter += 1
        yield dict_req

    def online_generator():
        pass