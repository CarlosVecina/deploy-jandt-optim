import datetime as dt
import pandas as pd

class DataImpactSerializer():
    @staticmethod
    def get_total_pool(pool: int, impact_data: list) -> int:
        '''Return the total pool size
        ---
        params:
            pool: current remaining pool
            impact_data: list of impacts
        returns:
            int: total pool size
        '''
        return int(pool + len(impact_data))

    @staticmethod
    def get_init_ts(now: dt.datetime, impact_data: list) -> dt.datetime:
        '''Estimate the init time w/o persistence, observing the impact data
        ---
        params:
            now: now datetime
            impact_data: list of impacts
        returns:
            init: returns the estimated init datetime of the job opening
        '''
        max_mins = pd.DataFrame(list(impact_data)).get('time_to_respond_ir_minutes')
        if max_mins is not None:
            max_mins = max_mins.max()
            return now + dt.timedelta(minutes=max_mins)
        else:
            return now

    @staticmethod
    def get_n_first_accepted(impact_data: list) -> int:
        """
        Count the trials before an accepted
        ---
        params:
            impact_data: list of impacts
        returns:
            t_def: perc of accepted in impact data
        """
        t_def = 0
        try:
            data = pd.DataFrame(list(impact_data))
            t_def = round(data.groupby('notification_status').notification_status.count()['ir_accepted'][0]/len(impact_data)*100)
        except:
            pass
        return t_def

    @staticmethod
    def get_t_response_accepted(impact_data: list) -> list:
        """
        Extract the avg response time for accepted
        ---
        params:
            impact_data: list of impacts
        returns:
            list of accepted response times if exist any in impact data
        """
        data = pd.DataFrame(list(impact_data))
        if 'notification_status' in data.columns:
            return data[data.notification_status=='ir_accepted'].time_to_respond_ir_minutes.values.astype(int)
        else:
            return [0]

    @staticmethod
    def get_avg_t_response_accepted(impact_data: list, default: int) -> int:
        """
        Extract the avg response time for accepted
        ---
        params:
            impact_data: list of impacts
            default: default value if can't extract
        returns:
            avg of response time of accepted
        """
        data = pd.DataFrame(list(impact_data))
        try:
            if (data.shape[0] >= 3) & (len(data[data.notification_status == 'ir_accepted'])):
                return data[
                    data.notification_status == 'ir_accepted'].time_to_respond_ir_minutes.mean(), data[data.notification_status == 'ir_accepted'
                ].time_to_respond_ir_minutes.count()
            else:
                return default, 0
        except:
            return default, 0

    @staticmethod
    def get_total_contract_accepted(impact_data: list) -> int:
        """
        Extract the total contract accepted ir_contract_accepted
        ---
        params:
            impact_data: list of impacts
        returns:
            t_acc: count of accepted offers
        """
        try:
            data = pd.DataFrame(list(impact_data))
            t_acc = data[data.candidate_status == 'offer_accepted'].candidate_status.count()
            return t_acc
        except:
            return 0
