import datetime as dt
import pandas as pd

class DataImpactSerializer():
    @staticmethod
    def get_total_pool(pool: int, impact_data: list) -> int:
        return int(pool/2)

    @staticmethod
    def get_init_ts(now: dt.datetime, impact_data: list) -> dt.datetime:
        max_mins = pd.DataFrame(list(impact_data)).get('time_to_respond_ir_minutes')
        if max_mins is not None:
            max_mins = max_mins.max()
            return now + dt.timedelta(minutes= max_mins)
        else:
            return now

    @staticmethod
    def get_n_first_accepted(impact_data) -> int:
        """
        Count the trials before an accepted
        """
        t_def = 0
        try:
            data = pd.DataFrame(list(impact_data))
            t_def = round(data.groupby('notification_status').notification_status.count()[0]/len(impact_data)*100)
        except:
            pass
        return t_def

    @staticmethod
    def get_t_response_accepted(impact_data: list) -> list:
        """
        Extract the avg response time for accepted
        """
        data = pd.DataFrame(list(impact_data))
        if 'notification_status' in data.columns:
            return data[data.notification_status == 'ir_accepted'].time_to_respond_ir_minutes.values.astype(int)
        else:
            return [0]

    @staticmethod
    def get_avg_t_response_accepted(impact_data: list, default: int) -> int:
        """
        Extract the avg response time for accepted
        """
        data = pd.DataFrame(list(impact_data))
        try:
            if (data.shape[0] >= 3) & (len(data[data.notification_status == 'ir_accepted'])):
                return data[data.notification_status == 'ir_accepted'].time_to_respond_ir_minutes.mean(), data[data.notification_status=='ir_accepted'].time_to_respond_ir_minutes.count()
            else:
                return default, 0
        except:
            return default, 0

    @staticmethod
    def get_total_contract_accepted(impact_data: list) -> int:
        """
        Extract the total contract accepted ir_contract_accepted
        """
        try:
            data = pd.DataFrame(list(impact_data))
            t_acc = data[data.candidate_status=='offer_accepted'].candidate_status.count()
            return t_acc
        except:
            return 0
