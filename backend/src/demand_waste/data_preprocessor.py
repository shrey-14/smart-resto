# import pandas as pd
# from datetime import datetime

# def load_inventory_data(input_path: str) -> pd.DataFrame:
#     df = pd.read_csv(input_path)
#     current_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
#     df['delivery_date'] = pd.to_datetime(df['delivery_date'])
#     df['days_since_delivery'] = (current_date - df['delivery_date']).dt.days
#     return df

import pandas as pd
from datetime import datetime

def load_inventory_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    current_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    df['days_since_delivery'] = (current_date - df['delivery_date']).dt.days
    return df