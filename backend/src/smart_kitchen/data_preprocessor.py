import pandas as pd

class DataPreprocessor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])

    def add_time_features(self):
        """Add seasonality features."""
        self.data['day_of_week'] = self.data['date'].dt.dayofweek
        self.data['month'] = self.data['date'].dt.month
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)

    def add_lag_features(self, lag_days=[1, 7]):
        """Add lagged sales features per item."""
        for item in self.data['item'].unique():
            item_data = self.data[self.data['item'] == item].sort_values('date')
            for lag in lag_days:
                self.data.loc[self.data['item'] == item, f'lag_{lag}'] = \
                    item_data['quantity'].shift(lag)
        self.data.dropna(inplace=True)

    def preprocess(self):
        """Run preprocessing pipeline."""
        self.add_time_features()
        self.add_lag_features()
        return self.data

    def save(self, output_path):
        self.data.to_csv(output_path, index=False)