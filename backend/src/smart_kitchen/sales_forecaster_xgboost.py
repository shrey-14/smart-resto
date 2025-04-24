import xgboost as xgb
import pandas as pd
import pickle
import yaml
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

class SalesForecaster:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_dir = self.config['model']['path'].replace('.pkl', '')
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {}

    def train(self, data):
        params = self.config['xgboost']
        for item in data['item'].unique():
            item_data = data[data['item'] == item]
            X = self.prepare_features(item_data)
            y = item_data['quantity']
            model = xgb.XGBRegressor(
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                objective='reg:squarederror'
            )
            model.fit(X, y)
            self.models[item] = model
            with open(f"{self.model_dir}/{item}_model.pkl", 'wb') as f:
                pickle.dump(model, f)

    def load_models(self):
        for item in os.listdir(self.model_dir):
            if item.endswith('_model.pkl'):
                item_name = item.replace('_model.pkl', '')
                with open(f"{self.model_dir}/{item}", 'rb') as f:
                    self.models[item_name] = pickle.load(f)

    def predict(self, future_data):
        if not self.models:
            self.load_models()
        predictions = []
        for item in future_data['item'].unique():
            item_data = future_data[future_data['item'] == item]
            X = self.prepare_features(item_data)
            if item in self.models:
                preds = self.models[item].predict(X)
                for date, pred in zip(item_data['date'], preds):
                    predictions.append({'date': date, 'item': item, 'predicted_quantity': pred})
        return pd.DataFrame(predictions)

    def evaluate(self, test_data):
        """Calculate RMSE and MAPE per ingredient."""
        if not self.models:
            self.load_models()
        results = []
        for item in test_data['item'].unique():
            item_data = test_data[test_data['item'] == item]
            X = self.prepare_features(item_data)
            y_true = item_data['quantity']
            if item in self.models:
                y_pred = self.models[item].predict(X)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # As percentage
                results.append({'item': item, 'rmse': rmse, 'mape': mape})
        return pd.DataFrame(results)

    def prepare_features(self, data):
        return data[['day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_7']]