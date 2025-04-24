from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import pandas as pd
import yaml
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

class SalesForecaster:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_dir = self.config['model']['path']
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {}

    def train(self, data):
        """Train a Prophet model per ingredient."""
        params = self.config['prophet']
        for item in data['item'].unique():
            item_data = data[data['item'] == item][['date', 'quantity']].rename(columns={'date': 'ds', 'quantity': 'y'})
            model = Prophet(
                yearly_seasonality=params['yearly_seasonality'],
                weekly_seasonality=params['weekly_seasonality'],
                daily_seasonality=params['daily_seasonality']
            )
            model.fit(item_data)
            self.models[item] = model
            # Save model to JSON
            with open(f"{self.model_dir}/{item}_model.json", 'w') as f:
                f.write(model_to_json(model))

    def load_models(self):
        """Load pre-trained Prophet models."""
        for item in os.listdir(self.model_dir):
            if item.endswith('_model.json'):
                item_name = item.replace('_model.json', '')
                with open(f"{self.model_dir}/{item}", 'r') as f:
                    self.models[item_name] = model_from_json(f.read())

    def predict(self, future_data):
        """Predict future quantities with 2 decimal places."""
        if not self.models:
            self.load_models()
        predictions = []
        for item in future_data['item'].unique():
            item_data = future_data[future_data['item'] == item][['date']].rename(columns={'date': 'ds'})
            if item in self.models:
                forecast = self.models[item].predict(item_data)
                for date, pred in zip(forecast['ds'], forecast['yhat']):
                    predictions.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'item': item,
                        'predicted_quantity': round(pred, 2)  # 2 decimals
                    })
        return pd.DataFrame(predictions)

    def evaluate(self, test_data):
        """Calculate RMSE and MAPE per ingredient."""
        if not self.models:
            self.load_models()
        results = []
        for item in test_data['item'].unique():
            item_data = test_data[test_data['item'] == item][['date', 'quantity']].rename(columns={'date': 'ds', 'quantity': 'y'})
            forecast = self.models[item].predict(item_data[['ds']])
            y_true = item_data['y']
            y_pred = forecast['yhat']
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            results.append({'item': item, 'rmse': rmse, 'mape': mape})
        return pd.DataFrame(results)