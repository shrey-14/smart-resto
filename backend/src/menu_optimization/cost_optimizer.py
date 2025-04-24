# src/cost_optimizer.py

import pandas as pd
import yaml
import os
import logging
from datetime import datetime

class CostOptimizer:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(), logging.FileHandler('cost_optimizer.log')]
        )
        self.logger = logging.getLogger(__name__)

    def _round_to_99(self, price):
        """Round to nearest value ending in .99"""
        # Round to the nearest 10, then adjust to end in .99
        rounded_up = int(price / 10.0 + 0.5) * 10  # Nearest 10
        return rounded_up - 0.01 if price <= rounded_up else rounded_up + 9.99

    def optimize_costs(self, target_profit_margin=0.3, profitability_threshold=0.7):
        """Optimize costs and suggest menu adjustments"""
        try:
            # Load cost data
            cost_df = pd.read_csv(self.config['data']['recipe_path'])
            
            # Check if cost data is provided
            if cost_df is None or cost_df.empty:
                print("Error: No cost optimization data available.")
                return None

            # No inventory data required; proceed with cost data only
            print("Note: Running cost optimization without inventory data. Expiry-based adjustments will be skipped.")

            # Calculate total cost per dish
            dish_costs = {}
            for dish_name, group in cost_df.groupby('recipe_name'):
                total_cost = group['prep_cost_inr'].sum()  # Sum preparation costs for all ingredients
                dish_costs[dish_name] = total_cost

            # Suggest menu adjustments (without inventory checks)
            suggestions = []
            for dish_name, cost in dish_costs.items():
                base_selling_price = cost / (1 - target_profit_margin)  # Base price before rounding
                selling_price = self._round_to_99(base_selling_price)  # Round to nearest .99
                profit_margin = 1 - (cost / selling_price)  # Recalculate margin with rounded price
                adjustment = "Maintain stock level"
                reasoning = f"Cost: {cost:.2f} INR, Suggested Price: {selling_price:.2f} INR, Margin: {profit_margin:.2%}."

                # Check for low profitability
                if cost / selling_price > profitability_threshold:
                    adjustment = "Review or Remove"
                    reasoning += f" {adjustment} - Cost ({cost:.2f} INR) exceeds {profitability_threshold*100}% of price."

                suggestions.append({
                    "dish_name": dish_name,
                    "dish_cost_inr": round(cost, 2),
                    "suggested_price_inr": selling_price,
                    "profit_margin": round(profit_margin * 100, 2),
                    "adjustment": adjustment,
                    "reasoning": reasoning
                })

            # Output suggestions to terminal
            if suggestions:
                print("\nCost Optimization and Menu Adjustment Suggestions:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"\nSuggestion {i}:")
                    print(f"Dish Name: {suggestion['dish_name']}")
                    print(f"Dish Cost: {suggestion['dish_cost_inr']} INR")
                    print(f"Suggested Selling Price: {suggestion['suggested_price_inr']} INR")
                    print(f"Profit Margin: {suggestion['profit_margin']}%")
                    print(f"Adjustment: {suggestion['adjustment']}")
                    print(f"Reasoning: {suggestion['reasoning']}")
                return suggestions
            else:
                print("No cost optimization suggestions generated.")
                self.logger.warning("No dishes generated for cost optimization.")
                return None

        except Exception as e:
            self.logger.error(f"Error in cost optimization: {e}")
            print(f"Error: An unexpected error occurred while optimizing costs: {e}")
            return None

if __name__ == "__main__":
    # This block will not be executed when called from main.py
    pass