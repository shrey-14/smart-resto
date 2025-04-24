import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from dateutil.parser import parse
import yaml
class RestaurantWasteTracker:
    def __init__(self, config_path=None, sales_data_file=None):
        """
        Initialize the waste tracker with the provided data files
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load actual data
        self.inventory_data =  pd.read_csv(self.config['data']['inventory_path'])
        self.recipe_data = pd.read_csv(self.config['data']['recipe_path'])
        self.output_dashboard_path = self.config['data']['output_dashboard_path']
        self.sales_data_file = self.config['data']['sales_data_file']

        # Standard portion size in grams (can be adjusted)
        self.standard_portion_size = 250
        
        # Average labor cost per dish in INR
        self.labor_cost_per_dish = 50
        
        # Process inventory and recipe data
        self.process_inventory_data()
        self.process_recipe_data()
        
        # Load or generate sales data
        if sales_data_file and os.path.exists(sales_data_file):
            self.sales_data = pd.read_csv(sales_data_file)
            self.sales_data['date'] = pd.to_datetime(self.sales_data['date'])
        else:
            self.sales_data = self.generate_sales_data()
            if sales_data_file:
                self.sales_data.to_csv(sales_data_file, index=False)
        
        # Generate waste data based on sales and inventory
        self.waste_data = self.generate_waste_data()
    
    def process_inventory_data(self):
        """Process inventory data and calculate costs"""
        # Convert delivery date to datetime
        self.inventory_data['delivery_date'] = pd.to_datetime(self.inventory_data['delivery_date'])
        
        # Calculate expiry date based on shelf life
        self.inventory_data['expiry_date'] = self.inventory_data['delivery_date'] + \
                                           pd.to_timedelta(self.inventory_data['shelf_life_days'], unit='d')
        
        # Add cost information (pricing in INR per kg)
        # Updated with more realistic market prices
        ingredient_costs = {
            'spinach': 40,
            'chicken': 180,
            'tomato': 30,
            'lettuce': 40,
            'fish': 200,
            'carrot': 25,
            'mushrooms': 120,
            'celery': 60,
            'onion': 25,
            'bread': 50,
            'butter': 400,
            'cucumber': 30,
            'zucchini': 50,
            'milk': 50,
            'peas': 60,
            'orange': 80,
            'ketchup': 100,
            'cornmeal': 40,
            'cream': 250,
            'egg': 5,  # per egg
            'strawberries': 150,
            'salmon': 500,
            'flour': 30,
            'rice': 60,
            'basil': 150,
            'frozen_peas': 80
        }
        
        # Add missing ingredients from recipe data
        recipe_ingredients = set(self.recipe_data['ingredient'].unique())
        for ingredient in recipe_ingredients:
            if ingredient not in ingredient_costs:
                # More realistic random prices for missing ingredients
                ingredient_costs[ingredient] = random.randint(30, 300)
        
        # Add cost to inventory data
        self.inventory_data['cost_per_kg'] = self.inventory_data['ingredient'].map(ingredient_costs)
        
        # Calculate value of current inventory
        self.inventory_data['current_value'] = self.inventory_data['stock_kg'] * self.inventory_data['cost_per_kg']
        
        # Create ingredient info dictionary for later use
        self.ingredient_info = {}
        for _, row in self.inventory_data.iterrows():
            self.ingredient_info[row['ingredient']] = {
                'cost_per_kg': row['cost_per_kg'],
                'shelf_life_days': row['shelf_life_days'],
                'storage_temp_c': row['storage_temp_c'],
                'weekly_usage_kg': row['weekly_usage_kg']
            }
        
        # Add any missing ingredients from recipes to ingredient_info
        for ingredient in recipe_ingredients:
            if ingredient not in self.ingredient_info:
                self.ingredient_info[ingredient] = {
                    'cost_per_kg': ingredient_costs.get(ingredient, 80),  # Default 80 INR if not found
                    'shelf_life_days': 7,  # Default shelf life
                    'storage_temp_c': 4,  # Default storage temp
                    'weekly_usage_kg': 3   # Default weekly usage
                }
    
    def process_recipe_data(self):
        """Process recipe data to get dish costs and ingredients"""
        # Group recipes by name to calculate total cost and ingredients
        self.recipe_info = {}
        
        for recipe_name in self.recipe_data['recipe_name'].unique():
            recipe_subset = self.recipe_data[self.recipe_data['recipe_name'] == recipe_name]
            
            # Calculate total ingredient cost
            total_prep_cost = recipe_subset['prep_cost_inr'].max()  # Use provided prep cost
            
            # Store ingredients with quantities
            ingredients = {}
            for _, row in recipe_subset.iterrows():
                # Convert to grams/ml for standardization
                if row['unit'] == 'kg':
                    quantity_grams = row['quantity'] * 1000
                elif row['unit'] == 'litre':
                    quantity_grams = row['quantity'] * 1000
                elif row['unit'] == 'dozen':
                    quantity_grams = row['quantity'] * 12 * 50  # Assuming average egg is 50g
                else:
                    quantity_grams = row['quantity'] * 1000  # Default to grams
                
                ingredients[row['ingredient']] = quantity_grams
            
            # Calculate selling price (markup of around 100% over prep cost)
            selling_price = total_prep_cost * 2
            
            # Store recipe information
            self.recipe_info[recipe_name] = {
                'ingredients': ingredients,
                'prep_cost': total_prep_cost,
                'selling_price': selling_price
            }
    
    def generate_sales_data(self, days=365):
        """Generate synthetic sales data for the specified number of days"""
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        sales_records = []
        
        # Create seasonal popularity for dishes
        seasonality = {
            'spring': ['Palak Paneer', 'Cucumber Raita', 'Fruit Chaat', 'Aloo Paratha', 'Mango Lassi'],
            'summer': ['Cucumber Raita', 'Mango Lassi', 'Fruit Chaat', 'Bhindi Masala', 'Poha'],
            'monsoon': ['Pakora', 'Masala Chai', 'Butter Chicken', 'Vegetable Biryani', 'Aloo Paratha'],
            'winter': ['Butter Chicken', 'Gajar Halwa', 'Palak Paneer', 'Aloo Gobi', 'Dal Tadka']
        }
        
        # Map months to seasons in India
        month_to_season = {
            1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'summer', 6: 'summer',
            7: 'monsoon', 8: 'monsoon', 9: 'monsoon', 10: 'autumn', 11: 'autumn', 12: 'winter'
        }
        
        # Base sales volume by day of week (higher on weekends)
        weekday_multiplier = {
            0: 0.8,  # Monday
            1: 0.9,  # Tuesday
            2: 1.0,  # Wednesday
            3: 1.1,  # Thursday
            4: 1.2,  # Friday
            5: 1.5,  # Saturday
            6: 1.4   # Sunday
        }
        
        for date in dates:
            # Get season for this date
            season = month_to_season[date.month]
            
            # Weekend factor (more sales on weekends)
            is_weekend = date.weekday() >= 5
            base_dishes = random.randint(80, 120) if is_weekend else random.randint(40, 80)
            
            # Apply weekday multiplier
            base_dishes = int(base_dishes * weekday_multiplier[date.weekday()])
            
            # Generate sales for dishes
            dishes_sold = []
            
            # Popular dishes for the season get more sales
            seasonal_popular = seasonality.get(season, [])
            
            for recipe_name in self.recipe_info.keys():
                # Base popularity
                popularity = 1.0
                
                # Seasonal popularity boost
                if recipe_name in seasonal_popular:
                    popularity *= 1.5
                
                # Random variation
                popularity *= random.uniform(0.8, 1.2)
                
                # Calculate number of dishes to sell today
                num_to_sell = int(base_dishes * popularity / len(self.recipe_info))
                num_to_sell = max(1, min(num_to_sell, 50))  # Cap between 1 and 50
                
                dishes_sold.append((recipe_name, num_to_sell))
            
            # Generate sales records
            for recipe_name, quantity in dishes_sold:
                recipe = self.recipe_info[recipe_name]
                
                # Calculate food cost for this dish
                food_cost = recipe['prep_cost']
                
                # Add labor cost
                total_cost = food_cost + self.labor_cost_per_dish
                
                # Calculate profit
                profit = recipe['selling_price'] - total_cost
                
                sales_records.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'day_of_week': date.strftime('%A'),
                    'month': date.strftime('%B'),
                    'quarter': f'Q{(date.month-1)//3 + 1}',
                    'dish': recipe_name,
                    'quantity': quantity,
                    'selling_price': recipe['selling_price'],
                    'food_cost': food_cost,
                    'labor_cost': self.labor_cost_per_dish,
                    'total_cost': total_cost,
                    'profit': profit
                })
        
        return pd.DataFrame(sales_records)
    
    def generate_waste_data(self):
        """Generate synthetic waste data based on sales and inventory"""
        waste_records = []
        
        # Get unique dates from sales data
        unique_dates = self.sales_data['date'].unique()
        
        # Industry standard waste percentages
        WASTE_PERCENTAGES = {
            'over_portioned': 0.03,  # 3% over-portioning waste
            'spoiled': 0.02,         # 2% spoilage waste
            'contaminated': 0.01      # 1% contamination waste
        }
        
        for date_obj in unique_dates:
            # Handle both string and Timestamp objects
            if isinstance(date_obj, str):
                date = datetime.strptime(date_obj, '%Y-%m-%d')
                date_str = date_obj
            else:
                date = date_obj
                date_str = date_obj.strftime('%Y-%m-%d')
            
            # Get sales for this date
            daily_sales = self.sales_data[self.sales_data['date'] == date_obj]
            
            # Track ingredients used on this day
            ingredients_used = {}
            
            # Calculate ingredient usage from dishes sold
            for _, sale in daily_sales.iterrows():
                dish_name = sale['dish']
                quantity = sale['quantity']
                recipe = self.recipe_info[dish_name]
                
                for ingredient, amount in recipe['ingredients'].items():
                    if ingredient in ingredients_used:
                        ingredients_used[ingredient] += amount * quantity / 1000  # Convert to kg
                    else:
                        ingredients_used[ingredient] = amount * quantity / 1000  # Convert to kg
            
            # OVER-PORTIONED WASTE: 3% of high-usage ingredients
            for ingredient, usage in ingredients_used.items():
                if usage > 1:  # Only consider significant usage
                    # 20% chance of over-portioning (not every day)
                    if random.random() < 0.2:
                        waste_amount = usage * WASTE_PERCENTAGES['over_portioned']
                        
                        # Cost of waste
                        ingredient_cost = self.ingredient_info.get(ingredient, {}).get('cost_per_kg', 100)
                        waste_cost = waste_amount * ingredient_cost
                        
                        waste_records.append({
                            'date': date_str,
                            'waste_type': 'over_portioned',
                            'ingredient': ingredient,
                            'amount_kg': round(waste_amount, 3),
                            'cost_inr': round(waste_cost, 2)
                        })
            
            # SPOILED WASTE: Check which ingredients might spoil based on shelf life
            for ingredient, info in self.ingredient_info.items():
                shelf_life = info.get('shelf_life_days', 7)
                
                # Lower chance of spoilage for items with longer shelf life
                spoilage_chance = 0.01 + (1 / (shelf_life + 1)) * 0.1
                
                # Higher chance of spoilage for low-usage items
                usage = ingredients_used.get(ingredient, 0)
                if usage < 0.5:  # Low usage
                    spoilage_chance += 0.05
                
                if random.random() < spoilage_chance:
                    # Base waste amount on usage and shelf life
                    base_waste = usage * WASTE_PERCENTAGES['spoiled']
                    waste_amount = min(base_waste, random.uniform(0.1, 1.0))  # Cap at 1kg
                    
                    # Cost of waste
                    ingredient_cost = info.get('cost_per_kg', 100)
                    waste_cost = waste_amount * ingredient_cost
                    
                    waste_records.append({
                        'date': date_str,
                        'waste_type': 'spoiled',
                        'ingredient': ingredient,
                        'amount_kg': round(waste_amount, 3),
                        'cost_inr': round(waste_cost, 2)
                    })
            
            # CONTAMINATED WASTE: Very rare occurrence
            used_ingredients = list(ingredients_used.keys())
            if used_ingredients and random.random() < 0.05:  # 5% chance of contamination per day
                # Select 1-2 random ingredients
                num_contaminated = random.randint(1, min(2, len(used_ingredients)))
                contaminated = random.sample(used_ingredients, num_contaminated)
                
                for ingredient in contaminated:
                    # Usually smaller amounts get contaminated
                    waste_amount = random.uniform(0.1, 0.5)  # 0.1 to 0.5 kg
                    
                    # Cost of waste
                    ingredient_cost = self.ingredient_info.get(ingredient, {}).get('cost_per_kg', 100)
                    waste_cost = waste_amount * ingredient_cost
                    
                    waste_records.append({
                        'date': date_str,
                        'waste_type': 'contaminated',
                        'ingredient': ingredient,
                        'amount_kg': round(waste_amount, 3),
                        'cost_inr': round(waste_cost, 2)
                    })
        
        return pd.DataFrame(waste_records)
    
    def calculate_waste_impact(self):
        """Calculate the financial impact of food waste"""
        # Group waste data by type and date
        waste_by_type = self.waste_data.groupby(['date', 'waste_type']).agg(
            total_cost=('cost_inr', 'sum'),
            total_amount=('amount_kg', 'sum')
        ).reset_index()
        
        # Calculate total waste by date
        waste_by_date = self.waste_data.groupby('date').agg(
            total_waste_cost=('cost_inr', 'sum'),
            total_waste_kg=('amount_kg', 'sum')
        ).reset_index()
        
        # Calculate sales and profit by date
        sales_by_date = self.sales_data.groupby('date').agg(
            total_sales=('selling_price', 'sum'),
            total_profit=('profit', 'sum'),
            dishes_sold=('dish', 'count')
        ).reset_index()
        
        # Merge waste and sales data
        combined_data = pd.merge(waste_by_date, sales_by_date, on='date', how='outer').fillna(0)
        
        # Calculate waste as percentage of sales
        combined_data['waste_pct_of_sales'] = (combined_data['total_waste_cost'] / combined_data['total_sales'] * 100).round(2)
        
        # Calculate adjusted profit (if waste was eliminated)
        combined_data['potential_profit'] = combined_data['total_profit'] + combined_data['total_waste_cost']
        combined_data['profit_loss_pct'] = ((combined_data['potential_profit'] - combined_data['total_profit']) / 
                                          combined_data['potential_profit'] * 100).round(2)
        
        return {
            'waste_by_type': waste_by_type,
            'combined_metrics': combined_data
        }
    
    def analyze_over_portioned_waste(self):
        """Analyze over-portioned waste to optimize future portions"""
        # Filter for over-portioned waste only
        over_portioned = self.waste_data[self.waste_data['waste_type'] == 'over_portioned']
        
        # Group by ingredient
        ingredient_waste = over_portioned.groupby('ingredient').agg(
            total_waste_kg=('amount_kg', 'sum'),
            total_waste_cost=('cost_inr', 'sum'),
            avg_waste_per_incident=('amount_kg', 'mean'),
            waste_incidents=('amount_kg', 'count')
        ).reset_index()
        
        # Add recipe information
        recipe_usage = {}
        for recipe_name, info in self.recipe_info.items():
            for ingredient, amount in info['ingredients'].items():
                if ingredient in recipe_usage:
                    recipe_usage[ingredient].append(recipe_name)
                else:
                    recipe_usage[ingredient] = [recipe_name]
        
        # Add recipe info to the analysis
        ingredient_waste['used_in_recipes'] = ingredient_waste['ingredient'].map(
            lambda x: ', '.join(recipe_usage.get(x, ['Unknown'])))
        
        # Calculate recommended portion adjustment
        ingredient_waste['avg_waste_pct'] = (ingredient_waste['avg_waste_per_incident'] / 
                                           (self.standard_portion_size/1000) * 100).round(2)
        
        ingredient_waste['recommended_portion_adjustment'] = (-ingredient_waste['avg_waste_per_incident'] * 1000 / 2).round(0)
        
        return ingredient_waste
    
    def analyze_spoilage_waste(self):
        """Analyze spoilage waste to improve inventory management"""
        # Filter for spoiled waste only
        spoiled = self.waste_data[self.waste_data['waste_type'] == 'spoiled']
        
        # Group by ingredient
        spoilage_by_ingredient = spoiled.groupby('ingredient').agg(
            total_waste_kg=('amount_kg', 'sum'),
            total_waste_cost=('cost_inr', 'sum'),
            waste_incidents=('amount_kg', 'count')
        ).reset_index()
        
        # Add shelf life information
        spoilage_by_ingredient['shelf_life_days'] = spoilage_by_ingredient['ingredient'].map(
            lambda x: self.ingredient_info.get(x, {}).get('shelf_life_days', 7))
        
        # Add weekly usage information
        spoilage_by_ingredient['weekly_usage_kg'] = spoilage_by_ingredient['ingredient'].map(
            lambda x: self.ingredient_info.get(x, {}).get('weekly_usage_kg', 5))
        
        # Calculate spoilage risk score (higher means more attention needed)
        spoilage_by_ingredient['spoilage_risk_score'] = (
            spoilage_by_ingredient['total_waste_cost'] / 
            (spoilage_by_ingredient['shelf_life_days'] * spoilage_by_ingredient['weekly_usage_kg'])
        ).round(2)
        
        # Calculate optimal order frequency
        spoilage_by_ingredient['optimal_order_frequency_days'] = np.ceil(
            7 * spoilage_by_ingredient['weekly_usage_kg'] / 
            (spoilage_by_ingredient['shelf_life_days'] * spoilage_by_ingredient['weekly_usage_kg'] / 
             (spoilage_by_ingredient['shelf_life_days'] / 2))
        )
        
        return spoilage_by_ingredient
    
    def forecast_ingredient_needs(self, days_to_forecast=7):
        """Forecast ingredient needs based on historical data and waste patterns"""
        # Get recent sales data (last 30 days)
        recent_dates = sorted(self.sales_data['date'].unique())[-30:]
        recent_sales = self.sales_data[self.sales_data['date'].isin(recent_dates)]
        
        # Calculate dish frequency
        dish_frequency = recent_sales.groupby('dish')['quantity'].sum().reset_index()
        dish_frequency['avg_daily_sales'] = (dish_frequency['quantity'] / 30).round(2)
        
        # Initialize forecast dataframe
        ingredient_forecast = {}
        
        # Calculate expected ingredient usage
        for _, row in dish_frequency.iterrows():
            dish = row['dish']
            daily_sales = row['avg_daily_sales']
            
            if dish in self.recipe_info:
                dish_ingredients = self.recipe_info[dish]['ingredients']
                for ingredient, amount in dish_ingredients.items():
                    # Amount is in grams, convert to kg
                    amount_kg = amount / 1000 * daily_sales
                    
                    if ingredient in ingredient_forecast:
                        ingredient_forecast[ingredient] += amount_kg
                    else:
                        ingredient_forecast[ingredient] = amount_kg
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame([
            {'ingredient': ing, 'daily_usage_kg': round(amount, 3), 
             'forecast_usage_kg': round(amount * days_to_forecast, 3)}
            for ing, amount in ingredient_forecast.items()
        ])
        
        # Get current inventory levels
        if hasattr(self, 'inventory_data'):
            inventory_map = dict(zip(self.inventory_data['ingredient'], self.inventory_data['stock_kg']))
            forecast_df['current_stock_kg'] = forecast_df['ingredient'].map(inventory_map).fillna(0)
        else:
            forecast_df['current_stock_kg'] = 0
        
        # Calculate waste adjustments from over-portioning analysis
        over_portioned_analysis = self.analyze_over_portioned_waste()
        
        # Create a mapping of ingredient to recommended adjustment (in kg)
        adjustment_map = dict(zip(
            over_portioned_analysis['ingredient'],
            over_portioned_analysis['recommended_portion_adjustment'] / 1000  # Convert from grams to kg
        ))
        
        # Apply adjustments to forecast
        forecast_df['portion_adjustment_kg'] = forecast_df['ingredient'].map(adjustment_map).fillna(0)
        forecast_df['adjusted_daily_usage_kg'] = (forecast_df['daily_usage_kg'] + 
                                             forecast_df['daily_usage_kg'] * forecast_df['portion_adjustment_kg'] / 
                                             (self.standard_portion_size/1000)).round(3)
        
        forecast_df['adjusted_forecast_usage_kg'] = forecast_df['adjusted_daily_usage_kg'] * days_to_forecast
        
        # Calculate needed purchase
        forecast_df['needed_purchase_kg'] = (forecast_df['adjusted_forecast_usage_kg'] - 
                                       forecast_df['current_stock_kg']).round(3)
        forecast_df['needed_purchase_kg'] = forecast_df['needed_purchase_kg'].clip(lower=0)
        
        # Add cost information
        forecast_df['cost_per_kg'] = forecast_df['ingredient'].map(
            lambda x: self.ingredient_info.get(x, {}).get('cost_per_kg', 100))
        
        forecast_df['purchase_cost_inr'] = (forecast_df['needed_purchase_kg'] * 
                                      forecast_df['cost_per_kg']).round(2)
        
        return forecast_df
    
    def generate_profit_loss_dashboard(self):
        """Generate a comprehensive profit and loss dashboard"""
        # Calculate various metrics
        waste_impact = self.calculate_waste_impact()
        over_portioned_analysis = self.analyze_over_portioned_waste()
        spoilage_analysis = self.analyze_spoilage_waste()
        forecast = self.forecast_ingredient_needs()
        
        # Summarize key metrics
        waste_by_type = self.waste_data.groupby('waste_type').agg(
            total_cost=('cost_inr', 'sum'),
            total_kg=('amount_kg', 'sum'),
            incidents=('waste_type', 'count')
        ).reset_index()
        
        # Expand sales data to account for quantity
        expanded_sales = self.sales_data.copy()
        expanded_sales['total_selling_price'] = expanded_sales['selling_price'] * expanded_sales['quantity']
        expanded_sales['total_profit'] = expanded_sales['profit'] * expanded_sales['quantity']
        
        total_sales = expanded_sales['total_selling_price'].sum()
        total_profit = expanded_sales['total_profit'].sum()
        total_waste_cost = self.waste_data['cost_inr'].sum()
        potential_profit = total_profit + total_waste_cost
        profit_impact_pct = (total_waste_cost / potential_profit * 100).round(2)
        
        dashboard = {
            'summary': {
                'total_sales_inr': round(total_sales, 2),
                'total_profit_inr': round(total_profit, 2),
                'total_waste_cost_inr': round(total_waste_cost, 2),
                'potential_profit_inr': round(potential_profit, 2),
                'profit_impact_pct': profit_impact_pct,
                'total_waste_incidents': len(self.waste_data)
            },
            'waste_by_type': waste_by_type.to_dict('records'),
            'daily_metrics': waste_impact['combined_metrics'].to_dict('records'),
            'over_portioned_analysis': over_portioned_analysis.to_dict('records'),
            'spoilage_analysis': spoilage_analysis.to_dict('records'),
            'ingredient_forecast': forecast.to_dict('records')
        }
        
        return dashboard
    
    def plot_waste_impact(self, output_dir='.'):
        """Create visualizations for waste impact on profit"""
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        waste_impact = self.calculate_waste_impact()
        combined_data = waste_impact['combined_metrics']
        
        # Select only a subset of dates for better visualization
        date_sample = sorted(combined_data['date'].unique())[::30]  # Every 30 days
        plot_data = combined_data[combined_data['date'].isin(date_sample)]
        
        # Save data to CSV instead of plotting
        plot_data.to_csv(os.path.join(output_dir, 'waste_impact_data.csv'), index=False)
        
        # Save waste by type data
        waste_by_type = self.waste_data.groupby('waste_type').agg(
            total_cost=('cost_inr', 'sum')
        ).reset_index()
        waste_by_type.to_csv(os.path.join(output_dir, 'waste_by_type_data.csv'), index=False)
        
        # Save top ingredients data
        top_ingredients = self.waste_data.groupby('ingredient')['cost_inr'].sum().sort_values(ascending=False).head(10)
        top_ingredients.to_csv(os.path.join(output_dir, 'top_wasted_ingredients.csv'))
        
        return True
    
    def generate_time_based_analysis(self):
        """Generate time-based analysis of profit and loss"""
        # Convert date column to datetime
        self.sales_data['date'] = pd.to_datetime(self.sales_data['date'])
        self.waste_data['date'] = pd.to_datetime(self.waste_data['date'])
        
        # Weekly analysis
        weekly_sales = self.sales_data.groupby(pd.Grouper(key='date', freq='W')).agg({
            'selling_price': 'sum',
            'total_cost': 'sum',
            'profit': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        weekly_waste = self.waste_data.groupby(pd.Grouper(key='date', freq='W')).agg({
            'cost_inr': 'sum',
            'amount_kg': 'sum'
        }).reset_index()
        
        weekly_analysis = pd.merge(weekly_sales, weekly_waste, on='date', how='outer').fillna(0)
        weekly_analysis['waste_pct_of_sales'] = (weekly_analysis['cost_inr'] / weekly_analysis['selling_price'] * 100).round(2)
        weekly_analysis['adjusted_profit'] = weekly_analysis['profit'] - weekly_analysis['cost_inr']
        
        # Monthly analysis
        monthly_sales = self.sales_data.groupby(pd.Grouper(key='date', freq='ME')).agg({
            'selling_price': 'sum',
            'total_cost': 'sum',
            'profit': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        monthly_waste = self.waste_data.groupby(pd.Grouper(key='date', freq='ME')).agg({
            'cost_inr': 'sum',
            'amount_kg': 'sum'
        }).reset_index()
        
        monthly_analysis = pd.merge(monthly_sales, monthly_waste, on='date', how='outer').fillna(0)
        monthly_analysis['waste_pct_of_sales'] = (monthly_analysis['cost_inr'] / monthly_analysis['selling_price'] * 100).round(2)
        monthly_analysis['adjusted_profit'] = monthly_analysis['profit'] - monthly_analysis['cost_inr']
        
        # Quarterly analysis
        quarterly_sales = self.sales_data.groupby(pd.Grouper(key='date', freq='QE')).agg({
            'selling_price': 'sum',
            'total_cost': 'sum',
            'profit': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        quarterly_waste = self.waste_data.groupby(pd.Grouper(key='date', freq='QE')).agg({
            'cost_inr': 'sum',
            'amount_kg': 'sum'
        }).reset_index()
        
        quarterly_analysis = pd.merge(quarterly_sales, quarterly_waste, on='date', how='outer').fillna(0)
        quarterly_analysis['waste_pct_of_sales'] = (quarterly_analysis['cost_inr'] / quarterly_analysis['selling_price'] * 100).round(2)
        quarterly_analysis['adjusted_profit'] = quarterly_analysis['profit'] - quarterly_analysis['cost_inr']
        
        return {
            'weekly': weekly_analysis,
            'monthly': monthly_analysis,
            'quarterly': quarterly_analysis
        }

    def plot_time_based_analysis(self, output_dir='.'):
        """Create visualizations for time-based analysis"""
        time_analysis = self.generate_time_based_analysis()
        
        # Save weekly analysis data
        time_analysis['weekly'].to_csv(os.path.join(output_dir, 'weekly_analysis_data.csv'), index=False)
        
        # Save monthly analysis data
        time_analysis['monthly'].to_csv(os.path.join(output_dir, 'monthly_analysis_data.csv'), index=False)
        
        # Save quarterly analysis data
        time_analysis['quarterly'].to_csv(os.path.join(output_dir, 'quarterly_analysis_data.csv'), index=False)
        
        return True

    def generate_waste_analysis_csvs(self, output_dir='.'):
        """Generate detailed CSV files for waste analysis and optimization"""
        # 1. Time-based waste analysis (weekly, monthly, quarterly)
        time_analysis = self.generate_time_based_analysis()
        
        # Weekly waste analysis
        weekly_waste = time_analysis['weekly'].copy()
        weekly_waste['potential_profit'] = weekly_waste['profit'] + weekly_waste['cost_inr']
        weekly_waste['profit_loss'] = weekly_waste['cost_inr']
        weekly_waste['date'] = weekly_waste['date'].dt.strftime('%Y-%m-%d')
        weekly_waste = weekly_waste[['date', 'cost_inr', 'profit', 'potential_profit', 'profit_loss']]
        weekly_waste.columns = ['Week', 'Waste_Cost', 'Actual_Profit', 'Potential_Profit', 'Profit_Loss']
        weekly_waste.to_csv(os.path.join(output_dir, 'weekly_waste_analysis.csv'), index=False)
        
        # Monthly waste analysis
        monthly_waste = time_analysis['monthly'].copy()
        monthly_waste['potential_profit'] = monthly_waste['profit'] + monthly_waste['cost_inr']
        monthly_waste['profit_loss'] = monthly_waste['cost_inr']
        monthly_waste['date'] = monthly_waste['date'].dt.strftime('%Y-%m')
        monthly_waste = monthly_waste[['date', 'cost_inr', 'profit', 'potential_profit', 'profit_loss']]
        monthly_waste.columns = ['Month', 'Waste_Cost', 'Actual_Profit', 'Potential_Profit', 'Profit_Loss']
        monthly_waste.to_csv(os.path.join(output_dir, 'monthly_waste_analysis.csv'), index=False)
        
        # Quarterly waste analysis
        quarterly_waste = time_analysis['quarterly'].copy()
        quarterly_waste['potential_profit'] = quarterly_waste['profit'] + quarterly_waste['cost_inr']
        quarterly_waste['profit_loss'] = quarterly_waste['cost_inr']
        quarterly_waste['date'] = quarterly_waste['date'].dt.to_period('Q').astype(str)
        quarterly_waste = quarterly_waste[['date', 'cost_inr', 'profit', 'potential_profit', 'profit_loss']]
        quarterly_waste.columns = ['Quarter', 'Waste_Cost', 'Actual_Profit', 'Potential_Profit', 'Profit_Loss']
        quarterly_waste.to_csv(os.path.join(output_dir, 'quarterly_waste_analysis.csv'), index=False)
        
        # 2. Waste type analysis
        # Over-portioning analysis
        over_portioned = self.waste_data[self.waste_data['waste_type'] == 'over_portioned']
        over_portioned_analysis = over_portioned.groupby('ingredient').agg({
            'amount_kg': 'sum',
            'cost_inr': 'sum',
            'date': 'count'
        }).reset_index()
        over_portioned_analysis.columns = ['Ingredient', 'Total_Waste_Kg', 'Total_Cost', 'Incidents']
        over_portioned_analysis['Avg_Waste_Per_Incident'] = over_portioned_analysis['Total_Waste_Kg'] / over_portioned_analysis['Incidents']
        over_portioned_analysis['Current_Portion_Size'] = self.standard_portion_size / 1000  # Convert to kg
        over_portioned_analysis['Recommended_Portion_Size'] = over_portioned_analysis['Current_Portion_Size'] * 0.97  # 3% reduction
        over_portioned_analysis.to_csv(os.path.join(output_dir, 'over_portioning_analysis.csv'), index=False)
        
        # Spoilage analysis
        spoiled = self.waste_data[self.waste_data['waste_type'] == 'spoiled']
        spoilage_analysis = spoiled.groupby('ingredient').agg({
            'amount_kg': 'sum',
            'cost_inr': 'sum',
            'date': 'count'
        }).reset_index()
        spoilage_analysis.columns = ['Ingredient', 'Total_Waste_Kg', 'Total_Cost', 'Incidents']
        
        # Add shelf life and current stock information
        spoilage_analysis['Shelf_Life_Days'] = spoilage_analysis['Ingredient'].map(
            lambda x: self.ingredient_info.get(x, {}).get('shelf_life_days', 7))
        spoilage_analysis['Current_Stock_Kg'] = spoilage_analysis['Ingredient'].map(
            lambda x: self.inventory_data[self.inventory_data['ingredient'] == x]['stock_kg'].sum())
        spoilage_analysis['Weekly_Usage_Kg'] = spoilage_analysis['Ingredient'].map(
            lambda x: self.ingredient_info.get(x, {}).get('weekly_usage_kg', 3))
        
        # Calculate optimal order quantity
        spoilage_analysis['Optimal_Order_Kg'] = (spoilage_analysis['Weekly_Usage_Kg'] * 
                                               (spoilage_analysis['Shelf_Life_Days'] / 7) * 0.8)  # 80% of shelf life
        spoilage_analysis.to_csv(os.path.join(output_dir, 'spoilage_analysis.csv'), index=False)
        
        # 3. Waste type summary
        waste_type_summary = self.waste_data.groupby('waste_type').agg({
            'amount_kg': 'sum',
            'cost_inr': 'sum',
            'date': 'count'
        }).reset_index()
        waste_type_summary.columns = ['Waste_Type', 'Total_Waste_Kg', 'Total_Cost', 'Incidents']
        waste_type_summary['Percentage_of_Total_Waste'] = (waste_type_summary['Total_Waste_Kg'] / 
                                                         waste_type_summary['Total_Waste_Kg'].sum() * 100).round(2)
        waste_type_summary.to_csv(os.path.join(output_dir, 'waste_type_summary.csv'), index=False)
        
        # 4. Optimization recommendations
        optimization_recommendations = []
        
        # Portion size recommendations
        for _, row in over_portioned_analysis.iterrows():
            optimization_recommendations.append({
                'Type': 'Portion_Size',
                'Item': row['Ingredient'],
                'Current_Value': row['Current_Portion_Size'],
                'Recommended_Value': row['Recommended_Portion_Size'],
                'Potential_Savings': row['Total_Cost'] * 0.03,  # 3% of waste cost
                'Implementation_Complexity': 'Low'
            })
        
        # Inventory management recommendations
        for _, row in spoilage_analysis.iterrows():
            optimization_recommendations.append({
                'Type': 'Inventory_Management',
                'Item': row['Ingredient'],
                'Current_Value': row['Current_Stock_Kg'],
                'Recommended_Value': row['Optimal_Order_Kg'],
                'Potential_Savings': row['Total_Cost'] * 0.5,  # 50% of waste cost
                'Implementation_Complexity': 'Medium'
            })
        
        pd.DataFrame(optimization_recommendations).to_csv(
            os.path.join(output_dir, 'optimization_recommendations.csv'), index=False)
        
        return True

    def save_data_to_csv(self, output_dir='.'):
        """Save generated data to CSV files"""
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # Clear existing files in the directory
            for file in os.listdir(output_dir):
                if file.endswith('.csv') or file.endswith('.png'):
                    os.remove(os.path.join(output_dir, file))
            
        # Save the raw data
        self.waste_data.to_csv(os.path.join(output_dir, 'waste_data.csv'), index=False)
        self.sales_data.to_csv(os.path.join(output_dir, 'sales_data.csv'), index=False)
        
        # Generate and save analysis CSV files
        self.generate_waste_analysis_csvs(output_dir)
        
        # Save dashboard metrics
        dashboard = self.generate_profit_loss_dashboard()
        pd.DataFrame(dashboard['daily_metrics']).to_csv(os.path.join(output_dir, 'daily_metrics.csv'), index=False)
        
        # Save summary as a text file
        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            f.write("=== RESTAURANT WASTE PROFIT & LOSS ANALYSIS ===\n")
            f.write(f"Total Sales: ₹{dashboard['summary']['total_sales_inr']}\n")
            f.write(f"Total Profit: ₹{dashboard['summary']['total_profit_inr']}\n")
            f.write(f"Total Waste Cost: ₹{dashboard['summary']['total_waste_cost_inr']}\n")
            f.write(f"Potential Profit (if no waste): ₹{dashboard['summary']['potential_profit_inr']}\n")
            f.write(f"Profit Impact: {dashboard['summary']['profit_impact_pct']}%\n")
            f.write(f"Total Waste Incidents: {dashboard['summary']['total_waste_incidents']}\n")
        
        return True

    def generate_chart_data_csvs(self, output_dir='.'):
        """Generate CSV files with data for creating interactive charts"""
        time_analysis = self.generate_time_based_analysis()
        
        # 1. Profit and Waste Analysis Data
        for period in ['weekly', 'monthly', 'quarterly']:
            data = time_analysis[period].copy()
            data['date'] = data['date'].dt.strftime('%Y-%m-%d')
            data = data[['date', 'profit', 'cost_inr', 'adjusted_profit']]
            data.columns = ['Date', 'Actual_Profit', 'Waste_Cost', 'Potential_Profit']
            data.to_csv(os.path.join(output_dir, f'profit_waste_{period}.csv'), index=False)
        
        # 2. Waste Type Distribution Data
        for period in ['weekly', 'monthly', 'quarterly']:
            data = self.waste_data.copy()
            data['date'] = pd.to_datetime(data['date'])
            if period == 'weekly':
                data = data.groupby([pd.Grouper(key='date', freq='W'), 'waste_type']).sum().reset_index()
            elif period == 'monthly':
                data = data.groupby([pd.Grouper(key='date', freq='ME'), 'waste_type']).sum().reset_index()
            else:
                data = data.groupby([pd.Grouper(key='date', freq='QE'), 'waste_type']).sum().reset_index()
            
            data['date'] = data['date'].dt.strftime('%Y-%m-%d')
            data = data[['date', 'waste_type', 'cost_inr', 'amount_kg']]
            data.columns = ['Date', 'Waste_Type', 'Cost_INR', 'Amount_Kg']
            data.to_csv(os.path.join(output_dir, f'waste_type_distribution_{period}.csv'), index=False)
        
        # 3. Detailed Waste Analysis Data
        for period in ['weekly', 'monthly', 'quarterly']:
            # For over-portioned waste, analyze by recipe
            over_portioned_data = self.waste_data[self.waste_data['waste_type'] == 'over_portioned'].copy()
            
            if not over_portioned_data.empty:
                over_portioned_data['date'] = pd.to_datetime(over_portioned_data['date'])
                
                # Get unique recipes from sales data
                recipes = self.sales_data['dish'].unique()
                
                # Create a mapping of ingredients to recipes
                recipe_mapping = {}
                for recipe in recipes:
                    recipe_ingredients = self.recipe_data[self.recipe_data['recipe_name'] == recipe]['ingredient'].unique()
                    for ingredient in recipe_ingredients:
                        recipe_mapping[ingredient] = recipe
                
                # Map ingredients to recipes
                over_portioned_data['recipe_name'] = over_portioned_data['ingredient'].map(recipe_mapping)
                
                # Group by date and recipe
                if period == 'weekly':
                    over_portioned_data = over_portioned_data.groupby([pd.Grouper(key='date', freq='W'), 'recipe_name']).agg({
                        'amount_kg': 'sum',
                        'cost_inr': 'sum'
                    }).reset_index()
                elif period == 'monthly':
                    over_portioned_data = over_portioned_data.groupby([pd.Grouper(key='date', freq='ME'), 'recipe_name']).agg({
                        'amount_kg': 'sum',
                        'cost_inr': 'sum'
                    }).reset_index()
                else:
                    over_portioned_data = over_portioned_data.groupby([pd.Grouper(key='date', freq='QE'), 'recipe_name']).agg({
                        'amount_kg': 'sum',
                        'cost_inr': 'sum'
                    }).reset_index()
                
                # Calculate recommended portion size (3% reduction from current)
                over_portioned_data['current_portion_size'] = self.standard_portion_size
                over_portioned_data['recommended_portion_size'] = self.standard_portion_size * 0.97
                over_portioned_data['date'] = over_portioned_data['date'].dt.strftime('%Y-%m-%d')
                
                over_portioned_data = over_portioned_data[['date', 'recipe_name', 'amount_kg', 'cost_inr', 
                                                         'current_portion_size', 'recommended_portion_size']]
                over_portioned_data.columns = ['Date', 'Dish', 'Waste_Amount_Kg', 'Waste_Cost_INR', 
                                             'Current_Portion_Size_g', 'Recommended_Portion_Size_g']
                
                # Remove any rows where Dish is NaN
                over_portioned_data = over_portioned_data.dropna(subset=['Dish'])
                
                if not over_portioned_data.empty:
                    over_portioned_data.to_csv(os.path.join(output_dir, f'over_portioned_analysis_{period}.csv'), index=False)
            
            # For spoiled and contaminated waste, keep ingredient analysis
            other_waste_data = self.waste_data[self.waste_data['waste_type'].isin(['spoiled', 'contaminated'])].copy()
            other_waste_data['date'] = pd.to_datetime(other_waste_data['date'])
            
            if period == 'weekly':
                other_waste_data = other_waste_data.groupby([pd.Grouper(key='date', freq='W'), 'waste_type', 'ingredient']).sum().reset_index()
            elif period == 'monthly':
                other_waste_data = other_waste_data.groupby([pd.Grouper(key='date', freq='ME'), 'waste_type', 'ingredient']).sum().reset_index()
            else:
                other_waste_data = other_waste_data.groupby([pd.Grouper(key='date', freq='QE'), 'waste_type', 'ingredient']).sum().reset_index()
            
            other_waste_data['date'] = other_waste_data['date'].dt.strftime('%Y-%m-%d')
            other_waste_data = other_waste_data[['date', 'waste_type', 'ingredient', 'amount_kg', 'cost_inr']]
            other_waste_data.columns = ['Date', 'Waste_Type', 'Ingredient', 'Amount_Kg', 'Cost_INR']
            other_waste_data.to_csv(os.path.join(output_dir, f'other_waste_analysis_{period}.csv'), index=False)
        
        return True


