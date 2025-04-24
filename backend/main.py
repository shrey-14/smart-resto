import os
import argparse
import yaml
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def run_demand_waste_module():
    """Run the demand waste prediction module"""
    try:
        print("\n=== Running Demand Waste Predictor Module ===")
        # Import here to avoid loading unnecessary dependencies
        from src.demand_waste.data_preprocessor import load_inventory_data
        from src.demand_waste.waste_predictor import predict_waste

        input_path = "data/raw/inventory_data.csv"
        print(f"Loading data from: {input_path}")
        
        df = load_inventory_data(input_path)
        predictions = predict_waste(df)
        
        print("\nPrediction Results:")
        print(predictions.to_string(index=False, justify='left', col_space=10))
        
        if not predictions.empty:
            output_path = "data/processed/waste_prediction_results.csv"
            predictions.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
        else:
            print("Warning: No predictions generated")
            
    except Exception as e:
        print(f"Error in Demand Waste Predictor: {str(e)}")
        raise

def run_smart_kitchen_module():
    """Run the smart kitchen sales forecasting module"""
    try:
        print("\n=== Running Smart Kitchen Sales Module ===")
        # Import here to avoid loading unnecessary dependencies
        from src.smart_kitchen.data_preprocessor import DataPreprocessor
        from src.smart_kitchen.sales_forecaster_prophet import SalesForecaster

        # Load config
        config_path = "config/config.yaml"
        print(f"Loading config from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Preprocess data
        processed_data = pd.read_csv(config['data']['raw_path'])
        processed_data['date'] = pd.to_datetime(processed_data['date'])

        # Train/test split
        train_cutoff = pd.to_datetime('2024-12-01')
        train_data = processed_data[processed_data['date'] <= train_cutoff]
        test_data = processed_data[processed_data['date'] > train_cutoff]
        print(test_data)
        # Train and evaluate model
        forecaster = SalesForecaster(config_path)
        # Check if models exist
        model_files = [f for f in os.listdir(config['model']['path']) if f.endswith('_model.json')]
        if len(model_files) < len(processed_data['item'].unique()):
            print("Training models...")
            forecaster.train(train_data)
        else:
            print("Loading existing models...")
            forecaster.load_models()
        accuracy = forecaster.evaluate(test_data)
        
        print("\nAccuracy Metrics (Test Set):")
        print(accuracy)

        # Generate future predictions
        future_data = generate_future_data(processed_data, config['prediction']['days_ahead'])
        future_preds = forecaster.predict(future_data)
        print("\nFuture Predictions:")
        print(future_preds)
        
        # Save predictions to CSV
        future_preds.to_csv(config['data']['output_path'], index=False)
        print(f"\nPredictions saved to {config['data']['output_path']}")
        
    except Exception as e:
        print(f"Error in Smart Kitchen Sales: {str(e)}")
        raise

def generate_future_data(historical_data, days_ahead):
    """Generate future data for sales forecasting"""
    # Use current date as starting point instead of last historical date
    current_date = datetime.now()
    future_dates = pd.date_range(start=current_date, periods=days_ahead)
    future_data = []
    
    for date in future_dates:
        for item in historical_data['item'].unique():
            item_data = historical_data[historical_data['item'] == item].tail(7)
            last_row = item_data.iloc[-1]
            lag_7 = item_data['quantity'].iloc[0] if len(item_data) >= 7 else 0
            future_data.append({
                'date': date,
                'item': item,
                'day_of_week': date.dayofweek,
                'month': date.month,
                'is_weekend': 1 if date.dayofweek in [5, 6] else 0,
                'lag_1': last_row['quantity'],
                'lag_7': lag_7
            })
    return pd.DataFrame(future_data)

def run_recipe_recommender():
    """Run the recipe recommender module"""
    try:
        print("\n=== Running Recipe Recommender Module ===")
        # Import here to avoid loading unnecessary dependencies
        from src.menu_optimization.recipe_recommender import RecipeRecommender

        # Load config
        config_path = "config/config.yaml"
        print(f"Loading config from: {config_path}")
        
        # Initialize recommender
        recommender = RecipeRecommender(config_path)
        
        # Get current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get recipe recommendation
        recommended_recipe = recommender.suggest_daily_special(current_date)
        
        print("\nRecipe Recommendation:")
        print(f"Recommended Daily Special: {recommended_recipe}")
        
    except Exception as e:
        print(f"Error in Recipe Recommender: {str(e)}")
        raise

def run_recipe_generator():
    """Run the recipe generator module"""
    try:
        print("\n=== Running Recipe Generator Module ===")
        # Import here to avoid loading unnecessary dependencies
        from src.menu_optimization.recipe_generator import RecipeGenerator

        # Load config
        config_path = "config/config.yaml"
        print(f"Loading config from: {config_path}")
        
        # Initialize generator
        generator = RecipeGenerator(config_path)
        
        # Generate recipes
        recipes = generator.generate_recipes()
        
        # print("\nGenerated Recipes:")
        # print(recipes)
        
    except Exception as e:
        print(f"Error in Recipe Generator: {str(e)}")
        raise

def run_cost_optimizer():
    """Run the cost optimizer module"""
    try:
        print("\n=== Running Cost Optimizer Module ===")
        # Import here to avoid loading unnecessary dependencies
        from src.menu_optimization.cost_optimizer import CostOptimizer

        # Load config
        config_path = "config/config.yaml"
        print(f"Loading config from: {config_path}")
        
        # Initialize optimizer
        optimizer = CostOptimizer(config_path)
        
        # Optimize costs
        optimized_costs = optimizer.optimize_costs()
        
        # print("\nCost Optimization Results:")
        # print(optimized_costs)
        
    except Exception as e:
        print(f"Error in Cost Optimizer: {str(e)}")
        raise

def run_food_spoilage_detection(image_path=None):
    """Run the food spoilage detection module"""
    try:
        print("\n=== Running Food Spoilage Detection Module ===")
        # Import here to avoid loading unnecessary dependencies
        from src.food_spoilage_detection.food_spoilage_detection import FoodSpoilageDetector

        # Load config
        config_path = "config/config.yaml"
        print(f"Loading config from: {config_path}")
        
        # Initialize detector
        detector = FoodSpoilageDetector(config_path)
        
        # If no image path is provided, use the first sample image
        if image_path is None:
            sample_images = detector.get_sample_images()
            print(sample_images)
            if sample_images:
                image_path = os.path.join(detector.config['data']['raw_spoilage_image_path'], sample_images[2])
                print(f"Using sample image: {image_path}")
            else:
                print("Error: No sample images found")
                return
        
        # Detect spoilage
        result = detector.detect_spoilage(image_path)
        
        # print("\nSpoilage Detection Result:")
        # print(result)
        
    except Exception as e:
        print(f"Error in Food Spoilage Detection: {str(e)}")
        raise

def run_waste_classification(image_path=None):
    """Run the waste classification module"""
    try:
        print("\n=== Running Waste Classification Module ===")
        # Import here to avoid loading unnecessary dependencies
        from src.vision_analyis.food_waste_classification import FoodWasteClassifier

        # Load config
        config_path = "config/config.yaml"
        print(f"Loading config from: {config_path}")
        
        # Initialize classifier
        classifier = FoodWasteClassifier(config_path)
        
        # If no image path is provided, use the first sample image
        if image_path is None:
            sample_images = classifier.config['data']['sample_waste_images']
            if sample_images:
                image_path = os.path.join(classifier.config['data']['raw_waste_image_path'], sample_images[6])
                print(f"Using sample image: {image_path}")
            else:
                print("Error: No sample images found")
                return
        
        # Classify waste
        result = classifier.detect_food_waste(image_path)
        
        print("\nWaste Classification Result:")
        print(result)
        
    except Exception as e:
        print(f"Error in Waste Classification: {str(e)}")
        raise

def run_waste_heatmap(image_path=None):
    """Run the waste heatmap generation module"""
    try:
        print("\n=== Running Waste Heatmap Generation Module ===")
        # Import here to avoid loading unnecessary dependencies
        from src.vision_analyis.waste_heatmap import WasteHeatmapGenerator

        # Load config
        config_path = "config/config.yaml"
        print(f"Loading config from: {config_path}")
        
        # Initialize generator
        generator = WasteHeatmapGenerator(config_path)
        
        # If no image path is provided, use the first sample image
        if image_path is None:
            sample_images = generator.config['data']['sample_waste_heatmap_images']
            if sample_images:
                image_path = os.path.join(generator.config['data']['raw_waste_heatmap_path'], sample_images[2])
                print(f"Using sample image: {image_path}")
            else:
                print("Error: No sample images found")
                return
        
        # Generate heatmap
        heatmap_path, detections_path = generator.create_waste_heatmap(image_path)
        
        print("\nHeatmap Generation Results:")
        print(f"Heatmap saved to: {heatmap_path}")
        print(f"Detections saved to: {detections_path}")
        
    except Exception as e:
        print(f"Error in Waste Heatmap Generation: {str(e)}")
        raise

def run_inventory_tracking():
    """Run the inventory tracking module"""
    try:
        print("\n=== Running Inventory Tracking Module ===")
        # Import here to avoid loading unnecessary dependencies
        from src.inventory_tracking.inventory_tracking import InventoryTracker

        # Load config
        config_path = "config/config.yaml"
        print(f"Loading config from: {config_path}")
        
        # Initialize tracker
        tracker = InventoryTracker(config_path)
        
        # Run inventory detection
        results = tracker.detect_inventory()
        
        print("\nInventory Detection Results:")
        print(f"Results saved to: {tracker.csv_output_path}")
        print(f"Annotated images saved to: {tracker.annotated_image_path}")
        
    except Exception as e:
        print(f"Error in Inventory Tracking: {str(e)}")
        raise

def run_stock_detection():
    """Run the stock detection module"""
    try:
        print("\n=== Running Stock Detection Module ===")
        # Import here to avoid loading unnecessary dependencies
        from src.inventory_tracking.stock_detection import StockDetector

        # Load config
        config_path = "config/config.yaml"
        print(f"Loading config from: {config_path}")
        # Initialize detector
        detector = StockDetector(config_path)
        
        # Run stock detection
        results = detector.detect_stock()
        
        print("\nStock Detection Results:")
        print(f"Output video saved to: {detector.output_video_path}")
        print("\nDetected items:")
        for item, count in results.items():
            print(f"{item}: {count}")
        
    except Exception as e:
        print(f"Error in Stock Detection: {str(e)}")
        raise

def run_dashboard():
    """Run the dashboard module"""
    try:
        print("\n=== Running Dashboard Module ===")
        # Import here to avoid loading unnecessary dependencies
        from src.vision_analyis.PlDashboard import RestaurantWasteTracker
        
        # Load config
        config_path = "config/config.yaml"
        print(f"Loading config from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        output_dashboard_path = config['data']['output_dashboard_path']

        # Initialize dashboard
        tracker = RestaurantWasteTracker(config_path)
        
        # Generate and save all reports
        tracker.save_data_to_csv(output_dashboard_path)
        tracker.generate_chart_data_csvs(output_dashboard_path)
        
        # Generate summary data
        dashboard = tracker.generate_profit_loss_dashboard()
        time_analysis = tracker.generate_time_based_analysis()
        
        return {
            'dashboard': dashboard,
            'time_analysis': time_analysis
        }
        
    except Exception as e:
        print(f"Error in Dashboard: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description='Run various kitchen management modules',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('module', choices=[
        'demand', 'sales', 'recipe', 'recipe_gen',
        'cost_opt', 'spoilage', 'inventory', 'detect_stock', 'waste_class', 'waste_heatmap', 'dashboard'
    ], help='Module to run')
    parser.add_argument('--image-path', help='Path to image file (for spoilage, waste classification, or heatmap)')
    
    args = parser.parse_args()
    
    if args.module == 'demand':
        run_demand_waste_module()
    elif args.module == 'sales':
        run_smart_kitchen_module()
    elif args.module == 'recipe':
        run_recipe_recommender()
    elif args.module == 'recipe_gen':
        run_recipe_generator()
    elif args.module == 'cost_opt':
        run_cost_optimizer()
    elif args.module == 'spoilage':
        run_food_spoilage_detection(args.image_path)
    elif args.module == 'inventory':
        run_inventory_tracking()
    elif args.module == 'detect_stock':
        run_stock_detection()
    elif args.module == 'waste_class':
        run_waste_classification(args.image_path)
    elif args.module == 'waste_heatmap':
        run_waste_heatmap(args.image_path)
    elif args.module == 'dashboard':
        run_dashboard()
    else:
        print("Please specify a module to run (--module)")

if __name__ == '__main__':
    main()