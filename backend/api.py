import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yaml
import json
from pathlib import Path
import tempfile
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uvicorn
import cv2
import time
import asyncio
import traceback

from src.demand_waste.data_preprocessor import load_inventory_data
from src.demand_waste.waste_predictor import predict_waste
from src.smart_kitchen.data_preprocessor import DataPreprocessor
from src.smart_kitchen.sales_forecaster_prophet import SalesForecaster
from src.menu_optimization.recipe_recommender import RecipeRecommender
from src.menu_optimization.recipe_generator import RecipeGenerator
from src.menu_optimization.cost_optimizer import CostOptimizer
from src.food_spoilage_detection.food_spoilage_detection import FoodSpoilageDetector
from src.inventory_tracking.inventory_tracking import InventoryTracker
from src.inventory_tracking.stock_detection import StockDetector
from src.vision_analyis.food_waste_classification import FoodWasteClassifier
from src.vision_analyis.waste_heatmap import WasteHeatmapGenerator
from src.vision_analyis.PlDashboard import RestaurantWasteTracker

def delete_file(file_path: str) -> None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file {file_path}: {str(e)}")

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

app = FastAPI(
    title="Kitchen Management API",
    description="API for kitchen management operations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:8080", "http://127.0.0.1:8080", "http://192.168.1.26:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

config_path = "config/config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

class DemandWasteRequest(BaseModel):
    input_path: Optional[str] = "data/raw/inventory_data.csv"

class SalesForecastRequest(BaseModel):
    days_ahead: Optional[int] = 7

class RecipeRequest(BaseModel):
    current_date: Optional[str] = None

class SpoilageDetectionRequest(BaseModel):
    image_path: Optional[str] = None

class InventoryDetectionRequest(BaseModel):
    image_path: Optional[str] = None

class WasteClassificationRequest(BaseModel):
    image_path: Optional[str] = None

class WasteHeatmapRequest(BaseModel):
    image_path: Optional[str] = None

def get_temp_file_path(extension: str) -> str:
    return str(TEMP_DIR / f"{uuid.uuid4()}.{extension}")

task_tracker = {}

@app.get("/")
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kitchen Management API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #333;
            }
            .card {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .button {
                display: inline-block;
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>Kitchen Management API Server</h1>
        
        <div class="card">
            <h2>Main Application</h2>
            <p>Access the main frontend application.</p>
            <a href="http://localhost:8081" class="button">Go to Main Application</a>
        </div>
        
        <div class="card">
            <h2>API Tester</h2>
            <p>Test the API endpoints directly in your browser.</p>
            <a href="/static/api_tester.html" class="button">Go to API Tester</a>
        </div>
        
        <div class="card">
            <h2>API Documentation</h2>
            <p>View the auto-generated FastAPI documentation.</p>
            <a href="/docs" class="button">View API Documentation</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/api/test")
async def test_api():
    return {
        "status": "success",
        "message": "API is working correctly!",
        "endpoints": [
            {"path": "/api/demand-waste-prediction", "method": "GET/POST"},
            {"path": "/api/sales-forecasting", "method": "GET"},
            {"path": "/api/recipe-recommendation", "method": "GET"},
            {"path": "/api/recipe-generation", "method": "GET"},
            {"path": "/api/cost-optimization", "method": "GET"},
            {"path": "/api/spoilage-detection", "method": "GET/POST"},
            {"path": "/api/waste-classification", "method": "GET/POST"},
            {"path": "/api/inventory-tracking", "method": "GET/POST"},
            {"path": "/api/stock-detection", "method": "GET/POST"},
            {"path": "/api/waste-heatmap", "method": "GET/POST"},
            {"path": "/api/dashboard", "method": "GET"}
        ]
    }

@app.post("/api/demand-waste-prediction")
@app.get("/api/demand-waste-prediction")
async def run_demand_waste():
    try:
        print("\n=== Running Demand Waste Predictor Module ===")
        
        input_path = config.get('data', {}).get('waste_inventory_path', "data/raw/inventory_data.csv")
        print(f"Loading data from: {input_path}")
        
        df = load_inventory_data(input_path)
        predictions = predict_waste(df)
        
        result = predictions.to_dict(orient='records')
        
        return JSONResponse(content={
            "status": "success",
            "predictions": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Demand Waste Predictor: {str(e)}")

@app.get("/api/sales-forecasting")
async def run_sales_forecast():
    try:
        print("\n=== Running Smart Kitchen Sales Module ===")
        
        print(f"Loading config from: {config_path}")
        
        processed_data = pd.read_csv(config['data']['raw_path'])
        processed_data['date'] = pd.to_datetime(processed_data['date'])

        train_cutoff = pd.to_datetime('2024-12-01')
        train_data = processed_data[processed_data['date'] <= train_cutoff]
        test_data = processed_data[processed_data['date'] > train_cutoff]
        
        forecaster = SalesForecaster(config_path)
        model_files = [f for f in os.listdir(config['model']['path']) if f.endswith('_model.json')]
        if len(model_files) < len(processed_data['item'].unique()):
            print("Training models...")
            forecaster.train(train_data)
        else:
            print("Loading existing models...")
            forecaster.load_models()
        accuracy = forecaster.evaluate(test_data)
        
        future_data = generate_future_data(processed_data, 7)
        future_preds = forecaster.predict(future_data)
        
        accuracy_dict = accuracy.to_dict()
        future_preds_dict = future_preds.to_dict(orient='records')
        
        return JSONResponse(content={
            "status": "success",
            "accuracy": accuracy_dict,
            "future_predictions": future_preds_dict
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Smart Kitchen Sales: {str(e)}")

@app.post("/api/sales-forecasting")
async def upload_sales_data(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None)
):
    try:
        print("\n=== Processing Uploaded Sales Data ===")
        
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ['csv', 'xls', 'xlsx']:
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload CSV or Excel files.")
        
        temp_file_path = get_temp_file_path(file_extension)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        background_tasks.add_task(os.remove, temp_file_path)
        
        if file_extension == 'csv':
            processed_data = pd.read_csv(temp_file_path)
        else:
            processed_data = pd.read_excel(temp_file_path)
        
        required_columns = ['date', 'item', 'quantity']
        missing_columns = [col for col in required_columns if col not in processed_data.columns]
        if missing_columns:
            raise HTTPException(status_code=400, 
                                detail=f"Missing required columns: {', '.join(missing_columns)}. File must include date, item, and quantity columns.")
        
        processed_data['date'] = pd.to_datetime(processed_data['date'])
        
        train_cutoff = processed_data['date'].max() - timedelta(days=7)
        train_data = processed_data[processed_data['date'] <= train_cutoff]
        test_data = processed_data[processed_data['date'] > train_cutoff]
        
        if len(test_data) == 0:
            train_size = int(len(processed_data) * 0.9)
            train_data = processed_data.iloc[:train_size]
            test_data = processed_data.iloc[train_size:]
        
        forecaster = SalesForecaster(config_path)
        print("Training models on uploaded data...")
        forecaster.train(train_data)
        accuracy = forecaster.evaluate(test_data)
        
        future_data = generate_future_data(processed_data, 7)
        future_preds = forecaster.predict(future_data)
        
        accuracy_dict = accuracy.to_dict()
        future_preds_dict = future_preds.to_dict(orient='records')
        
        return JSONResponse(content={
            "status": "success",
            "message": "Sales data processed successfully",
            "accuracy": accuracy_dict,
            "future_predictions": future_preds_dict
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing sales data: {str(e)}")

def generate_future_data(historical_data, days_ahead):
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

@app.get("/api/recipe-recommendation")
async def run_recipe_recommender():
    try:
        print("\n=== Running Recipe Recommender Module ===")
        
        recommender = RecipeRecommender(config_path)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        recommended_recipe = recommender.suggest_daily_special(current_date)
        
        if recommended_recipe["recipe_name"] != "No suitable special found":
            recipe_name = recommended_recipe["recipe_name"]
            recipes_df = pd.read_csv(recommender.config['data']['recipe_path'])
            recipe_data = recipes_df[recipes_df['recipe_name'] == recipe_name]
            
            if not recipe_data.empty:
                ingredients_list = []
                for _, row in recipe_data.iterrows():
                    ingredient_text = f"{row['quantity']} {row['unit']} {row['ingredient']}"
                    ingredients_list.append(ingredient_text)
                
                recommended_recipe["ingredients"] = ingredients_list
        
        return JSONResponse(content={
            "status": "success",
            "current_date": current_date,
            "recommended_recipe": recommended_recipe
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Recipe Recommender: {str(e)}")

@app.get("/api/recipe-generation")
async def run_recipe_generator():
    try:
        print("\n=== Running Recipe Generator Module ===")
        
        generator = RecipeGenerator(config_path)
        
        recipes = generator.generate_recipes()
        
        return JSONResponse(content={
            "status": "success",
            "recipes": recipes
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Recipe Generator: {str(e)}")

@app.get("/api/cost-optimization")
async def run_cost_optimizer():
    try:
        print("\n=== Running Cost Optimizer Module ===")
        
        optimizer = CostOptimizer(config_path)
        
        optimized_costs = optimizer.optimize_costs()
        
        optimized_costs = convert_numpy_types(optimized_costs)
        
        return JSONResponse(content={
            "status": "success",
            "optimized_costs": optimized_costs
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Cost Optimizer: {str(e)}")

@app.get("/api/spoilage-detection")
async def run_spoilage_detection_get():
    try:
        print("\n=== Running Food Spoilage Detection Module (GET) ===")
        
        detector = FoodSpoilageDetector(config_path)
        
        sample_images = detector.get_sample_images()
        if sample_images:
            image_path = os.path.join(detector.config['data']['raw_spoilage_image_path'], sample_images[0])
        else:
            raise HTTPException(status_code=400, detail="No sample images found")
        
        result = detector.detect_spoilage(image_path)
        
        return JSONResponse(content={
            "status": "success",
            "image_path": image_path,
            "result": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Food Spoilage Detection: {str(e)}")

@app.post("/api/spoilage-detection")
async def run_spoilage_detection_post(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    try:
        print("\n=== Running Food Spoilage Detection Module (POST) ===")
        
        detector = FoodSpoilageDetector(config_path)
        
        file_extension = file.filename.split('.')[-1].lower()
        temp_file_path = get_temp_file_path(file_extension)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        background_tasks.add_task(os.remove, temp_file_path)
        
        image_path = temp_file_path
        
        result = detector.detect_spoilage(image_path)
        
        return JSONResponse(content={
            "status": "success",
            "image_path": image_path,
            "result": result
        })
    except Exception as e:
        print(f"Error in food spoilage detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Food Spoilage Detection: {str(e)}")

@app.get("/api/waste-classification")
async def run_waste_classification_get():
    try:
        print("\n=== Running Waste Classification Module ===")
        
        classifier = FoodWasteClassifier(config_path)
        
        sample_images = classifier.config['data']['sample_waste_images']
        if not sample_images:
            raise HTTPException(status_code=400, detail="No sample images found in configuration")
            
        image_path = os.path.join(classifier.config['data']['raw_waste_image_path'], sample_images[6])
        if not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail=f"Sample image not found at path: {image_path}")
            
        print(f"Using sample image: {image_path}")
        
        result = classifier.detect_food_waste(image_path)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to classify waste - no result returned")
        
        result = convert_numpy_types(result)
        
        return JSONResponse(content={
            "status": "success",
            "classification": result,
            "image_path": image_path
        })
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in waste classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Waste Classification: {str(e)}")

@app.get("/api/inventory-tracking")
async def inventory_tracking():
    try:
        tracker = InventoryTracker(config_path)
        print("tracker")
        results = tracker.detect_inventory()
        print(results)
        return {
            "status": "success",
            "message": "Inventory tracking completed successfully",
            "results": results,
            "output_path": tracker.annotated_image_path
        }
    except Exception as e:
        print(f"Error in inventory tracking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Inventory Tracking: {str(e)}")

@app.post("/api/inventory-tracking")
async def inventory_tracking_post(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        temp_file_path = get_temp_file_path(".jpg")
        try:
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
        
        try:
            tracker = InventoryTracker(config_path)
            tracker.input_image_path = temp_file_path
            
            results = tracker.detect_inventory()
            
            output_dir = Path("data/output/detection_images")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            original_filename = file.filename
            output_filename = f"detected_{original_filename}"
            output_path = output_dir / output_filename
            
            if tracker.annotated_image_path:
                output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), tracker.annotated_image)
            
            static_dir = Path("static")
            static_dir.mkdir(exist_ok=True)
            
            static_path = static_dir / output_filename
            shutil.copy2(output_path, static_path)
            
            background_tasks.add_task(delete_file, temp_file_path)
            
            serializable_results = convert_numpy_types(results)
            
            return JSONResponse(content={
                "status": "success",
                "message": "Inventory tracking completed successfully",
                "results": serializable_results,
                "output_path": f"/static/{output_filename}"
            })
        except Exception as e:
            background_tasks.add_task(delete_file, temp_file_path)
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in inventory tracking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Inventory Tracking: {str(e)}")

@app.get("/api/stock-detection")
async def stock_detection():
    try:
        print("stock detection")
        print(f"Loading config from: {config_path}")
        
        detector = StockDetector(config_path=config_path)
        results = detector.detect_stock()
        print(results)
        return {
            "status": "success",
            "message": "Stock detection completed successfully",
            "results": results,
            "video_url": "/static/output_video.mp4"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stock-detection")
async def upload_stock_detection_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a video for stock detection"""
    try:
        print("Uploading video for stock detection")
        
        file_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        video_filename = f"uploaded_video_{file_id}.mp4"
        video_path = os.path.join(config['data']['video_image_path'].rsplit('/', 1)[0], video_filename)
        
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        config['data']['video_image_path'] = video_path
        
        # Define absolute path to fina/frontend/public/static
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        frontend_static_path = os.path.join(project_root, "fina", "frontend", "public", "static")
        os.makedirs(frontend_static_path, exist_ok=True)
        output_filename = "output_video.mp4"
        output_path = os.path.join(frontend_static_path, output_filename)
        config['data']['output_video_image_path'] = output_path
        
        task_tracker[task_id] = {
            "status": "processing",
            "progress": 0,
            "start_time": time.time(),
            "result": None,
            "error": None,
            "file_id": file_id
        }
        
        async def process_video():
            try:
                task_tracker[task_id]["status"] = "processing"
                
                detector = StockDetector(config_path=config_path)
                detector.video_path = video_path
                detector.output_video_path = output_path  # Use the absolute path
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                for progress in range(10, 90, 10):
                    await asyncio.sleep(1)
                    task_tracker[task_id]["progress"] = progress
                
                results = detector.detect_stock()
                print(f"Detection results: {results}")
                
                task_tracker[task_id]["progress"] = 90
                
                if os.path.exists(output_path):
                    print(f"Output video exists at: {output_path}")
                else:
                    raise Exception(f"Output video not found at: {output_path}")
                
                video_url = f"/static/output_video.mp4"
                
                task_tracker[task_id]["progress"] = 100
                
                task_tracker[task_id]["status"] = "completed"
                
                task_tracker[task_id]["result"] = {
                    "results": results,
                    "video_url": video_url,
                    "timestamp": datetime.now().isoformat()
                }
                task_tracker[task_id]["end_time"] = time.time()
                
                print(f"Task {task_id} completed successfully. Video available at: {video_url}")
                
            except Exception as e:
                task_tracker[task_id]["status"] = "failed"
                task_tracker[task_id]["error"] = str(e)
                task_tracker[task_id]["end_time"] = time.time()
                
                print(f"Error in background processing for task {task_id}: {str(e)}")
                print(f"Error details: {traceback.format_exc()}")
                
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        print(f"Cleaned up temporary video file: {video_path}")
                except Exception as cleanup_error:
                    print(f"Error during cleanup: {str(cleanup_error)}")
        
        background_tasks.add_task(process_video)
        
        return {
            "status": "success",
            "message": "Video uploaded and processing started",
            "task_id": task_id
        }
    except Exception as e:
        print(f"Error uploading video: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error uploading video: {str(e)}")

@app.get("/api/task-status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in task_tracker:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_tracker[task_id]
    
    elapsed_time = time.time() - task["start_time"]
    
    response = {
        "status": task["status"],
        "progress": task["progress"],
        "elapsed_time": round(elapsed_time, 2)
    }
    
    if task["status"] == "completed":
        response["result"] = task["result"]
    elif task["status"] == "failed":
        response["error"] = task["error"]
    
    return response

@app.post("/api/waste-classification")
async def run_waste_classification_post(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None)
):
    try:
        print("\n=== Running Waste Classification Module ===")
        
        classifier = FoodWasteClassifier(config_path)
        
        if file:
            file_extension = file.filename.split('.')[-1].lower()
            temp_file_path = get_temp_file_path(file_extension)
            
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            background_tasks.add_task(os.remove, temp_file_path)
            
            image_path = temp_file_path
        else:
            sample_images = classifier.config['data']['sample_waste_images']
            if sample_images:
                image_path = os.path.join(classifier.config['data']['raw_waste_image_path'], sample_images[2])
                print(f"Using sample image: {image_path}")
            else:
                raise HTTPException(status_code=400, detail="No sample images found")
        
        result = classifier.detect_food_waste(image_path)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to classify waste - no result returned")
        
        result = convert_numpy_types(result)
        
        return JSONResponse(content={
            "status": "success",
            "classification": result,
            "image_path": image_path
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Waste Classification: {str(e)}")

@app.get("/api/waste-heatmap")
async def run_waste_heatmap_get():
    try:
        print("\n=== Running Waste Heatmap Generation Module ===")
        
        generator = WasteHeatmapGenerator(config_path)
        
        sample_images = generator.config['data']['sample_waste_heatmap_images']
        if not sample_images:
            raise HTTPException(status_code=400, detail="No sample images found in configuration")
            
        image_path = os.path.join(generator.config['data']['raw_waste_heatmap_path'], sample_images[2])
        if not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail=f"Sample image not found at path: {image_path}")
            
        print(f"Using sample image: {image_path}")
        
        heatmap_path, detections_path = generator.create_waste_heatmap(image_path)
        if not heatmap_path or not detections_path:
            raise HTTPException(status_code=500, detail="Failed to generate heatmap or detections")
            
        if not os.path.exists(heatmap_path) or not os.path.exists(detections_path):
            raise HTTPException(status_code=500, detail="Generated files not found")
        
        heatmap_url = f"/static/{os.path.basename(heatmap_path)}"
        detections_url = f"/static/{os.path.basename(detections_path)}"
        
        os.makedirs("static", exist_ok=True)
        
        try:
            shutil.copy2(heatmap_path, os.path.join("static", os.path.basename(heatmap_path)))
            shutil.copy2(detections_path, os.path.join("static", os.path.basename(detections_path)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to copy files to static directory: {str(e)}")
        
        return JSONResponse(content={
            "status": "success",
            "heatmap_url": heatmap_url,
            "detections_url": detections_url,
            "image_path": image_path
        })
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in waste heatmap generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Waste Heatmap Generation: {str(e)}")

@app.post("/api/waste-heatmap")
async def run_waste_heatmap_post(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None)
):
    try:
        print("\n=== Running Waste Heatmap Generation Module ===")
        
        generator = WasteHeatmapGenerator(config_path)
        
        if file:
            file_extension = file.filename.split('.')[-1].lower()
            temp_file_path = get_temp_file_path(file_extension)
            
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            background_tasks.add_task(os.remove, temp_file_path)
            
            image_path = temp_file_path
        else:
            sample_images = generator.config['data']['sample_waste_heatmap_images']
            if sample_images:
                image_path = os.path.join(generator.config['data']['raw_waste_heatmap_path'], sample_images[2])
                print(f"Using sample image: {image_path}")
            else:
                raise HTTPException(status_code=400, detail="No sample images found")
        
        heatmap_path, detections_path = generator.create_waste_heatmap(image_path)
        if not heatmap_path or not detections_path:
            raise HTTPException(status_code=500, detail="Failed to generate heatmap or detections")
            
        if not os.path.exists(heatmap_path) or not os.path.exists(detections_path):
            raise HTTPException(status_code=500, detail="Generated files not found")
        
        heatmap_url = f"/static/{os.path.basename(heatmap_path)}"
        detections_url = f"/static/{os.path.basename(detections_path)}"
        
        os.makedirs("static", exist_ok=True)
        
        try:
            shutil.copy2(heatmap_path, os.path.join("static", os.path.basename(heatmap_path)))
            shutil.copy2(detections_path, os.path.join("static", os.path.basename(detections_path)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to copy files to static directory: {str(e)}")
        
        return JSONResponse(content={
            "status": "success",
            "heatmap_url": heatmap_url,
            "detections_url": detections_url,
            "image_path": image_path
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Waste Heatmap Generation: {str(e)}")

@app.get("/api/dashboard")
async def run_dashboard():
    try:
        print("\n=== Running Dashboard Module ===")
        
        output_dashboard_path = config['data']['output_dashboard_path']
        
        tracker = RestaurantWasteTracker(config_path)
        
        tracker.save_data_to_csv(output_dashboard_path)
        tracker.generate_chart_data_csvs(output_dashboard_path)
        
        dashboard = tracker.generate_profit_loss_dashboard()
        time_analysis = tracker.generate_time_based_analysis()
        
        dashboard = convert_numpy_types(dashboard)
        time_analysis = {
            'weekly': convert_numpy_types(time_analysis['weekly'].to_dict('records')),
            'monthly': convert_numpy_types(time_analysis['monthly'].to_dict('records')),
            'quarterly': convert_numpy_types(time_analysis['quarterly'].to_dict('records'))
        }
        
        return JSONResponse(content={
            "status": "success",
            "output_path": output_dashboard_path
        })
    except Exception as e:
        print(f"Error in Dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in Dashboard: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)