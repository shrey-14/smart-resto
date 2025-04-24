import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from collections import Counter
import yaml

class InventoryTracker:
    def __init__(self, config_path=None):
        # Get the workspace root directory
        self.WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load config if provided
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Define paths relative to workspace root
        self.model_path = self.config['model']['inventory_model_path']
        self.output_dir = os.path.join(self.WORKSPACE_ROOT, "data/output/detection_images")
        self.csv_output_path = os.path.join(self.WORKSPACE_ROOT, "data/output/stock_prediction/predicted_items.csv")
        
        # Class names mapping
        self.items_list = {
            "Apple": 0, "Cheese": 1, "Cucumber": 2, "Egg": 3, "Grape": 4, "Zucchini": 5, 
            "Mushroom": 6, "Strawberry": 7, "Tomato": 8, "Banana": 9, "Lemon": 10, 
            "Broccoli": 11, "Orange": 12, "Carrot": 13
        }
        self.id_to_name = {v: k for k, v in self.items_list.items()}  
        self.all_categories = list(self.items_list.keys())
        
        # Initialize model
        self.model = None
        self.input_image_path = None
        self.annotated_image_path = None
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_output_path), exist_ok=True)
        
    def load_model(self, weights_path=None):
        """Load the YOLO model with the specified weights."""
        if weights_path is None:
            weights_path = self.model_path
            
        if os.path.isdir(weights_path):
            potential_path = os.path.join(weights_path, "best.pt")
            if os.path.exists(potential_path):
                weights_path = potential_path
            else:
                raise FileNotFoundError(f"No best.pt found in directory {weights_path}")
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
        self.model = YOLO(weights_path)
        print(f"Loaded model from {weights_path}")
        return self.model

    def draw_predictions(self, image, predictions):
        """Draw bounding boxes and labels on the image."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # For display
        image_bgr = image.copy()  # For saving (cv2 uses BGR)
        
        for pred in predictions:
            x_min, y_min, x_max, y_max = map(int, pred[:4])  # Bounding box coordinates
            conf = pred[4]  # Confidence score
            class_id = int(pred[5])  # Class ID
            class_name = self.id_to_name.get(class_id, f"Unknown_{class_id}")
            
            # Draw rectangle
            cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Add label with class name and confidence
            label = f"{class_name} ({conf:.2f})"
            cv2.putText(image_rgb, label, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image_bgr, label, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return image_rgb, image_bgr

    def predict_and_plot(self):
        """Run predictions on the input image, plot results, and save annotated images."""
        if self.model is None:
            self.load_model()
            
        if not self.input_image_path:
            raise ValueError("No input image path provided")
            
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get the filename from the input path
        filename = os.path.basename(self.input_image_path)
        output_filename = f"detected_{filename}"
        self.annotated_image_path = os.path.join(self.output_dir, output_filename)
        
        print(f"Processing image: {self.input_image_path}")
        print(f"Output file: {self.annotated_image_path}")

        # Run prediction
        results = self.model.predict(self.input_image_path, conf=0.25, imgsz=640)
        detected_items_list = []

        # Read the image
        image = cv2.imread(self.input_image_path)
        if image is None:
            print(f"Warning: Could not load {self.input_image_path}")
            return []

        # Get predictions
        predictions = results[0].boxes.data.cpu().numpy()

        if len(predictions) == 0:
            print(f"No predictions for {filename}")
            detected_items_list.append({filename: {}})
            # Set the original image as annotated image if no predictions
            self.annotated_image = image
        else:
            # Count items
            class_ids = [int(pred[5]) for pred in predictions]
            item_counts = Counter([self.id_to_name.get(cid, f"Unknown_{cid}") for cid in class_ids])
            detected_items_list.append({filename: dict(item_counts)})
            
            # Draw predictions
            annotated_image_rgb, annotated_image_bgr = self.draw_predictions(image, predictions)
            
            # Set the annotated image attribute
            self.annotated_image = annotated_image_bgr
            
            # Save annotated image
            cv2.imwrite(self.annotated_image_path, annotated_image_bgr)
            print(f"Saved predicted image: {self.annotated_image_path}")

            # Save inventory data to CSV
            if detected_items_list:
                inventory_data = []
                for item_dict in detected_items_list:
                    for img_name, items in item_dict.items():
                        for item_name, count in items.items():
                            inventory_data.append({
                                'image': img_name,
                                'item': item_name,
                                'count': count
                            })
                
                if inventory_data:
                    df = pd.DataFrame(inventory_data)
                    df.to_csv(self.csv_output_path, index=False)
                    print(f"Saved inventory data to CSV: {self.csv_output_path}")

        return detected_items_list

    def detect_inventory(self):
        """Main method to run the inventory detection process."""
        try:
            if self.model is None:
                self.load_model()
            detected_items = self.predict_and_plot()
            return detected_items
        except Exception as e:
            print(f"Error in inventory detection: {e}")
            return []

# def inventory_detection():    
#     tracker = InventoryTracker()    
#     return tracker.detect_inventory()