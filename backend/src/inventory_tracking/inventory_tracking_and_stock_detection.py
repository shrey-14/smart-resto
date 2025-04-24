import os
import cv2
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ultralytics import YOLO
from collections import Counter
import yaml
import torch


class InventoryTracker:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Define paths from config
        self.images_dir = self.config['data']['raw_inventory_image_path']
        self.model_path = self.config['model']['inventory_model_path']
        self.output_dir = self.config['data']['output_inventory_image_path']
        self.csv_output_path = self.config['data']['output_inventory_csv_path']
        
        # Class names mapping
        self.items_list = {
            "Apple": 0, "Cheese": 1, "Cucumber": 2, "Egg": 3, "Grape": 4, "Zucchini": 5, 
            "Mushroom": 6, "Strawberry": 7, "Tomato": 8, "Banana": 9, "Lemon": 10, 
            "Broccoli": 11, "Orange": 12, "Carrot": 13
        }
        self.id_to_name = {v: k for k, v in self.items_list.items()}
        self.all_categories = list(self.items_list.keys())
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_output_path), exist_ok=True)

    def load_model(self):
        """Load the YOLO model"""
        weights_path = self.model_path
        if os.path.isdir(weights_path):
            potential_path = os.path.join(weights_path, "best.pt")
            if os.path.exists(potential_path):
                weights_path = potential_path
            else:
                raise FileNotFoundError(f"No best.pt found in directory {weights_path}")
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
        model = YOLO(weights_path)
        print(f"Loaded model from {weights_path}")
        return model

    def get_image_paths(self):
        """Get all image paths from the input directory"""
        image_paths = [os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_paths:
            raise ValueError(f"No images found in {self.images_dir}!")
        print(f"Found {len(image_paths)} images in {self.images_dir}")
        return image_paths

    def draw_predictions(self, image, predictions):
        """Draw predictions on the image"""
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
            cv2.putText(image_rgb, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image_bgr, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image_rgb, image_bgr

    def save_to_csv(self, detected_items_list):
        """Save predictions to CSV"""
        # Prepare data for DataFrame
        data = []
        for item_dict in detected_items_list:
            image_name = list(item_dict.keys())[0]
            counts = item_dict[image_name]
            # Create a row with counts for all categories, defaulting to 0
            row = {"Image name": image_name}
            for category in self.all_categories:
                row[category] = counts.get(category, 0)
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=["Image name"] + self.all_categories)
        
        # Save to CSV
        df.to_csv(self.csv_output_path, index=False)
        print(f"Saved predictions to {self.csv_output_path}")

    def predict_and_plot(self):
        """Run predictions, plot, save, and store items"""
        model = self.load_model()
        image_paths = self.get_image_paths()
        num_images = len(image_paths)

        results = model.predict(image_paths, conf=0.25, imgsz=640)
        detected_items_list = []

        # Calculate grid dimensions
        n_cols = min(5, num_images)
        n_rows = (num_images + n_cols - 1) // n_cols
        
        # Create figure and axes
        if num_images == 1:
            fig, ax = plt.subplots(figsize=(10, 10))
            axes = np.array([ax])
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
            axes = axes.flatten()

        for i, (image_path, result) in enumerate(zip(image_paths, results)):
            if i >= len(axes):
                print(f"Warning: More images ({num_images}) than subplots ({len(axes)}). Truncating.")
                break
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load {image_path}")
                axes[i].text(0.5, 0.5, "Image Load Failed", ha="center", va="center")
                axes[i].axis('off')
                continue
            
            predictions = result.boxes.data.cpu().numpy()
            if len(predictions) == 0:
                print(f"No predictions for {os.path.basename(image_path)}")
                detected_items_list.append({os.path.basename(image_path): {}})
            else:
                class_ids = [int(pred[5]) for pred in predictions]
                item_counts = Counter([self.id_to_name.get(cid, f"Unknown_{cid}") for cid in class_ids])
                detected_items_list.append({os.path.basename(image_path): dict(item_counts)})
            
            annotated_image_rgb, annotated_image_bgr = self.draw_predictions(image, predictions)
            
            axes[i].imshow(annotated_image_rgb)
            axes[i].set_title(os.path.basename(image_path), fontsize=10)
            axes[i].axis('off')

            output_path = os.path.join(self.output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, annotated_image_bgr)
            print(f"Saved predicted image: {output_path}")

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        
        # Save the figure instead of displaying it
        figure_path = os.path.join(self.output_dir, 'detection_summary.png')
        plt.savefig(figure_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved detection summary to: {figure_path}")

        print("\nDetected items per image:")
        for item in detected_items_list:
            print(item)
        
        # Save to CSV
        self.save_to_csv(detected_items_list)
        return detected_items_list

    def run_inventory_detection(self):
        """Main method to run inventory detection"""
        try:
            detected_items = self.predict_and_plot()
            return detected_items
        except Exception as e:
            print(f"Error in inventory detection: {str(e)}")
            raise