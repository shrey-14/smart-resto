import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time
import yaml
import os
from collections import defaultdict
import subprocess
import tempfile

class StockDetector:
    def __init__(self, config_path=None):
        # Get the workspace root directory
        self.WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Initialize config
        self.config = None
        
        # Load config if path is provided
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            # Define paths from config
            self.video_path = self.config['data']['video_image_path']
            self.output_video_path = self.config['data']['output_video_image_path']
            self.model_path = self.config['model']['inventory_model_path']
            self.output_csv_count_path = self.config['data']['output_csv_count_path']
        else:
            # Use default paths if no config is provided
            self.video_path = os.path.join(self.WORKSPACE_ROOT, '../data/raw/stock_videos/sample_video.mp4')
            self.output_video_path = os.path.join(self.WORKSPACE_ROOT, '../data/output/stock_prediction/output_video.mp4')
            self.model_path = os.path.join(self.WORKSPACE_ROOT, '../models/food_detection_model/best.pt')
            self.output_csv_count_path = os.path.join(self.WORKSPACE_ROOT, '../data/output/stock_prediction/food_count.csv')
        
        # Define inventory items and their colors
        self.inventory_items = {
            "Apple": 0, "Cheese": 1, "Cucumber": 2, "Egg": 3, "Grape": 4, "Zucchini": 5,
            "Mushroom": 6, "Strawberry": 7, "Tomato": 8, "Banana": 9, "Lemon": 10,
            "Broccoli": 11, "Orange": 12, "Carrot": 13
        }
        
        self.food_colors = {
            "Apple": (255, 0, 0),      # Red
            "Cheese": (255, 255, 0),   # Yellow
            "Cucumber": (0, 255, 0),   # Green
            "Egg": (255, 255, 255),    # White
            "Grape": (128, 0, 128),    # Purple
            "Zucchini": (0, 128, 0),   # Dark Green
            "Mushroom": (192, 192, 192), # Silver
            "Strawberry": (255, 0, 255), # Magenta
            "Tomato": (0, 0, 255),     # Blue
            "Banana": (0, 255, 255),   # Cyan
            "Lemon": (0, 255, 0),      # Lime
            "Broccoli": (0, 128, 0),   # Dark Green
            "Orange": (0, 165, 255),   # Orange
            "Carrot": (0, 128, 255)    # Light Blue
        }
        
        # Define parameters
        self.min_box_size = 0.001  # 0.1% of frame size
        self.optimal_width = 640
        self.optimal_height = 640
        self.temporal_window = 5
        self.spatial_window = 10
        self.class_history_window = 10
        
        # Initialize tracking variables
        self.prev_detections = {}
        self.spatial_history = {}
        self.class_history = {}
        self.food_items = {}
        
        # Initialize model
        self.model = None
        self.class_mapping = {}
    
    def load_model(self):
        """Load and initialize the YOLO model."""
        print(f"Loading YOLO model from: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            print("YOLO model loaded successfully")
            
            print("Model information:")
            print(f"Model type: {type(self.model)}")
            print(f"Model names: {self.model.names}")
            
            model_classes = list(self.model.names.values())
            print(f"Model classes: {model_classes}")
            
            for item in self.inventory_items.keys():
                if item not in model_classes:
                    print(f"WARNING: '{item}' is not in the model's class list")
            
            for i, class_name in enumerate(model_classes):
                for item in self.inventory_items.keys():
                    if item.lower() in class_name.lower() or class_name.lower() in item.lower():
                        self.class_mapping[i] = item
                        print(f"Mapped model class '{class_name}' to inventory item '{item}'")
                        break
            
            print(f"Class mapping: {self.class_mapping}")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
    
    def check_spatial_consistency(self, x1, y1, x2, y2, food_type):
        if food_type not in self.spatial_history:
            self.spatial_history[food_type] = []
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        self.spatial_history[food_type].append((center_x, center_y))
        
        if len(self.spatial_history[food_type]) > self.spatial_window:
            self.spatial_history[food_type].pop(0)
        
        if len(self.spatial_history[food_type]) >= 3:
            avg_x = sum(x for x, _ in self.spatial_history[food_type]) / len(self.spatial_history[food_type])
            avg_y = sum(y for _, y in self.spatial_history[food_type]) / len(self.spatial_history[food_type])
            
            distance = np.sqrt((center_x - avg_x)**2 + (center_y - avg_y)**2)
            frame_diagonal = np.sqrt(self.optimal_width**2 + self.optimal_height**2)
            normalized_distance = distance / frame_diagonal
            
            if normalized_distance > 0.9:
                return False
        
        return True
    
    def check_class_consistency(self, food_type, cls, conf):
        if food_type not in self.class_history:
            self.class_history[food_type] = []
        
        self.class_history[food_type].append((cls, conf))
        
        if len(self.class_history[food_type]) > self.class_history_window:
            self.class_history[food_type].pop(0)
        
        if len(self.class_history[food_type]) >= 3:
            class_counts = {}
            for c, _ in self.class_history[food_type]:
                class_counts[c] = class_counts.get(c, 0) + 1
            
            most_common_class = max(class_counts.items(), key=lambda x: x[1])[0]
            if cls != most_common_class:
                avg_conf = sum(c for _, c in self.class_history[food_type] if _ == cls) / class_counts.get(cls, 1)
                if conf < avg_conf * 1.05:
                    return False
        
        return True
    
    def check_color_consistency(self, frame, x1, y1, x2, y2, food_type):
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            return True
        
        avg_color = np.mean(roi, axis=(0, 1))
        
        color_ranges = {
            "Apple": [(0, 50, 50), (50, 255, 255)],
            "Cheese": [(0, 200, 200), (100, 255, 255)],
            "Cucumber": [(50, 100, 0), (150, 255, 100)],
            "Egg": [(200, 200, 200), (255, 255, 255)],
            "Grape": [(100, 0, 100), (200, 100, 200)],
            "Zucchini": [(50, 100, 0), (150, 255, 100)],
            "Mushroom": [(150, 150, 150), (220, 220, 220)],
            "Strawberry": [(0, 0, 200), (100, 100, 255)],
            "Tomato": [(0, 0, 150), (50, 50, 255)],
            "Banana": [(0, 200, 200), (100, 255, 255)],
            "Lemon": [(0, 200, 0), (100, 255, 100)],
            "Broccoli": [(50, 100, 0), (150, 255, 100)],
            "Orange": [(0, 150, 200), (50, 255, 255)],
            "Carrot": [(0, 100, 200), (50, 255, 255)]
        }
        
        if food_type not in color_ranges:
            return True
        
        min_color, max_color = color_ranges[food_type]
        for i in range(3):
            if avg_color[i] < min_color[i] or avg_color[i] > max_color[i]:
                if avg_color[i] < min_color[i] * 0.5 or avg_color[i] > max_color[i] * 1.5:
                    return False
        
        return True
    
    def process_frame(self, frame, scale_x, scale_y):
        resized_frame = cv2.resize(frame, (self.optimal_width, self.optimal_height))
        
        try:
            results = self.model(resized_frame, conf=0.03, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = result.names[cls]
                    
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height
                    frame_area = self.optimal_width * self.optimal_height
                    box_size_percent = box_area / frame_area
                    
                    if box_size_percent < self.min_box_size:
                        continue
                    
                    food_type = None
                    if cls in self.class_mapping:
                        food_type = self.class_mapping[cls]
                    elif class_name in self.inventory_items:
                        food_type = class_name
                    
                    if food_type is None:
                        continue
                    
                    if not self.check_spatial_consistency(x1, y1, x2, y2, food_type):
                        continue
                    
                    if not self.check_class_consistency(food_type, cls, conf):
                        continue
                    
                    orig_x1 = int(x1 * scale_x)
                    orig_y1 = int(y1 * scale_y)
                    orig_x2 = int(x2 * scale_x)
                    orig_y2 = int(y2 * scale_y)
                    
                    if not self.check_color_consistency(frame, orig_x1, orig_y1, orig_x2, orig_y2, food_type):
                        continue
                    
                    detections.append([x1, y1, x2, y2, food_type, conf])
            
            return detections
            
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
            return []
    
    def draw_results(self, frame, detections, scale_x, scale_y, frame_counts):
        for det in detections:
            bbox = det[:4]
            food_type = det[4]
            conf = det[5]
            
            x1, y1, x2, y2 = bbox
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            color = self.food_colors.get(food_type, (0, 255, 0))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{food_type} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            frame_counts[food_type] += 1
        
        y_offset = 30
        for food_type, count in frame_counts.items():
            if count > 0:
                text = f"{food_type}: {count}"
                cv2.putText(frame, text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 25
        
        return frame
    
    def reencode_to_h264(self, input_path, output_path):
        """Re-encode the video to H.264/AAC using FFmpeg."""
        try:
            print(f"Re-encoding video to H.264: {input_path} -> {output_path}")
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',
                '-profile:v', 'main',
                '-c:a', 'aac',
                '-strict', '-2',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"FFmpeg re-encoding successful: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error during FFmpeg re-encoding: {e.stderr}")
            raise
        except FileNotFoundError:
            print("Error: FFmpeg not found. Please install FFmpeg and ensure it's in your system PATH.")
            raise
    
    def detect_stock(self):
        """Main method to run stock detection on video."""
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Initialize video capture
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return {}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Original video properties: {frame_width}x{frame_height} @ {fps}fps")
        
        # Calculate scaling factors
        scale_x = frame_width / self.optimal_width
        scale_y = frame_height / self.optimal_height
        
        # Create a temporary file for the initial OpenCV output
        temp_fd, temp_video_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)  # Close the file descriptor
        
        # Initialize video writer with mp4v (temporary)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
        
        # Initialize tracking variables
        frame_count = 0
        food_counts = {item: 0 for item in self.inventory_items.keys()}
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of video file")
                    break
                
                frame_count += 1
                display_frame = frame.copy()
                
                # Process frame
                detections = self.process_frame(frame, scale_x, scale_y)
                
                # Initialize frame counts
                frame_counts = {item: 0 for item in self.inventory_items.keys()}
                
                # Draw results
                display_frame = self.draw_results(display_frame, detections, scale_x, scale_y, frame_counts)
                
                # Calculate and display FPS
                elapsed_time = time.time() - start_time
                fps_current = frame_count / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(display_frame, f"FPS: {fps_current:.2f}", (10, frame_height - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Write frame to temporary video
                out.write(display_frame)
                
                # Update total counts
                for food_type, count in frame_counts.items():
                    food_counts[food_type] = max(food_counts[food_type], count)
                
                # Print progress every 10 frames
                if frame_count % 10 == 0:
                    print(f"Processed frame {frame_count}")
        
        finally:
            # Cleanup
            cap.release()
            out.release()
        
        # Re-encode the temporary video to H.264
        try:
            os.makedirs(os.path.dirname(self.output_video_path), exist_ok=True)
            self.reencode_to_h264(temp_video_path, self.output_video_path)
        finally:
            # Remove the temporary file
            try:
                os.remove(temp_video_path)
                print(f"Temporary file removed: {temp_video_path}")
            except Exception as e:
                print(f"Error removing temporary file: {e}")
        
        print("\nProcessing complete!")
        print("\nVideo saved at: ", self.output_video_path)
        print("\nTotal food items detected:")
        
        for food_type, count in food_counts.items():
            if count > 0:
                self.food_items[food_type] = count
                print(f"{food_type}: {count}")
        
        return self.food_items

# For backward compatibility
def detect_stock_in_video():
    detector = StockDetector()
    return detector.detect_stock()