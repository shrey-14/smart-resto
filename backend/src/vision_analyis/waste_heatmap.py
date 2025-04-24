import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from dotenv import load_dotenv
from huggingface_hub import login
import logging
import yaml

class WasteHeatmapGenerator:
    def __init__(self, config_path="config/config.yaml"):
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get API key from environment
        self.hf_api = os.environ.get("HF_API")
        if not self.hf_api:
            raise ValueError("HF_API environment variable not set")
        
        # Login to Hugging Face
        login(token=self.hf_api)
        
        # Configure logging
        log_path = self.config.get('data', {}).get('log_path', "data/output/waste_heatmap/waste_heatmap.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set output paths
        self.heatmap_output_path = self.config.get('data', {}).get('output_waste_heatmap_path', 
                                                                  "data/output/waste_heatmap/food_waste_heatmap.jpg")
        self.detections_output_path = self.config.get('data', {}).get('output_waste_detections_path', 
                                                                     "data/output/waste_heatmap/food_waste_detections.jpg")
        os.makedirs(os.path.dirname(self.heatmap_output_path), exist_ok=True)
        
        # Set input paths
        self.raw_waste_heatmap_path = self.config.get('data', {}).get('raw_waste_heatmap_path', 
                                                                     "data/raw/waste_heatmap_dataset")
        self.sample_waste_heatmap_images = self.config.get('data', {}).get('sample_waste_heatmap_images', [])
        
        # Load model and processor
        self.logger.info("Loading OWL-ViT model and processor...")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Define waste types
        self.waste_types = [
            "fruit/vegetable peels", "food waste", "empty cartons", "plastic scrap",
            "vegetable scraps", "spoiled food", "contaminated food",
            "plastic bottles", "fresh food"  # Added to contrast with waste
        ]
        
        # Define color map
        self.color_map = {
            "fruit/vegetable peels": (255, 0, 0),      # Red
            "food waste": (0, 255, 0),                 # Green
            "empty cartons": (0, 0, 255),              # Blue
            "plastic scrap": (255, 255, 0),            # Yellow
            "vegetable scraps": (0, 255, 255),         # Cyan
            "spoiled food": (255, 0, 255),             # Magenta
            "contaminated food": (128, 0, 128),        # Purple
            "plastic bottles": (0, 128, 128),          # Teal
            "fresh food": (255, 255, 255)              # White for fresh food
        }
    
    def create_waste_heatmap(self, image_path=None):
        """
        Create a waste heatmap for the given image
        
        Args:
            image_path (str, optional): Path to the image file. If None, uses the first sample image.
            
        Returns:
            tuple: Paths to the generated heatmap and detections images
        """
        # Use provided image path or default to first sample image
        if not image_path and self.sample_waste_heatmap_images:
            image_path = os.path.join(self.raw_waste_heatmap_path, self.sample_waste_heatmap_images[0])
        
        if not image_path:
            self.logger.error("No image path provided and no sample images available")
            return None, None
        
        try:
            # Load and preprocess image
            self.logger.info(f"Loading image from {image_path}")
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return None, None

        image_source = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)  # For OpenCV
        image_rgb = np.array(image_pil)  # For plotting
        image_gray = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY)  # For contour detection

        # Detect multiple types of waste
        threshold = 0.1  # Increased to reduce false positives
        h, w = image_source.shape[:2]
        
        self.logger.info("Processing image with OWL-ViT model...")
        inputs = self.processor(text=self.waste_types, images=image_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process outputs
        target_sizes = torch.tensor([image_pil.size[::-1]])  # [height, width]
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]

        # Extract detections
        boxes = results["boxes"].cpu().numpy()  # [x_min, y_min, x_max, y_max]
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        # Collect all detections initially
        waste_items = []
        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            waste_items.append({
                "type": self.waste_types[label],
                "coords": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "score": float(score)
            })

        # Identify fresh food regions and resolve conflicts strictly
        fresh_food_items = [item for item in waste_items if item["type"] == "fresh food"]
        other_items = [item for item in waste_items if item["type"] != "fresh food"]

        # Function to check if two boxes overlap
        def boxes_overlap(box1, box2):
            x1, y1, x2, y2 = box1
            x1_opp, y1_opp, x2_opp, y2_opp = box2
            xi1 = max(x1, x1_opp)
            yi1 = max(y1, y1_opp)
            xi2 = min(x2, x2_opp)
            yi2 = min(y2, y2_opp)
            return xi2 > xi1 and yi2 > yi1  # True if there is any overlap

        # Filter out waste items that overlap with any fresh food region
        filtered_items = fresh_food_items.copy()
        for other_item in other_items:
            overlaps = False
            for fresh_item in fresh_food_items:
                if boxes_overlap(other_item["coords"], fresh_item["coords"]):
                    overlaps = True
                    break
            if not overlaps:
                filtered_items.append(other_item)

        # Update waste_items with filtered results
        waste_items = filtered_items

        # Plot irregular shapes using enhanced segmentation
        image_with_contours = image_source.copy()

        for item in waste_items:
            x1, y1, x2, y2 = item["coords"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Extract ROI
            roi_color = image_source[y1:y2, x1:x2]
            roi_gray = image_gray[y1:y2, x1:x2]
            if roi_color.size == 0:
                continue

            # Initialize mask for GrabCut
            mask = np.zeros(roi_color.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            rect = (5, 5, roi_color.shape[1] - 5, roi_color.shape[0] - 5)

            # Run GrabCut
            try:
                cv2.grabCut(roi_color, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            except:
                # Fallback to adaptive thresholding
                thresh = cv2.adaptiveThreshold(
                    roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
                )
                mask2 = thresh // 255

            # Find contours
            contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            shifted_contours = [cnt + [x1, y1] for cnt in contours if cv2.contourArea(cnt) > 50]
            cv2.drawContours(
                image_with_contours,
                shifted_contours,
                -1,
                self.color_map.get(item["type"], (0, 255, 0)),
                2
            )

            # Calculate text position dynamically
            text = f"{item['type']} ({item['score']:.2f})"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_x = x1
            text_y = y1 - 5 if y1 > text_height + 5 else y2 + text_height + 5

            # Ensure text stays within image boundaries
            text_y = max(10, min(text_y, h - 10))  # Keep text within 10 pixels of edges

            cv2.putText(
                image_with_contours,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.color_map.get(item["type"], (0, 255, 0)),
                2
            )

        # Generate single heatmap for waste types only (exclude fresh food)
        heatmap_grid = np.zeros((h, w))
        for item in waste_items:
            if item["type"] == "fresh food":
                continue  # Skip fresh food for heatmap
            x1, y1, x2, y2 = item["coords"]
            x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
            x_min, x_max = max(0, x_center - 50), min(w, x_center + 50)
            y_min, y_max = max(0, y_center - 50), min(h, y_center + 50)
            heatmap_grid[y_min:y_max, x_min:x_max] += item["score"]

        heatmap_grid = heatmap_grid / max(np.max(heatmap_grid), 1e-10) if np.max(heatmap_grid) > 0 else heatmap_grid

        # Visualize single heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_grid, cmap="hot", alpha=0.6)
        plt.imshow(image_rgb, alpha=0.4)
        plt.axis("off")
        plt.title("Food Waste Heatmap")
        
        # Save heatmap
        self.logger.info(f"Saving heatmap to {self.heatmap_output_path}")
        plt.savefig(self.heatmap_output_path)
        plt.close()

        # Save image with contours
        self.logger.info(f"Saving detections to {self.detections_output_path}")
        cv2.imwrite(self.detections_output_path, image_with_contours)

        # Print report
        self.logger.info("Food Waste Detection Report")
        if not waste_items:
            self.logger.warning("No items detected. Try adjusting the threshold or waste types.")
        elif all(item["type"] == "fresh food" for item in waste_items):
            self.logger.info("Only fresh food detected. No food waste found.")
        else:
            for item in waste_items:
                self.logger.info(f"Detected {item['type']} at bounding box [{item['coords'][0]:.0f}, {item['coords'][1]:.0f}, {item['coords'][2]:.0f}, {item['coords'][3]:.0f}] with confidence {item['score']:.2f}")
        
        return self.heatmap_output_path, self.detections_output_path
    
    def visualize_heatmap(self, image_path=None):
        """
        Visualize the waste heatmap for the given image
        
        Args:
            image_path (str, optional): Path to the image file. If None, uses the first sample image.
            
        Returns:
            tuple: Paths to the generated heatmap and detections images
        """
        return this.create_waste_heatmap(image_path)


# For backward compatibility
def create_waste_heatmap():
    generator = WasteHeatmapGenerator()
    return generator.create_waste_heatmap()

def visualize_heatmap():
    generator = WasteHeatmapGenerator()
    return generator.visualize_heatmap()

