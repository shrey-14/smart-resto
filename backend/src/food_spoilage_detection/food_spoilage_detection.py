import os
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
import csv
import datetime
from datetime import timezone
import base64
from io import BytesIO
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import yaml

class FoodSpoilageDetector:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load environment variables
        load_dotenv()

        # Initialize the Groq client
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Get model name with default
        self.model_name = self.config.get('model', {}).get('name', "meta-llama/llama-4-scout-17b-16e-instruct")

        # Configure logging with defaults
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_format = self.config.get('logging', {}).get('format', '%(asctime)s - %(levelname)s - %(message)s')
        log_file = self.config.get('logging', {}).get('file', 'food_spoilage_detector.log')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Create output directory if it doesn't exist
        output_path = self.config.get('data', {}).get('output_spoilage_path', 'data/output/spoilage_detection/food_freshness_log.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use fixed values for the retry decorator
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
    def _invoke_llm_with_retry(self, messages):
        try:
            self.logger.info("Attempting API call with messages...")
            
            # Get model parameters with defaults
            max_tokens = self.config.get('model', {}).get('max_tokens', 512)
            temperature = self.config.get('model', {}).get('temperature', 0.0)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            model_response = response.choices[0].message.content
            self.logger.info(f"Generated response: {model_response[:100]}...")
            return model_response
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            raise

    def get_sample_images(self):
        """Get list of available sample images"""
        # Get raw data path with default
        sample_dir = self.config.get('data', {}).get('raw_spoilage_image_path', 'data/raw/fresh_rotten_dataset')
        
        if not os.path.exists(sample_dir):
            self.logger.warning(f"Sample directory not found: {sample_dir}")
            return []
        
        return [img for img in os.listdir(sample_dir) 
                if img.endswith(('.jpg', '.jpeg', '.png'))]

    def detect_spoilage(self, image_path):
        """Detect food spoilage in the given image"""
        try:
            # Derive image format from extension and validate
            image_format = image_path.split(".")[-1].upper()
            format_to_mime = {
                "JPEG": "image/jpeg",
                "JPG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
                "BMP": "image/bmp",
                "GIF": "image/gif"
            }
            if image_format not in format_to_mime:
                raise ValueError(f"Unsupported image format '{image_format}'. Supported formats: {list(format_to_mime.keys())}")

            # Normalize "JPG" to "JPEG" for PIL save compatibility
            save_format = "JPEG" if image_format == "JPG" else image_format

            # Verify image accessibility and load it
            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"Image file not found at {image_path}. "
                    f"Ensure the file exists and the path is correct."
                )
            
            image = Image.open(image_path).convert("RGB")
            
            # Convert image to base64 for Groq API
            buffered = BytesIO()
            try:
                image.save(buffered, format=save_format)
            except KeyError:
                raise ValueError(f"PIL does not support saving as '{save_format}' for this image")
            
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Generate a unique image_name
            image_name = os.path.basename(image_path)

            # Get current timestamp (timezone-aware)
            timestamp = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            data_url = f"data:{format_to_mime[image_format]};base64,{image_base64}"

            # Define the query and instructions
            query = "Is the food in the image fresh or rotten?"
            instructions = f"""Analyze the image and provide a response in the following CSV format:
            "timestamp","image_name","food_item","freshness","status"
            "{timestamp}","{image_name}","[food name]","[fresh/rotten]","success"

            Rules:
            1. food_item: Name of the food detected (or 'Unknown' if unclear)
            2. freshness: Must be either 'fresh' or 'rotten'
            3. status: Must be 'success' if food is detected, 'error' if not
            4. Return ONLY the CSV line, no other text
            5. Use double quotes for all fields
            6. Do not include the header row in your response

            Example response:
            "2024-03-14T10:30:00Z","banana.jpg","banana","rotten","success"
            """

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructions},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]

            # Get model response
            model_response = self._invoke_llm_with_retry(messages)
            
            # Clean and validate the response
            response_lines = [line.strip() for line in model_response.split('\n') if line.strip()]
            if not response_lines:
                raise ValueError("Empty response from model")
            
            # Get the first non-empty line that looks like our CSV format
            csv_line = None
            for line in response_lines:
                if line.startswith('"') and line.count('"') >= 8:  # At least 4 quoted fields
                    csv_line = line
                    break
            
            if not csv_line:
                raise ValueError("No valid CSV line found in model response")

            # Save to CSV
            csv_file = self.config.get('data', {}).get('output_spoilage_path', 'data/output/spoilage_detection/food_freshness_log.csv')
            mode = 'w' if not os.path.exists(csv_file) else 'a'
            
            with open(csv_file, mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if mode == 'w':
                    writer.writerow(["timestamp", "image_name", "food_item", "freshness", "status"])
                
                try:
                    # Parse the CSV line
                    data_row = next(csv.reader([csv_line]))
                    if len(data_row) != 5:
                        raise ValueError(f"Invalid number of fields: {len(data_row)}")
                    
                    # Validate freshness value
                    if data_row[3].lower() not in ['fresh', 'rotten']:
                        data_row[3] = 'Unknown'
                    
                    writer.writerow(data_row)
                    self.logger.info(f"Successfully wrote detection result: {data_row}")
                except csv.Error as e:
                    self.logger.error(f"CSV parsing error: {str(e)}")
                    writer.writerow([timestamp, image_name, "Unknown", "Unknown", "error: Invalid CSV format"])

            return csv_line

        except Exception as e:
            self.logger.error(f"Error in food spoilage detection: {str(e)}")
            # Log error to CSV
            csv_file = self.config.get('data', {}).get('output_spoilage_path', 'data/output/spoilage_detection/food_freshness_log.csv')
            mode = 'w' if not os.path.exists(csv_file) else 'a'
            with open(csv_file, mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if mode == 'w':
                    writer.writerow(["timestamp", "image_name", "food_item", "freshness", "status"])
                writer.writerow([timestamp, image_name, "Unknown", "Unknown", f"error: {str(e)}"])
            return f"An error occurred during food spoilage detection: {str(e)}"

if __name__ == "__main__":
    # This block will not be executed when called from main.py
    pass
