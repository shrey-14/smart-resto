# import requests
# from PIL import Image
# from io import BytesIO
# from groq import Groq
# import os
# from dotenv import load_dotenv
# import csv
# import datetime
# from datetime import datetime, timezone
# import base64
# import json
# import re
# import logging
# import yaml

# class FoodWasteClassifier:
#     def __init__(self, config_path="config/config.yaml"):
#         # Load environment variables
#         load_dotenv()
        
#         # Load configuration
#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)
        
#         # Get API key from environment
#         self.groq_api_key = os.environ.get("GROQ_API_KEY")
#         if not self.groq_api_key:
#             raise ValueError("GROQ_API_KEY environment variable not set")
        
#         # Initialize the Groq client
#         self.client = Groq(api_key=self.groq_api_key)
        
#         # Get model name from config or use default
#         self.model_name = self.config.get('model', {}).get('name', "llama-3.2-11b-vision-preview")
        
#         # Configure logging
#         log_path = self.config.get('data', {}).get('log_path', "data/output/waste_classification/food_waste_classifier.log")
#         os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
#         logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(levelname)s - %(message)s',
#             handlers=[
#                 logging.FileHandler(log_path),
#                 logging.StreamHandler()
#             ]
#         )
#         self.logger = logging.getLogger(__name__)
        
#         # Set output paths
#         self.csv_output_path = self.config.get('data', {}).get('output_waste_classification_path', 
#                                                               "data/output/waste_classification/food_waste_log.csv")
#         os.makedirs(os.path.dirname(self.csv_output_path), exist_ok=True)
        
#         # Set input paths
#         self.raw_waste_image_path = self.config.get('data', {}).get('raw_waste_image_path', 
#                                                                    "data/raw/waste_food_dataset")
#         self.sample_waste_images = self.config.get('data', {}).get('sample_waste_images', [])
    
#     def detect_food_waste(self, image_path=None):
#         """
#         Detect food waste in the given image
        
#         Args:
#             image_path (str, optional): Path to the image file. If None, uses the first sample image.
            
#         Returns:
#             str: JSON response from the model
#         """
#         # Use provided image path or default to first sample image
#         if not image_path and self.sample_waste_images:
#             image_path = os.path.join(self.raw_waste_image_path, self.sample_waste_images[0])
        
#         if not image_path:
#             self.logger.error("No image path provided and no sample images available")
#             return "Error: No image path provided"
        
#         # Derive image format from extension and validate
#         image_format = image_path.split(".")[-1].upper()
#         format_to_mime = {
#             "JPEG": "image/jpeg",
#             "JPG": "image/jpeg",
#             "PNG": "image/png",
#             "WEBP": "image/webp",
#             "BMP": "image/bmp",
#             "GIF": "image/gif"
#         }
#         if image_format not in format_to_mime:
#             self.logger.error(f"Unsupported image format '{image_format}'. Supported formats: {list(format_to_mime.keys())}")
#             return f"Error: Unsupported image format '{image_format}'"

#         # Normalize "JPG" to "JPEG" for PIL save compatibility
#         save_format = "JPEG" if image_format == "JPG" else image_format

#         # Verify image accessibility and load it
#         try:
#             if not os.path.exists(image_path):
#                 raise FileNotFoundError(
#                     f"Image file not found at {image_path}. "
#                     f"Ensure the file exists and the path is correct."
#                 )
#             image = Image.open(image_path).convert("RGB")
            
#             # Convert image to base64 for Groq API
#             buffered = BytesIO()
#             try:
#                 image.save(buffered, format=save_format)
#             except KeyError:
#                 self.logger.error(f"PIL does not support saving as '{save_format}' for this image")
#                 return f"Error: PIL does not support saving as '{save_format}' for this image"
#             image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
#         except Exception as e:
#             self.logger.error(f"Error loading image: {e}")
#             return f"Error loading image: {e}"

#         # Generate image_id from the original filename
#         image_id = os.path.basename(image_path)  # e.g., "xyz.jpg"

#         # Get current timestamp (timezone-aware)
#         timestamp = datetime.now(timezone.utc).isoformat()

#         # Prepare the data URL
#         data_url = f"data:{format_to_mime[image_format]};base64,{image_base64}"
#         self.logger.debug(f"Data URL length: {len(data_url)}")
#         self.logger.debug(f"Data URL preview: {data_url[:100]}...")

#         prompt = f"""
#         IMPORTANT: Reply ONLY with JSON. Do not add any explanatory text before or after the JSON.
        
#         Your task is to analyze the provided image and determine if it contains food, then classify any food waste according to specific categories. Follow this step-by-step reasoning process to arrive at the output:

#         **Step 1: Check for Food Presence**
#         - Examine the image carefully to identify any food items (e.g., fruits, vegetables, breads, eggs shells, fruit peels, vegetable peels, food packets, milk carton).
#         - If no food items are visible (e.g., the image shows an empty table, objects like a chair, or non-food items), conclude that no food is present.
#         - If no food is detected, respond with this exact JSON:
#         {{
#           "timestamp": "{timestamp}",
#           "image_id": "{image_id}",
#           "contains_food": false,
#           "is_waste": false,
#           "categories": []
#         }}

#         **Step 2: Identify Food Items**
#         - If food is present, identify the specific food items in the image (e.g., apple, banana, pasta, apple peel, carrot peel, milk carton, coke bottles, breads).
#         - Note each distinct food item clearly for further analysis.

#         **Step 3: Check for Waste Categories**
#         - For each identified food item, evaluate whether it belongs to any of the following waste categories:
#           - Over-portion: Uneaten food in large quantities (e.g., half a plate of pasta left untouched, few pieces of pizza is left, half of bread is uneaten).
#           - Spoiled: Visibly unfit for consumption (e.g., spoiled tomato, spoiled orange, moldy cheese, rotten fruit or vegetables with discoloration or mold, mold on bread).
#           - Inedible: Non-edible food parts (e.g., banana peels, bones, eggshells, potato peels, cucumber peels).
#           - Contaminated: Food mixed with non-food waste (e.g., food scraps wrapped in plastic, food with dirt or debris, food peels with near dustbin, fruit or vegetables peels lying in dustbin).
#         - If none of the food items fall into these categories (e.g., the food appears fresh and ready to eat), conclude that the food is not waste.
#         - If no waste is detected but food is present, respond with this exact JSON:
#         {{
#           "timestamp": "{timestamp}",
#           "image_id": "{image_id}",
#           "contains_food": true,
#           "is_waste": false,
#           "categories": []
#         }}

#         **Step 4: Classify Food Waste**
#         - For each food item identified as waste, determine which waste category (or categories) it belongs to.
#         - For each waste classification:
#           - Specify the food type as a single item (e.g., tomato, banana peel, onion peel, cucumber piece). Do not group multiple food items together.
#           - Assign the category name (over-portion, spoiled, inedible, contaminated).
#           - Estimate a confidence score (a float between 0 and 1) based on how clear the evidence is.
#           - Provide a brief explanation (e.g., "Mold and discoloration detected", "Spoilage detected").
#         - Collect all waste classifications into a list, ensuring each classification represents exactly one food item.

#         **Step 5: Produce Output** 
#         - YOU MUST RESPOND WITH VALID JSON ONLY. No other text, explanations, or analysis should be included.
#         - If waste is detected, select the top three items from the waste classifications with the highest confidence score.
#         - Return this JSON structure:

#         {{
#           "timestamp": "{timestamp}",
#           "image_id": "{image_id}",
#           "contains_food": true,
#           "is_waste": true,
#           "categories": [
#             {{
#               "food_type": "detected food item (e.g., banana peel)",
#               "name": "category name (e.g., inedible)",
#               "confidence": 0.95,
#               "explanation": "brief reason for classification"
#             }}
#           ]
#         }}

#         EXAMPLE FOR AN IMAGE WITH A DISCARDED BANANA PEEL:

#         {{
#           "timestamp": "{timestamp}",
#           "image_id": "{image_id}",
#           "contains_food": true,
#           "is_waste": true,
#           "categories": [
#             {{
#               "food_type": "banana peel",
#               "name": "inedible",
#               "confidence": 0.95,
#               "explanation": "Discarded peel of banana"
#             }}
#           ]
#         }}

#         RESPOND WITH VALID JSON ONLY. NO OTHER TEXT.
#         """

#         # Prepare input for the Groq API
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image_url", "image_url": {"url": data_url}}
#                 ]
#             }
#         ]

#         # Attempt inference with the model
#         try:
#             self.logger.info(f"Trying inference with {self.model_name}...")
            
#             # Define request parameters
#             request_params = {
#                 "model": self.model_name,
#                 "messages": [
#                     {
#                         "role": "system",
#                         "content": "You are a food waste classification AI. ALWAYS respond with valid JSON objects only, following the exact format specified. Never include explanatory text outside the JSON structure."
#                     },
#                     *messages
#                 ],
#                 "max_tokens": self.config.get('model', {}).get('max_tokens', 1024),
#                 "temperature": self.config.get('model', {}).get('temperature', 0.0),
#                 "stream": False
#             }
            
#             # Remove response_format parameter as it might not be supported by all models
            
#             response = self.client.chat.completions.create(**request_params)
#             model_response = response.choices[0].message.content
#             self.logger.info("Model Response received")
#             self.logger.debug(f"Model Response: {model_response}")
            
#             # Extract JSON from the response using regex - more forgiving approach
#             # Try multiple regex patterns to extract JSON
#             json_patterns = [
#                 r'\{.*\}',  # Standard JSON object pattern
#                 r'\{[^{]*\}', # Try to get the first complete JSON object
#                 r'\{.*"contains_food".*\}', # Look for specific field pattern
#             ]
            
#             result = None
#             json_extracted = False
            
#             for pattern in json_patterns:
#                 if json_extracted:
#                     break
                    
#                 json_match = re.search(pattern, model_response, re.DOTALL)
#                 if json_match:
#                     json_str = json_match.group(0)
#                     try:
#                         result = json.loads(json_str)
#                         json_extracted = True
#                         self.logger.info(f"Successfully extracted JSON with pattern: {pattern}")
#                         break
#                     except json.JSONDecodeError:
#                         self.logger.warning(f"Failed to parse JSON with pattern: {pattern}")
#                         continue
            
#             # If no JSON was extracted with any patterns
#             if not json_extracted:
#                 self.logger.error("No valid JSON found in model response")
                
#                 # Improved handling for text responses
#                 # Check if the response indicates food detection even in text form
#                 food_detected = False
#                 waste_detected = False
                
#                 # Better text analysis - look for positive and negative phrases
#                 positive_food_phrases = ["food is detected", "food was detected", "contains food", "food item", "food items"]
#                 negative_food_phrases = ["no food detected", "no food was detected", "does not contain food", "doesn't contain food"]
                
#                 # Check for negations first
#                 for phrase in negative_food_phrases:
#                     if phrase in model_response.lower():
#                         food_detected = False
#                         break
#                 else:
#                     # Only check positive phrases if no negative phrases were found
#                     for phrase in positive_food_phrases:
#                         if phrase in model_response.lower():
#                             food_detected = True
#                             break
                
#                 # Check for waste detection phrases
#                 if "no waste" in model_response.lower() or "not waste" in model_response.lower():
#                     waste_detected = False
#                 elif "waste" in model_response.lower():
#                     waste_detected = True
                    
#                 # Check for specific food items mentioned in the text
#                 common_food_items = [
#                     "apple", "banana", "orange", "tomato", "bread", "potato", "carrot", 
#                     "fruit", "vegetable", "food", "peel", "leftover", "pizza", "pasta",
#                     "meat", "rice", "cheese", "milk", "egg", "sandwich", "salad"
#                 ]
                
#                 for food_item in common_food_items:
#                     if food_item in model_response.lower():
#                         food_detected = True
#                         break
                        
#                 self.logger.info(f"Text analysis fallback - Food detected: {food_detected}, Waste detected: {waste_detected}")
                
#                 result = {
#                     "timestamp": timestamp,
#                     "image_id": image_id,
#                     "contains_food": food_detected,
#                     "is_waste": waste_detected,
#                     "categories": [],
#                     "error": "No JSON found in response",
#                     "raw_response": model_response
#                 }

#             # Ensure result is properly structured regardless of source
#             if result and not isinstance(result, dict):
#                 result = {
#                     "timestamp": timestamp,
#                     "image_id": image_id,
#                     "contains_food": False,
#                     "is_waste": False,
#                     "categories": [],
#                     "error": "Invalid result structure",
#                     "raw_response": model_response
#                 }

#             # Determine mode based on file existence
#             mode = 'w' if not os.path.exists(self.csv_output_path) else 'a'
            
#             # Save to CSV
#             with open(self.csv_output_path, mode, newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 # Write header if file is being created
#                 if mode == 'w':
#                     writer.writerow(["timestamp", "image_id", "contains_food", "is_waste", "category", "food_type", "confidence", "explanation"])
                
#                 # Write data
#                 if result.get("categories"):
#                     for category in result["categories"]:
#                         writer.writerow([
#                             result.get("timestamp", timestamp),
#                             result.get("image_id", image_id),
#                             result.get("contains_food", False),
#                             result.get("is_waste", False),
#                             category.get("name", ""),
#                             category.get("food_type", ""),
#                             category.get("confidence", ""),
#                             category.get("explanation", "")
#                         ])
#                 else:
#                     writer.writerow([
#                         result.get("timestamp", timestamp),
#                         result.get("image_id", image_id),
#                         result.get("contains_food", False),
#                         result.get("is_waste", False),
#                         "", "", "", ""  # Empty fields for no categories
#                     ])
#                     if "error" in result:
#                         self.logger.warning(f"Logged error: {result['error']}")

#             # Return the full result object rather than just the raw response
#             return result

#         except Exception as e:
#             self.logger.error(f"Error with {self.model_name}: {e}")
#             # Log error to CSV
#             mode = 'w' if not os.path.exists(self.csv_output_path) else 'a'
#             with open(self.csv_output_path, mode, newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 if mode == 'w':
#                     writer.writerow(["timestamp", "image_id", "contains_food", "is_waste", "category", "food_type", "confidence", "explanation"])
#                 writer.writerow([timestamp, image_id, False, False, "", "", "", f"error: {str(e)}"])

#             # Return structured error response
#             return {
#                 "timestamp": timestamp,
#                 "image_id": image_id,
#                 "contains_food": False,
#                 "is_waste": False,
#                 "categories": [],
#                 "error": f"An error occurred in food waste classification: {str(e)}",
#                 "success": False
#             }
    
#     def identify_food_waste(self, image_path=None):
#         """
#         Identify food waste in the given image
        
#         Args:
#             image_path (str, optional): Path to the image file. If None, uses the first sample image.
            
#         Returns:
#             str: JSON response from the model
#         """
#         return self.detect_food_waste(image_path)


# # For backward compatibility
# def detect_food_waste():
#     classifier = FoodWasteClassifier()
#     return classifier.detect_food_waste()

# def identify_food_waste():
#     classifier = FoodWasteClassifier()
#     return classifier.identify_food_waste()





import requests
from PIL import Image
from io import BytesIO
from groq import Groq
import os
from dotenv import load_dotenv
import csv
import datetime
from datetime import datetime, timezone
import base64
import json
import re
import logging
import yaml

class FoodWasteClassifier:
    def __init__(self, config_path="config/config.yaml"):
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get API key from environment
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        # Initialize the Groq client
        self.client = Groq(api_key=self.groq_api_key)
        
        # Get model name from config or use default
        self.model_name = self.config.get('model', {}).get('name', "meta-llama/llama-4-scout-17b-16e-instruct")
        
        # Configure logging
        log_path = self.config.get('data', {}).get('log_path', "data/output/waste_classification/food_waste_classifier.log")
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
        self.csv_output_path = self.config.get('data', {}).get('output_waste_classification_path', 
                                                              "data/output/waste_classification/food_waste_log.csv")
        os.makedirs(os.path.dirname(self.csv_output_path), exist_ok=True)
        
        # Set input paths
        self.raw_waste_image_path = self.config.get('data', {}).get('raw_waste_image_path', 
                                                                   "data/raw/waste_food_dataset")
        self.sample_waste_images = self.config.get('data', {}).get('sample_waste_images', [])
    
    def detect_food_waste(self, image_path=None):
        """
        Detect food waste in the given image
        
        Args:
            image_path (str, optional): Path to the image file. If None, uses the first sample image.
            
        Returns:
            str: JSON response from the model
        """
        # Use provided image path or default to first sample image
        if not image_path and self.sample_waste_images:
            image_path = os.path.join(self.raw_waste_image_path, self.sample_waste_images[0])
        
        if not image_path:
            self.logger.error("No image path provided and no sample images available")
            return "Error: No image path provided"
        
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
            self.logger.error(f"Unsupported image format '{image_format}'. Supported formats: {list(format_to_mime.keys())}")
            return f"Error: Unsupported image format '{image_format}'"

        # Normalize "JPG" to "JPEG" for PIL save compatibility
        save_format = "JPEG" if image_format == "JPG" else image_format

        # Verify image accessibility and load it
        try:
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
                self.logger.error(f"PIL does not support saving as '{save_format}' for this image")
                return f"Error: PIL does not support saving as '{save_format}' for this image"
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return f"Error loading image: {e}"

        # Generate image_id from the original filename
        image_id = os.path.basename(image_path)  # e.g., "xyz.jpg"

        # Get current timestamp (timezone-aware)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Prepare the data URL
        data_url = f"data:{format_to_mime[image_format]};base64,{image_base64}"
        self.logger.debug(f"Data URL length: {len(data_url)}")
        self.logger.debug(f"Data URL preview: {data_url[:100]}...")

        prompt = f"""
        IMPORTANT: Reply ONLY with valid JSON. Do not include any explanatory text, comments, or additional content before or after the JSON.

        Your task is to analyze an image to determine if it contains food and, if so, classify any food waste into predefined categories. Follow the steps below to process the image and produce the required output.

        Step 1: Check for Food Presence
        Objective: Determine if the image contains any food items.
        Definition of Food Items:
        Includes, but is not limited to:
        Fruits (e.g., apples, bananas)
        Vegetables (e.g., carrots, tomatoes)
        Breads
        Eggshells
        Fruit peels (e.g., banana peels, apple peels)
        Vegetable peels (e.g., potato peels, cucumber peels)
        Food packets (e.g., chips, snacks)
        Milk cartons
        Action:
        If no food items are detected (e.g., the image shows an empty table, a chair, or non-food objects), conclude that no food is present.
        Output for No Food:
        {{
        "timestamp": "{timestamp}",
        "image_id": "{image_id}",
        "contains_food": false,
        "is_waste": false,
        "categories": []
        }}


        Step 2: Identify Food Items
        Objective: If food is present, list each distinct food item in the image.
        Examples of Food Items:
        A whole apple
        A banana peel
        A piece of bread
        A tomato
        A milk carton
        Action:
        Identify and note each food item for further analysis in the next step.


        Step 3: Check for Waste Categories
        Objective: Evaluate each food item to determine if it qualifies as waste.
        Waste Categories and Definitions:
        Over-portion: Uneaten food in large quantities.
        Examples: Half a plate of pasta left untouched, several slices of pizza uneaten, half a loaf of bread untouched.
        Spoiled: Visibly unfit for consumption.
        Examples: Moldy cheese, rotten fruit or vegetables with discoloration or mold, mold on bread.
        Inedible: Non-edible food parts.
        Examples: Banana peels, bones, eggshells, potato peels, cucumber peels.
        Contaminated: Food mixed with non-food waste.
        Examples: Food scraps wrapped in plastic, food with dirt or debris, peels near or in a dustbin.
        Action:
        If none of the food items fall into these categories (e.g., the food is fresh and edible), conclude that no waste is present.
        Output for Food Present, No Waste:
        {{
        "timestamp": "{timestamp}",
        "image_id": "{image_id}",
        "contains_food": true,
        "is_waste": false,
        "categories": []
        }}

        
        Step 4: Classify Food Waste
        Objective: For each food item identified as waste, assign it to the appropriate waste category.
        Classification Details:
        Food Type: Specify the food item (e.g., "tomato", "banana peel", "bread"). Each classification must represent one item, not a group.
        Category Name: Assign one of the categories: "over-portion", "spoiled", "inedible", or "contaminated".
        Confidence Score: Provide a float between 0 and 1, reflecting the certainty of the classification based on visual evidence.
        Explanation: Include a brief reason (e.g., "Mold detected", "Non-edible part").
        Action:
        Compile a list of waste classifications, one per food item identified as waste.


        Step 5: Produce Output
        Objective: Format the analysis into a JSON response.
        Rules:
        Replace {timestamp} and {image_id} with the actual values provided.
        If waste is detected, include up to three classifications in "categories", selecting those with the highest confidence scores.
        If fewer than three waste items are detected, include only those available.
        If no waste or no food is present, set "categories" to an empty list.
        Output Structure:
        {{
        "timestamp": "{timestamp}",
        "image_id": "{image_id}",
        "contains_food": true,
        "is_waste": true,
        "categories": [
            {{
            "food_type": "detected food item",
            "name": "category name",
            "confidence": 0.95,
            "explanation": "brief reason for classification"
            }}
        ]
        }}
        """

        # Prepare input for the Groq API
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ]

        # Attempt inference with the model
        try:
            self.logger.info(f"Trying inference with {self.model_name}...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.config.get('model', {}).get('max_tokens', 1024),
                temperature=self.config.get('model', {}).get('temperature', 0.0),
                stream=False
            )
            model_response = response.choices[0].message.content
            self.logger.info("Model Response received")
            self.logger.debug(f"Model Response: {model_response}")
            
            # Extract JSON from the response using regex
            json_match = re.search(r'\{.*\}', model_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    self.logger.error("Extracted JSON is invalid")
                    result = {
                        "timestamp": timestamp,
                        "image_id": image_id,
                        "contains_food": False,
                        "is_waste": False,
                        "categories": [],
                        "error": "Invalid JSON extracted from response"
                    }
            else:
                self.logger.error("No JSON found in model response")
                result = {
                    "timestamp": timestamp,
                    "image_id": image_id,
                    "contains_food": False,
                    "is_waste": False,
                    "categories": [],
                    "error": "No JSON found in response"
                }

            # Determine mode based on file existence
            mode = 'w' if not os.path.exists(self.csv_output_path) else 'a'
            
            # Save to CSV
            with open(self.csv_output_path, mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header if file is being created
                if mode == 'w':
                    writer.writerow(["timestamp", "image_id", "contains_food", "is_waste", "category", "food_type", "confidence", "explanation"])
                
                # Write data
                if result.get("categories"):
                    for category in result["categories"]:
                        writer.writerow([
                            result["timestamp"],
                            result["image_id"],
                            result["contains_food"],
                            result["is_waste"],
                            category["name"],
                            category["food_type"],
                            category["confidence"],
                            category["explanation"]
                        ])
                else:
                    writer.writerow([
                        result["timestamp"],
                        result["image_id"],
                        result.get("contains_food", False),
                        result.get("is_waste", False),
                        "", "", "", ""  # Empty fields for no categories
                    ])
                    if "error" in result:
                        self.logger.warning(f"Logged error: {result['error']}")

            return model_response

        except Exception as e:
            self.logger.error(f"Error with {self.model_name}: {e}")
            # Log error to CSV
            mode = 'w' if not os.path.exists(self.csv_output_path) else 'a'
            with open(self.csv_output_path, mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if mode == 'w':
                    writer.writerow(["timestamp", "image_id", "contains_food", "is_waste", "category", "food_type", "confidence", "explanation"])
                writer.writerow([timestamp, image_id, False, False, "", "", "", f"error: {str(e)}"])

            return f"An error occurred in food waste classification: {str(e)}"
    
    def identify_food_waste(self, image_path=None):
        """
        Identify food waste in the given image
        
        Args:
            image_path (str, optional): Path to the image file. If None, uses the first sample image.
            
        Returns:
            str: JSON response from the model
        """
        return self.detect_food_waste(image_path)


# For backward compatibility
def detect_food_waste():
    classifier = FoodWasteClassifier()
    return classifier.detect_food_waste()

def identify_food_waste():
    classifier = FoodWasteClassifier()
    return classifier.identify_food_waste()