

# # src/waste_predictor.py
# import pandas as pd
# import os
# import json
# import signal
# import ast
# import time
# from datetime import datetime
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from langchain_groq import ChatGroq
# from langchain.schema import SystemMessage, HumanMessage
# from dotenv import load_dotenv
# import re
# import logging
# from tenacity import retry, stop_after_attempt, wait_exponential
# from jsonschema import validate, ValidationError

# # Load environment variables
# load_dotenv()

# # Initialize the Groq model with increased timeout
# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model_name="Llama-3.1-8b-Instant",
#     timeout=30  # Increased timeout to handle slow responses
# )

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler(), logging.FileHandler('waste_predictor.log')]
# )
# logger = logging.getLogger(__name__)

# # Retry decorator with exponential backoff
# @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=15))
# def invoke_llm_with_retry(llm, messages):
#     try:
#         response = llm.invoke(messages)
#         logger.debug(f"Full LLM response: {response.content}")
#         return response
#     except Exception as e:
#         logger.error(f"API call failed: {e}")
#         raise


# # Enhanced response validation with iterative repair and schema validation
# import json
# import re
# from jsonschema import validate

# def validate_response(response: str) -> tuple[bool, str]:
#     try:
#         # Step 1: Clean the response
#         response = response.strip().strip('"')

#         # Step 2: Try parsing with ast.literal_eval to handle single quotes
#         try:
#             data = ast.literal_eval(response)
#         except (SyntaxError, ValueError) as e:
#             logger.debug(f"Initial parse failed with ast.literal_eval: {e}")
#             # Fallback to regex-based cleaning
#             response = re.sub(r"\'(\w+)\'(\s*:\s*)", r'"\1"\2', response)  # Convert single-quoted keys
#             response = re.sub(r"\'(.*?)\'(?=(?:,|\]|\s*}))", r'"\1"', response)  # Convert single-quoted values
#             response = re.sub(r'(}\s*{)', r'}, {', response)  # Insert missing commas between objects
#             response = re.sub(r'\s+', ' ', response).strip()  # Normalize whitespace
#             data = ast.literal_eval(response)

#         # Step 3: Convert to standard JSON
#         json_data = json.dumps(data)

#         # Step 4: Validate against schema
#         schema = {
#             "type": "object",
#             "properties": {
#                 "analyses": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "recommended_actions": {"type": "array", "items": {"type": "string"}}
#                         },
#                         "required": ["recommended_actions"]
#                     }
#                 }
#             },
#             "required": ["analyses"]
#         }
#         validate(instance=data, schema=schema)
#         return True, json_data
#     except (SyntaxError, ValueError, TypeError, ValidationError) as e:
#         logger.error(f"Validation failed: {e} for raw response: {repr(response)}")
#         return False, f"Invalid JSON format: {e}"

# # Extract recommended actions from LLM response
# def extract_recommended_actions(response: str, ingredients, valid_rows):
#     is_valid, json_data_or_error = validate_response(response)
#     logger.info(f"Validation result: {json_data_or_error if is_valid else 'Failed - ' + json_data_or_error}")
#     if not is_valid:
#         logger.error(f"Raw invalid response: {response}")
#         return {ingr: [f"Fallback: Consult inventory manager (Error: {json_data_or_error}, Raw: {response[:100]}...)"] for ingr in ingredients}

#     try:
#         data = json.loads(json_data_or_error)
#         analyses = data.get("analyses", [])
#         if len(analyses) != len(ingredients):
#             logger.warning(f"Mismatch: Expected {len(ingredients)} analyses, got {len(analyses)}. Using available data.")
#         return {
#             ingr: [action for action in (analyses[i].get("recommended_actions", []) if i < len(analyses) else [])]
#             for i, ingr in enumerate(ingredients)
#         }
#     except (ValueError, KeyError) as e:
#         logger.error(f"Extraction failed: {e}")
#         return {ingr: [f"Fallback: Consult inventory manager (Error: {e})"] for ingr in ingredients}

# # Process a single batch
# def process_batch(batch, batch_ingredients, valid_rows, system_prompt):
#     user_input = "Analyze the following ingredients for waste prevention:\n" + "\n".join(
#         f"Ingredient: {row['ingredient']}, Stock: {row['stock_kg']}kg, "
#         f"Days Since Delivery: {row['days_since_delivery']}, "
#         f"Shelf Life (Days): {row['shelf_life_days']}, "
#         f"Storage Temp: {row['storage_temp_c']}°C, "
#         f"Weekly Usage: {row['weekly_usage_kg']}kg"
#         for row in batch
#     )

#     try:
#         response = invoke_llm_with_retry(llm, [
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=user_input)
#         ])
#         logger.info(f"Raw response for batch {batch[0]['ingredient'][:10]}...: {response.content[:100]}...")

#         recommended_actions_dict = extract_recommended_actions(response.content, batch_ingredients, valid_rows)
#         batch_results = [
#             {
#                 "ingredient": row["ingredient"],
#                 "stock_kg": row["stock_kg"],
#                 "days_since_delivery": row["days_since_delivery"],
#                 "shelf_life_days": row["shelf_life_days"],
#                 "storage_temp_c": row["storage_temp_c"],
#                 "weekly_usage_kg": row["weekly_usage_kg"],
#                 "suggested_replenishment_kg": row["suggested_replenishment_kg"],
#                 "spoilage_risk": calculate_spoilage_risk(row["days_since_delivery"], row["shelf_life_days"]),
#                 "overuse_risk": calculate_overuse_risk(row["stock_kg"], row["weekly_usage_kg"], row["shelf_life_days"]),
#                 "recommended_actions": recommended_actions_dict.get(row["ingredient"], ["No actions available"])
#             }
#             for row in batch
#         ]
#         return batch_results
#     except Exception as e:
#         logger.error(f"Error processing batch starting with {batch[0]['ingredient']}: {e}")
#         return [
#             {
#                 "ingredient": row["ingredient"],
#                 "stock_kg": row["stock_kg"],
#                 "days_since_delivery": row["days_since_delivery"],
#                 "shelf_life_days": row["shelf_life_days"],
#                 "storage_temp_c": row["storage_temp_c"],
#                 "weekly_usage_kg": row["weekly_usage_kg"],
#                 "suggested_replenishment_kg": row["suggested_replenishment_kg"],
#                 "spoilage_risk": "Error: Batch processing failed",
#                 "overuse_risk": "Error: Batch processing failed",
#                 "recommended_actions": ["Fallback: Consult inventory manager"]
#             }
#             for row in batch
#         ]

# # Heuristic calculations
# def calculate_spoilage_risk(days_since_delivery: int, shelf_life_days: int) -> float:
#     """Heuristically calculate spoilage risk based on days since delivery and shelf life."""
#     days_past = days_since_delivery - shelf_life_days
#     if days_past <= 0:
#         return 0.0
#     elif 0 < days_past <= 2:
#         return 0.5
#     elif 2 < days_past <= 5:
#         return 0.8
#     else:
#         return 1.0

# def calculate_overuse_risk(stock_kg: float, weekly_usage_kg: float, shelf_life_days: int) -> float:
#     """Heuristically calculate overuse risk based on stock duration."""
#     stock_duration = (stock_kg / (weekly_usage_kg / 7)) if weekly_usage_kg > 0 else float('inf')
#     if stock_duration >= shelf_life_days:
#         return 0.0
#     elif stock_duration >= 7:
#         return 0.3
#     elif stock_duration >= 3:
#         return 0.7
#     else:
#         return 1.0

# # Predict waste and suggest replenishment for ingredients
# def predict_waste(df: pd.DataFrame) -> pd.DataFrame:
#     results = []

#     # Handle Ctrl+C gracefully
#     def signal_handler(sig, frame):
#         logger.info("Process interrupted by user. Saving partial results...")
#         partial_df = pd.DataFrame(results)
#         partial_df.to_csv("partial_waste_predictions.csv", index=False)
#         exit(0)

#     signal.signal(signal.SIGINT, signal_handler)

#     start_time = time.time()  # Track execution time

#     # Calculate days_since_delivery dynamically using today's date if not provided
#     current_date = datetime.now()
#     if 'days_since_delivery' not in df.columns:
#         if 'delivery_date' not in df.columns:
#             raise ValueError("Either 'days_since_delivery' or 'delivery_date' must be provided in the DataFrame.")
#         df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
#         df['days_since_delivery'] = (current_date - df['delivery_date']).dt.days

#     # Validate input data and build batch input
#     valid_rows = []
#     ingredients = []
#     for _, row in df.iterrows():
#         required_fields = ['ingredient', 'stock_kg', 'days_since_delivery', 'shelf_life_days', 'storage_temp_c', 'weekly_usage_kg']
#         if any(pd.isna(row[field]) for field in required_fields):
#             logger.warning(f"Missing data for {row['ingredient']}. Skipping incomplete row.")
#             results.append({
#                 "ingredient": row["ingredient"],
#                 "stock_kg": None if pd.isna(row["stock_kg"]) else row["stock_kg"],
#                 "days_since_delivery": None if pd.isna(row["days_since_delivery"]) else row["days_since_delivery"],
#                 "shelf_life_days": None if pd.isna(row["shelf_life_days"]) else row["shelf_life_days"],
#                 "storage_temp_c": None if pd.isna(row["storage_temp_c"]) else row["storage_temp_c"],
#                 "weekly_usage_kg": None if pd.isna(row["weekly_usage_kg"]) else row["weekly_usage_kg"],
#                 "suggested_replenishment_kg": 0.0,
#                 "spoilage_risk": "Error: Missing data",
#                 "overuse_risk": "Error: Missing data",
#                 "recommended_actions": []
#             })
#             continue

#         buffer_storage = 0.10 * row['weekly_usage_kg']
#         suggested_replenishment_kg = max(0, (row['weekly_usage_kg'] + buffer_storage) - row['stock_kg'])
#         valid_rows.append({
#             "ingredient": row["ingredient"],
#             "stock_kg": row["stock_kg"],
#             "days_since_delivery": row["days_since_delivery"],
#             "shelf_life_days": row["shelf_life_days"],
#             "storage_temp_c": row["storage_temp_c"],
#             "weekly_usage_kg": row["weekly_usage_kg"],
#             "suggested_replenishment_kg": round(suggested_replenishment_kg, 2)
#         })
#         ingredients.append(row["ingredient"])

#     if not valid_rows:
#         logger.error("No valid data to process.")
#         return pd.DataFrame(results)

#     # Define system prompt
#     system_prompt = (
#         "You are an AI kitchen assistant for a restaurant aiming to minimize waste and optimize inventory. "
#         "For each ingredient provided in the input, provide a list of recommended actions to prevent spoilage or overuse. "
#         "Actions should be detailed and actionable, considering the ingredient type and current conditions (e.g., 'Use spinach in salads within 2 days', 'Adjust storage to 2-4°C for fish', 'Prioritize chicken in tomorrow's menu'). "
#         "Return your response as a VALID JSON object with a single key 'analyses' containing a list of objects, where each object has a 'recommended_actions' key with a list of strings. "
#         "Ensure all keys (e.g., 'analyses', 'recommended_actions') are followed by a colon ':' and their values. The order of analyses must match the order of ingredients in the input. "
#         "Do not include spoilage_risk, overuse_risk, confidence_score, or reasoning. Do not include any additional text or markdown outside the JSON.\n"
#         "Example: For inputs: 'Ingredient: tomato, Stock: 5kg, Days Since Delivery: 12, Shelf Life: 10, Storage Temp: 4°C, Weekly Usage: 7kg' and 'Ingredient: eggplant, Stock: 2kg, Days Since Delivery: 13, Shelf Life: 10, Storage Temp: 6°C, Weekly Usage: 3kg'\n"
#         "{\n"
#         "  'analyses': [\n"
#         "    {\n"
#         "      'recommended_actions': ['Replenish stock by 2.2kg', 'Use in next menu', 'Monitor storage temperature']\n"
#         "    },\n"
#         "    {\n"
#         "      'recommended_actions': ['Replenish stock by 1.7kg', 'Discard if spoiled', 'Adjust storage temp to 2-4°C']\n"
#         "    }\n"
#         "  ]\n"
#         "}"
#     )

#     # Process batches in parallel
#     batch_size = 5
#     batches = [valid_rows[i:i + batch_size] for i in range(0, len(valid_rows), batch_size)]
#     batch_ingredients = [ingredients[i:i + batch_size] for i in range(0, len(ingredients), batch_size)]

#     with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
#         future_to_batch = {
#             executor.submit(process_batch, batch, batch_ingr, valid_rows, system_prompt): (batch, batch_ingr)
#             for batch, batch_ingr in zip(batches, batch_ingredients)
#         }
#         for future in as_completed(future_to_batch):
#             batch_results = future.result()
#             results.extend(batch_results)
#             time.sleep(1)  # Rate limiting delay between batches

#     result_df = pd.DataFrame(results)
#     pd.set_option('display.max_colwidth', 200)
#     pd.set_option('display.width', 200)

#     # Return DataFrame without saving or printing (handled by main.py)
#     return result_df




# src/waste_predictor.py
import pandas as pd
import os
import json
import signal
import ast
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import re
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from jsonschema import validate, ValidationError

# Load environment variables
load_dotenv()

# Initialize the Groq model with increased timeout
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="Llama-3.1-8b-Instant",
    timeout=30  # Increased timeout to handle slow responses
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('waste_predictor.log')]
)
logger = logging.getLogger(__name__)

# Retry decorator with exponential backoff
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=15))
def invoke_llm_with_retry(llm, messages):
    try:
        response = llm.invoke(messages)
        logger.debug(f"Full LLM response: {response.content}")
        return response
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise


# Enhanced response validation with iterative repair and schema validation
import json
import re
from jsonschema import validate

def validate_response(response: str) -> tuple[bool, str]:
    try:
        # Step 1: Clean the response
        response = response.strip().strip('"')

        # Step 2: Try parsing with ast.literal_eval to handle single quotes
        try:
            data = ast.literal_eval(response)
        except (SyntaxError, ValueError) as e:
            logger.debug(f"Initial parse failed with ast.literal_eval: {e}")
            # Fallback to regex-based cleaning
            response = re.sub(r"\'(\w+)\'(\s*:\s*)", r'"\1"\2', response)  # Convert single-quoted keys
            response = re.sub(r"\'(.*?)\'(?=(?:,|\]|\s*}))", r'"\1"', response)  # Convert single-quoted values
            response = re.sub(r'(}\s*{)', r'}, {', response)  # Insert missing commas between objects
            response = re.sub(r'\s+', ' ', response).strip()  # Normalize whitespace
            data = ast.literal_eval(response)

        # Step 3: Convert to standard JSON
        json_data = json.dumps(data)

        # Step 4: Validate against schema
        schema = {
            "type": "object",
            "properties": {
                "analyses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "recommended_actions": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["recommended_actions"]
                    }
                }
            },
            "required": ["analyses"]
        }
        validate(instance=data, schema=schema)
        return True, json_data
    except (SyntaxError, ValueError, TypeError, ValidationError) as e:
        logger.error(f"Validation failed: {e} for raw response: {repr(response)}")
        return False, f"Invalid JSON format: {e}"

# Extract recommended actions from LLM response
def extract_recommended_actions(response: str, ingredients, valid_rows):
    is_valid, json_data_or_error = validate_response(response)
    logger.info(f"Validation result: {json_data_or_error if is_valid else 'Failed - ' + json_data_or_error}")
    if not is_valid:
        logger.error(f"Raw invalid response: {response}")
        return {ingr: [f"Fallback: Consult inventory manager (Error: {json_data_or_error}, Raw: {response[:100]}...)"] for ingr in ingredients}

    try:
        data = json.loads(json_data_or_error)
        analyses = data.get("analyses", [])
        if len(analyses) != len(ingredients):
            logger.warning(f"Mismatch: Expected {len(ingredients)} analyses, got {len(analyses)}. Using available data.")
        return {
            ingr: [action for action in (analyses[i].get("recommended_actions", []) if i < len(analyses) else [])]
            for i, ingr in enumerate(ingredients)
        }
    except (ValueError, KeyError) as e:
        logger.error(f"Extraction failed: {e}")
        return {ingr: [f"Fallback: Consult inventory manager (Error: {e})"] for ingr in ingredients}

# Process a single batch
def process_batch(batch, batch_ingredients, valid_rows, system_prompt):
    user_input = "Analyze the following ingredients for waste prevention:\n" + "\n".join(
        f"Ingredient: {row['ingredient']}, Stock: {row['stock_kg']}kg, "
        f"Days Since Delivery: {row['days_since_delivery']}, "
        f"Shelf Life (Days): {row['shelf_life_days']}, "
        f"Storage Temp: {row['storage_temp_c']}°C, "
        f"Weekly Usage: {row['weekly_usage_kg']}kg"
        for row in batch
    )

    try:
        response = invoke_llm_with_retry(llm, [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ])
        logger.info(f"Raw response for batch {batch[0]['ingredient'][:10]}...: {response.content[:100]}...")

        recommended_actions_dict = extract_recommended_actions(response.content, batch_ingredients, valid_rows)
        batch_results = [
            {
                "ingredient": row["ingredient"],
                "stock_kg": row["stock_kg"],
                "days_since_delivery": row["days_since_delivery"],
                "shelf_life_days": row["shelf_life_days"],
                "storage_temp_c": row["storage_temp_c"],
                "weekly_usage_kg": row["weekly_usage_kg"],
                "suggested_replenishment_kg": row["suggested_replenishment_kg"],
                "spoilage_risk": calculate_spoilage_risk(row["days_since_delivery"], row["shelf_life_days"]),
                "overuse_risk": calculate_overuse_risk(row["stock_kg"], row["weekly_usage_kg"], row["shelf_life_days"]),
                "recommended_actions": recommended_actions_dict.get(row["ingredient"], ["No actions available"])
            }
            for row in batch
        ]
        return batch_results
    except Exception as e:
        logger.error(f"Error processing batch starting with {batch[0]['ingredient']}: {e}")
        return [
            {
                "ingredient": row["ingredient"],
                "stock_kg": row["stock_kg"],
                "days_since_delivery": row["days_since_delivery"],
                "shelf_life_days": row["shelf_life_days"],
                "storage_temp_c": row["storage_temp_c"],
                "weekly_usage_kg": row["weekly_usage_kg"],
                "suggested_replenishment_kg": row["suggested_replenishment_kg"],
                "spoilage_risk": "Error: Batch processing failed",
                "overuse_risk": "Error: Batch processing failed",
                "recommended_actions": ["Fallback: Consult inventory manager"]
            }
            for row in batch
        ]

# Heuristic calculations
def calculate_spoilage_risk(days_since_delivery: int, shelf_life_days: int) -> float:
    """Heuristically calculate spoilage risk based on days since delivery and shelf life."""
    days_past = days_since_delivery - shelf_life_days
    if days_past <= 0:
        return 0.0
    elif 0 < days_past <= 2:
        return 0.5
    elif 2 < days_past <= 5:
        return 0.8
    else:
        return 1.0

def calculate_overuse_risk(stock_kg: float, weekly_usage_kg: float, shelf_life_days: int) -> float:
    """Heuristically calculate overuse risk based on stock duration."""
    stock_duration = (stock_kg / (weekly_usage_kg / 7)) if weekly_usage_kg > 0 else float('inf')
    if stock_duration >= shelf_life_days:
        return 0.0
    elif stock_duration >= 7:
        return 0.3
    elif stock_duration >= 3:
        return 0.7
    else:
        return 1.0

# Predict waste and suggest replenishment for ingredients
def predict_waste(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("Process interrupted by user. Saving partial results...")
        partial_df = pd.DataFrame(results)
        partial_df.to_csv("partial_waste_predictions.csv", index=False)
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    start_time = time.time()  # Track execution time

    # Calculate days_since_delivery dynamically using today's date if not provided
    current_date = datetime.now()
    if 'days_since_delivery' not in df.columns:
        if 'delivery_date' not in df.columns:
            raise ValueError("Either 'days_since_delivery' or 'delivery_date' must be provided in the DataFrame.")
        df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
        df['days_since_delivery'] = (current_date - df['delivery_date']).dt.days

    # Validate input data and build batch input
    valid_rows = []
    ingredients = []
    for _, row in df.iterrows():
        required_fields = ['ingredient', 'stock_kg', 'days_since_delivery', 'shelf_life_days', 'storage_temp_c', 'weekly_usage_kg']
        if any(pd.isna(row[field]) for field in required_fields):
            logger.warning(f"Missing data for {row['ingredient']}. Skipping incomplete row.")
            results.append({
                "ingredient": row["ingredient"],
                "stock_kg": None if pd.isna(row["stock_kg"]) else row["stock_kg"],
                "days_since_delivery": None if pd.isna(row["days_since_delivery"]) else row["days_since_delivery"],
                "shelf_life_days": None if pd.isna(row["shelf_life_days"]) else row["shelf_life_days"],
                "storage_temp_c": None if pd.isna(row["storage_temp_c"]) else row["storage_temp_c"],
                "weekly_usage_kg": None if pd.isna(row["weekly_usage_kg"]) else row["weekly_usage_kg"],
                "suggested_replenishment_kg": 0.0,
                "spoilage_risk": "Error: Missing data",
                "overuse_risk": "Error: Missing data",
                "recommended_actions": []
            })
            continue

        buffer_storage = 0.10 * row['weekly_usage_kg']
        suggested_replenishment_kg = max(0, (row['weekly_usage_kg'] + buffer_storage) - row['stock_kg'])
        valid_rows.append({
            "ingredient": row["ingredient"],
            "stock_kg": row["stock_kg"],
            "days_since_delivery": row["days_since_delivery"],
            "shelf_life_days": row["shelf_life_days"],
            "storage_temp_c": row["storage_temp_c"],
            "weekly_usage_kg": row["weekly_usage_kg"],
            "suggested_replenishment_kg": round(suggested_replenishment_kg, 2)
        })
        ingredients.append(row["ingredient"])

    if not valid_rows:
        logger.error("No valid data to process.")
        return pd.DataFrame(results)

    # Define system prompt
    system_prompt = (
        "You are an AI kitchen assistant for a restaurant aiming to minimize waste and optimize inventory. "
        "For each ingredient provided in the input, provide a list of recommended actions to prevent spoilage or overuse. "
        "Actions should be detailed and actionable, considering the ingredient type and current conditions (e.g., 'Use spinach in salads within 2 days', 'Adjust storage to 2-4°C for fish', 'Prioritize chicken in tomorrow's menu'). "
        "Return your response as a VALID JSON object with a single key 'analyses' containing a list of objects, where each object has a 'recommended_actions' key with a list of strings. "
        "Ensure all keys (e.g., 'analyses', 'recommended_actions') are followed by a colon ':' and their values. The order of analyses must match the order of ingredients in the input. "
        "Do not include spoilage_risk, overuse_risk, confidence_score, or reasoning. Do not include any additional text or markdown outside the JSON.\n"
        "Example: For inputs: 'Ingredient: tomato, Stock: 5kg, Days Since Delivery: 12, Shelf Life: 10, Storage Temp: 4°C, Weekly Usage: 7kg' and 'Ingredient: eggplant, Stock: 2kg, Days Since Delivery: 13, Shelf Life: 10, Storage Temp: 6°C, Weekly Usage: 3kg'\n"
        "{\n"
        "  'analyses': [\n"
        "    {\n"
        "      'recommended_actions': ['Replenish stock by 2.2kg', 'Use in next menu', 'Monitor storage temperature']\n"
        "    },\n"
        "    {\n"
        "      'recommended_actions': ['Replenish stock by 1.7kg', 'Discard if spoiled', 'Adjust storage temp to 2-4°C']\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    # Process batches in parallel
    batch_size = 5
    batches = [valid_rows[i:i + batch_size] for i in range(0, len(valid_rows), batch_size)]
    batch_ingredients = [ingredients[i:i + batch_size] for i in range(0, len(ingredients), batch_size)]

    with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
        future_to_batch = {
            executor.submit(process_batch, batch, batch_ingr, valid_rows, system_prompt): (batch, batch_ingr)
            for batch, batch_ingr in zip(batches, batch_ingredients)
        }
        for future in as_completed(future_to_batch):
            batch_results = future.result()
            results.extend(batch_results)
            time.sleep(1)  # Rate limiting delay between batches

    result_df = pd.DataFrame(results)
    pd.set_option('display.max_colwidth', 200)
    pd.set_option('display.width', 200)

    # Return DataFrame without saving or printing (handled by main.py)
    return result_df