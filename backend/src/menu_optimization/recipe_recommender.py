import pandas as pd
import yaml
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
import re
import json
# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LangChain Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="Llama-3.1-8b-Instant",  # Adjust if you want to try others
    timeout=30
)

# Retry LLM call with exponential backoff
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=15))
def invoke_llm_with_retry(messages):
    try:
        response = llm.invoke(messages)
        logger.debug(f"LLM Response: {response.content}")
        return response
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


class RecipeRecommender:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.inventory = pd.read_csv(self.config['data']['inventory_path'])
        self.recipes = pd.read_csv(self.config['data']['recipe_path'])
        self.threshold_days = self.config['recommendation']['expiration_threshold_days']

    def preprocess_inventory(self, current_date):
        self.inventory['delivery_date'] = pd.to_datetime(self.inventory['delivery_date'])
        self.inventory['expiration_date'] = self.inventory['delivery_date'] + pd.to_timedelta(self.inventory['shelf_life_days'], unit='D')
        current_date = pd.to_datetime(current_date)
        threshold_date = current_date + timedelta(days=self.threshold_days)
        self.inventory['surplus_kg'] = (self.inventory['stock_kg'] - self.inventory['weekly_usage_kg']).clip(lower=0)
        self.inventory['soon_to_expire'] = self.inventory['expiration_date'] <= threshold_date
        return self.inventory

    def get_recipe_details(self, recipe_name):
        if not recipe_name or recipe_name == "No suitable special found":
            return {
                "description": "No recipe recommended due to insufficient inventory.",
                "cuisine": "",
                "prep_time_minutes": 0,
                "serving_size": 0
            }

        recipe = self.recipes[self.recipes['recipe_name'] == recipe_name]
        ingredients = recipe[['ingredient', 'quantity', 'unit']].to_dict('records')
        ingredients_str = ", ".join([f"{row['quantity']} {row['unit']} {row['ingredient']}" for row in ingredients])

        prompt = f"""
        Provide concise details for the Indian dish "{recipe_name}" using these ingredients: {ingredients_str}.
        Return the response as a JSON object with the following fields:
        - description: A 2-3 sentence engaging summary of the dish.
        - cuisine: Type of cuisine (e.g., North Indian, South Indian, Indo-Chinese).
        - prep_time_minutes: Estimated preparation time.
        - serving_size: Number of people it can serve (assume 4-6 people).
        Only return valid JSON.
        """

        messages = [
            SystemMessage(content="You are a helpful culinary assistant for an Indian restaurant."),
            HumanMessage(content=prompt)
        ]

        try:
            response = invoke_llm_with_retry(messages)
            # Try parsing JSON from the content
            import json
            content = response.content
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            else:
                return {"description": content.strip(), "cuisine": "", "prep_time_minutes": 0, "serving_size": 0}
        except Exception as e:
            logger.error(f"Failed to generate description: {e}")
            return {
                "description": f"Could not fetch details for {recipe_name}.",
                "cuisine": "Unknown",
                "prep_time_minutes": 0,
                "serving_size": 0
            }

    def suggest_daily_special(self, current_date):
        inventory = self.preprocess_inventory(current_date)
        priority_items = inventory[(inventory['surplus_kg'] > 0) | (inventory['soon_to_expire'])]['ingredient'].tolist()
        logger.info(f"Priority items: {priority_items}")

        viable_recipes = []
        for recipe_name in self.recipes['recipe_name'].unique():
            recipe = self.recipes[self.recipes['recipe_name'] == recipe_name]
            can_make = True
            priority_used_kg = 0

            for _, row in recipe.iterrows():
                item = row['ingredient']
                needed = row['quantity']
                unit = row['unit']
                available = inventory[inventory['ingredient'] == item]['stock_kg'].values[0] if item in inventory['ingredient'].values else 0

                if unit == 'dozen':
                    needed_kg = needed * 0.6
                elif unit == 'litre':
                    needed_kg = needed * 1
                else:
                    needed_kg = needed

                if available < needed_kg:
                    can_make = False
                    break
                if item in priority_items:
                    surplus = inventory[inventory['ingredient'] == item]['surplus_kg'].values[0]
                    soon_to_expire = inventory[inventory['ingredient'] == item]['soon_to_expire'].values[0]
                    priority_used_kg += needed_kg if soon_to_expire or surplus > 0 else 0

            if can_make and any(row['ingredient'] in priority_items for _, row in recipe.iterrows()):
                viable_recipes.append({'name': recipe_name, 'priority_used_kg': priority_used_kg})

        if viable_recipes:
            best_recipe = max(viable_recipes, key=lambda x: x['priority_used_kg'])['name']
            details = self.get_recipe_details(best_recipe)
            return {
                "recipe_name": best_recipe,
                "details": details
            }
        return {
            "recipe_name": "No suitable special found",
            "details": self.get_recipe_details(None)
        }
