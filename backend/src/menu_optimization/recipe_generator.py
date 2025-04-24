# src/recipe_generator.py

import pandas as pd
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime
import json
import re
import yaml

class RecipeGenerator:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load environment variables
        load_dotenv()

        # Initialize the Groq model
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="Llama-3.1-8b-Instant",
            timeout=30
        )

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(), logging.FileHandler('recipe_generator.log')]
        )
        self.logger = logging.getLogger(__name__)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _invoke_llm_with_retry(self, messages):
        try:
            self.logger.info("Attempting API call with messages:")
            response = self.llm.invoke(messages)
            self.logger.info(f"Generated recipe response: {response.content[:100]}...")
            return response.content
        except TypeError as e:
            self.logger.error(f"TypeError in API call: {str(e)}")
            self.logger.error(f"Messages type: {type(messages)}")
            self.logger.error(f"Messages content: {messages}")
            raise
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            raise

    def generate_recipes(self):
        """Generate recipes based on inventory data"""
        try:
            # Load inventory data
            inventory_df = pd.read_csv(self.config['data']['inventory_path'])
            # print(inventory_df.head())
            # Validate input DataFrame
            required_columns = ['ingredient', 'stock_kg', 'shelf_life_days', 'weekly_usage_kg', 'delivery_date']
            if not all(col in inventory_df.columns for col in required_columns):
                self.logger.error(f"Inventory DataFrame missing required columns: {required_columns}")
                print("Error: Inventory data is missing required columns.")
                return None

            # Calculate days since delivery
            current_date = datetime.now()
            inventory_df['days_since_delivery'] = (current_date - pd.to_datetime(inventory_df['delivery_date'])).dt.days

            # Filter ingredients with surplus or near expiry
            surplus_ingredients = inventory_df[
                (inventory_df['stock_kg'] > inventory_df['weekly_usage_kg'] * 0.5) |  # Surplus
                (inventory_df['days_since_delivery'] > inventory_df['shelf_life_days'] * 0.8)  # Near expiry
            ]

            # Add debug logging
            self.logger.info("Inventory analysis:")
            for _, row in inventory_df.iterrows():
                surplus_condition = row['stock_kg'] > row['weekly_usage_kg'] * 0.5
                expiry_condition = row['days_since_delivery'] > row['shelf_life_days'] * 0.8
                self.logger.info(f"Ingredient: {row['ingredient']}")
                self.logger.info(f"  - Stock: {row['stock_kg']}kg, Weekly Usage: {row['weekly_usage_kg']}kg")
                self.logger.info(f"  - Days since delivery: {row['days_since_delivery']}, Shelf life: {row['shelf_life_days']} days")
                self.logger.info(f"  - Meets surplus condition: {surplus_condition}")
                self.logger.info(f"  - Meets expiry condition: {expiry_condition}")

            if surplus_ingredients.empty:
                self.logger.warning("No surplus or near-expiry ingredients found.")
                print("Warning: No surplus or near-expiry ingredients found to generate recipes.")
                return None

            # Prepare prompt for LLM
            ingredients_list = "\n".join(
                f"- {row['ingredient']} ({row['stock_kg']}kg, {row['days_since_delivery']} days since delivery, shelf life: {row['shelf_life_days']} days)"
                for _, row in surplus_ingredients.iterrows()
            )
            # print(ingredients_list)
            prompt = f"""
            You are a culinary AI assistant. Based on the following inventory of surplus or near-expiry ingredients, generate creative dish ideas. 
            Prioritize using ingredients that are nearing expiry or in excess. Provide:
            1. A dish name possibly indian dish.
            2. A brief detailed of the recipe.
            3. A list of required ingredients (including quantities if possible).
            4. Reasoning for the suggestion (e.g., why the dish uses specific ingredients).

            Inventory:
            {ingredients_list}

            Return ONLY the JSON object with a key 'recipes' containing a list of objects, each with keys: dish_name, description, ingredients, reasoning. Use double quotes for keys and values. Do NOT include any additional text, Markdown, or notes outside the JSON.
            Example:
            {{
                "recipes": [
                    {{
                        "dish_name": "Tomato Basil Soup",
                        "description": "A warm soup using surplus tomatoes.",
                        "ingredients": ["5kg tomatoes", "0.5kg basil", "1L water"],
                        "reasoning": "Uses 5kg of tomatoes nearing expiry (12 days past shelf life) and 0.5kg of basil in stock."
                    }}
                ]
            }}
            """
            # print(prompt)
            
            response = self._invoke_llm_with_retry([
                {"role": "system", "content": "You are a culinary AI assistant."},
                {"role": "user", "content": prompt}
            ])
            # print(response)
            # Extract and clean JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON content detected", response, 0)
            
            json_str = json_match.group(0).strip()
            recipes_data = json.loads(json_str)
            recipes = recipes_data.get("recipes", [])
            # print(recipes_data)
            # Output recipes to terminal
            if recipes:
                print("\nGenerated Recipes:")
                for i, recipe in enumerate(recipes, 1):
                    print(f"\nRecipe {i}:")
                    print(f"Dish Name: {recipe.get('dish_name', 'N/A')}")
                    print(f"Description: {recipe.get('description', 'N/A')}")
                    print(f"Ingredients: {', '.join(recipe.get('ingredients', ['N/A']))}")
                    print(f"Reasoning: {recipe.get('reasoning', 'N/A')}")
                return recipes
            else:
                print("No recipes generated from the response.")
                self.logger.warning("No recipes found in the LLM response.")
                return None

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}. Raw response: {response}")
            print(f"Error: Failed to parse recipe data due to invalid JSON format: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error generating recipes: {e}")
            print(f"Error: An unexpected error occurred while generating recipes: {e}")
            return None

if __name__ == "__main__":
    # This block will not be executed when called from main.py
    pass