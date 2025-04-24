import os
import argparse
from menu_optimization.recipe_generator import RecipeGenerator
from food_spoilage_detection.food_spoilage_detection import FoodSpoilageDetector
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('unified_project.log')
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Unified Project - Recipe Generation and Food Spoilage Detection')
    parser.add_argument('--mode', choices=['recipe', 'spoilage', 'both'], default='both',
                      help='Operation mode: recipe generation, spoilage detection, or both')
    parser.add_argument('--image-path', type=str, 
                      default='data/raw/fresh_rotten_dataset/rotten-strawberry.jpg',
                      help='Path to image for spoilage detection')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("Starting Unified Project")

    try:
        # Initialize components
        recipe_generator = RecipeGenerator("src/menu_optimization/config.yaml")
        spoilage_detector = FoodSpoilageDetector("src/food_spoilage_detection/config.yaml")

        # Run recipe generation if requested
        if args.mode in ['recipe', 'both']:
            logger.info("Starting recipe generation")
            recipes = recipe_generator.generate_recipes()
            if recipes:
                logger.info(f"Generated {len(recipes)} recipes successfully")
            else:
                logger.warning("No recipes were generated")

        # Run spoilage detection if requested
        if args.mode in ['spoilage', 'both']:
            # Check if image exists
            if not os.path.exists(args.image_path):
                logger.error(f"Image not found at path: {args.image_path}")
                logger.info("Available sample images in data/raw/fresh_rotten_dataset/:")
                sample_dir = "data/raw/fresh_rotten_dataset"
                if os.path.exists(sample_dir):
                    for img in os.listdir(sample_dir):
                        if img.endswith(('.jpg', '.jpeg', '.png')):
                            logger.info(f"  - {img}")
                return
            
            logger.info(f"Starting spoilage detection for image: {args.image_path}")
            result = spoilage_detector.detect_spoilage(args.image_path)
            logger.info(f"Spoilage detection result: {result}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 