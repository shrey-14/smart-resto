import React, { useState, useEffect } from 'react';
import { getRecipeRecommendation } from '@/services/api';
import { Loader2, RefreshCw, Utensils, Clock, ChefHat } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

interface RecipeDetails {
  description: string;
  cuisine: string;
  prep_time_minutes: number;
  serving_size: number;
}

interface BackendRecipe {
  recipe_name: string;
  details: RecipeDetails;
  ingredients?: string[]; // May not be present in backend response
}

interface Recipe {
  dish_name: string;
  description: string;
  preparation_time?: string;
  difficulty?: string;
  calories?: number;
  ingredients?: string[];
}

const DailySpecials = () => {
  const [loading, setLoading] = useState(false);
  const [recipe, setRecipe] = useState<Recipe | null>(null);
  const [date, setDate] = useState<string>('');

  // Fetch recipe recommendation when component mounts
  useEffect(() => {
    fetchRecipeRecommendation();
    // Set current date
    const today = new Date();
    setDate(today.toLocaleDateString('en-US', { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    }));
  }, []);

  const fetchRecipeRecommendation = async () => {
    setLoading(true);
    try {
      const response = await getRecipeRecommendation();
      console.log('Recipe response:', response); // Debug log
      
      if (response.status === 'success' && response.recommended_recipe) {
        const backendRecipe = response.recommended_recipe as BackendRecipe;
        
        // Transform backend response to match our component's expected structure
        const transformedRecipe: Recipe = {
          dish_name: backendRecipe.recipe_name,
          description: backendRecipe.details.description,
          preparation_time: backendRecipe.details.prep_time_minutes 
            ? `${backendRecipe.details.prep_time_minutes} minutes` 
            : undefined,
          difficulty: backendRecipe.details.cuisine || undefined,
          ingredients: backendRecipe.ingredients || []
        };
        
        setRecipe(transformedRecipe);
        toast.success('Daily special loaded successfully');
      } else {
        toast.error('Failed to load daily special');
      }
    } catch (error) {
      console.error('Error fetching recipe recommendation:', error);
      toast.error('Error fetching daily special');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-100">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-800">Today's Special</h2>
          <p className="text-sm text-gray-500">{date}</p>
        </div>
        <Button
          onClick={fetchRecipeRecommendation}
          disabled={loading}
          className="bg-restaurant-primary hover:bg-restaurant-primary/90 text-white"
        >
          {loading ? (
            <>
              <Loader2 size={16} className="mr-2 animate-spin" />
              Loading...
            </>
          ) : (
            <>
              <RefreshCw size={16} className="mr-2" />
              Refresh
            </>
          )}
        </Button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 size={30} className="animate-spin text-restaurant-primary mr-3" />
          <p className="text-lg text-gray-600">Loading today's special...</p>
        </div>
      ) : recipe ? (
        <div className="animate-fade-in">
          <div className="mb-6 pb-6 border-b border-gray-100">
            <h3 className="text-2xl font-bold text-restaurant-primary flex items-center">
              <Utensils size={24} className="mr-3" />
              {recipe.dish_name}
            </h3>
            
            <div className="mt-4 text-gray-700">
              <p>{recipe.description}</p>
            </div>
            
            {(recipe.preparation_time || recipe.difficulty || recipe.calories) && (
              <div className="mt-4 flex flex-wrap gap-4">
                {recipe.preparation_time && (
                  <div className="flex items-center text-sm text-gray-600">
                    <Clock size={16} className="mr-1" />
                    <span>Prep time: {recipe.preparation_time}</span>
                  </div>
                )}
                
                {recipe.difficulty && (
                  <div className="flex items-center text-sm text-gray-600">
                    <ChefHat size={16} className="mr-1" />
                    <span>Cuisine type: {recipe.difficulty}</span>
                  </div>
                )}
                
                {recipe.calories && (
                  <div className="flex items-center text-sm text-gray-600">
                    <span className="font-medium">{recipe.calories} cal</span>
                  </div>
                )}
              </div>
            )}
          </div>
          
          <div>
            <h4 className="text-lg font-semibold text-gray-800 mb-3">Ingredients</h4>
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {recipe.ingredients && recipe.ingredients.length > 0 ? (
                recipe.ingredients.map((ingredient, index) => (
                  <li key={index} className="flex items-center text-gray-700">
                    <div className="w-2 h-2 bg-restaurant-primary rounded-full mr-2"></div>
                    {ingredient}
                  </li>
                ))
              ) : (
                <li className="text-gray-600">No ingredients information available</li>
              )}
            </ul>
          </div>
        </div>
      ) : (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <Utensils size={48} className="mx-auto text-gray-300 mb-4" />
          <p className="text-lg text-gray-600">
            No daily special available
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Click "Refresh" to fetch today's special recommendation
          </p>
        </div>
      )}
    </div>
  );
};

export default DailySpecials; 