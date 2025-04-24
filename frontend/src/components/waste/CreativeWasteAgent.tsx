import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { Loader2, Utensils, RefreshCw } from 'lucide-react';
import { getRecipeGeneration } from '@/services/api';

interface Recipe {
  dish_name: string;
  description: string;
  ingredients: string[];
}

const CreativeWasteAgent = () => {
  const [loading, setLoading] = useState(false);
  const [recipes, setRecipes] = useState<Recipe[]>([]);

  // Fetch recipes when component mounts
  useEffect(() => {
    fetchRecipes();
  }, []);

  const fetchRecipes = async () => {
    setLoading(true);
    try {
      const response = await getRecipeGeneration();
      if (response.status === 'success' && response.recipes) {
        setRecipes(response.recipes);
        toast.success('Recipes loaded successfully');
      } else {
        toast.error('Failed to load recipes');
      }
    } catch (error) {
      console.error('Error fetching recipes:', error);
      toast.error('Error fetching recipes');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-100">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-800">Creative Waste Agent</h2>
          <p className="text-sm text-gray-500">Recipe suggestions for reducing food waste</p>
        </div>
        <Button
          onClick={fetchRecipes}
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
              Refresh Recipes
            </>
          )}
        </Button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 size={30} className="animate-spin text-restaurant-primary mr-3" />
          <p className="text-lg text-gray-600">Loading recipe suggestions...</p>
        </div>
      ) : recipes.length > 0 ? (
        <div className="grid md:grid-cols-2 gap-6">
          {recipes.map((recipe, index) => (
            <div key={index} className="bg-gray-50 rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-2 text-gray-800 flex items-center">
                <Utensils size={20} className="mr-2 text-restaurant-primary" />
                {recipe.dish_name}
              </h3>
              
              <p className="text-sm text-gray-600 mb-4">{recipe.description}</p>
              
              <div className="bg-white p-4 rounded-md shadow-sm">
                <h4 className="font-medium text-gray-700 mb-2">Ingredients</h4>
                <ul className="list-disc pl-5 space-y-1">
                  {recipe.ingredients.map((ingredient, idx) => (
                    <li key={idx} className="text-sm text-gray-600">{ingredient}</li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <Utensils size={48} className="mx-auto text-gray-300 mb-4" />
          <p className="text-lg text-gray-600">
            No recipes available
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Click "Refresh Recipes" to fetch recipe suggestions
          </p>
        </div>
      )}
    </div>
  );
};

export default CreativeWasteAgent; 