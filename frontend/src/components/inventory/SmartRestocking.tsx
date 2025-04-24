import React, { useState, useEffect } from 'react';
import { getRestockingRecommendations } from '@/services/api';
import { Loader2, RefreshCw, ArrowUpDown, ShoppingCart } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

interface RestockingItem {
  ingredient: string;
  stock_kg: number;
  suggested_replenishment_kg: number;
  // Other fields are still in the interface but not displayed prominently
  days_since_delivery: number;
  shelf_life_days: number;
  storage_temp_c: number;
  weekly_usage_kg: number;
  spoilage_risk: number;
  overuse_risk: number;
  recommended_actions: string[];
}

const SmartRestocking = () => {
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<RestockingItem[]>([]);
  const [sortField, setSortField] = useState<'ingredient' | 'suggested_replenishment_kg'>('suggested_replenishment_kg');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // Auto-fetch recommendations on component mount
  useEffect(() => {
    fetchRecommendations();
  }, []);

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const response = await getRestockingRecommendations();
      if (response.status === 'success' && response.predictions) {
        setRecommendations(response.predictions);
        toast.success('Restocking recommendations loaded successfully');
      }
    } catch (error) {
      console.error('Error fetching restocking recommendations:', error);
      // Toast error is already handled in the API service
    } finally {
      setLoading(false);
    }
  };

  const handleSort = (field: 'ingredient' | 'suggested_replenishment_kg') => {
    if (field === sortField) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc'); // Default to descending for new field
    }
  };

  // Filter to only include items that need restocking
  const needsRestocking = recommendations.filter(item => item.suggested_replenishment_kg > 0);
  
  // Sort the filtered recommendations
  const sortedRecommendations = [...needsRestocking].sort((a, b) => {
    if (sortField === 'ingredient') {
      return sortDirection === 'asc' 
        ? a.ingredient.localeCompare(b.ingredient) 
        : b.ingredient.localeCompare(a.ingredient);
    } else {
      return sortDirection === 'asc' 
        ? a.suggested_replenishment_kg - b.suggested_replenishment_kg 
        : b.suggested_replenishment_kg - a.suggested_replenishment_kg;
    }
  });

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-100">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-800">Smart Restocking Recommendations</h2>
          <p className="text-sm text-gray-500">Items that need to be restocked</p>
        </div>
        <Button
          onClick={fetchRecommendations}
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
          <p className="text-lg text-gray-600">Loading recommendations...</p>
        </div>
      ) : sortedRecommendations.length > 0 ? (
        <div className="overflow-auto">
          <table className="w-full text-sm text-left text-gray-500">
            <thead className="text-xs text-gray-700 uppercase bg-gray-50">
              <tr>
                <th 
                  className="px-6 py-3 cursor-pointer"
                  onClick={() => handleSort('ingredient')}
                >
                  <div className="flex items-center">
                    Ingredient
                    {sortField === 'ingredient' && (
                      <ArrowUpDown size={14} className="ml-1" />
                    )}
                  </div>
                </th>
                <th 
                  className="px-6 py-3 cursor-pointer"
                  onClick={() => handleSort('suggested_replenishment_kg')}
                >
                  <div className="flex items-center">
                    Suggested Restock (kg)
                    {sortField === 'suggested_replenishment_kg' && (
                      <ArrowUpDown size={14} className="ml-1" />
                    )}
                  </div>
                </th>
                <th className="px-6 py-3">Action</th>
              </tr>
            </thead>
            <tbody>
              {sortedRecommendations.map((item, index) => (
                <tr 
                  key={`${item.ingredient}-${index}`} 
                  className="bg-white border-b hover:bg-gray-50"
                >
                  <td className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap">
                    {item.ingredient.replace(/_/g, ' ')}
                  </td>
                  <td className="px-6 py-4 font-medium text-blue-600">
                    {item.suggested_replenishment_kg.toFixed(1)}
                  </td>
                  <td className="px-6 py-4">
                    <Button
                      variant="outline" 
                      size="sm"
                      className="text-restaurant-primary border-restaurant-primary/30 hover:bg-restaurant-primary/10"
                    >
                      <ShoppingCart size={14} className="mr-1" />
                      Add to Order
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <p className="text-lg text-gray-600">
            Good news! No items need to be restocked at this time.
          </p>
          <p className="text-sm text-gray-500 mt-2">
            All your inventory levels are sufficient based on your usage patterns.
          </p>
        </div>
      )}
    </div>
  );
};

export default SmartRestocking; 