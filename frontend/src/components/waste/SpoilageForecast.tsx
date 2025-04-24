import React, { useState, useEffect } from 'react';
import { getRestockingRecommendations } from '@/services/api';
import { Loader2, RefreshCw, ArrowUpDown, AlertTriangle, AlertCircle, Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

interface SpoilageItem {
  ingredient: string;
  spoilage_risk: number;
  overuse_risk: number;
  recommended_actions: string[];
  // Other fields might exist in the API response but we're focusing on these
  stock_kg?: number;
  days_since_delivery?: number;
  shelf_life_days?: number;
  storage_temp_c?: number;
  weekly_usage_kg?: number;
}

const SpoilageForecast = () => {
  const [loading, setLoading] = useState(false);
  const [items, setItems] = useState<SpoilageItem[]>([]);
  const [sortField, setSortField] = useState<'ingredient' | 'spoilage_risk' | 'overuse_risk'>('spoilage_risk');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // Auto-fetch data on component mount
  useEffect(() => {
    fetchSpoilageData();
  }, []);

  const fetchSpoilageData = async () => {
    setLoading(true);
    try {
      // This uses the same endpoint as smart restocking but we'll display different fields
      const response = await getRestockingRecommendations();
      if (response.status === 'success' && response.predictions) {
        setItems(response.predictions);
        toast.success('Spoilage forecast loaded successfully');
      }
    } catch (error) {
      console.error('Error fetching spoilage forecast data:', error);
      // Toast error is already handled in the API service
    } finally {
      setLoading(false);
    }
  };

  const handleSort = (field: 'ingredient' | 'spoilage_risk' | 'overuse_risk') => {
    if (field === sortField) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc'); // Default to descending for new field
    }
  };

  // Filter to only include items with some risk
  const hasRisk = items.filter(item => item.spoilage_risk > 0 || item.overuse_risk > 0);
  
  // Sort the filtered items
  const sortedItems = [...hasRisk].sort((a, b) => {
    if (sortField === 'ingredient') {
      return sortDirection === 'asc' 
        ? a.ingredient.localeCompare(b.ingredient) 
        : b.ingredient.localeCompare(a.ingredient);
    } else if (sortField === 'spoilage_risk') {
      return sortDirection === 'asc' 
        ? a.spoilage_risk - b.spoilage_risk 
        : b.spoilage_risk - a.spoilage_risk;
    } else {
      return sortDirection === 'asc' 
        ? a.overuse_risk - b.overuse_risk 
        : b.overuse_risk - a.overuse_risk;
    }
  });

  // Helper function to get color class based on risk level
  const getRiskColorClass = (risk: number) => {
    if (risk >= 0.8) return 'text-red-600';
    if (risk >= 0.5) return 'text-amber-500';
    if (risk >= 0.3) return 'text-yellow-500';
    return 'text-green-600';
  };

  // Helper function to format risk level as percentage
  const formatRiskLevel = (risk: number) => {
    return `${(risk * 100).toFixed(0)}%`;
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-100">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-800">Spoilage Forecast</h2>
          <p className="text-sm text-gray-500">Predict which items may spoil soon based on inventory age</p>
        </div>
        <Button
          onClick={fetchSpoilageData}
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
          <p className="text-lg text-gray-600">Forecasting potential spoilage risks...</p>
        </div>
      ) : sortedItems.length > 0 ? (
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
                  onClick={() => handleSort('spoilage_risk')}
                >
                  <div className="flex items-center">
                    Spoilage Risk
                    {sortField === 'spoilage_risk' && (
                      <ArrowUpDown size={14} className="ml-1" />
                    )}
                  </div>
                </th>
                <th 
                  className="px-6 py-3 cursor-pointer"
                  onClick={() => handleSort('overuse_risk')}
                >
                  <div className="flex items-center">
                    Overuse Risk
                    {sortField === 'overuse_risk' && (
                      <ArrowUpDown size={14} className="ml-1" />
                    )}
                  </div>
                </th>
                <th className="px-6 py-3">
                  <div className="flex items-center">
                    Recommended Actions
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedItems.map((item, index) => (
                <tr 
                  key={`${item.ingredient}-${index}`} 
                  className="bg-white border-b hover:bg-gray-50"
                >
                  <td className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap">
                    {item.ingredient.replace(/_/g, ' ')}
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center">
                      {item.spoilage_risk >= 0.5 && (
                        <AlertTriangle size={16} className={`mr-2 ${getRiskColorClass(item.spoilage_risk)}`} />
                      )}
                      <span className={`font-medium ${getRiskColorClass(item.spoilage_risk)}`}>
                        {formatRiskLevel(item.spoilage_risk)}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center">
                      {item.overuse_risk >= 0.5 && (
                        <AlertCircle size={16} className={`mr-2 ${getRiskColorClass(item.overuse_risk)}`} />
                      )}
                      <span className={`font-medium ${getRiskColorClass(item.overuse_risk)}`}>
                        {formatRiskLevel(item.overuse_risk)}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    {item.recommended_actions && item.recommended_actions.length > 0 ? (
                      <ul className="list-disc list-inside text-gray-700">
                        {item.recommended_actions.map((action, idx) => (
                          <li key={idx}>{action}</li>
                        ))}
                      </ul>
                    ) : (
                      <span className="text-gray-500 italic">No actions needed</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <p className="text-lg text-gray-600">
            Good news! No items forecasted to spoil soon.
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Your inventory is in good condition with no detected spoilage risks.
          </p>
        </div>
      )}
    </div>
  );
};

export default SpoilageForecast; 