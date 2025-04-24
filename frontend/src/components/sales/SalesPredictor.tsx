import React, { useState, useEffect } from 'react';
import { getSalesForecast } from '@/services/api';
import { Loader2, RefreshCw, ArrowUpDown, TrendingUp, Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { format } from 'date-fns';

// Interface matching the API response format
interface ForecastItem {
  date: string;
  item: string;
  predicted_quantity: number;
}

const SalesPredictor = () => {
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState<ForecastItem[]>([]);
  const [sortField, setSortField] = useState<'date' | 'item' | 'predicted_quantity'>('date');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  // Auto-fetch forecasts on component mount
  useEffect(() => {
    fetchSalesForecast();
  }, []);

  const fetchSalesForecast = async () => {
    setLoading(true);
    try {
      const response = await getSalesForecast();
      console.log('API Response:', response); // Debug log
      
      if (response.status === 'success' && response.future_predictions) {
        setPredictions(response.future_predictions);
        toast.success('Sales forecast loaded successfully');
      } else {
        toast.error('Failed to load sales forecast data');
      }
    } catch (error) {
      console.error('Error fetching sales forecast:', error);
      toast.error('Error loading sales forecast');
    } finally {
      setLoading(false);
    }
  };

  const handleSort = (field: 'date' | 'item' | 'predicted_quantity') => {
    if (field === sortField) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection(field === 'date' ? 'asc' : 'desc'); // Default sorting
    }
  };

  // Sort the predictions based on selected field and direction
  const sortedPredictions = [...predictions].sort((a, b) => {
    if (sortField === 'date') {
      return sortDirection === 'asc' 
        ? new Date(a.date).getTime() - new Date(b.date).getTime() 
        : new Date(b.date).getTime() - new Date(a.date).getTime();
    } else if (sortField === 'item') {
      return sortDirection === 'asc' 
        ? a.item.localeCompare(b.item) 
        : b.item.localeCompare(a.item);
    } else {
      return sortDirection === 'asc' 
        ? a.predicted_quantity - b.predicted_quantity 
        : b.predicted_quantity - a.predicted_quantity;
    }
  });

  // Group predictions by date for better visualization
  const predictionsByDate = sortedPredictions.reduce((acc, item) => {
    const date = item.date.split('T')[0]; // Extract just the date part
    if (!acc[date]) {
      acc[date] = [];
    }
    acc[date].push(item);
    return acc;
  }, {} as Record<string, ForecastItem[]>);

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-100">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-800">Sales Forecast</h2>
          <p className="text-sm text-gray-500">Predicted sales for the next 7 days</p>
        </div>
        <Button
          onClick={fetchSalesForecast}
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
          <p className="text-lg text-gray-600">Loading forecast...</p>
        </div>
      ) : Object.keys(predictionsByDate).length > 0 ? (
        <div className="space-y-6">
          {Object.entries(predictionsByDate).map(([date, items]) => (
            <div key={date} className="border rounded-lg overflow-hidden">
              <div className="bg-gray-50 p-4 border-b flex items-center">
                <Calendar size={18} className="text-gray-600 mr-2" />
                <h3 className="font-medium text-gray-800">
                  {format(new Date(date), 'EEEE, MMMM d, yyyy')}
                </h3>
              </div>
              <div className="overflow-auto">
                <table className="w-full text-sm text-left text-gray-500">
                  <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                    <tr>
                      <th 
                        className="px-6 py-3 cursor-pointer"
                        onClick={() => handleSort('item')}
                      >
                        <div className="flex items-center">
                          Item
                          {sortField === 'item' && (
                            <ArrowUpDown size={14} className="ml-1" />
                          )}
                        </div>
                      </th>
                      <th 
                        className="px-6 py-3 cursor-pointer"
                        onClick={() => handleSort('predicted_quantity')}
                      >
                        <div className="flex items-center">
                          Predicted Sales
                          {sortField === 'predicted_quantity' && (
                            <ArrowUpDown size={14} className="ml-1" />
                          )}
                        </div>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((item, index) => (
                      <tr 
                        key={`${item.item}-${index}`} 
                        className="bg-white border-b hover:bg-gray-50"
                      >
                        <td className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap">
                          {item.item.replace(/_/g, ' ')}
                        </td>
                        <td className="px-6 py-4 font-medium text-blue-600">
                          <div className="flex items-center">
                            <TrendingUp size={16} className="mr-2" />
                            {item.predicted_quantity.toFixed(1)} units
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <p className="text-lg text-gray-600">
            No sales forecast data available.
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Try refreshing or upload your sales data to generate predictions.
          </p>
        </div>
      )}
    </div>
  );
};

export default SalesPredictor; 