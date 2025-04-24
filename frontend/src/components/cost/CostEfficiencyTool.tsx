import React, { useState, useEffect } from 'react';
import { getCostOptimization } from '@/services/api';
import { Loader2, RefreshCw, ArrowUpDown } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

// API response interface
interface ApiMenuItem {
  dish_name: string;
  dish_cost_inr: number;
  suggested_price_inr: number;
  profit_margin: number;
  adjustment: string;
}

// Interface for displaying menu items in the component - exactly matching API fields
interface MenuItem {
  dish_name: string;
  dish_cost_inr: number;
  suggested_price_inr: number;
  profit_margin: number;
  adjustment: string;
}

const CostEfficiencyTool = () => {
  const [loading, setLoading] = useState(false);
  const [menuItems, setMenuItems] = useState<MenuItem[]>([]);
  const [sortField, setSortField] = useState<'dish_name' | 'dish_cost_inr' | 'suggested_price_inr' | 'profit_margin'>('dish_name');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [selectedItem, setSelectedItem] = useState<MenuItem | null>(null);

  // Auto-fetch cost optimization data on component mount
  useEffect(() => {
    fetchCostOptimization();
    
    // Direct fetch to validate CORS
    console.log('Directly validating API access...');
    fetch('http://0.0.0.0:8000/api/cost-optimization')
      .then(response => {
        console.log('Direct API connection test - Status:', response.status);
        return response.json();
      })
      .then(data => {
        console.log('Direct API connection test - Data:', data);
      })
      .catch(error => {
        console.error('Direct API connection test - Error:', error);
      });
  }, []);

  const fetchCostOptimization = async () => {
    setLoading(true);
    try {
      console.log('Fetching cost optimization data...');
      const response = await getCostOptimization();
      console.log('Cost Optimization Response:', response); // Debug log
      
      // Checking for data in multiple possible response formats
      const optimizedCostsData = response.optimized_costs || response.data || response.items || [];
      console.log('Processing optimization data:', optimizedCostsData);
      
      if (response.status === 'success' && optimizedCostsData) {
        // Handle various response formats - checking if it's an array
        const apiItems: ApiMenuItem[] = Array.isArray(optimizedCostsData) 
          ? optimizedCostsData 
          : optimizedCostsData.menu_items || [];
          
        // Map API data directly to display format - keeping only the original fields 
        // with no additional calculations or manipulations
        const menuItemsToDisplay: MenuItem[] = apiItems.map(item => ({
          dish_name: item.dish_name,
          dish_cost_inr: item.dish_cost_inr,
          suggested_price_inr: item.suggested_price_inr,
          profit_margin: item.profit_margin,
          adjustment: item.adjustment
        }));
          
        console.log('Menu items to display:', menuItemsToDisplay);
        
        setMenuItems(menuItemsToDisplay);
        toast.success('Cost optimization data loaded successfully');
      } else {
        console.error('Invalid API response format:', response);
        toast.error('Failed to load cost optimization data: Invalid response format');
        setMenuItems([]);
      }
    } catch (error) {
      console.error('Error fetching cost optimization data:', error);
      toast.error(`Error loading cost optimization data: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setMenuItems([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSort = (field: 'dish_name' | 'dish_cost_inr' | 'suggested_price_inr' | 'profit_margin') => {
    if (field === sortField) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection(field === 'dish_name' ? 'asc' : 'desc'); // Default sorting
    }
  };

  // Sort menu items based on the selected field and direction
  const sortedMenuItems = [...menuItems].sort((a, b) => {
    if (sortField === 'dish_name') {
      return sortDirection === 'asc' 
        ? a.dish_name.localeCompare(b.dish_name) 
        : b.dish_name.localeCompare(a.dish_name);
    } else if (sortField === 'dish_cost_inr') {
      return sortDirection === 'asc' 
        ? a.dish_cost_inr - b.dish_cost_inr 
        : b.dish_cost_inr - a.dish_cost_inr;
    } else if (sortField === 'suggested_price_inr') {
      return sortDirection === 'asc' 
        ? a.suggested_price_inr - b.suggested_price_inr 
        : b.suggested_price_inr - a.suggested_price_inr;
    } else {
      return sortDirection === 'asc' 
        ? a.profit_margin - b.profit_margin 
        : b.profit_margin - a.profit_margin;
    }
  });

  const handleItemClick = (item: MenuItem) => {
    setSelectedItem(item === selectedItem ? null : item);
  };

  return (
    <div className="space-y-6">
      {/* Overview Card */}
      <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-100">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-xl font-semibold text-gray-800">Cost Optimization Data</h2>
            <p className="text-sm text-gray-500">Direct data from cost optimization API</p>
          </div>
          <Button
            onClick={fetchCostOptimization}
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
            <p className="text-lg text-gray-600">Loading cost optimization data...</p>
          </div>
        ) : menuItems.length > 0 ? (
          <div className="overflow-auto">
            <table className="w-full text-sm text-left text-gray-500">
              <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                <tr>
                  <th 
                    className="px-6 py-3 cursor-pointer"
                    onClick={() => handleSort('dish_name')}
                  >
                    <div className="flex items-center">
                      Dish Name
                      {sortField === 'dish_name' && (
                        <ArrowUpDown size={14} className="ml-1" />
                      )}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-3 cursor-pointer"
                    onClick={() => handleSort('dish_cost_inr')}
                  >
                    <div className="flex items-center">
                      Dish Cost (INR)
                      {sortField === 'dish_cost_inr' && (
                        <ArrowUpDown size={14} className="ml-1" />
                      )}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-3 cursor-pointer"
                    onClick={() => handleSort('suggested_price_inr')}
                  >
                    <div className="flex items-center">
                      Suggested Price (INR)
                      {sortField === 'suggested_price_inr' && (
                        <ArrowUpDown size={14} className="ml-1" />
                      )}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-3 cursor-pointer"
                    onClick={() => handleSort('profit_margin')}
                  >
                    <div className="flex items-center">
                      Profit Margin %
                      {sortField === 'profit_margin' && (
                        <ArrowUpDown size={14} className="ml-1" />
                      )}
                    </div>
                  </th>
                  <th className="px-6 py-3">
                    <div className="flex items-center">
                      Adjustment
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedMenuItems.map((item, index) => (
                  <React.Fragment key={`${item.dish_name}-${index}`}>
                    <tr 
                      className={`bg-white border-b hover:bg-gray-50 cursor-pointer ${selectedItem === item ? 'bg-blue-50' : ''}`}
                      onClick={() => handleItemClick(item)}
                    >
                      <td className="px-6 py-4 font-medium text-gray-900">
                        {item.dish_name}
                      </td>
                      <td className="px-6 py-4">
                        ₹{item.dish_cost_inr.toFixed(2)}
                      </td>
                      <td className="px-6 py-4">
                        ₹{item.suggested_price_inr.toFixed(2)}
                      </td>
                      <td className="px-6 py-4">
                        <span className="text-green-600 font-medium">
                          {item.profit_margin.toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        {item.adjustment}
                      </td>
                    </tr>
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12 bg-gray-50 rounded-lg">
            <p className="text-lg text-gray-600">
              No cost optimization data available.
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Try refreshing to load data from the API.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default CostEfficiencyTool; 