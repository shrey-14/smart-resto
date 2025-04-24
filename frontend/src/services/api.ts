import { toast } from 'sonner';

// API base URL configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '' // In production, use relative URLs with proxy
  : 'http://0.0.0.0:8000'; // In development, connect directly to the backend

console.log(`Using API base URL: ${API_BASE_URL || 'Proxy configuration'}`);

interface ApiResponse<T> {
  status: string;
  [key: string]: any;
}

/**
 * General fetch wrapper with error handling
 */
const fetchApi = async <T>(
  endpoint: string, 
  options: RequestInit = {}
): Promise<T> => {
  try {
    const url = `${API_BASE_URL}${endpoint}`;
    console.log(`API Request to: ${url}`);
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error(`API Error (${response.status}):`, errorData);
      throw new Error(errorData.detail || 'An error occurred');
    }

    const data = await response.json();
    console.log(`API Response from ${url}:`, data);
    return data as T;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('API Error:', errorMessage);
    toast.error(errorMessage);
    throw error;
  }
};

/**
 * File upload helper function
 */
const uploadFile = async <T>(
  endpoint: string,
  file: File
): Promise<T> => {
  try {
    const url = `${API_BASE_URL}${endpoint}`;
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'An error occurred');
    }

    const data = await response.json();
    return data as T;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    toast.error(errorMessage);
    throw error;
  }
};

// Smart Restocking API - connects to demand-waste-prediction
export const getRestockingRecommendations = async (): Promise<ApiResponse<any>> => {
  return fetchApi<ApiResponse<any>>('/api/demand-waste-prediction', {
    method: 'POST',
  });
};

// Upload inventory data for smart restocking
export const uploadInventoryData = async (file: File): Promise<ApiResponse<any>> => {
  return uploadFile<ApiResponse<any>>('/api/demand-waste-prediction', file);
};

// Sales Forecasting API
export const getSalesForecast = async (): Promise<ApiResponse<any>> => {
  return fetchApi<ApiResponse<any>>('/api/sales-forecasting', {
    method: 'GET',
  });
};

// Upload sales data for forecasting
export const uploadSalesData = async (file: File): Promise<ApiResponse<any>> => {
  return uploadFile<ApiResponse<any>>('/api/sales-forecasting', file);
};

// Cost Optimization API
export const getCostOptimization = async (): Promise<ApiResponse<any>> => {
  return fetchApi<ApiResponse<any>>('/api/cost-optimization', {
    method: 'GET',
  });
};

// Upload menu data for cost optimization
export const uploadMenuData = async (file: File): Promise<ApiResponse<any>> => {
  return uploadFile<ApiResponse<any>>('/api/cost-optimization/menu', file);
};

// Upload ingredient pricing data for cost optimization
export const uploadIngredientData = async (file: File): Promise<ApiResponse<any>> => {
  return uploadFile<ApiResponse<any>>('/api/cost-optimization/ingredients', file);
};

// Recipe Recommendation API
export const getRecipeRecommendation = async (): Promise<ApiResponse<any>> => {
  return fetchApi<ApiResponse<any>>('/api/recipe-recommendation', {
    method: 'GET',
  });
};

// Recipe Generation API
export const getRecipeGeneration = async (): Promise<ApiResponse<any>> => {
  return fetchApi<ApiResponse<any>>('/api/recipe-generation', {
    method: 'GET',
  });
};

// Food Spoilage Detection API
export const detectFoodSpoilage = async (file?: File): Promise<ApiResponse<any>> => {
  if (file) {
    return uploadFile<ApiResponse<any>>('/api/spoilage-detection', file);
  }
  return fetchApi<ApiResponse<any>>('/api/spoilage-detection', {
    method: 'GET',
  });
};

// Waste Classification API
export const classifyWaste = async (file?: File): Promise<ApiResponse<any>> => {
  if (file) {
    return uploadFile<ApiResponse<any>>('/api/waste-classification', file);
  }
  return fetchApi<ApiResponse<any>>('/api/waste-classification', {
    method: 'GET',
  });
};

// Waste Heatmap API
export const generateWasteHeatmap = async (file?: File): Promise<ApiResponse<any>> => {
  if (file) {
    return uploadFile<ApiResponse<any>>('/api/waste-heatmap', file);
  }
  return fetchApi<ApiResponse<any>>('/api/waste-heatmap', {
    method: 'GET',
  });
};

// Inventory Tracking API
export const scanInventory = async (file?: File): Promise<ApiResponse<any>> => {
  try {
    if (file) {
      console.log('Uploading file to inventory tracking API:', file.name);
      
      // Create form data
      const formData = new FormData();
      formData.append('file', file);
      
      // Make the API request
      const url = `${API_BASE_URL}/api/inventory-tracking`;
      console.log('Making POST request to:', url);
      
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header - browser will set it with proper boundary
      });
      
      if (!response.ok) {
        console.error('Error response from API:', response.status, response.statusText);
        let errorMessage = 'Failed to upload image';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
          console.error('Error details:', errorData);
        } catch (e) {
          console.error('Could not parse error response');
        }
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      console.log('API response received:', data);
      return data;
    }
    
    // If no file is provided, make a GET request
    console.log('Making GET request to inventory tracking API');
    return fetchApi<ApiResponse<any>>('/api/inventory-tracking', {
      method: 'GET',
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error in inventory tracking';
    console.error('Error in inventory tracking API:', errorMessage);
    toast.error(errorMessage);
    throw error;
  }
};

// Stock Detection API
export const detectStock = async (file?: File): Promise<ApiResponse<any>> => {
  try {
    if (file) {
      console.log('Uploading video to stock detection API:', file.name);
      
      // Create form data
      const formData = new FormData();
      formData.append('file', file);
      
      // Make the API request
      const url = `${API_BASE_URL}/api/stock-detection`;
      console.log('Making POST request to:', url);
      
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header - browser will set it with proper boundary
      });
      
      if (!response.ok) {
        console.error('Error response from API:', response.status, response.statusText);
        let errorMessage = 'Failed to upload video';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
          console.error('Error details:', errorData);
        } catch (e) {
          console.error('Could not parse error response');
        }
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      console.log('Stock detection API response:', data);
      
      // Make sure we return all necessary information including task_id
      return {
        status: data.status || 'success',
        message: data.message || '',
        video_url: data.video_url || null,
        results: data.results || null,
        processing: !!data.processing,
        task_id: data.task_id || null,
        ...data
      };
    }
    
    // If no file is provided, make a GET request
    console.log('Making GET request to stock detection API');
    return fetchApi<ApiResponse<any>>('/api/stock-detection', {
      method: 'GET',
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error in stock detection';
    console.error('Error in stock detection API:', errorMessage);
    toast.error(errorMessage);
    throw error;
  }
};

// Task Status API
export const getTaskStatus = async (taskId: string): Promise<ApiResponse<any>> => {
  try {
    console.log('Fetching task status for task:', taskId);
    return fetchApi<ApiResponse<any>>(`/api/task-status/${taskId}`, {
      method: 'GET',
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error in task status';
    console.error('Error in task status API:', errorMessage);
    throw error;
  }
};

export default {
  getRestockingRecommendations,
  uploadInventoryData,
  getSalesForecast,
  uploadSalesData,
  getCostOptimization,
  uploadMenuData,
  uploadIngredientData,
  getRecipeRecommendation,
  getRecipeGeneration,
  detectFoodSpoilage,
  classifyWaste,
  generateWasteHeatmap,
  scanInventory,
  detectStock,
  getTaskStatus,
}; 