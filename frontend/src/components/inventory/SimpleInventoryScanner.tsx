import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { Loader2, Upload, Info, ImageIcon } from 'lucide-react';

interface DetectedItem {
  image: string;
  item: string;
  count: number;
}

const SimpleInventoryScanner = () => {
  const [loading, setLoading] = useState(false);
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Function to handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    console.log('File selected:', file.name);
    setImage(file);
    
    // Create image preview
    const reader = new FileReader();
    reader.onload = (event) => {
      setImagePreview(event.target?.result as string);
    };
    reader.readAsDataURL(file);
  };
  
  // Function to process inventory scanning
  const handleScan = async () => {
    if (!image) {
      toast.error('Please select an image first');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      // Create form data for the API request
      const formData = new FormData();
      formData.append('file', image);
      
      // Make API call to backend
      console.log('Sending request to API with image:', image.name);
      const response = await fetch('http://0.0.0.0:8000/api/inventory-tracking', {
        method: 'POST',
        body: formData,
      });
      
      // Log the raw response
      console.log('Raw response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }
      
      // Parse the JSON response
      const data = await response.json();
      console.log('API response:', data);
      
      // Store the result
      setResult(data);
      
      // Show success message
      toast.success('Inventory scan completed successfully');
    } catch (err) {
      console.error('Error during scan:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      toast.error('Failed to process inventory scan');
    } finally {
      setLoading(false);
    }
  };
  
  // Function to render detected items if available
  const renderDetectedItems = () => {
    if (!result) return null;
    
    // Extract detected items from result
    const items: DetectedItem[] = [];
    if (result.results && Array.isArray(result.results)) {
      for (const resultItem of result.results) {
        Object.entries(resultItem).forEach(([image, detections]) => {
          Object.entries(detections as Record<string, number>).forEach(([item, count]) => {
            items.push({ image, item, count });
          });
        });
      }
    }
    
    if (items.length === 0) {
      return <p className="text-gray-500">No items detected</p>;
    }
    
    return (
      <div className="space-y-2">
        <h4 className="font-medium text-lg">Detected Items:</h4>
        <div className="grid gap-2 max-h-[300px] overflow-y-auto pr-2">
          {items.map((item, index) => (
            <div key={index} className="flex justify-between bg-white p-2 rounded shadow-sm">
              <span className="font-medium capitalize">{item.item}</span>
              <span className="bg-blue-100 text-blue-800 px-2 py-0.5 rounded">
                Count: {item.count}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };
  
  // Function to get annotated image URL from result
  const getAnnotatedImageUrl = () => {
    if (!result) return null;
    
    let imageUrl = null;
    
    // Check different possible fields for the image path
    if (result.output_path) {
      imageUrl = result.output_path;
    } else if (result.annotated_image_path) {
      imageUrl = result.annotated_image_path;
    } else if (result.output_image_path) {
      imageUrl = result.output_image_path;
    }
    
    if (!imageUrl) return null;
    
    // Ensure it's a full URL
    if (!imageUrl.startsWith('http')) {
      imageUrl = `http://0.0.0.0:8000/static/${imageUrl.split('/').pop()}`;
    }
    
    return imageUrl;
  };
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-5xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Inventory Scanner</h2>
      
      {/* Error display */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6">
          <p className="text-red-700">{error}</p>
        </div>
      )}
      
      {/* File upload section */}
      <div className="mb-6">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
          {imagePreview ? (
            <img 
              src={imagePreview} 
              alt="Preview" 
              className="max-h-64 mx-auto mb-4 rounded" 
            />
          ) : (
            <div className="py-12">
              <Upload size={48} className="mx-auto text-gray-400 mb-4" />
              <p className="text-gray-600">
                Select an image to scan inventory
              </p>
            </div>
          )}
          
          <input
            type="file"
            id="inventory-image"
            className="hidden"
            accept="image/jpeg,image/png,image/jpg"
            onChange={handleFileChange}
          />
          
          <div className="flex justify-center mt-4 gap-4">
            <Button
              onClick={() => document.getElementById('inventory-image')?.click()}
              className="bg-blue-600 hover:bg-blue-700"
            >
              <Upload className="w-4 h-4 mr-2" />
              Select Image
            </Button>
            
            <Button
              onClick={handleScan}
              disabled={loading || !image}
              className="bg-green-600 hover:bg-green-700"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Scanning...
                </>
              ) : (
                'Scan Inventory'
              )}
            </Button>
          </div>
        </div>
      </div>
      
      {/* Results section */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-12 bg-gray-50 rounded-lg">
          <Loader2 size={48} className="text-blue-600 animate-spin mb-4" />
          <p className="text-gray-600">Analyzing your inventory...</p>
          <p className="text-gray-500 text-sm mt-2">This may take a few moments</p>
        </div>
      ) : result ? (
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4 border-b pb-2">Scan Results</h3>
          
          {/* Images comparison section */}
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            {/* Original image */}
            <div className="bg-white p-4 rounded shadow-sm">
              <div className="flex items-center mb-3">
                <ImageIcon className="w-5 h-5 text-blue-500 mr-2" />
                <h4 className="font-medium">Original Image</h4>
              </div>
              {imagePreview && (
                <div className="relative">
                  <img 
                    src={imagePreview} 
                    alt="Original" 
                    className="w-full rounded border border-gray-200" 
                  />
                  <div className="absolute top-0 left-0 bg-blue-500/70 text-white px-2 py-1 text-xs rounded-br">
                    Original
                  </div>
                </div>
              )}
            </div>
            
            {/* Annotated image */}
            <div className="bg-white p-4 rounded shadow-sm">
              <div className="flex items-center mb-3">
                <Info className="w-5 h-5 text-green-500 mr-2" />
                <h4 className="font-medium">Detected Items</h4>
              </div>
              {getAnnotatedImageUrl() ? (
                <div className="relative">
                  <img 
                    src={getAnnotatedImageUrl()} 
                    alt="Detected Items" 
                    className="w-full rounded border border-gray-200" 
                    onError={(e) => {
                      console.error('Error loading annotated image');
                      e.currentTarget.style.display = 'none';
                      toast.error('Failed to load annotated image');
                      
                      // Show error message
                      const errorDiv = document.createElement('div');
                      errorDiv.className = 'text-red-500 text-center py-4';
                      errorDiv.textContent = 'Failed to load annotated image';
                      e.currentTarget.parentNode?.appendChild(errorDiv);
                    }}
                  />
                  <div className="absolute top-0 left-0 bg-green-500/70 text-white px-2 py-1 text-xs rounded-br">
                    Detected
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  No annotated image available
                </div>
              )}
            </div>
          </div>
          
          {/* Detected items list */}
          {renderDetectedItems()}
          
          {/* Instructions */}
          <div className="mt-6 bg-blue-50 p-4 rounded">
            <h3 className="font-medium text-blue-800 mb-2">Tips:</h3>
            <ul className="list-disc list-inside text-blue-700 space-y-1 pl-2">
              <li>Compare the original image with the detected items</li>
              <li>The annotated image shows what items were detected</li>
              <li>Check the item counts in the list below the images</li>
            </ul>
          </div>
        </div>
      ) : (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <p className="text-lg text-gray-600">
            Upload an image and click "Scan Inventory" to see results
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Our AI will analyze your shelf image and detect inventory items
          </p>
        </div>
      )}
    </div>
  );
};

export default SimpleInventoryScanner; 