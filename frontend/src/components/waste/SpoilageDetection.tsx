import React, { useState, useEffect, useRef } from 'react';
import { detectFoodSpoilage } from '@/services/api';
import { Loader2, RefreshCw, Upload, CheckCircle, XCircle, ImageIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

interface SpoilageResult {
  timestamp: string;
  image_name: string;
  food_item: string;
  freshness: 'fresh' | 'rotten';
  status: string;
}

const SpoilageDetection = () => {
  const [loading, setLoading] = useState(false);
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<SpoilageResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setImage(selectedFile);
      
      // Create a preview
      const reader = new FileReader();
      reader.onload = (event) => {
        setImagePreview(event.target?.result as string);
      };
      reader.readAsDataURL(selectedFile);
      
      // Reset any previous results
      setResult(null);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      setImage(droppedFile);
      
      // Create a preview
      const reader = new FileReader();
      reader.onload = (event) => {
        setImagePreview(event.target?.result as string);
      };
      reader.readAsDataURL(droppedFile);
      
      // Reset any previous results
      setResult(null);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDetectSpoilage = async () => {
    if (!image) {
      toast.error('Please select an image first');
      return;
    }

    setLoading(true);
    try {
      console.log('Sending image to API:', image.name, image.type, image.size);
      const response = await detectFoodSpoilage(image);
      
      console.log('API Response:', response);
      
      if (response.status === 'success' && response.result) {
        // Parse the CSV-like string from API response
        // Format: "timestamp","image_name","food_item","freshness","status"
        const csvLine = response.result;
        console.log('CSV Line from API:', csvLine);
        
        // Simple parsing (assumes no commas in the values)
        const parts = csvLine.match(/"([^"]*)"/g)?.map(part => part.slice(1, -1));
        console.log('Parsed parts:', parts);
        
        if (parts && parts.length >= 5) {
          const parsedResult: SpoilageResult = {
            timestamp: parts[0],
            image_name: parts[1],
            food_item: parts[2],
            freshness: parts[3] as 'fresh' | 'rotten',
            status: parts[4]
          };
          
          console.log('Final parsed result:', parsedResult);
          setResult(parsedResult);
          toast.success('Spoilage detection completed');
        } else {
          console.error('Invalid response format - could not parse parts:', parts);
          toast.error('Invalid response format from API');
        }
      } else {
        console.error('API did not return success status or result is missing:', response);
        toast.error('Failed to detect spoilage');
      }
    } catch (error) {
      console.error('Error detecting spoilage:', error);
      // Toast error is already handled in the API service
    } finally {
      setLoading(false);
    }
  };

  const handleClickUpload = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-100">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-800">Food Spoilage Detection</h2>
          <p className="text-sm text-gray-500">Upload food images to detect spoilage</p>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Image Upload Section */}
        <div>
          <div 
            className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-restaurant-primary transition-colors"
            onClick={handleClickUpload}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <input 
              type="file" 
              accept="image/*" 
              className="hidden" 
              onChange={handleImageChange}
              ref={fileInputRef}
            />
            
            {imagePreview ? (
              <div className="mb-4">
                <img 
                  src={imagePreview} 
                  alt="Food Preview" 
                  className="max-h-64 mx-auto rounded-md" 
                />
              </div>
            ) : (
              <div className="py-8">
                <ImageIcon size={48} className="mx-auto text-gray-400 mb-4" />
                <p className="text-gray-600 mb-2">Drag and drop your image here</p>
                <p className="text-gray-500 text-sm">or click to browse</p>
              </div>
            )}
            
            <Button 
              className="mt-4 bg-restaurant-primary hover:bg-restaurant-primary/90 text-white"
              onClick={(e) => {
                e.stopPropagation();
                handleClickUpload();
              }}
            >
              <Upload size={16} className="mr-2" />
              Select Image
            </Button>
          </div>
          
          <Button 
            onClick={handleDetectSpoilage}
            disabled={loading || !image}
            className="mt-4 w-full bg-restaurant-primary hover:bg-restaurant-primary/90 text-white"
          >
            {loading ? (
              <>
                <Loader2 size={16} className="mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <RefreshCw size={16} className="mr-2" />
                Detect Spoilage
              </>
            )}
          </Button>
        </div>
        
        {/* Results Section */}
        <div className="bg-gray-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">Detection Results</h3>
          
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 size={30} className="animate-spin text-restaurant-primary mr-3" />
              <p className="text-lg text-gray-600">Analyzing food image...</p>
            </div>
          ) : result ? (
            <div className="space-y-4">
              <div className="bg-white p-4 rounded-md shadow-sm">
                <div className="flex items-center mb-2">
                  <span className="font-medium text-gray-700 w-32">Food Item:</span>
                  <span className="text-gray-900">{result.food_item}</span>
                </div>
                
                <div className="flex items-center mb-2">
                  <span className="font-medium text-gray-700 w-32">Freshness:</span>
                  <div className="flex items-center">
                    {result.freshness === 'fresh' ? (
                      <>
                        <CheckCircle size={18} className="text-green-500 mr-2" />
                        <span className="text-green-600 font-medium">Fresh</span>
                      </>
                    ) : (
                      <>
                        <XCircle size={18} className="text-red-500 mr-2" />
                        <span className="text-red-600 font-medium">Rotten</span>
                      </>
                    )}
                  </div>
                </div>
                
                <div className="flex items-center mb-2">
                  <span className="font-medium text-gray-700 w-32">Image:</span>
                  <span className="text-gray-900">{result.image_name}</span>
                </div>
                
                <div className="flex items-center">
                  <span className="font-medium text-gray-700 w-32">Time:</span>
                  <span className="text-gray-900">{new Date(result.timestamp).toLocaleString()}</span>
                </div>
              </div>
              
              <div className="bg-white p-4 rounded-md shadow-sm">
                <h4 className="font-medium text-gray-800 mb-2">Recommendations:</h4>
                {result.freshness === 'fresh' ? (
                  <p className="text-green-700">This food item appears to be fresh and safe to consume.</p>
                ) : (
                  <div className="text-red-700 space-y-2">
                    <p>This food item appears to be spoiled. Consider the following actions:</p>
                    <ul className="list-disc list-inside">
                      <li>Dispose of the food item safely</li>
                      <li>Check other similar items for spoilage</li>
                      <li>Review storage conditions</li>
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-lg text-gray-600">
                Upload an image and click "Detect Spoilage" to see results
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Our AI will analyze your food image and determine freshness
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SpoilageDetection; 