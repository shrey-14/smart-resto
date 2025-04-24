import React, { useState, useRef } from 'react';
import { classifyWaste } from '@/services/api';
import { Loader2, RefreshCw, Upload, ImageIcon, AlertCircle, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { 
  ClassificationResult, 
  WasteCategory, 
  createClassificationResult,
  parseResponseWithDescription,
  extractEssentialWasteData
} from './WasteTypes';

const WasteCategorization = () => {
  const [loading, setLoading] = useState(false);
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [noFoodDetected, setNoFoodDetected] = useState(false);
  const [noWasteDetected, setNoWasteDetected] = useState(false);
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
      setNoFoodDetected(false);
      setNoWasteDetected(false);
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
      setNoFoodDetected(false);
      setNoWasteDetected(false);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleClassifyWaste = async () => {
    if (!image) {
      toast.error('Please select an image first');
      return;
    }

    // Reset states
    setResult(null);
    setNoFoodDetected(false);
    setNoWasteDetected(false);
    setLoading(true);
    
    try {
      console.log('Sending image to API:', image.name, image.type, image.size);
      const response = await classifyWaste(image);
      
      console.log('API Response:', response);
      
      if (response.status === 'success' && response.classification) {
        // Parse the classification string which is a JSON string
        try {
          // The classification field contains a JSON string that needs to be parsed
          const classificationData = typeof response.classification === 'string' 
            ? JSON.parse(response.classification) 
            : response.classification;
            
          console.log('Parsed classification data:', classificationData);
          
          // Handle the three cases based on contains_food and is_waste flags
          if (!classificationData.contains_food) {
            // Case 1: No food detected
            setNoFoodDetected(true);
            toast.info('No food detected in the image');
          } else if (classificationData.contains_food && !classificationData.is_waste) {
            // Case 2: Food detected, but no waste
            setNoWasteDetected(true);
            toast.info('Food detected, but no waste found');
          } else if (classificationData.contains_food && classificationData.is_waste) {
            // Case 3: Food waste detected with categories
            const result: ClassificationResult = {
              timestamp: classificationData.timestamp,
              image_id: classificationData.image_id,
              contains_food: classificationData.contains_food,
              is_waste: classificationData.is_waste,
              categories: classificationData.categories || []
            };
            
            setResult(result);
            toast.success('Waste classification completed');
          } else {
            // Unexpected state
            toast.error('Unexpected classification result');
            console.error('Unexpected classification result:', classificationData);
          }
        } catch (parseError) {
          console.error('Error parsing classification data:', parseError, response.classification);
          
          // Fallback to the old parsing mechanisms if JSON.parse fails
          const classificationText = response.classification;
          
          // Try old parsing methods one by one
          // First check for text-based indicators
          if (typeof classificationText === 'string') {
            if (classificationText.includes('contains_food": false')) {
              setNoFoodDetected(true);
              toast.info('No food detected in the image');
              return;
            }
            
            if (classificationText.includes('contains_food": true') && 
                classificationText.includes('is_waste": false')) {
              setNoWasteDetected(true);
              toast.info('Food detected, but no waste found');
              return;
            }
            
            // If we reach here, try the different parsers
            const parsedResult = parseResponseWithDescription(classificationText) || 
                               createClassificationResult(classificationText);
            
            if (parsedResult) {
              setResult(parsedResult);
              toast.success('Waste classification completed');
              return;
            }
            
            // Last resort - extract categories
            const essentialCategories = extractEssentialWasteData(classificationText);
            if (essentialCategories && essentialCategories.length > 0) {
              const minimalResult: ClassificationResult = {
                timestamp: new Date().toISOString(),
                image_id: image.name || 'uploaded_image.jpg',
                contains_food: true,
                is_waste: true,
                categories: essentialCategories
              };
              setResult(minimalResult);
              toast.success('Waste categories identified');
              return;
            }
            
            // If all parsing attempts fail
            toast.error('Could not parse classification result');
            console.error('Could not parse classification result:', classificationText);
          }
        }
      } else {
        console.error('API did not return success status or classification is missing:', response);
        toast.error('Failed to classify waste');
      }
    } catch (error) {
      console.error('Error classifying waste:', error);
      toast.error('Error processing waste classification');
    } finally {
      setLoading(false);
    }
  };

  const handleClickUpload = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Helper function to format confidence as percentage
  const formatConfidence = (confidence: number): string => {
    return `${Math.round(confidence * 100)}%`;
  };

  // Helper function to get color class based on confidence
  const getConfidenceColorClass = (confidence: number): string => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.5) return 'text-amber-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-100">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-800">Waste Categorization</h2>
          <p className="text-sm text-gray-500">Upload food waste images to analyze and categorize</p>
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
                  alt="Waste Preview" 
                  className="max-h-64 mx-auto rounded-md" 
                />
              </div>
            ) : (
              <div className="py-8">
                <ImageIcon size={48} className="mx-auto text-gray-400 mb-4" />
                <p className="text-gray-600 mb-2">Drag and drop your waste image here</p>
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
            onClick={handleClassifyWaste}
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
                Classify Waste
              </>
            )}
          </Button>
        </div>
        
        {/* Results Section */}
        <div className="bg-gray-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">Classification Results</h3>
          
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 size={30} className="animate-spin text-restaurant-primary mr-3" />
              <p className="text-lg text-gray-600">Analyzing waste image...</p>
            </div>
          ) : noFoodDetected ? (
            <div className="text-center py-8 bg-white rounded-md shadow-sm">
              <AlertCircle size={48} className="mx-auto text-blue-500 mb-4" />
              <h4 className="text-xl font-medium text-gray-800 mb-2">No Food Detected</h4>
              <p className="text-gray-600">
                We couldn't detect any food items in this image. 
                Please upload an image containing food to analyze waste.
              </p>
            </div>
          ) : noWasteDetected ? (
            <div className="text-center py-8 bg-white rounded-md shadow-sm">
              <Info size={48} className="mx-auto text-green-500 mb-4" />
              <h4 className="text-xl font-medium text-gray-800 mb-2">No Food Waste Detected</h4>
              <p className="text-gray-600">
                Food was detected in the image, but no waste was identified.
                The food appears to be fresh and consumable.
              </p>
            </div>
          ) : result ? (
            <div className="space-y-5">
              {/* Summary */}
              <div className="bg-white p-4 rounded-md shadow-sm">
                <div className="flex items-center mb-3">
                  <Info size={20} className="text-blue-500 mr-2" />
                  <h4 className="font-medium text-gray-800">Image Summary</h4>
                </div>
                
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-gray-600">Image ID:</p>
                    <p className="font-medium">{result.image_id}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Contains Food:</p>
                    <p className="font-medium">{result.contains_food ? 'Yes' : 'No'}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Is Waste:</p>
                    <p className="font-medium">{result.is_waste ? 'Yes' : 'No'}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Analyzed:</p>
                    <p className="font-medium">{new Date(result.timestamp).toLocaleString()}</p>
                  </div>
                </div>
              </div>
              
              {/* Categories */}
              <div>
                <h4 className="font-medium text-gray-800 mb-3 flex items-center">
                  <AlertCircle size={18} className="text-restaurant-primary mr-2" />
                  Waste Categories
                </h4>
                
                {result.categories && result.categories.length > 0 ? (
                  <div className="space-y-3">
                    {result.categories.map((category, index) => (
                      <div key={index} className="bg-white p-4 rounded-md shadow-sm">
                        <div className="flex justify-between items-center mb-2">
                          <div className="flex items-center">
                            <span className="font-medium text-gray-800 capitalize">{category.name}</span>
                            <span className={`ml-2 text-sm ${getConfidenceColorClass(category.confidence)}`}>
                              ({formatConfidence(category.confidence)})
                            </span>
                          </div>
                          <span className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-xs capitalize">
                            {category.food_type}
                          </span>
                        </div>
                        <p className="text-gray-600 text-sm">{category.explanation}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500 text-center py-4 bg-white rounded-md">
                    No waste categories detected
                  </p>
                )}
              </div>
              
              {/* Recommendations based on waste type */}
              <div className="bg-white p-4 rounded-md shadow-sm">
                <h4 className="font-medium text-gray-800 mb-2">Recommendations:</h4>
                <ul className="list-disc list-inside text-gray-700 space-y-1 text-sm">
                  <li>Consider composting organic waste materials</li>
                  <li>Review food preparation processes to reduce waste</li>
                  <li>Track waste patterns to identify optimization opportunities</li>
                  <li>Implement portion control measures to minimize over-production</li>
                </ul>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-lg text-gray-600">
                Upload an image and click "Classify Waste" to see results
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Our AI will analyze your waste image and determine food type and waste categories
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default WasteCategorization; 