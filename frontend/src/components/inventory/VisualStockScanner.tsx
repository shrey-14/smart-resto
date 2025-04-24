import React, { useState, useRef } from 'react';
import { Loader2, RefreshCw, Upload, ImageIcon, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { scanInventory } from '@/services/api';

interface DetectedItem {
  image: string;
  item: string;
  count: number;
}

interface InventoryTrackingResponse {
  status: string;
  message: string;
  results: Array<{ [key: string]: { [key: string]: number } }>;
  output_path: string;
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg'];

const VisualStockScanner = () => {
  const [loading, setLoading] = useState(false);
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [detectedItems, setDetectedItems] = useState<DetectedItem[]>([]);
  const [annotatedImageUrl, setAnnotatedImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): boolean => {
    console.log('Validating file:', { name: file.name, type: file.type, size: file.size });
    
    if (!ALLOWED_TYPES.includes(file.type)) {
      setError('Please upload a valid image file (JPEG, PNG)');
      toast.error('Invalid file type');
      return false;
    }

    if (file.size > MAX_FILE_SIZE) {
      setError('File size should be less than 10MB');
      toast.error('File too large');
      return false;
    }

    return true;
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      console.log('No file selected');
      return;
    }

    console.log('File selected:', { name: file.name, type: file.type, size: file.size });

    if (!validateFile(file)) {
      e.target.value = ''; // Reset input
      return;
    }

    setImage(file);
    setError(null);
    
    // Create a preview
    const reader = new FileReader();
    reader.onload = (event) => {
      setImagePreview(event.target?.result as string);
      console.log('Preview created successfully');
    };
    reader.onerror = (error) => {
      console.error('Error reading file:', error);
      setError('Error reading file');
      toast.error('Error reading file');
    };
    reader.readAsDataURL(file);
    
    // Reset any previous results
    setDetectedItems([]);
    setAnnotatedImageUrl(null);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    
    const file = e.dataTransfer.files?.[0];
    if (!file) {
      console.log('No file dropped');
      return;
    }

    console.log('File dropped:', { name: file.name, type: file.type, size: file.size });

    if (!validateFile(file)) return;

    setImage(file);
    setError(null);
    
    // Create a preview
    const reader = new FileReader();
    reader.onload = (event) => {
      setImagePreview(event.target?.result as string);
      console.log('Preview created successfully');
    };
    reader.onerror = (error) => {
      console.error('Error reading file:', error);
      setError('Error reading file');
      toast.error('Error reading file');
    };
    reader.readAsDataURL(file);
    
    // Reset any previous results
    setDetectedItems([]);
    setAnnotatedImageUrl(null);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleScanInventory = async () => {
    if (!image) {
      toast.error('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      console.log('Starting inventory scan with image:', {
        name: image.name,
        type: image.type,
        size: image.size
      });

      const response = await scanInventory(image);
      console.log('Scan response received:', response);

      if (response && response.status === 'success') {
        // Process detected items
        const items: DetectedItem[] = [];
        
        // Handle results based on the response structure
        if (response.results && Array.isArray(response.results)) {
          console.log('Processing array results:', response.results);
          response.results.forEach(result => {
            Object.entries(result).forEach(([image, detections]) => {
              Object.entries(detections).forEach(([item, count]) => {
                items.push({ image, item, count });
              });
            });
          });
        } else if (response.detected_items) {
          // Alternative structure that might be returned
          console.log('Processing detected_items:', response.detected_items);
          
          if (Array.isArray(response.detected_items)) {
            response.detected_items.forEach(item => {
              if (typeof item === 'object') {
                const image = Object.keys(item)[0] || 'unknown';
                const detections = item[image] || {};
                
                Object.entries(detections).forEach(([itemName, count]) => {
                  items.push({ 
                    image, 
                    item: itemName, 
                    count: typeof count === 'number' ? count : 1 
                  });
                });
              }
            });
          }
        }
        
        console.log('Final processed items:', items);
        setDetectedItems(items);
        
        // Set the annotated image URL based on the response structure
        let imageUrl = null;
        
        if (response.output_path) {
          imageUrl = response.output_path;
        } else if (response.annotated_image_path) {
          imageUrl = response.annotated_image_path;
        } else if (response.output_image_path) {
          imageUrl = response.output_image_path;
        }
        
        if (imageUrl) {
          // Ensure it's a full URL
          if (!imageUrl.startsWith('http')) {
            imageUrl = `http://0.0.0.0:8000/static/${imageUrl.split('/').pop()}`;
          }
          
          console.log('Setting annotated image URL:', imageUrl);
          setAnnotatedImageUrl(imageUrl);
        } else {
          console.warn('No image URL found in response');
          toast.warning('No annotated image available');
        }
        
        toast.success('Inventory scan completed successfully');
      } else {
        console.error('Scan failed or invalid response:', response);
        setError('Failed to process inventory scan or invalid response');
        toast.error('Failed to process inventory scan');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Error processing inventory scan';
      console.error('Error scanning inventory:', error);
      setError(errorMessage);
      toast.error(errorMessage);
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
          <h2 className="text-xl font-semibold text-gray-800">Visual Stock Scanner</h2>
          <p className="text-sm text-gray-500">Upload shelf images to analyze inventory</p>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-md">
          <p className="text-red-600">{error}</p>
        </div>
      )}

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
                  alt="Stock Preview" 
                  className="max-h-64 mx-auto rounded-md" 
                />
              </div>
            ) : (
              <div className="py-8">
                <ImageIcon size={48} className="mx-auto text-gray-400 mb-4" />
                <p className="text-gray-600 mb-2">Drag and drop your shelf image here</p>
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
            onClick={handleScanInventory}
            disabled={loading || !image}
            className="mt-4 w-full bg-restaurant-primary hover:bg-restaurant-primary/90 text-white"
          >
            {loading ? (
              <>
                <Loader2 size={16} className="mr-2 animate-spin" />
                Scanning...
              </>
            ) : (
              <>
                <RefreshCw size={16} className="mr-2" />
                Scan Inventory
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
              <p className="text-lg text-gray-600">Scanning inventory...</p>
            </div>
          ) : annotatedImageUrl ? (
            <div className="space-y-5">
              {/* Detected Image */}
              <div className="bg-white p-4 rounded-md shadow-sm">
                <div className="flex items-center mb-3">
                  <Info size={20} className="text-blue-500 mr-2" />
                  <h4 className="font-medium text-gray-800">Detected Items</h4>
                </div>
                
                <div className="relative">
                  <img 
                    src={annotatedImageUrl} 
                    alt="Detected Items" 
                    className="w-full rounded-md mb-4" 
                    onError={(e) => {
                      console.error('Error loading annotated image:', annotatedImageUrl);
                      
                      // Try to determine if it's a CORS issue
                      const imgSrc = e.currentTarget.src;
                      if (imgSrc.startsWith('http://0.0.0.0:8000/')) {
                        // Try to replace with relative URL
                        const relativePath = imgSrc.replace('http://0.0.0.0:8000/', '/');
                        console.log('Trying relative path instead:', relativePath);
                        e.currentTarget.src = relativePath;
                      } else {
                        // Hide the image on error
                        e.currentTarget.style.display = 'none';
                        
                        // Show error message
                        const errorDiv = document.createElement('div');
                        errorDiv.className = 'text-red-500 text-center py-4';
                        errorDiv.textContent = 'Failed to load annotated image';
                        e.currentTarget.parentNode?.appendChild(errorDiv);
                      }
                    }}
                  />
                  <div className="absolute top-0 left-0 bg-black/60 text-white px-2 py-1 text-xs rounded-br">
                    Detected
                  </div>
                </div>
                
                {/* Detected Items List */}
                {detectedItems.length > 0 ? (
                  <div className="space-y-3">
                    {detectedItems.map((item, index) => (
                      <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                        <span className="font-medium text-gray-800 capitalize">{item.item}</span>
                        <span className="bg-restaurant-primary/10 text-restaurant-primary px-2 py-1 rounded">
                          Count: {item.count}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500 text-center py-2">
                    No items detected
                  </p>
                )}
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-lg text-gray-600">
                Upload an image and click "Scan Inventory" to see results
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Our AI will analyze your shelf image and detect inventory items
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VisualStockScanner; 