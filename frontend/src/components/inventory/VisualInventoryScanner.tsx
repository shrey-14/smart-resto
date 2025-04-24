import React, { useState, useRef } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import { Upload, Camera, Loader2, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface DetectedItem {
  name: string;
  count: number;
}

const VisualInventoryScanner: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [detectedItems, setDetectedItems] = useState<DetectedItem[]>([]);
  const [outputImageUrl, setOutputImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.match(/image\/(jpeg|png)/)) {
      toast.error('Please upload a JPG or PNG image');
      return;
    }

    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      toast.error('Image size should be less than 10MB');
      return;
    }

    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);
    setSelectedFile(file);
    setDetectedItems([]);
    setOutputImageUrl(null);
    setError(null);
  };

  const handleScanInventory = async () => {
    if (!selectedFile) {
      toast.error('Please upload an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch('http://0.0.0.0:8000/api/inventory-tracking', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to scan inventory');
      }

      const data = await response.json();
      
      if (data.status === 'success' && data.results) {
        // Convert results to the expected format
        const items = Object.entries(data.results[0]).flatMap(([imageName, items]) => 
          Object.entries(items as Record<string, number>).map(([name, count]) => ({
            name,
            count,
            image: '' // Not needed anymore
          }))
        );
        
        setDetectedItems(items);
        
        // Set the output image URL
        if (data.output_path) {
          // Extract filename from the path
          const filename = data.output_path.split('/').pop();
          setOutputImageUrl(`http://0.0.0.0:8000/static/${filename}`);
        }
        
        toast.success('Inventory scanned successfully!');
      }
    } catch (err) {
      console.error('Error scanning inventory:', err);
      setError(err instanceof Error ? err.message : 'Error scanning inventory');
      toast.error('Failed to scan inventory');
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveImage = () => {
    setImagePreview(null);
    setSelectedFile(null);
    setDetectedItems([]);
    setOutputImageUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100 mb-6">
      <Toaster position="top-right" />
      <h3 className="text-lg font-medium text-gray-800 mb-2">Visual Stock Scanner</h3>
      <p className="text-sm text-gray-500 mb-4">
        Take clear photos of your inventory shelves to automatically identify and count items.
      </p>
      
      {error && (
        <div className="mb-4 p-4 bg-red-50 border-l-4 border-red-500 rounded-md">
          <p className="text-red-700">{error}</p>
        </div>
      )}
      
      <div className="grid md:grid-cols-2 gap-6">
        {/* Image upload area */}
        <div>
          <div className="border-2 border-dashed rounded-lg p-6 text-center border-gray-300 hover:border-restaurant-primary transition-colors">
            {imagePreview ? (
              <div className="space-y-4">
                <div className="flex flex-col items-center">
                  <img
                    src={imagePreview}
                    alt="Preview"
                    className="max-h-64 object-contain mb-2" 
                  />
                  <p className="text-sm text-gray-600 truncate max-w-full">
                    {selectedFile?.name || 'Preview'}
                  </p>
                </div>
                <Button
                  variant="outline"
                  onClick={handleRemoveImage}
                  className="text-red-500 border-red-300 hover:bg-red-50 hover:text-red-600"
                >
                  Remove Image
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-col items-center">
                  <Upload size={36} className="text-gray-400 mb-2" />
                  <p className="text-sm text-gray-600">
                    Drag and drop your inventory image here, or click to browse
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    Supported formats: JPG, PNG (max 10MB)
                  </p>
                </div>
                <div className="mt-4">
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    onChange={handleImageUpload}
                    accept="image/jpeg,image/png,image/jpg"
                  />
                  <Button
                    variant="outline"
                    className="text-restaurant-primary border-restaurant-primary/30 hover:bg-restaurant-primary/10"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Camera size={18} className="mr-2" />
                    Select Image
                  </Button>
                </div>
              </div>
            )}
          </div>
          
          <Button
            className="w-full mt-4 bg-restaurant-primary hover:bg-restaurant-primary/90 text-white"
            onClick={handleScanInventory}
            disabled={!selectedFile || loading}
          >
            {loading ? (
              <>
                <Loader2 size={16} className="mr-2 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Camera size={16} className="mr-2" />
                Scan Inventory
              </>
            )}
          </Button>
        </div>
        
        {/* Results area */}
        <div className="bg-gray-50 rounded-lg p-6">
          <h4 className="font-medium text-gray-800 mb-4">Detection Results</h4>
          
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 size={30} className="animate-spin text-restaurant-primary mr-3" />
              <p className="text-gray-600">Analyzing inventory...</p>
            </div>
          ) : outputImageUrl ? (
            <div className="space-y-4">
              {/* Detected Image */}
              <div className="bg-white p-3 rounded-md shadow-sm">
                <p className="text-sm font-medium text-gray-600 mb-2">Detected Items</p>
                <img 
                  src={outputImageUrl} 
                  alt="Detected Items" 
                  className="w-full rounded-md"
                />
              </div>
              
              {/* Detected Items List */}
              <div className="bg-white rounded-md shadow-sm overflow-hidden">
                <div className="p-3 bg-gray-50 border-b flex items-center">
                  <Info size={16} className="text-blue-500 mr-2" />
                  <h5 className="text-sm font-medium">Item Counts</h5>
                </div>
                <div className="p-4">
                  {detectedItems.length > 0 ? (
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {detectedItems.map((item, index) => (
                        <div 
                          key={index} 
                          className="flex justify-between items-center p-2 bg-gray-50 rounded text-sm"
                        >
                          <span className="font-medium capitalize">{item.name}</span>
                          <span className="bg-restaurant-primary/10 text-restaurant-primary px-2 py-0.5 rounded">
                            Count: {item.count}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-4">
                      <p className="text-gray-500">No items detected in the image</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-gray-500">
                Upload an image and click "Scan Inventory" to see results
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VisualInventoryScanner; 