import React, { useState, useRef } from 'react';
import { Upload, Camera, X, CheckCircle, Loader2, ImageIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { generateWasteHeatmap } from '@/services/api';

interface WasteMappingResult {
  heatmapUrl: string;
  detectionsUrl: string;
}

const WasteMapping: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState<WasteMappingResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    
    if (!selectedFile) {
      return;
    }
    
    // Check if file is an image
    if (!selectedFile.type.startsWith('image/')) {
      toast.error('Please upload an image file');
      return;
    }
    
    setFile(selectedFile);
    
    // Create a preview
    const imageUrl = URL.createObjectURL(selectedFile);
    setPreview(imageUrl);
    
    // Reset results when new file is selected
    setResults(null);
  };

  // Handle drag and drop
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    
    const droppedFile = e.dataTransfer.files?.[0];
    if (!droppedFile) return;
    
    // Check if file is an image
    if (!droppedFile.type.startsWith('image/')) {
      toast.error('Please upload an image file');
      return;
    }
    
    setFile(droppedFile);
    
    // Create a preview
    const imageUrl = URL.createObjectURL(droppedFile);
    setPreview(imageUrl);
    
    // Reset results when new file is selected
    setResults(null);
  };
  
  const handleUpload = async () => {
    if (!file) {
      toast.error('Please select an image file first');
      return;
    }
    
    setUploading(true);
    
    try {
      const response = await generateWasteHeatmap(file);
      
      if (response.status === 'success') {
        // Add timestamp to URLs to prevent caching
        const timestamp = new Date().getTime();
        // Use the full URL including the API_BASE_URL or window.location.origin if it's a relative URL
        const apiBaseUrl = process.env.NODE_ENV === 'production' ? window.location.origin : 'http://0.0.0.0:8000';
        
        const heatmapUrl = response.heatmap_url.startsWith('http') 
          ? `${response.heatmap_url}?t=${timestamp}`
          : `${apiBaseUrl}${response.heatmap_url}?t=${timestamp}`;
          
        const detectionsUrl = response.detections_url.startsWith('http')
          ? `${response.detections_url}?t=${timestamp}`
          : `${apiBaseUrl}${response.detections_url}?t=${timestamp}`;
        
        setResults({
          heatmapUrl,
          detectionsUrl
        });
        
        // Log URLs for debugging
        console.log('Heatmap URL:', heatmapUrl);
        console.log('Detections URL:', detectionsUrl);
        
        toast.success('Waste heatmap generated successfully');
      } else {
        toast.error('Failed to generate waste heatmap');
      }
    } catch (error) {
      console.error('Error generating waste heatmap:', error);
      toast.error('Error generating waste heatmap. Please try again.');
    } finally {
      setUploading(false);
    }
  };
  
  const handleClear = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setFile(null);
    setPreview(null);
    setResults(null);
  };
  
  const selectFileAlternative = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  
  return (
    <div className="space-y-6">
      {/* Image Upload Section */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
        <h3 className="text-lg font-medium text-gray-800 mb-2">Upload Floor Plan</h3>
        <p className="text-sm text-gray-500 mb-4">
          Upload a floor plan or kitchen layout image to map waste sources.
        </p>
        
        <div className="space-y-4">
          {/* File selection area */}
          <div 
            className={`border-2 border-dashed rounded-lg p-6 text-center ${
              preview ? 'border-restaurant-primary' : 'border-gray-300 hover:border-restaurant-primary'
            } transition-colors`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            {preview ? (
              <div className="space-y-4">
                <div className="flex flex-col items-center">
                  <img 
                    src={preview} 
                    alt="Floor Plan Preview" 
                    className="max-w-full max-h-64 object-contain mb-2 rounded-md" 
                  />
                  <p className="text-sm text-gray-600 truncate max-w-full">
                    {file?.name} ({Math.round((file?.size || 0) / 1024)}KB)
                  </p>
                </div>
                <Button
                  variant="outline"
                  onClick={handleClear}
                  className="text-red-500 border-red-300 hover:bg-red-50 hover:text-red-600"
                >
                  <X size={16} className="mr-2" />
                  Remove
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-col items-center">
                  <ImageIcon size={36} className="text-gray-400 mb-2" />
                  <p className="text-sm text-gray-600">
                    Drag and drop your image here, or click to browse
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    Supported formats: JPG, PNG, GIF (max 5MB)
                  </p>
                </div>
                <div className="mt-4">
                  {/* Hidden file input */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    onChange={handleFileChange}
                    accept="image/*"
                  />
                  <Button
                    variant="outline"
                    type="button" 
                    onClick={selectFileAlternative}
                    className="text-restaurant-primary border-restaurant-primary/30 hover:bg-restaurant-primary/10"
                  >
                    <Camera size={18} className="mr-2" />
                    Select Image
                  </Button>
                </div>
              </div>
            )}
          </div>
          
          {/* Upload button */}
          {file && (
            <Button
              className="w-full bg-restaurant-primary hover:bg-restaurant-primary/90 text-white"
              onClick={handleUpload}
              disabled={uploading}
            >
              {uploading ? (
                <>
                  <Loader2 size={16} className="mr-2 animate-spin" />
                  Generating Heatmap...
                </>
              ) : (
                <>
                  <Upload size={16} className="mr-2" />
                  Generate Waste Heatmap
                </>
              )}
            </Button>
          )}
        </div>
      </div>
      
      {/* Results Section */}
      {results && (
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-medium text-gray-800">Waste Mapping Results</h3>
            <div className="flex items-center text-green-600 text-sm">
              <CheckCircle size={16} className="mr-1" />
              Analysis Complete
            </div>
          </div>
          
          
          
          <div className="grid md:grid-cols-2 gap-6">
            {/* Heatmap */}
            <div className="space-y-2">
              <div className="bg-gray-50 p-3 rounded-md">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Waste Heatmap</h4>
                <img 
                  src={results.heatmapUrl} 
                  alt="Waste Heatmap" 
                  className="w-full rounded-md shadow-sm border border-gray-200" 
                  onError={(e) => {
                    console.error('Failed to load heatmap image');
                    e.currentTarget.style.border = '1px solid red';
                    e.currentTarget.style.padding = '10px';
                    e.currentTarget.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMiI+SW1hZ2UgZmFpbGVkIHRvIGxvYWQ8L3RleHQ+PC9zdmc+';
                  }}
                />
                <p className="text-xs text-gray-500 mt-2">
                  Colored overlay showing waste concentration areas.
                </p>
              </div>
            </div>
            
            {/* Detections */}
            <div className="space-y-2">
              <div className="bg-gray-50 p-3 rounded-md">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Waste Detections</h4>
                <img 
                  src={results.detectionsUrl} 
                  alt="Waste Detections" 
                  className="w-full rounded-md shadow-sm border border-gray-200" 
                  onError={(e) => {
                    console.error('Failed to load detections image');
                    e.currentTarget.style.border = '1px solid red';
                    e.currentTarget.style.padding = '10px';
                    e.currentTarget.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMiI+SW1hZ2UgZmFpbGVkIHRvIGxvYWQ8L3RleHQ+PC9zdmc+';
                  }}
                />
                <p className="text-xs text-gray-500 mt-2">
                  Identified waste objects with bounding boxes.
                </p>
              </div>
            </div>
          </div>
          
          <div className="mt-6 bg-blue-50 p-4 rounded-md">
            <h4 className="text-sm font-medium text-blue-700 mb-2">Analysis Summary</h4>
            <p className="text-sm text-blue-600">
              The analysis identified key waste generation areas in your kitchen or facility.
              Red zones indicate high waste concentration while green areas show minimal waste.
              Use this information to optimize your waste management strategies.
            </p>
          </div>
        </div>
      )}
      
      {/* Instructions Section */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
        <h3 className="text-lg font-medium text-gray-800 mb-3">How to Use Waste Mapping</h3>
        <ol className="list-decimal pl-5 space-y-2 text-gray-700">
          <li>Upload an image of your kitchen layout or floor plan</li>
          <li>Our AI will analyze the image to detect potential waste areas</li>
          <li>View the generated heatmap showing waste concentration zones</li>
          <li>Use the detection image to identify specific waste sources</li>
          <li>Implement targeted waste reduction strategies for high-waste areas</li>
        </ol>
        <div className="mt-4 p-3 bg-blue-50 rounded-md text-blue-700 text-sm">
          <p><strong>Pro Tip:</strong> For best results, use clear, well-lit images of your kitchen or storage areas. The more detailed the image, the more accurate the waste mapping will be.</p>
        </div>
      </div>
    </div>
  );
};

export default WasteMapping; 