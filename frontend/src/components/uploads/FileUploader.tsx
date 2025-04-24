import React, { useState } from 'react';
import { Upload, Camera, X, Check, FileUp, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { 
  uploadInventoryData, 
  uploadSalesData, 
  uploadMenuData, 
  uploadIngredientData 
} from '@/services/api';

interface FileUploaderProps {
  type: 'image' | 'data' | 'camera';
  title: string;
  description: string;
  allowedFileTypes?: string;
  onUploadSuccess?: (data: any) => void;
  endpoint?: string;
}

const FileUploader = ({ 
  type, 
  title, 
  description, 
  allowedFileTypes = "image/*",
  onUploadSuccess,
  endpoint
}: FileUploaderProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    
    setFile(selectedFile);
    
    // Create preview for images
    if (type === 'image' && selectedFile.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setPreview(event.target?.result as string);
      };
      reader.readAsDataURL(selectedFile);
    } else {
      setPreview(null);
    }
  };
  
  const handleUpload = async () => {
    if (!file) return;
    
    setUploading(true);
    
    try {
      // If we're in the smart restocking section and uploading inventory data
      if (type === 'data' && title.includes('Inventory Data')) {
        const response = await uploadInventoryData(file);
        if (response.status === 'success') {
          toast.success('Inventory data uploaded successfully!');
          if (onUploadSuccess) {
            onUploadSuccess(response);
          }
        }
      } 
      // If we're in the sales predictor section and uploading sales data
      else if (type === 'data' && (title.includes('Sales Data') || title.includes('Seasonal Data'))) {
        const response = await uploadSalesData(file);
        if (response.status === 'success') {
          toast.success('Sales data uploaded successfully!');
          if (onUploadSuccess) {
            onUploadSuccess(response);
          }
        }
      }
      // If we're in the cost efficiency section and uploading menu data
      else if (type === 'data' && title.includes('Menu Data')) {
        const response = await uploadMenuData(file);
        if (response.status === 'success') {
          toast.success('Menu data uploaded successfully!');
          if (onUploadSuccess) {
            onUploadSuccess(response);
          }
        }
      }
      // If we're in the cost efficiency section and uploading ingredient pricing data
      else if (type === 'data' && title.includes('Ingredient Pricing')) {
        const response = await uploadIngredientData(file);
        if (response.status === 'success') {
          toast.success('Ingredient pricing data uploaded successfully!');
          if (onUploadSuccess) {
            onUploadSuccess(response);
          }
        }
      }
      else {
        // Simulate upload process for other file types (for now)
        await new Promise(resolve => setTimeout(resolve, 1500));
        toast.success('File uploaded successfully!');
      }
      
      setUploadComplete(true);
      
      // Reset the complete status after a delay
      setTimeout(() => {
        setUploadComplete(false);
      }, 3000);
    } catch (error) {
      // Error is already handled in the API service
      console.error('Upload error:', error);
    } finally {
      setUploading(false);
    }
  };
  
  const handleClear = () => {
    setFile(null);
    setPreview(null);
  };
  
  const getButtonIcon = () => {
    if (type === 'image') return <Camera size={18} className="mr-2" />;
    if (type === 'camera') return <Camera size={18} className="mr-2" />;
    return <FileUp size={18} className="mr-2" />;
  };
  
  const getButtonText = () => {
    if (type === 'image') return 'Select Image';
    if (type === 'camera') return 'Connect Camera';
    return 'Select File';
  };
  
  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100 mb-6">
      <h3 className="text-lg font-medium text-gray-800 mb-2">{title}</h3>
      <p className="text-sm text-gray-500 mb-4">{description}</p>
      
      <div className="space-y-4">
        {/* File selection area */}
        <div 
          className={`border-2 border-dashed rounded-lg p-6 text-center ${
            preview ? 'border-restaurant-primary' : 'border-gray-300 hover:border-restaurant-primary'
          } transition-colors`}
        >
          {preview ? (
            <div className="space-y-4">
              <div className="flex flex-col items-center">
                <img src={preview} alt="Preview" className="max-w-full max-h-48 object-contain mb-2" />
                <p className="text-sm text-gray-600 truncate max-w-full">
                  {file?.name} ({Math.round(file?.size / 1024)}KB)
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
                <Upload size={36} className="text-gray-400 mb-2" />
                <p className="text-sm text-gray-600">
                  Drag and drop your file here, or click to browse
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  {type === 'image' 
                    ? 'Supported formats: JPG, PNG, GIF' 
                    : 'Supported formats: CSV, XLS, XLSX'}
                </p>
              </div>
              <div className="mt-4">
                <label className="cursor-pointer">
                  <input
                    type="file"
                    className="hidden"
                    onChange={handleFileChange}
                    accept={allowedFileTypes}
                  />
                  <Button
                    variant="outline"
                    className="text-restaurant-primary border-restaurant-primary/30 hover:bg-restaurant-primary/10"
                  >
                    {getButtonIcon()}
                    {getButtonText()}
                  </Button>
                </label>
              </div>
            </div>
          )}
        </div>
        
        {/* Upload button */}
        {file && (
          <Button
            className="w-full bg-restaurant-primary hover:bg-restaurant-primary/90 text-white"
            onClick={handleUpload}
            disabled={uploading || uploadComplete}
          >
            {uploading ? (
              <>
                <Loader2 size={16} className="mr-2 animate-spin" />
                Uploading...
              </>
            ) : uploadComplete ? (
              <>
                <Check size={16} className="mr-2" />
                Uploaded
              </>
            ) : (
              <>
                <Upload size={16} className="mr-2" />
                Upload {type === 'image' ? 'Image' : 'File'}
              </>
            )}
          </Button>
        )}
      </div>
    </div>
  );
};

export default FileUploader;
