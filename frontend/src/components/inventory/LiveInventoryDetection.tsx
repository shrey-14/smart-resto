import React, { useState, useRef } from 'react';
import { Upload, Camera, X, Check, VideoIcon, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { detectStock, getTaskStatus } from '@/services/api';
import StockDetectionDisplay from './StockDetectionDisplay';

const LiveInventoryDetection: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [detectionResults, setDetectionResults] = useState<Record<string, number> | null>(null);
  // const [detectionVideo, setDetectionVideo] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log('File input change detected');
    const selectedFile = e.target.files?.[0];
    
    if (!selectedFile) {
      console.log('No file selected');
      return;
    }
    
    console.log('Selected file:', selectedFile.name, 'Type:', selectedFile.type);
    
    // Check if file is a video
    if (!selectedFile.type.startsWith('video/')) {
      console.log('File is not a video type:', selectedFile.type);
      toast.error('Please upload a video file');
      return;
    }
    
    setFile(selectedFile);
    
    // Create a preview for the video
    const videoUrl = URL.createObjectURL(selectedFile);
    console.log('Created preview URL:', videoUrl);
    setPreview(videoUrl);
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
    
    // Check if file is a video
    if (!droppedFile.type.startsWith('video/')) {
      toast.error('Please upload a video file');
      return;
    }
    
    setFile(droppedFile);
    
    // Create a preview for the video
    const videoUrl = URL.createObjectURL(droppedFile);
    setPreview(videoUrl);
  };
  
  const handleUpload = async () => {
    if (!file) return;
    
    setUploading(true);
    setProgress(0);
    
    try {
      const response = await detectStock(file);
      
      if (response.status === 'success') {
        toast.success('Video upload successful, processing started');
        
        // Check if the server is processing the video
        if (response.task_id) {
          // Start polling for task status
          let pollInterval: number;
          
          const pollTaskStatus = async () => {
            try {
              // Use the getTaskStatus function to poll
              const statusData = await getTaskStatus(response.task_id);
              console.log('Task status:', statusData);
              
              // Update progress
              if (statusData.progress) {
                console.log(`Processing progress: ${statusData.progress}%`);
                setProgress(statusData.progress);
              }
              
              if (statusData.status === 'completed' && statusData.result) {
                // Clear the polling interval
                clearInterval(pollInterval);
                
                // Set the results
                setDetectionResults(statusData.result.results);
                
                // Ensure the video URL includes a timestamp to avoid caching issues
                // let videoUrl = statusData.result.video_url;
                // if (videoUrl) {
                //   videoUrl = `${videoUrl}?t=${new Date().getTime()}`;
                //   console.log('Setting video URL:', videoUrl);
                //   setDetectionVideo(videoUrl);
                // } else {
                //   console.warn('No video URL in task result');
                // }
                
                setUploadComplete(true);
                setUploading(false);
                toast.success('Video processing completed!');
              } else if (statusData.status === 'failed') {
                // Clear the polling interval
                clearInterval(pollInterval);
                setUploading(false);
                toast.error(`Processing failed: ${statusData.error || 'Unknown error'}`);
              }
            } catch (error) {
              console.error('Error polling for task status:', error);
              // Don't stop polling on error, just log it
            }
          };
          
          // Poll every 3 seconds
          pollInterval = window.setInterval(pollTaskStatus, 3000);
          
          // Set a timeout to stop polling after 2 minutes
          setTimeout(() => {
            clearInterval(pollInterval);
            if (!uploadComplete) {
              toast.error('Video processing taking too long, please try again later');
              setUploading(false);
            }
          }, 120000); // 2 minutes
        } else {
          // Results are already available
          if (response.results) {
            setDetectionResults(response.results);
          }
          
          // Set detection video URL if available
          // if (response.video_url) {
          //   const videoUrl = `${response.video_url}?t=${new Date().getTime()}`;
          //   console.log('Setting immediate video URL:', videoUrl);
          //   setDetectionVideo(videoUrl);
          // }
          
          setUploadComplete(true);
          setUploading(false);
        }
      } else {
        toast.error('Error processing video');
        setUploading(false);
      }
    } catch (error) {
      console.error('Error uploading video:', error);
      toast.error('Error uploading video. Please try again.');
      setUploading(false);
    }
  };
  
  const handleClear = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setFile(null);
    setPreview(null);
    setUploadComplete(false);
    setDetectionResults(null);
    // setDetectionVideo(null);
    setProgress(0);
  };
  
  const selectFileAlternative = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  
  return (
    <div>
      {/* Video Upload Section */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100 mb-6">
        <h3 className="text-lg font-medium text-gray-800 mb-2">Upload Inventory Video</h3>
        <p className="text-sm text-gray-500 mb-4">
          Upload a video of your storage area for real-time inventory detection.
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
                  <video 
                    src={preview} 
                    className="max-w-full max-h-64 object-contain mb-2" 
                    controls
                  />
                  <p className="text-sm text-gray-600 truncate max-w-full">
                    {file?.name} ({Math.round((file?.size || 0) / 1024 / 1024)}MB)
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
                  <VideoIcon size={36} className="text-gray-400 mb-2" />
                  <p className="text-sm text-gray-600">
                    Drag and drop your video here, or click to browse
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    Supported formats: MP4, MOV, AVI (max 100MB)
                  </p>
                </div>
                <div className="mt-4">
                  {/* Hidden file input */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    onChange={handleFileChange}
                    accept="video/*"
                  />
                  <Button
                    variant="outline"
                    type="button" 
                    onClick={selectFileAlternative}
                    className="text-restaurant-primary border-restaurant-primary/30 hover:bg-restaurant-primary/10"
                  >
                    <Camera size={18} className="mr-2" />
                    Select Video
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
              disabled={uploading || uploadComplete}
            >
              {uploading ? (
                <>
                  <Loader2 size={16} className="mr-2 animate-spin" />
                  Processing Video...
                </>
              ) : uploadComplete ? (
                <>
                  <Check size={16} className="mr-2" />
                  Video Processed
                </>
              ) : (
                <>
                  <Upload size={16} className="mr-2" />
                  Process Video
                </>
              )}
            </Button>
          )}

          {/* Debug message */}
          <div className="text-xs text-gray-400 mt-2 text-center">
            Click "Select Video" to choose a video file from your device.
            {file && <p className="text-green-500">✓ File selected: {file.name}</p>}
            {!file && (
              <p 
                className="text-blue-500 mt-2 cursor-pointer" 
                onClick={selectFileAlternative}
              >
                If "Select Video" doesn't work, click here to try an alternative method
              </p>
            )}
          </div>
        </div>
      </div>
      
      {/* Detection Results Section */}
      {(detectionResults || uploading) && (
        <StockDetectionDisplay 
          // videoUrl={detectionVideo || undefined}
          detectionResults={detectionResults || undefined}
          isLoading={uploading}
          progress={progress}
        />
      )}
      
      {/* Instructions Section */}
      <div className="mt-8 p-6 bg-white rounded-lg shadow-sm border border-gray-100">
        <h3 className="text-lg font-medium text-gray-800 mb-3">How Stock Detection Works</h3>
        <ol className="list-decimal pl-5 space-y-2 text-gray-700">
          <li>Upload a video of your storage area or kitchen inventory.</li>
          <li>Our AI system will process the video and detect all food items present.</li>
          <li>You'll receive a detailed count of all detected items.</li>
          <li>The system will also provide an annotated video highlighting all detected items.</li>
          <li>Use these results to track your inventory in real-time.</li>
        </ol>
        <div className="mt-4 p-3 bg-blue-50 rounded-md text-blue-700 text-sm">
          <p><strong>Pro Tip:</strong> For best results, ensure your video has good lighting and a clear view of all items. Move the camera slowly across your storage area for more accurate detection.</p>
        </div>
      </div>
    </div>
  );
};

export default LiveInventoryDetection; 