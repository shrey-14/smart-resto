import React, { useState, useEffect } from 'react';
import { detectStock } from '@/services/api';
import { RefreshCw, CheckCircle, ClipboardList } from 'lucide-react';
import { toast } from 'sonner';

interface StockItem {
  name: string;
  count: number;
}

interface StockDetectionDisplayProps {
  // videoUrl?: string;
  detectionResults?: Record<string, number>;
  isLoading?: boolean;
  progress?: number;
}

const StockDetectionDisplay: React.FC<StockDetectionDisplayProps> = ({
  // videoUrl,
  detectionResults,
  isLoading = false,
  progress = 0
}) => {
  const [stockItems, setStockItems] = useState<StockItem[]>([]);
  const [processing, setProcessing] = useState<boolean>(isLoading);
  // const [resultVideo, setResultVideo] = useState<string | null>(videoUrl || null);
  const [processingProgress, setProcessingProgress] = useState<number>(progress);
  
  // Process detection results when they change
  useEffect(() => {
    if (detectionResults && !isLoading) {
      console.log("Processing detection results:", detectionResults);
      const items = Object.entries(detectionResults).map(([name, count]) => ({
        name,
        count: Number(count)
      })).sort((a, b) => b.count - a.count);
      
      setStockItems(items);
      setProcessing(false);
    } else if (isLoading) {
      setProcessing(true);
    }
  }, [detectionResults, isLoading]);

  // Handle the video URL when it changes
  // useEffect(() => {
  //   if (videoUrl) {
  //     console.log("Setting video URL:", videoUrl);
  //     setResultVideo(videoUrl);
  //   }
  // }, [videoUrl]);
  
  // Update progress when it changes
  useEffect(() => {
    setProcessingProgress(progress);
  }, [progress]);

  // Get latest detection results
  const refreshResults = async () => {
    setProcessing(true);
    try {
      const response = await detectStock();
      if (response && response.results) {
        const items = Object.entries(response.results).map(([name, count]) => ({
          name,
          count: Number(count)
        })).sort((a, b) => b.count - a.count);
        
        setStockItems(items);
        
        // If there's a video URL in the response
        // if (response.video_url) {
        //   const videoUrlWithTimestamp = `${response.video_url}?t=${new Date().getTime()}`; // Add timestamp to bypass cache
        //   console.log("Setting refreshed video URL:", videoUrlWithTimestamp);
        //   setResultVideo(videoUrlWithTimestamp);
        // }
      }
    } catch (error) {
      console.error('Error fetching detection results:', error);
      toast.error('Failed to refresh results. Please try again.');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden">
      {/* Header Section */}
      <div className="p-4 border-b border-gray-100 flex justify-between items-center">
        <div>
          <h3 className="text-lg font-medium text-gray-800">Stock Detection Results</h3>
          <p className="text-sm text-gray-500">Real-time inventory detection from video feed</p>
        </div>
        <button
          onClick={refreshResults}
          disabled={processing}
          className="flex items-center px-3 py-1.5 text-sm rounded-md border border-gray-200 hover:bg-gray-50 transition-colors"
        >
          {processing ? (
            <RefreshCw className="w-4 h-4 mr-1.5 animate-spin" />
          ) : (
            <RefreshCw className="w-4 h-4 mr-1.5" />
          )}
          Refresh
        </button>
      </div>

      <div className="grid md:grid-cols-2 gap-4 p-4">
        {/* Video Section */}
        <div className="bg-gray-50 rounded-lg overflow-hidden">
          {stockItems.length > 0 ? (
            <video
              src="/static/output_video.mp4"
              controls
              className="w-full aspect-video object-cover"
              poster="/images/video-placeholder.jpg"
              // key={resultVideo}  // Add key to force re-render when URL changes
            />
          ) : (
            <div className="w-full aspect-video flex items-center justify-center bg-gray-100">
              <p className="text-gray-500">No detection video available</p>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex justify-between items-center mb-3">
            <h4 className="font-medium text-gray-700">Detected Items</h4>
            {stockItems.length > 0 && (
              <span className="text-xs bg-green-50 text-green-700 px-2 py-1 rounded-full flex items-center">
                <CheckCircle className="w-3 h-3 mr-1" />
                {stockItems.length} items found
              </span>
            )}
          </div>

          {processing ? (
            <div className="flex items-center justify-center h-60">
              <div className="flex flex-col items-center">
                <RefreshCw className="w-8 h-8 text-gray-400 animate-spin mb-2" />
                <p className="text-gray-500 text-sm">Processing video...</p>
                
                {/* Progress bar */}
                {processingProgress > 0 && (
                  <div className="w-full mt-4 max-w-xs">
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div 
                        className="bg-restaurant-primary h-2.5 rounded-full" 
                        style={{ width: `${processingProgress}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-500 text-center mt-1">
                      {processingProgress}% complete
                    </p>
                  </div>
                )}
              </div>
            </div>
          ) : stockItems.length > 0 ? (
            <div className="overflow-y-auto max-h-60">
              <table className="min-w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Item</th>
                    <th className="py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Count</th>
                  </tr>
                </thead>
                <tbody>
                  {stockItems.map((item, index) => (
                    <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="py-2 text-sm text-gray-700">{item.name}</td>
                      <td className="py-2 text-sm text-right font-medium text-gray-900">{item.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-60 text-center">
              <ClipboardList className="w-10 h-10 text-gray-300 mb-2" />
              <p className="text-gray-500">No items detected yet</p>
              <p className="text-gray-400 text-xs mt-1">Upload a video to start detection</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StockDetectionDisplay; 