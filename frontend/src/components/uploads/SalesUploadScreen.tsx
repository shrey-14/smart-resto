import React, { useState } from 'react';
import FileUploader from './FileUploader';
import SalesPredictor from '@/components/sales/SalesPredictor';

const SalesUploadScreen = () => {
  const [showUploadSection, setShowUploadSection] = useState(false);

  return (
    <div>
      {/* Always show SalesPredictor component first */}
      <SalesPredictor />
      
      {/* Optional upload section that can be toggled */}
      {showUploadSection && (
        <div className="mt-8">
          <h3 className="text-lg font-medium text-gray-800 mb-4">Upload Custom Data (Optional)</h3>
          
          <FileUploader 
            type="data"
            title="Upload Sales Data"
            description="Upload your historical sales data to improve forecast accuracy."
            allowedFileTypes=".csv,.xls,.xlsx"
          />
          
          <FileUploader 
            type="data"
            title="Upload Seasonal Data"
            description="Upload seasonal factors or special event data to enhance predictions."
            allowedFileTypes=".csv,.xls,.xlsx"
          />
        </div>
      )}
      
      {/* Toggle button for upload section */}
      <div className="mt-4 text-center">
        <button 
          onClick={() => setShowUploadSection(!showUploadSection)}
          className="text-restaurant-primary underline text-sm hover:text-restaurant-primary/80"
        >
          {showUploadSection ? 'Hide' : 'Show'} custom data upload options
        </button>
      </div>
    </div>
  );
};

export default SalesUploadScreen; 