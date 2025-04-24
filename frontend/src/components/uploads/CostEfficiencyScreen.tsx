import React, { useState } from 'react';
import FileUploader from './FileUploader';
import CostEfficiencyTool from '@/components/cost/CostEfficiencyTool';

const CostEfficiencyScreen = () => {
  const [showUploadSection, setShowUploadSection] = useState(false);

  return (
    <div>
      {/* Always show CostEfficiencyTool component first */}
      <CostEfficiencyTool />
      
      {/* Optional upload section that can be toggled */}
      {showUploadSection && (
        <div className="mt-8">
          <h3 className="text-lg font-medium text-gray-800 mb-4">Upload Custom Data (Optional)</h3>
          
          <FileUploader 
            type="data"
            title="Upload Menu Data"
            description="Upload your menu items with ingredient costs to receive personalized cost optimization recommendations."
            allowedFileTypes=".csv,.xls,.xlsx"
          />
          
          <FileUploader 
            type="data"
            title="Upload Ingredient Pricing"
            description="Upload current market pricing for ingredients to improve cost optimization accuracy."
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

export default CostEfficiencyScreen; 