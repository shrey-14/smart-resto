import React, { useState } from 'react';
import FileUploader from './FileUploader';
import SmartRestocking from '@/components/inventory/SmartRestocking';
import SalesUploadScreen from './SalesUploadScreen';
import CostEfficiencyScreen from './CostEfficiencyScreen';
import SpoilageDetection from '@/components/waste/SpoilageDetection';
import SpoilageForecast from '@/components/waste/SpoilageForecast';
import WasteCategorization from '@/components/waste/WasteCategorization';
import CreativeWasteAgent from '@/components/waste/CreativeWasteAgent';
import VisualInventoryScanner from '@/components/inventory/VisualInventoryScanner';
import LiveInventoryDetection from '@/components/inventory/LiveInventoryDetection';
import WasteMapping from '@/components/waste/WasteMapping';

interface UploadScreenProps {
  category: string;
  subCategory: string;
}

const UploadScreen = ({ category, subCategory }: UploadScreenProps) => {
  const [showUploadSection, setShowUploadSection] = useState(false);
  
  // Return appropriate upload components based on subcategory
  switch (subCategory) {
    case 'visual-stock-scanner':
      return <VisualInventoryScanner />;
      
    case 'live-inventory-detection':
      return <LiveInventoryDetection />;
      
    case 'smart-restocking':
      return (
        <div>
          {/* Always show SmartRestocking component first */}
          <SmartRestocking />
          
          {/* Optional upload section that can be toggled */}
          {showUploadSection && (
            <div className="mt-8">
              <h3 className="text-lg font-medium text-gray-800 mb-4">Upload Custom Data (Optional)</h3>
              
              <FileUploader 
                type="data"
                title="Upload Inventory Data"
                description="Upload your custom inventory data to receive more personalized restocking recommendations."
                allowedFileTypes=".csv,.xls,.xlsx"
              />
              
              <FileUploader 
                type="data"
                title="Upload Sales Data"
                description="Upload your sales data to improve accuracy of restocking suggestions."
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
      
    case 'sales-predictor':
      return <SalesUploadScreen />;
      
    case 'cost-efficiency-tools':
      return <CostEfficiencyScreen />;
      
    case 'spoilage-forecast':
      return <SpoilageForecast />;
    
    case 'spoilage-detection':
      return <SpoilageDetection />;
      
    case 'waste-categorization':
      return <WasteCategorization />;
      
    case 'creative-waste-agent':
      return <CreativeWasteAgent />;
      
    case 'waste-mapping':
      return <WasteMapping />;
      
    default:
      return null;
  }
};

export default UploadScreen;
