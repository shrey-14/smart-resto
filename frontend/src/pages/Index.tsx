import React from 'react';
import PageLayout from '@/components/layout/PageLayout';
import { NavigationProvider, useNavigation } from '@/context/NavigationContext';
import { Brain, ChefHat } from 'lucide-react';
import UploadScreen from '@/components/uploads/UploadScreen';
import DailySpecials from '@/components/dashboard/DailySpecials';
import WasteQuantifier from '@/components/waste/WasteQuantifier';
import { cn } from '@/lib/utils';

const Index = () => {
  return (
    <NavigationProvider>
      <PageLayout>
        <IndexContent />
      </PageLayout>
    </NavigationProvider>
  );
};

// Feature descriptions for each subcategory
const featureDescriptions = {
  'inventory': {
    'visual-stock-scanner': {
      title: 'Visual Stock Scanner',
      description: 'Upload photos of your inventory shelves to automatically identify and log the ingredients present in the image.',
      input: 'Upload clear images of your storage areas, shelves, or refrigerator.',
      output: 'Automated inventory list with identified items, quantities, and confidence scores.'
    },
    'live-inventory-detection': {
      title: 'Live Inventory Detection',
      description: 'Connect to a live camera feed to detect and track your current stock in real time using visual inputs.',
      input: 'Connect a camera feed from your storage area or use your device camera.',
      output: 'Real-time inventory tracking dashboard with alerts for low stock items.'
    },
    'smart-restocking': {
      title: 'Smart Restocking',
      description: 'Based on your inventory trends and usage history, get intelligent suggestions on what items to restock and in what quantity.',
      input: 'Your historical inventory data and sales information.',
      output: 'Optimized restocking recommendations to minimize waste and shortages.'
    }
  },
  'analytics': {
    'sales-predictor': {
      title: 'Sales Predictor',
      description: 'Use past sales data to forecast future demand, helping you prepare and stock more efficiently.',
      input: 'Historical sales data over a minimum period of 3 months.',
      output: 'Sales forecasts for upcoming days, weeks, or months with confidence intervals.'
    },
    'cost-efficiency-tools': {
      title: 'Cost Efficiency Tools',
      description: 'Enter your dish recipes and ingredient prices to calculate real-time dish costs and get suggestions to optimize your menu.',
      input: 'Recipes with ingredients and current market prices.',
      output: 'Dish cost breakdown, profit margins, and menu optimization suggestions.'
    },
    'profit-monitor': {
      title: 'Daily Specials',
      description: 'View a dynamic dashboard that visually represents your current profits, losses, trends, and actionable financial insights.',
      input: 'Connect your POS system or upload sales and expense data.',
      output: 'Comprehensive financial dashboard with trends, warnings, and actionable insights.'
    }
  },
  'waste-management': {
    'spoilage-forecast': {
      title: 'Spoilage Forecast',
      description: 'Based on expiry data and inventory age, predict which items are most likely to go bad soon.',
      input: 'Inventory list with purchase dates and expiry information.',
      output: 'Prioritized list of items at risk of spoilage with recommended actions.'
    },
    'spoilage-detection': {
      title: 'Spoilage Detection',
      description: 'Upload images or connect a sensor feed to detect spoiled items automatically using image recognition or sensor analysis.',
      input: 'Images of food items or sensor data (temperature, humidity, etc.).',
      output: 'Spoilage alerts with confidence scores and suggested actions.'
    },
    'waste-categorization': {
      title: 'Waste Categorization',
      description: 'Upload logs or photos of waste to classify it into categories like organic, expired, or overcooked food waste.',
      input: 'Waste logs or images of kitchen waste.',
      output: 'Categorized waste report with insights on reducing each type of waste.'
    },
    'waste-mapping': {
      title: 'Waste Mapping',
      description: 'Generate a visual heatmap showing where and how much waste is being generated across different areas of your kitchen or storage.',
      input: 'Waste logs with location data or manual input of waste by location.',
      output: 'Visual heatmap identifying high-waste areas in your establishment.'
    },
    'total-waste-overview': {
      title: 'Waste Quantifier',
      description: 'View a consolidated dashboard that summarizes your historical waste trends and current metrics in one place.',
      input: 'Historical waste data and current inventory information.',
      output: 'Comprehensive waste metrics dashboard with trend analysis and improvement suggestions.'
    },
    'creative-waste-agent': {
      title: 'Creative Waste Agent',
      description: 'Let an AI assistant suggest unique recipes using ingredients close to expiry, turning potential waste into delicious dishes.',
      input: 'Your inventory list with expiry dates highlighted.',
      output: 'Creative recipe suggestions that utilize at-risk ingredients, complete with preparation instructions.'
    }
  }
};

// Determine if a subcategory needs file upload functionality
const needsUploadButton = (category: string, subCategory: string) => {
  const uploadCategories = [
    'visual-stock-scanner',
    'live-inventory-detection',
    'smart-restocking',
    'sales-predictor',
    'cost-efficiency-tools',
    'spoilage-forecast',
    'spoilage-detection',
    'waste-categorization',
    'waste-mapping',
    'creative-waste-agent'
  ];
  
  return uploadCategories.includes(subCategory);
};

// Content changes based on active category and subcategory
const IndexContent = () => {
  const { activeCategory, activeSubCategory } = useNavigation();
  
  // Get the feature description for the active category and subcategory
  const featureInfo = featureDescriptions[activeCategory]?.[activeSubCategory];
  
  return (
    <div className="max-w-7xl mx-auto">
      {/* Page header - Now shows the selected tab name instead of "Smart Restaurant Management" */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">
            {featureInfo?.title || 
             activeSubCategory.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
          </h1>
          <p className="text-sm text-gray-500">
            {featureInfo?.description || 
             "AI-powered tools to optimize your restaurant operations"}
          </p>
        </div>
        {/* <div className="hidden sm:flex items-center space-x-3">
          <button className="px-4 py-2 bg-restaurant-primary text-white rounded-md text-sm font-medium hover:bg-restaurant-primary/90 transition-colors flex items-center">
            <Brain size={16} className="mr-2" />
            Ask AI Assistant
          </button>
          <button className="px-4 py-2 border border-restaurant-secondary text-restaurant-secondary rounded-md text-sm font-medium hover:bg-gray-100 transition-colors flex items-center">
            <ChefHat size={16} className="mr-2" />
            Kitchen View
          </button>
        </div> */}
      </div>
      
      {/* Dashboard content */}
      <div className="animate-fade-in">
        {/* Show upload component for specific subcategories */}
        {needsUploadButton(activeCategory, activeSubCategory) && (
          <UploadScreen category={activeCategory} subCategory={activeSubCategory} />
        )}
        
        {/* Render DailySpecials component for profit-monitor subcategory */}
        {activeSubCategory === 'profit-monitor' && (
          <div className="mb-8">
            <DailySpecials />
          </div>
        )}
        
        {/* Render WasteQuantifier component for total-waste-overview subcategory */}
        {activeSubCategory === 'total-waste-overview' && (
          <div className="mb-8">
            <WasteQuantifier />
          </div>
        )}
        
        {/* Only show the 'How It Works' section if we're not showing the Waste Quantifier dashboard */}
        {activeSubCategory !== 'total-waste-overview' && (
          <div className="mt-8 p-6 bg-white rounded-lg shadow-sm border border-gray-100">
            <h2 className="text-xl font-semibold mb-2 text-restaurant-primary">
              How It Works
            </h2>
            
            <div className="mb-4 pb-4 border-b border-gray-100">
              <p className="text-gray-700">
                {featureInfo?.description || 
                 "This feature will help you optimize your restaurant operations with AI-powered insights."}
              </p>
            </div>
            
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gray-50 p-4 rounded-md">
                <h3 className="font-medium text-gray-800 mb-2">What you'll need to provide:</h3>
                <p className="text-gray-600">
                  {featureInfo?.input || 
                   "Relevant restaurant data to help the AI generate accurate insights."}
                </p>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-md">
                <h3 className="font-medium text-gray-800 mb-2">What you'll receive:</h3>
                <p className="text-gray-600">
                  {featureInfo?.output || 
                   "Actionable insights to help you make better decisions for your restaurant."}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Index;
