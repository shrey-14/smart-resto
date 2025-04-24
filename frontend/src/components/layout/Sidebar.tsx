import React from 'react';
import { useNavigation } from '@/context/NavigationContext';
import { 
  ChevronRight, 
  Camera, 
  VideoIcon, 
  RefreshCw, 
  TrendingUp, 
  DollarSign, 
  BarChart, 
  Calendar, 
  ImageIcon, 
  FileStack, 
  Map, 
  LayoutDashboard, 
  ChefHat
} from 'lucide-react';
import { cn } from '@/lib/utils';

const getIconForCategory = (category: string) => {
  switch (category) {
    case 'inventory':
      return <Camera size={18} />;
    case 'analytics':
      return <BarChart size={18} />;
    case 'waste-management':
      return <FileStack size={18} />;
    default:
      return <ChevronRight size={18} />;
  }
};

const getIconForSubCategory = (category: string, subCategory: string) => {
  // Inventory subcategory icons
  if (category === 'inventory') {
    switch (subCategory) {
      case 'visual-stock-scanner':
        return <Camera size={18} />;
      case 'live-inventory-detection':
        return <VideoIcon size={18} />;
      case 'smart-restocking':
        return <RefreshCw size={18} />;
      default:
        return <ChevronRight size={18} />;
    }
  }
  
  // Analytics subcategory icons
  if (category === 'analytics') {
    switch (subCategory) {
      case 'sales-predictor':
        return <TrendingUp size={18} />;
      case 'cost-efficiency-tools':
        return <DollarSign size={18} />;
      case 'profit-monitor':
        return <BarChart size={18} />;
      default:
        return <ChevronRight size={18} />;
    }
  }
  
  // Waste Management subcategory icons
  if (category === 'waste-management') {
    switch (subCategory) {
      case 'spoilage-forecast':
        return <Calendar size={18} />;
      case 'spoilage-detection':
        return <ImageIcon size={18} />;
      case 'waste-categorization':
        return <FileStack size={18} />;
      case 'waste-mapping':
        return <Map size={18} />;
      case 'total-waste-overview':
        return <LayoutDashboard size={18} />;
      case 'creative-waste-agent':
        return <ChefHat size={18} />;
      default:
        return <ChevronRight size={18} />;
    }
  }
  
  return <ChevronRight size={18} />;
};

const Sidebar = () => {
  const { 
    activeCategory, 
    activeSubCategory, 
    isSidebarOpen, 
    getSubCategories, 
    setActiveSubCategory 
  } = useNavigation();
  
  const subCategories = getSubCategories(activeCategory);

  return (
    <div 
      className={cn(
        "fixed inset-y-0 left-0 bg-white border-r border-gray-200 transition-all duration-300 ease-in-out z-20",
        isSidebarOpen ? "w-64" : "w-0 md:w-16 overflow-hidden"
      )}
    >
      <div className="h-16">
        {/* Spacer to align with navbar */}
      </div>
      
      <div className="p-4">
        <div className="flex items-center mb-6">
          {isSidebarOpen ? (
            <h2 className="text-lg font-semibold text-gray-800 flex items-center">
              {getIconForCategory(activeCategory)}
              <span className="ml-2">{activeCategory.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</span>
            </h2>
          ) : (
            <div className="mx-auto">
              {getIconForCategory(activeCategory)}
            </div>
          )}
        </div>
        
        <nav>
          <ul className="space-y-1">
            {subCategories.map((subCategory) => (
              <li key={subCategory.id}>
                <button
                  onClick={() => setActiveSubCategory(subCategory.id)}
                  className={cn(
                    "w-full text-left px-3 py-2 rounded-md text-sm transition-colors flex items-center",
                    activeSubCategory === subCategory.id
                      ? "bg-restaurant-primary text-white"
                      : "text-gray-700 hover:bg-gray-100",
                    !isSidebarOpen && "justify-center"
                  )}
                >
                  {getIconForSubCategory(activeCategory, subCategory.id)}
                  {isSidebarOpen && (
                    <span className="ml-2">{subCategory.name}</span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </div>
  );
};

export default Sidebar;
