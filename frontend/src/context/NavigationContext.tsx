
import React, { createContext, useState, useContext, ReactNode } from 'react';

// Define types for our navigation items
export type NavigationCategory = 'inventory' | 'analytics' | 'waste-management';

export type SubCategory = {
  id: string;
  name: string;
  icon?: string;
};

type NavigationContextType = {
  activeCategory: NavigationCategory;
  activeSubCategory: string;
  isSidebarOpen: boolean;
  setActiveCategory: (category: NavigationCategory) => void;
  setActiveSubCategory: (subCategory: string) => void;
  toggleSidebar: () => void;
  getSubCategories: (category: NavigationCategory) => SubCategory[];
};

const NavigationContext = createContext<NavigationContextType | undefined>(undefined);

// Define subcategories for each main category
const subCategoriesMap: Record<NavigationCategory, SubCategory[]> = {
  'inventory': [
    { id: 'visual-stock-scanner', name: 'Visual Stock Scanner' },
    { id: 'live-inventory-detection', name: 'Live Inventory Detection' },
    { id: 'smart-restocking', name: 'Smart Restocking' }
  ],
  'analytics': [
    { id: 'sales-predictor', name: 'Sales Predictor' },
    { id: 'cost-efficiency-tools', name: 'Cost Efficiency Tools' },
    { id: 'profit-monitor', name: 'Daily Specials' }
  ],
  'waste-management': [
    { id: 'spoilage-forecast', name: 'Spoilage Forecast' },
    { id: 'spoilage-detection', name: 'Spoilage Detection' },
    { id: 'waste-categorization', name: 'Waste Categorization' },
    { id: 'waste-mapping', name: 'Waste Mapping' },
    { id: 'total-waste-overview', name: 'Waste Quantifier' },
    { id: 'creative-waste-agent', name: 'Creative Waste Agent' }
  ]
};

export const NavigationProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [activeCategory, setActiveCategory] = useState<NavigationCategory>('inventory');
  const [activeSubCategory, setActiveSubCategory] = useState<string>('visual-stock-scanner');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const toggleSidebar = () => {
    setIsSidebarOpen(prev => !prev);
  };

  const getSubCategories = (category: NavigationCategory): SubCategory[] => {
    return subCategoriesMap[category] || [];
  };

  return (
    <NavigationContext.Provider
      value={{
        activeCategory,
        activeSubCategory,
        isSidebarOpen,
        setActiveCategory,
        setActiveSubCategory,
        toggleSidebar,
        getSubCategories
      }}
    >
      {children}
    </NavigationContext.Provider>
  );
};

export const useNavigation = (): NavigationContextType => {
  const context = useContext(NavigationContext);
  if (!context) {
    throw new Error('useNavigation must be used within a NavigationProvider');
  }
  return context;
};
