import React from 'react';
import { Menu, Bell, Search, User, LayoutDashboard, Package, BarChart2 } from 'lucide-react';
import { NavigationCategory, useNavigation } from '@/context/NavigationContext';

const navItems: { id: NavigationCategory; label: string; icon: React.ElementType }[] = [
  { id: 'inventory', label: 'Inventory', icon: Package },
  { id: 'analytics', label: 'Analytics', icon: BarChart2 },
  { id: 'waste-management', label: 'Waste Management', icon: LayoutDashboard }
];

const Navbar = () => {
  const { activeCategory, setActiveCategory, toggleSidebar, setActiveSubCategory, getSubCategories } = useNavigation();

  const handleCategoryChange = (category: NavigationCategory) => {
    setActiveCategory(category);
    // Set the first subcategory as active when changing categories
    const subcategories = getSubCategories(category);
    if (subcategories.length > 0) {
      setActiveSubCategory(subcategories[0].id);
    }
  };

  return (
    <div className="bg-white border-b border-gray-200 sticky top-0 z-30 shadow-sm">
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center space-x-4">
          <button
            onClick={toggleSidebar}
            className="p-2 rounded-md hover:bg-gray-100 text-gray-600 focus:outline-none"
            aria-label="Toggle sidebar"
          >
            <Menu size={20} />
          </button>
          <div className="flex items-center">
            <span className="text-restaurant-primary font-bold text-xl">Smart</span>
            <span className="text-restaurant-secondary font-bold text-xl">Resto</span>
          </div>
        </div>

        <nav className="hidden md:flex space-x-2">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => handleCategoryChange(item.id)}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors flex items-center ${
                activeCategory === item.id
                  ? 'text-restaurant-primary border-b-2 border-restaurant-primary'
                  : 'text-gray-700 hover:text-restaurant-primary hover:bg-gray-50'
              }`}
            >
              <item.icon className="mr-2" size={16} />
              {item.label}
            </button>
          ))}
        </nav>

        <div className="flex items-center space-x-3">
          <button className="p-2 rounded-full hover:bg-gray-100 text-gray-600 focus:outline-none">
            <Search size={18} />
          </button>
          <button className="p-2 rounded-full hover:bg-gray-100 text-gray-600 focus:outline-none relative">
            <Bell size={18} />
            <span className="absolute top-0 right-0 w-2 h-2 bg-restaurant-primary rounded-full"></span>
          </button>
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-restaurant-primary rounded-full flex items-center justify-center text-white">
              <User size={16} />
            </div>
            <span className="text-sm font-medium text-gray-700 hidden sm:inline-block">Admin</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
