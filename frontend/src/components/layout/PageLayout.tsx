
import React, { ReactNode } from 'react';
import Navbar from './Navbar';
import Sidebar from './Sidebar';
import { useNavigation } from '@/context/NavigationContext';
import { cn } from '@/lib/utils';

interface PageLayoutProps {
  children: ReactNode;
}

const PageLayout = ({ children }: PageLayoutProps) => {
  const { isSidebarOpen } = useNavigation();

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      <Navbar />
      <div className="flex flex-1 relative">
        <Sidebar />
        <main 
          className={cn(
            "flex-1 transition-all duration-300 p-6",
            isSidebarOpen ? "ml-64" : "ml-0 md:ml-16"
          )}
        >
          <div className="animate-fade-in">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

export default PageLayout;
