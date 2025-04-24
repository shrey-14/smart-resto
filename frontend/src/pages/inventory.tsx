import React from 'react';
import SimpleInventoryScanner from '@/components/inventory/SimpleInventoryScanner';

const InventoryPage = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-800 mb-8">Inventory Management</h1>
      <SimpleInventoryScanner />
    </div>
  );
};

export default InventoryPage; 