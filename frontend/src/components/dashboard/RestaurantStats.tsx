
import React from 'react';
import { BarChart, Bar, XAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { restaurantStatsData } from '@/data/mockData';

const RestaurantStats = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
      {/* Daily summary */}
      <div className="bg-white rounded-lg shadow-sm p-5 border border-gray-100">
        <h3 className="text-sm font-medium text-gray-500 mb-4">Today's Summary</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-gray-500">Sales</p>
            <p className="text-xl font-semibold text-restaurant-primary">${restaurantStatsData.dailySales.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Orders</p>
            <p className="text-xl font-semibold text-restaurant-secondary">{restaurantStatsData.orderCount}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Avg. Order</p>
            <p className="text-xl font-semibold text-restaurant-primary">${restaurantStatsData.averageOrderValue.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Customers</p>
            <p className="text-xl font-semibold text-restaurant-secondary">{restaurantStatsData.customerCount}</p>
          </div>
        </div>
      </div>

      {/* Top selling items */}
      <div className="bg-white rounded-lg shadow-sm p-5 border border-gray-100">
        <h3 className="text-sm font-medium text-gray-500 mb-4">Top Selling Items</h3>
        <div className="space-y-3">
          {restaurantStatsData.topSellingItems.map((item, index) => (
            <div key={index} className="flex justify-between items-center">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-700">{item.name}</p>
                <p className="text-xs text-gray-500">{item.count} orders</p>
              </div>
              <p className="text-sm font-semibold text-restaurant-primary">${item.revenue.toFixed(2)}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Sales by hour */}
      <div className="bg-white rounded-lg shadow-sm p-5 border border-gray-100">
        <h3 className="text-sm font-medium text-gray-500 mb-2">Today's Sales by Hour</h3>
        <div className="h-36">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={restaurantStatsData.salesByHour}>
              <XAxis dataKey="hour" tick={{ fontSize: 10 }} />
              <Tooltip 
                formatter={(value) => [`$${value}`, 'Sales']}
                contentStyle={{ fontSize: 12 }}
              />
              <Bar dataKey="sales" fill="#8B0000" barSize={20} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default RestaurantStats;
