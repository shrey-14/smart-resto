
import React from 'react';
import { recentOrdersData } from '@/data/mockData';
import { Clock, CheckCircle, RefreshCw, ChefHat } from 'lucide-react';

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'Completed':
      return <CheckCircle className="text-green-500" size={16} />;
    case 'In Progress':
      return <RefreshCw className="text-blue-500" size={16} />;
    case 'Preparing':
      return <ChefHat className="text-amber-500" size={16} />;
    case 'Ready':
      return <Clock className="text-purple-500" size={16} />;
    default:
      return null;
  }
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'Completed':
      return 'bg-green-100 text-green-800';
    case 'In Progress':
      return 'bg-blue-100 text-blue-800';
    case 'Preparing':
      return 'bg-amber-100 text-amber-800';
    case 'Ready':
      return 'bg-purple-100 text-purple-800';
    default:
      return 'bg-gray-100 text-gray-800';
  }
};

const RecentOrders = () => {
  return (
    <div className="bg-white rounded-lg shadow-sm p-5 border border-gray-100">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-medium text-gray-700">Recent Orders</h3>
        <button className="text-xs text-restaurant-primary hover:underline">View All</button>
      </div>
      
      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead>
            <tr className="text-left text-xs text-gray-500 border-b border-gray-100">
              <th className="pb-2 font-medium">Order ID</th>
              <th className="pb-2 font-medium">Customer</th>
              <th className="pb-2 font-medium">Items</th>
              <th className="pb-2 font-medium">Total</th>
              <th className="pb-2 font-medium">Status</th>
              <th className="pb-2 font-medium">Time</th>
            </tr>
          </thead>
          <tbody>
            {recentOrdersData.map((order) => (
              <tr key={order.id} className="border-b border-gray-50 hover:bg-gray-50">
                <td className="py-3 text-xs font-medium text-gray-800">{order.id}</td>
                <td className="py-3 text-xs text-gray-600">{order.customer}</td>
                <td className="py-3 text-xs text-gray-600">
                  <div className="flex flex-col">
                    {order.items.slice(0, 2).map((item, i) => (
                      <span key={i}>{item}</span>
                    ))}
                    {order.items.length > 2 && (
                      <span className="text-gray-400">+{order.items.length - 2} more</span>
                    )}
                  </div>
                </td>
                <td className="py-3 text-xs font-medium text-gray-800">${order.total.toFixed(2)}</td>
                <td className="py-3">
                  <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs ${getStatusColor(order.status)}`}>
                    <span className="mr-1">{getStatusIcon(order.status)}</span>
                    {order.status}
                  </span>
                </td>
                <td className="py-3 text-xs text-gray-600">{order.time}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default RecentOrders;
