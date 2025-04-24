
import React from 'react';
import { aiInsightsData, inventoryAlertData } from '@/data/mockData';
import { AlertTriangle, Award, TrendingUp, Zap } from 'lucide-react';

const getInsightIcon = (type: string) => {
  switch (type) {
    case 'Opportunity':
      return <Award className="text-green-500" size={18} />;
    case 'Alert':
      return <AlertTriangle className="text-amber-500" size={18} />;
    case 'Trend':
      return <TrendingUp className="text-blue-500" size={18} />;
    case 'Efficiency':
      return <Zap className="text-purple-500" size={18} />;
    default:
      return <TrendingUp className="text-blue-500" size={18} />;
  }
};

const AIInsights = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
      <div className="md:col-span-2 bg-white rounded-lg shadow-sm p-5 border border-gray-100">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-gray-700">AI Insights & Recommendations</h3>
          <span className="text-xs px-2 py-1 bg-restaurant-accent/20 text-restaurant-primary rounded-full">Powered by AI</span>
        </div>
        
        <div className="space-y-4">
          {aiInsightsData.map(insight => (
            <div 
              key={insight.id} 
              className="p-3 rounded-md border border-gray-100 hover:shadow-md transition-shadow"
            >
              <div className="flex items-start">
                <div className="mr-3 mt-0.5">
                  {getInsightIcon(insight.type)}
                </div>
                <div className="flex-1">
                  <div className="flex justify-between items-center mb-1">
                    <h4 className="text-sm font-medium text-gray-800">{insight.title}</h4>
                    <span className="text-xs text-gray-500">{insight.type}</span>
                  </div>
                  <p className="text-xs text-gray-600 mb-2">{insight.description}</p>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-restaurant-secondary">{insight.potentialImpact}</span>
                    <span className="text-gray-500">AI Confidence: {insight.aiConfidence}%</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="bg-white rounded-lg shadow-sm p-5 border border-gray-100">
        <h3 className="text-sm font-medium text-gray-700 mb-4">Inventory Alerts</h3>
        
        {inventoryAlertData.map((alert, index) => (
          <div key={index} className="mb-3 last:mb-0">
            <div className="flex justify-between items-center mb-1">
              <p className="text-sm font-medium text-gray-700">{alert.item}</p>
              <span 
                className={`text-xs px-2 py-0.5 rounded-full ${
                  alert.status === 'critical' 
                    ? 'bg-red-100 text-red-600' 
                    : 'bg-amber-100 text-amber-600'
                }`}
              >
                {alert.status}
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${
                  alert.status === 'critical' ? 'bg-red-500' : 'bg-amber-500'
                }`}
                style={{ width: `${(alert.currentStock / 10) * 100}%` }}
              ></div>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              {alert.currentStock} {alert.unit} remaining
            </p>
          </div>
        ))}
        
        <div className="mt-4 pt-4 border-t border-gray-100">
          <button className="w-full px-3 py-2 text-xs font-medium text-restaurant-primary border border-restaurant-primary rounded-md hover:bg-restaurant-primary hover:text-white transition-colors">
            Run AI Inventory Optimization
          </button>
        </div>
      </div>
    </div>
  );
};

export default AIInsights;
