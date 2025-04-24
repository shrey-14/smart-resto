
export const recentOrdersData = [
  {
    id: 'ORD-001',
    customer: 'Sarah Johnson',
    items: ['Grilled Salmon', 'Caesar Salad', 'Iced Tea'],
    total: 42.99,
    status: 'Completed',
    time: '12:30 PM'
  },
  {
    id: 'ORD-002',
    customer: 'Michael Chen',
    items: ['Margherita Pizza', 'Tiramisu', 'Sparkling Water'],
    total: 37.50,
    status: 'In Progress',
    time: '12:45 PM'
  },
  {
    id: 'ORD-003',
    customer: 'Jessica Williams',
    items: ['Chicken Parmesan', 'Garlic Bread', 'Soda'],
    total: 31.25,
    status: 'Preparing',
    time: '1:00 PM'
  },
  {
    id: 'ORD-004',
    customer: 'David Rodriguez',
    items: ['Beef Burrito', 'Chips & Salsa', 'Margarita'],
    total: 28.75,
    status: 'Ready',
    time: '1:15 PM'
  }
];

export const restaurantStatsData = {
  dailySales: 3247.89,
  orderCount: 78,
  averageOrderValue: 41.64,
  customerCount: 64,
  topSellingItems: [
    { name: 'Margherita Pizza', count: 14, revenue: 294.00 },
    { name: 'Grilled Salmon', count: 12, revenue: 275.88 },
    { name: 'Chicken Parmesan', count: 9, revenue: 179.91 }
  ],
  salesByHour: [
    { hour: '10 AM', sales: 245.75 },
    { hour: '11 AM', sales: 387.25 },
    { hour: '12 PM', sales: 856.50 },
    { hour: '1 PM', sales: 742.15 },
    { hour: '2 PM', sales: 431.80 },
    { hour: '3 PM', sales: 293.45 },
    { hour: '4 PM', sales: 197.25 }
  ]
};

export const aiInsightsData = [
  {
    id: 1,
    type: 'Opportunity',
    title: 'Menu Optimization',
    description: 'Based on sales patterns, consider featuring "Grilled Salmon" as a weekend special to increase profitability.',
    potentialImpact: 'Estimated 8% increase in weekend revenue',
    aiConfidence: 87
  },
  {
    id: 2,
    type: 'Alert',
    title: 'Inventory Warning',
    description: 'Fresh tomatoes will likely run out before the next delivery. Consider placing an emergency order.',
    potentialImpact: 'Could affect 3 popular menu items',
    aiConfidence: 93
  },
  {
    id: 3,
    type: 'Trend',
    title: 'Customer Preference Shift',
    description: 'Vegetarian options have seen a 23% increase in orders over the last month.',
    potentialImpact: 'Opportunity to expand vegetarian menu options',
    aiConfidence: 78
  },
  {
    id: 4,
    type: 'Efficiency',
    title: 'Staff Scheduling',
    description: 'Tuesday lunch shifts are overstaffed based on recent customer traffic patterns.',
    potentialImpact: 'Potential labor cost savings of $120/week',
    aiConfidence: 85
  }
];

export const inventoryAlertData = [
  { item: 'Fresh Tomatoes', currentStock: 2.3, unit: 'kg', status: 'critical' },
  { item: 'Chicken Breast', currentStock: 4.5, unit: 'kg', status: 'warning' },
  { item: 'Olive Oil', currentStock: 1.2, unit: 'L', status: 'warning' }
];
