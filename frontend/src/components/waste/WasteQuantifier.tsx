import React from 'react';

const WasteQuantifier = () => {
  return (
    <div className="waste-quantifier">
      <iframe
        srcDoc={`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Restaurant Waste Analysis Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .chart-container {
            height: 500px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
        .card {
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <h1 class="mb-4">Restaurant Waste Analysis Dashboard</h1>
        
        <!-- Summary Cards -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Total Waste Cost</h5>
                        <h2 class="card-text">₹100,000</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Waste Percentage</h5>
                        <h2 class="card-text">7%</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Profit Impact</h5>
                        <h2 class="card-text">₹20,000</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Potential Savings</h5>
                        <h2 class="card-text">₹120,000</h2>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Grid -->
        <div class="row">
            <!-- Profit vs Waste Analysis -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Profit vs Waste Analysis</h5>
                        <div class="chart-container" id="profitWasteChart"></div>
                    </div>
                </div>
            </div>

            <!-- Waste Distribution -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Waste Distribution</h5>
                        <div class="chart-container" id="wasteDistributionChart"></div>
                    </div>
                </div>
            </div>

            <!-- Over Portion Analysis -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Over Portion Analysis</h5>
                        <div class="chart-container" id="overPortionChart"></div>
                    </div>
                </div>
            </div>

            <!-- Spoilage Analysis -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Spoilage Analysis</h5>
                        <div class="chart-container" id="spoilageChart"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Profit vs Waste Analysis Chart
        const profitWasteTrace1 = {
            x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            y: [50000, 45000, 60000, 55000, 65000, 70000],
            type: 'scatter',
            name: 'Profit',
            line: {color: '#2ecc71', width: 3}
        };

        const profitWasteTrace2 = {
            x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            y: [10000, 12000, 15000, 13000, 14000, 16000],
            type: 'scatter',
            name: 'Waste',
            line: {color: '#e74c3c', width: 3}
        };

        const profitWasteLayout = {
            title: 'Monthly Profit vs Waste Analysis',
            xaxis: {title: 'Month'},
            yaxis: {title: 'Amount (₹)'},
            height: 450,
            showlegend: true,
            legend: {x: 0, y: 1},
            margin: {l: 50, r: 50, t: 50, b: 50}
        };

        Plotly.newPlot('profitWasteChart', [profitWasteTrace1, profitWasteTrace2], profitWasteLayout);

        // Waste Distribution Chart
        const wasteDistributionData = [{
            values: [45, 30, 25],
            labels: ['Over-portioned', 'Spoilage', 'Contamination'],
            type: 'pie',
            marker: {
                colors: ['#3498db', '#e74c3c', '#f1c40f']
            },
            textinfo: 'label+percent',
            hoverinfo: 'label+percent+value'
        }];

        const wasteDistributionLayout = {
            title: 'Waste Distribution by Type',
            height: 450,
            showlegend: true,
            legend: {x: 1, y: 0.5}
        };

        Plotly.newPlot('wasteDistributionChart', wasteDistributionData, wasteDistributionLayout);

        // Over Portion Analysis Chart
        const overPortionData = [{
            x: ['Chicken', 'Rice', 'Vegetables', 'Fish', 'Bread'],
            y: [120, 80, 60, 40, 30],
            type: 'bar',
            marker: {color: '#3498db'},
            text: [120, 80, 60, 40, 30],
            textposition: 'auto'
        }];

        const overPortionLayout = {
            title: 'Over Portion Analysis by Ingredient',
            xaxis: {title: 'Ingredient'},
            yaxis: {title: 'Waste Amount (kg)'},
            height: 450,
            margin: {l: 50, r: 50, t: 50, b: 50}
        };

        Plotly.newPlot('overPortionChart', overPortionData, overPortionLayout);

        // Spoilage Analysis Chart
        const spoilageData = [{
            x: ['Tomatoes', 'Lettuce', 'Milk', 'Cheese', 'Yogurt', 'Meat', 'Fish', 'Bread', 'Fruits', 'Vegetables'],
            y: [25, 20, 18, 15, 12, 10, 8, 6, 5, 4],
            type: 'bar',
            marker: {color: '#e74c3c'},
            text: [25, 20, 18, 15, 12, 10, 8, 6, 5, 4],
            textposition: 'auto'
        }];

        const spoilageLayout = {
            title: 'Top 10 Ingredients with Highest Spoilage Waste',
            xaxis: {
                title: 'Ingredient',
                tickangle: -45
            },
            yaxis: {title: 'Waste Amount (kg)'},
            height: 450,
            margin: {l: 50, r: 50, t: 50, b: 100}
        };

        Plotly.newPlot('spoilageChart', spoilageData, spoilageLayout);
    </script>
</body>
</html>`}
        style={{
          width: '100%',
          height: '2000px',  // Make it tall enough to fit all charts
          border: 'none',
          overflow: 'hidden'
        }}
        title="Waste Analysis Dashboard"
      />
    </div>
  );
};

export default WasteQuantifier; 