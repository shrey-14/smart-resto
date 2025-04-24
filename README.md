# SmartResto

**SmartResto** is a smart kitchen management solution developed as part of a hackathon-winning project. It uses AI to tackle food inventory tracking, spoilage detection, waste management, and dynamic menu optimization. The goal: to reduce food waste and improve kitchen efficiency using cutting-edge machine learning and computer vision technologies.

---

## 💡 Key Features

### 🔍 Smart Inventory Management  
- Visual detection and tracking of kitchen ingredients using **YOLOv8**  
- Real-time stock level monitoring and spoilage detection via camera feeds  
- Integrated with an interactive web interface

### 📊 AI-Powered Demand & Waste Prediction  
- Sales forecasting using **XGBoost** and **Prophet**  
- Spoilage and overuse risk prediction using **LLaMA 3.1-8B Instruct**  
- Dynamic inventory replenishment suggestions

### 🍲 Intelligent Menu Optimization  
- Suggests daily specials based on near-expiry items  
- Recommends recipes and calculates cost-effective meal plans  
- Uses **LLaMA 3.1-8B Instruct** for generative dish creation

### 🗑️ Vision-Powered Waste Analysis  
- Classifies waste types via camera feed  
- Generates heatmaps of waste-prone areas  
- Converts waste to financial impact with loss-to-profit dashboard

---

## 🧠 Tech Stack

| Area                      | Tools / Models Used |
|--------------------------|---------------------|
| Visual Detection          | YOLOv8              |
| Forecasting               | XGBoost, Prophet    |
| Generative AI             | LLaMA 3.1 / 3.2 via Groq API |
| Waste Visualization       | OWL-ViT, Plotly     |
| Interface                 | Web UI (custom)     |

---

## 📁 Folder Structure

```
smart-resto/
├── backend/
│   ├── config/
│   ├── src/
│   ├── api.py
│   ├── main.py
│   ├── partial_waste_predictions.csv
│   ├── requirements.txt
│   ├── run_api.sh
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── components.json
│   ├── eslint.config.js
│   ├── index.html
│   ├── package-lock.json
│   ├── postcss.config.js
│   ├── tailwind.config.ts
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
```

---

## 🛠️ Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/shrey-14/smart-resto.git
cd SmartResto
```

### 2. Install requirements

```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the application

Start the frontend services:
```bash
cd frontend
npm run dev
```

Start the backend services:
```bash
cd backend
./run_api.sh
```

---

## 📈 Evaluation Metrics

| Task                          | Model                       | Metric             |
|-------------------------------|-----------------------------|--------------------|
| Inventory Tracking            | YOLOv8                      | mAP@50 = 0.526     |
| Food Spoilage Detection       | LLaMA 3.2 - 90b Vision       | Accuracy = 70%     |
| Sales Forecasting             | XGBoost, Prophet            | MAPE = 10–14%      |
| Waste Prediction              | LLaMA 3.1 - 8b Instruct      | ROUGE metrics      |
| Waste Heatmap & Dashboard     | OWL-ViT + Plotly             | Visual Accuracy    |

---


