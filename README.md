# 🌾 Crop Price Prediction App

A Streamlit-based web application that predicts future crop prices using Machine Learning and Time Series Forecasting. This tool helps farmers, traders, and policymakers make informed decisions based on projected trends.

## 🚀 Features

- 📈 Predict crop prices using **XGBoost Regressor** and **Holt-Winters Exponential Smoothing**
- 📊 Interactive **Plotly** charts for real-time trend visualization
- 🧪 Clean preprocessing using **MinMaxScaler**
- ✅ Simple UI built with **Streamlit**
- 🔁 Option to handle multiple crops with time-based forecasts

## 📂 Tech Stack

- Python 🐍
- Streamlit 🌐
- XGBoost 📦
- Statsmodels (Holt-Winters) 📉
- Scikit-learn 🔍
- Plotly 📊
- Pandas & NumPy 📋

## 🧠 How it Works

1. Upload or load your crop price dataset (with date and price columns).
2. The app preprocesses the data and scales it for model input.
3. Two models are trained:
   - A **Machine Learning model** using `XGBRegressor`
   - A **Time Series model** using `ExponentialSmoothing`
4. Future price predictions are generated and plotted for visual analysis.

## 🖥️ Running the App Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/crop-price-prediction.git
   cd crop-price-prediction
