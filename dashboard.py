import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from datetime import datetime
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from PIL import Image

def dashboard(crop_choice, start_date, end_date):
    # your logic to forecast prices using crop_choice, start_date, end_date
    st.write(f"Forecasting for {crop_choice} from {start_date} to {end_date}")
    st.markdown("## 📊 Crop Price Forecasting Dashboard")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.write("""
    Select a crop from the dropdown below to view historical price trends and future forecasts powered by machine learning.
    """)

    crop_choice = st.selectbox("Select a Crop", [
        "Arecanut", "Black Pepper", "Copra", "Groundnut", "Maize", 
        "Mustard", "Ragi", "Rice", "Turmeric", "Wheat"
    ])

    crop_info = {
        "Arecanut": {"image": "arecanut.jpg", "desc": "### Arecanut\n- **Uses:** Cultural stimulant, traditional medicine\n- **Climate:** Warm, humid climate\n", "file": "Arecanut.xls"},
        "Black Pepper": {"image": "blackpepper.jpg", "desc": "### Black Pepper\n- **Uses:** Culinary spice, medicine, export crop\n- **Climate:** Hot, humid\n", "file": "BlackPepper.xls"},
        "Copra": {"image": "copra.png", "desc": "### Copra\n- **Uses:** Oil extraction\n- **Climate:** Coastal, tropical\n", "file": "Copra.xls"},
        "Groundnut": {"image": "groundnut.png", "desc": "### Groundnut\n- **Uses:** Edible oil, food\n- **Climate:** Warm, dry regions\n", "file": "Groundnut.xls"},
        "Maize": {"image": "maize.jpg", "desc": "### Maize\n- **Uses:** Cereal food, fodder\n- **Climate:** Warm, moderate rainfall\n", "file": "Maize.xls"},
        "Mustard": {"image": "mustard.jpg", "desc": "### Mustard\n- **Uses:** Oil, spice\n- **Climate:** Cool, dry\n", "file": "Mustard.xls"},
        "Ragi": {"image": "ragi.jpg", "desc": "### Ragi\n- **Uses:** Millet crop\n- **Climate:** Drought-resistant\n", "file": "Ragi.xls"},
        "Rice": {"image": "rice.jpg", "desc": "### Rice\n- **Uses:** Staple food\n- **Climate:** Wet, tropical\n", "file": "Rice.xls"},
        "Turmeric": {"image": "turmeric.jpg", "desc": "### Turmeric\n- **Uses:** Spice, medicinal\n- **Climate:** Hot, moist\n", "file": "Turmeric.xls"},
        "Wheat": {"image": "wheat.jpg", "desc": "### Wheat\n- **Uses:** Staple grain\n- **Climate:** Cool, dry\n", "file": "Wheat.xls"},
    }

    selected = crop_info[crop_choice]
    st.image(selected["image"], width=400, use_column_width=False)
    st.markdown(selected["desc"])
    file_path = selected["file"]

    @st.cache_data
    def load_data(file_path):
        df_list = pd.read_html(file_path)
        df = df_list[0]
        df = df[["Price Date", "Modal Price (Rs./Quintal)"]].copy()
        df.columns = ["Date", "Price"]
        df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
        return df.sort_values("Date")

    df = load_data(file_path)
    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

    if df.empty:
     st.warning("No data available for the selected date range.")
    return


    wpi_data = {
        "Date": pd.date_range(start="2020-01-01", end="2025-04-01", freq="MS"),
        "WPI": np.linspace(198.9, 259.1, 64)
    }
    wpi_df = pd.DataFrame(wpi_data)

    def extend_wpi(wpi_df, months=12):
        wpi_ts = wpi_df.set_index('Date')['WPI']
        model = ExponentialSmoothing(wpi_ts, trend='add', seasonal=None)
        model_fit = model.fit()
        forecast = model_fit.forecast(months)
        future_dates = pd.date_range(start=wpi_df['Date'].max() + pd.DateOffset(months=1), periods=months, freq='MS')
        future_wpi_df = pd.DataFrame({'Date': future_dates, 'WPI': forecast.values})
        return pd.concat([wpi_df, future_wpi_df], ignore_index=True)

    wpi_df = extend_wpi(wpi_df)

    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    merged_df = pd.merge(df, wpi_df, left_on="Month", right_on="Date", how="left")
    merged_df.drop(columns=["Month", "Date_y"], inplace=True)
    merged_df.rename(columns={"Date_x": "Date"}, inplace=True)
    merged_df.dropna(inplace=True)

    scaler = MinMaxScaler()
    merged_df[['Price', 'WPI']] = scaler.fit_transform(merged_df[['Price', 'WPI']])

    def create_features(data, n_lags=3):
        df_feat = data.copy()
        for lag in range(1, n_lags + 1):
            df_feat[f'Price_lag_{lag}'] = df_feat['Price'].shift(lag)
            df_feat[f'WPI_lag_{lag}'] = df_feat['WPI'].shift(lag)
        df_feat = df_feat.dropna().reset_index(drop=True)
        return df_feat

    df_features = create_features(merged_df, n_lags=3)

    train_size = int(len(df_features) * 0.85)
    train, test = df_features[:train_size], df_features[train_size:]
    feature_cols = [f'Price_lag_{i}' for i in range(1, 4)] + [f'WPI_lag_{i}' for i in range(1, 4)]
    X_train, y_train = train[feature_cols], train['Price']
    X_test, y_test = test[feature_cols], test['Price']

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    st.markdown("### 🔍 Model Performance")
    st.write(f"**R² Score:** {r2:.3f}")
    st.write(f"**RMSE:** {rmse:.3f} (scaled units)")

    months_to_forecast = st.selectbox("Select number of months to forecast", [1, 2, 3, 4, 5, 6], index=2)
    forecast_start = pd.Timestamp(datetime.today().strftime('%Y-%m-01')) + pd.DateOffset(months=1)
    last_known = df_features.iloc[-1:].copy()
    forecasted_prices_scaled, forecast_dates = [], []
    current_row = last_known.copy()

    for month in range(months_to_forecast):
        features, next_date = [], forecast_start + pd.DateOffset(months=month)
        for lag in range(1, 4):
            features.append(current_row['Price'].values[0] if lag == 1 else forecasted_prices_scaled[-lag + 1] if len(forecasted_prices_scaled) >= lag - 1 else current_row[f'Price_lag_{lag}'].values[0])
            wpi_val = wpi_df.loc[wpi_df['Date'] == next_date, 'WPI']
            features.append(wpi_val.values[0] if not wpi_val.empty else current_row[f'WPI_lag_{lag}'].values[0])
        pred_features = pd.DataFrame([features], columns=feature_cols)
        pred_price_scaled = model.predict(pred_features)[0]
        forecasted_prices_scaled.append(pred_price_scaled)
        forecast_dates.append(next_date)
        new_row = {
            'Date': next_date,
            'Price': pred_price_scaled,
            'WPI': pred_features['WPI_lag_1'].values[0],
            'Price_lag_1': pred_price_scaled,
            'Price_lag_2': current_row['Price_lag_1'].values[0],
            'Price_lag_3': current_row['Price_lag_2'].values[0],
            'WPI_lag_1': pred_features['WPI_lag_1'].values[0],
            'WPI_lag_2': current_row['WPI_lag_1'].values[0],
            'WPI_lag_3': current_row['WPI_lag_2'].values[0],
        }
        current_row = pd.DataFrame([new_row])

    forecasted_prices_scaled = np.array(forecasted_prices_scaled).reshape(-1, 1)
    dummy_wpi = np.zeros_like(forecasted_prices_scaled)
    forecasted_prices = scaler.inverse_transform(np.hstack([forecasted_prices_scaled, dummy_wpi]))[:, 0]

    historical_dates = merged_df['Date']
    historical_prices = scaler.inverse_transform(merged_df[['Price', 'WPI']])[:, 0]
    trace1 = go.Scatter(x=historical_dates, y=historical_prices, mode='lines', name='Historical Price')
    trace2 = go.Scatter(x=forecast_dates, y=forecasted_prices, mode='lines+markers', name='Forecasted Price')
    layout = go.Layout(title=f'{crop_choice} Price Forecast', xaxis=dict(title='Date'), yaxis=dict(title='Price (Rs./Quintal)'), hovermode='x')
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    df_forecast = pd.DataFrame({
    "Month": forecast_dates,
    "Forecasted Price (Rs./Quintal)": np.round(forecasted_prices, 2)
})

# Sort by actual datetime
    df_forecast = df_forecast.sort_values("Month")
    df_forecast["Month"] = df_forecast["Month"].dt.strftime("%Y-%m")
    st.subheader(f"🗓 Forecasted Prices for Next {months_to_forecast} Month(s)")
    st.dataframe(df_forecast)
 