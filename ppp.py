import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objs as go
import base64
st.set_page_config(layout="wide")

st.markdown("""
<style>
/* Make main content full width */
.reportview-container .main .block-container,
.appview-container .main .block-container,
.st-emotion-cache-1jicfl2 {
    max-width: 100vw !important;
    width: 100vw !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}
</style>
""", unsafe_allow_html=True)



# --- KisanSathi Logo in Header ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    return b64

logo_b64 = get_base64_image("img.png")
st.markdown(f"""
<style>
.kisansathi-header {{
    width: 100vw;
    position: relative;
    left: 50%;
    right: 50%;
    margin-left: -50vw;
    margin-right: -50vw;
    box-sizing: border-box;
    background-color: #f1faee;
    padding: 12px 0;
    border-radius: 0 0 12px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.kisansathi-header .left, .kisansathi-header .right {{
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}}
.kisansathi-header .center {{
    flex: 2;
    text-align: center;
}}
</style>
<div class="kisansathi-header">
    <div class="left">
        <img src="data:image/png;base64,{logo_b64}" alt="KisanSathi Logo" style="height: 180px;">
    </div>
    <div class="center">
        <h1 style="color:#457b9d; margin-bottom:0;">KisanSathi</h1>
        <h3 style="color:#1d3557; margin-top:0;"></h3>
    </div>
    <div class="right"></div>
</div>
""", unsafe_allow_html=True)



# --- Real crop info and WPI index dictionaries ---
crop_info = {
    "Arecanut": {
        "file": "Arecanut.xls",
        "image": "Arecanut.jpg",
        "markdown": """**Arecanut (Areca catechu)**

**Scientific Name:** Areca catechu  
**Common Names:** Arecanut, Betel nut  

**Places Grown:**  
Primarily grown in India (Kerala, Karnataka, Assam, Tamil Nadu), Bangladesh, Indonesia, Philippines, and other tropical regions.

**Description:**  
Arecanut is a tropical crop cultivated for its nut which is commonly chewed wrapped in betel leaves. It is an important commercial crop in many countries, used for both traditional and economic purposes.

**Uses:**  
- Widely used as a stimulant in cultural chewing practices.  
- Ingredient in traditional medicine.  
- Economic crop for many farmers.

**Climatic Needs:**  
- Requires warm, humid climate.  
- Grows best in well-drained soils with adequate rainfall.
"""
    },
    "Black Pepper": {
        "file": "pepper.xls",
        "image": "pepper.png",
        "markdown": """**Black Pepper (Piper nigrum)**

**Scientific Name:** Piper nigrum  
**Common Names:** Black Pepper, Kali Mirch  

**Places Grown:**  
India (Kerala, Karnataka, Tamil Nadu), Vietnam, Indonesia, Brazil, Sri Lanka.

**Description:**  
Black pepper is a flowering vine cultivated for its fruit, known as a peppercorn, which is dried and used as a spice and seasoning.

**Uses:**  
- Culinary spice and flavoring  
- Used in traditional medicine  
- Export commodity

**Climatic Needs:**  
- Hot, humid tropical climate  
- Well-drained soils
"""
    },
    "Copra": {
        "file": "copra.xls",
        "image": "copra.png",
        "markdown": """**Copra (Dried Coconut Kernel)**

**Scientific Name:** Cocos nucifera  
**Common Names:** Copra, Dried Coconut  

**Places Grown:**  
India (Kerala, Tamil Nadu, Karnataka, Andhra Pradesh), Philippines, Indonesia, Sri Lanka.

**Description:**  
Copra is the dried kernel of coconut, used to extract coconut oil and as animal feed.

**Uses:**  
- Coconut oil extraction  
- Animal feed  
- Industrial uses

**Climatic Needs:**  
- Tropical, coastal regions  
- High rainfall, sandy soils
"""
    },
    "Groundnut": {
        "file": "Groundnut.xls",
        "image": "grountnut.png",
        "markdown": """**Groundnut (Arachis hypogaea)**

**Scientific Name:** Arachis hypogaea  
**Common Names:** Groundnut, Peanut  

**Places Grown:**  
India (Gujarat, Andhra Pradesh, Tamil Nadu, Karnataka), China, Nigeria, USA.

**Description:**  
Groundnut is a legume crop grown mainly for its edible seeds, which are a rich source of oil and protein.

**Uses:**  
- Edible oil  
- Direct consumption (roasted, boiled)  
- Animal feed

**Climatic Needs:**  
- Warm, dry climate  
- Well-drained sandy loam soils
"""
    },
    "Maize": {
        "file": "maize.xls",
        "image": "maize.jpg",
        "markdown": """**Maize (Zea mays)**

**Scientific Name:** Zea mays  
**Common Names:** Maize, Corn, Makka  

**Places Grown:**  
India (Karnataka, Andhra Pradesh, Maharashtra, Bihar), USA, China, Brazil.

**Description:**  
Maize is a cereal grain used as food, fodder, and in various industrial products.

**Uses:**  
- Human consumption  
- Animal feed  
- Industrial uses (starch, ethanol)

**Climatic Needs:**  
- Warm climate  
- Moderate rainfall
"""
    },
    "Mustard": {
        "file": "Mustard.xls",
        "image": "mustard.jpg",
        "markdown": """**Mustard (Brassica spp.)**

**Scientific Name:** Brassica juncea, Brassica nigra  
**Common Names:** Mustard, Sarson  

**Places Grown:**  
India (Rajasthan, Uttar Pradesh, Haryana, Madhya Pradesh), Canada, China.

**Description:**  
Mustard is grown for its seeds, which are used to produce oil and as a spice.

**Uses:**  
- Edible oil  
- Spice  
- Green manure

**Climatic Needs:**  
- Cool, dry climate  
- Well-drained soils
"""
    },
    "Ragi": {
        "file": "Ragi.xls",
        "image": "ragi.jpg",
        "markdown": """**Ragi (Eleusine coracana)**

**Scientific Name:** Eleusine coracana  
**Common Names:** Ragi, Finger Millet  

**Places Grown:**  
India (Karnataka, Tamil Nadu, Andhra Pradesh, Maharashtra), Africa.

**Description:**  
Ragi is a cereal crop known for its high nutritional value and drought resistance.

**Uses:**  
- Staple food  
- Malted products  
- Baby food

**Climatic Needs:**  
- Drought-resistant  
- Grows in poor soils
"""
    },
    "Rice": {
        "file": "Rice.xls",
        "image": "rice.jpg",
        "markdown": """**Rice (Oryza sativa)**

**Scientific Name:** Oryza sativa  
**Common Names:** Rice, Chawal  

**Places Grown:**  
India (West Bengal, Uttar Pradesh, Punjab, Andhra Pradesh, Tamil Nadu), China, Indonesia.

**Description:**  
Rice is the most widely consumed staple food in the world.

**Uses:**  
- Human consumption  
- Rice bran oil  
- Brewing

**Climatic Needs:**  
- Wet, tropical climate  
- Requires standing water
"""
    },
    "Turmeric": {
        "file": "Turmeric.xls",
        "image": "turmeric.jpg",
        "markdown": """**Turmeric (Curcuma longa)**

**Scientific Name:** Curcuma longa  
**Common Names:** Turmeric, Haldi  

**Places Grown:**  
India (Andhra Pradesh, Tamil Nadu, Karnataka, Odisha, West Bengal, Maharashtra), also grown in Bangladesh, China, Indonesia.

**Description:**  
Turmeric is a perennial herbaceous plant of the ginger family, widely used as a spice, coloring agent, and in traditional medicine.

**Uses:**  
- Culinary spice and coloring agent.  
- Used in Ayurvedic and traditional medicine.  
- Cosmetic and dye industries.

**Climatic Needs:**  
- Prefers hot, moist climate.  
- Thrives in well-drained, fertile soils.
"""
    },
    "Wheat": {
        "file": "wheat.xls",
        "image": "wheat.jpg",
        "markdown": """**Wheat (Triticum aestivum)**

**Scientific Name:** Triticum aestivum  
**Common Names:** Wheat, Gehun  

**Places Grown:**  
India (Uttar Pradesh, Punjab, Haryana, Madhya Pradesh), China, Russia, USA.

**Description:**  
Wheat is a staple cereal grain used for bread, pasta, and many food products.

**Uses:**  
- Bread, pasta, bakery products  
- Animal feed  
- Brewing

**Climatic Needs:**  
- Cool, dry climate  
- Fertile, well-drained soils
"""
    }
}

# --- WPI data for each crop (FULL, as provided) ---
turmeric_wpi = {
    2020: [109.7, 109.3, 117, 116.7, 116.5, 114.9, 113.2, 115.4, 113.5, 111.5, 112.2, 113],
    2021: [113.7, 118.6, 126.3, 132.6, 128.5, 127.4, 121.4, 120.8, 118.5, 117.2, 117.3, 123.1],
    2022: [127.1, 127.2, 126.6, 120.7, 119.7, 118.3, 117.9, 115.2, 114.9, 110.4, 114, 115.7],
    2023: [116.6, 114.2, 115, 112.5, 110.1, 111, 127.1, 156.6, 163.1, 163.8, 171.7, 172.7],
    2024: [172, 176, 203.4, 216, 221.1, 218.9, 209.7, 207.7, 198.5, 186.4, 191.8, 183]
}
arecanut_wpi = {
    2020: [198.9, 195.2, 187.8, 185.4, 189.6, 198.5, 207.7, 217.7, 220.6, 229, 241.3, 230.3],
    2021: [226, 239.4, 235.2, 235, 234.6, 239.1, 242.4, 253.2, 269.2, 279.9, 280.5, 285.6],
    2022: [279.1, 266.5, 260.8, 266, 265.1, 264.5, 265.1, 271.1, 281.7, 287.4, 272.3, 261.5],
    2023: [268.1, 265.6, 261.5, 261.5, 267.7, 272.3, 277.4, 284.4, 278.6, 277.6, 267.2, 263.6],
    2024: [267.1, 258.3, 255.9, 261.2, 266.8, 265.2, 255.9, 255.1, 242.6, 240.6, 235, 231]
}
blackpepper_wpi = {
    2020: [124.8, 123.8, 121.9, 122.3, 124, 124.8, 123.4, 123.7, 123.5, 123.2, 124.3, 125.2],
    2021: [122.2, 121.5, 126.9, 130.3, 130.6, 136.5, 136, 135.2, 137.9, 145, 159.7, 163.8],
    2022: [165, 166.1, 167.9, 168.7, 166.6, 165.4, 164, 165.6, 165.7, 164.7, 164.1, 167],
    2023: [168.5, 167.4, 165.7, 161.2, 162.7, 162.9, 168.7, 194.3, 199.4, 201.4, 200.6, 201.1],
    2024: [200.7, 190, 181.9, 189.1, 197.1, 211.3, 213.9, 212.8, 214.1, 210.8, 210.4, 213.1]
}
copra_wpi = {
    2020: [186, 186.3, 184.1, 184.2, 181.4, 179.1, 180.7, 184.7, 187.8, 192.4, 195.9, 205.9],
    2021: [206.4, 210.2, 219.6, 218.5, 215.2, 217.2, 211.5, 209, 208.4, 207.3, 206.5, 206.9],
    2022: [205.7, 204.9, 202.6, 200, 195.8, 185.8, 184.1, 183.5, 185.2, 183.2, 182.2, 180.6],
    2023: [173.5, 167.9, 160.8, 159.8, 159.3, 148.2, 149.2, 154.3, 147.9, 150.1, 148.6, 148.5],
    2024: [153.8, 158.4, 156.1, 155.3, 154.1, 156.1, 159.6, 170.9, 194.9, 209.3, 213.1, 221.5]
}
groundnut_wpi = {
    2020: [139, 142.7, 149.8, 155.8, 157.7, 158.8, 156.5, 154.2, 145.5, 145.1, 146.3, 149.1],
    2021: [156.4, 162.1, 170.4, 167.7, 163.4, 163.3, 160, 161.9, 162.9, 160.4, 163.6, 165.3],
    2022: [167.3, 165.7, 169.1, 171.2, 174.3, 171, 172, 175.3, 175.2, 175.2, 176.6, 181.6],
    2023: [189.7, 198.3, 197.4, 197.8, 196.8, 200.1, 203.9, 203, 201, 201.1, 197.7, 200.7],
    2024: [195.9, 190.4, 189.5, 187.7, 189.9, 185.5, 182.2, 180, 176.7, 175.8, 177.5, 175.7]
}
maize_wpi = {
    2020: [178.8, 167, 158.5, 155.4, 152, 148.4, 147.6, 146.9, 142.5, 137.9, 139.8, 140.5],
    2021: [137.4, 135.4, 135.7, 139.9, 141.5, 142.6, 147.6, 153.6, 154.8, 151.6, 150.4, 154.2],
    2022: [157.4, 162.4, 169.5, 181.1, 182.3, 178.3, 186.1, 192.4, 193.4, 186.1, 182.4, 186.6],
    2023: [190.9, 193.1, 193.7, 189.8, 184.5, 181.9, 186, 186.9, 183.9, 183.5, 185.7, 188.3],
    2024: [195.6, 198.4, 196.4, 198.7, 200.9, 202.7, 207.2, 213.4, 216.5, 215.6, 213.3, 212.9]
}
mustard_wpi = {
    2020: [152.9, 150.5, 148.2, 147.3, 148.6, 151.9, 154.6, 159.3, 164, 166.4, 171.9, 171.3],
    2021: [177.5, 178.9, 172.9, 181.1, 196, 200.2, 202.3, 210.9, 219.1, 224, 225.1, 222.8],
    2022: [216.8, 216.6, 210.9, 208.3, 210.8, 206, 204, 201.6, 198.2, 197.4, 201.9, 199.3],
    2023: [196.2, 185.9, 176.2, 171.7, 167.5, 162, 163.8, 166.2, 165.5, 166, 167.1, 167.4],
    2024: [165.7, 162.8, 162.2, 163, 164.5, 172, 174.9, 175.7, 182.5, 191.9, 191.9, 190.5]
}
ragi_wpi = {
    2020: [213.8, 211.9, 215, 224.5, 229.6, 228.2, 232.7, 231.9, 225.1, 223.9, 228.4, 230.1],
    2021: [226.2, 226.8, 232, 231.8, 233.7, 235.2, 233.1, 230.1, 227.8, 227.1, 229.3, 229.6],
    2022: [232.2, 235.4, 234.9, 233.4, 231.1, 225.9, 219.4, 209.7, 203.9, 207, 208.6, 213.3],
    2023: [221.6, 222, 224.3, 233.8, 237, 237.1, 245.4, 256.6, 259, 263.1, 269.5, 279.2],
    2024: [278.9, 285.6, 291.9, 294.5, 300.4, 303.3, 309.2, 321.5, 321.2, 316.1, 312.8, 316.9]
}
rice_wpi = {
    2020: [162.3, 162.3, 159.3, 162.7, 163.1, 165.6, 165.8, 165.4, 165, 163.7, 163.1, 162.2],
    2021: [162, 161.7, 161.5, 162, 162.2, 161.7, 161.2, 161.8, 162.3, 162.8, 162.8, 162.6],
    2022: [162.9, 161.7, 163.1, 164.4, 165.1, 165.5, 166.2, 168.8, 171.7, 173.6, 173.3, 173.7],
    2023: [174.6, 175.6, 175.4, 176.2, 177.2, 178.2, 181.2, 184.3, 187, 190.2, 191.4, 192],
    2024: [191.2, 193.6, 196, 197.4, 197.7, 199.4, 201.1, 202, 203.4, 204.4, 205.9, 205.3]
}
wheat_wpi = {
    2020: [168.7, 167.3, 162.9, 161.2, 161.5, 158.5, 157.3, 154.2, 150, 147.4, 147.9, 147.3],
    2021: [149.1, 149.6, 150.3, 156.1, 157.4, 155.7, 152.8, 153.9, 156.6, 159.4, 162.9, 164.1],
    2022: [164.6, 166.1, 171.4, 173.3, 174.1, 171.8, 173.6, 180.6, 181.8, 185.3, 192.4, 198.2],
    2023: [203.9, 196.8, 187.1, 186.4, 184.9, 187.3, 187.1, 188.3, 190.7, 194, 197.6, 200.5],
    2024: [200.1, 201.5, 201.1, 197, 196.5, 199, 200.2, 202.2, 205.4, 209.6, 213.8, 215.5]
}

wpi_dicts = {
    "Arecanut": arecanut_wpi,
    "Black Pepper": blackpepper_wpi,
    "Copra": copra_wpi,
    "Groundnut": groundnut_wpi,
    "Maize": maize_wpi,
    "Mustard": mustard_wpi,
    "Ragi": ragi_wpi,
    "Rice": rice_wpi,
    "Turmeric": turmeric_wpi,
    "Wheat": wheat_wpi,
}

def wpi_table_to_df(wpi_dict):
    records = []
    for year, values in wpi_dict.items():
        for month, wpi in enumerate(values):
            records.append({"Date": pd.Timestamp(year=year, month=month+1, day=1), "WPI": wpi})
    return pd.DataFrame(records)

@st.cache_data
def load_crop_data(file_path):
    try:
        df_list = pd.read_html(file_path)
        df = df_list[0]
        df = df[["Price Date", "Modal Price (Rs./Quintal)"]].copy()
        df.columns = ["Date", "Price"]
        df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
        return df.sort_values("Date")
    except Exception as e:
        st.error(f"Could not load data from {file_path}: {e}")
        return pd.DataFrame()

# --- Page Navigation ---
# Center the radio and increase font size
# Increase font size for the radio label and options
st.markdown("""
    <style>
    div[role="radiogroup"] > label, div[role="radiogroup"] > div {
        font-size: 1.3rem !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# Center the radio using columns
col1, col2, col3 = st.columns([1,1,1])
with col2:
    page = st.radio(
        "",
        ["Top Gainers & Losers", "Price Prediction"],
        horizontal=True,
        index=0,
        key="main_nav"
    )



crops = list(crop_info.keys())
st.markdown("""
    <style>
    /* Reduce sidebar width */
    section[data-testid="stSidebar"] {
        min-width: 80px !important;
        max-width: 80px !important;
        width: 80px !important;
    }
    /* Optional: Reduce sidebar content padding */
    .st-emotion-cache-1v0mbdj {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- Let user select number of forecast months (1-6), always from July 2025 ---
months_to_forecast = st.slider("Select number of forecast months ", 1, 6, 6)
forecast_start = pd.Timestamp('2025-07-01')
forecast_dates = [forecast_start + pd.DateOffset(months=i) for i in range(months_to_forecast)]
forecast_months = [d.strftime('%Y-%m') for d in forecast_dates]

# --- Hybrid Forecast for All Crops (for Top Gainers & Losers) ---
@st.cache_data
def get_forecast_for_all_crops(crops, months_to_forecast, forecast_months):
    forecasted_prices_dict = {}
    for crop in crops:
        price_df = load_crop_data(crop_info[crop]["file"])
        wpi_df = wpi_table_to_df(wpi_dicts[crop])
        price_df["Month"] = price_df["Date"].dt.to_period("M").dt.to_timestamp()
        merged_df = pd.merge(price_df, wpi_df, left_on="Month", right_on="Date", how="left")
        merged_df.drop(columns=["Month", "Date_y"], inplace=True)
        merged_df.rename(columns={"Date_x": "Date"}, inplace=True)
        merged_df.dropna(inplace=True)
        scaler = MinMaxScaler()
        merged_df[["Price", "WPI"]] = scaler.fit_transform(merged_df[["Price", "WPI"]])
        def create_features(data, n_lags=3):
            df_feat = data.copy()
            for lag in range(1, n_lags + 1):
                df_feat[f'Price_lag_{lag}'] = df_feat['Price'].shift(lag)
                df_feat[f'WPI_lag_{lag}'] = df_feat['WPI'].shift(lag)
            df_feat = df_feat.dropna().reset_index(drop=True)
            return df_feat
        df_features = create_features(merged_df, n_lags=3)
        if df_features.empty or len(df_features) < 10:
            forecasted_prices_dict[crop] = [np.nan]*months_to_forecast
            continue
        train_size = int(len(df_features) * 0.85)
        train = df_features[:train_size]
        feature_cols = [f'Price_lag_{i}' for i in range(1, 4)] + [f'WPI_lag_{i}' for i in range(1, 4)]
        X_train, y_train = train[feature_cols], train['Price']
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        hw_series = merged_df.copy()
        hw_series['Price_unscaled'] = scaler.inverse_transform(merged_df[["Price", "WPI"]])[:,0]
        hw_model = ExponentialSmoothing(
            hw_series['Price_unscaled'],
            trend='add',
            seasonal='add',
            seasonal_periods=12
        ).fit()
        hw_forecast = hw_model.forecast(months_to_forecast)
        last_known = df_features.iloc[-1:].copy()
        forecasted_prices_scaled = []
        current_row = last_known.copy()
        for month in range(months_to_forecast):
            features = []
            for lag in range(1, 4):
                if lag == 1:
                    features.append(current_row['Price'].values[0])
                else:
                    idx = -lag + 1
                    if len(forecasted_prices_scaled) >= abs(idx):
                        features.append(forecasted_prices_scaled[idx])
                    else:
                        features.append(current_row[f'Price_lag_{lag}'].values[0])
                wpi_val = wpi_df['WPI'].values[-1] if len(wpi_df) == 0 else wpi_df['WPI'].values[-1]
                features.append(wpi_val)
            pred_features = pd.DataFrame([features], columns=feature_cols)
            pred_price_scaled = model.predict(pred_features)[0]
            forecasted_prices_scaled.append(pred_price_scaled)
            new_row = {
                'Date': None,
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
        xgb_forecasted_prices = scaler.inverse_transform(np.hstack([forecasted_prices_scaled, dummy_wpi]))[:, 0]
        hybrid_forecast = (xgb_forecasted_prices + hw_forecast.values) / 2
        forecasted_prices_dict[crop] = list(np.round(hybrid_forecast, 2))
    return forecasted_prices_dict

# --- Top Gainers & Losers Page ---
if page == "Top Gainers & Losers":
    st.markdown("## 🌾 Top Gainers & Losers (Forecasted Crops per Month)")
    with st.spinner("Generating forecasts for all crops..."):
        forecasted_prices_dict = get_forecast_for_all_crops(crops, months_to_forecast, forecast_months)
    forecast_df = pd.DataFrame([forecasted_prices_dict[crop] for crop in crops], index=crops, columns=forecast_months)
    price_changes = forecast_df.diff(axis=1).fillna(0)
    summary = {}
    for i, month in enumerate(forecast_months):
        if i == 0:
            top_gainers = []
            top_losers = []
        else:
            prev_month = forecast_months[i - 1]
            changes = forecast_df[month] - forecast_df[prev_month]
            top_gainers = changes.sort_values(ascending=False).head(3).index.tolist()
            top_losers = changes.sort_values(ascending=True).head(3).index.tolist()
        summary[month] = {
            "Top Gainers": top_gainers,
            "Top Losers": top_losers
        }
    for month in forecast_months[1:]:
        st.markdown(f"### {month}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Gainers:**")
            for crop in summary[month]["Top Gainers"]:
                st.success(f"🟢 {crop} (+₹{int(forecast_df.loc[crop, month] - forecast_df.loc[crop, forecast_months[forecast_months.index(month)-1]])})")
        with col2:
            st.markdown("**Top Losers:**")
            for crop in summary[month]["Top Losers"]:
                st.error(f"🔴 {crop} ({int(forecast_df.loc[crop, month] - forecast_df.loc[crop, forecast_months[forecast_months.index(month)-1]])})")
        # --- Chart for this month ---
        gainers_crops = summary[month]["Top Gainers"]
        gainers_changes = [forecast_df.loc[c, month] - forecast_df.loc[c, forecast_months[forecast_months.index(month)-1]] for c in gainers_crops]
        losers_crops = summary[month]["Top Losers"]
        losers_changes = [forecast_df.loc[c, month] - forecast_df.loc[c, forecast_months[forecast_months.index(month)-1]] for c in losers_crops]
        chart_df = pd.DataFrame({
            "Crop": gainers_crops + losers_crops,
            "Change": gainers_changes + losers_changes,
            "Type": ["Gainer"]*len(gainers_crops) + ["Loser"]*len(losers_crops)
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_df[chart_df["Type"] == "Gainer"]["Crop"],
            y=chart_df[chart_df["Type"] == "Gainer"]["Change"],
            name="Gainers",
            marker_color="green"
        ))
        fig.add_trace(go.Bar(
            x=chart_df[chart_df["Type"] == "Loser"]["Crop"],
            y=chart_df[chart_df["Type"] == "Loser"]["Change"],
            name="Losers",
            marker_color="red"
        ))
        fig.update_layout(
            title=f"Top Gainers and Losers for {month}",
            xaxis_title="Crop",
            yaxis_title="Price Change (Rs./Quintal)",
            barmode="group",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Price Prediction Page (Full hybrid dashboard for selected crop) ---
elif page == "Price Prediction":
    crop = st.selectbox("Select Crop", crops)
    col1, col2 = st.columns([1,2])
    with col1:
        try:
            st.image(crop_info[crop]["image"], width=300)
        except Exception:
            st.image("https://via.placeholder.com/300x200.png?text=No+Image", width=300)
    with col2:
        st.markdown(crop_info[crop]["markdown"])

    # --- Load data and WPI for this crop
    price_df = load_crop_data(crop_info[crop]["file"])
    wpi_df = wpi_table_to_df(wpi_dicts[crop])
    price_df["Month"] = price_df["Date"].dt.to_period("M").dt.to_timestamp()
    merged_df = pd.merge(price_df, wpi_df, left_on="Month", right_on="Date", how="left")
    merged_df.drop(columns=["Month", "Date_y"], inplace=True)
    merged_df.rename(columns={"Date_x": "Date"}, inplace=True)
    merged_df.dropna(inplace=True)
    scaler = MinMaxScaler()
    merged_df[["Price", "WPI"]] = scaler.fit_transform(merged_df[["Price", "WPI"]])
    def create_features(data, n_lags=3):
        df_feat = data.copy()
        for lag in range(1, n_lags + 1):
            df_feat[f'Price_lag_{lag}'] = df_feat['Price'].shift(lag)
            df_feat[f'WPI_lag_{lag}'] = df_feat['WPI'].shift(lag)
        df_feat = df_feat.dropna().reset_index(drop=True)
        return df_feat
    df_features = create_features(merged_df, n_lags=3)
    if df_features.empty or len(df_features) < 10:
        st.warning("Not enough data for modeling after feature engineering.")
        st.stop()
    train_size = int(len(df_features) * 0.85)
    train, test = df_features[:train_size], df_features[train_size:]
    feature_cols = [f'Price_lag_{i}' for i in range(1, 4)] + [f'WPI_lag_{i}' for i in range(1, 4)]
    X_train, y_train = train[feature_cols], train['Price']
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    hw_series = merged_df.copy()
    hw_series['Price_unscaled'] = scaler.inverse_transform(merged_df[["Price", "WPI"]])[:,0]
    hw_model = ExponentialSmoothing(
        hw_series['Price_unscaled'],
        trend='add',
        seasonal='add',
        seasonal_periods=12
    ).fit()
    hw_forecast = hw_model.forecast(months_to_forecast)
    last_known = df_features.iloc[-1:].copy()
    forecasted_prices_scaled = []
    current_row = last_known.copy()
    for month in range(months_to_forecast):
        features = []
        for lag in range(1, 4):
            if lag == 1:
                features.append(current_row['Price'].values[0])
            else:
                idx = -lag + 1
                if len(forecasted_prices_scaled) >= abs(idx):
                    features.append(forecasted_prices_scaled[idx])
                else:
                    features.append(current_row[f'Price_lag_{lag}'].values[0])
            wpi_val = wpi_df['WPI'].values[-1] if len(wpi_df) == 0 else wpi_df['WPI'].values[-1]
            features.append(wpi_val)
        pred_features = pd.DataFrame([features], columns=feature_cols)
        pred_price_scaled = model.predict(pred_features)[0]
        forecasted_prices_scaled.append(pred_price_scaled)
        new_row = {
            'Date': None,
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
    xgb_forecasted_prices = scaler.inverse_transform(np.hstack([forecasted_prices_scaled, dummy_wpi]))[:, 0]
    hybrid_forecast = (xgb_forecasted_prices + hw_forecast.values) / 2

    # Always forecast for July–December 2025 (or up to N months)
    forecast_start = pd.Timestamp('2025-07-01')
    forecast_dates = [forecast_start + pd.DateOffset(months=i) for i in range(months_to_forecast)]
    forecast_months = [d.strftime('%Y-%m') for d in forecast_dates]

    # --- Show price trend for this crop (last 12 months + forecast)
    last_12 = price_df.sort_values("Date").tail(12)
    hist_months = last_12["Date"].dt.strftime("%Y-%m").tolist()
    hist_prices = last_12["Price"].tolist()
    all_months = hist_months + forecast_months
    all_prices = hist_prices + list(hybrid_forecast)
    all_colors = ["gray"]*len(hist_months) + ["blue"]*len(forecast_months)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=all_months, y=all_prices, mode="lines+markers",
        marker=dict(color=all_colors),
        line=dict(color="gray"),
        name="Price Trend"
    ))
    fig.update_layout(title=f"{crop} Price Trend (Historical + Forecast)",
                      xaxis_title="Month",
                      yaxis_title="Price (Rs./Quintal)",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # --- Recommendation Bar Chart ---
    recommendations = []
    rec_color_map = {"Hold": "green", "Sell": "red", "Stable": "orange", "Start": "blue"}
    bar_colors = []
    for i in range(len(hybrid_forecast)):
        if i == 0:
            recommendations.append("Start")
            bar_colors.append("blue")
        else:
            change = hybrid_forecast[i] - hybrid_forecast[i-1]
            if change > 300:
                recommendations.append("Hold")
                bar_colors.append("green")
            elif change < -300:
                recommendations.append("Sell")
                bar_colors.append("red")
            else:
                recommendations.append("Stable")
                bar_colors.append("orange")

    bar_fig = go.Figure(go.Bar(
        x=forecast_months,
        y=hybrid_forecast,
        marker_color=bar_colors,
        text=recommendations,
        textposition='outside',
    ))
    bar_fig.update_layout(
        title="Monthly Recommendation (Color-coded)",
        xaxis_title="Month",
        yaxis_title="Forecasted Price (Rs./Quintal)",
        showlegend=False,
        template="plotly_white"
    )
    st.plotly_chart(bar_fig, use_container_width=True)
    st.caption("Color-coded bar chart showing monthly recommendations for the selected crop.")

    # --- Forecast Table ---
    st.subheader(f"Hybrid Forecasted Prices for July–December 2025")
    df_forecast = pd.DataFrame({
        "Month": forecast_months,
        "Hybrid Forecast (Rs./Quintal)": np.round(hybrid_forecast, 2),
        "Recommendation": recommendations
    })
    st.dataframe(df_forecast)

    # --- Final Farmer Suggestion with Regional Tips ---
    st.subheader("\U0001F4E2 Final Farmer Suggestion")
    overall_change = hybrid_forecast[-1] - hybrid_forecast[0]
    if overall_change > 500:
        st.success("🟢 Prices expected to rise. Recommendation: **Hold**")
        st.markdown("### 🗣 Tip in Kannada")
        st.info("**ಬೆಲೆ ಏರಿಕೆಯಾಗುತ್ತಿದೆ. ನಿಮ್ಮ ಉತ್ಪನ್ನವನ್ನು ಇನ್ನೂ ಕೆಲವು ದಿನ ಇರಿಸಿಕೊಳ್ಳಿ.**")
        st.markdown("### 🗣 Tip in Hindi")
        st.info("**कीमतें बढ़ रही हैं। अभी बेचने से बचें और कुछ समय इंतजार करें।**")
    elif overall_change < -500:
        st.warning("🔴 Prices expected to fall. Recommendation: **Sell Soon**")
        st.markdown("### 🗣 Tip in Kannada")
        st.info("**ಬೆಲೆ ಇಳಿಕೆಯಾಗುತ್ತಿದೆ. ಇನ್ನೂ ಹೆಚ್ಚಿನ ನಷ್ಟದಿಂದ ತಪ್ಪಿಸಿಕೊಳ್ಳಲು ಶೀಘ್ರವೇ ಮಾರಾಟ ಮಾಡಿ.**")
        st.markdown("### 🗣 Tip in Hindi")
        st.info("**कीमतें गिर रही हैं। नुकसान से बचने के लिए जल्दी बेचें।**")
    else:
        st.info("🟡 Prices stable. Recommendation: **Sell in phases or hold based on need**")
        st.markdown("### 🗣 Tip in Kannada")
        st.info("**ಬೆಲೆ ಸ್ಥಿರವಾಗಿದೆ. ನಿಮ್ಮ ಅಗತ್ಯವನ್ನು ಅವಲಂಬಿಸಿ ಮಾರಾಟ ಮಾಡಿ ಅಥವಾ ಕಾಯಿರಿ.**")
        st.markdown("### 🗣 Tip in Hindi")
        st.info("**कीमतें स्थिर हैं। आवश्यकता के अनुसार बेचें या थोड़ा और प्रतीक्षा करें।**")

# --- Footer ---
st.markdown("""
<style>
.kisansathi-footer {
    width: 100vw;
    position: relative;
    left: 50%;
    right: 50%;
    margin-left: -50vw;
    margin-right: -50vw;
    box-sizing: border-box;
    background-color: #457b9d;
    color: white;
    border-radius: 0 0 10px 10px;
    text-align: center;
    padding: 20px 0;
    margin-top: 40px;
}
</style>
<div class="kisansathi-footer">
    <p>© 2025 KisanSathi. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)

