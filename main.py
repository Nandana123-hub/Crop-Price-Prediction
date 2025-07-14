import streamlit as st
import datetime
import base64
from dashboard import dashboard  # Ensure dashboard.py has a function dashboard(crop_choice, start_date, end_date)

st.set_page_config(
    layout="wide",
    page_title="KisanSathi - Crop Price Forecast",
    initial_sidebar_state="auto"
)

# Custom CSS
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .main {padding-top: 0rem !important;}
        .custom-navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #457b9d;
            padding: 0 32px;
            height: 70px;
            border-radius: 0 0 12px 12px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .custom-navbar-logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .custom-navbar-logo img {
            height: 94px;
            border-radius: 6px;
        }
        .custom-navbar-title {
            color: #f1faee;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: 1px;
        }
        .custom-navbar-links {
            display: flex;
            align-items: center;
            gap: 180px;
            margin-left: 60px;
            margin-right: 60px;
        }
        .custom-navbar-link {
            color: white !important;
            text-decoration: none !important;
            font-size: 1.1rem;
            font-weight: 500;
            transition: color 0.2s;
        }
        .custom-navbar-link:hover {
            color: #ffd166 !important;
        }
        .custom-navbar-select select {
            background: #1d3557;
            color: #f1faee;
            border-radius: 5px;
            border: none;
            padding: 7px 18px;
            font-size: 1rem;
            font-weight: 500;
            outline: none;
        }
    </style>
""", unsafe_allow_html=True)

# Logo Handling
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    img_base64 = get_base64_of_bin_file("img.png")
    logo_img_tag = f'<img src="data:image/png;base64,{img_base64}"/>'
except Exception:
    logo_img_tag = '<img src="https://via.placeholder.com/120x60.png?text=Logo"/>'

# Navbar
st.markdown(f"""
<nav class="custom-navbar">
    <div class="custom-navbar-logo">
        {logo_img_tag}
        <span class="custom-navbar-title">KisanSathi</span>
    </div>
    <div class="custom-navbar-links">
        <a class="custom-navbar-link" href="#">Home</a>
        <a class="custom-navbar-link" href="#">Price Trend</a>
        <a class="custom-navbar-link" href="#">Contact</a>
    </div>
    <div class="custom-navbar-select">
        <select>
            <option>English</option>
        </select>
    </div>
</nav>
""", unsafe_allow_html=True)

# Form
with st.form(key="filter_form"):
    f1, f2, f3, f4 = st.columns(4, vertical_alignment="bottom")
    f5, f6, f7, f8 = st.columns(4, vertical_alignment="bottom")

    with f1:
        st.selectbox("Price", ["Modal Price"])
    with f2:
        commodity = st.selectbox("Commodity", [
            "Arecanut", "Black Pepper", "Copra", "Groundnut", "Maize",
            "Mustard", "Ragi", "Rice", "Turmeric", "Wheat"
        ])
    with f3:
        st.selectbox("State", ["Karnataka"])
    with f4:
        st.selectbox("District", ["Bangalore"])
    with f5:
        st.selectbox("Market", ["Bangalore"])
    with f6:
        date_from = st.date_input("Date From", datetime.date.today())
    with f7:
        date_to = st.date_input("Date To", datetime.date.today())
    with f8:
        submit = st.form_submit_button("Go", use_container_width=True)

# Run dashboard on button click
if submit:
    dashboard(commodity, date_from, date_to)

# Optional static dashboard images below
st.markdown('<h3 style="color:#1d3557;">📊 Dashboard</h3>', unsafe_allow_html=True)

with st.container():
    d1, d2 = st.columns(2)
    with d1:
        st.image("https://via.placeholder.com/500x300.png?text=Commodity+Graph", use_column_width=True)
        st.caption("Commodity-wise Price Trend")
    with d2:
        st.image("https://via.placeholder.com/500x300.png?text=Market+Graph", use_column_width=True)
        st.caption("Market-wise Price Comparison")

with st.container():
    d3, d4 = st.columns(2)
    with d3:
        st.image("https://via.placeholder.com/500x300.png?text=Daily+Market+Report", use_column_width=True)
        st.caption("Daily Market Report")
    with d4:
        st.image("https://via.placeholder.com/500x300.png?text=Forecasted+Prices", use_column_width=True)
        st.caption("Forecasted Price Graph")

# Footer
st.markdown("""
<div style="text-align: center; padding: 20px; margin-top: 40px; background-color: #457b9d; color: white; border-radius: 10px;">
    <p>© 2024 KisanSathi. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
