import streamlit as st
import pandas as pd
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= LOAD TRAINED PIPELINE =================
model = joblib.load("Flight_Price_Pipeline.pkl")

# ================= DESKTOP STYLE CSS =================
st.markdown("""
<style>

/* App background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #eef2f3, #ffffff);
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1f2937;
}
[data-testid="stSidebar"] * {
    color: white;
}

/* Cards */
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Button */
.stButton>button {
    width: 100%;
    height: 55px;
    background: linear-gradient(90deg, #4CAF50, #2E8B57);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    border: none;
}

/* Prediction box */
.result-box {
    background: #e8f5e9;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #1b5e20;
    margin-top: 20px;
}

/* Section headers */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("✈️ Flight Price Prediction App")
st.write("Desktop-style web application using Machine Learning pipeline")

# ================= SIDEBAR =================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Insights"])

# ================= HOME PAGE =================
if page == "Home":

    st.markdown("## ✈️ Predict Your Flight Price")

    col1, col2 = st.columns(2)

    # -------- LEFT CARD --------
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Flight Details</div>', unsafe_allow_html=True)

        airline = st.selectbox(
            "Airline",
            ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air India']
        )

        source_city = st.selectbox(
            "Source City",
            ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
        )

        destination_city = st.selectbox(
            "Destination City",
            ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
        )

        stops = st.selectbox(
            "Stops",
            ['zero', 'one', 'two_or_more']
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # -------- RIGHT CARD --------
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Time & Class</div>', unsafe_allow_html=True)

        departure_time = st.selectbox(
            "Departure Time",
            ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night']
        )

        arrival_time = st.selectbox(
            "Arrival Time",
            ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night']
        )

        flight_class = st.selectbox(
            "Class",
            ['Economy', 'Business']
        )

        duration_hours = st.number_input("Duration (Hours)", 0, 24, 2)
        duration_minutes = st.number_input("Duration (Minutes)", 0, 59, 30)

        days_left = st.slider("Days Left for Departure", 1, 50, 20)

        st.markdown('</div>', unsafe_allow_html=True)

    # -------- PREDICTION --------
    duration = duration_hours * 60 + duration_minutes

    if st.button("Predict Price"):

        input_data = pd.DataFrame([{
            'airline': airline,
            'source_city': source_city,
            'destination_city': destination_city,
            'stops': stops,
            'departure_time': departure_time,
            'arrival_time': arrival_time,
            'class': flight_class,
            'duration': duration,
            'days_left': days_left
        }])

        prediction = model.predict(input_data)[0]

        st.markdown(
            f"""
            <div class="result-box">
                💰 Estimated Flight Price <br>
                ₹ {int(prediction)}
            </div>
            """,
            unsafe_allow_html=True
        )

# ================= DATA INSIGHTS PAGE =================
elif page == "Data Insights":

    st.subheader("📊 Project Insights")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.write("""
    • Random Forest Regression model  
    • Preprocessing + Model combined using Pipeline  
    • No manual encoding during deployment  
    • Same transformations applied during training & prediction  
    • Production-safe and deployment-ready architecture  
    """)

    st.markdown('</div>', unsafe_allow_html=True)
