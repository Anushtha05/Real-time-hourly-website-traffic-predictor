import streamlit as st
from pytrends.request import TrendReq
from neuralprophet import NeuralProphet
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import time
import random

# -------------------- Utility --------------------
def human_delay(min_sec=0.5, max_sec=2):
    time.sleep(random.uniform(min_sec, max_sec))

# -------------------- Streamlit App Config --------------------
st.set_page_config(page_title="Hourly Website Traffic Forecast", layout="wide")
st.title("⏱️ Hourly Website Traffic Forecast (Google Trends + NeuralProphet)")

# -------------------- Select Website --------------------
websites = ['youtube.com', 'instagram.com', 'facebook.com', 'amazon.com', 'linkedin.com', 'bitmesra.ac.in']
selected_site = st.selectbox("Choose a Website:", websites)

# -------------------- Select Training Period --------------------
training_options = {
    "Last 1 Day": 1,
    "Last 3 Days": 3,
    "Last 7 Days": 7
}
selected_period_label = st.radio("Training Data Period:", list(training_options.keys()))
selected_days = training_options[selected_period_label]

# -------------------- Fetch Google Trends Data --------------------
@st.cache_data
def get_trend_data(keyword):
    pytrends = TrendReq()
    pytrends.build_payload([keyword], timeframe='now 7-d')  # hourly data for last 7 days
    df = pytrends.interest_over_time().reset_index()
    df = df.rename(columns={'date': 'ds', keyword: 'y'})
    df = df[['ds', 'y']]
    df = df.set_index('ds').resample('H').mean().fillna(0).reset_index()
    return df

with st.spinner("📡 Fetching data..."):
    human_delay(1, 2)
    data = get_trend_data(selected_site)

st.subheader("📊 Historical Hourly Google Search Interest")
st.line_chart(data.set_index('ds'))

# -------------------- Filter Training Data --------------------
cutoff_date = data['ds'].max() - pd.Timedelta(days=selected_days)
data_small = data[data['ds'] >= cutoff_date].reset_index(drop=True)

# Debug: Show training data sample
st.write(f"🧪 Sample of training data (last {selected_days} day(s)) — last 5 rows:")
st.write(data_small.tail())

# -------------------- Train NeuralProphet Model --------------------
st.info(f"🤖 Training model on the last {selected_days} day(s) of hourly data...")
with st.spinner("Training in progress..."):
    model = NeuralProphet(epochs=50)
    metrics = model.fit(data_small, freq="H", progress="bar")

st.success("✅ Model training completed!")

# -------------------- Forecast Next 48 Hours --------------------
future = model.make_future_dataframe(data_small, periods=48)
forecast = model.predict(future)

# Debug: Show prediction sample
st.write("📈 Sample forecast (last 5 predictions):")
st.write(forecast[['ds', 'yhat1']].tail())

# -------------------- Separate Past & Future --------------------
forecast['ds'] = pd.to_datetime(forecast['ds'])
actual_end = data_small['ds'].max()

past_forecast = forecast[forecast['ds'] <= actual_end]
future_forecast = forecast[forecast['ds'] > actual_end]

combined = pd.concat([
    past_forecast[['ds', 'yhat1']].assign(Type="Past Forecast"),
    future_forecast[['ds', 'yhat1']].assign(Type="Future Forecast")
])

# -------------------- Display Forecast Table --------------------
st.subheader("🕒 Forecast Table (Past + Next 48 Hours)")
st.dataframe(combined.tail(60).rename(columns={'ds': 'Datetime', 'yhat1': 'Forecast'}))

# -------------------- Plot Combined Forecast --------------------
st.subheader("📈 Forecast Visualization")
chart = alt.Chart(combined).mark_line().encode(
    x='ds:T',
    y='yhat1:Q',
    color='Type:N'
).properties(width=800, height=400)
st.altair_chart(chart)

# -------------------- Forecast Components --------------------
with st.expander("📉 Show Forecast Components (Trend, Seasonality)"):
    figs = model.plot_components(forecast)
    if isinstance(figs, list):
        for fig in figs:
            st.plotly_chart(fig)
    else:
        st.plotly_chart(figs)
