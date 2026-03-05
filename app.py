import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# ----------------------------------------------------
# Page Setup
# ----------------------------------------------------

st.set_page_config(page_title="GoldVision AI", layout="wide")

st_autorefresh(interval=60000)

# ----------------------------------------------------
# Custom UI
# ----------------------------------------------------

st.markdown("""
<style>

body{
background-color:#0f172a;
}

.title{
font-size:45px;
font-weight:bold;
color:#FFD700;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">GoldVision AI — Indian Gold Analytics Dashboard</p>', unsafe_allow_html=True)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.header("Dashboard Controls")

years = st.sidebar.slider("Historical Data (Years)",1,10,5)

city = st.sidebar.selectbox(
"Select City",
["Delhi","Mumbai","Chennai","Hyderabad","Kolkata"]
)

# ----------------------------------------------------
# Download Data
# ----------------------------------------------------

gold = yf.download("GC=F",period=f"{years}y")

gold.reset_index(inplace=True)

usd_inr = float(yf.download("USDINR=X",period="1d")["Close"].iloc[-1])

# ----------------------------------------------------
# Convert to Indian Gold Price
# ----------------------------------------------------

gold_usd = gold["Close"]

gold_inr_per_gram = (gold_usd*usd_inr)/31.1035

gold_24k = gold_inr_per_gram*10

india_premium = 1.12

gold_24k = gold_24k*india_premium

gold["Gold_24K"]=gold_24k
gold["Gold_22K"]=gold_24k*0.916
gold["Gold_18K"]=gold_24k*0.75

# ----------------------------------------------------
# City Prices
# ----------------------------------------------------

city_factor={
"Delhi":1.00,
"Mumbai":0.998,
"Chennai":1.01,
"Hyderabad":1.005,
"Kolkata":0.997
}

city_price=float(gold["Gold_24K"].iloc[-1])*city_factor[city]

# ----------------------------------------------------
# Price Cards
# ----------------------------------------------------

st.subheader("Live Gold Prices")

col1,col2,col3,col4 = st.columns(4)

col1.metric("24K Gold",f"₹{float(gold['Gold_24K'].iloc[-1]):.2f}")
col2.metric("22K Gold",f"₹{float(gold['Gold_22K'].iloc[-1]):.2f}")
col3.metric("18K Gold",f"₹{float(gold['Gold_18K'].iloc[-1]):.2f}")
col4.metric(f"{city} Price",f"₹{city_price:.2f}")

# ----------------------------------------------------
# Tabs
# ----------------------------------------------------

tab1,tab2,tab3,tab4,tab5 = st.tabs([
"Daily Trend",
"Monthly Trend",
"Yearly Trend",
"Candlestick",
"AI Prediction"
])

# ----------------------------------------------------
# Daily Chart
# ----------------------------------------------------

with tab1:

    fig = go.Figure()

    fig.add_trace(go.Scatter(
    x=gold["Date"],
    y=gold["Gold_24K"],
    name="24K",
    line=dict(color="gold",width=3)
    ))

    fig.add_trace(go.Scatter(
    x=gold["Date"],
    y=gold["Gold_22K"],
    name="22K",
    line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
    x=gold["Date"],
    y=gold["Gold_18K"],
    name="18K",
    line=dict(color="cyan")
    ))

    fig.update_layout(template="plotly_dark",height=600)

    st.plotly_chart(fig,use_container_width=True)

# ----------------------------------------------------
# Monthly Chart
# ----------------------------------------------------

with tab2:

    monthly = gold.resample("M",on="Date").mean().reset_index()

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=monthly["Date"],y=monthly["Gold_24K"],name="24K"))
    fig2.add_trace(go.Scatter(x=monthly["Date"],y=monthly["Gold_22K"],name="22K"))
    fig2.add_trace(go.Scatter(x=monthly["Date"],y=monthly["Gold_18K"],name="18K"))

    fig2.update_layout(template="plotly_dark",height=600)

    st.plotly_chart(fig2,use_container_width=True)

# ----------------------------------------------------
# Yearly Chart
# ----------------------------------------------------

with tab3:

    yearly = gold.resample("Y",on="Date").mean().reset_index()

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(x=yearly["Date"],y=yearly["Gold_24K"],name="24K"))
    fig3.add_trace(go.Scatter(x=yearly["Date"],y=yearly["Gold_22K"],name="22K"))
    fig3.add_trace(go.Scatter(x=yearly["Date"],y=yearly["Gold_18K"],name="18K"))

    fig3.update_layout(template="plotly_dark",height=600)

    st.plotly_chart(fig3,use_container_width=True)

# ----------------------------------------------------
# Candlestick Chart
# ----------------------------------------------------

with tab4:

    candle = go.Figure(data=[go.Candlestick(
    x=gold["Date"],
    open=gold["Open"],
    high=gold["High"],
    low=gold["Low"],
    close=gold["Close"],
    increasing_line_color="green",
    decreasing_line_color="red"
    )])

    candle.update_layout(template="plotly_dark",height=600)

    st.plotly_chart(candle,use_container_width=True)

# ----------------------------------------------------
# Moving Average
# ----------------------------------------------------

gold["MA20"]=gold["Gold_24K"].rolling(20).mean()

st.subheader("Gold Price with Moving Average")

fig_ma = go.Figure()

fig_ma.add_trace(go.Scatter(
x=gold["Date"],
y=gold["Gold_24K"],
name="Price"
))

fig_ma.add_trace(go.Scatter(
x=gold["Date"],
y=gold["MA20"],
name="MA20"
))

fig_ma.update_layout(template="plotly_dark")

st.plotly_chart(fig_ma,use_container_width=True)

# ----------------------------------------------------
# AI Prediction
# ----------------------------------------------------

with tab5:

    df_ml = gold.copy()

    df_ml["Index"]=np.arange(len(df_ml))

    X=df_ml[["Index"]]

    y=df_ml["Gold_24K"]

    model=LinearRegression()

    model.fit(X,y)

    future_days=np.arange(len(df_ml),len(df_ml)+30).reshape(-1,1)

    prediction=model.predict(future_days)

    future_dates=pd.date_range(
    start=gold["Date"].iloc[-1],
    periods=30,
    freq="D"
    )

    fig4=go.Figure()

    fig4.add_trace(go.Scatter(
    x=gold["Date"],
    y=gold["Gold_24K"],
    name="Historical Price"
    ))

    fig4.add_trace(go.Scatter(
    x=future_dates,
    y=prediction,
    name="AI Prediction",
    line=dict(color="lime",dash="dash")
    ))

    fig4.update_layout(template="plotly_dark")

    st.plotly_chart(fig4,use_container_width=True)
    st.subheader("RSI Indicator")

delta = gold["Gold_24K"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()

avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

gold["RSI"] = 100 - (100/(1+rs))

fig_rsi = go.Figure()

fig_rsi.add_trace(go.Scatter(
x=gold["Date"],
y=gold["RSI"],
name="RSI"
))

fig_rsi.update_layout(template="plotly_dark")

st.plotly_chart(fig_rsi,use_container_width=True)
st.subheader("MACD Indicator")

exp1 = gold["Gold_24K"].ewm(span=12,adjust=False).mean()

exp2 = gold["Gold_24K"].ewm(span=26,adjust=False).mean()

gold["MACD"] = exp1-exp2

gold["Signal"] = gold["MACD"].ewm(span=9,adjust=False).mean()

fig_macd = go.Figure()

fig_macd.add_trace(go.Scatter(
x=gold["Date"],
y=gold["MACD"],
name="MACD"
))

fig_macd.add_trace(go.Scatter(
x=gold["Date"],
y=gold["Signal"],
name="Signal"
))

fig_macd.update_layout(template="plotly_dark")

st.plotly_chart(fig_macd,use_container_width=True)
st.subheader("Bollinger Bands")

gold["MA20"] = gold["Gold_24K"].rolling(20).mean()

gold["STD"] = gold["Gold_24K"].rolling(20).std()

gold["Upper"] = gold["MA20"] + (gold["STD"]*2)

gold["Lower"] = gold["MA20"] - (gold["STD"]*2)

fig_bb = go.Figure()

fig_bb.add_trace(go.Scatter(
x=gold["Date"],
y=gold["Gold_24K"],
name="Price"
))

fig_bb.add_trace(go.Scatter(
x=gold["Date"],
y=gold["Upper"],
name="Upper Band"
))

fig_bb.add_trace(go.Scatter(
x=gold["Date"],
y=gold["Lower"],
name="Lower Band"
))

fig_bb.update_layout(template="plotly_dark")

st.plotly_chart(fig_bb,use_container_width=True)