import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# --- Configuración de la página ---
st.set_page_config(page_title="Top 10 S&P 500 con AI", layout="wide")

st.title("📊 Ranking S&P 500 con AI")
st.markdown("Prototipo con datos de Yahoo Finance y predicciones semanales usando **Random Forest**")

# --- Descargar lista de empresas del S&P500 ---
@st.cache_data
def load_sp500_tickers():
    sp500 = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    )[0]
    return sp500['Symbol'].tolist()

tickers = load_sp500_tickers()

# --- Descargar precios últimos 90 días ---
end = datetime.today()
start = end - timedelta(days=90)
data = yf.download(tickers, start=start, end=end)['Adj Close']

# --- Cálculo de revalorización (últimos 30 días) ---
returns = (data.iloc[-1] / data.iloc[-22] - 1).sort_values(ascending=False)
top10 = returns.head(10)

st.subheader("🏆 Top 10 empresas con mayor revalorización (últimos 30 días)")
st.dataframe(top10)

# --- AI: Random Forest para predicciones ---
st.subheader("🤖 Predicción con Machine Learning (Random Forest)")

proyecciones = {}

for ticker in tickers[:50]:  # limitar a primeras 50 para no saturar
    precios = data[ticker].dropna()
    if len(precios) < 30:
        continue

    # Features: rendimientos pasados
    X, y = [], []
    for i in range(10, len(precios) - 5):
        past_returns = (precios.values[i-10:i] / precios.values[i-11:i-1] - 1)
        if np.isnan(past_returns).any():
            continue
        X.append(past_returns)
        future_return = (precios.values[i+5] - precios.values[i]) / precios.values[i]
        y.append(future_return)

    if len(X) < 20:
        continue

    X, y = np.array(X), np.array(y)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predicción usando últimos 10 días
    last_returns = (precios.values[-10:] / precios.values[-11:-1] - 1)
    pred = model.predict([last_returns])[0]

    proyecciones[ticker] = pred

# Ranking de predicciones
predicciones_df = pd.Series(proyecciones).sort_values(ascending=False).head(10)
st.dataframe(predicciones_df)

# --- Visualización de una empresa ---
st.subheader("🔍 Ver gráfico de una empresa")
empresa = st.selectbox("Elige ticker:", top10.index.tolist())
if empresa:
    fig, ax = plt.subplots()
    data[empresa].plot(ax=ax)
    ax.set_title(f"Evolución de {empresa}")
    st.pyplot(fig)
