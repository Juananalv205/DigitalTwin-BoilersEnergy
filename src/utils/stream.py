import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import altair as alt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# 🧭 Rutas absolutas seguras
# -------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DATA_PATH = os.path.join(BASE_DIR, "data", "datos_generacion.csv")
AUTOENCODER_PATH = os.path.join(DOCS_DIR, "autoencoder_modelo.h5")

# -------------------------
# ⚠️ Verificar existencia
# -------------------------
if not os.path.exists(AUTOENCODER_PATH):
    raise FileNotFoundError(f"❌ No se encontró el Autoencoder en:\n{AUTOENCODER_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ No se encontró el archivo de datos en:\n{DATA_PATH}")

# -------------------------
# 📦 Cargar modelo
# -------------------------
autoencoder = load_model(AUTOENCODER_PATH, compile=False)

# -------------------------
# 📊 Cargar y preprocesar datos
# -------------------------
def cargar_y_preparar_datos():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")

    meses = {"ene": "Jan", "feb": "Feb", "mar": "Mar", "abr": "Apr",
             "may": "May", "jun": "Jun", "jul": "Jul", "ago": "Aug",
             "sep": "Sep", "oct": "Oct", "nov": "Nov", "dic": "Dec"}
    for esp, eng in meses.items():
        df["Fecha"] = df["Fecha"].str.replace(esp, eng, regex=False)

    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d-%b-%y %H:%M:%S")

    for col in df.columns:
        if col != "Fecha":
            df[col] = (
                df[col]
                .str.replace(",", ".", regex=False)
                .apply(lambda x: x if isinstance(x, str) and x.replace(".", "", 1).isdigit() else np.nan)
                .astype(float)
            )

    df = df.dropna()
    features = df.drop(columns=["Fecha", "total energia generada"])
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, df["Fecha"].reset_index(drop=True)

# -------------------------
# 🚨 Simulación de streaming con corrección gráfica
# -------------------------
def simular_streaming(data, fechas, umbral=0.0211, velocidad=0.3):
    if "errores" not in st.session_state:
        st.session_state.errores = []
        st.session_state.anomalias = []
        st.session_state.fechas_anomalias = []
        st.session_state.valores_anomalias = []

    status = st.empty()
    chart_placeholder = st.empty()
    tabla_placeholder = st.empty()

    for i in range(len(data)):
        fila = data[i].reshape(1, -1)
        reconstruccion = autoencoder.predict(fila, verbose=0)
        error = np.mean(np.square(fila - reconstruccion))

        st.session_state.errores.append(error)

        fecha = fechas[i]

        if error > umbral:
            status.error(f"❌ {fecha} | Anomalía detectada | Error: {error:.4f}")
            st.session_state.anomalias.append(i)
            st.session_state.fechas_anomalias.append(fecha)
            st.session_state.valores_anomalias.append(error)
        else:
            status.success(f"✅ {fecha} | Operación normal | Último error: {error:.4f}")

        # --- Gráfico persistente ---
        df_plot = pd.DataFrame({
            "Índice": list(range(len(st.session_state.errores))),
            "Error": st.session_state.errores
        })
        df_plot["Anomalía"] = df_plot["Índice"].isin(st.session_state.anomalias)

        linea = alt.Chart(df_plot).mark_line().encode(
            x="Índice",
            y="Error"
        )

        puntos_rojos = alt.Chart(df_plot[df_plot["Anomalía"] == True]).mark_circle(color="red", size=60).encode(
            x="Índice",
            y="Error"
        )

        chart_placeholder.altair_chart(linea + puntos_rojos, use_container_width=True)

        # --- Tabla de anomalías ---
        if st.session_state.anomalias:
            df_anomalias = pd.DataFrame({
                "Índice": st.session_state.anomalias,
                "Fecha": st.session_state.fechas_anomalias,
                "Error": np.round(st.session_state.valores_anomalias, 5)
            })
            tabla_placeholder.dataframe(df_anomalias, use_container_width=True)

        time.sleep(velocidad)

# -------------------------
# 🎛️ Interfaz Streamlit
# -------------------------
st.set_page_config(page_title="Gemelo Digital - Anomalías", layout="wide")
st.title("🌡️ Gemelo Digital - Sistema de Generación de Vapor")
st.subheader("🔍 Detección de Anomalías en Tiempo Real")

if st.button("▶ Iniciar simulación"):
    st.info("Cargando datos y modelo...")
    data, fechas = cargar_y_preparar_datos()
    st.success("✅ Datos listos. Iniciando monitoreo...")
    simular_streaming(data, fechas, umbral=0.0211, velocidad=0.3)

