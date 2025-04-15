import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import trimesh
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# 🧭 Rutas absolutas seguras
# -------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DATA_PATH = os.path.join(BASE_DIR, "data", "datos_generacion.csv")
STL_PATH = os.path.join(BASE_DIR, "data", "caldera.stl")
AUTOENCODER_PATH = os.path.join(DOCS_DIR, "autoencoder_modelo.h5")

# -------------------------
# ⚠️ Verificar existencia
# -------------------------
if not os.path.exists(AUTOENCODER_PATH):
    raise FileNotFoundError(f"❌ No se encontró el Autoencoder en:\n{AUTOENCODER_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ No se encontró el archivo de datos en:\n{DATA_PATH}")
if not os.path.exists(STL_PATH):
    raise FileNotFoundError(f"❌ No se encontró el archivo STL en:\n{STL_PATH}")

# -------------------------
# 📦 Cargar modelo
# -------------------------
autoencoder = load_model(AUTOENCODER_PATH, compile=False)

# -------------------------
# 📊 Cargar y preprocesar datos
# -------------------------
def cargar_y_preparar_datos():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    meses = {"ene": "Jan", "feb": "Feb", "mar": "Mar", "abr": "Apr", "may": "May", "jun": "Jun",
             "jul": "Jul", "ago": "Aug", "sep": "Sep", "oct": "Oct", "nov": "Nov", "dic": "Dec"}
    for esp, eng in meses.items():
        df["Fecha"] = df["Fecha"].str.replace(esp, eng, regex=False)
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d-%b-%y %H:%M:%S")
    for col in df.columns:
        if col != "Fecha":
            df[col] = (df[col]
                .str.replace(",", ".", regex=False)
                .apply(lambda x: x if isinstance(x, str) and x.replace(".", "", 1).isdigit() else np.nan)
                .astype(float))
    df = df.dropna()
    features = df.drop(columns=["Fecha", "total energia generada"])
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, df["Fecha"].reset_index(drop=True)

# -------------------------
# 🚨 Simulación
# -------------------------
def simular_streaming(data, fechas, umbral=0.0211, velocidad=0.3):
    if "errores" not in st.session_state:
        st.session_state.errores = []
        st.session_state.anomalias = []
        st.session_state.fechas_anomalias = []
        st.session_state.valores_anomalias = []
        st.session_state.stl_vertices = trimesh.load(STL_PATH).vertices
        st.session_state.stl_faces = trimesh.load(STL_PATH).faces

    status = st.empty()
    tabla_placeholder = st.empty()
    col1, col2 = st.columns([1.2, 1])
    stl_placeholder = col1.empty()
    chart_placeholder = col2.empty()

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
            status.success(f"✅ {fecha} | Operación normal | Error: {error:.4f}")

        # STL
        v = st.session_state.stl_vertices
        f = st.session_state.stl_faces
        center = v.mean(axis=0)
        v_halo = (v - center) * 1.03 + center
        x1, y1, z1 = v.T
        x2, y2, z2 = v_halo.T
        i_, j_, k_ = f.T

        if error < 0.015:
            halo_color = "rgba(0,255,0,0.6)"
            estado = "Normal"
        elif error < 0.025:
            halo_color = "rgba(255,215,0,0.6)"
            estado = "Precaución"
        else:
            halo_color = "rgba(255,0,0,0.6)"
            estado = "Anomalía"

        fig_stl = go.Figure(data=[
            go.Mesh3d(x=x2, y=y2, z=z2, i=i_, j=j_, k=k_, color=halo_color, opacity=0.2),
            go.Mesh3d(x=x1, y=y1, z=z1, i=i_, j=j_, k=k_, color="lightgray", opacity=1.0)
        ])
        fig_stl.update_layout(title=f"🧊 Modelo STL - Estado: {estado}",
                              scene=dict(aspectmode="data"),
                              margin=dict(l=0, r=0, b=0, t=40))

        stl_placeholder.plotly_chart(fig_stl, use_container_width=True, key=f"stl_chart_{i}")


        # Gráfico de errores
        df_plot = pd.DataFrame({
            "Índice": list(range(len(st.session_state.errores))),
            "Error": st.session_state.errores
        })
        df_plot["Anomalía"] = df_plot["Índice"].isin(st.session_state.anomalias)
        linea = alt.Chart(df_plot).mark_line().encode(x="Índice", y="Error")
        puntos_rojos = alt.Chart(df_plot[df_plot["Anomalía"] == True])\
            .mark_circle(color="red", size=60).encode(x="Índice", y="Error")

        chart_placeholder.altair_chart(linea + puntos_rojos, use_container_width=True)

        # Tabla
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
st.subheader("🔍 Dashboard de Detección de Anomalías en Tiempo Real")

if st.button("▶ Iniciar simulación"):
    st.info("Cargando datos y modelo...")
    data, fechas = cargar_y_preparar_datos()
    st.success("✅ Datos listos. Iniciando monitoreo...")
    simular_streaming(data, fechas, umbral=0.0211, velocidad=0.3)
