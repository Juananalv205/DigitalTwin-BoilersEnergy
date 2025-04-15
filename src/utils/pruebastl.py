import streamlit as st
import plotly.graph_objects as go
import trimesh
import os

# Configurar la página
st.set_page_config(layout="wide")
st.title("🔥 Visualizador 3D de Caldera con Halo Dinámico")

# Selección de estado
estado = st.radio("🧪 Estado del sistema:", ["Normal", "Precaución", "Anomalía"], index=0, horizontal=True)

# Ruta al STL
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
STL_PATH = os.path.join(BASE_DIR, "data", "caldera.stl")

# Cargar el STL solo una vez
@st.cache_resource
def cargar_malla(path):
    return trimesh.load(path)

if not os.path.exists(STL_PATH):
    st.error(f"❌ No se encontró el archivo STL en:\n{STL_PATH}")
    st.stop()

mesh = cargar_malla(STL_PATH)
vertices = mesh.vertices
faces = mesh.faces

# Extraer datos
x1, y1, z1 = vertices.T
i, j, k = faces.T

# Generar halo (capa expandida)
expansion_factor = 1.03
center = vertices.mean(axis=0)
expanded_vertices = (vertices - center) * expansion_factor + center
x2, y2, z2 = expanded_vertices.T

# Elegir color según estado
if estado == "Normal":
    halo_color = "rgba(0,255,0,0.6)"      # verde
elif estado == "Precaución":
    halo_color = "rgba(255,215,0,0.6)"    # amarillo dorado
else:
    halo_color = "rgba(255,0,0,0.6)"      # rojo

# Crear el gráfico Plotly
fig = go.Figure(data=[
    go.Mesh3d(
        x=x2, y=y2, z=z2,
        i=i, j=j, k=k,
        color=halo_color,
        opacity=0.2,
        name="Halo",
        flatshading=True,
        showscale=False
    ),
    go.Mesh3d(
        x=x1, y=y1, z=z1,
        i=i, j=j, k=k,
        color="lightgray",
        opacity=1.0,
        name="Caldera",
        flatshading=True,
        showscale=False
    )
])

fig.update_layout(
    title=f"Modelo 3D - Estado: {estado}",
    scene=dict(aspectmode="data"),
    margin=dict(l=0, r=0, t=40, b=0)
)

# Mostrar gráfico con actualizaciones rápidas
plot_placeholder = st.empty()
plot_placeholder.plotly_chart(fig, use_container_width=True)
