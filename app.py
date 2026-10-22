import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Configuración de la página web
st.set_page_config(
    page_title="Predictor de Fútbol - Media_punta", 
    page_icon="⚽", 
    layout="wide"
)

st.title("⚽ Predictor Inteligente de Fútbol")
st.markdown("Aplicación conectada directamente a los datos de tu repositorio en GitHub.")

# ==========================================
# CARGAR DATOS DIRECTAMENTE DESDE GITHUB
# ==========================================
@st.cache_resource
def cargar_entorno_desde_github():
    # Enlaces directos a los archivos en tu repositorio de GitHub (Modifica los nombres si tus CSV se llaman distinto)
    url_historico = "https://raw.githubusercontent.com/TU_USUARIO_GITHUB/Media_punta/main/historico_4_temporadas.csv"
    url_jornada = "https://raw.githubusercontent.com/TU_USUARIO_GITHUB/Media_punta/main/jornada_actual.csv"
    
    try:
        df_hist = pd.read_csv(url_historico)
    except Exception:
        # Datos de respaldo por si falla la descarga inicial
        df_hist = pd.DataFrame({
            'HomeTeam': ['Real Madrid', 'Barcelona', 'Atlético', 'Sevilla'],
            'AwayTeam': ['Valencia', 'Villarreal', 'Betis', 'Real Sociedad'],
            'goles_local_previos': [2.1, 2.0, 1.6, 1.4],
            'goles_visitante_previos': [1.1, 1.3, 1.0, 1.2],
            'forma_local': [13, 12, 10, 8],
            'forma_visitante': [7, 9, 8, 7],
            'resultado': [1, 1, 0, 2]
        })
    
    # Entrenar el modelo de Machine Learning
    X = df_hist[['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante']]
    y = df_hist['resultado']
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    try:
        df_jorn = pd.read_csv(url_jornada)
    except Exception:
        df_jorn = pd.DataFrame({
            'Jornada': [28, 28, 28, 28],
            'HomeTeam': ['Real Madrid', 'Barcelona', 'Atlético', 'Sevilla'],
            'AwayTeam': ['Valencia', 'Villarreal', 'Betis', 'Real Sociedad'],
            'goles_local_previos': [2.0, 1.9, 1.5, 1.3],
            'goles_visitante_previos': [1.0, 1.2, 1.1, 1.1],
            'forma_local': [12, 11, 9, 7],
            'forma_visitante': [6, 8, 7, 6]
        })
        
    return df_hist, modelo, df_jorn

df_historico, modelo_ml, df_jornada = cargar_entorno_desde_github()

# ==========================================
# INTERFAZ PRINCIPAL DE LA APLICACIÓN
# ==========================================
menu = st.radio(
    "Selecciona una sección:", 
    ["📅 Pronosticar Jornada Actual", "🔍 Simulador Individual (H2H)", "📊 Ver Base de Datos"], 
    horizontal=True
)
st.markdown("---")

if menu == "📅 Pronosticar Jornada Actual":
    st.header("🎯 Predicciones para la Jornada")
    
    if 'Jornada' in df_jornada.columns:
        jornada_elegida = st.selectbox("Elige el número de Jornada", sorted(df_jornada['Jornada'].unique()))
        partidos_activos = df_jornada[df_jornada['Jornada'] == jornada_elegida]
    else:
        partidos_activos = df_jornada
        
    for idx, row in partidos_activos.iterrows():
        local = row['HomeTeam']
        visitante = row['AwayTeam']
        
        datos_partido = pd.DataFrame([[
            row['goles_local_previos'], 
            row['goles_visitante_previos'], 
            row['forma_local'], 
            row['forma_visitante']
        ]], columns=['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante'])
        
        pred = modelo_ml.predict(datos_partido)[0]
        probs = modelo_ml.predict_proba(datos_partido)[0]
        
        mapa_res = {0: "Empate 🤝", 1: f"Victoria de {local} 🏠", 2: f"Victoria de {visitante} ✈️"}
        
        with st.expander(f"⚽ {local} vs {visitante}  ➡️  **{mapa_res[pred]}**"):
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Local ({local})", f"{probs[1]*100:.1f}%")
            c2.metric("Empate", f"{probs[0]*100:.1f}%")
            c3.metric(f"Visitante ({visitante})", f"{probs[2]*100:.1f}%")

elif menu == "🔍 Simulador Individual (H2H)":
    st.header("⚙️ Simulación Cara a Cara")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Equipo Local")
        g_loc = st.slider("Goles esperados (Local)", 0.0, 4.0, 1.6, 0.1)
        f_loc = st.slider("Puntos últimos 5 partidos (Local)", 0, 15, 9)
    with col2:
        st.subheader("Equipo Visitante")
        g_vis = st.slider("Goles esperados (Visitante)", 0.0, 4.0, 1.2, 0.1)
        f_vis = st.slider("Puntos últimos 5 partidos (Visitante)", 0, 15, 6)
        
    if st.button("🚀 Calcular Pronóstico", use_container_width=True):
        custom_df = pd.DataFrame([[g_loc, g_vis, f_loc, f_vis]], 
                                    columns=['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante'])
        p_res = modelo_ml.predict(custom_df)[0]
        p_probs = modelo_ml.predict_proba(custom_df)[0]
        
        mapa_res = {0: "Empate 🤝", 1: "Victoria Local 🏠", 2: "Victoria Visitante ✈️"}
        st.success(f"### Pronóstico del Modelo: {mapa_res[p_res]}")
        
        chart_df = pd.DataFrame({
            'Resultado': ['Victoria Local', 'Empate', 'Victoria Visitante'],
            'Probabilidad (%)': [p_probs[1]*100, p_probs[0]*100, p_probs[2]*100]
        }).set_index('Resultado')
        st.bar_chart(chart_df)

else:
    st.header("📊 Resumen del Histórico Cargado")
    st.info(f"Partidos totales en la base de datos: **{len(df_historico):,}**")
    st.dataframe(df_historico.head(20), use_container_width=True)
