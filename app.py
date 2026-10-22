import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Media_punta - Predictor de Fútbol", page_icon="⚽", layout="wide")

st.title("⚽ Media_punta: Predictor de Fútbol")

@st.cache_resource
def cargar_datos_maestros():
    try:
        # Intentamos leer los archivos de la raíz
        df_hist = pd.read_csv("historico_maestro.csv")
        df_jorn = pd.read_csv("jornada_maestra.csv")
        
        # Mostramos un aviso de éxito temporal para confirmar que los leyó
        st.success(f"✅ ¡Archivos leídos con éxito! Histórico: {len(df_hist)} filas | Jornada: {len(df_jorn)} filas")
        
        # Entrenar el modelo de Machine Learning
        X = df_hist[['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante']]
        y = df_hist['resultado']
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)
        
        return df_hist, modelo, df_jorn
        
    except Exception as e:
        st.error(f"❌ Error crítico al leer los CSVs: {e}")
        return None, None, None

# Cargamos los datos
df_historico, modelo_ml, df_jornada = cargar_datos_maestros()

# Si ha habido un error al cargar, detenemos la app para que no pete
if df_historico is None or modelo_ml is None or df_jornada is None:
    st.warning("⚠️ Comprueba que los archivos `historico_maestro.csv` y `jornada_maestra.csv` están exactamente en la raíz principal de tu GitHub y que las columnas coinciden.")
    st.stop()

# Menú de navegación
menu = st.radio("Selecciona una sección:", ["📅 Pronosticar Jornada Actual", "🔍 Simulador Cara a Cara", "📊 Ver Base de Datos"], horizontal=True)
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

elif menu == "🔍 Simulador Cara a Cara":
    st.header("⚙️ Simulador Individual (H2H)")
    col1, col2 = st.columns(2)
    with col1:
        g_loc = st.slider("Goles esperados (Local)", 0.0, 4.0, 1.6, 0.1)
        f_loc = st.slider("Puntos últimos 5 partidos (Local)", 0, 15, 9)
    with col2:
        g_vis = st.slider("Goles esperados (Visitante)", 0.0, 4.0, 1.2, 0.1)
        f_vis = st.slider("Puntos últimos 5 partidos (Visitante)", 0, 15, 6)
        
    if st.button("🚀 Calcular Pronóstico", use_container_width=True):
        custom_df = pd.DataFrame([[g_loc, g_vis, f_loc, f_vis]], columns=['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante'])
        p_res = modelo_ml.predict(custom_df)[0]
        p_probs = modelo_ml.predict_proba(custom_df)[0]
        mapa_res = {0: "Empate 🤝", 1: "Victoria Local 🏠", 2: "Victoria Visitante ✈️"}
        st.success(f"### Pronóstico del Modelo: {mapa_res[p_res]}")

else:
    st.header("📊 Resumen del Histórico Maestro")
    st.info(f"Partidos totales cargados en el sistema: **{len(df_historico):,}**")
    st.dataframe(df_historico.head(20), use_container_width=True)
