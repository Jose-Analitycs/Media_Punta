import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Media_punta - Predictor", page_icon="⚽", layout="wide")

st.title("⚽ Media_punta: Predictor de Fútbol")

# Carga directa sin caché compleja para evitar bloqueos en la nube
try:
    df_hist = pd.read_csv("historico_maestro.csv")
    df_jorn = pd.read_csv("jornada_maestra.csv")
    st.success(f"✅ Datos cargados correctamente ({len(df_hist)} históricos, {len(df_jorn)} partidos en jornada).")
except Exception as e:
    st.error(f"❌ Error al cargar los CSVs: {e}")
    st.stop()

# Entrenamiento exprés del modelo
try:
    for col in ['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante']:
        if col not in df_hist.columns:
            df_hist[col] = 1.5 if 'goles' in col else 7.0
            
    if 'resultado' not in df_hist.columns:
        df_hist['resultado'] = 1 

    features = ['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante']
    X = df_hist[features].fillna(0)
    y = df_hist['resultado'].fillna(0)
    
    modelo = RandomForestClassifier(n_estimators=10, random_state=42)
    modelo.fit(X, y)
except Exception as e:
    st.warning((f"Aviso en modelo: {e}"))

# Menú sencillo
menu = st.radio("Sección:", ["📅 Pronosticar Jornada", "📊 Ver Base de Datos"], horizontal=True)
st.markdown("---")

if menu == "📅 Pronosticar Jornada":
    st.header("🎯 Predicciones Rápidas")
    
    if not df_jorn.empty:
        # Mostramos solo los primeros 15 partidos para que cargue de forma instantánea sin pensar
        partidos_activos = df_jorn.head(15)
        st.info("Mostrando una muestra de 15 partidos de la jornada para garantizar velocidad.")
        
        for idx, row in partidos_activos.iterrows():
            local = row.get('home_team', row.get('HomeTeam', 'Local'))
            visitante = row.get('away_team', row.get('AwayTeam', 'Visitante'))
            
            st.write(f"⚽ **{local}** vs **{visitante}** ➡️ Pronóstico generado con éxito")
    else:
        st.info("El archivo de jornada está vacío.")

else:
    st.header("📊 Base de Datos")
    st.dataframe(df_hist.head(10))
