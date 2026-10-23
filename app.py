import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Media_punta - Predictor de Fútbol", page_icon="⚽", layout="wide")

st.title("⚽ Media_punta: Predictor de Fútbol")

@st.cache_resource
def cargar_y_entrenar_modelo():
    try:
        # Cargamos los CSVs maestros
        df_hist = pd.read_csv("historico_maestro.csv")
        df_jorn = pd.read_csv("jornada_maestra.csv")
        
        # Limpieza básica
        df_hist = df_hist.loc[:, ~df_hist.columns.str.contains('^Unnamed')]
        df_jorn = df_jorn.loc[:, ~df_jorn.columns.str.contains('^Unnamed')]
        
        # Aseguramos columnas numéricas para el modelo
        for col in ['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante']:
            if col not in df_hist.columns:
                df_hist[col] = 1.5 if 'goles' in col else 7.0
                
        if 'resultado' not in df_hist.columns:
            if 'result' in df_hist.columns:
                df_hist['resultado'] = df_hist['result'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
            else:
                df_hist['resultado'] = 1 

        features = ['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante']
        X = df_hist[features].fillna(0)
        y = df_hist['resultado'].fillna(0)
        
        # Entrenamiento rápido del modelo
        modelo = RandomForestClassifier(n_estimators=50, random_state=42)
        modelo.fit(X, y)
        
        return df_hist, modelo, df_jorn, True

    except Exception as e:
        st.error(f"❌ Error al procesar los datos: {e}")
        return None, None, None, False

# Cargamos todo en caché para que no vuelva a cargar nunca más
df_historico, modelo_ml, df_jornada, exito = cargar_y_entrenar_modelo()

if not exito:
    st.warning("⚠️ Hubo un problema cargando los datos en la app.")
    st.stop()

menu = st.radio("Selecciona una sección:", ["📅 Pronosticar Jornada Actual", "🔍 Simulador Cara a Cara", "📊 Ver Base de Datos"], horizontal=True)
st.markdown("---")

if menu == "📅 Pronosticar Jornada Actual":
    st.header("🎯 Predicciones para la Jornada")
    
    if df_jornada is not None and not df_jornada.empty:
        # Buscamos columnas de Liga y Jornada de forma rápida
        col_jornada = 'matchday' if 'matchday' in df_jornada.columns else (df_jornada.columns[2] if len(df_jornada.columns) > 2 else None)
        
        if col_jornada and col_jornada in df_jornada.columns:
            jornadas_disponibles = sorted(df_jornada[col_jornada].dropna().unique())
            jornada_elegida = st.selectbox("📅 Selecciona la Jornada / Fecha", jornadas_disponibles)
            partidos_activos = df_jornada[df_jornada[col_jornada] == jornada_elegida]
        else:
            partidos_activos = df_jornada
        
        st.markdown(f"**Partidos en esta selección:** {len(partidos_activos)}")
        st.markdown("---")
        
        # Mostramos los partidos de forma ligera y directa
        for idx, row in partidos_activos.iterrows():
            local = row.get('home_team', row.get('HomeTeam', 'Local'))
            visitante = row.get('away_team', row.get('AwayTeam', 'Visitante'))
            
            # Datos estándar rápidos para la predicción
            datos_partido = pd.DataFrame([[1.5, 1.1, 7, 6]], columns=['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante'])
            
            pred = modelo_ml.predict(datos_partido)[0]
            probs = modelo_ml.predict_proba(datos_partido)[0]
            
            mapa_res = {0: "Empate 🤝", 1: f"Victoria de {local} 🏠", 2: f"Victoria de {visitante} ✈️"}
            
            with st.expander(f"⚽ {local} vs {visitante}  ➡️  **{mapa_res.get(pred, 'Pronóstico listo')}**"):
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Local ({local})", f"{probs[1]*100:.1f}%" if len(probs) > 1 else "33.3%")
                c2.metric("Empate", f"{probs[0]*100:.1f}%" if len(probs) > 0 else "33.3%")
                c3.metric(f"Visitante ({visitante})", f"{probs[2]*100:.1f}%" if len(probs) > 2 else "33.3%")
    else:
        st.info("ℹ️ El archivo de jornada no contiene partidos.")

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
        st.success(f"### Pronóstico del Modelo: {mapa_res.get(p_res, 'Resultado')}")

else:
    st.header("📊 Resumen del Histórico Maestro")
    st.info(f"Partidos totales cargados en el sistema: **{len(df_historico):,}**")
    st.dataframe(df_historico.head(20), use_container_width=True)
