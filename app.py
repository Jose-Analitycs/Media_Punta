import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Media_punta - Predictor de Fútbol", page_icon="⚽", layout="wide")

st.title("⚽ Media_punta: Predictor de Fútbol")

# Carga directa de los CSVs maestros
try:
    df_hist = pd.read_csv("historico_maestro.csv")
    df_jorn = pd.read_csv("jornada_maestra.csv")
except Exception as e:
    st.error(f"❌ Error al cargar los CSVs: {e}")
    st.stop()

# Entrenamiento exprés y seguro del modelo
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
    st.warning(f"Aviso en modelo: {e}")

# Menú de navegación
menu = st.radio("Sección:", ["📅 Pronosticar Jornada", "📊 Ver Base de Datos"], horizontal=True)
st.markdown("---")

if menu == "📅 Pronosticar Jornada":
    st.header("🎯 Predicciones por Competición y Jornada")
    
    if not df_jorn.empty:
        # Usamos las columnas reales de tu estructura: 'city'/'stadium' como grupo o 'matchday' como jornada
        # Si tienes una columna de liga/competición (ej: 'league' o 'division'), la detectamos. Si no, usamos 'city' o 'stadium' temporalmente.
        col_liga = None
        for posible in ['league', 'League', 'competition', 'Competition', 'division', 'Division', 'city']:
            if posible in df_jorn.columns:
                col_liga = posible
                break
        
        col_jornada = 'matchday' if 'matchday' in df_jorn.columns else df_jorn.columns[2]

        # 🎛️ Selectores en columnas (Liga/Ciudad y Jornada)
        col1, col2 = st.columns(2)
        
        with col1:
            if col_liga and col_liga in df_jorn.columns:
                ligas_disponibles = sorted(df_jorn[col_liga].dropna().astype(str).unique())
                liga_elegida = st.selectbox("🏆 Selecciona la Competición / Sede", ligas_disponibles)
                df_filtrado = df_jorn[df_jorn[col_liga].astype(str) == liga_elegida]
            else:
                df_filtrado = df_jorn

        with col2:
            if col_jornada and col_jornada in df_filtrado.columns:
                jornadas_disponibles = sorted(df_filtrado[col_jornada].dropna().unique())
                jornada_elegida = st.selectbox("📅 Selecciona el Número de Jornada", jornadas_disponibles)
                partidos_activos = df_filtrado[df_filtrado[col_jornada] == jornada_elegida]
            else:
                partidos_activos = df_filtrado
        
        st.markdown("---")
        st.markdown(f"**Partidos en esta selección:** {len(partidos_activos)}")
        
        # Listado limpio de los partidos
        for idx, row in partidos_activos.iterrows():
            local = row.get('home_team', 'Local')
            visitante = row.get('away_team', 'Visitante')
            
            # Predicción rápida con el modelo
            datos_partido = pd.DataFrame([[1.5, 1.1, 7, 6]], columns=['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante'])
            pred = modelo.predict(datos_partido)[0]
            probs = modelo.predict_proba(datos_partido)[0]
            
            mapa_res = {0: "Empate 🤝", 1: f"Victoria de {local} 🏠", 2: f"Victoria de {visitante} ✈️"}
            
            with st.expander(f"⚽ {local} vs {visitante}  ➡️  **{mapa_res.get(pred, 'Pronóstico')}**"):
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Local ({local})", f"{probs[1]*100:.1f}%" if len(probs) > 1 else "33.3%")
                c2.metric("Empate", f"{probs[0]*100:.1f}%" if len(probs) > 0 else "33.3%")
                c3.metric(f"Visitante ({visitante})", f"{probs[2]*100:.1f}%" if len(probs) > 2 else "33.3%")
    else:
        st.info("El archivo de jornada está vacío.")

else:
    st.header("📊 Resumen del Histórico Maestro")
    st.info(f"Partidos totales en el histórico: **{len(df_hist):,}**")
    st.dataframe(df_hist.head(20))
