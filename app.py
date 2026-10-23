import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Media_punta - Predictor de Fútbol", page_icon="⚽", layout="wide")

st.title("⚽ Media_punta: Predictor de Fútbol")

@st.cache_resource
def cargar_y_entrenar_modelo():
    try:
        df_hist = pd.read_csv("historico_maestro.csv")
        df_jorn = pd.read_csv("jornada_maestra.csv")
        
        df_hist = df_hist.loc[:, ~df_hist.columns.str.contains('^Unnamed')]
        df_jorn = df_jorn.loc[:, ~df_jorn.columns.str.contains('^Unnamed')]
        
        if 'goles_local_previos' not in df_hist.columns:
            df_hist['goles_local_previos'] = 1.5
        if 'goles_visitante_previos' not in df_hist.columns:
            df_hist['goles_visitante_previos'] = 1.1
        if 'forma_local' not in df_hist.columns:
            df_hist['forma_local'] = 7
        if 'forma_visitante' not in df_hist.columns:
            df_hist['forma_visitante'] = 6
            
        if 'resultado' not in df_hist.columns:
            if 'result' in df_hist.columns:
                df_hist['resultado'] = df_hist['result'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
            else:
                df_hist['resultado'] = 1 

        features = ['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante']
        for f in features:
            if f not in df_hist.columns:
                df_hist[f] = 0.0

        X = df_hist[features].fillna(0)
        y = df_hist['resultado'].fillna(0)
        
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)
        
        return df_hist, modelo, df_jorn, True

    except Exception as e:
        st.error(f"❌ Error al procesar los datos: {e}")
        return None, None, None, False

df_historico, modelo_ml, df_jornada, exito = cargar_y_entrenar_modelo()

if not exito:
    st.warning("⚠️ Hubo un problema cargando los datos en la app.")
    st.stop()

def obtener_estadisticas_equipo(df_hist, nombre_equipo):
    try:
        goles_medios = 1.4
        forma_media = 7.0
        if not df_hist.empty:
            if 'goles_local_previos' in df_hist.columns:
                goles_medios = float(df_hist['goles_local_previos'].mean())
            if 'forma_local' in df_hist.columns:
                forma_media = float(df_hist['forma_local'].mean())
        return goles_medios, forma_media
    except:
        return 1.4, 7.0

menu = st.radio("Selecciona una sección:", ["📅 Pronosticar Jornada Actual", "🔍 Simulador Cara a Cara", "📊 Ver Base de Datos"], horizontal=True)
st.markdown("---")

if menu == "📅 Pronosticar Jornada Actual":
    st.header("🎯 Predicciones Inteligentes para la Jornada")
    
    if df_jornada is not None and not df_jornada.empty:
        # Buscamos de forma más amplia cualquier columna que pueda representar la Liga o Competición
        col_liga = None
        posibles_nombres = ['league', 'League', 'competition', 'Competition', 'Pais', 'Div', 'division', 'Division', 'tournament', 'competicion', 'liga']
        for posible in posibles_nombres:
            if posible in df_jornada.columns:
                col_liga = posible
                break
        
        # Si aún no la encuentra, buscamos la primera columna que tenga texto (tipo object/string)
        if not col_liga:
            for col in df_jornada.columns:
                if df_jornada[col].dtype == 'object' and col not in ['home_team', 'away_team', 'HomeTeam', 'AwayTeam', 'date', 'time', 'status', 'city', 'stadium']:
                    col_liga = col
                    break

        col_jornada = 'matchday' if 'matchday' in df_jornada.columns else (df_jornada.columns[2] if len(df_jornada.columns) > 2 else None)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if col_liga:
                ligas_disponibles = sorted(df_jornada[col_liga].dropna().astype(str).unique())
                liga_elegida = st.selectbox(f"🏆 Selecciona la Liga (Columna: {col_liga})", ligas_disponibles)
                df_filtrado_liga = df_jornada[df_jornada[col_liga].astype(str) == liga_elegida]
            else:
                df_filtrado_liga = df_jornada
                st.info("ℹ️ No se detectó columna de liga específica. Mostrando todos los partidos.")
        
        with col2:
            if col_jornada and col_jornada in df_filtrado_liga.columns:
                jornadas_disponibles = sorted(df_filtrado_liga[col_jornada].dropna().unique())
                jornada_elegida = st.selectbox("📅 Selecciona la Jornada / Fecha", jornadas_disponibles)
                partidos_activos = df_filtrado_liga[df_filtrado_liga[col_jornada] == jornada_elegida]
            else:
                partidos_activos = df_filtrado_liga
        
        st.markdown("---")
        st.subheader(f"Partidos analizados: {len(partidos_activos)}")
        
        for idx, row in partidos_activos.iterrows():
            local = row.get('home_team', row.get('HomeTeam', 'Local'))
            visitante = row.get('away_team', row.get('AwayTeam', 'Visitante'))
            
            g_loc, f_loc = obtener_estadisticas_equipo(df_historico, str(local))
            g_vis, f_vis = obtener_estadisticas_equipo(df_historico, str(visitante))
            
            datos_partido = pd.DataFrame([[g_loc, g_vis, f_loc, f_vis]], columns=['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante'])
            
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
