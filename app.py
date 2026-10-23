import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Media_punta - Predictor de Fútbol", page_icon="⚽", layout="wide")

st.title("⚽ Media_punta: Predictor Inteligente")

# 1. Carga directa de los CSVs maestros
@st.cache_resource
def cargar_datos():
    try:
        df_hist = pd.read_csv("historico_maestro.csv")
        df_jorn = pd.read_csv("jornada_maestra.csv")
        
        df_hist = df_hist.loc[:, ~df_hist.columns.str.contains('^Unnamed')]
        df_jorn = df_jorn.loc[:, ~df_jorn.columns.str.contains('^Unnamed')]
        
        return df_hist, df_jorn, True
    except Exception as e:
        return None, None, str(e)

df_hist, df_jorn, cargado = cargar_datos()

if not cargado:
    st.error(f"❌ Error al cargar los CSVs: {df_jorn}")
    st.stop()

# 2. Entrenamiento del modelo
@st.cache_resource
def entrenar_modelo(df):
    try:
        for col in ['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante']:
            if col not in df.columns:
                df[col] = 1.5 if 'goles' in col else 7.0
                
        if 'resultado' not in df.columns:
            if 'result' in df.columns:
                df['resultado'] = df['result'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
            else:
                df['resultado'] = 1 

        features = ['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante']
        X = df[features].fillna(0)
        y = df['resultado'].fillna(0)
        
        modelo = RandomForestClassifier(n_estimators=30, random_state=42)
        modelo.fit(X, y)
        return modelo
    except Exception as e:
        return None

modelo_ml = entrenar_modelo(df_hist)

# 3. Función de estadísticas históricas por equipo
def obtener_estadisticas_equipo(df_historico, nombre_equipo):
    try:
        col_loc = 'home_team' if 'home_team' in df_historico.columns else ('HomeTeam' if 'HomeTeam' in df_historico.columns else None)
        if not col_loc:
            return 1.4, 7.0
            
        partidos_loc = df_historico[df_historico[col_loc].astype(str).str.contains(nombre_equipo, case=False, na=False)]
        goles_favor = 1.4
        forma = 7.0
        
        if not partidos_loc.empty:
            if 'goles_local_previos' in partidos_loc.columns:
                goles_favor = float(partidos_loc['goles_local_previos'].mean())
            if 'forma_local' in partidos_loc.columns:
                forma = float(partidos_loc['forma_local'].mean())
                
        return goles_favor, forma
    except:
        return 1.4, 7.0

# Menú de navegación
menu = st.radio("Sección:", ["📅 Pronosticar por Liga y Jornada", "📊 Ver Base de Datos"], horizontal=True)
st.markdown("---")

if menu == "📅 Pronosticar por Liga y Jornada":
    st.header("🎯 Predicciones de España (E1, E2) y Otras Ligas")
    
    if not df_jorn.empty:
        # Detectamos la columna de liga (la primera columna del archivo de jornada suele contener E1, E2, etc.)
        col_liga = df_jorn.columns[0]
        col_jornada = 'matchday' if 'matchday' in df_jorn.columns else (df_jorn.columns[2] if len(df_jorn.columns) > 2 else None)

        # 🎛️ Selectores limpios en tres columnas: Liga, Jornada y Buscador opcional
        c1, c2, c3 = st.columns(3)
        
        with c1:
            ligas_disponibles = sorted(df_jorn[col_liga].dropna().astype(str).unique())
            liga_elegida = st.selectbox("🏆 Competición (Ej: E1, E2...)", ligas_disponibles)
            df_filtrado_liga = df_jorn[df_jorn[col_liga].astype(str) == liga_elegida]

        with c2:
            if col_jornada and col_jornada in df_filtrado_liga.columns:
                jornadas_disponibles = sorted(df_filtrado_liga[col_jornada].dropna().unique())
                jornada_elegida = st.selectbox("📅 Jornada", jornadas_disponibles)
                partidos_activos = df_filtrado_liga[df_filtrado_liga[col_jornada] == jornada_elegida]
            else:
                partidos_activos = df_filtrado_liga

        with c3:
            busqueda_equipo = st.text_input("🔍 Filtrar equipo (opcional)", "").strip()
            if busqueda_equipo:
                col_loc = 'home_team' if 'home_team' in partidos_activos.columns else 'HomeTeam'
                col_vis = 'away_team' if 'away_team' in partidos_activos.columns else 'AwayTeam'
                partidos_activos = partidos_activos[
                    partidos_activos[col_loc].astype(str).str.contains(busqueda_equipo, case=False, na=False) |
                    partidos_activos[col_vis].astype(str).str.contains(busqueda_equipo, case=False, na=False)
                ]
        
        st.markdown("---")
        st.markdown(f"**Partidos en {liga_elegida}:** {len(partidos_activos)}")
        
        # Listado de partidos pronosticados
        for idx, row in partidos_activos.iterrows():
            local = row.get('home_team', row.get('HomeTeam', 'Local'))
            visitante = row.get('away_team', row.get('AwayTeam', 'Visitante'))
            
            # Consultamos las estadísticas reales en el histórico maestro
            g_loc, f_loc = obtener_estadisticas_equipo(df_hist, str(local))
            g_vis, f_vis = obtener_estadisticas_equipo(df_hist, str(visitante))
            
            datos_partido = pd.DataFrame([[g_loc, g_vis, f_loc, f_vis]], 
                                         columns=['goles_local_previos', 'goles_visitante_previos', 'forma_local', 'forma_visitante'])
            
            if modelo_ml:
                pred = modelo_ml.predict(datos_partido)[0]
                probs = modelo_ml.predict_proba(datos_partido)[0]
            else:
                pred, probs = 1, [0.33, 0.34, 0.33]
            
            mapa_res = {0: "Empate 🤝", 1: f"Victoria de {local} 🏠", 2: f"Victoria de {visitante} ✈️"}
            
            with st.expander(f"⚽ {local} vs {visitante}  ➡️  **{mapa_res.get(pred, 'Pronóstico')}**"):
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric(f"Local ({local})", f"{probs[1]*100:.1f}%" if len(probs) > 1 else "33.3%")
                col_m2.metric("Empate", f"{probs[0]*100:.1f}%" if len(probs) > 0 else "33.3%")
                col_m3.metric(f"Visitante ({visitante})", f"{probs[2]*100:.1f}%" if len(probs) > 2 else "33.3%")
                
                st.caption(f"📊 Estadísticas históricas aplicadas -> Goles Local: {g_loc:.2f} | Forma Local: {f_loc:.1f} | Goles Visitante: {g_vis:.2f} | Forma Visitante: {f_vis:.1f}")
    else:
        st.info("El archivo de jornada está vacío.")

else:
    st.header("📊 Resumen del Histórico Maestro")
    st.info(f"Partidos totales en el histórico: **{len(df_hist):,}**")
    st.dataframe(df_hist.head(20))
