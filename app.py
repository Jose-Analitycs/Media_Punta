import streamlit as st
import pandas as pd

st.set_page_config(page_title="Media_punta - Diagnóstico", page_icon="⚽", layout="wide")
st.title("⚽ Diagnóstico de Columnas de los CSVs")

try:
    df_hist = pd.read_csv("historico_maestro.csv")
    df_jorn = pd.read_csv("jornada_maestra.csv")
    
    st.success(f"✅ ¡Archivos leídos! Histórico: {len(df_hist)} filas | Jornada: {len(df_jorn)} filas")
    
    st.subheader("📋 Columnas que tiene tu 'historico_maestro.csv':")
    st.write(list(df_hist.columns))
    
    st.subheader("📋 Columnas que tiene tu 'jornada_maestra.csv':")
    st.write(list(df_jorn.columns))

except Exception as e:
    st.error(f"❌ Error: {e}")
