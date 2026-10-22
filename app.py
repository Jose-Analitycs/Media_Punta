import streamlit as st
import pandas as pd

st.set_page_config(page_title="Media_punta - Texto", layout="wide")
st.title("⚽ Ver datos en texto plano")

try:
    df_hist = pd.read_csv("historico_maestro.csv")
    df_jorn = pd.read_csv("jornada_maestra.csv")
    
    st.write("### 📅 JORNADA MAESTRA (Primeras columnas y filas):")
    st.write(df_jorn.head(2))
    st.write("Columnas de jornada:", list(df_jorn.columns))
    
    st.write("---")
    
    st.write("### 📊 HISTÓRICO MAESTRO (Primeras filas):")
    st.write(df_hist.head(2))
    st.write("Primeras 10 columnas del histórico:", list(df_hist.columns[:10]))

except Exception as e:
    st.error(f"❌ Error: {e}")
