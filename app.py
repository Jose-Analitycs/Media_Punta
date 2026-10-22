import streamlit as st
import pandas as pd

st.set_page_config(page_title="Media_punta - Inspección", page_icon="⚽", layout="wide")
st.title("⚽ Inspeccionando el contenido de los CSVs")

try:
    df_hist = pd.read_csv("historico_maestro.csv")
    df_jorn = pd.read_csv("jornada_maestra.csv")
    
    st.success(f"Archivos leídos correctamente.")
    
    st.subheader("📌 Primeras filas de 'jornada_maestra.csv':")
    st.dataframe(df_jorn.head(3))
    
    st.subheader("📌 Primeras filas de 'historico_maestro.csv':")
    st.dataframe(df_hist.head(3))

except Exception as e:
    st.error(f"❌ Error: {e}")
