# -*- coding: utf-8 -*-
"""
Streamlit app: Industria — Nowcast & Forecast por Ciudad/Ramo (incluye proyección mes a mes 2026)
- Carga automáticamente la Hoja1 del spreadsheet (ID/GID definidos) usando la URL de export CSV.
- Si la hoja no es pública la app mostrará instrucciones (no hace upload).
- Normaliza columnas mínimas (FECHA, VALOR, TIPO_VALOR, CIUDAD, RAMO, COMPANIA, ESTADO, etc).
- Forecast por ciudad o por ramo (configurable). Calcula:
    * Nowcast / cierre estimado del año seleccionado por ciudad/ramo
    * Proyección mes-a-mes para 2026 por ciudad/ramo (y total industria)
    * Comparativa "Solo ESTADO" (mi empresa) vs Industria
- Descarga Excel con resultados.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from typing import Optional, Dict, List

# Time series models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# ---------- Config ----------
DEFAULT_SHEET_ID = "1VljNnZtRPDA3TkTUP6w8AviZCPIfIlqe"
DEFAULT_GID = "293107109"

st.set_page_config(page_title="Industria · Forecast 2026 por Ciudad / Ramo", layout="wide")

# ---------- Utilities ----------
def export_csv_url(sheet_id: str, gid: str) -> str:
    """Genera URL de exportación correcta (SIN espacios)"""
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def parse_number_co(series: pd.Series) -> pd.Series:
    s = series.astype(str).fillna("")
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def ensure_monthly(ts: pd.Series) -> pd.Series:
    ts = ts.asfreq("MS")
    ts = ts.interpolate(method="linear", limit_area="inside")
    return ts

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100

def sanitize_trailing_zeros(ts: pd.Series, ref_year: int) -> pd.Series:
    ts = ensure_monthly(ts.copy())
    year_series = ts[ts.index.year == ref_year]
    if year_series.empty:
        return ts.dropna()
    mask = (year_series[::-1] == 0)
    run, flag = [], True
    for v in mask:
        if flag and bool(v):
            run.append(True)
        else:
            flag = False
            run.append(False)
    trailing_zeros = pd.Series(run[::-1], index=year_series.index)
    ts.loc[trailing_zeros.index[trailing_zeros]] = np.nan
    if ts.last_valid_index() is not None:
        ts = ts.loc[:ts.last_valid_index()]
    return ts.dropna()

def split_series_excluding_partial_current(ts: pd.Series, ref_year: int, today_like: pd.Timestamp):
    ts = ensure_monthly(ts.copy())
    cur_m = pd.Timestamp(year=today_like.year, month=today_like.month, day=1)
    if len(ts) == 0:
        return ts, None, False
    end_of_month = (cur_m + pd.offsets.MonthEnd(0)).day
    if today_like.day < end_of_month:
        ts.loc[cur_m] = np.nan
        return ts.dropna(), cur_m, True
    return ts.dropna(), None, False

def fmt_cop(x):
    try:
        if pd.isna(x):
            return "-"
    except Exception:
        return "-"
    try:
        return "$" + f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return x

# ---------- Forecast helper ----------
def fit_forecast(ts_m: pd.Series, steps: int, eval_months:int=6, conservative_factor: float = 1.0):
    """Forecast historical series ts_m and return hist_df, fc_df, smape_last"""
    if steps < 1:
        steps = 1
    ts = ensure_monthly(ts_m.copy())
    if ts.empty:
        return pd.DataFrame(columns=["FECHA","Mensual","ACUM"]), pd.DataFrame(columns=["FECHA","Forecast_mensual","Forecast_acum","IC_lo","IC_hi"]), np.nan
    y = np.log1p(ts.replace(0, np.nan).dropna())  # log1p on nonzero series
    if y.empty:
        return pd.DataFrame({"FECHA":ts.index, "Mensual":ts.values}), pd.DataFrame(), np.nan
    smapes = []
    start = max(len(y)-eval_months, 12)
    if len(y) >= start+1:
        for t in range(start, len(y)):
            y_tr = y.iloc[:t]
            y_te = y.iloc[t:t+1]
            try:
                m = SARIMAX(y_tr, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
                r = m.fit(disp=False)
                p = r.get_forecast(steps=1).predicted_mean
            except Exception:
                r = ARIMA(y_tr, order=(1,1,1)).fit()
                p = r.get_forecast(steps=1).predicted_mean
            smapes.append(smape(np.expm1(y_te.values), np.expm1(p.values)))
    smape_last = float(np.mean(smapes)) if smapes else np.nan
    def _adj(arr):
        return np.expm1(arr) * conservative_factor
    try:
        m_full = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
        r_full = m_full.fit(disp=False)
        pred = r_full.get_forecast(steps=steps)
        mean = _adj(pred.predicted_mean)
        ci = np.expm1(pred.conf_int(alpha=0.05)) * conservative_factor
    except Exception:
        r_full = ARIMA(y, order=(1,1,1)).fit()
        pred = r_full.get_forecast(steps=steps)
        mean = _adj(pred.predicted_mean)
        ci = np.expm1(pred.conf_int(alpha=0.05)) * conservative_factor
    future_idx = pd.date_range(ts.index.max() + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    hist_acum = ts.cumsum()
    forecast_acum = np.cumsum(mean) + (hist_acum.iloc[-1] if len(hist_acum) > 0 else 0.0)
    fc_df = pd.DataFrame({"FECHA": future_idx, "Forecast_mensual": mean.values.clip(min=0), "Forecast_acum": forecast_acum.values.clip(min=0)})
    # attach IC if ci available
    if hasattr(ci, 'iloc'):
        try:
            fc_df["IC_lo"] = ci.iloc[:,0].values.clip(min=0)
            fc_df["IC_hi"] = ci.iloc[:,1].values.clip(min=0)
        except Exception:
            fc_df["IC_lo"] = np.nan
            fc_df["IC_hi"] = np.nan
    hist_df = pd.DataFrame({"FECHA": ts.index, "Mensual": ts.values, "ACUM": hist_acum.values if len(ts) > 0 else []})
    return hist_df, fc_df, smape_last

def forecast_year_monthly_for_series(series: pd.Series, target_year: int = 2026, conservative_factor: float = 1.0):
    """
    Forecast monthly values for target_year (Jan..Dec) given historical monthly series.
    Returns a Series indexed by month start dates for the target year.
    """
    if series is None or series.empty:
        # return zeros for 12 months
        idx = pd.date_range(start=f"{target_year}-01-01", periods=12, freq="MS")
        return pd.Series([0.0]*12, index=idx)
    last = series.index.max()
    last_year = last.year
    last_month = last.month
    steps = (target_year - last_year) * 12 + (12 - last_month)
    # steps is number of months from month after last to Dec target_year inclusive
    steps = int(max(1, steps))
    hist_df, fc_df, _ = fit_forecast(series, steps=steps, eval_months=6, conservative_factor=conservative_factor)
    if fc_df.empty:
        # fallback: use last 12-month average
        avg = series.tail(12).mean() if len(series) > 0 else 0.0
        idx = pd.date_range(start=f"{target_year}-01-01", periods=12, freq="MS")
        return pd.Series([avg]*12, index=idx)
    # fc_df starts at month after last
    fc_df = fc_df.copy()
    # Filter fc_df rows corresponding to target_year
    fc_df['YEAR'] = fc_df['FECHA'].dt.year
    sel = fc_df[fc_df['YEAR'] == target_year]
    if sel.empty:
        # maybe fc_df covers beyond target_year; produce zeros
        idx = pd.date_range(start=f"{target_year}-01-01", periods=12, freq="MS")
        return pd.Series([0.0]*12, index=idx)
    # ensure months Jan..Dec present (if some missing, fill with 0)
    idx = pd.date_range(start=f"{target_year}-01-01", periods=12, freq="MS")
    out = pd.Series(0.0, index=idx)
    for _, r in sel.iterrows():
        d = pd.Timestamp(r['FECHA']).to_period('M').to_timestamp()
        if d.year == target_year:
            out.loc[d] = float(r['Forecast_mensual'])
    return out

# ---------- Data loading & normalization ----------
def normalize_industria(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas y tipos de datos"""
    df = df.rename(columns={c: c.strip() for c in df.columns})
    
    # Mapeo flexible de columnas
    colmap = {}
    for c in df.columns:
        cn = c.strip().lower()
        if 'homolog' in cn:
            colmap[c] = 'HOMO'
        elif cn in ['año','ano','year']:
            colmap[c] = 'ANIO'
        elif 'compañ' in cn or 'compania' in cn:
            colmap[c] = 'COMPANIA'
        elif 'ciudad' in cn:
            colmap[c] = 'CIUDAD'
        elif 'ram' in cn:
            colmap[c] = 'RAMO'
        elif ('primas' in cn and 'siniest' in cn) or 'primas/siniestros' in cn:
            colmap[c] = 'TIPO_VALOR'
        elif cn in ['primas','siniestros']:
            colmap[c] = 'TIPO_VALOR'
        elif 'fecha' in cn:
            colmap[c] = 'FECHA'
        elif 'valor' in cn or 'valor_mensual' in cn:
            colmap[c] = 'VALOR'
        elif 'depart' in cn:
            colmap[c] = 'DEPARTAMENTO'
        elif 'estado' in cn:
            colmap[c] = 'ESTADO'
    
    df = df.rename(columns=colmap)
    
    # Parse fecha
    if 'FECHA' in df.columns:
        df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
    else:
        if 'ANIO' in df.columns and 'MES' in df.columns:
            try:
                df['FECHA'] = pd.to_datetime(dict(year=df['ANIO'].astype(int), month=df['MES'].astype(int), day=1), errors='coerce')
            except Exception:
                df['FECHA'] = pd.NaT
        elif 'ANIO' in df.columns:
            try:
                df['FECHA'] = pd.to_datetime(df['ANIO'].astype(int).astype(str) + "-01-01", errors='coerce')
            except Exception:
                df['FECHA'] = pd.NaT
        else:
            df['FECHA'] = pd.NaT
    
    df['FECHA'] = df['FECHA'].dt.to_period("M").dt.to_timestamp()
    
    # Valor numérico
    if 'VALOR' in df.columns:
        df['VALOR'] = parse_number_co(df['VALOR'])
    else:
        for alt in ['Valor_Mensual','Valor Mensual','VALOR_MENSUAL','VALOR_MES']:
            if alt in df.columns:
                df['VALOR'] = parse_number_co(df[alt])
                break
        else:
            df['VALOR'] = pd.to_numeric(df.get('VALOR', pd.Series(dtype=float)), errors='coerce')
    
    # Tipo normalized
    if 'TIPO_VALOR' in df.columns:
        df['TIPO_VALOR'] = df['TIPO_VALOR'].astype(str).str.strip().str.lower()
    else:
        df['TIPO_VALOR'] = 'primas'
    
    # text clean
    for c in ['HOMO','COMPANIA','CIUDAD','RAMO','DEPARTAMENTO','ESTADO']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    
    if 'ANIO' not in df.columns:
        df['ANIO'] = df['FECHA'].dt.year
    
    keep = ['ANIO','FECHA','HOMO','COMPANIA','CIUDAD','RAMO','TIPO_VALOR','VALOR','DEPARTAMENTO','ESTADO']
    keep = [c for c in keep if c in df.columns]
    return df[keep].dropna(subset=['FECHA']).copy()

def try_load_industria_direct():
    """
    Intenta cargar datos desde Google Sheets con manejo robusto de errores
    Retorna DataFrame normalizado o vacío si falla
    """
    url = export_csv_url(DEFAULT_SHEET_ID, DEFAULT_GID)
    
    try:
        # Intentar cargar directamente
        df = pd.read_csv(url)
        
        if df.empty:
            raise ValueError("✅ Conexión exitosa pero la hoja está vacía")
        
        df = normalize_industria(df)
        
        if df.empty:
            raise ValueError("✅ Datos cargados pero sin registros válidos después de normalización")
        
        st.sidebar.success("✅ Conectado a Google Sheets")
        return df
        
    except Exception as e:
        st.error(f"❌ Error de conexión: {str(e)}")
        
        # Mostrar instrucciones VISUALES
        st.warning("🔧 **INSTRUCCIONES PARA SOLUCIONAR**")
        
        with st.expander("📖 PASO 1: Hacer el Google Sheets público (clic para ver detalles)", expanded=True):
            st.markdown("""
            1. Abre tu Google Sheets
            2. Haz clic en **"Compartir"** (arriba derecha, botón azul)
            3. En "General access", cambia de "Restricted" a **"Anyone with the link"**
            4. Asegúrate que el rol sea **"Viewer"** (Lector)
            5. Guarda los cambios
            
            **Verifica la conexión con este enlace:**
            """)
            st.code(url, language="text")
        
        with st.expander("📊 PASO 2: Prueba manual del enlace (opcional)", expanded=False):
            st.markdown("""
            Copia y pega este enlace en tu navegador:
            """)
            st.code(url, language="text")
            st.markdown("""
            - Si **descarga un CSV**, la conexión funcionará
            - Si ves **error 404**, verifica el ID y GID
            - Si pide **iniciar sesión**, la hoja NO es pública
            """)
        
        with st.expander("🛠️ PASO 3: Usar datos de ejemplo para pruebas", expanded=False):
            st.info("Mientras configuras la conexión, puedes usar datos de ejemplo para probar la app")
            if st.button("▶️ Cargar Datos de Ejemplo"):
                return generate_sample_data()
        
        # Retornar DataFrame vacío para evitar crash
        return pd.DataFrame()

def generate_sample_data():
    """Genera datos de ejemplo realistas para pruebas"""
    st.warning("⚠️ **Usando datos de ejemplo** - Configura tu Google Sheets para datos reales")
    
    dates = pd.date_range(start='2020-01-01', end='2025-07-31', freq='M')
    companias = ['ESTADO', 'MAPFRE', 'LIBERTY', 'AXA', 'MUNDIAL', 'PREVISORA', 'ALFA', 'ALLIANZ']
    ciudades = ['BOGOTA', 'MEDELLIN', 'CALI', 'BUCARAMANGA', 'BARRANQUILLA', 'CARTAGENA', 'TUNJA', 'BUENAVENTURA']
    ramos = ['VIDRIOS', 'INCENDIO', 'ROBO', 'RESPONSABILIDAD CIVIL', 'VEHICULOS', 'VIDA', 'SALUD']
    homologaciones = ['GENERALES', 'ESPECIALES', 'EXCLUIDOS']
    
    data = []
    for date in dates:
        for compania in companias[:3]:  # Reducido para velocidad
            for ciudad in ciudades[:4]:
                for ramo in ramos[:3]:
                    base_valor = np.random.normal(50000, 15000)
                    data.append({
                        'HOMOLOGACIÓN': np.random.choice(homologaciones),
                        'Año': date.year,
                        'COMPAÑÍA': compania,
                        'CIUDAD': ciudad,
                        'RAMOS': ramo,
                        'Primas/Siniestros': 'Primas',
                        'FECHA': date,
                        'Valor_Mensual': max(0, base_valor),
                        'DEPARTAMENTO': 'VALLE DEL CAUCA' if ciudad == 'BUENAVENTURA' else 'ANTIOQUIA'
                    })
                    # Agregar siniestros (20% de primas en promedio)
                    data.append({
                        'HOMOLOGACIÓN': np.random.choice(homologaciones),
                        'Año': date.year,
                        'COMPAÑÍA': compania,
                        'CIUDAD': ciudad,
                        'RAMOS': ramo,
                        'Primas/Siniestros': 'Siniestros',
                        'FECHA': date,
                        'Valor_Mensual': max(0, base_valor * np.random.normal(0.2, 0.05)),
                        'DEPARTAMENTO': 'VALLE DEL CAUCA' if ciudad == 'BUENAVENTURA' else 'ANTIOQUIA'
                    })
    
    return pd.DataFrame(data)

# ---------- App UI ----------
st.title("Industria · Forecast 2026 — por Ciudad / Ramo")
st.markdown("La app carga automáticamente la Hoja1 indicada y calcula previsiones mes-a-mes para 2026.")

# Carga de datos con manejo robusto de errores
df_ind = try_load_industria_direct()

# Si no hay datos, detener la app
if df_ind.empty:
    st.stop()

# Resto del código para el análisis...
# [Aquí va el resto de tu código original de análisis y forecasting]
# Asegúrate de envolver cada sección en try-except para manejar errores

# Agregar botón de recarga en sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Recargar Datos"):
    st.cache_data.clear()
    st.rerun()

# Footer
st.sidebar.info("""
**Conexión:** Google Sheets  
**Última actualización:** {}  
**Registros:** {:,}
""".format(
    df_ind['FECHA'].max().strftime('%Y-%m-%d') if not df_ind.empty else 'N/A',
    len(df_ind)
))
