# -*- coding: utf-8 -*-
"""
Streamlit app: Industria — Nowcast & Forecast por Ciudad (Primas y Siniestros)
- Carga Hoja1 del spreadsheet de Industria (ID y GID configurables) o permite cargar CSV manualmente.
- Normaliza columnas mínimas (FECHA, VALOR, TIPO_VALOR, CIUDAD, COMPANIA, ESTADO).
- Predice producción hasta fin de año por ciudad (Primas y Siniestros) usando SARIMAX/ARIMA.
- Compara "mi empresa" vs resto (por ESTADO si existe, o por selección de COMPANIA).
- Muestra métricas, gráficos, tablas y permite descargar resultados.
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
from io import BytesIO, StringIO
from typing import Optional, Dict, List

# Time series models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# ---------- Config ----------
DEFAULT_SHEET_ID = "1VljNnZtRPDA3TkTUP6w8AviZCPIfILqe"
DEFAULT_GID = "293107109"
DEFAULT_SHEET_NAME = "Hoja1"

st.set_page_config(page_title="Industria · Nowcast & Forecast por Ciudad", layout="wide")

# ---------- Utilities ----------
def gsheet_csv(sheet_id: str, sheet_name: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

def csv_by_gid(sheet_id: str, gid: str) -> str:
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

# ---------- Forecasting ----------
def fit_forecast(ts_m: pd.Series, steps: int, eval_months:int=6, conservative_factor: float = 1.0):
    if steps < 1:
        steps = 1
    ts = ensure_monthly(ts_m.copy())
    if ts.empty:
        return pd.DataFrame(columns=["FECHA","Mensual","ACUM"]), pd.DataFrame(columns=["FECHA","Forecast_mensual","Forecast_acum","IC_lo","IC_hi"]), np.nan
    y = np.log1p(ts)
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
    fc_df = pd.DataFrame({"FECHA": future_idx, "Forecast_mensual": mean.values.clip(min=0), "Forecast_acum": forecast_acum.values.clip(min=0), "IC_lo": ci.iloc[:,0].values.clip(min=0), "IC_hi": ci.iloc[:,1].values.clip(min=0)})
    hist_df = pd.DataFrame({"FECHA": ts.index, "Mensual": ts.values, "ACUM": hist_acum.values if len(ts) > 0 else []})
    return hist_df, fc_df, smape_last

# ---------- Data loading & normalization (Industria focused) ----------
@st.cache_data(show_spinner=False)
def load_industria_from_gsheet(sheet_id: str, sheet_name: str = "Hoja1", gid: Optional[str] = None) -> pd.DataFrame:
    """Try to load CSV from Google Sheets; may raise HTTPError if not public."""
    if gid:
        url = csv_by_gid(sheet_id, gid)
    else:
        url = gsheet_csv(sheet_id, sheet_name)
    df = pd.read_csv(url)
    return normalize_industria(df)

def normalize_industria(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # Flexible mapping
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
        # try to build from ANIO/MES if present
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
    # Valor numeric
    if 'VALOR' in df.columns:
        df['VALOR'] = parse_number_co(df['VALOR'])
    else:
        # try alternative column names
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

# ---------- App UI ----------
st.title("Industria · Nowcast & Forecast por Ciudad")
st.markdown("Predicción hasta fin de año de Primas y Siniestros por Ciudad. Comparativa 'mi empresa' vs resto.")

# Left column: source config & load
with st.sidebar:
    st.header("Origen de datos")
    sheet_id = st.text_input("Google Sheet ID", value=DEFAULT_SHEET_ID)
    gid = st.text_input("GID (opcional)", value=DEFAULT_GID)
    sheet_name = st.text_input("Sheet name", value=DEFAULT_SHEET_NAME)
    st.markdown("Si la hoja no es pública, sube un CSV alternativo abajo o haz pública la hoja.")

st_col1, st_col2 = st.columns([1,2])
with st_col1:
    uploaded = st.file_uploader("O subir CSV (override)", type=["csv"], help="Sube CSV exportado de la Hoja1 si la hoja no es pública.")
    load_button = st.button("Cargar datos")

# Data load: try gsheet if no uploaded, else use uploaded
df_ind = pd.DataFrame()
load_error = None
if uploaded is not None:
    try:
        df_ind = pd.read_csv(uploaded)
        df_ind = normalize_industria(df_ind)
        st.success("CSV cargado correctamente (upload).")
    except Exception as e:
        load_error = e
        st.error("Error leyendo CSV subido.")
else:
    if load_button:
        try:
            df_ind = load_industria_from_gsheet(sheet_id.strip() or DEFAULT_SHEET_ID, sheet_name.strip() or DEFAULT_SHEET_NAME, gid=gid.strip() or DEFAULT_GID)
            st.success("Datos cargados desde Google Sheet.")
        except Exception as e:
            load_error = e
            st.error("No fue posible cargar la hoja Industria. Verifique ID/GID/hoja y permisos.")
            st.exception(e)

# If still empty, show guidance
if df_ind.empty:
    st.info("No hay datos cargados. Haz pública la hoja (Compartir → Cualquiera con el enlace → Lector) o sube el CSV.")
    st.stop()

# Top controls: year, tipo, city selection and company selection
control_col1, control_col2, control_col3 = st.columns([2,2,2])
with control_col1:
    years = sorted(df_ind['ANIO'].dropna().unique().astype(int).tolist())
    year_sel = st.selectbox("Año análisis", options=years, index=max(0, len(years)-1))
    tipo_options = sorted(df_ind['TIPO_VALOR'].dropna().unique().astype(str).tolist())
    tipo_sel = st.multiselect("Tipos a incluir", options=tipo_options, default=tipo_options)
with control_col2:
    cities = sorted(df_ind['CIUDAD'].dropna().unique().astype(str).tolist())
    city_sel = st.multiselect("Ciudades (dejar vacío para top N)", options=cities)
    top_n = st.number_input("Top N ciudades (si no seleccionas)", min_value=1, max_value=100, value=10)
with control_col3:
    # company vs rest: detect ESTADO column or allow selecting my company
    if 'ESTADO' in df_ind.columns and df_ind['ESTADO'].notna().any():
        use_estado = st.checkbox("Usar columna ESTADO para mi empresa vs resto", value=True)
    else:
        use_estado = False
    companies = sorted(df_ind['COMPANIA'].dropna().unique().astype(str).tolist())
    my_company = st.selectbox("Mi compañía (para comparar con resto)", options=["(Ninguna)"] + companies, index=0)
    conservative_adj = st.slider("Ajuste conservador forecast (%)", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)

conservative_factor = 1.0 + (conservative_adj / 100.0)

# Helper to aggregate series for a given city and tipo
def agg_series_for(df: pd.DataFrame, tipo: str, city: Optional[str] = None, year_limit: Optional[int] = None) -> pd.Series:
    df2 = df.copy()
    if tipo:
        df2 = df2[df2['TIPO_VALOR'] == tipo]
    if city:
        df2 = df2[df2['CIUDAD'] == city]
    if year_limit:
        df2 = df2[df2['FECHA'].dt.year <= year_limit]
    s = df2.groupby('FECHA')['VALOR'].sum().sort_index()
    s.index = pd.to_datetime(s.index)
    return s

# Determine cities to run
if not city_sel:
    # compute top N by sum in selected year for selected tipos
    df_filtered_for_top = df_ind[(df_ind['FECHA'].dt.year == int(year_sel)) & (df_ind['TIPO_VALOR'].isin(tipo_sel))]
    city_sums = df_filtered_for_top.groupby('CIUDAD')['VALOR'].sum().reset_index().sort_values('VALOR', ascending=False)
    chosen_cities = city_sums.head(top_n)['CIUDAD'].tolist()
else:
    chosen_cities = city_sel

st.markdown(f"### Predicción por ciudad (Año {year_sel}) — tipos: {', '.join(tipo_sel)}")
st.info(f"Se generarán forecasts para {len(chosen_cities)} ciudades (puede tardar si son muchas). Pulsa 'Generar forecasts' cuando estés listo.")

if st.button("Generar forecasts por ciudad"):
    resultados = []
    resumen_rows = []
    meses_restantes = None
    for city in chosen_cities:
        city_row = {"CIUDAD": city}
        for tipo in tipo_sel:
            ser = agg_series_for(df_ind, tipo, city=city, year_limit=year_sel)
            # prepare series sanitized for year_sel
            ser = sanitize_trailing_zeros(ser, int(year_sel))
            ser_train, cur_month_ts, had_partial = split_series_excluding_partial_current(ser, int(year_sel), pd.Timestamp(datetime.now()))
            # calc months remaining until Dec
            if cur_month_ts is not None:
                last_closed_month = cur_month_ts.month - 1
            else:
                last_closed_month = ser_train.index.max().month if len(ser_train)>0 else 0
            meses_falt = int(max(0, 12 - last_closed_month))
            meses_restantes = meses_falt if meses_restantes is None else meses_restantes
            # Forecast
            try:
                hist_df, fc_df, smape_val = fit_forecast(ser_train, steps=max(1, meses_falt), eval_months=6, conservative_factor=conservative_factor)
            except Exception:
                hist_df, fc_df, smape_val = pd.DataFrame(), pd.DataFrame(), np.nan
            # nowcast if partial
            nowcast = None
            if had_partial and not fc_df.empty:
                if fc_df.iloc[0]["FECHA"].month != cur_month_ts.month or fc_df.iloc[0]["FECHA"].year != cur_month_ts.year:
                    fc_df.iloc[0, fc_df.columns.get_loc("FECHA")] = cur_month_ts
                nowcast = float(fc_df.iloc[0]["Forecast_mensual"])
                restante_fc = float(fc_df['Forecast_mensual'].iloc[1:].sum()) if len(fc_df)>1 else 0.0
            else:
                nowcast = None
                restante_fc = float(fc_df['Forecast_mensual'].sum()) if not fc_df.empty else 0.0
            ytd_actual = float(ser[ser.index.year == int(year_sel)].sum()) if not ser.empty else 0.0
            cierre_est = ytd_actual + (nowcast if nowcast is not None else 0.0) + restante_fc
            prev_year_total = float(agg_series_for(df_ind, tipo, city=city, year_limit=year_sel-1).sum())
            growth_abs = cierre_est - prev_year_total
            growth_pct = (cierre_est / prev_year_total - 1.0) * 100.0 if prev_year_total > 0 else np.nan
            # Collect results
            city_row[f"{tipo}_ytd_actual"] = ytd_actual
            city_row[f"{tipo}_nowcast_mes"] = nowcast if nowcast is not None else np.nan
            city_row[f"{tipo}_cierre_est"] = cierre_est
            city_row[f"{tipo}_prev_year_total"] = prev_year_total
            city_row[f"{tipo}_growth_abs"] = growth_abs
            city_row[f"{tipo}_growth_pct"] = growth_pct
        resultados.append(city_row)

    df_result = pd.DataFrame(resultados).fillna(0)
    # Compute industry totals for shares
    total_primas = df_result[[c for c in df_result.columns if c.endswith('_cierre_est') and 'primas' in c]].sum().sum() if any(['primas' in c for c in df_result.columns]) else 0.0
    # present table
    st.markdown("#### Resumen por Ciudad (cierre estimado año)")
    display_df = df_result.copy()
    # format for display
    def fmt_val(x): return fmt_cop(x) if not pd.isna(x) and x!=0 else "-"
    for col in display_df.columns:
        if col.endswith('_cierre_est') or col.endswith('_ytd_actual') or col.endswith('_prev_year_total') or col.endswith('_nowcast_mes') or col.endswith('_growth_abs'):
            display_df[col + "_fmt"] = display_df[col].map(fmt_val)
        if col.endswith('_growth_pct'):
            display_df[col + "_fmt"] = display_df[col].map(lambda v: f"{v:.1f}%" if (not pd.isna(v) and np.isfinite(v)) else "-")
    # show a compact table
    st.dataframe(display_df.head(200), use_container_width=True)

    # Charts: per tipo show top cities by cierre_est
    for tipo in tipo_sel:
        st.markdown(f"##### Top ciudades por cierre estimado — {tipo.capitalize()}")
        cierre_col = f"{tipo}_cierre_est"
        if cierre_col in df_result.columns:
            df_top = df_result.sort_values(cierre_col, ascending=False).head(20)
            fig = go.Figure([go.Bar(x=df_top['CIUDAD'], y=df_top[cierre_col], marker_color='#38bdf8')])
            fig.update_layout(yaxis_title="COP", xaxis_tickangle=-45, margin=dict(b=150))
            st.plotly_chart(fig, use_container_width=True)

    # Comparison: my company vs rest
    st.markdown("### Comparativa: Mi empresa vs Resto")
    # Build boolean mask for my company
    if use_estado and 'ESTADO' in df_ind.columns:
        # try to interpret ESTADO column: prefer explicit matches to 'mi_company' if provided
        empresa_mask = df_ind['ESTADO'].astype(str).str.strip().str.lower().isin(['true','si','yes','1'])  # generic truthy candidates
        # If ESTADO contains company identifier values, allow fallback to COMPANIA selection
        if not empresa_mask.any() and my_company != "(Ninguna)":
            empresa_mask = df_ind['COMPANIA'] == my_company
    else:
        empresa_mask = (df_ind['COMPANIA'] == my_company) if (my_company and my_company != "(Ninguna)") else pd.Series([False]*len(df_ind), index=df_ind.index)

    if empresa_mask.sum() == 0 and my_company != "(Ninguna)":
        st.warning("No se detectaron registros para la compañía seleccionada; se comparará por COMPANIA == selección.")
        empresa_mask = (df_ind['COMPANIA'] == my_company)

    # Compute aggregates monthly for company vs rest for each tipo
    for tipo in tipo_sel:
        df_tipo = df_ind[df_ind['TIPO_VALOR'] == tipo]
        comp_series = df_tipo[empresa_mask].groupby('FECHA')['VALOR'].sum().sort_index()
        rest_series = df_tipo[~empresa_mask].groupby('FECHA')['VALOR'].sum().sort_index()
        # Align and plot with forecast to year end
        # Prepare series until selected year
        comp_ser = comp_series[comp_series.index.year <= int(year_sel)]
        rest_ser = rest_series[rest_series.index.year <= int(year_sel)]
        comp_ser = sanitize_trailing_zeros(comp_ser, int(year_sel))
        rest_ser = sanitize_trailing_zeros(rest_ser, int(year_sel))
        # Forecast remaining months for both
        comp_train, cur_m_comp, had_partial_comp = split_series_excluding_partial_current(comp_ser, int(year_sel), pd.Timestamp(datetime.now()))
        rest_train, cur_m_rest, had_partial_rest = split_series_excluding_partial_current(rest_ser, int(year_sel), pd.Timestamp(datetime.now()))
        # months remaining
        if len(comp_train)>0:
            last_closed_month_comp = cur_m_comp.month-1 if had_partial_comp and cur_m_comp is not None else comp_train.index.max().month
        else:
            last_closed_month_comp = max(1, datetime.now().month) - 1
        meses_falt_comp = int(max(0, 12 - last_closed_month_comp))
        if len(rest_train)>0:
            last_closed_month_rest = cur_m_rest.month-1 if had_partial_rest and cur_m_rest is not None else rest_train.index.max().month
        else:
            last_closed_month_rest = max(1, datetime.now().month) - 1
        meses_falt_rest = int(max(0, 12 - last_closed_month_rest))
        try:
            _, fc_comp, _ = fit_forecast(comp_train, steps=max(1, meses_falt_comp), eval_months=6, conservative_factor=conservative_factor)
        except Exception:
            fc_comp = pd.DataFrame()
        try:
            _, fc_rest, _ = fit_forecast(rest_train, steps=max(1, meses_falt_rest), eval_months=6, conservative_factor=conservative_factor)
        except Exception:
            fc_rest = pd.DataFrame()
        # compute year-end totals
        ytd_comp = float(comp_ser[comp_ser.index.year == int(year_sel)].sum()) if not comp_ser.empty else 0.0
        ytd_rest = float(rest_ser[rest_ser.index.year == int(year_sel)].sum()) if not rest_ser.empty else 0.0
        remain_comp = fc_comp['Forecast_mensual'].iloc[1:].sum() if (not fc_comp.empty and (had_partial_comp and len(fc_comp)>1)) else (fc_comp['Forecast_mensual'].sum() if not fc_comp.empty else 0.0)
        remain_rest = fc_rest['Forecast_mensual'].iloc[1:].sum() if (not fc_rest.empty and (had_partial_rest and len(fc_rest)>1)) else (fc_rest['Forecast_mensual'].sum() if not fc_rest.empty else 0.0)
        cierre_comp = ytd_comp + (fc_comp['Forecast_mensual'].iloc[0] if (had_partial_comp and not fc_comp.empty) else 0.0) + remain_comp
        cierre_rest = ytd_rest + (fc_rest['Forecast_mensual'].iloc[0] if (had_partial_rest and not fc_rest.empty) else 0.0) + remain_rest
        # plot
        fig = go.Figure()
        if not comp_ser.empty:
            fig.add_trace(go.Scatter(x=comp_ser.index, y=comp_ser.values, mode="lines+markers", name=f"{my_company} (Hist)"))
        if not rest_ser.empty:
            fig.add_trace(go.Scatter(x=rest_ser.index, y=rest_ser.values, mode="lines+markers", name="Resto (Hist)"))
        if not fc_comp.empty:
            fig.add_trace(go.Scatter(x=fc_comp['FECHA'], y=fc_comp['Forecast_mensual'], mode="lines", name=f"{my_company} Forecast", line=dict(dash="dash")))
        if not fc_rest.empty:
            fig.add_trace(go.Scatter(x=fc_rest['FECHA'], y=fc_rest['Forecast_mensual'], mode="lines", name="Resto Forecast", line=dict(dash="dash")))
        fig.update_layout(title=f"Comparativa {tipo.capitalize()}: Mi empresa vs Resto", yaxis_title="VALOR", xaxis=dict(type="date"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"- Cierre estimado {tipo.capitalize()} — Mi empresa: {fmt_cop(cierre_comp)} · Resto: {fmt_cop(cierre_rest)}")

    # Additional analyses suggestions (and quick implementations)
    st.markdown("### Análisis adicionales realizados")
    st.markdown("""
    - Top ciudades por cierre estimado (ya mostrado).
    - Evolución YoY por ciudad (se puede mostrar en tabla/heatmap).
    - Ratio Siniestros/Primas por ciudad y su tendencia (útil para identificar problemas).
    - Share de cada ciudad sobre la industria.
    - Detección de outliers mensuales por ciudad (cambios > 50% mes a mes).
    """)

    # Compute S/P ratio per city (latest year)
    st.markdown("#### Ratio Siniestros / Primas por ciudad (estimado cierre año)")
    if 'primas' in tipo_sel and 'siniestros' in tipo_sel:
        ratios = []
        for idx, row in df_result.iterrows():
            city = row['CIUDAD']
            primas_cierre = row.get('primas_cierre_est', 0.0)
            sin_cierre = row.get('siniestros_cierre_est', 0.0)
            ratio = (sin_cierre / primas_cierre) if primas_cierre > 0 else np.nan
            ratios.append({'CIUDAD': city, 'PRIMAS_CIERRE': primas_cierre, 'SIN_CIERRE': sin_cierre, 'RATIO_S_P': ratio})
        ratios_df = pd.DataFrame(ratios).sort_values('RATIO_S_P', ascending=False)
        if not ratios_df.empty:
            st.dataframe(ratios_df.head(200), use_container_width=True)
    else:
        st.info("Selecciona ambos tipos (primas y siniestros) para ver ratios S/P.")

    # Download results
    try:
        with BytesIO() as output:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_result.to_excel(writer, sheet_name="resumen_ciudades", index=False)
                df_ind.to_excel(writer, sheet_name="raw_industria", index=False)
            data_xls = output.getvalue()
        st.download_button("⬇️ Descargar resultados (Excel)", data=data_xls, file_name=f"industria_forecast_ciudades_{year_sel}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.warning("No fue posible preparar descarga Excel.")

st.markdown("---")
st.markdown("Notas:")
st.markdown("- Si la hoja de Google Sheets no es pública, sube el CSV desde la barra lateral o publica la hoja.")
st.markdown("- El forecasting usa SARIMAX con fallback a ARIMA; para series muy cortas se usan promedios implícitos (no modelado robusto).")
st.markdown("- Si quieres, puedo optimizar para correr forecasts en paralelo y añadir opciones de parámetros por ciudad (p. ej. ajustar orden, usar XGBoost).")
