# -*- coding: utf-8 -*-
"""
AseguraView · Primas & Presupuesto
Versión completa con correcciones:
- Fix UnboundLocalError en fit_forecast
- Tarjetas por LINEA_PLUS en lugar de mini-tabla
- Presupuesto anual y previos usan dataset completo para evitar truncados por corte
- Preparar dataset segmentado y wrapper de tabla_presupuesto_2026_desagregado_cached robustos
  (conversión segura de FECHA y coerción numérica para evitar AttributeError .dt)
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO, StringIO
from datetime import date
from typing import Dict, Optional

# XGBoost fallback (opcional)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Page config and styles
st.set_page_config(page_title="AseguraView · Primas & Presupuesto", layout="wide")
st.markdown("""
<style>
:root { --bg:#071428; --fg:#f8fafc; --accent:#38bdf8; --muted:#9fb7cc; --card:rgba(255,255,255,0.03); --up:#16a34a; --down:#ef4444; --near:#f59e0b; }
body,.stApp {background:var(--bg);color:var(--fg);}
.block-container{padding-top:.6rem;}
.card{background:var(--card);border:1px solid rgba(255,255,255,0.04);border-radius:12px;padding:12px;margin-bottom:12px}
.table-wrap{overflow:auto;border:1px solid rgba(255,255,255,0.04);border-radius:12px;background:transparent;padding:6px}
.tbl{width:100%;border-collapse:collapse;font-size:14px;color:var(--fg)}
.tbl thead th{position:sticky;top:0;background:#033b63;color:#ffffff;padding:10px;border-bottom:1px solid rgba(255,255,255,0.06);text-align:left}
.tbl td{padding:8px;border-bottom:1px dashed rgba(255,255,255,0.03);white-space:nowrap;color:var(--fg)}
.bad{color:var(--down);font-weight:700}
.ok{color:var(--up);font-weight:700}
.near{color:var(--near);font-weight:700}
.muted{color:var(--muted)}
.small{font-size:12px;color:var(--muted)}
.vertical-summary{display:flex;gap:12px;flex-wrap:wrap}
.vert-left{flex:0 0 360px}
.vert-right{flex:1;min-height:160px}
.vrow{display:flex;justify-content:space-between;padding:8px 10px;border-bottom:1px dashed rgba(255,255,255,0.03)}
.vtitle{color:var(--muted)}
.vvalue{font-weight:700;color:var(--fg)}
.badge{padding:3px 6px;border-radius:6px}

/* New small card for each LINEA_PLUS */
.lplus-cards-wrap{display:flex;gap:10px;flex-wrap:nowrap;overflow:auto;padding:6px}
.lplus-card{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.04);border-radius:10px;padding:10px;min-width:240px;max-width:260px;flex:0 0 auto}
.lplus-title{font-weight:800;margin-bottom:6px;color:var(--fg);font-size:13px}
.lplus-row{display:flex;justify-content:space-between;padding:6px 2px;border-bottom:1px dashed rgba(255,255,255,0.03);font-size:13px}
.lplus-row .vtitle{color:var(--muted);font-size:12px}
.lplus-row .vvalue{font-weight:700;color:var(--fg);font-size:13px}
</style>
""", unsafe_allow_html=True)

# DATA SOURCE (Google Sheet)
SHEET_ID_DEFAULT = "1ThVwW3IbkL7Dw_Vrs9heT1QMiHDZw1Aj-n0XNbDi9i8"
SHEET_NAME_DATOS_DEFAULT = "Hoja1"
GID_FECHA_CORTE = "1567176042"

def gsheet_csv(sheet_id, sheet_name):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

def csv_by_gid(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

# ----------------- Utilities -----------------
@st.cache_data(show_spinner=False)
def load_cutoff_date(sheet_id: str, gid: str) -> pd.Timestamp:
    url = csv_by_gid(sheet_id, gid)
    try:
        df = pd.read_csv(url, header=None)
        raw = str(df.iloc[0, 0]).strip() if not df.empty else ""
    except Exception:
        ts_fallback = pd.Timestamp.today().normalize()
        return pd.Timestamp(ts_fallback.date())
    ts = pd.to_datetime(raw, dayfirst=True, errors='coerce')
    if pd.isna(ts):
        ts = pd.Timestamp.today().normalize()
    return pd.Timestamp(ts.date())

def parse_number_co(series: pd.Series) -> pd.Series:
    s = series.astype(str)
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
        return "$" + f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return x

def pct_plain(val):
    try:
        if pd.isna(val):
            return "-"
    except Exception:
        return "-"
    try:
        f = float(val)
    except Exception:
        return "-"
    return f"{f:.1f}%"

def badge_pct_html(val):
    try:
        if pd.isna(val):
            return "-"
    except Exception:
        return "-"
    try:
        f = float(val)
    except Exception:
        return "-"
    if f >= 100:
        cls = "ok"
    elif f >= 95:
        cls = "near"
    elif f >= 0:
        cls = "bad"
    else:
        cls = "bad"
    return f'<span class="{cls}">{f:.1f}%</span>'

def badge_growth_cop_html(val):
    try:
        if pd.isna(val):
            return "-"
    except Exception:
        return "-"
    try:
        f = float(val)
    except Exception:
        return "-"
    cls = "ok" if f >= 0 else "bad"
    return f'<span class="{cls}">{fmt_cop(f)}</span>'

def badge_growth_pct_html(val):
    try:
        if pd.isna(val):
            return "-"
    except Exception:
        return "-"
    try:
        f = float(val)
    except Exception:
        return "-"
    cls = "ok" if f >= 0 else "bad"
    return f'<span class="{cls}">{f:.1f}%</span>'

def business_days_left(fecha_corte: date, fecha_fin: date) -> int:
    if fecha_fin < fecha_corte:
        return 0
    rng = pd.date_range(start=fecha_corte, end=fecha_fin, freq="B")
    return len(rng)

# Helper render function (must be defined before first use)
def df_to_html(df_in: pd.DataFrame):
    html = '<div class="table-wrap"><table class="tbl"><thead><tr>'
    for c in df_in.columns:
        html += f'<th>{c}</th>'
    html += '</tr></thead><tbody>'
    for _, r in df_in.iterrows():
        html += '<tr>'
        for c in df_in.columns:
            html += f'<td>{r[c]}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html

# ----------------- Load & Normalize -----------------
@st.cache_data(show_spinner=False)
def load_gsheet(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_csv(gsheet_csv(sheet_id, sheet_name))
    return normalize_columns(df)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        'Año':'ANIO','ANO':'ANIO','YEAR':'ANIO',
        'Mes yyyy':'MES_TXT','MES YYYY':'MES_TXT','Mes':'MES_TXT','MES':'MES_TXT',
        'Codigo y Sucursal':'SUCURSAL','Código y Sucursal':'SUCURSAL',
        'Linea':'LINEA','Línea':'LINEA',
        'Compañía':'COMPANIA','COMPAÑÍA':'COMPANIA','COMPANIA':'COMPANIA',
        'Imp Prima':'IMP_PRIMA','Imp Prima Cuota':'IMP_PRIMA_CUOTA',
        'Linea +':'LINEA_PLUS','Línea +':'LINEA_PLUS','LINEA +':'LINEA_PLUS',
        'Código y Ramo':'CODIGO_RAMO','Codigo y Ramo':'CODIGO_RAMO'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    # FECHA
    if 'MES_TXT' in df.columns:
        df['FECHA'] = pd.to_datetime(df['MES_TXT'], dayfirst=True, errors='coerce')
    else:
        if 'ANIO' in df.columns and 'MES' in df.columns:
            df['FECHA'] = pd.to_datetime(dict(year=df['ANIO'].astype(int), month=df['MES'].astype(int), day=1), errors='coerce')
        else:
            df['FECHA'] = pd.to_datetime(df.get('ANIO', pd.Series()).astype(str)+"-01-01", errors='coerce')
    df['FECHA'] = df['FECHA'].dt.to_period("M").dt.to_timestamp()
    # NUMBERS
    if 'IMP_PRIMA' in df.columns:
        df['IMP_PRIMA'] = parse_number_co(df['IMP_PRIMA'])
    if 'IMP_PRIMA_CUOTA' in df.columns:
        df['IMP_PRIMA_CUOTA'] = parse_number_co(df['IMP_PRIMA_CUOTA'])
    else:
        st.error("Falta la columna 'Imp Prima Cuota' (PRESUPUESTO).")
        st.stop()
    df['PRESUPUESTO'] = df['IMP_PRIMA_CUOTA']
    for c in ['SUCURSAL','LINEA','LINEA_PLUS','COMPANIA','CODIGO_RAMO']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if 'CODIGO_RAMO' in df.columns:
        df['RAMO'] = df['CODIGO_RAMO']
    if 'ANIO' not in df.columns:
        df['ANIO'] = df['FECHA'].dt.year
    keep = ['ANIO','FECHA','SUCURSAL','LINEA','LINEA_PLUS','COMPANIA','CODIGO_RAMO','RAMO','IMP_PRIMA','PRESUPUESTO']
    keep = [x for x in keep if x in df.columns]
    return df[keep].dropna(subset=['FECHA']).copy()

# ----------------- Proportions & Segment summary -----------------
def proporciones_segmento_mes(df_scope: pd.DataFrame, col_segmento: str, mes_ts: pd.Timestamp, ventana_meses: int = 11) -> Dict[str, float]:
    if col_segmento not in df_scope.columns:
        return {}
    ventana_ini = mes_ts - pd.DateOffset(months=ventana_meses)
    tmp = df_scope[ (df_scope["FECHA"] >= ventana_ini) & (df_scope["FECHA"] <= mes_ts) ].groupby(col_segmento)["IMP_PRIMA"].sum()
    segs = sorted(df_scope[col_segmento].dropna().unique())
    if tmp.sum() > 0:
        prop = tmp / tmp.sum()
    else:
        prop = pd.Series([1/len(segs)]*len(segs), index=segs) if segs else pd.Series(dtype=float)
    return prop.to_dict()

def resumen_segmentado_df(df_scope: pd.DataFrame, col_segmento: str, ref_year: int, mes_ref: int, mes_ts: pd.Timestamp,
                          forecast_mes_total: float, habiles_restantes_mes: int, vista_select: str,
                          forecast_annual_total: Optional[float] = None,
                          habiles_restantes_anio: Optional[int] = None,
                          df_scope_full_for_presupuesto: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    df_scope: dataframe ya filtrado (puede estar truncado hasta fecha_corte para análisis)
    df_scope_full_for_presupuesto: si se provee, se usará para calcular PRESUPUESTO ANUAL completo por segmento
      y totals del año previo por segmento (evita que queden truncados por el corte).
    """
    if col_segmento not in df_scope.columns:
        return pd.DataFrame()
    df = df_scope.copy()
    mask_mes_actual = (df["FECHA"].dt.year == ref_year) & (df["FECHA"].dt.month == mes_ref)
    mask_mes_prev = (df["FECHA"].dt.year == ref_year-1) & (df["FECHA"].dt.month == mes_ref)
    mask_ytd = (df["FECHA"].dt.year == ref_year) & (df["FECHA"].dt.month <= mes_ref)
    mask_ytd_prev = (df["FECHA"].dt.year == ref_year-1) & (df["FECHA"].dt.month <= mes_ref)

    # pres_ytd: presupuesto acumulado hasta mes_ref basado en df_scope (mantener comportamiento YTD)
    pres_ytd = df[mask_ytd].groupby(col_segmento)["PRESUPUESTO"].sum()

    # pres_full: presupuesto anual completo. Intentamos tomarlo de df_scope_full_for_presupuesto (si se pasó);
    # si no se pasó, usamos df (posible que esté truncado -> por eso preferimos pasar df_sel_full desde el caller).
    if df_scope_full_for_presupuesto is not None and col_segmento in df_scope_full_for_presupuesto.columns:
        pres_full_series = df_scope_full_for_presupuesto[df_scope_full_for_presupuesto["FECHA"].dt.year == ref_year].groupby(col_segmento)["PRESUPUESTO"].sum()
    else:
        pres_full_series = df[df["FECHA"].dt.year == ref_year].groupby(col_segmento)["PRESUPUESTO"].sum()

    # PROD prev full (año previo completo) -> usado para la vista "Año" en crecimiento
    if df_scope_full_for_presupuesto is not None and col_segmento in df_scope_full_for_presupuesto.columns:
        prod_prev_full_series = df_scope_full_for_presupuesto[df_scope_full_for_presupuesto["FECHA"].dt.year == (ref_year-1)].groupby(col_segmento)["IMP_PRIMA"].sum()
    else:
        prod_prev_full_series = df[df["FECHA"].dt.year == (ref_year-1)].groupby(col_segmento)["IMP_PRIMA"].sum()

    prod_mes = df[mask_mes_actual].groupby(col_segmento)["IMP_PRIMA"].sum()
    pres_mes = df[mask_mes_actual].groupby(col_segmento)["PRESUPUESTO"].sum()
    prod_prev = df[mask_mes_prev].groupby(col_segmento)["IMP_PRIMA"].sum()
    prod_ytd = df[mask_ytd].groupby(col_segmento)["IMP_PRIMA"].sum()
    prod_ytd_prev = df[mask_ytd_prev].groupby(col_segmento)["IMP_PRIMA"].sum()

    segmentos = sorted(df[col_segmento].dropna().unique())
    prop = proporciones_segmento_mes(df_scope, col_segmento, mes_ts, ventana_meses=11)
    rows = []
    for seg in segmentos:
        prod_seg = float(prod_mes.get(seg, 0.0))
        pres_seg = float(pres_mes.get(seg, 0.0))
        prev_seg = float(prod_prev.get(seg, 0.0))
        ytd_seg = float(prod_ytd.get(seg, 0.0))
        ytd_prev_seg = float(prod_ytd_prev.get(seg, 0.0))
        share = float(prop.get(seg, 0.0))
        fc_seg = share * forecast_mes_total

        if vista_select == "Mes":
            falt = pres_seg - prod_seg
            pct_ejec = (prod_seg / pres_seg * 100) if pres_seg > 0 else np.nan
            pct_ejec_fc = (fc_seg / pres_seg * 100) if pres_seg > 0 else np.nan
            req_dia_fc = ((fc_seg - prod_seg) / habiles_restantes_mes) if habiles_restantes_mes > 0 else 0.0
            growth_abs = fc_seg - prev_seg
            growth_pct = (fc_seg / prev_seg - 1.0) * 100.0 if prev_seg > 0 else np.nan
            row = {
                col_segmento: seg,
                "Previo": fmt_cop(prev_seg),
                "Actual": fmt_cop(prod_seg),
                "Presupuesto": fmt_cop(pres_seg),
                "Faltante": f'<span class="{ "ok" if falt<=0 else "bad" }">{fmt_cop(falt)}</span>',
                "% Ejec.": badge_pct_html(pct_ejec),
                "Forecast (mes)": fmt_cop(fc_seg),
                "Forecast ejecución": badge_pct_html(pct_ejec_fc),
                "Crec. Fc (COP)": badge_growth_cop_html(growth_abs),
                "Crec. Fc (%)": badge_growth_pct_html(growth_pct),
                "Req x día Fc": fmt_cop(req_dia_fc)
            }
        elif vista_select == "Año":
            # Usar pres_full_series (presupuesto anual completo por segmento), no pres_ytd.
            pres_label = float(pres_full_series.get(seg, 0.0))
            # PREVIO para año: usar total del año previo completo (no YTD)
            prod_prev_label = float(prod_prev_full_series.get(seg, 0.0))
            # Actual (YTD) sí permanece como ytd_seg (suma hasta mes_ref del año actual)
            prod_act_label = ytd_seg
            if forecast_annual_total is not None:
                fc_ann = share * forecast_annual_total
            else:
                fc_ann = fc_seg * 12.0
            # comparar YTD actual vs presupuesto anual y forecast anual vs presupuesto anual
            pct_ejec = (prod_act_label / pres_label * 100) if pres_label > 0 else np.nan
            pct_ejec_fc = (fc_ann / pres_label * 100) if pres_label > 0 else np.nan
            req_dia_fc = ((fc_ann - prod_act_label) / habiles_restantes_anio) if (habiles_restantes_anio and habiles_restantes_anio > 0) else 0.0
            growth_abs = fc_ann - prod_prev_label
            growth_pct = (fc_ann / prod_prev_label - 1.0) * 100.0 if prod_prev_label > 0 else np.nan
            row = {
                col_segmento: seg,
                "Previo (año prev.)": fmt_cop(prod_prev_label),
                "Actual (YTD)": fmt_cop(prod_act_label),
                "Presupuesto (anual)": fmt_cop(pres_label),
                "Faltante": f'<span class="{ "ok" if (pres_label - prod_act_label)<=0 else "bad" }">{fmt_cop(pres_label - prod_act_label)}</span>',
                "% Ejec.": badge_pct_html(pct_ejec),
                "Forecast (anual est.)": fmt_cop(fc_ann),
                "Forecast ejecución": badge_pct_html(pct_ejec_fc),
                "Crec. Fc (COP)": badge_growth_cop_html(growth_abs),
                "Crec. Fc (%)": badge_growth_pct_html(growth_pct),
                "Req x día Fc (est.)": fmt_cop(req_dia_fc)
            }
        else:
            # Acumulado (YTD) view: prev = YTD año previo (ytd_prev_seg) — mantiene comportamiento
            falt = float(pres_ytd.get(seg, 0.0)) - ytd_seg
            pct_ejec = (ytd_seg / float(pres_ytd.get(seg, 0.0)) * 100) if pres_ytd.get(seg, 0.0) > 0 else np.nan
            delta_fc_month = fc_seg - prod_seg
            ytd_fc_seg = ytd_seg + max(delta_fc_month, 0)
            pct_ejec_fc = (ytd_fc_seg / float(pres_ytd.get(seg, 0.0)) * 100) if pres_ytd.get(seg, 0.0) > 0 else np.nan
            growth_abs = ytd_fc_seg - ytd_prev_seg
            growth_pct = (ytd_fc_seg / ytd_prev_seg - 1.0) * 100.0 if ytd_prev_seg > 0 else np.nan
            req_dia_fc = ((ytd_fc_seg - ytd_seg) / habiles_restantes_mes) if habiles_restantes_mes > 0 else 0.0
            row = {
                col_segmento: seg,
                "Previo (YTD)": fmt_cop(ytd_prev_seg),
                "Actual (YTD)": fmt_cop(ytd_seg),
                "Presupuesto (YTD)": fmt_cop(float(pres_ytd.get(seg, 0.0))),
                "Faltante": f'<span class="{ "ok" if falt<=0 else "bad" }">{fmt_cop(falt)}</span>',
                "% Ejec.": badge_pct_html(pct_ejec),
                "Forecast (YTD est.)": fmt_cop(ytd_fc_seg),
                "Forecast ejecución": badge_pct_html(pct_ejec_fc),
                "Crec. Fc (COP)": badge_growth_cop_html(growth_abs),
                "Crec. Fc (%)": badge_growth_pct_html(growth_pct),
                "Req x día Fc": fmt_cop(req_dia_fc)
            }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    # Ajustar orden de columnas por vista
    if vista_select == "Mes":
        cols_order = [col_segmento, "Previo", "Actual", "Presupuesto", "Faltante", "% Ejec.", "Forecast (mes)", "Forecast ejecución", "Crec. Fc (COP)", "Crec. Fc (%)", "Req x día Fc"]
    elif vista_select == "Año":
        # Mostramos "Previo (año prev.)" para claridad
        cols_order = [col_segmento, "Previo (año prev.)", "Actual (YTD)", "Presupuesto (anual)", "Faltante", "% Ejec.", "Forecast (anual est.)", "Forecast ejecución", "Crec. Fc (COP)", "Crec. Fc (%)", "Req x día Fc (est.)"]
    else:
        cols_order = [col_segmento, "Previo (YTD)", "Actual (YTD)", "Presupuesto (YTD)", "Faltante", "% Ejec.", "Forecast (YTD est.)", "Forecast ejecución", "Crec. Fc (COP)", "Crec. Fc (%)", "Req x día Fc"]
    cols_order = [c for c in cols_order if c in df_out.columns]
    df_out = df_out[cols_order]
    return df_out

# ----------------- Forecast (SARIMAX / ARIMA) -----------------
@st.cache_data(show_spinner=False)
def fit_forecast_cached(series_csv: str, steps: int, eval_months:int=6, conservative_factor: float = 1.0):
    s = pd.read_csv(StringIO(series_csv), index_col=0, parse_dates=True, squeeze=True)
    return fit_forecast(s, steps=steps, eval_months=eval_months, conservative_factor=conservative_factor)

def fit_forecast(ts_m: pd.Series, steps: int, eval_months:int=6, conservative_factor: float = 1.0):
    """Fixed: use parameter ts_m consistently to avoid UnboundLocalError."""
    if steps < 1:
        steps = 1
    # ensure we operate on the series passed in (ts_m)
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

# ----------------- Presupuesto 2026 helpers -----------------
def preparar_dataset_segmentado(df_segment: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que FECHA sea datetime y prepara el dataset agregado por FECHA,
    con columnas YEAR y MONTH. Se toleran entradas donde FECHA venga como texto.
    """
    df = df_segment.copy()

    # Intentar asegurarnos de que exista una columna FECHA datetime
    if 'FECHA' in df.columns:
        df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
    else:
        # Intentos alternativos si no hay FECHA explícita
        if 'MES_TXT' in df.columns:
            df['FECHA'] = pd.to_datetime(df['MES_TXT'], dayfirst=True, errors='coerce')
        elif 'ANIO' in df.columns and 'MES' in df.columns:
            # Convertir ANIO/MES a fecha
            try:
                df['FECHA'] = pd.to_datetime(dict(year=df['ANIO'].astype(int), month=df['MES'].astype(int), day=1), errors='coerce')
            except Exception:
                df['FECHA'] = pd.to_datetime(df.get('ANIO', pd.Series()).astype(str) + "-01-01", errors='coerce')
        else:
            # Último recurso: si existe ANIO, construir fecha 1-ene del año
            if 'ANIO' in df.columns:
                df['FECHA'] = pd.to_datetime(df['ANIO'].astype(str) + "-01-01", errors='coerce')
            else:
                # No tenemos FECHA construible -> crear serie vacía
                df['FECHA'] = pd.NaT

    # Asegurar columnas numéricas
    if 'IMP_PRIMA' in df.columns:
        df['IMP_PRIMA'] = pd.to_numeric(df['IMP_PRIMA'], errors='coerce')
    else:
        df['IMP_PRIMA'] = 0.0

    # Eliminar filas sin fecha válida antes de agrupar
    df = df.dropna(subset=['FECHA'])

    agg = df.groupby('FECHA')['IMP_PRIMA'].sum().sort_index().reset_index()
    # Si quedó vacío, devolver dataframe vacío con las columnas esperadas
    if agg.empty:
        agg = pd.DataFrame(columns=['FECHA','IMP_PRIMA','YEAR','MONTH'])
        return agg

    agg["YEAR"] = agg["FECHA"].dt.year
    agg["MONTH"] = agg["FECHA"].dt.month
    return agg

def forecast_segment_xgb_or_avg(df_segment: pd.DataFrame, target_year: int = None, conservative_factor_local: float = 1.0) -> float:
    if target_year is None:
        target_year = fecha_corte.year + 1
    agg = preparar_dataset_segmentado(df_segment)
    if len(agg) < 3:
        return float(agg["IMP_PRIMA"].sum()) if "IMP_PRIMA" in agg.columns else 0.0
    X_hist = agg[["YEAR","MONTH"]].values
    y_hist = agg["IMP_PRIMA"].values
    fut = pd.DataFrame({"YEAR":[target_year]*12,"MONTH":list(range(1,13))})
    X_fut = fut[["YEAR","MONTH"]].values
    if XGBOOST_AVAILABLE and len(np.unique(y_hist)) > 1:
        try:
            model = XGBRegressor(n_estimators=200, learning_rate=0.07, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
            model.fit(X_hist, y_hist)
            y_pred = model.predict(X_fut)
            y_pred = np.where(y_pred < 0, 0, y_pred)
            return float(np.sum(y_pred) * conservative_factor_local)
        except Exception:
            pass
    last6 = agg.tail(6)["IMP_PRIMA"]
    base_val = last6.mean() if len(last6) > 0 else agg["IMP_PRIMA"].mean()
    return float(base_val * 12.0 * conservative_factor_local)

@st.cache_data(show_spinner=False)
def tabla_presupuesto_2026_desagregado_cached(csv_text: str, ipc_pct: float, conservative_factor_local: float):
    """
    Lectura segura del CSV (producido por df_sel.to_csv) y saneamiento mínimo:
    - parsea FECHA a datetime si existe
    - garantiza que IMP_PRIMA y PRESUPUESTO sean numéricos
    - llama a la función de desagregado con el dataframe limpio
    """
    df_scope = pd.read_csv(StringIO(csv_text))

    # Normalizar FECHA si existe o intentar reconstruirla
    if 'FECHA' in df_scope.columns:
        df_scope['FECHA'] = pd.to_datetime(df_scope['FECHA'], errors='coerce')
    else:
        # Intentar con MES_TXT o ANIO/MES
        if 'MES_TXT' in df_scope.columns:
            df_scope['FECHA'] = pd.to_datetime(df_scope['MES_TXT'], dayfirst=True, errors='coerce')
        elif 'ANIO' in df_scope.columns and 'MES' in df_scope.columns:
            try:
                df_scope['FECHA'] = pd.to_datetime(dict(year=df_scope['ANIO'].astype(int), month=df_scope['MES'].astype(int), day=1), errors='coerce')
            except Exception:
                df_scope['FECHA'] = pd.NaT
        else:
            # si no hay forma de reconstruir, añadimos columna FECHA vacía para evitar fallos posteriores
            df_scope['FECHA'] = pd.NaT

    # Asegurar columnas numéricas importantes
    if 'IMP_PRIMA' in df_scope.columns:
        df_scope['IMP_PRIMA'] = pd.to_numeric(df_scope['IMP_PRIMA'], errors='coerce')
    if 'PRESUPUESTO' in df_scope.columns:
        df_scope['PRESUPUESTO'] = pd.to_numeric(df_scope['PRESUPUESTO'], errors='coerce')
    elif 'IMP_PRIMA_CUOTA' in df_scope.columns:
        # mantener compatibilidad con nombres originales si aparece IMP_PRIMA_CUOTA
        df_scope['PRESUPUESTO'] = pd.to_numeric(df_scope['IMP_PRIMA_CUOTA'], errors='coerce')

    # Quitar filas sin FECHA (no aportan al forecast mensual)
    df_scope = df_scope.dropna(subset=['FECHA'])

    return tabla_presupuesto_2026_desagregado(df_scope, ipc_pct=ipc_pct, conservative_factor_local=conservative_factor_local)

def tabla_presupuesto_2026_desagregado(df_scope: pd.DataFrame, ipc_pct: float = 0.0, conservative_factor_local: float = 1.0) -> pd.DataFrame:
    future_year = fecha_corte.year + 1
    groups = []
    has_suc = 'SUCURSAL' in df_scope.columns
    has_lplus = 'LINEA_PLUS' in df_scope.columns
    has_ramo = 'RAMO' in df_scope.columns
    if has_suc and has_lplus and has_ramo:
        combos = df_scope.dropna(subset=['SUCURSAL','LINEA_PLUS','RAMO'])[['SUCURSAL','LINEA_PLUS','RAMO']].drop_duplicates()
        for _, r in combos.iterrows():
            df_seg = df_scope[(df_scope['SUCURSAL']==r['SUCURSAL']) & (df_scope['LINEA_PLUS']==r['LINEA_PLUS']) & (df_scope['RAMO']==r['RAMO'])]
            val = forecast_segment_xgb_or_avg(df_seg, target_year=future_year, conservative_factor_local=conservative_factor_local)
            groups.append({'SUCURSAL': r['SUCURSAL'],'LINEA_PLUS': r['LINEA_PLUS'],'RAMO': r['RAMO'],'Presupuesto_Ideal_{}'.format(future_year): round(val,0),'Modelo': "XGBoost mensual" if XGBOOST_AVAILABLE else "Promedio últimos 6m x12"})
    else:
        group_cols = [c for c in ['SUCURSAL','LINEA_PLUS','RAMO'] if c in df_scope.columns]
        agg = df_scope.groupby(group_cols)['IMP_PRIMA'].sum().reset_index()
        for _, r in agg.iterrows():
            mask = True
            for c in group_cols:
                mask = mask & (df_scope[c] == r[c])
            val = forecast_segment_xgb_or_avg(df_scope[mask], target_year=future_year, conservative_factor_local=conservative_factor_local)
            row = {c: r[c] for c in group_cols}
            row.update({'Presupuesto_Ideal_{}'.format(future_year): round(val,0),'Modelo': "XGBoost mensual" if XGBOOST_AVAILABLE else "Promedio últimos 6m x12"})
            groups.append(row)
    out = pd.DataFrame(groups)
    ipc_factor = 1.0 + (ipc_pct / 100.0)
    if not out.empty:
        out['Presupuesto_Ideal_{}_ajustado'.format(future_year)] = (out['Presupuesto_Ideal_{}'.format(future_year)] * ipc_factor).round(0)
    return out

# ----------------- Load initial data -----------------
df = load_gsheet(SHEET_ID_DEFAULT, SHEET_NAME_DATOS_DEFAULT)
fecha_corte = load_cutoff_date(SHEET_ID_DEFAULT, GID_FECHA_CORTE)

# ----------------- Sidebar filters (default ajuste 0.0) -----------------
st.sidebar.header("Filtros")
suc_opts = ["TODAS"] + sorted(df['SUCURSAL'].dropna().unique()) if 'SUCURSAL' in df.columns else ["TODAS"]
lplus_opts= ["TODAS"] + sorted(df['LINEA_PLUS'].dropna().unique()) if 'LINEA_PLUS' in df.columns else ["TODAS"]
comp_opts = ["TODAS"] + sorted(df['COMPANIA'].dropna().unique()) if 'COMPANIA' in df.columns else ["TODAS"]

suc = st.sidebar.selectbox("Sucursal", suc_opts)
lplus= st.sidebar.selectbox("Línea +", lplus_opts)
comp = st.sidebar.selectbox("Compañía", comp_opts)

anio_analisis = st.sidebar.number_input("Año de análisis", min_value=2018, max_value=2100, value=fecha_corte.year, step=1)

st.sidebar.markdown("#### Ajuste conservador (%)")
ajuste_pct = st.sidebar.slider("Ajuste forecast", min_value=-20.0, max_value=10.0, step=0.5, value=0.0, help="Porcentaje aplicado a la proyección.")
nota_ajuste = st.sidebar.text_input("Nota del ajuste (opcional)", value="")
conservative_factor = 1.0 + (ajuste_pct / 100.0)

# ----------------- Header -----------------
st.markdown(f"""
<div style="display:flex;align-items:center;gap:18px;margin-bottom:6px">
  <div style="font-size:26px;font-weight:800;color:#f3f4f6">AseguraView · Primas & Presupuesto</div>
  <div style="opacity:.85;color:var(--muted);">Corte: {fecha_corte.strftime('%d/%m/%Y')}</div>
</div>
""", unsafe_allow_html=True)
st.caption("Nowcast del mes, cierre estimado del año, ejecución vs presupuesto y propuesta 2026.")

# ----------------- Apply selection filters -----------------
df_sel_full = df.copy()
if suc != "TODAS" and 'SUCURSAL' in df_sel_full.columns:
    df_sel_full = df_sel_full[df_sel_full['SUCURSAL'] == suc]
if lplus != "TODAS" and 'LINEA_PLUS' in df_sel_full.columns:
    df_sel_full = df_sel_full[df_sel_full['LINEA_PLUS'] == lplus]
if comp != "TODAS" and 'COMPANIA' in df_sel_full.columns:
    df_sel_full = df_sel_full[df_sel_full['COMPANIA'] == comp]

# df_sel: dataset que se usa para análisis y que *puede* estar truncado (para excluir meses posteriores al corte)
df_sel = df_sel_full.copy()
df_sel = df_sel[(df_sel['FECHA'].dt.year <= anio_analisis)]
df_sel = df_sel[ ~((df_sel['FECHA'].dt.year == anio_analisis) & (df_sel['FECHA'] > pd.Timestamp(anio_analisis, fecha_corte.month, 1)) ) ]

serie_prima_all = df_sel.groupby('FECHA')['IMP_PRIMA'].sum().sort_index()
serie_presu_all = df_sel.groupby('FECHA')['PRESUPUESTO'].sum().sort_index()
serie_prima_all_full = df_sel_full.groupby('FECHA')['IMP_PRIMA'].sum().sort_index()
serie_presu_all_full = df_sel_full.groupby('FECHA')['PRESUPUESTO'].sum().sort_index()

if serie_prima_all.empty and serie_prima_all_full.empty:
    st.warning("No hay datos de IMP_PRIMA con los filtros seleccionados.")
    st.stop()

# ----------------- TABS -----------------
tabs = st.tabs([ "🏠 Presentación", "📈 Primas", "📊 Presupuesto 2026" ])

# -------- PRESENTACIÓN --------
with tabs[0]:
    st.markdown("""
    <div class="card">
      <h3 style="margin:0 0 8px 0">Presentación</h3>
      <div style="color:#cfe7fb;line-height:1.5">
        Bienvenido a AseguraView. En este tablero verás:
        <ul>
          <li>Nowcast del mes actual (si el mes está parcial usamos nowcast)</li>
          <li>Proyección de cierre del año (forecast SARIMAX/ARIMA)</li>
          <li>Ejecución vs presupuesto (mensual y acumulado)</li>
          <li>Propuesta técnica de Presupuesto 2026 por Sucursal + Línea + Ramo (on-demand)</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

# -------- PRIMAS: forecast & cierre (main analytics) --------
with tabs[1]:
    ref_year = int(anio_analisis)
    base_series = sanitize_trailing_zeros(serie_prima_all.copy(), ref_year)
    serie_train, cur_month_ts, had_partial = split_series_excluding_partial_current(base_series, ref_year, today_like=fecha_corte)
    if len(serie_train) > 0:
        if had_partial and cur_month_ts is not None:
            last_closed_month = cur_month_ts.month - 1
        else:
            last_closed_month = serie_train.index.max().month
    else:
        last_closed_month = max(1, fecha_corte.month) - 1
    meses_faltantes = int(max(0, 12 - last_closed_month))

    # forecast
    try:
        series_csv = serie_train.to_csv()
        hist_df, fc_df, smape6 = fit_forecast_cached(series_csv, steps=max(1, meses_faltantes), eval_months=6, conservative_factor=conservative_factor)
    except Exception:
        hist_df, fc_df, smape6 = fit_forecast(serie_train, steps=max(1, meses_faltantes), eval_months=6, conservative_factor=conservative_factor)

    forecast_map = {}
    if not fc_df.empty:
        for i in range(len(fc_df)):
            forecast_map[pd.Timestamp(fc_df.iloc[i]["FECHA"])] = float(fc_df.iloc[i]["Forecast_mensual"])

    nowcast_actual = None
    if had_partial and not fc_df.empty and cur_month_ts is not None:
        if fc_df.iloc[0]["FECHA"] != cur_month_ts:
            fc_df.iloc[0, fc_df.columns.get_loc("FECHA")] = cur_month_ts
        nowcast_actual = float(fc_df.iloc[0]["Forecast_mensual"])

    # Aggregates
    prod_total_real_ref_year = float( serie_prima_all_full[serie_prima_all_full.index.year == ref_year].sum() )
    ejec_real_cerrado = float( serie_prima_all[ (serie_prima_all.index.year == ref_year) & (serie_prima_all.index <= pd.Timestamp(ref_year, last_closed_month, 1)) ].sum() ) if last_closed_month > 0 else 0.0
    ytd_total = ejec_real_cerrado + (nowcast_actual if nowcast_actual is not None else 0.0)
    resto = 0.0
    if not fc_df.empty:
        if nowcast_actual is not None:
            resto = fc_df['Forecast_mensual'].iloc[1:].sum()
        else:
            resto = fc_df['Forecast_mensual'].sum()
    cierre_est = float(ytd_total + resto)
    pres_anual_ref = float( serie_presu_all_full[serie_presu_all_full.index.year == ref_year].sum() )
    cierre_prev_year = float( serie_prima_all_full[serie_prima_all_full.index.year == (ref_year-1)].sum() )

    mes_ref = (cur_month_ts.month if (had_partial and cur_month_ts is not None) else last_closed_month)
    mes_ref = max(1, mes_ref)
    mes_ts = pd.Timestamp(ref_year, mes_ref, 1)
    prod_mes_corte = float(serie_prima_all.get(mes_ts, 0.0))
    proy_mes = float(forecast_map.get(mes_ts, prod_mes_corte))
    pres_mes = float( serie_presu_all_full[ (serie_presu_all_full.index.year == ref_year) & (serie_presu_all_full.index.month == mes_ref) ].sum() )
    prod_mes_prev = float( serie_prima_all_full[ (serie_prima_all_full.index.year == ref_year-1) & (serie_prima_all_full.index.month == mes_ref) ].sum() )

    month_end = (mes_ts + pd.offsets.MonthEnd(0)).to_pydatetime().date()
    year_end = pd.Timestamp(ref_year, 12, 31).to_pydatetime().date()
    habiles_restantes_mes = business_days_left(fecha_corte.date(), month_end)
    habiles_restantes_anio = business_days_left(fecha_corte.date(), year_end)

    # Vista radio (affects downstream tables)
    vista = st.radio("Vista del resumen", ["Mes", "Año", "Acumulado Mes"], horizontal=True)

    # Consolidated vertical summary
    if vista == "Mes":
        falt_mes_pres = pres_mes - prod_mes_corte
        pct_ejec_mes = (prod_mes_corte / pres_mes * 100) if pres_mes > 0 else np.nan
        pct_ejec_fc_mes = (proy_mes / pres_mes * 100) if pres_mes > 0 else np.nan
        req_dia_fc = ((proy_mes - prod_mes_corte) / habiles_restantes_mes) if habiles_restantes_mes > 0 else 0.0
        req_dia_pres = ((pres_mes - prod_mes_corte) / habiles_restantes_mes) if habiles_restantes_mes > 0 else 0.0
        growth_abs_tot = proy_mes - prod_mes_prev
        growth_pct_tot = (proy_mes / prod_mes_prev - 1.0) * 100.0 if prod_mes_prev > 0 else np.nan
        rows_metrics = [
            ("Periodo", f"{mes_ref:02d}/{ref_year}"),
            ("Prod. mes año previo", fmt_cop(prod_mes_prev)),
            ("Prod. mes (al corte)", fmt_cop(prod_mes_corte)),
            ("Presupuesto mes", fmt_cop(pres_mes)),
            ("Faltante", f'<span class="{ "ok" if falt_mes_pres<=0 else "bad" }">{fmt_cop(falt_mes_pres)}</span>'),
            ("% Ejec.", badge_pct_html(pct_ejec_mes)),
            ("Forecast mes", fmt_cop(proy_mes)),
            ("Forecast ejecución", badge_pct_html(pct_ejec_fc_mes)),
            ("Crec. Forecast (COP)", badge_growth_cop_html(growth_abs_tot)),
            ("Crec. Forecast (%)", badge_growth_pct_html(growth_pct_tot)),
            ("Req x día Fc", fmt_cop(max(req_dia_fc,0))),
            ("Req x día Pres", fmt_cop(max(req_dia_pres,0))),
        ]
    elif vista == "Año":
        falt_anual_pres = pres_anual_ref - prod_total_real_ref_year
        pct_ejec_total = (prod_total_real_ref_year / pres_anual_ref * 100) if pres_anual_ref > 0 else np.nan
        pct_ejec_fc_anual = (cierre_est / pres_anual_ref * 100) if pres_anual_ref > 0 else np.nan
        req_dia_fc_anual = ((cierre_est - ytd_total) / habiles_restantes_anio) if habiles_restantes_anio > 0 else 0.0
        req_dia_pres_anual = ((pres_anual_ref - ytd_total) / habiles_restantes_anio) if habiles_restantes_anio > 0 else 0.0
        growth_abs_tot = cierre_est - cierre_prev_year
        growth_pct_tot = (cierre_est / cierre_prev_year - 1.0) * 100.0 if cierre_prev_year > 0 else np.nan
        rows_metrics = [
            ("Periodo", f"Año {ref_year}"),
            ("Prod. año prev (cierre)", fmt_cop(cierre_prev_year)),
            ("Prod. año actual (real)", fmt_cop(prod_total_real_ref_year)),
            ("Presupuesto anual", fmt_cop(pres_anual_ref)),
            ("Faltante anual", f'<span class="{ "ok" if falt_anual_pres<=0 else "bad" }">{fmt_cop(falt_anual_pres)}</span>'),
            ("% Ejec. (real)", badge_pct_html(pct_ejec_total)),
            ("Forecast anual (est.)", fmt_cop(cierre_est)),
            ("Forecast ejecución", badge_pct_html(pct_ejec_fc_anual)),
            ("Crec. Forecast (COP)", badge_growth_cop_html(growth_abs_tot)),
            ("Crec. Forecast (%)", badge_growth_pct_html(growth_pct_tot)),
            ("Req x día Fc anual", fmt_cop(max(req_dia_fc_anual,0))),
            ("Req x día Pres anual", fmt_cop(max(req_dia_pres_anual,0))),
        ]
    else:
        pres_ytd = float( serie_presu_all_full[ (serie_presu_all_full.index.year == ref_year) & (serie_presu_all_full.index.month <= mes_ref) ].sum() )
        real_ytd_al_corte = ejec_real_cerrado + prod_mes_corte
        ytd_nowcast_mes = ejec_real_cerrado + proy_mes
        falt_ytd = pres_ytd - real_ytd_al_corte
        pct_ejec_ytd = (real_ytd_al_corte / pres_ytd * 100) if pres_ytd > 0 else np.nan
        pct_ejec_fc_ytd = (ytd_nowcast_mes / pres_ytd * 100) if pres_ytd > 0 else np.nan
        req_dia_fc_ytd = ((ytd_nowcast_mes - real_ytd_al_corte) / habiles_restantes_mes) if habiles_restantes_mes > 0 else 0.0
        req_dia_pres_ytd = ((pres_ytd - real_ytd_al_corte) / habiles_restantes_mes) if habiles_restantes_mes > 0 else 0.0
        prod_prev_ytd = float( serie_prima_all_full[ (serie_prima_all_full.index.year == ref_year-1) & (serie_prima_all_full.index.month <= mes_ref) ].sum() )
        growth_abs_tot = ytd_nowcast_mes - prod_prev_ytd
        growth_pct_tot = (ytd_nowcast_mes / prod_prev_ytd - 1.0) * 100.0 if prod_prev_ytd > 0 else np.nan
        rows_metrics = [
            ("Periodo", f"YTD hasta {mes_ref:02d}/{ref_year}"),
            ("Prod. YTD prev", fmt_cop(prod_prev_ytd)),
            ("Prod. YTD actual", fmt_cop(real_ytd_al_corte)),
            ("Presupuesto YTD", fmt_cop(pres_ytd)),
            ("Faltante YTD", f'<span class="{ "ok" if falt_ytd<=0 else "bad" }">{fmt_cop(falt_ytd)}</span>'),
            ("% Ejec. YTD", badge_pct_html(pct_ejec_ytd)),
            ("Forecast YTD", fmt_cop(ytd_nowcast_mes)),
            ("Forecast ejecución", badge_pct_html(pct_ejec_fc_ytd)),
            ("Crec. Fc (COP)", badge_growth_cop_html(growth_abs_tot)),
            ("Crec. Fc (%)", badge_growth_pct_html(growth_pct_tot)),
            ("Req x día Fc YTD", fmt_cop(max(req_dia_fc_ytd,0))),
            ("Req x día Pres YTD", fmt_cop(max(req_dia_pres_ytd,0))),
        ]

    # --- Prepare resumen por LINEA_PLUS usando df_sel (truncado) para shares / historicos
    # pero pasar df_sel_full como df_scope_full_for_presupuesto para obtener PRESUPUESTO ANUAL completo y previos año anterior completos por línea.
    try:
        df_lplus_res = resumen_segmentado_df(df_sel, "LINEA_PLUS", ref_year, mes_ref, mes_ts, proy_mes, habiles_restantes_mes, vista,
                                             forecast_annual_total=cierre_est, habiles_restantes_anio=habiles_restantes_anio,
                                             df_scope_full_for_presupuesto=df_sel_full)
    except Exception:
        df_lplus_res = pd.DataFrame()

    # render vertical summary with right-card containing a card per LINEA_PLUS (instead of a mini table)
    html_v = '<div class="table-wrap"><div class="vertical-summary"><div class="vert-left card">'
    for t, v in rows_metrics:
        html_v += f'<div class="vrow"><div class="vtitle">{t}</div><div class="vvalue">{v}</div></div>'
    html_v += '</div><div class="vert-right card">'
    nota_text = f' ({nota_ajuste})' if nota_ajuste and nota_ajuste.strip() else ''
    html_v += f'<div style="font-size:13px;color:var(--muted);margin-bottom:8px">Ajuste conservador: {ajuste_pct:.1f}%{nota_text}</div>'

    # New behavior: for each LINEA_PLUS show a small card with the same summary metrics (formatted) for that LINEA_PLUS.
    if not df_lplus_res.empty:
        html_v += '<div style="font-weight:700;margin-bottom:6px;color:var(--muted)">Resumen por Línea+</div>'
        # horizontal scrollable list of small cards
        html_v += '<div class="lplus-cards-wrap">'
        # decide how many to show inline (rest will be indicated)
        max_cards_inline = 12
        seg_col = df_lplus_res.columns[0] if len(df_lplus_res.columns) > 0 else "LINEA_PLUS"
        segmentos = df_lplus_res[seg_col].tolist()
        for idx, seg in enumerate(segmentos):
            if idx >= max_cards_inline:
                break
            row = df_lplus_res[df_lplus_res[seg_col] == seg].iloc[0]
            html_v += '<div class="lplus-card">'
            html_v += f'<div class="lplus-title">{seg}</div>'
            for col in df_lplus_res.columns:
                if col == seg_col:
                    continue
                val = row[col] if (col in row and pd.notna(row[col])) else "-"
                html_v += f'<div class="lplus-row"><div class="vtitle">{col}</div><div class="vvalue">{val}</div></div>'
            html_v += '</div>'  # end lplus-card
        html_v += '</div>'  # end lplus-cards-wrap
        if len(segmentos) > max_cards_inline:
            html_v += f'<div class="small" style="margin-top:6px;color:var(--muted)">Mostrando {max_cards_inline} de {len(segmentos)} líneas. Ver detalle completo abajo.</div>'
    else:
        html_v += '<div class="small" style="color:var(--muted)">No hay detalle de Línea + para los filtros seleccionados y la vista actual.</div>'

    html_v += '</div></div></div>'
    st.markdown(html_v, unsafe_allow_html=True)

    st.markdown(f"SMAPE validación (6 meses): {smape6:.2f}% · Corte de datos: {fecha_corte.strftime('%d/%m/%Y')}", unsafe_allow_html=True)

    # Monthly extended table -> show ONLY missing months (forecast months)
    st.markdown("##### Proyección - meses faltantes")
    if not fc_df.empty:
        rows_monthly = []
        for i in range(len(fc_df)):
            mes = pd.Timestamp(fc_df.iloc[i]["FECHA"])
            prod = float(serie_prima_all.get(mes, 0.0))
            pres = float(serie_presu_all_full.get(mes, 0.0))
            fc_m = float(fc_df.iloc[i]["Forecast_mensual"])
            falt = pres - prod
            pct_ejec = (prod / pres * 100) if pres > 0 else np.nan
            pct_ejec_fc = (fc_m / pres * 100) if pres > 0 else np.nan
            prev_same = pd.Timestamp(mes.year-1, mes.month, 1)
            prod_prev_same = float(serie_prima_all_full.get(prev_same, 0.0))
            growth_abs_m = fc_m - prod_prev_same
            growth_pct_m = (fc_m / prod_prev_same - 1.0) * 100.0 if prod_prev_same > 0 else np.nan
            rows_monthly.append({
                "Mes": mes.strftime("%b-%Y"),
                "Producción": fmt_cop(prod),
                "Presupuesto": fmt_cop(pres),
                "Faltante": f'<span class="{ "ok" if falt<=0 else "bad" }">{fmt_cop(falt)}</span>',
                "% Ejec.": pct_plain(pct_ejec),
                "Forecast (ajustado)": fmt_cop(fc_m),
                "Forecast ejecución": pct_plain(pct_ejec_fc),
                "Crec. Fc (COP)": badge_growth_cop_html(growth_abs_m),
                "Crec. Fc (%)": badge_growth_pct_html(growth_pct_m),
                "Dif. Fc − Pres": f'<span class="{ "ok" if (fc_m - pres) >= 0 else "bad" }">{fmt_cop(fc_m - pres)}</span>'
            })
        tabla_mensual_ext = pd.DataFrame(rows_monthly)
        st.markdown(df_to_html(tabla_mensual_ext), unsafe_allow_html=True)
    else:
        st.info("No hay meses faltantes / forecast para mostrar.")

    # Desglose - Ramo (inside expander)
    st.markdown("### Desglose - Ramo")
    with st.expander("Ver / Ocultar detalle por Ramo"):
        try:
            df_ramo_res = resumen_segmentado_df(df_sel, "RAMO", ref_year, mes_ref, mes_ts, proy_mes, habiles_restantes_mes, vista,
                                                forecast_annual_total=cierre_est, habiles_restantes_anio=habiles_restantes_anio,
                                                df_scope_full_for_presupuesto=df_sel_full)
            if not df_ramo_res.empty:
                st.markdown(df_to_html(df_ramo_res), unsafe_allow_html=True)
            else:
                st.info("No hay detalle por Ramo para los filtros seleccionados y la vista actual.")
        except Exception as e:
            st.error("Error generando desglose por Ramo.")
            st.exception(e)

    # Graph per LINEA_PLUS (use df_sel so filters apply)
    st.markdown("##### Primas mensuales por Línea+")
    try:
        fig = go.Figure()
        if "LINEA_PLUS" in df_sel.columns and not df_sel["LINEA_PLUS"].dropna().empty:
            lineas_plus = sorted(df_sel["LINEA_PLUS"].dropna().unique())
            for lp in lineas_plus:
                s = df_sel[df_sel["LINEA_PLUS"] == lp].groupby("FECHA")["IMP_PRIMA"].sum().sort_index()
                if not s.empty:
                    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=f"{lp} (Hist)"))
            for lp in lineas_plus:
                fc_vals = []
                fc_idx = []
                for i in range(len(fc_df)):
                    mes_ts_local = pd.Timestamp(fc_df.iloc[i]["FECHA"])
                    prop = proporciones_segmento_mes(df_sel, "LINEA_PLUS", mes_ts_local, ventana_meses=11)
                    share = prop.get(lp, 0.0)
                    fc_vals.append(float(fc_df.iloc[i]["Forecast_mensual"]) * share)
                    fc_idx.append(mes_ts_local)
                if any(fc_vals):
                    fig.add_trace(go.Scatter(x=fc_idx, y=fc_vals, mode="lines+markers", name=f"{lp} (Forecast)", line=dict(dash="dash")))
            if not fc_df.empty:
                fig.add_trace(go.Scatter(x=fc_df["FECHA"], y=fc_df["IC_hi"], mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=fc_df["FECHA"], y=fc_df["IC_lo"], mode="lines", fill='tonexty', line=dict(width=0), showlegend=False))
        else:
            st.info("No hay columna 'LINEA_PLUS' para generar gráfico por Línea+.")
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="COP", xaxis=dict(rangeslider=dict(visible=True), type="date"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("Error generando gráfico por Línea+.")
        st.exception(e)

    # Download hist & forecast
    try:
        with BytesIO() as output:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                hist_df.to_excel(writer, sheet_name="hist_mensual", index=False)
                fc_df.to_excel(writer, sheet_name="forecast", index=False)
            data_xls = output.getvalue()
        st.download_button("⬇️ Descargar Excel (histórico y forecast)", data=data_xls, file_name=f"forecast_{ref_year}_con_nowcast.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.warning("No fue posible preparar el Excel de descarga.")

# -------- PRESUPUESTO 2026 (on-demand) --------
with tabs[2]:
    st.subheader("Presupuesto ideal 2026")
    st.markdown("Propuesta técnica por SUCURSAL + LINEA + RAMO. Ajustador IPC/Incrementos de ley disponible.")
    ipc_pct = st.number_input("Ajustador IPC / Incrementos (%)", min_value=-50.0, max_value=200.0, value=0.0, step=0.1)
    st.markdown("Pulsa 'Generar propuesta 2026' para calcular la desagregación (on-demand).")
    if st.button("Generar propuesta 2026"):
        with st.spinner("Generando propuesta 2026..."):
            try:
                csv_buf = df_sel.to_csv(index=False)
                tabla_2026_desag = tabla_presupuesto_2026_desagregado_cached(csv_text=csv_buf, ipc_pct=ipc_pct, conservative_factor_local=conservative_factor)
                if not tabla_2026_desag.empty:
                    show_desag = tabla_2026_desag.copy()
                    future_year = fecha_corte.year + 1
                    if f"Presupuesto_Ideal_{future_year}" in show_desag.columns:
                        show_desag[f"Presupuesto_Ideal_{future_year}"] = show_desag[f"Presupuesto_Ideal_{future_year}"].map(lambda x: fmt_cop(x) if pd.notna(x) else x)
                    col_adj = f"Presupuesto_Ideal_{future_year}_ajustado"
                    if col_adj in show_desag.columns:
                        show_desag[col_adj] = show_desag[col_adj].map(lambda x: fmt_cop(x) if pd.notna(x) else x)
                    st.markdown("##### Propuesta por Sucursal + Línea + Ramo")
                    st.dataframe(show_desag, use_container_width=True, hide_index=True)
                    # aggregates
                    if 'LINEA_PLUS' in tabla_2026_desag.columns:
                        agg_linea = tabla_2026_desag.groupby('LINEA_PLUS')[f'Presupuesto_Ideal_{future_year}'].sum().reset_index()
                        agg_linea[f'Presupuesto_Ideal_{future_year}'] = agg_linea[f'Presupuesto_Ideal_{future_year}'].map(fmt_cop)
                        st.markdown("##### Agregado por Línea +")
                        st.dataframe(agg_linea, use_container_width=True, hide_index=True)
                    if 'RAMO' in tabla_2026_desag.columns:
                        agg_ramo = tabla_2026_desag.groupby('RAMO')[f'Presupuesto_Ideal_{future_year}'].sum().reset_index()
                        agg_ramo[f'Presupuesto_Ideal_{future_year}'] = agg_ramo[f'Presupuesto_Ideal_{future_year}'].map(fmt_cop)
                        st.markdown("##### Agregado por Ramo")
                        st.dataframe(agg_ramo, use_container_width=True, hide_index=True)
                    # download
                    with BytesIO() as output:
                        with pd.ExcelWriter(output, engine="openpyxl") as writer:
                            show_desag.to_excel(writer, sheet_name="Presupuesto_Suc_Linea_Ramo", index=False)
                            if 'agg_linea' in locals() and not agg_linea.empty:
                                agg_linea.to_excel(writer, sheet_name="Presupuesto_LINEA_PLUS", index=False)
                            if 'agg_ramo' in locals() and not agg_ramo.empty:
                                agg_ramo.to_excel(writer, sheet_name="Presupuesto_RAMO", index=False)
                        data_xls = output.getvalue()
                    st.download_button("⬇️ Descargar propuesta Presupuesto 2026", data=data_xls, file_name=f"presupuesto_2026_{future_year}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.info("No se generó propuesta 2026 (no hay datos o combinaciones).")
            except Exception as e:
                st.error("Error generando propuesta 2026. Ver logs.")
                st.exception(e)
    else:
        st.info("Pulsa 'Generar propuesta 2026' para crear la tabla desagregada y el archivo descargable.")

# End of file
