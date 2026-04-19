# -*- coding: utf-8 -*-
"""
Streamlit app: Predicción Fasecolda
- Análisis predictivo de Primas (R. CIVIL y R.C. PROFESIONAL) con SARIMAX, XGBoost y LightGBM
- Sistema de alertas de Siniestros con mapa interactivo de Colombia
- Sábana de presupuesto 2026
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import plotly.graph_objects as go
from io import BytesIO, StringIO
from typing import Optional, Dict, List, Tuple

# Time series
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

# ML models
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Maps
import folium
from streamlit_folium import st_folium

# ---------- Config ----------
SHEET_ID = "1jhgIO0k5BTOpjCT2v1GveNCPkYA2tEu02LO1GyitD5k"
GID_PRIMAS = "604069835"
GID_SINIESTROS = "1221556540"
RAMOS_FILTRO = ["1 - R. CIVIL", "3 - R. C. PROFESIONAL"]
MESES_ESP = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]

st.set_page_config(
    page_title="Predicción Fasecolda · Primas y Siniestros",
    layout="wide",
    page_icon="📊",
)

# Thresholds and constants
# SARIMAX(1,1,1)x(1,1,1,12) needs at least 24 months for reliable estimation
MIN_MONTHS_REQUIRED = 24
# Fraction of available ML features held out for test evaluation
ML_TEST_RATIO = 0.25
# Assumed coefficient of variation when historical std is zero
DEFAULT_CV_FALLBACK = 0.10
MUNICIPIOS_COORDS: Dict[str, Tuple[float, float]] = {
    "11001": (4.7110, -74.0721),
    "05001": (6.2442, -75.5812),
    "76001": (3.4516, -76.5320),
    "08001": (10.9639, -74.7964),
    "13001": (10.3910, -75.4794),
    "68001": (7.1193, -73.1227),
    "54001": (7.8939, -72.5078),
    "17001": (5.0689, -75.5174),
    "63001": (4.5339, -75.6811),
    "66001": (4.8133, -75.6961),
    "73001": (4.4380, -75.2322),
    "41001": (2.9273, -75.2819),
    "52001": (1.2136, -77.2811),
    "23001": (8.7575, -75.8839),
    "70001": (9.3047, -75.3978),
    "50001": (4.1533, -73.6350),
    "15001": (5.5353, -73.3672),
    "18001": (1.6144, -75.6062),
    "44001": (11.5444, -72.9072),
    "47001": (11.2408, -74.1997),
    "20001": (10.4631, -73.2532),
    "19001": (2.4448, -76.6147),
    "76109": (3.8965, -76.2983),
    "25754": (4.8580, -74.0665),
    "25307": (4.6855, -74.1372),
    "25269": (4.7064, -74.2226),
    "25290": (4.5167, -74.3333),
    "05088": (6.3480, -75.9736),
    "05266": (6.5500, -75.6800),
    "76520": (3.8670, -76.9810),
    "08573": (10.7400, -74.7900),
    "68081": (6.8700, -73.0000),
}

# ---------- Utilities ----------

def fmt_cop(x) -> str:
    """Formatea un número como pesos colombianos."""
    try:
        if pd.isna(x):
            return "-"
    except Exception:
        return "-"
    try:
        return "$" + f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return str(x)


def parse_number(series: pd.Series) -> pd.Series:
    """Parsea columnas numéricas en formato colombiano (punto=miles, coma=decimal).

    Handles three cases:
    - Both '.' and ',': '.' is thousands separator, ',' is decimal (e.g. '1.234.567,89')
    - Only ',': ',' is decimal separator (e.g. '567,89')
    - Only '.': in financial/insurance data, treated as thousands separator (e.g. '1.234' → 1234)
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)
    has_dot = s.str.contains(".", regex=False)
    has_comma = s.str.contains(",", regex=False)
    result = s.copy()
    # Case 1: both separators → dot=thousands, comma=decimal
    both = has_dot & has_comma
    result[both] = s[both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    # Case 2: only comma → comma=decimal
    only_comma = ~has_dot & has_comma
    result[only_comma] = s[only_comma].str.replace(",", ".", regex=False)
    # Case 3: only dot → treat as thousands separator (remove dot)
    only_dot = has_dot & ~has_comma
    result[only_dot] = s[only_dot].str.replace(".", "", regex=False)
    return pd.to_numeric(result, errors="coerce")


def smape(y_true, y_pred) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(100 * np.mean(2 * np.abs(y_pred - y_true) / denom))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.array(y_true, dtype=float) - np.array(y_pred, dtype=float)) ** 2)))


def obtener_coordenadas(cod_mun) -> Optional[Tuple[float, float]]:
    """Devuelve (lat, lon) para un código DANE, o None si no está en la tabla."""
    key = str(cod_mun).strip().zfill(5)
    return MUNICIPIOS_COORDS.get(key)


def exportar_excel(dfs: Dict[str, pd.DataFrame]) -> bytes:
    """Exporta un dict {nombre_hoja: DataFrame} a bytes de Excel."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


# ---------- Feature Engineering ----------

def create_features(df: pd.DataFrame, target_col: str, max_lag: int = 12) -> pd.DataFrame:
    """Crea lags, rolling stats y variables temporales para modelos ML."""
    d = df.copy()
    for lag in range(1, max_lag + 1):
        d[f"lag_{lag}"] = d[target_col].shift(lag)
    for window in [3, 6, 12]:
        d[f"rolling_mean_{window}"] = d[target_col].rolling(window).mean()
        d[f"rolling_std_{window}"] = d[target_col].rolling(window).std()
    d["month"] = d.index.month
    d["quarter"] = d.index.quarter
    d["year"] = d.index.year
    d["trend"] = range(len(d))
    return d.dropna()


# ---------- Data Loading ----------

@st.cache_data(ttl=3600)
def cargar_datos_primas() -> pd.DataFrame:
    """Carga la Hoja1 (Primas) desde Google Sheets."""
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_PRIMAS}"
    try:
        return pd.read_csv(url)
    except Exception as exc:
        st.error(f"❌ Error cargando primas: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def cargar_datos_siniestros() -> pd.DataFrame:
    """Carga la Hoja2 (Siniestros) desde Google Sheets."""
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_SINIESTROS}"
    try:
        return pd.read_csv(url)
    except Exception as exc:
        st.error(f"❌ Error cargando siniestros: {exc}")
        return pd.DataFrame()


def _find_col(df: pd.DataFrame, *keywords) -> Optional[str]:
    """Busca la primera columna cuyo nombre contenga TODAS las keywords dadas."""
    for col in df.columns:
        cl = col.lower()
        if all(kw in cl for kw in keywords):
            return col
    return None


def preparar_primas(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Filtra y limpia el DataFrame de primas."""
    if df_raw.empty:
        return pd.DataFrame()
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    cod_col = _find_col(df, "codigo") or _find_col(df, "ramo")
    if cod_col is None:
        st.warning("No se encontró la columna 'Codigo y Ramo' en la hoja de primas.")
        return pd.DataFrame()

    cod_clean = df[cod_col].astype(str).str.strip()
    mask = cod_clean.isin(RAMOS_FILTRO)
    if not mask.any():
        # Fallback: regex for flexible matching
        mask = cod_clean.str.upper().str.contains(
            r"1.*R.*CIVIL|3.*R.*C.*PROFESIONAL", regex=True, na=False
        )
    df = df[mask].copy()
    if df.empty:
        st.warning("No hay datos para los ramos 1-R.CIVIL y 3-R.C.PROFESIONAL.")
        return pd.DataFrame()

    # Fecha
    fecha_col = _find_col(df, "mes") or _find_col(df, "fecha")
    if fecha_col:
        df["FECHA"] = pd.to_datetime(df[fecha_col], dayfirst=True, errors="coerce")
    else:
        df["FECHA"] = pd.NaT
    df["FECHA"] = df["FECHA"].dt.to_period("M").dt.to_timestamp()

    # Imp Prima (real) — columna G ≈ no contiene "cuota"
    imp_col = None
    cuota_col = None
    for col in df.columns:
        cl = col.lower()
        if "imp" in cl and "prima" in cl and "cuota" not in cl:
            imp_col = col
        if "imp" in cl and "cuota" in cl:
            cuota_col = col

    df["Imp Prima"] = parse_number(df[imp_col]) if imp_col else np.nan
    df["Imp Prima Cuota"] = parse_number(df[cuota_col]) if cuota_col else np.nan
    df["Codigo y Ramo"] = df[cod_col].astype(str).str.strip()

    return df.dropna(subset=["FECHA"]).copy()


def preparar_siniestros(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Filtra y limpia el DataFrame de siniestros."""
    if df_raw.empty:
        return pd.DataFrame()
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    comp_col = _find_col(df, "compa") or _find_col(df, "compañ")
    ramo_col = _find_col(df, "ramo")

    if comp_col and ramo_col:
        mask = (
            df[comp_col].astype(str).str.strip().str.upper() == "ESTADO"
        ) & (
            df[ramo_col].astype(str).str.upper().str.contains("RESPONSABI", na=False)
        )
        df = df[mask].copy()

    if df.empty:
        st.warning("No hay datos de siniestros para ESTADO / RESPONSABILIDAD.")
        return pd.DataFrame()

    fecha_col = _find_col(df, "fecha")
    if fecha_col:
        df["FECHA"] = pd.to_datetime(df[fecha_col], dayfirst=True, errors="coerce")
    else:
        df["FECHA"] = pd.NaT
    df["FECHA"] = df["FECHA"].dt.to_period("M").dt.to_timestamp()

    valor_col = _find_col(df, "valor")
    if valor_col:
        df["Valor"] = parse_number(df[valor_col])

    cod_mun_col = _find_col(df, "cod", "mun") or _find_col(df, "codmun")
    if cod_mun_col:
        df["Cod_Mun"] = df[cod_mun_col].astype(str).str.strip()

    ciudad_col = _find_col(df, "ciudad")
    if ciudad_col:
        df["CIUDAD"] = df[ciudad_col].astype(str).str.strip().str.upper()

    dep_col = _find_col(df, "depart")
    if dep_col:
        df["DEPARTAMENTO"] = df[dep_col].astype(str).str.strip().str.upper()

    return df.dropna(subset=["FECHA"]).copy()


# ---------- Model Training ----------

def _forecast_ml_iterative(ts: pd.Series, model, target_year: int = 2026) -> pd.Series:
    """Genera pronóstico iterativo mes-a-mes para modelos ML (XGBoost / LightGBM)."""
    ts_ext = ts.copy()
    end_date = pd.Timestamp(f"{target_year}-12-01")
    last_date = ts_ext.index.max()

    while last_date < end_date:
        next_date = last_date + pd.offsets.MonthBegin()
        vals = ts_ext.values
        row: Dict = {}
        for lag in range(1, 13):
            row[f"lag_{lag}"] = vals[-lag] if lag <= len(vals) else np.nan
        for window in [3, 6, 12]:
            row[f"rolling_mean_{window}"] = np.mean(vals[-window:]) if len(vals) >= window else float(np.mean(vals))
            row[f"rolling_std_{window}"] = float(np.std(vals[-window:])) if len(vals) >= window else 0.0
        row["month"] = next_date.month
        row["quarter"] = (next_date.month - 1) // 3 + 1
        row["year"] = next_date.year
        row["trend"] = len(ts_ext)

        try:
            pred_log = model.predict(pd.DataFrame([row]))[0]
            pred = max(0.0, float(np.expm1(pred_log)))
        except Exception:
            pred = float(ts_ext.mean())

        ts_ext[next_date] = pred
        last_date = next_date

    return ts_ext[ts_ext.index.year == target_year]


@st.cache_data(show_spinner=False)
def entrenar_modelos(ts_json: str, target_year: int = 2026) -> Dict:
    """
    Entrena SARIMAX, XGBoost y LightGBM con validación walk-forward.

    Parámetros
    ----------
    ts_json : str
        Serie temporal mensual serializada como JSON.
    target_year : int
        Año para el cual se genera el pronóstico (sábana).

    Retorna
    -------
    dict con resultados, métricas y pronósticos por modelo.
    """
    ts = pd.read_json(StringIO(ts_json), typ="series")
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index().asfreq("MS").interpolate(method="linear").dropna()

    if len(ts) < MIN_MONTHS_REQUIRED:
        return {}

    n_test = 12
    train = ts.iloc[:-n_test]
    test = ts.iloc[-n_test:]
    results: Dict = {"_train": train, "_test": test}

    # ── SARIMAX ──────────────────────────────────────────────────────────
    try:
        y_log = np.log1p(train)
        sar = SARIMAX(
            y_log,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res_sar = sar.fit(disp=False)
        pred_sar = np.expm1(res_sar.get_forecast(steps=n_test).predicted_mean.values)

        # Full model for 2026 forecast
        y_full_log = np.log1p(ts)
        sar_full = SARIMAX(
            y_full_log,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res_full = sar_full.fit(disp=False)
        last_date = ts.index.max()
        steps_ahead = (target_year - last_date.year) * 12 + (12 - last_date.month)
        steps_ahead = max(steps_ahead, 12)
        fc_idx = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(), periods=steps_ahead, freq="MS"
        )
        fc_vals = np.expm1(res_full.get_forecast(steps=steps_ahead).predicted_mean.values)
        fc_series = pd.Series(fc_vals, index=fc_idx)
        fc_2026 = fc_series[fc_series.index.year == target_year]

        residuals = res_sar.resid.values
        lb = acorr_ljungbox(residuals, lags=12, return_df=True)

        results["SARIMAX"] = {
            "pred_test": pred_sar,
            "smape": smape(test.values, pred_sar),
            "rmse": rmse(test.values, pred_sar),
            "mae": float(mean_absolute_error(test.values, pred_sar)),
            "residuals": residuals,
            "lb_test": lb,
            "fc_2026": fc_2026,
        }
    except Exception as exc:
        st.warning(f"SARIMAX no convergió: {exc}")

    # ── Shared ML features ───────────────────────────────────────────────
    try:
        df_ml = pd.DataFrame({"valor": ts.values}, index=ts.index)
        df_feat = create_features(df_ml, "valor", max_lag=12)

        X_all = df_feat.drop(columns=["valor"])
        y_all = df_feat["valor"]
        y_log_all = np.log1p(y_all)

        n_feat = len(df_feat)
        n_test_ml = min(n_test, int(n_feat * ML_TEST_RATIO))
        X_tr, X_te = X_all.iloc[:-n_test_ml], X_all.iloc[-n_test_ml:]
        y_tr_log = y_log_all.iloc[:-n_test_ml]
        y_te = y_all.iloc[-n_test_ml:]

        results["_ml_test_index"] = X_te.index

        # ── XGBoost ──────────────────────────────────────────────────────
        xgb_m = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        xgb_m.fit(X_tr, y_tr_log)
        pred_xgb = np.expm1(xgb_m.predict(X_te))
        fi_xgb = (
            pd.DataFrame({"Feature": X_tr.columns, "Importancia": xgb_m.feature_importances_})
            .sort_values("Importancia", ascending=False)
        )
        fc_xgb = _forecast_ml_iterative(ts, xgb_m, target_year=target_year)

        results["XGBoost"] = {
            "pred_test": pred_xgb,
            "smape": smape(y_te.values, pred_xgb),
            "rmse": rmse(y_te.values, pred_xgb),
            "mae": float(mean_absolute_error(y_te.values, pred_xgb)),
            "feature_importance": fi_xgb,
            "fc_2026": fc_xgb,
        }

        # ── LightGBM ─────────────────────────────────────────────────────
        lgb_m = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        lgb_m.fit(X_tr, y_tr_log)
        pred_lgb = np.expm1(lgb_m.predict(X_te))
        fi_lgb = (
            pd.DataFrame({"Feature": X_tr.columns, "Importancia": lgb_m.feature_importances_})
            .sort_values("Importancia", ascending=False)
        )
        fc_lgb = _forecast_ml_iterative(ts, lgb_m, target_year=target_year)

        results["LightGBM"] = {
            "pred_test": pred_lgb,
            "smape": smape(y_te.values, pred_lgb),
            "rmse": rmse(y_te.values, pred_lgb),
            "mae": float(mean_absolute_error(y_te.values, pred_lgb)),
            "feature_importance": fi_lgb,
            "fc_2026": fc_lgb,
        }

    except Exception as exc:
        st.warning(f"Error en XGBoost/LightGBM: {exc}")

    return results


# ---------- Alert System ----------

def calcular_alerta(pronostico: float, historico_promedio: float, historico_std: float) -> Tuple[str, str]:
    """Clasifica el nivel de alerta por z-score respecto al histórico."""
    if historico_std == 0:
        # Assume a DEFAULT_CV_FALLBACK coefficient of variation when std is unavailable
        historico_std = historico_promedio * DEFAULT_CV_FALLBACK if historico_promedio > 0 else 1.0
    z = (pronostico - historico_promedio) / historico_std
    if z > 2:
        return "CRÍTICO", "#8B0000"
    if z > 1:
        return "ALTO", "#FF0000"
    if z > 0.5:
        return "MEDIO", "#FFA500"
    if z > 0:
        return "MODERADO", "#FFD700"
    return "BAJO", "#00CC00"


# ============================================================
# STREAMLIT APP
# ============================================================

st.title("📊 Predicción Fasecolda · Primas y Siniestros")
st.markdown(
    "**Análisis predictivo — Ramos Responsabilidad Civil (R.CIVIL y R.C.PROFESIONAL)**  \n"
    "Modelos: SARIMAX · XGBoost · LightGBM &nbsp;|&nbsp; Métricas: SMAPE · RMSE · MAE"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")
target_year = st.sidebar.number_input("📅 Año Objetivo", min_value=2026, max_value=2030, value=2026)
ramo_sel = st.sidebar.selectbox("📋 Ramo", ["Ambos"] + RAMOS_FILTRO)
ipc_pct = st.sidebar.slider("📈 Ajuste IPC (%)", 0.0, 10.0, 3.5, 0.5)
ipc_factor = ipc_pct / 100

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Recargar Datos"):
    st.cache_data.clear()
    st.rerun()

# ── Load & prepare data ───────────────────────────────────────────────────────
with st.spinner("⏳ Cargando datos desde Google Sheets…"):
    df_primas_raw = cargar_datos_primas()
    df_sin_raw = cargar_datos_siniestros()

df_primas = preparar_primas(df_primas_raw) if not df_primas_raw.empty else pd.DataFrame()
df_sin = preparar_siniestros(df_sin_raw) if not df_sin_raw.empty else pd.DataFrame()

# Apply ramo filter
if not df_primas.empty and ramo_sel != "Ambos":
    df_primas_f = df_primas[df_primas["Codigo y Ramo"] == ramo_sel].copy()
else:
    df_primas_f = df_primas.copy()

# Status bar
c1, c2, c3 = st.columns(3)
with c1:
    if not df_primas.empty:
        st.success(f"✅ Primas: {len(df_primas):,} registros · Último dato: {df_primas['FECHA'].max().strftime('%b %Y')}")
    else:
        st.error("❌ Sin datos de primas")
with c2:
    if not df_sin.empty:
        st.success(f"✅ Siniestros: {len(df_sin):,} registros · Último dato: {df_sin['FECHA'].max().strftime('%b %Y')}")
    else:
        st.warning("⚠️ Sin datos de siniestros")
with c3:
    if not df_primas.empty:
        ramos_disp = ", ".join(df_primas["Codigo y Ramo"].unique())
        st.info(f"📋 Ramos disponibles: {ramos_disp}")

st.markdown("---")

# ── Train models (cached) ────────────────────────────────────────────────────
resultados_primas: Dict = {}
resultados_sin: Dict = {}

if not df_primas_f.empty:
    ts_primas = (
        df_primas_f.groupby("FECHA")["Imp Prima"]
        .sum()
        .sort_index()
        .asfreq("MS")
        .interpolate(method="linear")
        .dropna()
    )
    if len(ts_primas) >= MIN_MONTHS_REQUIRED:
        with st.spinner("🤖 Entrenando modelos de Primas (SARIMAX · XGBoost · LightGBM)…"):
            try:
                resultados_primas = entrenar_modelos(ts_primas.to_json(), target_year=int(target_year))
            except Exception as exc:
                st.error(f"❌ Error entrenando modelos de primas: {exc}")
                resultados_primas = {}
    ts_cuota = (
        df_primas_f.groupby("FECHA")["Imp Prima Cuota"]
        .sum()
        .sort_index()
        .asfreq("MS")
        .fillna(0)
    )
else:
    ts_primas = pd.Series(dtype=float)
    ts_cuota = pd.Series(dtype=float)

if not df_sin.empty and "Valor" in df_sin.columns:
    ts_sin = (
        df_sin.groupby("FECHA")["Valor"]
        .sum()
        .sort_index()
        .asfreq("MS")
        .interpolate(method="linear")
        .dropna()
    )
    if len(ts_sin) >= MIN_MONTHS_REQUIRED:
        with st.spinner("🤖 Entrenando modelos de Siniestros…"):
            try:
                resultados_sin = entrenar_modelos(ts_sin.to_json(), target_year=int(target_year))
            except Exception as exc:
                st.error(f"❌ Error entrenando modelos de siniestros: {exc}")
                resultados_sin = {}
else:
    ts_sin = pd.Series(dtype=float)

# ── Sidebar footer ────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.info(
    "**📊 Predicción Fasecolda**  \n"
    "Ramos: R.Civil · R.C.Profesional  \n"
    "Modelos: SARIMAX · XGBoost · LightGBM"
)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(
    ["📊 Análisis de Primas", "🚨 Análisis de Siniestros", "📈 Comparación de Modelos"]
)

# ─────────────────────────────────────────────────────────────
# TAB 1 · ANÁLISIS DE PRIMAS
# ─────────────────────────────────────────────────────────────
with tab1:
    st.header("📊 Análisis Predictivo de Primas")

    if ts_primas.empty:
        st.warning("⚠️ No hay datos de primas con los filtros actuales.")
    else:
        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        cur_yr = ts_primas.index.max().year
        sum_yr = float(ts_primas[ts_primas.index.year == cur_yr].sum())
        sum_cuota_yr = float(ts_cuota[ts_cuota.index.year == cur_yr].sum()) if not ts_cuota.empty else 0.0
        pct_cumpl = (sum_yr / sum_cuota_yr * 100) if sum_cuota_yr > 0 else 0.0
        k1.metric("📈 Último mes primas", fmt_cop(ts_primas.iloc[-1]))
        k2.metric("📊 Promedio mensual (12m)", fmt_cop(ts_primas.tail(12).mean()))
        k3.metric(f"💰 Total {cur_yr}", fmt_cop(sum_yr))
        k4.metric("🎯 Cumplimiento vs Cuota", f"{pct_cumpl:.1f}%")

        # Histórico Primas vs Cuota
        st.subheader("📈 Histórico: Primas vs Presupuesto (Cuota)")
        df_comp = pd.DataFrame({"Primas Reales": ts_primas, "Presupuesto (Cuota)": ts_cuota}).dropna(subset=["Primas Reales"])
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=df_comp.index, y=df_comp["Primas Reales"], name="Primas Reales", line=dict(color="#2f5597", width=2)))
        fig_hist.add_trace(go.Scatter(x=df_comp.index, y=df_comp["Presupuesto (Cuota)"], name="Presupuesto (Cuota)", line=dict(color="#e67e22", width=2, dash="dash")))
        fig_hist.update_layout(title="Histórico Primas vs Presupuesto", xaxis_title="Fecha", yaxis_title="COP", height=380, hovermode="x unified")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Sábana 2026
        mejor = None
        if resultados_primas:
            metricas_p = {m: resultados_primas[m]["smape"] for m in ["SARIMAX", "XGBoost", "LightGBM"] if m in resultados_primas}
            if metricas_p:
                mejor = min(metricas_p, key=lambda m: metricas_p[m])

        st.subheader(f"📋 Sábana de Presupuesto {target_year}")
        if mejor and "fc_2026" in resultados_primas[mejor] and not resultados_primas[mejor]["fc_2026"].empty:
            fc = resultados_primas[mejor]["fc_2026"]
            fc_ipc = fc * (1 + ipc_factor)
            sabana = pd.DataFrame({
                "Mes": [MESES_ESP[m - 1] for m in fc.index.month],
                "Pronóstico Primas": fc.values,
                "Presupuesto Sugerido (+IPC)": fc_ipc.values,
                "Acumulado": fc.cumsum().values,
                "Acumulado +IPC": fc_ipc.cumsum().values,
            })

            ci1, ci2 = st.columns([3, 1])
            with ci1:
                st.info(f"✅ Modelo: **{mejor}** · SMAPE: **{metricas_p[mejor]:.2f}%** · IPC aplicado: **{ipc_pct:.1f}%**")
            with ci2:
                st.metric(f"💰 Total Proyectado {target_year}", fmt_cop(fc.sum()))

            sabana_disp = sabana.copy()
            for col in ["Pronóstico Primas", "Presupuesto Sugerido (+IPC)", "Acumulado", "Acumulado +IPC"]:
                sabana_disp[col] = sabana_disp[col].apply(fmt_cop)
            st.dataframe(sabana_disp, use_container_width=True)

            # Comparación histórica Primas vs Cuota
            comp_hist = df_comp.reset_index().rename(columns={"index": "Fecha"})
            comp_hist["Diferencia"] = comp_hist["Primas Reales"] - comp_hist["Presupuesto (Cuota)"]
            comp_hist["Cumplimiento %"] = (comp_hist["Primas Reales"] / comp_hist["Presupuesto (Cuota)"] * 100).round(1)

            excel_bytes = exportar_excel({"Sábana 2026": sabana, "Primas vs Cuota": comp_hist})
            st.download_button(
                "📥 Descargar Excel (Sábana + Comparativo)",
                data=excel_bytes,
                file_name=f"sabana_presupuesto_{target_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.warning("⚠️ Los modelos aún no generaron el pronóstico (se necesitan ≥ 24 meses de datos).")

        # Métricas comparativas
        if resultados_primas:
            st.subheader("📊 Comparativa de Métricas (Primas)")
            rows = []
            for m in ["SARIMAX", "XGBoost", "LightGBM"]:
                if m in resultados_primas:
                    rows.append({
                        "Modelo": m,
                        "SMAPE (%)": round(resultados_primas[m]["smape"], 2),
                        "RMSE": fmt_cop(round(resultados_primas[m]["rmse"], 0)),
                        "MAE": fmt_cop(round(resultados_primas[m]["mae"], 0)),
                    })
            if rows:
                df_met = pd.DataFrame(rows).set_index("Modelo")
                st.dataframe(df_met, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 2 · ANÁLISIS DE SINIESTROS
# ─────────────────────────────────────────────────────────────
with tab2:
    st.header("🚨 Sistema de Alertas de Siniestros")

    if ts_sin.empty:
        st.warning("⚠️ No hay datos de siniestros disponibles.")
    else:
        # KPIs
        s1, s2, s3 = st.columns(3)
        s1.metric("⚠️ Último mes siniestros", fmt_cop(ts_sin.iloc[-1]))
        s2.metric("📊 Promedio mensual (12m)", fmt_cop(ts_sin.tail(12).mean()))
        if "CIUDAD" in df_sin.columns:
            s3.metric("🏙️ Ciudades con siniestros", df_sin["CIUDAD"].nunique())

        # Histórico
        st.subheader("📈 Histórico de Siniestros")
        fig_sin_h = go.Figure()
        fig_sin_h.add_trace(go.Scatter(x=ts_sin.index, y=ts_sin.values, fill="tozeroy", name="Siniestros", line=dict(color="#e74c3c", width=2)))
        fig_sin_h.update_layout(title="Evolución Histórica de Siniestros", xaxis_title="Fecha", yaxis_title="COP", height=320)
        st.plotly_chart(fig_sin_h, use_container_width=True)

        # Alert map
        st.subheader("🗺️ Mapa de Alertas por Municipio")

        mejor_sin = None
        if resultados_sin:
            metricas_sin = {m: resultados_sin[m]["smape"] for m in ["SARIMAX", "XGBoost", "LightGBM"] if m in resultados_sin}
            if metricas_sin:
                mejor_sin = min(metricas_sin, key=lambda m: metricas_sin[m])

        fc_total_2026 = 0.0
        if mejor_sin and "fc_2026" in resultados_sin.get(mejor_sin, {}):
            fc_total_2026 = float(resultados_sin[mejor_sin]["fc_2026"].sum())
        elif not ts_sin.empty:
            fc_total_2026 = float(ts_sin.tail(12).sum() * 1.05)

        if "CIUDAD" in df_sin.columns and "Cod_Mun" in df_sin.columns:
            city_stats = (
                df_sin.groupby(["CIUDAD", "Cod_Mun"])["Valor"]
                .agg(["sum", "mean", "std"])
                .reset_index()
            )
            city_stats.columns = ["CIUDAD", "Cod_Mun", "Total", "Promedio", "Desv_Std"]
            city_prop = df_sin.groupby("CIUDAD")["Valor"].sum()
            city_prop = city_prop / city_prop.sum() if city_prop.sum() > 0 else city_prop

            alertas = []
            for _, row in city_stats.iterrows():
                prop = city_prop.get(row["CIUDAD"], 0.0)
                fc_anual = fc_total_2026 * prop
                fc_mensual = fc_anual / 12
                prom = row["Promedio"]
                std = row["Desv_Std"] if row["Desv_Std"] > 0 else prom * 0.2
                nivel, color = calcular_alerta(fc_mensual, prom, std)
                desv_pct = ((fc_mensual - prom) / prom * 100) if prom > 0 else 0.0
                alertas.append({
                    "Ciudad": row["CIUDAD"],
                    "Cod_Mun": str(row["Cod_Mun"]),
                    "Pronóstico Anual 2026": round(fc_anual, 0),
                    "Pronóstico Mensual": round(fc_mensual, 0),
                    "Promedio Histórico Mensual": round(prom, 0),
                    "Desviación vs Histórico (%)": round(desv_pct, 1),
                    "Nivel Alerta": nivel,
                    "Color": color,
                })

            df_alertas = pd.DataFrame(alertas).sort_values("Pronóstico Anual 2026", ascending=False)

            # Build Folium map
            mapa = folium.Map(location=[4.5709, -74.2973], zoom_start=6, tiles="OpenStreetMap")
            n_pins = 0
            for _, row in df_alertas.iterrows():
                coords = obtener_coordenadas(row["Cod_Mun"])
                if coords is None:
                    continue
                lat, lon = coords
                color_f = {
                    "#8B0000": "darkred",
                    "#FF0000": "red",
                    "#FFA500": "orange",
                    "#FFD700": "beige",
                    "#00CC00": "green",
                }.get(row["Color"], "blue")
                popup_html = (
                    f"<div style='font-family:Arial;width:200px'>"
                    f"<h4 style='margin:0'>{row['Ciudad']}</h4><hr style='margin:4px 0'>"
                    f"<p style='margin:2px 0'><b>Nivel:</b> "
                    f"<span style='color:{row['Color']};font-weight:bold'>{row['Nivel Alerta']}</span></p>"
                    f"<p style='margin:2px 0'><b>Pronóstico 2026:</b> {fmt_cop(row['Pronóstico Anual 2026'])}</p>"
                    f"<p style='margin:2px 0'><b>Prom. Mensual:</b> {fmt_cop(row['Promedio Histórico Mensual'])}</p>"
                    f"<p style='margin:2px 0'><b>Desviación:</b> {row['Desviación vs Histórico (%)']:.1f}%</p>"
                    f"</div>"
                )
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=12,
                    popup=folium.Popup(popup_html, max_width=260),
                    tooltip=f"{row['Ciudad']} — {row['Nivel Alerta']}",
                    color=row["Color"],
                    fill=True,
                    fillColor=row["Color"],
                    fillOpacity=0.75,
                    weight=2,
                ).add_to(mapa)
                n_pins += 1

            legend_html = (
                "<div style='position:fixed;bottom:50px;left:50px;background:white;"
                "padding:10px;border:2px solid grey;z-index:9999;font-size:13px;border-radius:5px'>"
                "<p style='margin:0;font-weight:bold'>🚨 Nivel de Alerta</p>"
                "<p><span style='color:#00CC00;font-size:18px'>●</span> BAJO</p>"
                "<p><span style='color:#FFD700;font-size:18px'>●</span> MODERADO</p>"
                "<p><span style='color:#FFA500;font-size:18px'>●</span> MEDIO</p>"
                "<p><span style='color:#FF0000;font-size:18px'>●</span> ALTO</p>"
                "<p><span style='color:#8B0000;font-size:18px'>●</span> CRÍTICO</p>"
                "</div>"
            )
            mapa.get_root().html.add_child(folium.Element(legend_html))
            st_folium(mapa, width=None, height=520)
            if n_pins > 0:
                st.caption(f"📍 {n_pins} municipios mapeados")
            else:
                st.info("ℹ️ No se encontraron coordenadas para los municipios en los datos.")

            # Alerts table
            st.subheader("📋 Detalle de Alertas por Ciudad")
            COLOR_BG = {
                "CRÍTICO": "background-color:#8B0000;color:white",
                "ALTO": "background-color:#FF6666",
                "MEDIO": "background-color:#FFD700",
                "MODERADO": "background-color:#FFFF99",
                "BAJO": "background-color:#90EE90",
            }

            def _color_nivel(val):
                return COLOR_BG.get(val, "")

            df_alert_disp = df_alertas.drop(columns=["Color"]).copy()
            for col in ["Pronóstico Anual 2026", "Pronóstico Mensual", "Promedio Histórico Mensual"]:
                df_alert_disp[col] = df_alert_disp[col].apply(fmt_cop)
            st.dataframe(df_alert_disp.style.map(_color_nivel, subset=["Nivel Alerta"]), use_container_width=True)

            # Insights
            st.subheader("💡 Insights y Recomendaciones")
            for _, row in df_alertas.iterrows():
                nivel = row["Nivel Alerta"]
                ciudad = row["Ciudad"]
                desv = row["Desviación vs Histórico (%)"]
                if nivel == "CRÍTICO":
                    st.error(f"🚨 **{ciudad}** (CRÍTICO): Incremento esperado **{desv:.1f}%**. ACCIÓN: Revisar suscripción, aumentar reservas técnicas y evaluar reaseguro.")
                elif nivel == "ALTO":
                    st.warning(f"⚠️ **{ciudad}** (ALTO): Incremento proyectado **{desv:.1f}%**. ACCIÓN: Monitoreo cercano, ajustar primas y condiciones de cobertura.")
                elif nivel == "MEDIO":
                    st.info(f"📊 **{ciudad}** (MEDIO): Variación esperada **{desv:.1f}%**. ACCIÓN: Seguimiento mensual, revisar tendencias del mercado local.")
            bajos = df_alertas[df_alertas["Nivel Alerta"].isin(["BAJO", "MODERADO"])]
            if len(bajos):
                st.success(f"✅ **{len(bajos)} ciudades** con nivel BAJO/MODERADO: comportamiento dentro de parámetros históricos normales.")

            # Sábana siniestros
            if mejor_sin and "fc_2026" in resultados_sin.get(mejor_sin, {}) and not resultados_sin[mejor_sin]["fc_2026"].empty:
                st.subheader(f"📋 Pronóstico de Siniestros por Mes {target_year}")
                fc_s = resultados_sin[mejor_sin]["fc_2026"]
                sab_sin = pd.DataFrame({
                    "Mes": [MESES_ESP[m - 1] for m in fc_s.index.month],
                    "Pronóstico Siniestros": fc_s.values,
                    "Acumulado": fc_s.cumsum().values,
                })
                sab_sin_disp = sab_sin.copy()
                for col in ["Pronóstico Siniestros", "Acumulado"]:
                    sab_sin_disp[col] = sab_sin_disp[col].apply(fmt_cop)
                st.dataframe(sab_sin_disp, use_container_width=True)
        else:
            st.warning("⚠️ Los datos de siniestros no contienen columnas CIUDAD y Cod_Mun necesarias para el mapa.")

# ─────────────────────────────────────────────────────────────
# TAB 3 · COMPARACIÓN DE MODELOS
# ─────────────────────────────────────────────────────────────
with tab3:
    st.header("📈 Comparación de Modelos Predictivos")

    # ── Primas ────────────────────────────────────────────────
    st.subheader("📊 Métricas de Desempeño — Primas")
    if resultados_primas:
        rows_p = []
        for m in ["SARIMAX", "XGBoost", "LightGBM"]:
            if m in resultados_primas:
                rows_p.append({
                    "Modelo": m,
                    "SMAPE (%)": round(resultados_primas[m]["smape"], 2),
                    "RMSE": fmt_cop(round(resultados_primas[m]["rmse"], 0)),
                    "MAE": fmt_cop(round(resultados_primas[m]["mae"], 0)),
                })
        if rows_p:
            mejor_p = min(rows_p, key=lambda r: r["SMAPE (%)"])["Modelo"]
            st.success(f"🏆 Mejor modelo (Primas): **{mejor_p}**")
            st.dataframe(pd.DataFrame(rows_p).set_index("Modelo"), use_container_width=True)

        # Comparative chart
        st.subheader("📈 Pronósticos Comparados — Primas")
        train_p = resultados_primas.get("_train", pd.Series(dtype=float))
        test_p = resultados_primas.get("_test", pd.Series(dtype=float))
        if not train_p.empty and not test_p.empty:
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(x=train_p.index, y=train_p.values, name="Entrenamiento", line=dict(color="#2f5597", width=2)))
            fig_c.add_trace(go.Scatter(x=test_p.index, y=test_p.values, name="Real", line=dict(color="#d62728", width=2.5)))
            ml_idx = resultados_primas.get("_ml_test_index", test_p.index)
            for model_name, color in [("SARIMAX", "#1f77b4"), ("XGBoost", "#ff7f0e"), ("LightGBM", "#2ca02c")]:
                if model_name in resultados_primas:
                    idx = test_p.index if model_name == "SARIMAX" else ml_idx
                    fig_c.add_trace(go.Scatter(x=idx, y=resultados_primas[model_name]["pred_test"], name=model_name, line=dict(color=color, width=2, dash="dash")))
            fig_c.add_vline(x=train_p.index[-1].strftime('%Y-%m-%d'), line_dash="dot", line_color="gray", annotation_text="Inicio del pronóstico")
            fig_c.update_layout(title="Comparación Modelos — Primas", xaxis_title="Fecha", yaxis_title="COP", height=480, hovermode="x unified")
            st.plotly_chart(fig_c, use_container_width=True)

        # SARIMAX residuals
        if "SARIMAX" in resultados_primas:
            with st.expander("🔬 Diagnóstico de Residuos SARIMAX — Primas", expanded=False):
                res = resultados_primas["SARIMAX"]["residuals"]
                lb = resultados_primas["SARIMAX"]["lb_test"]
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Media Residuos", f"{res.mean():.4f}")
                r2.metric("Desv. Estándar", f"{res.std():.4f}")
                r3.metric("Asimetría", f"{float(pd.Series(res).skew()):.4f}")
                r4.metric("Curtosis", f"{float(pd.Series(res).kurt()):.4f}")
                st.markdown("**Test de Ljung-Box (autocorrelación residuos):**")
                if (lb["lb_pvalue"] > 0.05).all():
                    st.success("✅ Residuos se comportan como ruido blanco (p > 0.05 en todos los lags)")
                else:
                    st.warning("⚠️ Residuos presentan autocorrelación significativa en algunos lags")
                st.dataframe(lb.round(4), use_container_width=True)
                fig_r, axes = plt.subplots(2, 2, figsize=(12, 7))
                axes[0, 0].plot(res, color="#2f5597")
                axes[0, 0].axhline(0, color="red", linestyle="--")
                axes[0, 0].set_title("Residuos en el tiempo")
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 1].hist(res, bins=20, color="#1f77b4", edgecolor="white", alpha=0.8)
                axes[0, 1].set_title("Histograma de Residuos")
                axes[0, 1].grid(True, alpha=0.3)
                scipy_stats.probplot(res, dist="norm", plot=axes[1, 0])
                axes[1, 0].set_title("Q-Q Plot")
                axes[1, 0].grid(True, alpha=0.3)
                plot_acf(res, lags=20, ax=axes[1, 1], color="#2f5597")
                axes[1, 1].set_title("ACF de Residuos")
                axes[1, 1].grid(True, alpha=0.3)
                plt.suptitle("Diagnóstico de Residuos — SARIMAX (Primas)", fontsize=13, weight="bold")
                plt.tight_layout()
                st.pyplot(fig_r)
                plt.close(fig_r)

        # Feature importance
        for model_name, color in [("XGBoost", "#ff7f0e"), ("LightGBM", "#2ca02c")]:
            if model_name in resultados_primas and "feature_importance" in resultados_primas[model_name]:
                with st.expander(f"🔍 Feature Importance — {model_name} (Primas)", expanded=False):
                    fi = resultados_primas[model_name]["feature_importance"].head(15)
                    fig_fi, ax_fi = plt.subplots(figsize=(9, 5))
                    ax_fi.barh(fi["Feature"][::-1], fi["Importancia"][::-1], color=color, alpha=0.85)
                    ax_fi.set_title(f"Top 15 Variables — {model_name} (Primas)", weight="bold")
                    ax_fi.set_xlabel("Importancia")
                    ax_fi.grid(True, alpha=0.3, axis="x")
                    plt.tight_layout()
                    st.pyplot(fig_fi)
                    plt.close(fig_fi)
    else:
        st.info("ℹ️ Los modelos de primas no están disponibles (necesitan ≥ 24 meses de datos).")

    # ── Siniestros ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Métricas de Desempeño — Siniestros")
    if resultados_sin:
        rows_s = []
        for m in ["SARIMAX", "XGBoost", "LightGBM"]:
            if m in resultados_sin:
                rows_s.append({
                    "Modelo": m,
                    "SMAPE (%)": round(resultados_sin[m]["smape"], 2),
                    "RMSE": fmt_cop(round(resultados_sin[m]["rmse"], 0)),
                    "MAE": fmt_cop(round(resultados_sin[m]["mae"], 0)),
                })
        if rows_s:
            mejor_s = min(rows_s, key=lambda r: r["SMAPE (%)"])["Modelo"]
            st.success(f"🏆 Mejor modelo (Siniestros): **{mejor_s}**")
            st.dataframe(pd.DataFrame(rows_s).set_index("Modelo"), use_container_width=True)

        st.subheader("📈 Pronósticos Comparados — Siniestros")
        train_s = resultados_sin.get("_train", pd.Series(dtype=float))
        test_s = resultados_sin.get("_test", pd.Series(dtype=float))
        if not train_s.empty and not test_s.empty:
            fig_cs = go.Figure()
            fig_cs.add_trace(go.Scatter(x=train_s.index, y=train_s.values, name="Entrenamiento", line=dict(color="#2f5597", width=2)))
            fig_cs.add_trace(go.Scatter(x=test_s.index, y=test_s.values, name="Real", line=dict(color="#d62728", width=2.5)))
            ml_idx_s = resultados_sin.get("_ml_test_index", test_s.index)
            for model_name, color in [("SARIMAX", "#1f77b4"), ("XGBoost", "#ff7f0e"), ("LightGBM", "#2ca02c")]:
                if model_name in resultados_sin:
                    idx = test_s.index if model_name == "SARIMAX" else ml_idx_s
                    fig_cs.add_trace(go.Scatter(x=idx, y=resultados_sin[model_name]["pred_test"], name=model_name, line=dict(color=color, width=2, dash="dash")))
            fig_cs.add_vline(x=train_s.index[-1].strftime('%Y-%m-%d'), line_dash="dot", line_color="gray", annotation_text="Inicio del pronóstico")
            fig_cs.update_layout(title="Comparación Modelos — Siniestros", xaxis_title="Fecha", yaxis_title="COP", height=450, hovermode="x unified")
            st.plotly_chart(fig_cs, use_container_width=True)

        if "SARIMAX" in resultados_sin:
            with st.expander("🔬 Diagnóstico de Residuos SARIMAX — Siniestros", expanded=False):
                res_s = resultados_sin["SARIMAX"]["residuals"]
                lb_s = resultados_sin["SARIMAX"]["lb_test"]
                rs1, rs2, rs3, rs4 = st.columns(4)
                rs1.metric("Media Residuos", f"{res_s.mean():.4f}")
                rs2.metric("Desv. Estándar", f"{res_s.std():.4f}")
                rs3.metric("Asimetría", f"{float(pd.Series(res_s).skew()):.4f}")
                rs4.metric("Curtosis", f"{float(pd.Series(res_s).kurt()):.4f}")
                if (lb_s["lb_pvalue"] > 0.05).all():
                    st.success("✅ Residuos se comportan como ruido blanco")
                else:
                    st.warning("⚠️ Residuos presentan autocorrelación significativa")
                st.dataframe(lb_s.round(4), use_container_width=True)

        for model_name, color in [("XGBoost", "#ff7f0e"), ("LightGBM", "#2ca02c")]:
            if model_name in resultados_sin and "feature_importance" in resultados_sin[model_name]:
                with st.expander(f"🔍 Feature Importance — {model_name} (Siniestros)", expanded=False):
                    fi_s = resultados_sin[model_name]["feature_importance"].head(15)
                    fig_fis, ax_fis = plt.subplots(figsize=(9, 5))
                    ax_fis.barh(fi_s["Feature"][::-1], fi_s["Importancia"][::-1], color=color, alpha=0.85)
                    ax_fis.set_title(f"Top 15 Variables — {model_name} (Siniestros)", weight="bold")
                    ax_fis.set_xlabel("Importancia")
                    ax_fis.grid(True, alpha=0.3, axis="x")
                    plt.tight_layout()
                    st.pyplot(fig_fis)
                    plt.close(fig_fis)
    else:
        st.info("ℹ️ Los modelos de siniestros no están disponibles (necesitan ≥ 24 meses de datos).")
