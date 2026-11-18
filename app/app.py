import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import lightgbm as lgb
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

# 1. Configuración general de página
st.set_page_config(
    page_title="Simulador de Rendimiento Agrícola",
    page_icon="🌾",
    layout="wide"
)

# Evitar que el traductor automático de Chrome congele los títulos dinámicos en el DOM
st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)

# 2. Carga en caché de los datos históricos de referencia
@st.cache_data
def load_reference_data() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "df_recortado.csv"
    df = pd.read_csv(data_path)
    df = df[df['superficie_cosechada_ha'] > 0].copy()
    return df

# 3. Carga o auto-entrenamiento en caché del modelo LightGBM
@st.cache_resource
def load_or_train_model(df_raw: pd.DataFrame):
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "lgbm_rendimiento.joblib"

    if model_path.exists():
        return joblib.load(model_path)

    # Si no existe el binario (ej. ignorado por .gitignore en despliegue cloud),
    # entrenamos el pipeline ganador directamente en memoria (~7 segundos)
    df = df_raw.copy()
    df['ALLSKY_SFC_SW_DWN'] = df['ALLSKY_SFC_SW_DWN'].replace(0, np.nan)
    df['log_sup_sembrada'] = np.log1p(df['superficie_sembrada_ha'])
    df['log_sup_cosechada'] = np.log1p(df['superficie_cosechada_ha'])
    df['ratio_cosechada'] = (
        df['superficie_cosechada_ha'] / df['superficie_sembrada_ha'].replace(0, np.nan)
    ).clip(upper=1.5)
    df['amplitud_termica'] = df['T2M_MAX'] - df['T2M_MIN']

    cat_ohe = ['cultivo', 'provincia']
    cat_te = ['departamento']
    num_cols = [
        'año', 'log_sup_sembrada', 'log_sup_cosechada', 'ratio_cosechada',
        'T2M', 'T2M_MAX', 'T2M_MIN', 'amplitud_termica',
        'ALLSKY_SFC_SW_DWN', 'RH2M', 'precip_total_mm'
    ]
    target = 'rendimiento_kgxha'

    df = df.dropna(subset=[target])
    X = df[cat_ohe + cat_te + num_cols]
    y = df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_ohe),
            ('te', TargetEncoder(random_state=42), cat_te),
            ('num', SimpleImputer(strategy='median'), num_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ))
    ])

    pipeline.fit(X, y)
    return pipeline

try:
    df_hist = load_reference_data()
    model = load_or_train_model(df_hist)
except Exception as e:
    st.error(f"Error al inicializar el modelo o los datos: {e}")
    st.stop()

# 4. Encabezado del Simulador
st.title("🌾 Simulador Predictivo de Rendimiento Agrícola (LightGBM)")
st.markdown(
    "Estimación de rendimiento (**kg/ha**) y producción total en base a ubicación geográfica, "
    "manejo de superficie y variables meteorológicas (NASA POWER)."
)
st.divider()

# 5. Panel Lateral: Controles de Simulación
st.sidebar.header("Parámetros de Simulación")

provincias = sorted(df_hist['provincia'].dropna().unique().tolist())
default_prov_idx = provincias.index("SANTA FE") if "SANTA FE" in provincias else 0
provincia_sel = st.sidebar.selectbox("Provincia:", provincias, index=default_prov_idx)

deptos_prov = sorted(df_hist[df_hist['provincia'] == provincia_sel]['departamento'].dropna().unique().tolist())
depto_sel = st.sidebar.selectbox("Departamento:", deptos_prov)

cultivos_prov = sorted(df_hist[df_hist['provincia'] == provincia_sel]['cultivo'].dropna().unique().tolist())
if not cultivos_prov:
    cultivos_prov = sorted(df_hist['cultivo'].dropna().unique().tolist())
cultivo_sel = st.sidebar.selectbox("Cultivo:", cultivos_prov)

anio_sel = st.sidebar.number_input(
    "Campaña (Año):",
    min_value=int(df_hist['año'].min()),
    max_value=2030,
    value=int(df_hist['año'].max())
)

st.sidebar.subheader("Superficie (Hectáreas)")
sup_sembrada = st.sidebar.number_input(
    "Superficie Sembrada (ha):",
    min_value=1.0,
    max_value=500000.0,
    value=1000.0,
    step=50.0
)
sup_cosechada = st.sidebar.number_input(
    "Superficie Cosechada (ha):",
    min_value=1.0,
    max_value=float(sup_sembrada),
    value=float(sup_sembrada),
    step=50.0
)

subset_zona = df_hist[(df_hist['provincia'] == provincia_sel) & (df_hist['departamento'] == depto_sel)]
if subset_zona.empty:
    subset_zona = df_hist[df_hist['provincia'] == provincia_sel]

rad_default = subset_zona['ALLSKY_SFC_SW_DWN'].replace(0, np.nan).mean()
if pd.isna(rad_default):
    rad_default = df_hist['ALLSKY_SFC_SW_DWN'].replace(0, np.nan).mean()

zona_key = f"{provincia_sel}_{depto_sel}"

st.sidebar.subheader("Escenario Climático")
t2m = st.sidebar.slider(
    "Temperatura Media — T2M (°C):", 5.0, 35.0,
    float(round(subset_zona['T2M'].mean(), 1)), 0.5, key=f"t2m_{zona_key}"
)
t2m_max = st.sidebar.slider(
    "Temperatura Máxima — T2M_MAX (°C):", 10.0, 45.0,
    float(round(subset_zona['T2M_MAX'].mean(), 1)), 0.5, key=f"tmax_{zona_key}"
)
t2m_min = st.sidebar.slider(
    "Temperatura Mínima — T2M_MIN (°C):", -5.0, 25.0,
    float(round(subset_zona['T2M_MIN'].mean(), 1)), 0.5, key=f"tmin_{zona_key}"
)
precip = st.sidebar.slider(
    "Precipitación Total Anual (mm):", 50.0, 2500.0,
    float(round(subset_zona['precip_total_mm'].mean(), 0)), 10.0, key=f"precip_{zona_key}"
)
rh2m = st.sidebar.slider(
    "Humedad Relativa — RH2M (%):", 20.0, 100.0,
    float(round(subset_zona['RH2M'].mean(), 1)), 1.0, key=f"rh_{zona_key}"
)
radiacion = st.sidebar.slider(
    "Radiación Solar (MJ/m²/día):", 5.0, 30.0,
    float(round(rad_default, 1)), 0.5, key=f"rad_{zona_key}"
)

# 6. Construcción del registro de entrada e inferencia
t_max_adj = max(t2m_max, t2m)
t_min_adj = min(t2m_min, t2m)

input_data = pd.DataFrame([{
    'cultivo': cultivo_sel,
    'provincia': provincia_sel,
    'departamento': depto_sel,
    'año': anio_sel,
    'log_sup_sembrada': np.log1p(sup_sembrada),
    'log_sup_cosechada': np.log1p(sup_cosechada),
    'ratio_cosechada': min(sup_cosechada / sup_sembrada, 1.5),
    'T2M': t2m,
    'T2M_MAX': t_max_adj,
    'T2M_MIN': t_min_adj,
    'amplitud_termica': t_max_adj - t_min_adj,
    'ALLSKY_SFC_SW_DWN': radiacion,
    'RH2M': rh2m,
    'precip_total_mm': precip
}])

pred_rendimiento = max(float(model.predict(input_data)[0]), 0.0)
produccion_estimada_tn = (pred_rendimiento * sup_cosechada) / 1000.0

hist_depto = df_hist[
    (df_hist['provincia'] == provincia_sel) &
    (df_hist['departamento'] == depto_sel) &
    (df_hist['cultivo'] == cultivo_sel)
]
hist_prov = df_hist[
    (df_hist['provincia'] == provincia_sel) &
    (df_hist['cultivo'] == cultivo_sel)
]

if not hist_depto.empty:
    media_hist = hist_depto['rendimiento_kgxha'].mean()
    ref_label = f"Media Histórica ({depto_sel})"
elif not hist_prov.empty:
    media_hist = hist_prov['rendimiento_kgxha'].mean()
    ref_label = f"Media Histórica ({provincia_sel})"
else:
    media_hist = df_hist[df_hist['cultivo'] == cultivo_sel]['rendimiento_kgxha'].mean()
    ref_label = "Media Histórica Nacional"

delta_pct = ((pred_rendimiento - media_hist) / media_hist * 100) if media_hist > 0 else 0.0

st.markdown(
    f'<div translate="no" style="padding: 0.75rem 1rem; border-radius: 0.5rem; background-color: #EFF6FF; color: #1E3A8A; border-left: 4px solid #2563EB; margin-bottom: 1rem;">'
    f'<b>Escenario Activo:</b> {cultivo_sel} &nbsp;|&nbsp; <b>Ubicación:</b> {depto_sel}, {provincia_sel} &nbsp;|&nbsp; <b>Campaña:</b> {anio_sel}'
    f'</div>',
    unsafe_allow_html=True
)

# 7. Resultados Principales (KPIs)
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    label="Rendimiento Estimado",
    value=f"{pred_rendimiento:,.0f} kg/ha",
    delta=f"{delta_pct:+.1f}% vs. referencia"
)
k2.metric(
    label=ref_label,
    value=f"{media_hist:,.0f} kg/ha"
)
k3.metric(
    label="Producción Total Proyectada",
    value=f"{produccion_estimada_tn:,.1f} tn"
)
k4.metric(
    label="Amplitud Térmica Simulada",
    value=f"{(t_max_adj - t_min_adj):.1f} °C"
)

st.divider()

# 8. Contexto Histórico Visual
g1, g2 = st.columns(2)

with g1:
    st.markdown(f"### Evolución Histórica: {cultivo_sel} en {provincia_sel}")
    if not hist_prov.empty:
        evol_prov = hist_prov.groupby('año', as_index=False)['rendimiento_kgxha'].mean()
        fig_evol = go.Figure()

        fig_evol.add_trace(go.Scatter(
            x=evol_prov['año'],
            y=evol_prov['rendimiento_kgxha'],
            mode='lines+markers',
            name=f"Promedio {provincia_sel}",
            line=dict(color='#16A34A', width=2.5)
        ))

        if not hist_depto.empty:
            evol_depto = hist_depto.groupby('año', as_index=False)['rendimiento_kgxha'].mean()
            fig_evol.add_trace(go.Scatter(
                x=evol_depto['año'],
                y=evol_depto['rendimiento_kgxha'],
                mode='lines+markers',
                name=f"Depto. {depto_sel}",
                line=dict(color='#0284C7', width=2, dash='dot')
            ))

        fig_evol.add_hline(
            y=pred_rendimiento,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text=f"Simulación: {pred_rendimiento:,.0f} kg/ha",
            annotation_position="top left"
        )

        fig_evol.update_layout(
            xaxis_title="Campaña",
            yaxis_title="Rendimiento (kg/ha)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=30, b=20, l=20, r=20),
            height=380
        )
        st.plotly_chart(fig_evol, use_container_width=True)
    else:
        st.info(f"No hay suficientes registros históricos para {cultivo_sel} en {provincia_sel}.")

with g2:
    st.markdown(f"### Ranking Provincial para {cultivo_sel}")
    df_cult = df_hist[df_hist['cultivo'] == cultivo_sel]
    if not df_cult.empty:
        rank_prov = (
            df_cult.groupby('provincia', as_index=False)['rendimiento_kgxha']
            .mean()
            .sort_values(by='rendimiento_kgxha', ascending=True)
            .tail(10)
        )
        fig_bar = px.bar(
            rank_prov,
            x='rendimiento_kgxha',
            y='provincia',
            orientation='h',
            labels={'rendimiento_kgxha': 'Rendimiento Promedio (kg/ha)', 'provincia': ''},
            color_discrete_sequence=['#2563EB'],
            template='plotly_white'
        )
        fig_bar.update_layout(margin=dict(t=30, b=20, l=20, r=20), height=380)
        st.plotly_chart(fig_bar, use_container_width=True)