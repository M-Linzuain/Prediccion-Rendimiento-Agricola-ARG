import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb

def build_and_save_model() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "df_recortado.csv"
    model_path = base_dir / "models" / "lgbm_rendimiento.joblib"

    print(f"Cargando dataset desde: {data_path}")
    df = pd.read_csv(data_path)

    # 1. Filtrado e Ingeniería de Características (idéntica al notebook)
    df = df[df['superficie_cosechada_ha'] > 0].copy()
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

    # Eliminar nulos en el target si existieran
    df = df.dropna(subset=[target])

    X = df[cat_ohe + cat_te + num_cols]
    y = df[target]

    # 2. Preprocesamiento encapsulado (evita data leakage)
    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_ohe),
            ('te', TargetEncoder(random_state=42), cat_te),
            ('num', SimpleImputer(strategy='median'), num_cols)
        ]
    )

    # 3. Pipeline completo con LightGBM
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

    print("Entrenando Pipeline LightGBM sobre los datos completos...")
    pipeline.fit(X, y)

    # 4. Guardar artefacto serializado
    joblib.dump(pipeline, model_path)
    print(f"Modelo guardado exitosamente en: {model_path}")

if __name__ == "__main__":
    build_and_save_model()