# --- PASO 0: IMPORTACIONES ---
# ==================================
import pandas as pd
import numpy as np
import ast
import warnings

# Sklearn (Procesamiento y Validación)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# CatBoost
from catboost import CatBoostRegressor, Pool

# Configuración General
warnings.filterwarnings('ignore')
print("Librerías cargadas. Iniciando pipeline FINAL (Submission #18 - Ensemble de Finalistas)...")


# --- PASO 1: CONFIGURACIÓN GLOBAL ---
# =====================================
PATH_TRAIN = r'C:\Users\fabia\OneDrive\Escritorio\UNI\FME2025\data\train.csv'
PATH_TEST = r'C:\Users\fabia\OneDrive\Escritorio\UNI\FME2025\data\test.csv'
N_PCA_COMPONENTS = 20
N_SPLITS = 5
# Pesos fijos. 60% a tu mejor modelo, 40% al subcampeón.
WEIGHT_MODEL_1 = 0.6
WEIGHT_MODEL_2 = 0.4


# --- PASO 2: INGENIERÍA DE CARACTERÍSTICAS (V14 - con LogFeatures) ---
# =======================================================
print(f"\n--- Iniciando [Paso 2]: Ingeniería de Características (V14) ---")
# (El código de 2.1 a 2.8 es idéntico a tu V14)
# --- 2.1: Carga de Datos ---
try:
    df_train_raw = pd.read_csv(PATH_TRAIN, sep=';')
    df_test_raw = pd.read_csv(PATH_TEST, sep=';')
    print(f"train.csv cargado. Shape: {df_train_raw.shape}")
    print(f"test.csv cargado. Shape: {df_test_raw.shape}")
except FileNotFoundError:
    print(f"Error: No se encontraron los archivos.")
    exit()
# --- 2.2: Agregación ---
static_cols = ['ID', 'id_season', 'family', 'category', 'fabric', 'color_name', 'image_embedding', 'length_type', 'silhouette_type', 'waist_type', 'sleeve_length_type', 'ocassion', 'phase_in', 'life_cycle_length', 'num_stores', 'num_sizes', 'price', 'Production']
static_cols = [col for col in static_cols if col in df_train_raw.columns]
df_agg = df_train_raw.groupby('ID').agg(Total_Demand=('weekly_demand', 'sum')).reset_index()
df_static = df_train_raw[static_cols].drop_duplicates(subset='ID').set_index('ID')
df_train = df_static.join(df_agg.set_index('ID')).reset_index()
df_train['source'] = 'train'
df_test_raw['source'] = 'test'
print(f"Train agregado. Shape: {df_train.shape}")
# --- 2.3: Combinación ---
df_combined = pd.concat([df_train.drop(['Total_Demand', 'Production'], axis=1, errors='ignore'), df_test_raw], ignore_index=True)
print(f"Shape combinado: {df_combined.shape}")
# --- 2.4: Fechas ---
try:
    df_combined['phase_in_dt'] = pd.to_datetime(df_combined['phase_in'], dayfirst=True)
    df_combined['phase_in_month'] = df_combined['phase_in_dt'].dt.month
    df_combined['phase_in_week'] = df_combined['phase_in_dt'].dt.isocalendar().week
    df_combined['phase_in_month'] = df_combined['phase_in_month'].fillna(0).astype(int)
    df_combined['phase_in_week'] = df_combined['phase_in_week'].fillna(0).astype(int)
except Exception as e:
    df_combined['phase_in_month'] = 0; df_combined['phase_in_week'] = 0
# --- 2.5: Categóricas ---
categorical_features = ['family', 'category', 'fabric', 'color_name', 'length_type', 'silhouette_type', 'waist_type', 'sleeve_length_type', 'ocassion', 'phase_in_month']
categorical_features = [col for col in categorical_features if col in df_combined.columns]
for col in categorical_features: df_combined[col] = df_combined[col].fillna("Desconocido")
# --- 2.6: Embeddings (PCA) ---
print(f"Procesando Embeddings con PCA...")
def parse_embedding(embed_str):
    if pd.isna(embed_str): return None
    try: return ast.literal_eval(embed_str)
    except (ValueError, SyntaxError):
        try: return [float(x) for x in embed_str.strip('[]').split()]
        except: return None
embedding_list = df_combined['image_embedding'].apply(parse_embedding).tolist()
dim = 0;
for emb in embedding_list:
    if emb is not None: dim = len(emb); break
if dim > 0:
    zero_vector = [0.0] * dim
    embeddings_matrix = [emb if emb is not None else zero_vector for emb in embedding_list]
    embeddings_matrix = np.array(embeddings_matrix)
    scaler = StandardScaler(); embeddings_scaled = scaler.fit_transform(embeddings_matrix)
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42); embeddings_pca = pca.fit_transform(embeddings_scaled)
    pca_cols = [f'pca_{i}' for i in range(N_PCA_COMPONENTS)]; df_pca = pd.DataFrame(embeddings_pca, columns=pca_cols, index=df_combined.index)
    df_combined = pd.concat([df_combined, df_pca], axis=1)
else:
    print("No se pudieron parsear los embeddings.")
# --- 2.7: Features Agregadas (V2 - 70 features) ---
print("Creando features agregadas (V2)...")
agg_cat_cols = ['family', 'category', 'ocassion', 'silhouette_type']
agg_cat_cols = [col for col in agg_cat_cols if col in df_combined.columns]
agg_num_cols = ['num_stores', 'price', 'life_cycle_length']
agg_num_cols = [col for col in agg_num_cols if col in df_combined.columns]
new_agg_features = []
for cat_col in agg_cat_cols:
    for num_col in agg_num_cols:
        group_stats = df_combined.groupby(cat_col)[num_col].agg(['mean', 'median'])
        mean_col = f'{num_col}_mean_by_{cat_col}'; median_col = f'{num_col}_median_by_{cat_col}'; ratio_mean_col = f'{num_col}_ratio_to_mean_{cat_col}'; ratio_median_col = f'{num_col}_ratio_to_median_{cat_col}' 
        new_agg_features.extend([mean_col, median_col, ratio_mean_col, ratio_median_col])
        df_combined[mean_col] = df_combined[cat_col].map(group_stats['mean']); df_combined[median_col] = df_combined[cat_col].map(group_stats['median'])
        df_combined[mean_col] = df_combined[mean_col].fillna(df_combined[num_col].mean()); df_combined[median_col] = df_combined[median_col].fillna(df_combined[num_col].median())
        df_combined[ratio_mean_col] = df_combined[num_col] / (df_combined[mean_col] + 1e-6); df_combined[ratio_median_col] = df_combined[num_col] / (df_combined[median_col] + 1e-6)
print("Nuevas features V2 creadas.")
# --- 2.8: Verificación de Duplicados ---
print("Verificando y eliminando columnas duplicadas...")
df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
print("Columnas duplicadas eliminadas.")
# --- 2.9: Normalización de Features Numéricas (V14) ---
print("Normalizando (log1p) features numéricas sesgadas...")
num_features_to_log = ['num_stores', 'num_sizes', 'price', 'life_cycle_length', 'phase_in_week']
agg_features_to_log = [col for col in new_agg_features if 'ratio' not in col]
num_features_to_log.extend(agg_features_to_log)
for col in num_features_to_log:
    if col in df_combined.columns:
        df_combined[col] = np.log1p(df_combined[col])
print("Features numéricas normalizadas.")
# --- 2.10: Creación de Datasets Finales ---
print("Creando datasets finales...")
features = ['num_stores', 'num_sizes', 'price', 'life_cycle_length', 'phase_in_week']
features += categorical_features
if 'pca_cols' in locals(): features += pca_cols
features += new_agg_features
features = list(dict.fromkeys(features)); features = [col for col in features if col in df_combined.columns]
categorical_features = [col for col in categorical_features if col in features]
df_train_final = df_combined[df_combined['source'] == 'train'].reset_index(drop=True)
df_test_final = df_combined[df_combined['source'] == 'test'].reset_index(drop=True)
X_train = df_train_final[features]; X_test = df_test_final[features]
y_train = np.log1p(df_train['Total_Demand']); test_ids = df_test_raw['ID'] 
print("\n--- [Paso 2] Completado! ---")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")


# --- PASO 3: FUNCIÓN DE ENTRENAMIENTO "FINALISTA" ---
# ===================================================
def train_finalist_model(alpha, depth, l2_reg, lr, model_name=""):
    """
    Entrena un modelo "Finalista" (train-val-retrain) y devuelve las predicciones del test.
    """
    print(f"\n--- Iniciando Entrenamiento {model_name}: alpha={alpha}, depth={depth}, l2={l2_reg}, lr={lr} ---")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    # 1. Definir el split 80/20
    train_index, val_index = list(tscv.split(X_train))[-1]
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    train_pool = Pool(data=X_train_fold, label=y_train_fold, cat_features=categorical_features)
    val_pool = Pool(data=X_val_fold, label=y_val_fold, cat_features=categorical_features)

    loss_function = f'Quantile:alpha={alpha}'

    # 2. Entrenar el modelo de validación
    model = CatBoostRegressor(
        iterations=10000, learning_rate=lr, depth=depth, l2_leaf_reg=l2_reg,
        loss_function=loss_function, eval_metric=loss_function,
        random_seed=42, verbose=500, early_stopping_rounds=400
    )
    model.fit(train_pool, eval_set=val_pool)
    
    best_iteration = model.get_best_iteration()
    print(f"  {model_name} - Mejor iteración: {best_iteration}")

    # 3. Re-entrenar el modelo FINAL
    full_train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
    final_model = CatBoostRegressor(
        iterations=best_iteration,  # Usamos la mejor iteración
        learning_rate=lr, depth=depth, l2_leaf_reg=l2_reg,
        loss_function=loss_function, eval_metric=loss_function,
        random_seed=42, verbose=0
    )
    final_model.fit(full_train_pool)
    print(f"  ¡Entrenamiento {model_name} completado y re-entrenado!")
    
    # 4. Devolver las predicciones finales
    return final_model.predict(X_test)

# --- PASO 4: ENTRENAR LOS DOS MODELOS "FINALISTAS" ---
# ====================================================

# 1. Entrenar Modelo A (Tu V17 Ganador)
preds_model_1 = train_finalist_model(
    alpha=0.78,
    depth=7,
    l2_reg=5,
    lr=0.01,
    model_name="V17-A (Paciente)"
)

# 2. Entrenar Modelo B (Tu V6 Ganador)
preds_model_2 = train_finalist_model(
    alpha=0.75,
    depth=7,
    l2_reg=5,
    lr=0.03,
    model_name="V17-B (Equilibrado)"
)

# --- PASO 5: CREACIÓN DE LA ENTREGA (Ensemble Ponderado) ---
# ==========================================================
print(f"\n--- Iniciando [Paso 5]: Generación de Submission (Ensemble de Finalistas) ---")

# Damos un peso fijo, 60% a tu mejor modelo (el V10/V17-A)
w1 = WEIGHT_MODEL_1 # 0.6
w2 = WEIGHT_MODEL_2 # 0.4
print(f"Peso Modelo 1 (alpha=0.78): {w1:.2f}")
print(f"Peso Modelo 2 (alpha=0.75): {w2:.2f}")

final_preds_log = (preds_model_1 * w1) + (preds_model_2 * w2)

final_preds_real = np.expm1(final_preds_log)
final_preds_safe = final_preds_real
final_preds_safe[final_preds_safe < 0] = 0

print(f"Predicción media (Ensemble): {final_preds_safe.mean():.2f} unidades")

submission = pd.DataFrame({
    'ID': test_ids,
    'TARGET': final_preds_safe
})
submission['ID'] = submission['ID'].astype(str)
submission['TARGET'] = submission['TARGET'].round(0).astype(int)

submission_filename = f'submission_catboost_V18_EnsembleFinalists.csv'
submission.to_csv(submission_filename, index=False)

print("\n--- ¡PIPELINE V18 (ENSEMBLE DE FINALISTAS) COMPLETADO! ---")
print(f"Archivo '{submission_filename}' creado con éxito.")
print("Esta es la estrategia de *blending* más robusta que hemos construido.")
print(submission.head())