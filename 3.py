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
print("Librerías cargadas. Iniciando pipeline FINAL (Submission #10 - Aprendiz Paciente)...")


# --- PASO 1: CONFIGURACIÓN GLOBAL ---
# =====================================
PATH_TRAIN = r'C:\Users\fabia\OneDrive\Escritorio\UNI\FME2025\data\train.csv'
PATH_TEST = r'C:\Users\fabia\OneDrive\Escritorio\UNI\FME2025\data\test.csv'
N_PCA_COMPONENTS = 20
N_SPLITS = 5

# --- ¡CAMBIO DE TUNING (V10)! ---
QUANTILE_ALPHA = 0.78 # Mantenemos el Alpha ganador
MODEL_DEPTH = 7       # Mantenemos el Depth ganador
MODEL_L2_REG = 5      # Mantenemos el L2 ganador
MODEL_LR = 0.01       # <-- ¡MÁS BAJO!


# --- PASO 2: INGENIERÍA DE CARACTERÍSTICAS (Versión V2 - 70 features) ---
# =======================================================
print(f"\n--- Iniciando [Paso 2]: Ingeniería de Características (Volvemos a V2) ---")

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

# --- 2.9: Creación de Datasets Finales ---
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


# --- PASO 3: ENTRENAMIENTO DEL MODELO (FINAL - V10) ---
# ===================================================
print(f"\n--- Iniciando [Paso 3]: Entrenamiento de CatBoost (V10 - Paciente) ---")
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

oof_predictions = np.zeros(X_train.shape[0])
test_predictions = np.zeros(X_test.shape[0])
all_feature_importance = pd.DataFrame() 

test_pool = Pool(data=X_test, cat_features=categorical_features)
loss_function = f'Quantile:alpha={QUANTILE_ALPHA}' # <-- Usa 0.75

for fold, (train_index, val_index) in enumerate(tscv.split(X_train)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    train_pool = Pool(data=X_train_fold, label=y_train_fold, cat_features=categorical_features)
    val_pool = Pool(data=X_val_fold, label=y_val_fold, cat_features=categorical_features)

    # --- ¡Nuevos parámetros de tuning! ---
    model = CatBoostRegressor(
        iterations=10000,         # <-- Más iteraciones
        learning_rate=MODEL_LR,   # <-- 0.01 (más lento)
        depth=MODEL_DEPTH,        # <-- 7
        l2_leaf_reg=MODEL_L2_REG, # <-- 5
        loss_function=loss_function,
        eval_metric=loss_function,
        random_seed=42,
        verbose=500,              # Imprime menos
        early_stopping_rounds=400 # <-- Más paciencia
    )

    model.fit(
        train_pool,
        eval_set=val_pool
    )

    val_preds = model.predict(X_val_fold)
    oof_predictions[val_index] = val_preds
    test_predictions += model.predict(test_pool) / N_SPLITS

    best_score = model.get_best_score()['validation'][loss_function]
    print(f"Fold {fold+1} Quantile Loss (alpha={QUANTILE_ALPHA}): {best_score:.4f}")

    fold_importance_df = pd.DataFrame({
        'feature': model.feature_names_,
        'importance': model.get_feature_importance(),
        'fold': fold + 1
    })
    all_feature_importance = pd.concat([all_feature_importance, fold_importance_df], axis=0)

# --- Evaluación OOF
oof_preds_real = np.expm1(oof_predictions)
y_train_real = np.expm1(y_train)
oof_rmse_real = np.sqrt(mean_squared_error(y_train_real, oof_preds_real))
oof_r2_real = r2_score(y_train_real, oof_preds_real)

print("\n--- Evaluación OOF (Fuera de Muestra) Completada ---")
print(f"RMSE (OOF) en Unidades Reales: {oof_rmse_real:.4f}")
print(f"R2 (escala real): {oof_r2_real:.4f}  <--- ¡COMPARA ESTE NÚMERO CON 0.5712!")

avg_importance = all_feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
print("\n--- Importancia Media de Features (Todos los Folds) ---")
print(avg_importance.head(20))


# --- PASO 4: CREACIÓN DE LA ENTREGA (SUBMISSION) ---
# ==================================================
print(f"\n--- Iniciando [Paso 4]: Generación de Submission ---")

final_preds_log = test_predictions
final_preds_real = np.expm1(final_preds_log)
final_preds_safe = final_preds_real
final_preds_safe[final_preds_safe < 0] = 0

print(f"Predicción media (con Quantile Loss alpha={QUANTILE_ALPHA}): {final_preds_safe.mean():.2f} unidades")

submission = pd.DataFrame({
    'ID': test_ids,
    'TARGET': final_preds_safe
})
submission['ID'] = submission['ID'].astype(str)
submission['TARGET'] = submission['TARGET'].round(0).astype(int)

submission_filename = f'submission_catboost_V10_PatientLearner.csv'
submission.to_csv(submission_filename, index=False)

print("\n--- ¡PIPELINE V10 COMPLETADO! ---")
print(f"Archivo '{submission_filename}' creado con éxito.")
print(f"Compara tu nuevo 'R2 (escala real)' con tu récord de 0.5712.")
print(submission.head())