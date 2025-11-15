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
print("Librerías cargadas. Iniciando pipeline completo...")


# --- PASO 1: CONFIGURACIÓN GLOBAL ---
# =====================================

# --- Configuración de Rutas
PATH_TRAIN = r'C:\Users\fabia\OneDrive\Escritorio\UNI\FME2025\data\train.csv'
PATH_TEST = r'C:\Users\fabia\OneDrive\Escritorio\UNI\FME2025\data\test.csv'

# --- Configuración de Features
N_PCA_COMPONENTS = 20 # Componentes para PCA de embeddings

# --- Configuración de Modelo
N_SPLITS = 5 # Número de Folds para TimeSeriesSplit
SAFETY_FACTOR = 1.15 # Factor de seguridad para evitar ventas perdidas (1.15 = 15% de colchón)
LOST_SALES_PENALTY = 1.5 # Penalización para la métrica de monitoreo


# --- PASO 2: INGENIERÍA DE CARACTERÍSTICAS (SCRIPT 1) ---
# =======================================================

print(f"\n--- Iniciando [Paso 2]: Ingeniería de Características ---")

# --- 2.1: Carga de Datos ---
try:
    df_train_raw = pd.read_csv(PATH_TRAIN, sep=';')
    df_test_raw = pd.read_csv(PATH_TEST, sep=';')
    print(f"train.csv cargado. Shape: {df_train_raw.shape}")
    print(f"test.csv cargado. Shape: {df_test_raw.shape}")
except FileNotFoundError:
    print(f"Error: No se encontraron los archivos en las rutas especificadas.")
    exit()

# --- 2.2: Agregación de Datos de Entrenamiento ---
print("Agregando datos de entrenamiento (train)...")
static_cols = [
    'ID', 'id_season', 'family', 'category', 'fabric', 'color_name',
    'image_embedding', 'length_type', 'silhouette_type', 'waist_type', 
    'sleeve_length_type', 'ocassion', 'phase_in', 'life_cycle_length', 
    'num_stores', 'num_sizes', 'price', 'Production'
]
static_cols = [col for col in static_cols if col in df_train_raw.columns]

df_agg = df_train_raw.groupby('ID').agg(
    Total_Demand=('weekly_demand', 'sum')
).reset_index()

df_static = df_train_raw[static_cols].drop_duplicates(subset='ID').set_index('ID')
df_train = df_static.join(df_agg.set_index('ID')).reset_index()

df_train['source'] = 'train'
df_test_raw['source'] = 'test'
print(f"Train agregado. Shape: {df_train.shape}")

# --- 2.3: Combinación Train/Test para Procesamiento ---
print("Combinando train y test para ingeniería de características...")
df_combined = pd.concat([
    df_train.drop(['Total_Demand', 'Production'], axis=1, errors='ignore'), 
    df_test_raw
], ignore_index=True)
print(f"Shape combinado: {df_combined.shape}")

# --- 2.4: Creación de Features de Fecha ---
print("Creando features de fecha...")
try:
    df_combined['phase_in_dt'] = pd.to_datetime(df_combined['phase_in'], dayfirst=True)
    df_combined['phase_in_month'] = df_combined['phase_in_dt'].dt.month
    df_combined['phase_in_week'] = df_combined['phase_in_dt'].dt.isocalendar().week
    df_combined['phase_in_month'] = df_combined['phase_in_month'].fillna(0).astype(int)
    df_combined['phase_in_week'] = df_combined['phase_in_week'].fillna(0).astype(int)
except Exception as e:
    print(f"Error procesando fechas: {e}. Se rellenarán con 0.")
    df_combined['phase_in_month'] = 0
    df_combined['phase_in_week'] = 0

# --- 2.5: Limpieza de Categóricas ---
print("Limpiando features categóricas...")
categorical_features = [
    'family', 'category', 'fabric', 'color_name', 'length_type', 
    'silhouette_type', 'waist_type', 'sleeve_length_type', 'ocassion',
    'phase_in_month' # La tratamos como categórica
]
categorical_features = [col for col in categorical_features if col in df_combined.columns]
for col in categorical_features:
    df_combined[col] = df_combined[col].fillna("Desconocido")

# --- 2.6: Embeddings (PCA) ---
print(f"Procesando Embeddings con PCA (n_components={N_PCA_COMPONENTS})...")

def parse_embedding(embed_str):
    if pd.isna(embed_str): return None
    try: return ast.literal_eval(embed_str)
    except (ValueError, SyntaxError):
        try: return [float(x) for x in embed_str.strip('[]').split()]
        except: return None

embedding_list = df_combined['image_embedding'].apply(parse_embedding).tolist()

dim = 0
for emb in embedding_list:
    if emb is not None:
        dim = len(emb)
        break

if dim > 0:
    print(f"Dimensión de Embeddings detectada: {dim}")
    zero_vector = [0.0] * dim
    embeddings_matrix = [emb if emb is not None else zero_vector for emb in embedding_list]
    embeddings_matrix = np.array(embeddings_matrix)

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_matrix)

    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_scaled)

    pca_cols = [f'pca_{i}' for i in range(N_PCA_COMPONENTS)]
    df_pca = pd.DataFrame(embeddings_pca, columns=pca_cols, index=df_combined.index)

    df_combined = pd.concat([df_combined, df_pca], axis=1)
    print("PCA completado. Features añadidas.")
else:
    print("No se pudo detectar la dimensión de los embeddings. Saltando PCA.")

# --- 2.7: Creación de Datasets Finales ---
print("Creando datasets finales...")
features = [
    'num_stores', 'num_sizes', 'price', 'life_cycle_length', 'phase_in_week'
]
features += categorical_features
if 'pca_cols' in locals():
    features += pca_cols

features = [col for col in features if col in df_combined.columns]
categorical_features = [col for col in categorical_features if col in features]

# Separamos de nuevo en Train y Test
df_train_final = df_combined[df_combined['source'] == 'train'].reset_index(drop=True)
df_test_final = df_combined[df_combined['source'] == 'test'].reset_index(drop=True)

# Creamos nuestros sets X, y
X_train = df_train_final[features]
X_test = df_test_final[features]

# Creamos el objetivo (y) usando el df_train original
# Aplicamos la transformación logarítmica que vimos en el EDA
y_train = np.log1p(df_train['Total_Demand'])

# Guardamos los IDs del test para la submission
test_ids = df_test_raw['ID'] 

print("\n--- [Paso 2] Completado! ---")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print("\nFeatures categóricas (para CatBoost):")
print(categorical_features)


# --- PASO 3: DEFINICIÓN DE MÉTRICA DE NEGOCIO ---
# ================================================
# (Para monitorizar el "dolor de negocio" en la validación)

class AsymmetricBusinessMetric:
    def __init__(self, penalty=LOST_SALES_PENALTY):
        self.penalty = penalty

    def is_max_optimal(self):
        return False  # Queremos minimizar este error

    def evaluate(self, approxes, target, weight):
        pred_real = np.expm1(approxes[0]) # Revertir log
        true_real = np.expm1(target)      # Revertir log
        diff = true_real - pred_real
        errors = np.where(diff > 0, diff * self.penalty, np.abs(diff))
        error_sum = np.sum(errors)
        weight_sum = np.sum(weight) if weight is not None else len(errors)
        return error_sum, weight_sum

    def get_final_error(self, error_sum, weight_sum):
        return error_sum / weight_sum


# --- PASO 4: ENTRENAMIENTO DEL MODELO (SCRIPT 2) ---
# ===================================================

print(f"\n--- Iniciando [Paso 4]: Entrenamiento de CatBoost ---")
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# Guardaremos las predicciones del test (OOF) y la importancia de features
oof_predictions = np.zeros(X_train.shape[0])
test_predictions = np.zeros(X_test.shape[0])

test_pool = Pool(data=X_test, cat_features=categorical_features)

for fold, (train_index, val_index) in enumerate(tscv.split(X_train)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    train_pool = Pool(data=X_train_fold, 
                      label=y_train_fold, 
                      cat_features=categorical_features)
    val_pool = Pool(data=X_val_fold, 
                    label=y_val_fold, 
                    cat_features=categorical_features)

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function='RMSE',       # Optimizamos RMSE en log
        eval_metric=AsymmetricBusinessMetric(), # Monitorizamos "dolor de negocio"
        random_seed=42,
        verbose=200,
        early_stopping_rounds=100
    )

    model.fit(
        train_pool,
        eval_set=val_pool
    )

    val_preds = model.predict(X_val_fold)
    oof_predictions[val_index] = val_preds
    
    test_predictions += model.predict(test_pool) / N_SPLITS

    print(f"Fold {fold+1} RMSE (en log): {np.sqrt(mean_squared_error(y_val_fold, val_preds)):.4f}")

# --- Evaluación OOF
oof_preds_real = np.expm1(oof_predictions)
y_train_real = np.expm1(y_train)
oof_business_error, _ = AsymmetricBusinessMetric().evaluate(oof_predictions, y_train, None)
oof_rmse_real = np.sqrt(mean_squared_error(y_train_real, oof_preds_real))

print("\n--- Evaluación OOF (Fuera de Muestra) Completada ---")
print(f"Score de Negocio (OOF): {oof_business_error / len(y_train):.4f}")
print(f"RMSE (OOF) en Unidades Reales: {oof_rmse_real:.4f}")
# Métricas tipo 'accuracy' (usamos R2) en log y en escala real
r2_log = r2_score(y_train, oof_predictions)
r2_real = r2_score(y_train_real, oof_preds_real)
accuracy = max(0.0, min(1.0, r2_real))  # Limitar entre 0 y 1
print(f"R2 (espacio log): {r2_log:.4f}")
print(f"R2 (escala real): {r2_real:.4f}")
print(f"Accuracy (0-1 basado en R2 real): {accuracy:.4f}")


# --- PASO 5: CREACIÓN DE LA ENTREGA (SUBMISSION) ---
# ==================================================
print(f"\n--- Iniciando [Paso 5]: Generación de Submission ---")

# 1. Post-Procesamiento de Predicciones del Test
final_preds_log = test_predictions
final_preds_real = np.expm1(final_preds_log)

# 2. ¡¡Aplicamos nuestro Factor de Seguridad!!
final_preds_safe = final_preds_real * SAFETY_FACTOR

# 3. Aseguramos que no haya predicciones negativas
final_preds_safe[final_preds_safe < 0] = 0

print(f"Predicción media (sin seguridad): {final_preds_real.mean():.2f} unidades")
print(f"Predicción media (con {SAFETY_FACTOR*100}% seguridad): {final_preds_safe.mean():.2f} unidades")

# 4. Crear el DataFrame de entrega
submission = pd.DataFrame({
    'ID': test_ids,
    'TARGET': final_preds_safe
})

# El datathon pide que el ID sea de tipo string.
submission['ID'] = submission['ID'].astype(str)

# Redondeamos a unidades enteras
submission['TARGET'] = submission['TARGET'].round(0).astype(int)

# 5. Guardar el archivo
submission.to_csv('submission_catboost.csv', index=False)

print("\n--- ¡PIPELINE COMPLETADO! ---")
print("Archivo 'submission_catboost.csv' creado con éxito.")
print(submission.head())