"""
Single-script training pipeline for FME2025.

What it does on execution:
- Loads data from `data/train.csv` and `data/test.csv`
- Builds features (embeddings PCA, categorical, aggregates, logs)
- Trains two CatBoost finalist models and re-trains on full data
- Blends predictions and writes `submission.csv` at the repository root
- Saves artifacts to `outputs/` (models, scaler, PCA, features)

How to run:
    python model.py
"""
import os
from pathlib import Path
import json
import warnings
import ast
import joblib

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# Default paths
PATH_TRAIN = DATA_DIR / 'train.csv'
PATH_TEST = DATA_DIR / 'test.csv'

# Global configuration
N_PCA_COMPONENTS = 20
N_SPLITS = 5
WEIGHT_MODEL_1 = 0.6
WEIGHT_MODEL_2 = 0.4


def parse_embedding(embed_str):
    """Parse an embedding stored as a string representation of a list (or None).
    Returns None when parsing fails or the embedding is missing.
    """
    if pd.isna(embed_str):
        return None
    try:
        return ast.literal_eval(embed_str)
    except (ValueError, SyntaxError):
        try:
            return [float(x) for x in embed_str.strip('[]').split()]
        except Exception:
            return None


def load_data(train_path=PATH_TRAIN, test_path=PATH_TEST):
    """Load train and test CSVs with sane defaults (sep=';').
    Raises FileNotFoundError if not found.
    """
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Train or test files not found at {train_path} and {test_path}")
    df_train_raw = pd.read_csv(train_path, sep=';')
    df_test_raw = pd.read_csv(test_path, sep=';')
    return df_train_raw, df_test_raw


def feature_engineering(df_train_raw, df_test_raw, n_pca_components=N_PCA_COMPONENTS):
    """Perform the same feature engineering pipeline used in original model.
    Returns: X_train, y_train, X_test, test_ids, metadata dict (scaler, pca, features, categorical features)
    """
    # 1) Aggregate training weekly data to product level
    static_cols = [
        'ID', 'id_season', 'family', 'category', 'fabric', 'color_name', 'image_embedding',
        'length_type', 'silhouette_type', 'waist_type', 'sleeve_length_type', 'ocassion', 'phase_in',
        'life_cycle_length', 'num_stores', 'num_sizes', 'price', 'Production'
    ]
    static_cols = [col for col in static_cols if col in df_train_raw.columns]
    df_agg = df_train_raw.groupby('ID').agg(Total_Demand=('weekly_demand', 'sum')).reset_index()
    df_static = df_train_raw[static_cols].drop_duplicates(subset='ID').set_index('ID')
    df_train = df_static.join(df_agg.set_index('ID')).reset_index()
    df_train['source'] = 'train'
    df_test_raw['source'] = 'test'

    # 2) Combine and create features
    df_combined = pd.concat([df_train.drop(['Total_Demand', 'Production'], axis=1, errors='ignore'),
                             df_test_raw], ignore_index=True)

    # Dates
    try:
        df_combined['phase_in_dt'] = pd.to_datetime(df_combined['phase_in'], dayfirst=True)
        df_combined['phase_in_month'] = df_combined['phase_in_dt'].dt.month.fillna(0).astype(int)
        df_combined['phase_in_week'] = df_combined['phase_in_dt'].dt.isocalendar().week.fillna(0).astype(int)
    except Exception:
        df_combined['phase_in_month'] = 0
        df_combined['phase_in_week'] = 0

    # Categorical fill
    categorical_features = [
        'family', 'category', 'fabric', 'color_name', 'length_type', 'silhouette_type',
        'waist_type', 'sleeve_length_type', 'ocassion', 'phase_in_month'
    ]
    categorical_features = [col for col in categorical_features if col in df_combined.columns]
    for col in categorical_features:
        df_combined[col] = df_combined[col].fillna('Unknown')

    # Embeddings (parse and PCA)
    embedding_list = df_combined['image_embedding'].apply(parse_embedding).tolist() if 'image_embedding' in df_combined.columns else []
    # Determine the embedding dimension (pad variable-length embeddings to max dim)
    dims = [len(emb) for emb in embedding_list if emb is not None]
    dim = int(max(dims)) if dims else 0
    scaler = None
    pca = None
    pca_cols = []
    if dim > 0:
        zero_vector = [0.0] * dim
        # pad shorter embeddings to the max dimension
        embeddings_matrix = []
        for emb in embedding_list:
            if emb is None:
                embeddings_matrix.append(zero_vector)
            elif len(emb) < dim:
                embeddings_matrix.append(list(emb) + [0.0] * (dim - len(emb)))
            else:
                embeddings_matrix.append(emb[:dim])
        embeddings_matrix = np.array(embeddings_matrix)
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_matrix)
        pca = PCA(n_components=n_pca_components, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings_scaled)
        pca_cols = [f'pca_{i}' for i in range(n_pca_components)]
        df_pca = pd.DataFrame(embeddings_pca, columns=pca_cols, index=df_combined.index)
        df_combined = pd.concat([df_combined, df_pca], axis=1)

    # Aggregation features
    agg_cat_cols = ['family', 'category', 'ocassion', 'silhouette_type']
    agg_cat_cols = [col for col in agg_cat_cols if col in df_combined.columns]
    agg_num_cols = ['num_stores', 'price', 'life_cycle_length']
    agg_num_cols = [col for col in agg_num_cols if col in df_combined.columns]
    new_agg_features = []
    for cat_col in agg_cat_cols:
        for num_col in agg_num_cols:
            group_stats = df_combined.groupby(cat_col)[num_col].agg(['mean', 'median'])
            mean_col = f'{num_col}_mean_by_{cat_col}'
            median_col = f'{num_col}_median_by_{cat_col}'
            ratio_mean_col = f'{num_col}_ratio_to_mean_{cat_col}'
            ratio_median_col = f'{num_col}_ratio_to_median_{cat_col}'
            new_agg_features.extend([mean_col, median_col, ratio_mean_col, ratio_median_col])
            df_combined[mean_col] = df_combined[cat_col].map(group_stats['mean'])
            df_combined[median_col] = df_combined[cat_col].map(group_stats['median'])
            df_combined[mean_col] = df_combined[mean_col].fillna(df_combined[num_col].mean())
            df_combined[median_col] = df_combined[median_col].fillna(df_combined[num_col].median())
            df_combined[ratio_mean_col] = df_combined[num_col] / (df_combined[mean_col] + 1e-6)
            df_combined[ratio_median_col] = df_combined[num_col] / (df_combined[median_col] + 1e-6)

    # Remove duplicated columns if any
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]

    # Log normalize skewed numeric features
    num_features_to_log = ['num_stores', 'num_sizes', 'price', 'life_cycle_length', 'phase_in_week']
    agg_features_to_log = [col for col in new_agg_features if 'ratio' not in col]
    num_features_to_log.extend(agg_features_to_log)
    for col in num_features_to_log:
        if col in df_combined.columns:
            # Replace negative or missing values with 0 before log
            df_combined[col] = df_combined[col].fillna(0)
            df_combined[col] = np.log1p(df_combined[col].clip(lower=0))

    # Final features
    features = ['num_stores', 'num_sizes', 'price', 'life_cycle_length', 'phase_in_week']
    features = [f for f in features if f in df_combined.columns]
    features += categorical_features
    if pca_cols:
        features += pca_cols
    features += new_agg_features
    # Deduplicate while keeping order
    features = list(dict.fromkeys(features))
    features = [col for col in features if col in df_combined.columns]
    categorical_features = [col for col in categorical_features if col in features]

    df_train_final = df_combined[df_combined['source'] == 'train'].reset_index(drop=True)
    df_test_final = df_combined[df_combined['source'] == 'test'].reset_index(drop=True)
    X_train = df_train_final[features].copy()
    X_test = df_test_final[features].copy()
    y_train = np.log1p(df_train['Total_Demand']).copy()
    test_ids = df_test_raw['ID'] if 'ID' in df_test_raw.columns else None

    metadata = {
        'features': features,
        'categorical_features': categorical_features,
        'pca_cols': pca_cols,
        'scaler': scaler,
        'pca': pca,
        'n_pca_components': n_pca_components
    }
    return X_train, y_train, X_test, test_ids, metadata


def train_finalist_model(X_train, y_train, X_test, categorical_features, alpha, depth, l2_reg, lr, n_splits=N_SPLITS, model_name=''):
    """Train model with a train/validation split using TimeSeriesSplit, then re-train on full data and return predictions + final model object."""
    print(f"Starting training {model_name}: alpha={alpha}, depth={depth}, l2={l2_reg}, lr={lr}")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    # Use the last split for an 80/20 train/val
    train_index, val_index = list(tscv.split(X_train))[-1]
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    train_pool = Pool(data=X_train_fold, label=y_train_fold, cat_features=categorical_features)
    val_pool = Pool(data=X_val_fold, label=y_val_fold, cat_features=categorical_features)

    loss_function = f'Quantile:alpha={alpha}'

    model = CatBoostRegressor(
        iterations=10000,
        learning_rate=lr,
        depth=depth,
        l2_leaf_reg=l2_reg,
        loss_function=loss_function,
        eval_metric=loss_function,
        random_seed=42,
        verbose=500,
        early_stopping_rounds=400
    )
    model.fit(train_pool, eval_set=val_pool)
    best_iteration = model.get_best_iteration()
    print(f"  {model_name} - Best iteration: {best_iteration}")

    # Re-train on full training data
    full_train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
    final_model = CatBoostRegressor(
        iterations=best_iteration,
        learning_rate=lr,
        depth=depth,
        l2_leaf_reg=l2_reg,
        loss_function=loss_function,
        eval_metric=loss_function,
        random_seed=42,
        verbose=0
    )
    final_model.fit(full_train_pool)
    print(f"Training {model_name} finished (re-trained on full data).")

    preds = final_model.predict(X_test)
    return preds, final_model


def save_artifacts(models, artifacts, prefix='finalist'):
    """Save models and data preprocessing artifacts to OUTPUT_DIR.
    models: dict with name->CatBoost model
    artifacts: dict with 'scaler', 'pca', 'features', 'categorical_features'
    """
    for name, model in models.items():
        model_path = OUTPUT_DIR / f'{prefix}_{name}.cbm'
        model.save_model(str(model_path))
        print(f"Saved model: {model_path}")
    # Save scaler/pca/feature lists via joblib
    if artifacts.get('scaler') is not None:
        joblib.dump(artifacts['scaler'], OUTPUT_DIR / f'{prefix}_scaler.joblib')
    if artifacts.get('pca') is not None:
        joblib.dump(artifacts['pca'], OUTPUT_DIR / f'{prefix}_pca.joblib')
    joblib.dump(artifacts['features'], OUTPUT_DIR / f'{prefix}_features.joblib')
    joblib.dump(artifacts['categorical_features'], OUTPUT_DIR / f'{prefix}_categorical_features.joblib')
    print(f"Saved artifacts to {OUTPUT_DIR}")


def run_pipeline(train_path=PATH_TRAIN, test_path=PATH_TEST):
    print("Starting training pipeline (refactored)")
    try:
        df_train_raw, df_test_raw = load_data(train_path, test_path)
    except FileNotFoundError as e:
        print(e)
        return

    X_train, y_train, X_test, test_ids, metadata = feature_engineering(df_train_raw, df_test_raw)
    print(f"Prepared data. X_train shape={X_train.shape}, X_test shape={X_test.shape}")

    # Train two finalist models and ensemble
    preds_1, model_1 = train_finalist_model(
        X_train, y_train, X_test, metadata['categorical_features'], alpha=0.78, depth=7, l2_reg=5, lr=0.01, model_name='V17-A'
    )
    preds_2, model_2 = train_finalist_model(
        X_train, y_train, X_test, metadata['categorical_features'], alpha=0.75, depth=7, l2_reg=5, lr=0.03, model_name='V17-B'
    )

    # Save models and artifacts
    models = {'V17-A': model_1, 'V17-B': model_2}
    artifacts = {
        'scaler': metadata.get('scaler'),
        'pca': metadata.get('pca'),
        'features': metadata.get('features'),
        'categorical_features': metadata.get('categorical_features')
    }
    save_artifacts(models, artifacts, prefix='finalist')

    # Ensemble weighted
    final_preds_log = (preds_1 * WEIGHT_MODEL_1) + (preds_2 * WEIGHT_MODEL_2)
    final_preds_real = np.expm1(final_preds_log)
    final_preds_safe = final_preds_real.copy()
    final_preds_safe[final_preds_safe < 0] = 0

    submission = pd.DataFrame({'ID': test_ids, 'TARGET': final_preds_safe})
    submission['ID'] = submission['ID'].astype(str)
    submission['TARGET'] = submission['TARGET'].round(0).astype(int)
    submission_filename = ROOT / 'submission.csv'
    submission.to_csv(submission_filename, index=False)
    print(f"Saved submission to {submission_filename}")


# Execute pipeline on import/run
run_pipeline()
