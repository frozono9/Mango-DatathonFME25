"""
Simple inference script for FME2025 ensemble models.

Usage examples:
  python inference.py --input-json sample_input.json
  python inference.py --input-json sample_input.json --output-json predictions.json

Input JSON should be a list of product-level records with the same fields as `data/test.csv`.
The script will load saved models/artifacts from `outputs/` and run the same feature pipeline.
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import joblib
import numpy as np
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'outputs'
DATA_DIR = ROOT / 'data'
PATH_TRAIN = DATA_DIR / 'train.csv'


def load_artifacts():
    features = joblib.load(OUTPUT_DIR / 'finalist_features.joblib')
    cat_features = joblib.load(OUTPUT_DIR / 'finalist_categorical_features.joblib')
    scaler = joblib.load(OUTPUT_DIR / 'finalist_scaler.joblib') if (OUTPUT_DIR / 'finalist_scaler.joblib').exists() else None
    pca = joblib.load(OUTPUT_DIR / 'finalist_pca.joblib') if (OUTPUT_DIR / 'finalist_pca.joblib').exists() else None
    model_1 = CatBoostRegressor(); model_1.load_model(str(OUTPUT_DIR / 'finalist_V17-A.cbm'))
    model_2 = CatBoostRegressor(); model_2.load_model(str(OUTPUT_DIR / 'finalist_V17-B.cbm'))
    return features, cat_features, scaler, pca, model_1, model_2


def predict_from_json(input_json_path, output_json_path=None):
    # Load input data
    with open(input_json_path, 'r') as f:
        payload = json.load(f)
    # If a dict is passed for single record, wrap into a list
    if isinstance(payload, dict):
        payload = [payload]
    df_test = pd.DataFrame(payload)

    # Load training data to compute group aggregations used during feature engineering
    if not PATH_TRAIN.exists():
        print("train.csv not found. Inference requires the train CSV to compute group aggregations used in feature engineering.")
        return
    df_train = pd.read_csv(PATH_TRAIN, sep=';')

    # We import feature_engineering from model.py so we can reuse the pipeline
    try:
        from model import feature_engineering
    except Exception as e:
        print("Could not import feature_engineering from model.py:", e)
        return

    _, _, X_test, test_ids, metadata = feature_engineering(df_train, df_test)
    features, cat_feats, scaler, pca, model_1, model_2 = load_artifacts()

    # Predict
    p1 = model_1.predict(X_test)
    p2 = model_2.predict(X_test)
    ensemble_log = (p1 * 0.6) + (p2 * 0.4)
    ensemble_real = np.expm1(ensemble_log)
    ensemble_real[ensemble_real < 0] = 0

    out_df = pd.DataFrame({'ID': test_ids.astype(str), 'TARGET': ensemble_real.round(0).astype(int)})
    if output_json_path:
        out_df.to_json(output_json_path, orient='records', indent=2)
        print(f"Predictions saved to {output_json_path}")
    else:
        print(out_df.to_json(orient='records', indent=2))
    return out_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using saved ensemble models and a JSON file containing product rows')
    parser.add_argument('--input-json', required=True, help='Path to input JSON file containing product rows')
    parser.add_argument('--output-json', required=False, help='Optional path to save predictions JSON')
    args = parser.parse_args()
    predict_from_json(args.input_json, args.output_json)
