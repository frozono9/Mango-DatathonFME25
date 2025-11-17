# Datathon FME 2025 – Mango

## Description

This repository contains our **final demand forecasting pipeline for Mango**, developed for the **Datathon FME 2025**.

Goal: **predict the optimal production quantity of garments for the next season**.

The final implementation is in `model.py` (cleaned and modularized), with `EDA.py` and `inference.py` helpers. The training pipeline produces an ensemble that was used in the original experiments.

---

## Repository Structure

```
.
├── data/
│   ├── train.csv        # Historical training data (semicolon-separated)
│   └── test.csv         # Test data for prediction
├── EDA.py               # Exploratory Data Analysis script (plots & visualization)
├── model.py             # Main training pipeline (single-run script)
├── inference.py         # Simple CLI to predict using saved models from a JSON input
└── outputs/             # Output models, artifacts and submissions
```

---

## Final Pipeline (`model.py`)

### Main steps:

1. **Library imports**

   * pandas, numpy, sklearn, catboost, etc.

2. **Global configuration**

   * Paths, PCA parameters, cross-validation, ensemble weights

3. **Feature engineering**

   * Data cleaning and aggregation
   * Parsing and PCA of image embeddings
   * Aggregated features by family, category, and attributes
   * Logarithmic normalization of numerical features

4. **Training of finalist models**

   * **Model A**: Alpha=0.78, learning_rate=0.01 (more stable)
   * **Model B**: Alpha=0.75, learning_rate=0.03 (more aggressive)
   * CatBoost with **K-Fold CV** to select optimal iterations

5. **Weighted ensemble**

   * 60% Model A + 40% Model B
   * Inverse log1p transformation to obtain real predictions

6. **Submission generation**

   * Writes `submission.csv` at the repository root

---

## Requirements

* Python >= 3.9
* pandas
* numpy
* scikit-learn
* catboost

```bash
pip install pandas numpy scikit-learn catboost
```

---

## Usage

1. Place `train.csv` and `test.csv` in the `data/` folder
2. Run the training pipeline (saves models to `outputs/` and writes `submission.csv` at repo root):

```bash
python model.py
```

3. You will get `submission.csv` and models/artifacts under `outputs/`.
## Exploratory Data Analysis (EDA)

`EDA.py` generates a set of plots to inspect the training dataset. To execute it (and open plots), run:

```bash
python EDA.py
```

If running in a headless environment, you may redirect or save each plot; the script prints status messages as it runs.

## Inference from JSON (optional)

If you want to test the model on a JSON payload mirroring `data/test.csv`, use `inference.py`:

```bash
python inference.py --input-json sample_input.json --output-json predictions.json
```

This outputs a list of `{ ID, TARGET }`. The script reuses the same feature pipeline and needs `data/train.csv` present for group aggregations.

## How to run the model

- Install requirements:

```bash
pip install -r requirements.txt
```

- Place CSVs under `data/` (semicolon-separated): `train.csv`, `test.csv`.
- Train and generate submission:

```bash
python model.py
```

- The submission is saved as `submission.csv` at the project root.

---

## Achievements and Learnings

* CatBoost model ensemble achieved **55.57900 accuracy**
* Robust feature engineering was more decisive than hypertuning complex models
* Combination of image embeddings, categorical attributes, and multi-season historical data was key
* Temporal validation (TimeSeriesSplit) avoided data leakage and enabled generalizable models

---

## Next Steps

* Train our own visual embeddings
* Explore TabNet or LightGBM with automatic tuning
* Add interpretability to the pipeline to understand which attributes generate more demand
* Automate the entire workflow for real production

---

## Credits

Team **Oink Oink** – AI Students UPC, Datathon FME 2025.
