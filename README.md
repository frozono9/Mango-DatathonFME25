# Datathon FME 2025 – Mango

## Description

This repository contains our **final demand forecasting pipeline for Mango**, developed for the **Datathon FME 2025**.

Goal: **predict the optimal production quantity of garments for the next season**.

Version **`8.py`** is the final one that allowed us to achieve **55.57900 accuracy**, combining the best models in a **weighted ensemble**.

---

## Repository Structure

```
.
├── data/
│   ├── train.csv        # Historical training data
│   └── test.csv         # Test data for prediction
├── 1.py … 7.py          # Previous experiment versions
└── 8.py                 # Final pipeline (ensemble of finalists)
```

---

## Final Pipeline (`8.py`)

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

   * File `submission_catboost_V18_EnsembleFinalists.csv` ready for Kaggle/Datathon

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
2. Run the final pipeline:

```bash
python 8.py
```

3. You will get `submission_catboost_V18_EnsembleFinalists.csv` with the final predictions.

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
