# Cirrhosis Patient Survival Prediction

Multiclass classification of cirrhosis patient outcomes: predicting the probability of each status (C / CL / D) with LightGBM. Built for the Kaggle competition *"Sirroz bemorlari omon qolish prognozini aniqlash"*.

**Result:** 5-fold CV log loss **0.373**, 48% better than the baseline.

---

## Task

For every patient, predict the probability of three outcomes:

| Class | Meaning |
|---|---|
| **C** | Alive at the end of follow-up |
| **CL** | Left follow-up due to a liver transplant |
| **D** | Died |

**Metric:** multi-class log loss. It rewards honest probabilities and heavily penalizes confident mistakes, so the model outputs `predict_proba` for all three classes, not a single label.

## Dataset

- **Train:** 15,000 rows × 20 columns · **Test:** 10,000 rows
- **Features:** follow-up time (`N_Days`), demographics (`Age` in days, `Sex`), treatment (`Drug`), clinical signs (`Ascites`, `Hepatomegaly`, `Spiders`, `Edema`), lab results (`Bilirubin`, `Cholesterol`, `Albumin`, `Copper`, `Alk_Phos`, `SGOT`, `Tryglicerides`, `Platelets`, `Prothrombin`) and disease `Stage`
- **Class balance:** C 67.0% · D 30.4% · CL 2.5%

## Data analysis

- **Label error:** one row had `Status = "Y"`, which is not a valid class. Removed.
- **Missing values come in a block.** `Drug`, `Ascites`, `Hepatomegaly`, `Spiders`, `Copper`, `Alk_Phos` and `SGOT` are missing *together* in ~43% of rows (6,465 rows miss all seven, 8,388 are complete). `Cholesterol` and `Tryglicerides` are ~55% missing.
- **Missingness is not informative.** The death rate differs by only 1–3 pp between rows with and without missing values, so missing indicators add little.
- **Test matches train.** Missing percentages in test are within 1 pp of train.
- **Outliers:** `Age` max ≈ 354 years and `N_Days` max ≈ 105 years are physically impossible. Tree models are robust to them.
- **Feature association (Cramér's V):** `Hepatomegaly` is the strongest categorical feature; `Drug` is the weakest (0.042).

## Preprocessing

- **Categorical NaN → `"Unknown"`** as its own category. Filling with the mode would silently mislabel unknown patients (e.g. `Hepatomegaly = "N"`).
- **LightGBM:** numeric NaN left as-is (handled natively), categoricals as `category` dtype.
- **Logistic Regression:** median imputation + missing indicators + `StandardScaler` + one-hot encoding, all inside a `Pipeline` to prevent data leakage.
- **Validation:** `StratifiedKFold` (5 folds, shuffled), scoring with `neg_log_loss`.

## Results

| Model | CV log loss | Avg. probability on the true class |
|---|---|---|
| Baseline (class priors only) | 0.7235 | 49% |
| Logistic Regression | 0.4615 ± 0.0124 | 63% |
| **LightGBM** | **0.3733 ± 0.0084** | **69%** |

LightGBM is **48% better** than the baseline, with **85.5% accuracy** on out-of-fold predictions.

### Why LightGBM

- Lowest log loss of all tested models
- Handles missing values natively, so 43–56% empty columns are used without imputation
- Captures non-linear interactions between features
- Robust to outliers and stable across folds (± 0.008)

## Error analysis

Based on out-of-fold predictions:

| Class | Rows | Avg. loss per row | Share of total loss | Recall | AUC |
|---|---|---|---|---|---|
| C | 10,053 | 0.201 | 36.0% | 0.93 | 0.924 |
| CL | 381 | 2.560 | 17.4% | 0.18 | 0.894 |
| D | 4,565 | 0.571 | **46.5%** | 0.74 | 0.927 |

- **C** is predicted best.
- **D** is the main source of error: 26% of patients who died were predicted as C, and this class produces nearly half of the total loss.
- **CL** has the highest per-row penalty because it is rare (2.5%). Its AUC of 0.894 shows the model still ranks these patients well; low absolute probabilities are the correct behavior for such a rare class under log loss.

## Next steps

- **Mayo risk score** feature (age, bilirubin, albumin, prothrombin, edema). These columns are available for almost every patient, which helps the 43% with missing lab blocks.
- **Ensemble** of LightGBM, XGBoost and CatBoost on shared folds
- **Tuning** the number of trees with a learning curve / early stopping
- **Calibration** check of predicted probabilities

## Project structure

```
├── cirrhosis_survival_prediction.ipynb   # analysis, modeling, evaluation
├── submission.csv                         # test set predictions
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
```

Download `train.csv`, `test.csv` and `sample_submission.csv` from the competition page, set `input_path` in the notebook, and run all cells. Notebook explanations are written in Uzbek.

## Tech stack

Python · pandas · NumPy · scikit-learn · LightGBM · Matplotlib · Seaborn

## Author

**Abdurahmon Qodirov** · [GitHub](https://github.com/abdurrohmanq) · [LinkedIn](https://linkedin.com/in/abdurrohman-qodirov)
