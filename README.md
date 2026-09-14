# Diabetes Classification Project

This repository contains a machine learning final project that predicts diabetes outcome from clinical measurements in `diabetes.csv`. The main analysis is implemented in the Jupyter notebook `머신러닝_최종과제_이명지.ipynb`.

The diabetes workflow covers exploratory data analysis, missing-value handling, feature scaling, logistic regression training, model evaluation, and hyperparameter tuning with cross-validation. The notebook also includes a separate Hugging Face/Gemma news-classification exercise near the end.

## Dataset

The project uses a diabetes classification dataset with 768 rows and 9 columns:

| Column | Description |
| --- | --- |
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin fold thickness |
| `Insulin` | 2-hour serum insulin |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age in years |
| `Outcome` | Target label, where `0` means non-diabetic and `1` means diabetic |

The target distribution is imbalanced:

| Outcome | Count | Share |
| --- | ---: | ---: |
| `0` | 500 | 65.1% |
| `1` | 268 | 34.9% |

The notebook found no explicit null values in the CSV file, but several medical fields contain `0` values that are not physiologically meaningful. These values are treated as missing data for:

| Column | Zero Values |
| --- | ---: |
| `Glucose` | 5 |
| `BloodPressure` | 35 |
| `SkinThickness` | 227 |
| `Insulin` | 374 |
| `BMI` | 11 |

## Preprocessing

The preprocessing pipeline follows these steps:

1. Split features and target:
   - `X`: all columns except `Outcome`
   - `y`: `Outcome`
2. Replace `0` with `NaN` in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`.
3. Split the data into training and test sets using:
   - `test_size=0.2`
   - `random_state=42`
   - `stratify=y`
4. Fill missing values using statistics calculated from the training set only:
   - Mean imputation for `Glucose`, `BloodPressure`, and `BMI`
   - Median imputation for `Insulin` and `SkinThickness`, because these columns show stronger outlier effects
5. Scale features with `StandardScaler`:
   - Fit on the training set
   - Transform both training and test sets

The final split contains 614 training rows and 154 test rows.

## Model

The diabetes classifier uses scikit-learn's `LogisticRegression` for binary classification.

The first model uses the default logistic regression settings. The improved-model section then checks overfitting and applies `GridSearchCV` with 5-fold cross-validation to tune the L2 regularization strength:

```python
param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
)
```

The best parameter found was `C=1`, which is also the default value for `LogisticRegression`.

## Evaluation Results

The notebook evaluates the model with accuracy and F1-score. Accuracy measures overall correctness, while F1-score is useful because the dataset has more non-diabetic cases than diabetic cases.

| Model | Accuracy | F1-score |
| --- | ---: | ---: |
| Baseline logistic regression | 0.7078 | 0.5455 |
| Tuned logistic regression | 0.7078 | 0.5455 |

The overfitting check for the baseline model reported:

| Split | Accuracy | F1-score |
| --- | ---: | ---: |
| Training | 0.7980 | 0.6737 |
| Test | 0.7078 | 0.5455 |

The best 5-fold cross-validation F1-score during tuning was `0.6539`.

## Key Findings

- `Glucose` is the clearest signal in the exploratory analysis. The diabetic group has a higher median glucose level, around 140 mg/dL, compared with about 107 mg/dL for the non-diabetic group.
- BMI and glucose do not show a strong linear relationship in the scatter plot, but higher glucose values are associated with a higher concentration of diabetic cases.
- `Insulin` and `SkinThickness` contain many `0` entries, so imputing them may remove some individual-level information from the original measurements.
- Hyperparameter tuning did not improve the final test score because the best value, `C=1`, matched the default logistic regression setting.
- The final model correctly classifies about 70.8% of the test set, but the lower F1-score shows that detecting diabetic cases remains harder than detecting non-diabetic cases.

## Project Structure

```text
.
├── README.md
├── diabetes.csv
├── 머신러닝_최종과제_이명지.ipynb
├── pyproject.toml
├── uv.lock
└── src/
    └── ml_final_pj/
        └── __init__.py
```

## Setup

This project is configured with `uv` and a `pyproject.toml` file. The recorded project configuration uses Python `>=3.14`.

Install dependencies with:

```bash
uv sync
```

Then launch the notebook from your preferred Jupyter environment. If Jupyter is available in the environment, you can run:

```bash
uv run jupyter lab
```

Open `머신러닝_최종과제_이명지.ipynb` and run the diabetes analysis cells from top to bottom.

If you are not using `uv`, create a Python environment and install the main diabetes-analysis dependencies manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

The notebook's final Hugging Face section also uses:

```bash
pip install torch transformers accelerate huggingface-hub safetensors
```

## How to Run the Diabetes Classifier

1. Make sure `diabetes.csv` is in the repository root.
2. Start Jupyter with `uv run jupyter lab` if Jupyter is installed, or use another Jupyter launcher.
3. Open `머신러닝_최종과제_이명지.ipynb`.
4. Run the cells through the logistic regression and tuning sections.
5. Compare the baseline and tuned model results in the printed evaluation table.

The notebook is the source of truth for the current implementation and reported results.
