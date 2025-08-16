# Calorie Expenditure Prediction - Baseline Model

This repository contains a baseline machine learning model for predicting calorie expenditure during workouts, based on physiological and workout parameters.

## Data Files

The implementation uses the following data files provided by the user:

- `data/train_subsample.csv` - Training data (75,000 samples)
- `data/test_full.csv` - Full test data for submission (250,000 samples)
- `data/test_subsample.csv` - Subsampled test data for validation (25,000 samples)

## Model Architecture

The baseline model uses XGBoost Regressor with the following key features:

1. **Target Transformation**: Log(1+y) transformation applied to optimize for RMSLE
2. **Feature Engineering**: One-hot encoding of the Gender/Sex categorical variable
3. **Validation**: Train/validation split to evaluate model performance before submission
4. **Prediction Post-processing**: Conversion back to original scale and clipping to ensure physically plausible results

## Requirements

- Python 3.7+
- pandas>=1.3.0
- numpy>=1.20.0
- scikit-learn>=1.0.0
- xgboost>=1.5.0

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

Run the baseline model:
```bash
python src/baseline_model.py
```

The script will:
1. Load and preprocess the training data
2. Train an XGBoost model on log-transformed targets
3. Validate model performance on a held-out validation set
4. If performance is acceptable, generate predictions on the full test set
5. Save predictions to `results/submission.csv`

## Output

The submission file will be saved to `results/submission.csv` with the following format:
```
id,Calories
1,300.5
2,250.2
...
```

This file is ready for submission to the Kaggle competition.