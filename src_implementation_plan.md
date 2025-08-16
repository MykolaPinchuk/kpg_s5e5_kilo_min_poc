# Source Code Implementation Plan

## Data Preprocessing Module

### File: `src/data_preprocessing.py`

```python
"""
Data preprocessing module for Kaggle Calorie Expenditure prediction.
Handles data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(train_file_path, test_file_path):
    """
    Load and preprocess training and test data.
    
    Parameters:
    train_file_path (str): Path to training data CSV
    test_file_path (str): Path to test data CSV
    
    Returns:
    tuple: Preprocessed training and test data (X_train, X_test, y_train, y_test)
    """
    # Load data
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    
    # Separate features and target
    X_train = train_data.drop(['Calories', 'id'], axis=1)
    y_train = train_data['Calories']
    
    # For test data, we don't have target values
    X_test = test_data.drop(['id'], axis=1)
    
    # Handle categorical variables (Gender)
    X_train_processed, X_test_processed = encode_categorical_variables(X_train, X_test)
    
    return X_train_processed, X_test_processed, y_train

def encode_categorical_variables(X_train, X_test):
    """
    One-hot encode categorical variables.
    
    Parameters:
    X_train (DataFrame): Training features
    X_test (DataFrame): Test features
    
    Returns:
    tuple: Encoded training and test features
    """
    # Initialize encoder
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    
    # Fit on training data and transform both training and test data
    X_train_encoded = encoder.fit_transform(X_train[['Gender']])
    X_test_encoded = encoder.transform(X_test[['Gender']])
    
    # Create DataFrames with encoded features
    gender_columns = [f"Gender_{cat}" for cat in encoder.categories_[0][1:]]
    X_train_gender = pd.DataFrame(X_train_encoded, columns=gender_columns, index=X_train.index)
    X_test_gender = pd.DataFrame(X_test_encoded, columns=gender_columns, index=X_test.index)
    
    # Combine with other features
    X_train_final = pd.concat([X_train.drop(['Gender'], axis=1), X_train_gender], axis=1)
    X_test_final = pd.concat([X_test.drop(['Gender'], axis=1), X_test_gender], axis=1)
    
    return X_train_final, X_test_final

def prepare_submission_data(test_file_path):
    """
    Prepare test data for submission predictions.
    
    Parameters:
    test_file_path (str): Path to full test data CSV for submission
    
    Returns:
    tuple: Processed features and IDs for submission
    """
    # Load data
    test_data = pd.read_csv(test_file_path)
    
    # Extract IDs
    ids = test_data['id']
    
    # Remove ID column
    X_test = test_data.drop(['id'], axis=1)
    
    # Handle categorical variables (Gender)
    X_test_processed, _ = encode_categorical_variables_for_submission(X_test)
    
    return X_test_processed, ids

def encode_categorical_variables_for_submission(X_test):
    """
    One-hot encode categorical variables for submission data.
    Uses pre-fitted encoder.
    
    Parameters:
    X_test (DataFrame): Test features
    
    Returns:
    DataFrame: Encoded test features
    """
    # This would use a pre-fitted encoder saved during training
    # Implementation would depend on how the encoder is saved
    pass
```

## Model Training Module

### File: `src/model_training.py`

```python
"""
Model training module for Kaggle Calorie Expenditure prediction.
Implements training and evaluation of baseline models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error
import joblib

def train_baseline_models(X_train, y_train):
    """
    Train baseline models: Linear Regression and Random Forest.
    
    Parameters:
    X_train (DataFrame): Training features
    y_train (Series): Training targets
    
    Returns:
    dict: Trained models
    """
    # Initialize models
    linear_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train models
    linear_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    
    # Store models
    models = {
        'linear_regression': linear_model,
        'random_forest': rf_model
    }
    
    return models

def evaluate_models(models, X_train, y_train):
    """
    Evaluate models using cross-validation and RMSLE.
    
    Parameters:
    models (dict): Dictionary of trained models
    X_train (DataFrame): Training features
    y_train (Series): Training targets
    
    Returns:
    dict: Evaluation results
    """
    results = {}
    
    for name, model in models.items():
        # Cross-validation with RMSLE
        # Note: sklearn doesn't have RMSLE directly, so we'll use MSE on log(y+1)
        scores = cross_val_score(model, X_train, np.log1p(y_train), 
                                cv=5, scoring='neg_mean_squared_error')
        rmsle_scores = np.sqrt(-scores)
        results[name] = {
            'mean_rmsle': np.mean(rmsle_scores),
            'std_rmsle': np.std(rmsle_scores)
        }
    
    return results

def select_best_model(models, evaluation_results):
    """
    Select the best model based on evaluation results.
    
    Parameters:
    models (dict): Dictionary of trained models
    evaluation_results (dict): Evaluation results for each model
    
    Returns:
    object: Best model
    """
    # Select model with lowest RMSLE
    best_model_name = min(evaluation_results, key=lambda x: evaluation_results[x]['mean_rmsle'])
    best_model = models[best_model_name]
    
    return best_model, best_model_name

def save_model(model, model_name, filepath):
    """
    Save trained model to disk.
    
    Parameters:
    model (object): Trained model
    model_name (str): Name of the model
    filepath (str): Path to save model
    """
    joblib.dump(model, filepath)
```

## Submission Generation Module

### File: `src/submission_generator.py`

```python
"""
Submission generation module for Kaggle Calorie Expenditure prediction.
Generates predictions and creates submission file.
"""

import pandas as pd
import joblib

def generate_predictions(model, X_test):
    """
    Generate predictions using trained model.
    
    Parameters:
    model (object): Trained model
    X_test (DataFrame): Test features
    
    Returns:
    array: Predictions
    """
    predictions = model.predict(X_test)
    return predictions

def create_submission_file(ids, predictions, filepath):
    """
    Create submission file in required format.
    
    Parameters:
    ids (Series): Test data IDs
    predictions (array): Model predictions
    filepath (str): Path to save submission file
    """
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': ids,
        'Calories': predictions
    })
    
    # Save to CSV
    submission.to_csv(filepath, index=False)
    
    return submission
```

## Main Execution Script

### File: `src/main.py`

```python
"""
Main execution script for Kaggle Calorie Expenditure prediction.
Orchestrates the entire workflow from data preprocessing to submission.
"""

from data_preprocessing import load_and_preprocess_data, prepare_submission_data
from model_training import train_baseline_models, evaluate_models, select_best_model, save_model
from submission_generator import generate_predictions, create_submission_file

def main():
    # File paths
    train_file = 'data/train_subsample.csv'
    test_file = 'data/test_subsample.csv'
    submission_file = 'data/full_test_for_submission.csv'
    output_submission = 'results/submission.csv'
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train = load_and_preprocess_data(train_file, test_file)
    
    # Train baseline models
    print("Training baseline models...")
    models = train_baseline_models(X_train, y_train)
    
    # Evaluate models
    print("Evaluating models...")
    evaluation_results = evaluate_models(models, X_train, y_train)
    
    # Print evaluation results
    for model_name, results in evaluation_results.items():
        print(f"{model_name}: RMSLE = {results['mean_rmsle']:.4f} (+/- {results['std_rmsle']:.4f})")
    
    # Select best model
    best_model, best_model_name = select_best_model(models, evaluation_results)
    print(f"Best model: {best_model_name}")
    
    # Save best model
    model_save_path = f"models/{best_model_name}_model.pkl"
    save_model(best_model, best_model_name, model_save_path)
    print(f"Best model saved to {model_save_path}")
    
    # Generate predictions for submission
    print("Generating predictions for submission...")
    X_submission, ids = prepare_submission_data(submission_file)
    predictions = generate_predictions(best_model, X_submission)
    
    # Create submission file
    submission = create_submission_file(ids, predictions, output_submission)
    print(f"Submission file created at {output_submission}")
    
    # Print submission summary
    print(f"Number of predictions: {len(predictions)}")
    print(f"Prediction range: {predictions.min():.2f} - {predictions.max():.2f}")

if __name__ == "__main__":
    main()
```

## Jupyter Notebook for EDA

### File:: `notebooks/eda_baseline_model.ipynb`

The notebook would contain:
1. Data loading and basic statistics
2. Feature distribution visualization
3. Correlation analysis
4. Missing value analysis
5. Baseline model training and evaluation
6. Results visualization

## Directory Structure Implementation

The implementation will create the following directory structure:
```
project/
├── data/
│   ├── train_subsample.csv
│   ├── test_subsample.csv
│   └── full_test_for_submission.csv
├── models/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── submission_generator.py
├── results/
├── baseline_model_plan.md
├── model_architecture.md
├── requirements.md
└── README.md