import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_train_data(filepath):
    """
    Load training data and preprocess it.
    
    Parameters:
    filepath (str): Path to the training data CSV file
    
    Returns:
    X (DataFrame): Preprocessed features
    y (Series): Target variable (log-transformed)
    encoder (OneHotEncoder): Fitted encoder for Gender/Sex column
    """
    # Load data
    data = pd.read_csv(filepath)
    
    # Separate features and target
    y = data['Calories']
    
    # Apply log transformation to target for RMSLE optimization
    y_log = np.log1p(y)
    
    # Check if 'Gender' or 'Sex' column exists in training data
    if 'Sex' in data.columns:
        # Use 'Sex' column
        gender_column = 'Sex'
    elif 'Gender' in data.columns:
        # Use 'Gender' column
        gender_column = 'Gender'
    else:
        raise ValueError("Neither 'Gender' nor 'Sex' column found in training data")
    
    # One-hot encode Gender/Sex column
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    gender_encoded = encoder.fit_transform(data[[gender_column]])
    
    # Create DataFrame with encoded gender
    gender_df = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out([gender_column]))
    
    # Combine with other features (excluding id and target)
    feature_columns = [col for col in data.columns if col not in ['id', 'Calories', gender_column]]
    X_processed = pd.concat([data[feature_columns], gender_df], axis=1)
    
    return X_processed, y_log, encoder

def load_and_preprocess_test_data(filepath, encoder):
    """
    Load test data and preprocess it using the same encoder.
    
    Parameters:
    filepath (str): Path to the test data CSV file
    encoder (OneHotEncoder): Fitted encoder for Gender/Sex column
    
    Returns:
    X (DataFrame): Preprocessed features
    ids (Series): ID column for submission
    """
    # Load data
    data = pd.read_csv(filepath)
    
    # Extract IDs for submission
    ids = data['id']
    
    # Check if 'Gender' or 'Sex' column exists in test data
    if 'Sex' in data.columns:
        # Use 'Sex' column
        gender_column = 'Sex'
    elif 'Gender' in data.columns:
        # Use 'Gender' column
        gender_column = 'Gender'
    else:
        raise ValueError("Neither 'Gender' nor 'Sex' column found in test data")
    
    # One-hot encode Gender/Sex column using the fitted encoder
    gender_encoded = encoder.transform(data[[gender_column]])
    
    # Create DataFrame with encoded gender
    gender_df = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out([gender_column]))
    
    # Combine with other features (excluding id and gender column)
    feature_columns = [col for col in data.columns if col not in ['id', gender_column]]
    X_processed = pd.concat([data[feature_columns], gender_df], axis=1)
    
    return X_processed, ids

def train_xgboost_model(X, y):
    """
    Train XGBoost model with default parameters on log-transformed target.
    
    Parameters:
    X (DataFrame): Preprocessed features
    y (Series): Log-transformed target variable
    
    Returns:
    model (XGBRegressor): Trained XGBoost model
    """
    # Initialize XGBoost regressor with default parameters
    model = XGBRegressor(random_state=42)
    
    # Train the model
    model.fit(X, y)
    
    return model

def validate_model(model, X_val, y_val):
    """
    Validate model performance using RMSLE.
    
    Parameters:
    model (XGBRegressor): Trained model
    X_val (DataFrame): Validation features
    y_val (Series): Log-transformed validation targets
    
    Returns:
    rmsle (float): Root Mean Squared Logarithmic Error
    """
    # Generate predictions
    y_pred_log = model.predict(X_val)
    
    # Convert back to original scale
    y_val_original = np.expm1(y_val)
    y_pred_original = np.expm1(y_pred_log)
    
    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y_val_original, y_pred_original))
    
    return rmsle

def generate_predictions(model, X_test):
    """
    Generate predictions using the trained model and convert back to original scale.
    
    Parameters:
    model (XGBRegressor): Trained model
    X_test (DataFrame): Preprocessed test features
    
    Returns:
    predictions (array): Model predictions in original scale
    """
    # Generate predictions (log scale)
    predictions_log = model.predict(X_test)
    
    # Convert back to original scale
    predictions = np.expm1(predictions_log)
    
    # Clip predictions to reasonable range
    predictions = np.clip(predictions, 0, 10000)
    
    return predictions

def save_submission(ids, predictions, filepath):
    """
    Save predictions to a CSV file in the required format.
    
    Parameters:
    ids (Series): ID column for submission
    predictions (array): Model predictions
    filepath (str): Path to save the submission file
    """
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': ids,
        'Calories': predictions
    })
    
    # Save to CSV
    submission.to_csv(filepath, index=False)
    
    print(f"Submission file saved to {filepath}")

def main():
    """
    Main function to run the entire pipeline.
    """
    print("Loading and preprocessing training data...")
    X_train, y_train, encoder = load_and_preprocess_train_data('data/train_subsample.csv')
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print("Training XGBoost model...")
    model = train_xgboost_model(X_train_split, y_train_split)
    
    print("Validating model...")
    rmsle = validate_model(model, X_val_split, y_val_split)
    print(f"Validation RMSLE: {rmsle:.4f}")
    
    # Only proceed if validation RMSLE is reasonable
    if rmsle < 0.5:  # More realistic threshold
        print("Model performance is acceptable. Generating submission...")
        
        # Load full test data for submission
        print("Loading and preprocessing full test data...")
        X_test, ids = load_and_preprocess_test_data('data/test_full.csv', encoder)
        
        # Generate predictions
        print("Generating predictions...")
        predictions = generate_predictions(model, X_test)
        
        # Save submission file
        print("Saving submission file...")
        save_submission(ids, predictions, 'results/submission.csv')
        
        print("Pipeline completed successfully!")
    else:
        print(f"Model performance is not acceptable (RMSLE: {rmsle:.4f}).")
        print("Please check the data or model parameters.")

if __name__ == "__main__":
    main()