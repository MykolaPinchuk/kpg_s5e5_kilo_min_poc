import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
    y (Series): Log-transformed target variable
    encoder (OneHotEncoder): Fitted encoder for Gender column
    """
    # Load data
    data = pd.read_csv(filepath)
    
    # Separate features and target
    X = data.drop('Calories', axis=1)
    y = data['Calories']
    
    # Apply log(1 + y) transformation to target
    y = np.log1p(y)
    
    # One-hot encode Gender column
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    gender_encoded = encoder.fit_transform(X[['Gender']])
    
    # Create DataFrame with encoded gender
    gender_df = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out(['Gender']))
    
    # Combine with other features
    X_processed = pd.concat([X.drop('Gender', axis=1), gender_df], axis=1)
    
    return X_processed, y, encoder

def train_xgboost_model(X, y):
    """
    Train XGBoost model with default parameters.
    
    Parameters:
    X (DataFrame): Preprocessed features
    y (Series): Target variable
    
    Returns:
    model (XGBRegressor): Trained XGBoost model
    """
    # Initialize XGBoost regressor with default parameters
    model = XGBRegressor(random_state=42)
    
    # Train the model
    model.fit(X, y)
    
    return model

def load_and_preprocess_test_data(filepath, encoder):
    """
    Load test data and preprocess it using the same encoder.
    
    Parameters:
    filepath (str): Path to the test data CSV file
    encoder (OneHotEncoder): Fitted encoder for Gender column
    
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
        # Rename 'Sex' column to 'Gender' to match training data
        data = data.rename(columns={'Sex': 'Gender'})
        gender_column = 'Gender'
    else:
        gender_column = 'Gender'  # Assume 'Gender' column exists
    
    # One-hot encode Gender column using the fitted encoder
    gender_encoded = encoder.transform(data[[gender_column]])
    
    # Create DataFrame with encoded gender
    gender_df = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out([gender_column]))
    
    # Combine with other features
    X_processed = pd.concat([data.drop(gender_column, axis=1), gender_df], axis=1)
    
    return X_processed, ids

def generate_predictions(model, X_test):
    """
    Generate predictions using the trained model.
    
    Parameters:
    model (XGBRegressor): Trained model
    X_test (DataFrame): Preprocessed test features
    
    Returns:
    predictions (array): Model predictions converted back to original scale
    """
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Apply exp(predictions) - 1 transformation to convert back to original scale
    predictions = np.expm1(predictions)
    
    # Clip predictions to [0, max_calorie] range
    max_calorie = 10000  # Set a reasonable upper limit for calories
    predictions = np.clip(predictions, 0, max_calorie)
    
    return predictions

def calculate_rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error (RMSLE).
    
    Parameters:
    y_true (array): True values
    y_pred (array): Predicted values
    
    Returns:
    rmsle (float): RMSLE score
    """
    # Apply log(1 + x) transformation to both true and predicted values
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    
    # Calculate MSE of log-transformed values
    mse = mean_squared_error(log_true, log_pred)
    
    # Return square root of MSE
    return np.sqrt(mse)

def validate_model(X, y, encoder, test_size=0.2, random_state=42):
    """
    Validate the model using an 80-20 train-validation split.
    
    Parameters:
    X (DataFrame): Preprocessed features
    y (Series): Log-transformed target variable
    encoder (OneHotEncoder): Fitted encoder for Gender column
    test_size (float): Proportion of data to use for validation
    random_state (int): Random seed for reproducibility
    
    Returns:
    rmsle (float): RMSLE score on validation set
    model (XGBRegressor): Trained model
    """
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model on training set
    model = train_xgboost_model(X_train, y_train)
    
    # Generate predictions on validation set
    val_predictions = model.predict(X_val)
    
    # Convert predictions back to original scale
    val_predictions_original = np.expm1(val_predictions)
    y_val_original = np.expm1(y_val)
    
    # Calculate RMSLE
    rmsle = calculate_rmsle(y_val_original, val_predictions_original)
    
    return rmsle, model

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
    # Step 1: Load and preprocess training data
    print("Loading and preprocessing training data...")
    X_train, y_train, encoder = load_and_preprocess_train_data('data/train_subsample.csv')
    
    # Step 2: Validate model with 80-20 train-validation split
    print("Validating model...")
    validation_rmsle, model = validate_model(X_train, y_train, encoder)
    print(f"Validation RMSLE: {validation_rmsle:.4f}")
    
    # Step 3: Check if validation RMSLE is less than 0.1
    if validation_rmsle < 0.1:
        print("Validation passed. Generating submission...")
        
        # Step 4: Load and preprocess test data
        print("Loading and preprocessing test data...")
        X_test, ids = load_and_preprocess_test_data('data/full_test_for_submission.csv', encoder)
        
        # Step 5: Generate predictions
        print("Generating predictions...")
        predictions = generate_predictions(model, X_test)
        
        # Step 6: Save submission file
        print("Saving submission file...")
        save_submission(ids, predictions, 'results/submission.csv')
        
        print("Pipeline completed successfully!")
    else:
        print(f"Validation failed. RMSLE {validation_rmsle:.4f} is not less than 0.1. No submission generated.")

if __name__ == "__main__":
    main()