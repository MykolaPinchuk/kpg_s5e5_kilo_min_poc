import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
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
    y (Series): Target variable
    encoder (OneHotEncoder): Fitted encoder for Gender column
    """
    # Load data
    data = pd.read_csv(filepath)
    
    # Separate features and target
    X = data.drop('Calories', axis=1)
    y = data['Calories']
    
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
    predictions (array): Model predictions
    """
    # Generate predictions
    predictions = model.predict(X_test)
    
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
    # Step 1: Load and preprocess training data
    print("Loading and preprocessing training data...")
    X_train, y_train, encoder = load_and_preprocess_train_data('data/train_subsample.csv')
    
    # Step 2: Train XGBoost model
    print("Training XGBoost model...")
    model = train_xgboost_model(X_train, y_train)
    
    # Step 3: Load and preprocess test data
    print("Loading and preprocessing test data...")
    X_test, ids = load_and_preprocess_test_data('data/full_test_for_submission.csv', encoder)
    
    # Step 4: Generate predictions
    print("Generating predictions...")
    predictions = generate_predictions(model, X_test)
    
    # Step 5: Save submission file
    print("Saving submission file...")
    save_submission(ids, predictions, 'results/submission.csv')
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()