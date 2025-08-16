# Model Architecture Diagram

## Workflow Overview

```mermaid
graph TD
    A[Data Files] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[Model Evaluation]
    E --> F[Select Best Model]
    F --> G[Generate Predictions]
    G --> H[Create Submission File]
    
    A1[train_subsample.csv] --> A
    A2[test_subsample.csv] --> A
    A3[full_test_for_submission.csv] --> A
    
    B1[Handle Missing Values] --> B
    B2[Encode Categorical Variables] --> B
    
    C1[One-Hot Encode Gender] --> C
    
    D1[Linear Regression] --> D
    D2[Random Forest] --> D
    
    E1[Cross-Validation] --> E
    E2[RMSLE Calculation] --> E
    
    F1[Model Comparison] --> F
    
    G1[Load Best Model] --> G
    G2[Preprocess Test Data] --> G
    G3[Model Prediction] --> G
    
    H1[Format Output] --> H
    H2[Save CSV] --> H
```

## Data Flow Explanation

1. **Data Input**: Three CSV files containing training and test data
2. **Preprocessing**: Handle missing values and encode categorical variables
3. **Feature Engineering**: One-hot encode the Gender column
4. **Model Training**: Train both Linear Regression and Random Forest models
5. **Evaluation**: Use cross-validation and RMSLE metric to compare models
6. **Selection**: Choose the better performing model based on evaluation
7. **Prediction**: Generate predictions on test data using best model
8. **Submission**: Format and save predictions in required CSV format

## Model Comparison Metrics

- **Linear Regression**: 
  - Pros: Simple, fast, interpretable
  - Cons: Assumes linear relationships
  
- **Random Forest**:
  - Pros: Handles non-linear relationships, feature interactions
  - Cons: More complex, slower to train

## Evaluation Strategy

- Use RMSLE (Root Mean Squared Logarithmic Error) as specified
- Implement k-fold cross-validation for robust evaluation
- Compare models on validation set before final prediction