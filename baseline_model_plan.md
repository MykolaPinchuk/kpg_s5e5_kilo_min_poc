# Baseline ML Model Plan for Kaggle Competition: Predict Calorie Expenditure

## Competition Overview
- **Goal**: Predict the number of calories burned during a workout (regression problem)
- **Evaluation Metric**: Root Mean Squared Logarithmic Error (RMSLE)
- **Data Fields**:
  - id: Unique identifier
  - Gender: Participant's gender
  - Age: Participant's age in years
  - Height: Participant's height in cm
  - Weight: Participant's weight in kg
  - Duration: Workout duration in minutes
  - Heart_Rate: Average heart rate during workout (bpm)
  - Body_Temp: Body temperature during workout (Celsius)
  - Calories: Total calories burned (target variable)

## Data Files
- `train_subsample.csv`: Training dataset with all features including target variable (Calories)
- `test_subsample.csv`: Test dataset without target variable
- `full_test_for_submission.csv`: Full test dataset for final submission

## Baseline Model Architecture

### Model Selection
For a quick POC baseline, we'll use:
1. **Linear Regression** - Simple, interpretable, and fast to train
2. **Random Forest Regressor** - Handles non-linear relationships better than linear models

### Data Preprocessing Steps
1. **Handle Categorical Variables**:
   - One-hot encode the 'Gender' column
   
2. **Feature Engineering**:
   - No complex feature engineering for baseline
   - Use raw features as provided
   
3. **Data Cleaning**:
   - Check for missing values
   - Handle outliers if any are found

### Model Training Pipeline
1. Load training data
2. Preprocess features (one-hot encoding for Gender)
3. Split data for validation (80/20 split)
4. Train both baseline models
5. Evaluate using RMSLE metric
6. Select better performing model

### Model Evaluation
- Use RMSLE as specified in competition
- Compare Linear Regression vs Random Forest
- Cross-validation for robustness

### Submission Generation
1. Load best performing model
2. Preprocess test data (same steps as training data)
3. Generate predictions on test set
4. Create submission file with 'id' and 'Calories' columns

## Implementation Plan

### File Structure
```
project/
├── data/
│   ├── train_subsample.csv
│   ├── test_subsample.csv
│   └── full_test_for_submission.csv
├── models/
│   ├── linear_regression_model.pkl
│   └── random_forest_model.pkl
├── notebooks/
│   └── eda_baseline_model.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── submission_generator.py
├── results/
│   └── submission.csv
├── requirements.txt
└── README.md
```

## Expected Approach
1. **Exploratory Data Analysis**:
   - Understand data distributions
   - Check for correlations between features and target
   - Identify any data quality issues

2. **Baseline Model Implementation**:
   - Implement data preprocessing pipeline
   - Train both Linear Regression and Random Forest models
   - Evaluate using RMSLE metric
   - Select the better performing model

3. **Submission Generation**:
   - Generate predictions on test set
   - Format predictions according to submission file requirements
   - Save submission file for competition upload

## Success Criteria
- Successfully trained baseline model
- Valid submission file generated
- RMSLE score calculated for validation set
- Code is well-documented and reproducible

## Timeline
- EDA and Data Preprocessing: 1-2 hours
- Model Implementation: 2-3 hours
- Evaluation and Submission: 1 hour
- Documentation: 1 hour

## Tools and Libraries
- Python 3.x
- pandas for data manipulation
- scikit-learn for machine learning models
- numpy for numerical operations
- matplotlib/seaborn for visualization
- joblib for model persistence