# Kaggle Competition: Predict Calorie Expenditure (Playground Series - Season 5, Episode 5)

## Project Overview

This repository contains a baseline machine learning model for the Kaggle competition "Predict Calorie Expenditure". The goal is to predict the number of calories burned during a workout based on various physiological and workout parameters.

## Competition Details

- **Task**: Regression problem to predict continuous target value (Calories burned)
- **Evaluation Metric**: Root Mean Squared Logarithmic Error (RMSLE)
- **Dataset**: Synthetic data generated from a deep learning model trained on the "Calories Burnt Prediction" dataset

## Data Description

The dataset contains the following features:
- `id`: Unique identifier for each entry
- `Gender`: Gender of the participant
- `Age`: Age of the participant in years
- `Height`: Height of the participant in cm
- `Weight`: Weight of the participant in kg
- `Duration`: Workout duration in minutes
- `Heart_Rate`: Average heart rate in beats per minute during the workout
- `Body_Temp`: Body temperature in Celsius during the workout
- `Calories`: Total calories burned (the target variable)

## Project Structure

```
project/
├── data/
│   ├── train_subsample.csv
│   ├── test_subsample.csv
│   └── full_test_for_submission.csv
├── models/
├── notebooks/
├── src/
├── results/
├── baseline_model_plan.md
├── model_architecture.md
├── requirements.md
└── README.md
```

## Baseline Model Approach

1. **Data Preprocessing**:
   - One-hot encoding for categorical variables
   - Handle missing values if any

2. **Model Selection**:
   - Linear Regression (simple baseline)
   - Random Forest Regressor (slightly more complex)

3. **Evaluation**:
   - Cross-validation with RMSLE metric
   - Model comparison and selection

4. **Submission**:
   - Generate predictions on test set
   - Format and save submission file

## Getting Started

1. Ensure all data files are in the `data/` directory
2. Install required dependencies (see `requirements.md`)
3. Run the preprocessing, training, and submission scripts

## Results

The baseline model provides a starting point for this competition. Further improvements can be made through:
- Feature engineering
- Hyperparameter tuning
- Advanced modeling techniques
- Ensemble methods

## Files Description

- `baseline_model_plan.md`: Detailed plan for the baseline model implementation
- `model_architecture.md`: Architecture diagrams and workflow explanation
- `requirements.md`: Project dependencies and installation instructions