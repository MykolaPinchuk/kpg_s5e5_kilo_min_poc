# Project Requirements

## Python Dependencies

- pandas>=1.3.0
- scikit-learn>=1.0.0
- numpy>=1.20.0
- matplotlib>=3.3.0
- seaborn>=0.11.0
- joblib>=1.0.0

## System Requirements

- Python 3.7 or higher
- At least 4GB RAM (8GB recommended)
- At least 1GB free disk space

## Installation Instructions

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Required Data Files

The following data files should be present in the `data/` directory:
- `train_subsample.csv`
- `test_subsample.csv`
- `full_test_for_submission.csv`

## Development Tools

- Jupyter Notebook (for exploratory data analysis)
- VS Code or any Python IDE
- Git (for version control)