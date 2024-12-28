# 🤖 ML AutoBuilder For Beginners

A Streamlit application that automates the machine learning pipeline, enabling users to build and deploy ML models without writing code. This app guides users through data processing, feature engineering, model building, and prediction generation.

## Features

### 1. Data Upload & Processing
- Upload custom CSV datasets (max 10MB)
- Use provided example dataset
- Automatic data type detection
- Missing value detection and handling
- Support for numerical, categorical, and temporal features

### 2. Feature Engineering
- Interactive feature selection
- Numerical feature scaling options:
  - StandardScaler
  - MinMaxScaler
  - No scaling
- Categorical encoding options:
  - Label Encoding
  - One-Hot Encoding
  - No encoding
- Automatic temporal feature detection and processing

### 3. Model Building
- Automatic problem type detection (Classification/Regression)
- Supported Models:
  - Classification: Logistic Regression, Random Forest Classifier
  - Regression: Linear Regression, Random Forest Regressor
- Model performance metrics:
  - Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Regression: MAE, MSE, RMSE, R² Score
- Feature importance visualization
- Customizable train-test split

### 4. Predictions
- Generate predictions on new data
- Use test set for model evaluation
- Visualization of results:
  - Regression: Actual vs Predicted scatter plot
  - Classification: Confusion Matrix
- Export predictions to CSV
- Download trained model for future use