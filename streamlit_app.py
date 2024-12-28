import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score)
import altair as alt
import time
import zipfile
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Page config
st.set_page_config(
    page_title='ML AutoBuilder For Beginners',
    page_icon='🤖',
    layout='wide'
)

# Session state initialization
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Main title
st.title('🤖 ML AutoBuilder For Beginners')

# Tabs for workflow
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Data Upload & Processing", 
    "2. Feature Engineering", 
    "3. Model Building", 
    "4. Predictions"
])

def get_numeric_categorical_columns(df):
    """Helper function to correctly identify numeric, temporal, and categorical columns"""
    # First try to convert potential datetime columns
    temporal_cols = []
    for col in df.columns:
        try:
            if df[col].dtype == 'object':
                pd.to_datetime(df[col], errors='raise')
                temporal_cols.append(col)
        except:
            continue
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
    
    # Get remaining columns as categorical (excluding temporal)
    categorical_cols = [col for col in df.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns 
                       if col not in temporal_cols]
    
    return list(numeric_cols), list(categorical_cols), list(temporal_cols)

# Data Upload & Processing
with tab1:
    st.header("Data Upload & Processing")
    
    # Data upload section
    upload_col, example_col = st.columns(2)
    
    with upload_col:
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload CSV file (max 10MB)",
            type=["csv"],
            help="Upload your dataset in CSV format"
        )
        
    with example_col:
        st.subheader("Use Example Data")
        use_example = st.checkbox("Load example dataset")
        
    # Data loading logic
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("Data uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    elif use_example:
        try:
            df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
            st.session_state.df = df
            st.success("Example data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading example data: {str(e)}")
            
    # If data is loaded, show data processing options
    if 'df' in st.session_state:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())
        
        # Add debug information about column types
        st.subheader("Column Types Information")
        column_info = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Type': st.session_state.df.dtypes,
            'Sample Values': [st.session_state.df[col].head(1).values[0] for col in st.session_state.df.columns]
        })
        st.dataframe(column_info)
        
        # Data processing options
        st.subheader("Data Processing Options")
        
        # Target variable selection
        target_col = st.selectbox(
            "Select target variable",
            options=st.session_state.df.columns
        )
        
        # Missing values handling
        st.subheader("Missing Values Treatment")
        # Check for both NaN and None values
        missing_values = st.session_state.df.isna().sum()  # This catches both NaN and None
        if missing_values.sum() > 0:
            # Show missing values summary
            missing_df = pd.DataFrame({
                'Column': missing_values.index,
                'Missing Count': missing_values.values,
                'Type': st.session_state.df[missing_values.index].dtypes,
                'Missing Percentage': (missing_values.values / len(st.session_state.df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            
            # Group columns by data type using the helper function
            numeric_cols, categorical_cols, temporal_cols = get_numeric_categorical_columns(st.session_state.df)
            
            # Filter for columns with missing values
            numeric_cols = [col for col in numeric_cols if col in missing_df['Column'].tolist()]
            categorical_cols = [col for col in categorical_cols if col in missing_df['Column'].tolist()]
            temporal_cols = [col for col in temporal_cols if col in missing_df['Column'].tolist()]
            
            st.write("Missing values summary:")
            # Format the missing percentage with % symbol
            missing_df['Missing Percentage'] = missing_df['Missing Percentage'].apply(lambda x: f"{x}%")
            st.dataframe(missing_df)
            
            # Batch imputation strategy
            col1, col2 = st.columns(2)
            
            with col1:
                if numeric_cols:
                    st.write("**Numeric Columns Strategy**")
                    numeric_strategy = st.radio(
                        "Choose strategy for numeric columns",
                        ["Mean", "Median", "Zero", "Drop rows"],
                        help="Mean: Replace with column mean\n"
                             "Median: Replace with column median\n"
                             "Zero: Replace with 0\n"
                             "Drop rows: Remove rows with missing values"
                    )
            
            with col2:
                if categorical_cols:
                    st.write("**Categorical Columns Strategy**")
                    categorical_strategy = st.radio(
                        "Choose strategy for categorical columns",
                        ["Mode", "Custom value", "Drop rows"],
                        help="Mode: Replace with most frequent value\n"
                             "Custom value: Replace with specified value\n"
                             "Drop rows: Remove rows with missing values"
                    )
            
            if categorical_cols and categorical_strategy == "Custom value":
                custom_value = st.text_input("Enter custom value for categorical columns")

            if st.button("Apply Imputation"):
                df_imputed = st.session_state.df.copy()
                
                # Apply numeric strategy
                if numeric_cols:
                    for col in numeric_cols:
                        if numeric_strategy == "Mean":
                            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
                        elif numeric_strategy == "Median":
                            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
                        elif numeric_strategy == "Zero":
                            df_imputed[col] = df_imputed[col].fillna(0)
                        else:  # Drop rows
                            df_imputed = df_imputed.dropna(subset=[col])
                
                # Apply categorical strategy
                if categorical_cols:
                    for col in categorical_cols:
                        if categorical_strategy == "Mode":
                            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])
                        elif categorical_strategy == "Custom value":
                            df_imputed[col] = df_imputed[col].fillna(custom_value)
                        else:  # Drop rows
                            df_imputed = df_imputed.dropna(subset=[col])
                
                # Update the session state with imputed data
                st.session_state.df = df_imputed
                
                # Show before/after comparison
                st.success("Imputation completed!")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Before imputation (missing values):")
                    st.write(missing_df)
                with col2:
                    st.write("After imputation (missing values):")
                    new_missing = pd.DataFrame({
                        'Column': st.session_state.df.isna().sum().index,
                        'Missing Count': st.session_state.df.isna().sum().values,
                        'Missing Percentage': (st.session_state.df.isna().sum().values / len(st.session_state.df) * 100).round(2)
                    })
                    new_missing = new_missing[new_missing['Missing Count'] > 0]
                    if len(new_missing) == 0:
                        st.write("No missing values remaining!")
                    else:
                        st.write(new_missing)

# Display all current session state in your Streamlit app.
st.write(st.session_state)

# Feature Engineering
with tab2:
    if 'df' not in st.session_state:
        st.warning("Please upload or select data in the Data Upload & Processing tab first.")
    else:
        st.header("Feature Selection Pool")

        # Get all available columns except target
        available_columns = [col for col in st.session_state.df.columns if col != target_col]

        # Multi-select for features without default selection
        selected_features_pool = st.multiselect(
            "Select features to use in your model",
            options=available_columns,
            help="Choose which columns to include as features in your model"
        )

        if not selected_features_pool:
            st.warning("Please select at least one feature to proceed with feature engineering.")
        else:
            # Store selected features in session state
            st.session_state.feature_pool = selected_features_pool
            
            # Continue with feature engineering only for selected columns
            st.header("Feature Engineering")
            
            # Update numerical and categorical columns based on selection
            numeric_cols, categorical_cols, temporal_cols = get_numeric_categorical_columns(st.session_state.df)
            numerical_columns = [col for col in selected_features_pool if col in numeric_cols]
            categorical_columns = [col for col in selected_features_pool if col in categorical_cols]
            temporal_columns = [col for col in selected_features_pool if col in temporal_cols]
            
            # Create columns for different feature engineering steps
            num_col, cat_col = st.columns(2)
            
            with num_col:
                st.subheader("Numerical Features")
                if numerical_columns:
                    st.write("Selected numerical features:", numerical_columns)
                    st.write("Select scaling method for numerical features:")
                    scaling_method = st.selectbox(
                        "Scaling Method",
                        ["None", "StandardScaler", "MinMaxScaler"],
                        help="StandardScaler: zero mean, unit variance\nMinMaxScaler: scale to range [0,1]"
                    )
                    
                    # Preview scaled data
                    if scaling_method != "None":
                        scaler = StandardScaler() if scaling_method == "StandardScaler" else MinMaxScaler()
                        scaled_data = scaler.fit_transform(st.session_state.df[numerical_columns])
                        scaled_df = pd.DataFrame(scaled_data, columns=numerical_columns)
                        
                        st.write("Preview of scaled numerical features:")
                        st.dataframe(scaled_df.head())
                        
                        # Store scaler in session state
                        st.session_state.scaler = scaler
                        st.session_state.scaled_numerical = scaled_df
                else:
                    st.info("No numerical features selected.")
            
            with cat_col:
                st.subheader("Categorical & Temporal Features")
                
                # Handle temporal features
                if temporal_columns:
                    st.write("**Temporal Features**")
                    st.write("The following columns were identified as temporal (datetime) features:", temporal_columns)
                    st.info("Temporal features will be converted to datetime format automatically.")
                    
                # Handle categorical features individually
                if categorical_columns:
                    st.write("**Categorical Features**")
                    st.write("Select encoding method for each categorical feature:")
                    
                    # Dictionary to store encoding choices
                    encoding_choices = {}
                    encoders = {}
                    encoded_data = pd.DataFrame()
                    
                    for col in categorical_columns:
                        st.write(f"\nFeature: **{col}**")
                        encoding_method = st.selectbox(
                            f"Encoding method for {col}",
                            ["None", "Label Encoding", "One-Hot Encoding"],
                            key=f"encoding_{col}",
                            help=f"Choose encoding method for {col}"
                        )
                        encoding_choices[col] = encoding_method
                        
                        if encoding_method != "None":
                            if encoding_method == "Label Encoding":
                                label_encoder = LabelEncoder()
                                encoded_col = label_encoder.fit_transform(st.session_state.df[col])
                                encoded_data[col] = encoded_col
                                encoders[col] = ('label', label_encoder)
                                
                                # Show value mapping
                                mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
                                st.write(f"Label mapping for {col}:", mapping)
                                
                            else:  # One-Hot Encoding
                                onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                                encoded_cols = onehot_encoder.fit_transform(st.session_state.df[[col]])
                                col_names = [f"{col}_{val}" for val in onehot_encoder.categories_[0]]
                                encoded_data = pd.concat([
                                    encoded_data,
                                    pd.DataFrame(encoded_cols, columns=col_names)
                                ], axis=1)
                                encoders[col] = ('onehot', onehot_encoder)
                    
                    if encoded_data.shape[1] > 0:
                        st.write("\nPreview of encoded categorical features:")
                        st.dataframe(encoded_data.head())
                        
                        # Store encoders and encoded data in session state
                        st.session_state.categorical_encoders = encoders
                        st.session_state.encoded_categorical = encoded_data
                else:
                    st.info("No categorical features selected.")
            
            # Proceed button
            if st.button("Proceed to Model Building"):
                if 'scaled_numerical' in st.session_state or 'encoded_categorical' in st.session_state:
                    st.session_state.data_processed = True
                    st.success("Feature engineering completed! You can now proceed to Model Building.")
                else:
                    st.warning("Please apply at least one transformation before proceeding.")

# Model Building
with tab3:
    if not st.session_state.data_processed:
        st.warning("Please complete the Feature Engineering step first.")
    else:
        st.header("Model Building")
        
        # Determine problem type based on target variable
        target_data = st.session_state.df[target_col]
        
        # Check if target is numeric and number of unique values
        is_numeric = np.issubdtype(target_data.dtype, np.number)
        unique_values = target_data.nunique()
        
        # Determine if classification or regression
        if not is_numeric or (is_numeric and unique_values <= 10):
            problem_type = "Classification"
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest Classifier": RandomForestClassifier()
            }
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        else:
            problem_type = "Regression"
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor()
            }
            metrics = ["MAE", "MSE", "RMSE", "R² Score"]
        
        st.write(f"**Detected Problem Type:** {problem_type}")
        
        # Model Selection
        st.subheader("Model Selection")
        selected_models = st.multiselect(
            "Select models to train",
            options=list(models.keys()),
            default=list(models.keys())[0]
        )
        
        # Train-Test Split Parameters
        st.subheader("Train-Test Split")
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", 0, 999, 42)
        
        # Prepare feature matrix
        X = pd.DataFrame()
        
        # Add scaled numerical features if they exist
        if 'scaled_numerical' in st.session_state:
            X = pd.concat([X, st.session_state.scaled_numerical], axis=1)
        
        # Add encoded categorical features if they exist
        if 'encoded_categorical' in st.session_state:
            X = pd.concat([X, st.session_state.encoded_categorical], axis=1)
        
        # Convert temporal features to datetime and extract features if they exist
        if temporal_columns:
            for col in temporal_columns:
                # Convert to datetime
                datetime_col = pd.to_datetime(st.session_state.df[col])
                # Extract useful features
                X[f"{col}_year"] = datetime_col.dt.year
                X[f"{col}_month"] = datetime_col.dt.month
                X[f"{col}_day"] = datetime_col.dt.day
                X[f"{col}_dayofweek"] = datetime_col.dt.dayofweek
        
        y = target_data
        
        if st.button("Train Models"):
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Dictionary to store results
            results = {}
            
            # Training progress bar
            progress_bar = st.progress(0)
            
            for i, model_name in enumerate(selected_models):
                model = models[model_name]
                
                # Train model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                if problem_type == "Classification":
                    results[model_name] = {
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred, average='weighted'),
                        "Recall": recall_score(y_test, y_pred, average='weighted'),
                        "F1-Score": f1_score(y_test, y_pred, average='weighted')
                    }
                    if len(np.unique(y)) == 2:  # Binary classification
                        results[model_name]["ROC-AUC"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                else:
                    results[model_name] = {
                        "MAE": mean_squared_error(y_test, y_pred),
                        "MSE": mean_squared_error(y_test, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "R² Score": r2_score(y_test, y_pred)
                    }
                
                # Store model in session state
                st.session_state[f"model_{model_name}"] = model
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(selected_models))
            
            # Display results
            st.subheader("Model Performance Comparison")
            results_df = pd.DataFrame(results).round(4)
            st.dataframe(results_df)
            
            # Plot performance comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            results_df.plot(kind='bar', ax=ax)
            plt.title("Model Performance Comparison")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Store results in session state
            st.session_state.model_results = results
            st.session_state.model_trained = True
            
            # Feature importance for Random Forest models
            for model_name in selected_models:
                if "Random Forest" in model_name:
                    st.subheader(f"{model_name} Feature Importance")
                    model = st.session_state[f"model_{model_name}"]
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')
                    plt.title(f"Top 10 Feature Importance - {model_name}")
                    st.pyplot(fig)
