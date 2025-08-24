# Overview

This is a comprehensive Cybersecurity Intrusion Detection System built with Python and Streamlit. The application provides an end-to-end machine learning pipeline for detecting network intrusions and cyber threats, targeting 92% accuracy. It supports multiple ML algorithms including Random Forest, Gradient Boosting, SVM, and Logistic Regression, with capabilities for data preprocessing, feature engineering, model training, evaluation, and real-time prediction through an interactive web interface.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Application**: Single-page application with sidebar navigation providing five main sections: Data Overview, Feature Engineering, Model Training, Model Evaluation, and Real-time Prediction
- **Interactive Visualizations**: Uses Plotly for dynamic charts and graphs, with matplotlib/seaborn for statistical plots
- **Session State Management**: Maintains application state across page interactions, storing trained models and results in Streamlit session state

## Backend Architecture
- **Modular Component Design**: Six core modules handling distinct responsibilities:
  - `DataProcessor`: Data loading, cleaning, and preprocessing with sklearn scalers and encoders
  - `FeatureEngineer`: Advanced feature creation including statistical features, ratio calculations, and feature selection
  - `ModelTrainer`: Multi-algorithm training with hyperparameter optimization using GridSearchCV
  - `ModelEvaluator`: Comprehensive metrics calculation including accuracy, precision, recall, F1-score, and ROC analysis
  - `Predictor`: Real-time prediction interface for individual samples
  - `DatasetLoader`: Dataset management supporting KDD Cup 99/NSL-KDD and synthetic data generation

## Data Processing Pipeline
- **Dataset Support**: Primary focus on KDD Cup 99 and NSL-KDD datasets with 41 standardized features
- **Attack Classification**: Five-category classification system (normal, dos, r2l, u2r, probe)
- **Feature Engineering**: Statistical transformations, ratio features, log transformations, and error rate aggregations
- **Data Preprocessing**: StandardScaler for numerical features, LabelEncoder for categorical features

## Machine Learning Architecture
- **Multi-Model Training**: Supports Random Forest, Gradient Boosting, SVM, and Logistic Regression with automated hyperparameter tuning
- **Cross-Validation**: Stratified k-fold validation with configurable fold count
- **Model Persistence**: Pickle-based serialization for trained models and preprocessing components
- **Performance Evaluation**: Matthews Correlation Coefficient, ROC curves, confusion matrices, and comprehensive classification reports

## Error Handling and Logging
- **Comprehensive Logging**: Structured logging system with configurable levels and file output
- **Data Validation**: Input validation for DataFrame format, required columns, and data integrity
- **Exception Management**: Try-catch blocks throughout the pipeline with informative error messages

# External Dependencies

## Core ML and Data Science Libraries
- **scikit-learn**: Primary machine learning framework for model training, evaluation, and preprocessing
- **pandas/numpy**: Data manipulation and numerical computation
- **plotly**: Interactive visualization library for charts and graphs
- **matplotlib/seaborn**: Statistical plotting and visualization

## Web Framework
- **streamlit**: Web application framework providing the user interface and session management

## Data Processing
- **pickle**: Model serialization and persistence
- **urllib**: Dataset downloading capabilities for remote data sources
- **gzip**: Compressed file handling for dataset archives

## Development and Utilities
- **logging**: Built-in Python logging for system monitoring and debugging
- **datetime**: Timestamp management for logging and data tracking
- **json**: Configuration and metadata handling

Note: The system is designed to work with cybersecurity datasets but includes synthetic data generation capabilities for demonstration purposes when real datasets are not available.