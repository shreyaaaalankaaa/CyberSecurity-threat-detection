import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

class DataProcessor:
    """
    Handles data loading, cleaning, and preprocessing for cybersecurity datasets
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def load_kdd_data(self, file_path):
        """
        Load KDD Cup 99 or NSL-KDD dataset
        """
        try:
            # KDD Cup 99 column names
            column_names = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
            ]
            
            # Load data
            data = pd.read_csv(file_path, names=column_names, header=None)
            
            # Remove trailing dots from labels if present
            data['label'] = data['label'].str.replace('.', '', regex=False)
            
            return data
            
        except Exception as e:
            logging.error(f"Error loading KDD data: {str(e)}")
            raise
    
    def clean_data(self, data):
        """
        Clean and preprocess the dataset
        """
        try:
            # Make a copy to avoid modifying original data
            cleaned_data = data.copy()
            
            # Remove duplicates
            initial_rows = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            logging.info(f"Removed {initial_rows - len(cleaned_data)} duplicate rows")
            
            # Handle missing values
            if cleaned_data.isnull().sum().sum() > 0:
                # For numerical columns, fill with median
                numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    if cleaned_data[col].isnull().sum() > 0:
                        cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
                
                # For categorical columns, fill with mode
                categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if cleaned_data[col].isnull().sum() > 0:
                        cleaned_data[col].fillna(cleaned_data[col].mode()[0], inplace=True)
            
            # Remove any infinite values
            cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
            cleaned_data = cleaned_data.dropna()
            
            return cleaned_data
            
        except Exception as e:
            logging.error(f"Error cleaning data: {str(e)}")
            raise
    
    def encode_categorical_features(self, data, fit=True):
        """
        Encode categorical features using LabelEncoder
        """
        try:
            encoded_data = data.copy()
            categorical_columns = ['protocol_type', 'service', 'flag']
            
            for column in categorical_columns:
                if column in encoded_data.columns:
                    if fit:
                        # Fit and transform
                        self.encoders[column] = LabelEncoder()
                        encoded_data[column] = self.encoders[column].fit_transform(encoded_data[column].astype(str))
                    else:
                        # Transform only using existing encoder
                        if column in self.encoders:
                            # Handle unseen categories
                            unique_values = set(self.encoders[column].classes_)
                            encoded_data[column] = encoded_data[column].astype(str)
                            encoded_data[column] = encoded_data[column].apply(
                                lambda x: x if x in unique_values else 'unknown'
                            )
                            
                            # Add 'unknown' to encoder if not present
                            if 'unknown' not in self.encoders[column].classes_:
                                self.encoders[column].classes_ = np.append(self.encoders[column].classes_, 'unknown')
                            
                            encoded_data[column] = self.encoders[column].transform(encoded_data[column])
            
            return encoded_data
            
        except Exception as e:
            logging.error(f"Error encoding categorical features: {str(e)}")
            raise
    
    def scale_numerical_features(self, data, fit=True):
        """
        Scale numerical features using StandardScaler
        """
        try:
            scaled_data = data.copy()
            
            # Identify numerical columns (excluding label)
            numerical_columns = scaled_data.select_dtypes(include=[np.number]).columns.tolist()
            if 'label' in numerical_columns:
                numerical_columns.remove('label')
            
            if fit:
                # Fit and transform
                self.scalers['numerical'] = StandardScaler()
                scaled_data[numerical_columns] = self.scalers['numerical'].fit_transform(scaled_data[numerical_columns])
            else:
                # Transform only using existing scaler
                if 'numerical' in self.scalers:
                    scaled_data[numerical_columns] = self.scalers['numerical'].transform(scaled_data[numerical_columns])
            
            return scaled_data
            
        except Exception as e:
            logging.error(f"Error scaling numerical features: {str(e)}")
            raise
    
    def prepare_labels(self, data):
        """
        Prepare labels for binary classification (normal vs attack)
        """
        try:
            # Convert to binary classification
            # 'normal' = 0, any attack type = 1
            labels = data['label'].apply(lambda x: 0 if x == 'normal' else 1)
            return labels
            
        except Exception as e:
            logging.error(f"Error preparing labels: {str(e)}")
            raise
    
    def process_data(self, data, fit=True):
        """
        Complete data processing pipeline
        """
        try:
            logging.info("Starting data processing pipeline")
            
            # Step 1: Clean data
            cleaned_data = self.clean_data(data)
            logging.info(f"Data cleaned. Shape: {cleaned_data.shape}")
            
            # Step 2: Separate features and labels
            if 'label' in cleaned_data.columns:
                labels = self.prepare_labels(cleaned_data)
                features = cleaned_data.drop('label', axis=1)
            else:
                labels = None
                features = cleaned_data
            
            # Step 3: Encode categorical features
            encoded_features = self.encode_categorical_features(features, fit=fit)
            logging.info("Categorical features encoded")
            
            # Step 4: Scale numerical features
            scaled_features = self.scale_numerical_features(encoded_features, fit=fit)
            logging.info("Numerical features scaled")
            
            # Store feature names
            if fit:
                self.feature_names = scaled_features.columns.tolist()
            
            return scaled_features, labels
            
        except Exception as e:
            logging.error(f"Error in data processing pipeline: {str(e)}")
            raise
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            logging.info(f"Data split completed:")
            logging.info(f"Training set: {X_train.shape[0]} samples")
            logging.info(f"Testing set: {X_test.shape[0]} samples")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error splitting data: {str(e)}")
            raise
    
    def get_feature_statistics(self, data):
        """
        Get comprehensive statistics about the dataset
        """
        try:
            stats = {
                'shape': data.shape,
                'columns': data.columns.tolist(),
                'dtypes': data.dtypes.to_dict(),
                'missing_values': data.isnull().sum().to_dict(),
                'numerical_summary': data.describe().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 0 else {},
                'categorical_summary': {col: data[col].value_counts().to_dict() 
                                     for col in data.select_dtypes(include=['object']).columns}
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting feature statistics: {str(e)}")
            raise
