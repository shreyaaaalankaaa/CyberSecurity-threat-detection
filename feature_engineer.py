import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import logging

class FeatureEngineer:
    """
    Advanced feature engineering for cybersecurity intrusion detection
    """
    
    def __init__(self):
        self.feature_selector = None
        self.pca = None
        self.poly_features = None
        self.feature_importance_scores = None
        self.selected_features = None
        self.feature_names = []
        
    def create_statistical_features(self, data):
        """
        Create statistical features from existing network data
        """
        try:
            statistical_data = data.copy()
            
            # Ratio features
            statistical_data['bytes_ratio'] = (statistical_data['src_bytes'] + 1) / (statistical_data['dst_bytes'] + 1)
            statistical_data['service_count_ratio'] = (statistical_data['srv_count'] + 1) / (statistical_data['count'] + 1)
            
            # Log transformations for highly skewed features
            log_features = ['src_bytes', 'dst_bytes', 'count', 'srv_count']
            for feature in log_features:
                if feature in statistical_data.columns:
                    statistical_data[f'{feature}_log'] = np.log1p(statistical_data[feature])
            
            # Binary indicators
            statistical_data['has_src_bytes'] = (statistical_data['src_bytes'] > 0).astype(int)
            statistical_data['has_dst_bytes'] = (statistical_data['dst_bytes'] > 0).astype(int)
            statistical_data['is_logged_in'] = statistical_data.get('logged_in', 0)
            
            # Error rate aggregations
            error_features = ['serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate']
            available_error_features = [f for f in error_features if f in statistical_data.columns]
            if available_error_features:
                statistical_data['total_error_rate'] = statistical_data[available_error_features].sum(axis=1)
                statistical_data['avg_error_rate'] = statistical_data[available_error_features].mean(axis=1)
            
            # Host-based features aggregation
            host_features = [col for col in statistical_data.columns if 'dst_host' in col]
            if len(host_features) > 0:
                statistical_data['dst_host_avg'] = statistical_data[host_features].mean(axis=1)
                statistical_data['dst_host_max'] = statistical_data[host_features].max(axis=1)
            
            logging.info(f"Created statistical features. New shape: {statistical_data.shape}")
            return statistical_data
            
        except Exception as e:
            logging.error(f"Error creating statistical features: {str(e)}")
            raise
    
    def create_temporal_features(self, data):
        """
        Create temporal features based on duration and timing patterns
        """
        try:
            temporal_data = data.copy()
            
            # Duration-based features
            if 'duration' in temporal_data.columns:
                # Categorize duration
                temporal_data['duration_category'] = pd.cut(
                    temporal_data['duration'],
                    bins=[-1, 0, 1, 10, 100, float('inf')],
                    labels=[0, 1, 2, 3, 4]
                ).astype(int)
                
                # Duration indicators
                temporal_data['is_short_duration'] = (temporal_data['duration'] <= 1).astype(int)
                temporal_data['is_long_duration'] = (temporal_data['duration'] > 100).astype(int)
                temporal_data['zero_duration'] = (temporal_data['duration'] == 0).astype(int)
            
            # Connection frequency features
            if 'count' in temporal_data.columns:
                temporal_data['high_frequency'] = (temporal_data['count'] > 10).astype(int)
                temporal_data['single_connection'] = (temporal_data['count'] == 1).astype(int)
            
            logging.info(f"Created temporal features. New shape: {temporal_data.shape}")
            return temporal_data
            
        except Exception as e:
            logging.error(f"Error creating temporal features: {str(e)}")
            raise
    
    def create_behavioral_features(self, data):
        """
        Create behavioral features that capture attack patterns
        """
        try:
            behavioral_data = data.copy()
            
            # Authentication-related features
            auth_features = ['num_failed_logins', 'logged_in', 'is_guest_login']
            available_auth = [f for f in auth_features if f in behavioral_data.columns]
            if available_auth:
                behavioral_data['auth_risk_score'] = behavioral_data[available_auth].sum(axis=1)
            
            # Privilege escalation indicators
            priv_features = ['root_shell', 'su_attempted', 'num_root']
            available_priv = [f for f in priv_features if f in behavioral_data.columns]
            if available_priv:
                behavioral_data['privilege_risk'] = behavioral_data[available_priv].sum(axis=1)
            
            # File system access patterns
            file_features = ['num_file_creations', 'num_shells', 'num_access_files']
            available_file = [f for f in file_features if f in behavioral_data.columns]
            if available_file:
                behavioral_data['file_activity'] = behavioral_data[available_file].sum(axis=1)
            
            # Network anomaly indicators
            if 'wrong_fragment' in behavioral_data.columns and 'urgent' in behavioral_data.columns:
                behavioral_data['network_anomaly'] = behavioral_data['wrong_fragment'] + behavioral_data['urgent']
            
            # Service consistency
            if 'same_srv_rate' in behavioral_data.columns and 'diff_srv_rate' in behavioral_data.columns:
                behavioral_data['service_consistency'] = behavioral_data['same_srv_rate'] - behavioral_data['diff_srv_rate']
            
            logging.info(f"Created behavioral features. New shape: {behavioral_data.shape}")
            return behavioral_data
            
        except Exception as e:
            logging.error(f"Error creating behavioral features: {str(e)}")
            raise
    
    def create_interaction_features(self, data, max_interactions=10):
        """
        Create interaction features between important variables
        """
        try:
            interaction_data = data.copy()
            
            # Important feature pairs for interactions
            important_pairs = [
                ('src_bytes', 'dst_bytes'),
                ('count', 'srv_count'),
                ('serror_rate', 'rerror_rate'),
                ('same_srv_rate', 'diff_srv_rate')
            ]
            
            interaction_count = 0
            for feature1, feature2 in important_pairs:
                if interaction_count >= max_interactions:
                    break
                    
                if feature1 in interaction_data.columns and feature2 in interaction_data.columns:
                    # Multiplication interaction
                    interaction_data[f'{feature1}_{feature2}_mult'] = (
                        interaction_data[feature1] * interaction_data[feature2]
                    )
                    
                    # Ratio interaction (avoid division by zero)
                    interaction_data[f'{feature1}_{feature2}_ratio'] = (
                        interaction_data[feature1] / (interaction_data[feature2] + 1e-8)
                    )
                    
                    interaction_count += 2
            
            logging.info(f"Created {interaction_count} interaction features")
            return interaction_data
            
        except Exception as e:
            logging.error(f"Error creating interaction features: {str(e)}")
            raise
    
    def select_features(self, X, y, method='random_forest', k=30):
        """
        Select the most important features using various methods
        """
        try:
            if method == 'random_forest':
                # Use RandomForest feature importance
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                
                feature_importance = rf.feature_importances_
                self.feature_importance_scores = feature_importance
                
                # Select top k features
                top_indices = np.argsort(feature_importance)[::-1][:k]
                self.selected_features = top_indices
                
                selected_X = X.iloc[:, top_indices] if isinstance(X, pd.DataFrame) else X[:, top_indices]
                
            elif method == 'univariate':
                # Use univariate statistical test
                self.feature_selector = SelectKBest(score_func=f_classif, k=k)
                selected_X = self.feature_selector.fit_transform(X, y)
                self.selected_features = self.feature_selector.get_support(indices=True)
                
            elif method == 'rfe':
                # Use Recursive Feature Elimination
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                self.feature_selector = RFE(estimator=rf, n_features_to_select=k, step=1)
                selected_X = self.feature_selector.fit_transform(X, y)
                self.selected_features = self.feature_selector.get_support(indices=True)
            
            else:
                raise ValueError(f"Unknown feature selection method: {method}")
            
            logging.info(f"Selected {len(self.selected_features)} features using {method} method")
            return selected_X
            
        except Exception as e:
            logging.error(f"Error in feature selection: {str(e)}")
            raise
    
    def apply_pca(self, X, n_components=0.95):
        """
        Apply PCA for dimensionality reduction
        """
        try:
            self.pca = PCA(n_components=n_components, random_state=42)
            X_pca = self.pca.fit_transform(X)
            
            logging.info(f"PCA applied. Original features: {X.shape[1]}, "
                        f"PCA components: {X_pca.shape[1]}")
            logging.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
            
            return X_pca
            
        except Exception as e:
            logging.error(f"Error applying PCA: {str(e)}")
            raise
    
    def fit_transform(self, data):
        """
        Complete feature engineering pipeline - fit and transform
        """
        try:
            logging.info("Starting feature engineering pipeline (fit_transform)")
            
            # Separate features from labels
            if 'label' in data.columns:
                X = data.drop('label', axis=1)
                y = data['label'].apply(lambda x: 0 if x == 'normal' else 1)
            else:
                X = data
                y = None
            
            # Step 1: Create statistical features
            X_statistical = self.create_statistical_features(X)
            
            # Step 2: Create temporal features
            X_temporal = self.create_temporal_features(X_statistical)
            
            # Step 3: Create behavioral features
            X_behavioral = self.create_behavioral_features(X_temporal)
            
            # Step 4: Create interaction features
            X_interactions = self.create_interaction_features(X_behavioral)
            
            # Store feature names
            self.feature_names = X_interactions.columns.tolist()
            
            # Step 5: Feature selection (if labels are available)
            if y is not None:
                X_selected = self.select_features(X_interactions, y, method='random_forest', k=50)
                X_final = pd.DataFrame(X_selected, index=X_interactions.index)
            else:
                X_final = X_interactions
            
            logging.info(f"Feature engineering completed. Final shape: {X_final.shape}")
            return X_final
            
        except Exception as e:
            logging.error(f"Error in fit_transform: {str(e)}")
            raise
    
    def transform(self, data):
        """
        Transform new data using fitted transformers
        """
        try:
            logging.info("Starting feature engineering pipeline (transform)")
            
            # Separate features from labels
            if 'label' in data.columns:
                X = data.drop('label', axis=1)
            else:
                X = data
            
            # Apply the same transformations as in fit_transform
            X_statistical = self.create_statistical_features(X)
            X_temporal = self.create_temporal_features(X_statistical)
            X_behavioral = self.create_behavioral_features(X_temporal)
            X_interactions = self.create_interaction_features(X_behavioral)
            
            # Apply feature selection if fitted
            if self.selected_features is not None:
                if isinstance(X_interactions, pd.DataFrame):
                    X_selected = X_interactions.iloc[:, self.selected_features]
                else:
                    X_selected = X_interactions[:, self.selected_features]
                X_final = pd.DataFrame(X_selected, index=X_interactions.index)
            else:
                X_final = X_interactions
            
            logging.info(f"Transform completed. Final shape: {X_final.shape}")
            return X_final
            
        except Exception as e:
            logging.error(f"Error in transform: {str(e)}")
            raise
    
    def get_feature_importance(self):
        """
        Get feature importance scores if available
        """
        return self.feature_importance_scores
    
    def get_selected_features(self):
        """
        Get indices of selected features
        """
        return self.selected_features
    
    def get_feature_names(self):
        """
        Get names of engineered features
        """
        return self.feature_names
