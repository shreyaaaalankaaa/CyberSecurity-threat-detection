import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import logging
import time

class ModelTrainer:
    """
    Train and optimize multiple machine learning models for intrusion detection
    """
    
    def __init__(self, use_hyperparameter_tuning=True, cv_folds=5, test_size=0.2, random_state=42):
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.training_scores = {}
        
    def get_model_configs(self):
        """
        Get model configurations with hyperparameter grids
        """
        configs = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'SVM': {
                'model': SVC(random_state=self.random_state, probability=True),
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 1],
                    'kernel': ['rbf', 'linear'],
                    'class_weight': ['balanced', None]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'param_grid': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': ['balanced', None]
                }
            }
        }
        
        return configs
    
    def train_single_model(self, model_name, X_train, y_train):
        """
        Train a single model with or without hyperparameter tuning
        """
        try:
            logging.info(f"Training {model_name}...")
            configs = self.get_model_configs()
            
            if model_name not in configs:
                raise ValueError(f"Unknown model: {model_name}")
            
            model_config = configs[model_name]
            base_model = model_config['model']
            param_grid = model_config['param_grid']
            
            start_time = time.time()
            
            if self.use_hyperparameter_tuning and len(param_grid) > 0:
                # Hyperparameter tuning with GridSearchCV
                cv_strategy = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                                            random_state=self.random_state)
                
                # Reduce param grid size for SVM to avoid long training times
                if model_name == 'SVM':
                    param_grid = {
                        'C': [0.1, 1, 10],
                        'gamma': ['scale', 0.1],
                        'kernel': ['rbf'],
                        'class_weight': ['balanced']
                    }
                
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=cv_strategy,
                    scoring='accuracy',
                    n_jobs=-1 if model_name != 'SVM' else 1,  # SVM can be memory intensive
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                self.best_params[model_name] = grid_search.best_params_
                
                logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                
            else:
                # Train with default parameters
                best_model = base_model
                best_model.fit(X_train, y_train)
                self.best_params[model_name] = "Default parameters used"
            
            # Cross-validation scores
            cv_strategy = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                                        random_state=self.random_state)
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_strategy, 
                                      scoring='accuracy', n_jobs=-1)
            
            training_time = time.time() - start_time
            
            # Store results
            self.models[model_name] = best_model
            self.training_scores[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'training_time': training_time,
                'best_params': self.best_params[model_name]
            }
            
            logging.info(f"{model_name} trained successfully in {training_time:.2f} seconds")
            logging.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return best_model, self.training_scores[model_name]
            
        except Exception as e:
            logging.error(f"Error training {model_name}: {str(e)}")
            raise
    
    def train_models(self, X_train, y_train, selected_models, progress_callback=None):
        """
        Train multiple selected models
        """
        try:
            logging.info(f"Starting training for models: {selected_models}")
            
            trained_models = {}
            results = {}
            
            for i, model_name in enumerate(selected_models):
                try:
                    # Update progress
                    if progress_callback:
                        progress_callback((i + 1) / len(selected_models))
                    
                    # Train individual model
                    model, scores = self.train_single_model(model_name, X_train, y_train)
                    
                    trained_models[model_name] = model
                    results[model_name] = scores
                    
                    # Add accuracy to results for easy access
                    results[model_name]['accuracy'] = scores['cv_mean']
                    
                except Exception as e:
                    logging.error(f"Failed to train {model_name}: {str(e)}")
                    # Continue with other models
                    continue
            
            # Find best model based on CV accuracy
            if results:
                best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
                logging.info(f"Best model: {best_model_name} with CV accuracy: {results[best_model_name]['cv_mean']:.4f}")
            
            return trained_models, results
            
        except Exception as e:
            logging.error(f"Error in train_models: {str(e)}")
            raise
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models on test set
        """
        try:
            evaluation_results = {}
            
            for model_name, model in self.models.items():
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Calculate false positive rate
                tn = np.sum((y_test == 0) & (y_pred == 0))
                fp = np.sum((y_test == 0) & (y_pred == 1))
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                evaluation_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'false_positive_rate': fpr,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logging.info(f"{model_name} - Test Accuracy: {accuracy:.4f}, "
                           f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                           f"F1: {f1:.4f}, FPR: {fpr:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            logging.error(f"Error evaluating models: {str(e)}")
            raise
    
    def save_models(self, filepath_prefix="model"):
        """
        Save trained models to disk
        """
        try:
            saved_files = []
            
            for model_name, model in self.models.items():
                filename = f"{filepath_prefix}_{model_name.replace(' ', '_').lower()}.pkl"
                
                with open(filename, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'model_name': model_name,
                        'best_params': self.best_params.get(model_name),
                        'training_scores': self.training_scores.get(model_name)
                    }, f)
                
                saved_files.append(filename)
                logging.info(f"Model {model_name} saved to {filename}")
            
            return saved_files
            
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
            raise
    
    def load_model(self, filepath):
        """
        Load a saved model from disk
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            model_name = model_data['model_name']
            self.models[model_name] = model_data['model']
            self.best_params[model_name] = model_data.get('best_params', {})
            self.training_scores[model_name] = model_data.get('training_scores', {})
            
            logging.info(f"Model {model_name} loaded from {filepath}")
            return model_data['model']
            
        except Exception as e:
            logging.error(f"Error loading model from {filepath}: {str(e)}")
            raise
    
    def get_model_summary(self):
        """
        Get a summary of all trained models
        """
        try:
            summary = {}
            
            for model_name in self.models.keys():
                summary[model_name] = {
                    'training_accuracy': self.training_scores.get(model_name, {}).get('cv_mean', 'N/A'),
                    'training_std': self.training_scores.get(model_name, {}).get('cv_std', 'N/A'),
                    'training_time': self.training_scores.get(model_name, {}).get('training_time', 'N/A'),
                    'best_parameters': self.best_params.get(model_name, 'N/A'),
                    'model_type': type(self.models[model_name]).__name__
                }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error getting model summary: {str(e)}")
            raise
