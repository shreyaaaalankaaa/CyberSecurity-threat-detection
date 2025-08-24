import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator
import pickle

class Predictor:
    """
    Real-time prediction interface for trained intrusion detection models
    """
    
    def __init__(self, model=None, feature_engineer=None):
        self.model = model
        self.feature_engineer = feature_engineer
        self.prediction_history = []
        
    def load_model(self, model_path):
        """
        Load a trained model from disk
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            if not self.model:
                raise ValueError("No model found in the loaded file")
            
            logging.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_sample(self, sample_data):
        """
        Predict on a single network sample
        
        Args:
            sample_data: pandas DataFrame with single row or dict with feature values
            
        Returns:
            tuple: (prediction, confidence_score)
        """
        try:
            if self.model is None:
                raise ValueError("No model loaded. Please load a model first.")
            
            # Convert to DataFrame if needed
            if isinstance(sample_data, dict):
                sample_df = pd.DataFrame([sample_data])
            elif isinstance(sample_data, pd.Series):
                sample_df = pd.DataFrame([sample_data])
            else:
                sample_df = sample_data.copy()
            
            # Apply feature engineering if available
            if self.feature_engineer is not None:
                try:
                    processed_sample = self.feature_engineer.transform(sample_df)
                except Exception as e:
                    logging.warning(f"Feature engineering failed, using original features: {str(e)}")
                    processed_sample = sample_df
            else:
                processed_sample = sample_df
            
            # Make prediction
            prediction = self.model.predict(processed_sample)[0]
            
            # Get confidence score
            confidence_score = 0.5  # Default confidence
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(processed_sample)[0]
                    confidence_score = max(probabilities)  # Confidence is the max probability
                except Exception as e:
                    logging.warning(f"Could not get prediction probabilities: {str(e)}")
            elif hasattr(self.model, 'decision_function'):
                try:
                    decision_score = self.model.decision_function(processed_sample)[0]
                    # Convert decision score to confidence (0-1 range)
                    confidence_score = 1 / (1 + np.exp(-abs(decision_score)))
                except Exception as e:
                    logging.warning(f"Could not get decision function score: {str(e)}")
            
            # Store prediction in history
            prediction_record = {
                'timestamp': pd.Timestamp.now(),
                'prediction': int(prediction),
                'confidence': float(confidence_score),
                'input_shape': processed_sample.shape
            }
            self.prediction_history.append(prediction_record)
            
            # Keep only last 1000 predictions in memory
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            return int(prediction), float(confidence_score)
            
        except Exception as e:
            logging.error(f"Error in predict_sample: {str(e)}")
            raise
    
    def predict_batch(self, batch_data):
        """
        Predict on a batch of network samples
        
        Args:
            batch_data: pandas DataFrame with multiple rows
            
        Returns:
            dict: predictions, confidences, and summary statistics
        """
        try:
            if self.model is None:
                raise ValueError("No model loaded. Please load a model first.")
            
            # Apply feature engineering if available
            if self.feature_engineer is not None:
                try:
                    processed_batch = self.feature_engineer.transform(batch_data)
                except Exception as e:
                    logging.warning(f"Feature engineering failed, using original features: {str(e)}")
                    processed_batch = batch_data
            else:
                processed_batch = batch_data
            
            # Make predictions
            predictions = self.model.predict(processed_batch)
            
            # Get confidence scores
            confidences = np.full(len(predictions), 0.5)  # Default confidence
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(processed_batch)
                    confidences = np.max(probabilities, axis=1)
                except Exception as e:
                    logging.warning(f"Could not get prediction probabilities: {str(e)}")
            elif hasattr(self.model, 'decision_function'):
                try:
                    decision_scores = self.model.decision_function(processed_batch)
                    confidences = 1 / (1 + np.exp(-np.abs(decision_scores)))
                except Exception as e:
                    logging.warning(f"Could not get decision function scores: {str(e)}")
            
            # Calculate summary statistics
            num_attacks = np.sum(predictions == 1)
            num_normal = np.sum(predictions == 0)
            attack_rate = num_attacks / len(predictions)
            avg_confidence = np.mean(confidences)
            
            results = {
                'predictions': predictions.tolist(),
                'confidences': confidences.tolist(),
                'summary': {
                    'total_samples': len(predictions),
                    'attacks_detected': int(num_attacks),
                    'normal_traffic': int(num_normal),
                    'attack_rate': float(attack_rate),
                    'average_confidence': float(avg_confidence),
                    'high_confidence_predictions': int(np.sum(confidences > 0.8)),
                    'low_confidence_predictions': int(np.sum(confidences < 0.6))
                }
            }
            
            # Store batch predictions in history
            for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                prediction_record = {
                    'timestamp': pd.Timestamp.now(),
                    'prediction': int(pred),
                    'confidence': float(conf),
                    'batch_id': len(self.prediction_history),
                    'sample_id': i
                }
                self.prediction_history.append(prediction_record)
            
            # Keep only last 1000 predictions in memory
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            return results
            
        except Exception as e:
            logging.error(f"Error in predict_batch: {str(e)}")
            raise
    
    def get_prediction_statistics(self):
        """
        Get statistics about recent predictions
        """
        try:
            if not self.prediction_history:
                return {"message": "No predictions made yet"}
            
            df = pd.DataFrame(self.prediction_history)
            
            # Recent predictions (last 100 or all if less)
            recent_df = df.tail(100)
            
            stats = {
                'total_predictions': len(self.prediction_history),
                'recent_predictions': len(recent_df),
                'attack_rate': float(recent_df['prediction'].mean()),
                'average_confidence': float(recent_df['confidence'].mean()),
                'high_confidence_rate': float((recent_df['confidence'] > 0.8).mean()),
                'low_confidence_rate': float((recent_df['confidence'] < 0.6).mean()),
                'predictions_per_class': {
                    'normal': int((recent_df['prediction'] == 0).sum()),
                    'attack': int((recent_df['prediction'] == 1).sum())
                }
            }
            
            # Time-based statistics if we have enough data
            if len(recent_df) > 10:
                # Calculate prediction rate (predictions per minute)
                time_diff = (recent_df['timestamp'].max() - recent_df['timestamp'].min()).total_seconds()
                if time_diff > 0:
                    stats['predictions_per_minute'] = float(len(recent_df) / (time_diff / 60))
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting prediction statistics: {str(e)}")
            return {"error": str(e)}
    
    def create_sample_from_features(self, **kwargs):
        """
        Create a sample from individual feature values
        
        This is useful for manual input in the web interface
        """
        try:
            # Define default values for all expected features
            default_features = {
                'duration': 0.0,
                'protocol_type': 'tcp',
                'service': 'http',
                'flag': 'SF',
                'src_bytes': 0,
                'dst_bytes': 0,
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': 1,
                'srv_count': 1,
                'serror_rate': 0.0,
                'srv_serror_rate': 0.0,
                'rerror_rate': 0.0,
                'srv_rerror_rate': 0.0,
                'same_srv_rate': 1.0,
                'diff_srv_rate': 0.0,
                'srv_diff_host_rate': 0.0,
                'dst_host_count': 1,
                'dst_host_srv_count': 1,
                'dst_host_same_srv_rate': 1.0,
                'dst_host_diff_srv_rate': 0.0,
                'dst_host_same_src_port_rate': 0.0,
                'dst_host_srv_diff_host_rate': 0.0,
                'dst_host_serror_rate': 0.0,
                'dst_host_srv_serror_rate': 0.0,
                'dst_host_rerror_rate': 0.0,
                'dst_host_srv_rerror_rate': 0.0
            }
            
            # Update with provided values
            default_features.update(kwargs)
            
            # Create DataFrame
            sample_df = pd.DataFrame([default_features])
            
            return sample_df
            
        except Exception as e:
            logging.error(f"Error creating sample from features: {str(e)}")
            raise
    
    def get_model_info(self):
        """
        Get information about the loaded model
        """
        try:
            if self.model is None:
                return {"message": "No model loaded"}
            
            info = {
                'model_type': type(self.model).__name__,
                'has_predict_proba': hasattr(self.model, 'predict_proba'),
                'has_decision_function': hasattr(self.model, 'decision_function'),
                'feature_engineer_available': self.feature_engineer is not None
            }
            
            # Try to get additional model-specific information
            if hasattr(self.model, 'n_estimators'):
                info['n_estimators'] = self.model.n_estimators
            
            if hasattr(self.model, 'feature_importances_'):
                info['has_feature_importances'] = True
                info['n_features'] = len(self.model.feature_importances_)
            
            if hasattr(self.model, 'classes_'):
                info['classes'] = self.model.classes_.tolist()
            
            return info
            
        except Exception as e:
            logging.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}
    
    def clear_history(self):
        """
        Clear prediction history
        """
        self.prediction_history = []
        logging.info("Prediction history cleared")
