import logging
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging configuration for the intrusion detection system
    """
    # Create logs directory if it doesn't exist
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging_config = {
        'level': log_level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if log_file:
        logging_config['filename'] = log_file
        logging_config['filemode'] = 'a'
    else:
        logging_config['stream'] = sys.stdout
    
    logging.basicConfig(**logging_config)
    
    # Set specific loggers to appropriate levels
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    
    return logger

def validate_data_format(data, required_columns=None):
    """
    Validate that data has the correct format for intrusion detection
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        if required_columns:
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for infinite or NaN values
        if data.isnull().sum().sum() > 0:
            logging.warning("Data contains NaN values")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(data[col]).sum() > 0:
                logging.warning(f"Column {col} contains infinite values")
        
        return True
        
    except Exception as e:
        logging.error(f"Data validation failed: {str(e)}")
        raise

def calculate_class_weights(y):
    """
    Calculate balanced class weights for imbalanced datasets
    """
    try:
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        weight_dict = dict(zip(classes, weights))
        
        logging.info(f"Calculated class weights: {weight_dict}")
        return weight_dict
        
    except Exception as e:
        logging.error(f"Error calculating class weights: {str(e)}")
        raise

def memory_usage_check(data, threshold_mb=1000):
    """
    Check memory usage of data and warn if it exceeds threshold
    """
    try:
        memory_mb = data.memory_usage(deep=True).sum() / 1024**2
        
        if memory_mb > threshold_mb:
            logging.warning(f"Data memory usage ({memory_mb:.2f} MB) exceeds threshold ({threshold_mb} MB)")
            return False
        else:
            logging.info(f"Data memory usage: {memory_mb:.2f} MB")
            return True
            
    except Exception as e:
        logging.error(f"Error checking memory usage: {str(e)}")
        return True

def save_results_to_json(results, filename):
    """
    Save model results to JSON file with proper serialization
    """
    try:
        # Convert numpy arrays and other non-serializable objects
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
        
        # Recursively convert results
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {key: deep_convert(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(item) for item in obj]
            else:
                return convert_for_json(obj)
        
        serializable_results = deep_convert(results)
        
        # Add metadata
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': serializable_results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logging.info(f"Results saved to {filename}")
        
    except Exception as e:
        logging.error(f"Error saving results to JSON: {str(e)}")
        raise

def load_results_from_json(filename):
    """
    Load model results from JSON file
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        logging.info(f"Results loaded from {filename}")
        return data.get('results', data)
        
    except Exception as e:
        logging.error(f"Error loading results from JSON: {str(e)}")
        raise

def generate_performance_summary(evaluation_results, target_accuracy=0.92):
    """
    Generate a summary of model performance against targets
    """
    try:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'target_accuracy': target_accuracy,
            'models_evaluated': len(evaluation_results),
            'performance_summary': {}
        }
        
        best_model = None
        best_accuracy = 0
        
        for model_name, results in evaluation_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                accuracy = metrics.get('accuracy', 0)
                
                model_summary = {
                    'accuracy': accuracy,
                    'meets_target': accuracy >= target_accuracy,
                    'accuracy_gap': target_accuracy - accuracy,
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'false_positive_rate': metrics.get('false_positive_rate', 0)
                }
                
                summary['performance_summary'][model_name] = model_summary
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
        
        summary['best_model'] = best_model
        summary['best_accuracy'] = best_accuracy
        summary['target_achieved'] = best_accuracy >= target_accuracy
        
        return summary
        
    except Exception as e:
        logging.error(f"Error generating performance summary: {str(e)}")
        raise

def create_feature_importance_report(model, feature_names, top_k=20):
    """
    Create a report of feature importance from a trained model
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            logging.warning("Model does not have feature_importances_ attribute")
            return None
        
        importances = model.feature_importances_
        
        if len(feature_names) != len(importances):
            logging.warning("Feature names and importances length mismatch")
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        # Create feature importance DataFrame
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Get top k features
        top_features = feature_df.head(top_k)
        
        report = {
            'total_features': len(importances),
            'top_features_count': min(top_k, len(importances)),
            'top_features': top_features.to_dict('records'),
            'importance_statistics': {
                'mean': float(np.mean(importances)),
                'std': float(np.std(importances)),
                'max': float(np.max(importances)),
                'min': float(np.min(importances))
            }
        }
        
        return report
        
    except Exception as e:
        logging.error(f"Error creating feature importance report: {str(e)}")
        return None

def format_performance_metrics(metrics):
    """
    Format performance metrics for display
    """
    try:
        formatted = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, np.integer)):
                formatted[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                if key in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']:
                    formatted[key] = f"{value:.4f} ({value*100:.2f}%)"
                else:
                    formatted[key] = f"{value:.6f}"
            else:
                formatted[key] = str(value)
        
        return formatted
        
    except Exception as e:
        logging.error(f"Error formatting metrics: {str(e)}")
        return metrics

def check_system_resources():
    """
    Check available system resources
    """
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        resources = {
            'cpu_usage_percent': cpu_percent,
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_usage_percent': memory.percent,
            'disk_total_gb': disk.total / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'disk_usage_percent': (disk.used / disk.total) * 100
        }
        
        # Check for resource constraints
        warnings = []
        if cpu_percent > 80:
            warnings.append("High CPU usage detected")
        if memory.percent > 80:
            warnings.append("High memory usage detected")
        if (disk.used / disk.total) * 100 > 80:
            warnings.append("Low disk space available")
        
        resources['warnings'] = warnings
        
        return resources
        
    except ImportError:
        logging.warning("psutil not available, cannot check system resources")
        return {"message": "Resource monitoring not available"}
    except Exception as e:
        logging.error(f"Error checking system resources: {str(e)}")
        return {"error": str(e)}

# Initialize logging on module import
logger = setup_logging()
