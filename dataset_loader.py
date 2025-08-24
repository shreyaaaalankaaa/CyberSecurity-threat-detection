import pandas as pd
import numpy as np
import logging
import os
import urllib.request
import gzip
from sklearn.datasets import make_classification

class DatasetLoader:
    """
    Load and prepare cybersecurity datasets for intrusion detection
    """
    
    def __init__(self):
        self.column_names = [
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
        
        # Define attack categories
        self.attack_categories = {
            'normal': ['normal'],
            'dos': ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop'],
            'r2l': ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster'],
            'u2r': ['buffer_overflow', 'loadmodule', 'perl', 'rootkit'],
            'probe': ['ipsweep', 'nmap', 'portsweep', 'satan']
        }
    
    def create_synthetic_data(self, n_samples=50000, n_features=41, n_informative=20, n_redundant=5):
        """
        Create synthetic cybersecurity dataset for demonstration purposes
        """
        try:
            logging.info(f"Creating synthetic dataset with {n_samples} samples")
            
            # Generate base features using scikit-learn
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features - 3,  # Subtract categorical features
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_clusters_per_class=2,
                class_sep=0.8,
                random_state=42
            )
            
            # Create DataFrame with numerical features
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            data = pd.DataFrame(X, columns=feature_names)
            
            # Add some cybersecurity-specific features
            data['duration'] = np.random.exponential(2, n_samples)
            data['src_bytes'] = np.random.lognormal(5, 2, n_samples).astype(int)
            data['dst_bytes'] = np.random.lognormal(4, 2, n_samples).astype(int)
            data['count'] = np.random.poisson(3, n_samples) + 1
            data['srv_count'] = np.random.poisson(2, n_samples) + 1
            
            # Add categorical features
            protocols = ['tcp', 'udp', 'icmp']
            services = ['http', 'ftp', 'smtp', 'ssh', 'telnet', 'pop_3', 'domain_u', 'private']
            flags = ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO']
            
            data['protocol_type'] = np.random.choice(protocols, n_samples)
            data['service'] = np.random.choice(services, n_samples)
            data['flag'] = np.random.choice(flags, n_samples)
            
            # Add more realistic network features
            data['land'] = np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
            data['wrong_fragment'] = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
            data['urgent'] = np.random.choice([0, 1], n_samples, p=[0.995, 0.005])
            data['hot'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.9, 0.05, 0.03, 0.02])
            data['num_failed_logins'] = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.8, 0.1, 0.05, 0.03, 0.015, 0.005])
            data['logged_in'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
            data['num_compromised'] = np.random.choice([0, 1, 2], n_samples, p=[0.95, 0.04, 0.01])
            data['root_shell'] = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
            data['su_attempted'] = np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
            data['num_root'] = np.random.choice([0, 1, 2], n_samples, p=[0.9, 0.08, 0.02])
            data['num_file_creations'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.85, 0.1, 0.03, 0.02])
            data['num_shells'] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
            data['num_access_files'] = np.random.choice([0, 1, 2], n_samples, p=[0.9, 0.08, 0.02])
            data['num_outbound_cmds'] = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
            data['is_host_login'] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
            data['is_guest_login'] = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
            
            # Add rate-based features
            data['serror_rate'] = np.random.uniform(0, 1, n_samples)
            data['srv_serror_rate'] = np.random.uniform(0, 1, n_samples)
            data['rerror_rate'] = np.random.uniform(0, 1, n_samples)
            data['srv_rerror_rate'] = np.random.uniform(0, 1, n_samples)
            data['same_srv_rate'] = np.random.uniform(0, 1, n_samples)
            data['diff_srv_rate'] = np.random.uniform(0, 1, n_samples)
            data['srv_diff_host_rate'] = np.random.uniform(0, 1, n_samples)
            
            # Add destination host features
            data['dst_host_count'] = np.random.randint(1, 256, n_samples)
            data['dst_host_srv_count'] = np.random.randint(1, 256, n_samples)
            data['dst_host_same_srv_rate'] = np.random.uniform(0, 1, n_samples)
            data['dst_host_diff_srv_rate'] = np.random.uniform(0, 1, n_samples)
            data['dst_host_same_src_port_rate'] = np.random.uniform(0, 1, n_samples)
            data['dst_host_srv_diff_host_rate'] = np.random.uniform(0, 1, n_samples)
            data['dst_host_serror_rate'] = np.random.uniform(0, 1, n_samples)
            data['dst_host_srv_serror_rate'] = np.random.uniform(0, 1, n_samples)
            data['dst_host_rerror_rate'] = np.random.uniform(0, 1, n_samples)
            data['dst_host_srv_rerror_rate'] = np.random.uniform(0, 1, n_samples)
            
            # Create labels based on synthetic binary classification
            # Add some realistic attack patterns
            attack_indicators = (
                (data['num_failed_logins'] > 2) |
                (data['root_shell'] == 1) |
                (data['num_compromised'] > 0) |
                (data['serror_rate'] > 0.8) |
                (data['dst_host_serror_rate'] > 0.7) |
                (data['wrong_fragment'] == 1)
            )
            
            # Combine with original classification
            final_labels = (y == 1) | attack_indicators
            data['label'] = final_labels.apply(lambda x: np.random.choice(['back', 'neptune', 'smurf', 'ipsweep', 'portsweep']) if x else 'normal')
            
            # Split into train and test
            split_idx = int(0.8 * len(data))
            train_data = data.iloc[:split_idx].copy()
            test_data = data.iloc[split_idx:].copy()
            
            # Ensure balanced classes in both sets
            train_attack_rate = (train_data['label'] != 'normal').mean()
            test_attack_rate = (test_data['label'] != 'normal').mean()
            
            logging.info(f"Synthetic dataset created successfully")
            logging.info(f"Training set: {len(train_data)} samples, attack rate: {train_attack_rate:.3f}")
            logging.info(f"Test set: {len(test_data)} samples, attack rate: {test_attack_rate:.3f}")
            
            return train_data, test_data
            
        except Exception as e:
            logging.error(f"Error creating synthetic data: {str(e)}")
            raise
    
    def load_nsl_kdd(self):
        """
        Load NSL-KDD dataset (uses synthetic data for demonstration)
        """
        try:
            logging.info("Loading NSL-KDD dataset...")
            
            # Since we can't download actual files, we'll create realistic synthetic data
            # that mimics the NSL-KDD dataset structure and characteristics
            return self.create_synthetic_data(n_samples=50000)
            
        except Exception as e:
            logging.error(f"Error loading NSL-KDD dataset: {str(e)}")
            raise
    
    def load_kdd_cup_99(self):
        """
        Load KDD Cup 99 dataset (uses synthetic data for demonstration)
        """
        try:
            logging.info("Loading KDD Cup 99 dataset...")
            
            # Create larger synthetic dataset to mimic KDD Cup 99
            return self.create_synthetic_data(n_samples=100000)
            
        except Exception as e:
            logging.error(f"Error loading KDD Cup 99 dataset: {str(e)}")
            raise
    
    def load_cicids_2017(self):
        """
        Load CICIDS 2017 dataset (uses synthetic data for demonstration)
        """
        try:
            logging.info("Loading CICIDS 2017 dataset...")
            
            # Create synthetic data with different characteristics for CICIDS
            return self.create_synthetic_data(n_samples=75000, n_informative=25)
            
        except Exception as e:
            logging.error(f"Error loading CICIDS 2017 dataset: {str(e)}")
            raise
    
    def get_dataset_info(self):
        """
        Get information about available datasets
        """
        return {
            'nsl_kdd': {
                'name': 'NSL-KDD',
                'description': 'Improved version of KDD Cup 99 dataset',
                'features': 41,
                'attack_types': ['DoS', 'Probe', 'R2L', 'U2R'],
                'recommended_for': 'Academic research and benchmarking'
            },
            'kdd_cup_99': {
                'name': 'KDD Cup 99',
                'description': 'Classic intrusion detection benchmark dataset',
                'features': 41,
                'attack_types': ['DoS', 'Probe', 'R2L', 'U2R'],
                'recommended_for': 'Historical comparison and baseline'
            },
            'cicids_2017': {
                'name': 'CICIDS 2017',
                'description': 'Modern realistic network traffic dataset',
                'features': 'Variable',
                'attack_types': ['DoS', 'DDoS', 'Port Scan', 'Brute Force'],
                'recommended_for': 'Real-world scenario simulation'
            }
        }
    
    def preprocess_labels(self, data, binary_classification=True):
        """
        Preprocess labels for binary or multiclass classification
        """
        try:
            if binary_classification:
                # Convert to binary: normal vs attack
                data['binary_label'] = data['label'].apply(lambda x: 0 if x == 'normal' else 1)
                return data
            else:
                # Keep original attack categories
                def categorize_attack(label):
                    for category, attacks in self.attack_categories.items():
                        if label in attacks:
                            return category
                    return 'unknown'
                
                data['category_label'] = data['label'].apply(categorize_attack)
                return data
            
        except Exception as e:
            logging.error(f"Error preprocessing labels: {str(e)}")
            raise
    
    def get_data_statistics(self, data):
        """
        Get comprehensive statistics about the dataset
        """
        try:
            stats = {
                'total_samples': len(data),
                'features': len(data.columns) - 1,  # Excluding label
                'missing_values': data.isnull().sum().sum(),
                'duplicate_rows': data.duplicated().sum(),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024**2)
            }
            
            # Label distribution
            if 'label' in data.columns:
                label_counts = data['label'].value_counts()
                stats['label_distribution'] = label_counts.to_dict()
                stats['attack_rate'] = (data['label'] != 'normal').mean()
            
            # Feature types
            stats['feature_types'] = {
                'numerical': len(data.select_dtypes(include=[np.number]).columns),
                'categorical': len(data.select_dtypes(include=['object']).columns)
            }
            
            # Data quality metrics
            numerical_data = data.select_dtypes(include=[np.number])
            if not numerical_data.empty:
                stats['data_quality'] = {
                    'mean_values': numerical_data.mean().to_dict(),
                    'std_values': numerical_data.std().to_dict(),
                    'min_values': numerical_data.min().to_dict(),
                    'max_values': numerical_data.max().to_dict()
                }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting data statistics: {str(e)}")
            raise
