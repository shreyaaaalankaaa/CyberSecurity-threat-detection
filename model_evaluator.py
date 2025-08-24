import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

class ModelEvaluator:
    """
    Comprehensive model evaluation for intrusion detection systems
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.baseline_metrics = {}
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive evaluation metrics
        """
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Matthews Correlation Coefficient
            mcc = matthews_corrcoef(y_true, y_pred)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'false_positive_rate': false_positive_rate,
                'false_negative_rate': false_negative_rate,
                'matthews_corrcoef': mcc,
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            }
            
            # ROC AUC if probabilities are available
            if y_pred_proba is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    metrics['roc_auc'] = roc_auc
                    metrics['roc_fpr'] = fpr.tolist()
                    metrics['roc_tpr'] = tpr.tolist()
                    
                    # Precision-Recall curve
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
                    pr_auc = auc(recall_curve, precision_curve)
                    metrics['pr_auc'] = pr_auc
                    metrics['pr_precision'] = precision_curve.tolist()
                    metrics['pr_recall'] = recall_curve.tolist()
                    
                except Exception as e:
                    logging.warning(f"Could not calculate ROC/PR curves: {str(e)}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def calculate_baseline_metrics(self, y_true):
        """
        Calculate baseline metrics for comparison
        """
        try:
            # Majority class baseline
            majority_class = 1 if np.sum(y_true) > len(y_true) / 2 else 0
            majority_pred = np.full_like(y_true, majority_class)
            
            # Random baseline
            np.random.seed(42)
            class_ratio = np.mean(y_true)
            random_pred = np.random.choice([0, 1], size=len(y_true), p=[1-class_ratio, class_ratio])
            
            self.baseline_metrics = {
                'majority_class': self.calculate_metrics(y_true, majority_pred),
                'random': self.calculate_metrics(y_true, random_pred)
            }
            
            return self.baseline_metrics
            
        except Exception as e:
            logging.error(f"Error calculating baseline metrics: {str(e)}")
            raise
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Comprehensive evaluation of a single model
        """
        try:
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                except:
                    pass
            elif hasattr(model, 'decision_function'):
                try:
                    y_pred_proba = model.decision_function(X_test)
                    # Normalize to [0,1] range
                    y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
                except:
                    pass
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Store results
            self.evaluation_results[model_name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'model': model
            }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating model {model_name}: {str(e)}")
            raise
    
    def compare_models(self, models, X_test, y_test):
        """
        Compare multiple models
        """
        try:
            comparison_results = {}
            
            for model_name, model in models.items():
                metrics = self.evaluate_model(model, X_test, y_test, model_name)
                comparison_results[model_name] = metrics
            
            # Calculate baseline for comparison
            self.calculate_baseline_metrics(y_test)
            
            return comparison_results
            
        except Exception as e:
            logging.error(f"Error comparing models: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model"):
        """
        Plot confusion matrix
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'Attack'],
                       yticklabels=['Normal', 'Attack'])
            
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'Confusion Matrix - {model_name}')
            
            # Add percentage annotations
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1f}%)', 
                           ha='center', va='center', fontsize=10, color='gray')
            
            return fig
            
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {str(e)}")
            raise
    
    def plot_roc_curve(self, model_results):
        """
        Plot ROC curves for multiple models
        """
        try:
            fig = go.Figure()
            
            for model_name, results in model_results.items():
                metrics = results.get('metrics', {})
                if 'roc_fpr' in metrics and 'roc_tpr' in metrics:
                    fpr = metrics['roc_fpr']
                    tpr = metrics['roc_tpr']
                    auc_score = metrics.get('roc_auc', 0)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{model_name} (AUC = {auc_score:.3f})',
                        line=dict(width=2)
                    ))
            
            # Add random classifier line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_layout(
                title='ROC Curves Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=600, height=500
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error plotting ROC curve: {str(e)}")
            raise
    
    def plot_precision_recall_curve(self, model_results):
        """
        Plot Precision-Recall curves for multiple models
        """
        try:
            fig = go.Figure()
            
            for model_name, results in model_results.items():
                metrics = results.get('metrics', {})
                if 'pr_precision' in metrics and 'pr_recall' in metrics:
                    precision = metrics['pr_precision']
                    recall = metrics['pr_recall']
                    auc_score = metrics.get('pr_auc', 0)
                    
                    fig.add_trace(go.Scatter(
                        x=recall, y=precision,
                        mode='lines',
                        name=f'{model_name} (AUC = {auc_score:.3f})',
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title='Precision-Recall Curves Comparison',
                xaxis_title='Recall',
                yaxis_title='Precision',
                width=600, height=500
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error plotting PR curve: {str(e)}")
            raise
    
    def generate_evaluation_report(self, model_results, target_accuracy=0.92, target_fpr_reduction=0.18):
        """
        Generate comprehensive evaluation report
        """
        try:
            report = {
                'summary': {},
                'detailed_metrics': {},
                'performance_analysis': {},
                'recommendations': []
            }
            
            # Summary statistics
            accuracies = [results['metrics']['accuracy'] for results in model_results.values()]
            fprs = [results['metrics']['false_positive_rate'] for results in model_results.values()]
            
            report['summary'] = {
                'total_models_evaluated': len(model_results),
                'best_accuracy': max(accuracies),
                'average_accuracy': np.mean(accuracies),
                'lowest_fpr': min(fprs),
                'average_fpr': np.mean(fprs)
            }
            
            # Detailed metrics for each model
            report['detailed_metrics'] = model_results
            
            # Performance analysis
            best_model = max(model_results.keys(), key=lambda x: model_results[x]['metrics']['accuracy'])
            best_accuracy = model_results[best_model]['metrics']['accuracy']
            best_fpr = model_results[best_model]['metrics']['false_positive_rate']
            
            # Calculate FPR reduction compared to baseline
            baseline_fpr = self.baseline_metrics.get('majority_class', {}).get('false_positive_rate', 0.1)
            fpr_reduction = (baseline_fpr - best_fpr) / baseline_fpr if baseline_fpr > 0 else 0
            
            report['performance_analysis'] = {
                'best_model': best_model,
                'accuracy_target_met': best_accuracy >= target_accuracy,
                'accuracy_gap': target_accuracy - best_accuracy,
                'fpr_reduction_achieved': fpr_reduction,
                'fpr_target_met': fpr_reduction >= target_fpr_reduction,
                'fpr_reduction_gap': target_fpr_reduction - fpr_reduction
            }
            
            # Recommendations
            if best_accuracy < target_accuracy:
                report['recommendations'].append(
                    f"Consider feature engineering or ensemble methods to improve accuracy by {target_accuracy - best_accuracy:.3f}"
                )
            
            if fpr_reduction < target_fpr_reduction:
                report['recommendations'].append(
                    f"Focus on reducing false positives. Current reduction: {fpr_reduction:.3f}, target: {target_fpr_reduction}"
                )
            
            if best_accuracy >= target_accuracy and fpr_reduction >= target_fpr_reduction:
                report['recommendations'].append("Performance targets achieved! Consider deployment.")
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating evaluation report: {str(e)}")
            raise
    
    def plot_metrics_comparison(self, model_results):
        """
        Plot comparison of key metrics across models
        """
        try:
            models = list(model_results.keys())
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=metrics_to_plot,
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            for i, metric in enumerate(metrics_to_plot):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                values = [model_results[model]['metrics'][metric] for model in models]
                
                fig.add_trace(
                    go.Bar(x=models, y=values, name=metric.replace('_', ' ').title()),
                    row=row, col=col
                )
            
            fig.update_layout(
                title="Model Performance Comparison",
                showlegend=False,
                height=600
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error plotting metrics comparison: {str(e)}")
            raise
