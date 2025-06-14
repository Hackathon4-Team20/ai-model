import pandas as pd
import numpy as np
from satisfaction_tracker import SatisfactionTracker
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

class ModelEvaluator:
    def __init__(self, openrouter_api_key: str):
        self.tracker = SatisfactionTracker(openrouter_api_key=openrouter_api_key)
        self.results = []
        self.setup_english_matplotlib()
        
    def setup_english_matplotlib(self):
        """
        Setup matplotlib for English display
        """
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.facecolor'] = 'white'
        
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data = []
        for line in lines:
            if '\t' in line:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    text = parts[0]
                    label = parts[1]
                    data.append({'text': text, 'label': label})
        
        return pd.DataFrame(data)
    
    def map_labels_to_satisfaction(self, label: str) -> int:
        """
        Map labels to satisfaction scores (1-5)
        """
        label_mapping = {
            'POS': 4,      # Positive
            'NEG': 2,      # Negative  
            'NEUTRAL': 3,  # Neutral
            'OBJ': 3       # Objective (neutral)
        }
        return label_mapping.get(label, 3)
    
    def predict_satisfaction(self, text: str) -> Dict:
        """
        Predict satisfaction score for text
        """
        self.tracker = SatisfactionTracker(openrouter_api_key=self.tracker.api_key)
        result = self.tracker.add_message('user', text)
        return result
    
    def evaluate_sample_enhanced(self, df: pd.DataFrame, sample_size: int = 50) -> Dict:
        """
        Enhanced sampling to ensure diverse representation and minimum 10+ data points
        """
        print(f"Original dataset distribution:")
        print(df['label'].value_counts())
        print(f"Original dataset percentages:")
        print(df['label'].value_counts(normalize=True) * 100)
        
        # Ensure minimum sample size for meaningful analysis
        min_sample_size = max(sample_size, 20)  # At least 20 samples
        
        # Enhanced stratified sampling
        sample_dfs = []
        
        # Get samples from each class with guaranteed diversity
        for label in ['NEG', 'POS', 'NEUTRAL', 'OBJ']:  # Prioritize order for diversity
            if label in df['label'].values:
                label_data = df[df['label'] == label]
                
                # Calculate samples per class (minimum 2, maximum available)
                if label == 'NEG':
                    # Ensure we get negative samples for better correlation
                    samples_needed = min(max(5, min_sample_size // 6), len(label_data))
                elif label == 'POS':
                    # Ensure we get positive samples
                    samples_needed = min(max(5, min_sample_size // 6), len(label_data))
                else:
                    # Neutral and objective
                    samples_needed = min(max(3, min_sample_size // 8), len(label_data))
                
                if len(label_data) > 0 and samples_needed > 0:
                    # Use different random states for variety
                    random_state = hash(label) % 1000
                    label_sample = label_data.sample(n=samples_needed, 
                                                   random_state=random_state, 
                                                   replace=False)
                    sample_dfs.append(label_sample)
                    print(f"Selected {samples_needed} samples from {label} class")
        
        # Fill remaining slots if needed
        if sample_dfs:
            current_size = sum(len(df) for df in sample_dfs)
            if current_size < min_sample_size:
                remaining_needed = min_sample_size - current_size
                used_indices = pd.concat(sample_dfs).index if sample_dfs else pd.Index([])
                remaining_df = df.drop(used_indices)
                
                if len(remaining_df) > 0:
                    additional_sample = remaining_df.sample(
                        n=min(remaining_needed, len(remaining_df)), 
                        random_state=123
                    )
                    sample_dfs.append(additional_sample)
                    print(f"Added {len(additional_sample)} additional diverse samples")
        
        # Combine all samples
        if sample_dfs:
            sample_df = pd.concat(sample_dfs, ignore_index=True)
        else:
            # Fallback: random sampling
            sample_df = df.sample(n=min(min_sample_size, len(df)), random_state=42)
        
        # Shuffle the final sample
        sample_df = sample_df.sample(frac=1, random_state=456).reset_index(drop=True)
        
        print(f"\nFinal sample distribution:")
        print(sample_df['label'].value_counts())
        print(f"Final sample percentages:")
        print(sample_df['label'].value_counts(normalize=True) * 100)
        
        true_scores = []
        predicted_scores = []
        predictions_details = []
        
        print(f"\nStarting evaluation of {len(sample_df)} samples...")
        
        for idx, row in sample_df.iterrows():
            text = row['text']
            true_label = row['label']
            true_score = self.map_labels_to_satisfaction(true_label)
            
            try:
                prediction = self.predict_satisfaction(text)
                predicted_score = prediction['updated_score']
                
                true_scores.append(true_score)
                predicted_scores.append(predicted_score)
                
                predictions_details.append({
                    'text': text,
                    'true_label': true_label,
                    'true_score': true_score,
                    'predicted_score': predicted_score,
                    'prediction_status': prediction['status'],
                    'reason': prediction['reason']
                })
                
                # Progress indicator every 5 samples
                progress = len(predictions_details)
                if progress % 5 == 0 or progress == len(sample_df):
                    print(f"Progress: {progress}/{len(sample_df)} ({progress/len(sample_df)*100:.1f}%)")
                    print(f"  Sample {progress} - Label: {true_label} | True: {true_score} | Predicted: {predicted_score}")
                
            except Exception as e:
                print(f"Error processing text: {text[:50]}... - {e}")
                continue
        
        print(f"\nCompleted evaluation with {len(true_scores)} successful predictions")
        print(f"Score diversity - True scores: {set(true_scores)}")
        print(f"Score diversity - Predicted scores: {set(predicted_scores)}")
        
        return {
            'true_scores': true_scores,
            'predicted_scores': predicted_scores,
            'details': predictions_details,
            'sample_distribution': sample_df['label'].value_counts().to_dict()
        }
    
    def calculate_comprehensive_metrics(self, true_scores: List[int], predicted_scores: List[int]) -> Dict:
        """
        Calculate comprehensive evaluation metrics with better error handling
        """
        # Convert scores to binary classification (satisfied/unsatisfied)
        true_binary = [1 if score >= 3 else 0 for score in true_scores]
        pred_binary = [1 if score >= 3 else 0 for score in predicted_scores]
        
        # Basic metrics
        accuracy = accuracy_score(true_binary, pred_binary)
        precision = precision_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary, zero_division=0)
        f1 = f1_score(true_binary, pred_binary, zero_division=0)
        
        # Regression metrics for score prediction
        mae = np.mean(np.abs(np.array(true_scores) - np.array(predicted_scores)))
        rmse = np.sqrt(np.mean((np.array(true_scores) - np.array(predicted_scores)) ** 2))
        
        # Enhanced correlation calculation
        try:
            true_array = np.array(true_scores)
            pred_array = np.array(predicted_scores)
            
            # Check for variance in both arrays
            if len(set(true_scores)) > 1 and len(set(predicted_scores)) > 1:
                if np.var(true_array) > 0 and np.var(pred_array) > 0:
                    correlation = np.corrcoef(true_array, pred_array)[0, 1]
                    if np.isnan(correlation) or np.isinf(correlation):
                        correlation = 0.0
                else:
                    correlation = 0.0
            else:
                correlation = 0.0
                
            print(f"Correlation calculation - True variance: {np.var(true_array):.3f}, Pred variance: {np.var(pred_array):.3f}")
            print(f"Correlation coefficient: {correlation:.3f}")
            
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            correlation = 0.0
        
        # Confusion matrix components
        cm = confusion_matrix(true_binary, pred_binary)
        
        # Handle different confusion matrix sizes
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:
            # Only one class present
            if len(set(true_binary)) == 1:
                if true_binary[0] == 1:
                    tp, tn, fp, fn = cm[0, 0], 0, 0, 0
                else:
                    tp, tn, fp, fn = 0, cm[0, 0], 0, 0
            else:
                tp, tn, fp, fn = 0, 0, 0, 0
        else:
            tp, tn, fp, fn = 0, 0, 0, 0
        
        # Specificity (True Negative Rate) with error handling
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'confusion_matrix': cm,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'classification_report': classification_report(true_binary, pred_binary, 
                                                         target_names=['Unsatisfied', 'Satisfied'])
        }
    
    def plot_comprehensive_results(self, true_scores: List[int], predicted_scores: List[int], metrics: Dict):
        """
        Create comprehensive visualization WITHOUT Precision-Recall curve
        """
        fig = plt.figure(figsize=(18, 12))
        
        # Convert to binary for some plots
        true_binary = [1 if score >= 3 else 0 for score in true_scores]
        pred_binary = [1 if score >= 3 else 0 for score in predicted_scores]
        
        # 1. Enhanced Confusion Matrix
        ax1 = plt.subplot(2, 4, 1)
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Unsatisfied', 'Satisfied'], 
                   yticklabels=['Unsatisfied', 'Satisfied'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Values', fontsize=12)
        plt.xlabel('Predicted Values', fontsize=12)
        
        # 2. Metrics Bar Chart
        ax2 = plt.subplot(2, 4, 2)
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                        metrics['f1_score']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8)
        plt.title('Classification Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Score Distribution Comparison
        ax3 = plt.subplot(2, 4, 3)
        plt.hist(true_scores, alpha=0.7, label='True Scores', bins=5, color='skyblue', edgecolor='black')
        plt.hist(predicted_scores, alpha=0.7, label='Predicted Scores', bins=5, color='orange', edgecolor='black')
        plt.title('Score Distribution Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Satisfaction Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Enhanced Scatter Plot with more points and better visualization
        ax4 = plt.subplot(2, 4, 4)
        
        # Add small random jitter to avoid overlapping points
        true_jittered = np.array(true_scores) + np.random.normal(0, 0.05, len(true_scores))
        pred_jittered = np.array(predicted_scores) + np.random.normal(0, 0.05, len(predicted_scores))
        
        # Create scatter plot with larger, more visible points
        scatter = plt.scatter(true_jittered, pred_jittered, 
                            alpha=0.7, s=80, color='purple', 
                            edgecolors='black', linewidth=0.5)
        
        # Add perfect prediction line
        plt.plot([1, 5], [1, 5], 'r--', label='Perfect Prediction', linewidth=2)
        
        # Add trend line if correlation exists
        if abs(metrics['correlation']) > 0.1:
            z = np.polyfit(true_scores, predicted_scores, 1)
            p = np.poly1d(z)
            plt.plot([1, 5], [p(1), p(5)], 'g-', alpha=0.8, linewidth=2, 
                    label=f'Trend Line (r={metrics["correlation"]:.3f})')
        
        plt.title('True vs Predicted Values', fontsize=14, fontweight='bold')
        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.5, 5.5)
        plt.ylim(0.5, 5.5)
        
        # Enhanced correlation display
        correlation_text = f'{metrics["correlation"]:.3f}' if not np.isnan(metrics["correlation"]) else 'N/A'
        correlation_color = 'lightgreen' if abs(metrics["correlation"]) > 0.3 else 'yellow'
        plt.text(0.05, 0.95, f'Correlation: {correlation_text}\nPoints: {len(true_scores)}', 
                transform=ax4.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=correlation_color, alpha=0.7))
        
        # 5. Error Distribution
        ax5 = plt.subplot(2, 4, 5)
        errors = np.array(predicted_scores) - np.array(true_scores)
        plt.hist(errors, bins=min(10, len(errors)//2 + 1), edgecolor='black', alpha=0.7, color='lightcoral')
        plt.title('Error Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Error (Predicted - True)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(x=0, color='red', linestyle='--', label='No Error', linewidth=2)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add MAE and RMSE
        plt.text(0.05, 0.95, f'MAE: {metrics["mae"]:.3f}\nRMSE: {metrics["rmse"]:.3f}', 
                transform=ax5.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # 6. Class Distribution
        ax6 = plt.subplot(2, 4, 6)
        true_counts = [sum(1 for x in true_binary if x == 0), sum(1 for x in true_binary if x == 1)]
        pred_counts = [sum(1 for x in pred_binary if x == 0), sum(1 for x in pred_binary if x == 1)]
        
        x = np.arange(2)
        width = 0.35
        plt.bar(x - width/2, true_counts, width, label='True', color='skyblue', alpha=0.8)
        plt.bar(x + width/2, pred_counts, width, label='Predicted', color='orange', alpha=0.8)
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(x, ['Unsatisfied', 'Satisfied'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. REMOVED - Precision-Recall Curve
        # This space is now empty
        
        # 8. Performance Summary Table
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('tight')
        ax8.axis('off')
        
        table_data = [
            ['Metric', 'Value'],
            ['Accuracy', f'{metrics["accuracy"]:.3f}'],
            ['Precision', f'{metrics["precision"]:.3f}'],
            ['Recall', f'{metrics["recall"]:.3f}'],
            ['F1-Score', f'{metrics["f1_score"]:.3f}'],
            ['MAE', f'{metrics["mae"]:.3f}'],
            ['RMSE', f'{metrics["rmse"]:.3f}'],
            ['True Positive', f'{metrics["tp"]}'],
            ['True Negative', f'{metrics["tn"]}'],
            ['False Positive', f'{metrics["fp"]}'],
            ['False Negative', f'{metrics["fn"]}']
        ]
        
        table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.5, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        
        # Style the header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style confusion matrix rows
        for i in range(7, 11):
            table[(i, 0)].set_facecolor('#E3F2FD')
            table[(i, 1)].set_facecolor('#E3F2FD')
        
        plt.title('Performance Summary', fontsize=14, fontweight='bold', pad=5)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.35, wspace=0.3)
        plt.savefig('comprehensive_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_detailed_results(self, details: List[Dict], filename: str = 'detailed_results.csv'):
        """
        Save detailed results
        """
        df_results = pd.DataFrame(details)
        df_results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"Detailed results saved to {filename}")
        
    def generate_comprehensive_report(self, metrics: Dict, sample_size: int, sample_distribution: Dict = None):
        """
        Generate comprehensive evaluation report
        """
        distribution_text = ""
        if sample_distribution:
            distribution_lines = []
            for label, count in sample_distribution.items():
                percentage = count/sample_size*100
                distribution_lines.append(f"- {label}: {count} samples ({percentage:.1f}%)")
            distribution_text = "\nSAMPLE DISTRIBUTION:\n" + "\n".join(distribution_lines) + "\n"
        
        report = "=" * 70 + "\n"
        report += "COMPREHENSIVE SENTIMENT ANALYSIS MODEL EVALUATION REPORT\n"
        report += "=" * 70 + "\n\n"
        
        report += "EVALUATION INFORMATION:\n"
        report += f"- Sample Size: {sample_size}\n"
        report += "- Model Type: Satisfaction Tracker with LLM Backend\n"
        report += distribution_text + "\n"
        
        report += "CLASSIFICATION METRICS:\n"
        report += f"- Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)\n"
        report += f"- Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)\n"
        report += f"- Recall (Sensitivity): {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)\n"
        report += f"- F1-Score: {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)\n\n"
        
        report += "REGRESSION METRICS (Score Prediction):\n"
        report += f"- Mean Absolute Error (MAE): {metrics['mae']:.3f}\n"
        report += f"- Root Mean Square Error (RMSE): {metrics['rmse']:.3f}\n"
        report += f"- Correlation Coefficient: {metrics['correlation']:.3f}\n\n"
        
        report += "CONFUSION MATRIX BREAKDOWN:\n"
        report += f"- True Positives (TP): {metrics['tp']}\n"
        report += f"- True Negatives (TN): {metrics['tn']}\n"
        report += f"- False Positives (FP): {metrics['fp']}\n"
        report += f"- False Negatives (FN): {metrics['fn']}\n\n"
        
        report += "PERFORMANCE INTERPRETATION:\n"
        report += "- Accuracy < 70%: Needs significant improvement\n"
        report += "- Accuracy 70-85%: Acceptable performance\n"
        report += "- Accuracy > 85%: Good performance\n"
        report += "- F1-Score > 0.8: Strong balanced performance\n"
        report += "- Correlation > 0.7: Strong linear relationship\n\n"
        
        report += "MODEL STRENGTHS:\n"
        report += self._analyze_strengths(metrics) + "\n\n"
        
        report += "AREAS FOR IMPROVEMENT:\n"
        report += self._analyze_weaknesses(metrics) + "\n\n"
        
        report += "DETAILED CLASSIFICATION REPORT:\n"
        report += metrics['classification_report'] + "\n\n"
        
        report += "=" * 70 + "\n"
        
        print(report)
        
        with open('comprehensive_evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
    
    def _analyze_strengths(self, metrics: Dict) -> str:
        """Analyze model strengths"""
        strengths = []
        if metrics['accuracy'] > 0.85:
            strengths.append("- High overall accuracy indicates strong performance")
        if metrics['precision'] > 0.8:
            strengths.append("- High precision means low false positive rate")
        if metrics['recall'] > 0.8:
            strengths.append("- High recall indicates good detection of positive cases")
        if metrics['f1_score'] > 0.8:
            strengths.append("- Excellent F1-score shows balanced precision-recall performance")
        if abs(metrics['correlation']) > 0.7:
            strengths.append("- Strong correlation between predicted and actual scores")
        
        return '\n'.join(strengths) if strengths else "- Model shows room for improvement across all metrics"
    
    def _analyze_weaknesses(self, metrics: Dict) -> str:
        """Analyze model weaknesses"""
        weaknesses = []
        if metrics['accuracy'] < 0.7:
            weaknesses.append("- Low accuracy suggests fundamental prediction issues")
        if metrics['precision'] < 0.7:
            weaknesses.append("- Low precision indicates high false positive rate")
        if metrics['recall'] < 0.7:
            weaknesses.append("- Low recall suggests missing many positive cases")
        if metrics['mae'] > 1.0:
            weaknesses.append("- High MAE indicates significant score prediction errors")
        if abs(metrics['correlation']) < 0.5:
            weaknesses.append("- Weak correlation suggests poor score prediction alignment")
        
        return '\n'.join(weaknesses) if weaknesses else "- Model performs well across all major metrics"

def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found")
        return
    
    evaluator = ModelEvaluator(api_key.strip())
    
    print("Loading dataset...")
    try:
        df = evaluator.load_dataset('Tweets.txt')
        print(f"Loaded {len(df)} samples")
    except FileNotFoundError:
        print("Error: Tweets.txt file not found")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print("\nOriginal Label Distribution:")
    label_counts = df['label'].value_counts()
    print(label_counts)
    
    # Enhanced sample size for better evaluation - CHANGED TO 100
    sample_size = 100  # Increased to 100 for better evaluation
    print(f"\nStarting enhanced evaluation for sample of {sample_size} texts...")
    
    try:
        # Use enhanced sampling method
        results = evaluator.evaluate_sample_enhanced(df, sample_size=sample_size)
        
        if not results['true_scores']:
            print("Error: No valid results obtained")
            return
        
        if len(results['true_scores']) < 10:
            print(f"Warning: Only {len(results['true_scores'])} data points obtained. Consider increasing sample size.")
        
        print("\nCalculating comprehensive performance metrics...")
        metrics = evaluator.calculate_comprehensive_metrics(results['true_scores'], results['predicted_scores'])
        
        # Generate comprehensive report with sample distribution
        evaluator.generate_comprehensive_report(metrics, len(results['true_scores']), 
                                              results.get('sample_distribution'))
        
        print("\nGenerating comprehensive visualizations...")
        evaluator.plot_comprehensive_results(results['true_scores'], results['predicted_scores'], metrics)
        
        evaluator.save_detailed_results(results['details'])
        
        print(f"\nComprehensive evaluation completed successfully for {len(results['true_scores'])} samples!")
        print("Files saved:")
        print("- comprehensive_evaluation_results.png (comprehensive visualizations)")
        print("- detailed_results.csv (detailed results)")
        print("- comprehensive_evaluation_report.txt (comprehensive report)")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
