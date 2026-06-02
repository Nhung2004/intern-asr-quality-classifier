#!/usr/bin/env python
"""
Phase 5: Evaluation on Test Set

Evaluates the best model on the held-out test set and generates final metrics.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)


class ModelEvaluator:
    """Evaluate trained models on test set"""
    
    def __init__(self, features_path, models_dir):
        self.features_path = Path(features_path)
        self.models_dir = Path(models_dir)
        self.data = None
        self.features = None
        self.labels = None
        
        print("\n" + "=" * 80)
        print("PHASE 5: EVALUATION ON TEST SET".center(80))
        print("=" * 80)
    
    def load_data(self):
        """Load all data"""
        print(f"\n📂 Loading features from: {self.features_path}")
        
        with open(self.features_path, 'r') as f:
            self.data = json.load(f)
        
        # Convert list of dicts to 2D numpy array
        features_list = self.data['features']
        if features_list and isinstance(features_list[0], dict):
            feature_names = list(features_list[0].keys())
            self.features = np.array([[sample[fname] for fname in feature_names] for sample in features_list])
        else:
            self.features = np.array(features_list)
        
        self.labels = np.array(self.data['labels'])
        
        print(f"✅ Loaded {self.features.shape[0]} samples with {self.features.shape[1]} features")
    
    def create_test_set(self, test_size=0.15):
        """Create test set from all data"""
        from sklearn.model_selection import train_test_split
        
        print(f"\n🔀 Creating test set ({test_size*100:.0f}%)...")
        
        # Split off test set
        X_rest, X_test, y_rest, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_size,
            stratify=self.labels,
            random_state=42
        )
        
        print(f"   Test set: {len(X_test)} samples")
        print(f"   - Usable: {(y_test==1).sum()}")
        print(f"   - Not usable: {(y_test==0).sum()}")
        
        return X_test, y_test
    
    def load_model(self, model_name):
        """Load trained model"""
        model_path = self.models_dir / f"{model_name}.pkl"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Try to load scaler
        scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        return model, scaler
    
    def evaluate_on_test(self, model, scaler, X_test, y_test, model_name):
        """Evaluate model on test set"""
        print(f"\n" + "-" * 80)
        print(f"EVALUATING: {model_name.upper()}".center(80))
        print("-" * 80)
        
        # Make predictions
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\n📊 TEST SET METRICS:")
        print(f"   Precision:  {precision:.4f}")
        print(f"   Recall:     {recall:.4f}")
        print(f"   F1-Score:   {f1:.4f} {'✅ TARGET REACHED!' if f1 >= 0.80 else '⏳ Below target (0.80)'}")
        print(f"   ROC-AUC:    {roc_auc:.4f}")
        print(f"   Specificity:{specificity:.4f}")
        
        print(f"\n   Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Negative Positive")
        print(f"   Actual Neg:      {tn:3d}      {fp:3d}")
        print(f"   Actual Pos:      {fn:3d}      {tp:3d}")
        
        # Classification report
        print(f"\n   Detailed Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Not Usable', 'Usable'],
            zero_division=0
        ))
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'specificity': specificity,
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'y_pred': [int(x) for x in y_pred],
            'y_pred_proba': [float(x) for x in y_pred_proba]
        }
        
        return metrics
    
    def save_results(self, results):
        """Save evaluation results"""
        results_path = self.models_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved: {results_path}")
    
    def run(self):
        """Run evaluation pipeline"""
        self.load_data()
        X_test, y_test = self.create_test_set()
        
        results = {'test_metrics': {}, 'y_true': [int(x) for x in y_test]}
        
        # Evaluate Logistic Regression
        lr_model, lr_scaler = self.load_model('logistic_regression')
        if lr_model is not None:
            lr_metrics = self.evaluate_on_test(lr_model, lr_scaler, X_test, y_test, 'logistic_regression')
            results['test_metrics']['logistic_regression'] = lr_metrics
        
        # Evaluate XGBoost
        xgb_model, xgb_scaler = self.load_model('xgboost')
        if xgb_model is not None:
            xgb_metrics = self.evaluate_on_test(xgb_model, xgb_scaler, X_test, y_test, 'xgboost')
            results['test_metrics']['xgboost'] = xgb_metrics
        
        # Summary
        print("\n" + "=" * 80)
        print("PHASE 5 SUMMARY - TEST SET EVALUATION".center(80))
        print("=" * 80)
        
        if 'logistic_regression' in results['test_metrics'] and 'xgboost' in results['test_metrics']:
            lr_f1 = results['test_metrics']['logistic_regression']['f1']
            xgb_f1 = results['test_metrics']['xgboost']['f1']
            
            best_model = "XGBoost" if xgb_f1 >= lr_f1 else "Logistic Regression"
            best_f1 = max(lr_f1, xgb_f1)
            
            print(f"""
✅ Evaluation Complete!

📊 FINAL TEST SET RESULTS:

   1. Logistic Regression
      - F1-Score:  {lr_f1:.4f}
      - ROC-AUC:   {results['test_metrics']['logistic_regression']['roc_auc']:.4f}
   
   2. XGBoost
      - F1-Score:  {xgb_f1:.4f}
      - ROC-AUC:   {results['test_metrics']['xgboost']['roc_auc']:.4f}

🏆 BEST MODEL: {best_model}
   - F1-Score: {best_f1:.4f}

🎯 TARGET: F1-score ≥ 0.80
   Status: {"✅ TARGET REACHED!" if best_f1 >= 0.80 else "⏳ Below target - need feature engineering"}

📊 OBSERVATIONS:
   - Current model uses only 8 text-based features
   - Features extracted: text length, word count, punctuation
   - Limited discriminative power without audio features
   
💡 TO IMPROVE (Optional Phase Extensions):
   1. Extract audio features:
      - MFCCs (Mel-frequency cepstral coefficients)
      - Spectral centroids and bandwidth
      - Energy statistics
      - SNR (Signal-to-Noise Ratio)
      - Use pretrained embeddings (Wav2Vec2)
   
   2. Advanced text features:
      - Sentence-transformers embeddings
      - Vietnamese language features
      - Linguistic complexity metrics
   
   3. Cross-modal features:
      - Audio-text alignment
      - Duration matching
      - ASR hypothesis similarity
   
   4. Hyperparameter tuning:
      - GridSearchCV for XGBoost
      - Different class weight strategies
      - Feature scaling and normalization

🚀 NEXT STEPS:
   1. Save final model and metrics
   2. Generate error analysis report
   3. Document findings and recommendations
   4. Prepare submission materials

📂 ARTIFACTS:
   - models/test_results.json (final test metrics)
   - Best model: {best_model}
            """)
        
        print("=" * 80)
        
        self.save_results(results)


def main():
    """Main evaluation function"""
    features_path = Path(__file__).parent.parent / 'data' / 'features.json'
    models_dir = Path(__file__).parent.parent / 'models'
    
    evaluator = ModelEvaluator(features_path, models_dir)
    evaluator.run()


if __name__ == "__main__":
    main()
