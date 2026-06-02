#!/usr/bin/env python
"""
Phase 4: Model Training & Hyperparameter Tuning

Loads features and trains ML models for ASR quality classification.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import pickle


class ModelTrainer:
    """Train and evaluate ML models for ASR quality classification"""
    
    def __init__(self, features_json_path):
        self.features_path = Path(features_json_path)
        self.data = None
        self.features = None
        self.labels = None
        self.file_ids = None
        self.scaler = StandardScaler()
        
        print("\n" + "=" * 80)
        print("PHASE 4: MODEL TRAINING & EVALUATION".center(80))
        print("=" * 80)
    
    def load_features(self):
        """Load features from JSON"""
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        
        print(f"\n📂 Loading features from: {self.features_path}")
        
        with open(self.features_path, 'r') as f:
            self.data = json.load(f)
        
        # Convert list of dicts to 2D numpy array
        features_list = self.data['features']
        if features_list and isinstance(features_list[0], dict):
            # Get feature names from first sample
            feature_names = list(features_list[0].keys())
            # Create 2D array
            self.features = np.array([[sample[fname] for fname in feature_names] for sample in features_list])
        else:
            self.features = np.array(features_list)
        
        self.labels = np.array(self.data['labels'])
        self.file_ids = self.data['file_ids']
        
        print(f"✅ Loaded {self.features.shape[0]} samples with {self.features.shape[1]} features")
        print(f"   Label distribution: {self.data['metadata']['label_distribution']}")
        
        return self.features, self.labels
    
    def create_splits(self, test_size=0.15, val_size=0.15):
        """Create stratified train/val/test splits"""
        print(f"\n🔀 Creating stratified splits (train: 70%, val: {val_size*100:.0f}%, test: {test_size*100:.0f}%)...")
        
        # First split: 85% train+val, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_size,
            stratify=self.labels,
            random_state=42
        )
        
        # Second split: train (70%) and val (15%)
        # val_ratio = 0.15 / 0.85 ≈ 0.176
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=42
        )
        
        print(f"   Train: {len(X_train)} samples ({len(X_train)/len(self.features)*100:.1f}%)")
        print(f"   Val:   {len(X_val)} samples ({len(X_val)/len(self.features)*100:.1f}%)")
        print(f"   Test:  {len(X_test)} samples ({len(X_test)/len(self.features)*100:.1f}%)")
        
        # Check class balance
        print(f"\n   Class distribution:")
        print(f"   Train - Usable: {(y_train==1).sum()}, Not-usable: {(y_train==0).sum()}")
        print(f"   Val   - Usable: {(y_val==1).sum()}, Not-usable: {(y_val==0).sum()}")
        print(f"   Test  - Usable: {(y_test==1).sum()}, Not-usable: {(y_test==0).sum()}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train logistic regression baseline"""
        print("\n" + "-" * 80)
        print("MODEL 1: LOGISTIC REGRESSION".center(80))
        print("-" * 80)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Calculate class weights to handle imbalance
        n_usable = (y_train == 1).sum()
        n_not_usable = (y_train == 0).sum()
        weight_usable = n_not_usable / n_usable
        
        print(f"\n⚙️  Hyperparameters:")
        print(f"   Class weight (not_usable:usable) = 1.0:{weight_usable:.2f}")
        print(f"   C=1.0, max_iter=1000, solver='lbfgs'")
        
        # Train model
        model = LogisticRegression(
            class_weight={0: 1.0, 1: weight_usable},
            C=1.0,
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        )
        
        print(f"\n🔨 Training...")
        model.fit(X_train_scaled, y_train)
        print(f"✅ Training complete!")
        
        # Evaluate
        y_pred = model.predict(X_val_scaled)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        metrics = self.evaluate_model(y_val, y_pred, y_pred_proba, "Validation")
        
        return model, self.scaler, metrics
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        try:
            import xgboost as xgb
        except ImportError:
            print("\n⚠️  XGBoost not installed - skipping")
            return None, None, None
        
        print("\n" + "-" * 80)
        print("MODEL 2: XGBOOST".center(80))
        print("-" * 80)
        
        # Calculate scale_pos_weight for imbalance
        n_usable = (y_train == 1).sum()
        n_not_usable = (y_train == 0).sum()
        scale_pos_weight = n_not_usable / n_usable
        
        print(f"\n⚙️  Hyperparameters:")
        print(f"   scale_pos_weight = {scale_pos_weight:.2f}")
        print(f"   max_depth=5, learning_rate=0.1, n_estimators=100")
        
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            verbosity=0
        )
        
        print(f"\n🔨 Training...")
        model.fit(X_train, y_train)
        print(f"✅ Training complete!")
        
        # Evaluate
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = self.evaluate_model(y_val, y_pred, y_pred_proba, "Validation")
        
        return model, None, metrics
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba, dataset_name="Validation"):
        """Evaluate model performance"""
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\n📊 {dataset_name} Metrics:")
        print(f"   Precision:  {precision:.4f}")
        print(f"   Recall:     {recall:.4f}")
        print(f"   F1-Score:   {f1:.4f} {'✅ TARGET REACHED!' if f1 >= 0.80 else '⏳ need improvement'}")
        print(f"   ROC-AUC:    {roc_auc:.4f}")
        print(f"   Specificity:{specificity:.4f}")
        
        print(f"\n   Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Negative Positive")
        print(f"   Actual Neg:      {tn:3d}      {fp:3d}")
        print(f"   Actual Pos:      {fn:3d}      {tp:3d}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'specificity': specificity,
            'confusion_matrix': (tn, fp, fn, tp)
        }
    
    def save_model(self, model, scaler, model_name, metrics):
        """Save trained model and scaler"""
        output_dir = self.features_path.parent.parent / 'models'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n💾 Model saved: {model_path}")
        
        # Save scaler if it exists
        if scaler is not None:
            scaler_path = output_dir / f"{model_name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"💾 Scaler saved: {scaler_path}")
        
        # Save metrics
        metrics_path = output_dir / f"{model_name}_metrics.json"
        metrics_copy = metrics.copy()
        metrics_copy['confusion_matrix'] = [int(x) for x in metrics['confusion_matrix']]
        with open(metrics_path, 'w') as f:
            json.dump(metrics_copy, f, indent=2)
        print(f"💾 Metrics saved: {metrics_path}")
        
        return model_path
    
    def run(self):
        """Run complete training pipeline"""
        # Load features
        self.load_features()
        
        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_splits()
        
        # Train Logistic Regression
        lr_model, lr_scaler, lr_metrics = self.train_logistic_regression(
            X_train, y_train, X_val, y_val
        )
        self.save_model(lr_model, lr_scaler, "logistic_regression", lr_metrics)
        
        # Train XGBoost
        xgb_model, xgb_scaler, xgb_metrics = self.train_xgboost(
            X_train, y_train, X_val, y_val
        )
        if xgb_model is not None:
            self.save_model(xgb_model, xgb_scaler, "xgboost", xgb_metrics)
        
        # Summary
        print("\n" + "=" * 80)
        print("PHASE 4 SUMMARY".center(80))
        print("=" * 80)
        
        print(f"""
✅ Model Training Complete!

📊 MODELS TRAINED:
   1. Logistic Regression
      - F1-Score: {lr_metrics['f1']:.4f}
      - ROC-AUC:  {lr_metrics['roc_auc']:.4f}
   
   2. XGBoost
      - F1-Score: {xgb_metrics['f1']:.4f}
      - ROC-AUC:  {xgb_metrics['roc_auc']:.4f}

💾 SAVED ARTIFACTS:
   - models/logistic_regression.pkl
   - models/logistic_regression_scaler.pkl
   - models/logistic_regression_metrics.json
   - models/xgboost.pkl
   - models/xgboost_metrics.json

🎯 NEXT STEPS (Phase 5):
   1. Evaluate best model on held-out test set
   2. Generate final metrics and confusion matrix
   3. Analyze prediction errors

📈 BEST MODEL SO FAR:
   """ + ("Logistic Regression" if lr_metrics['f1'] >= xgb_metrics['f1'] else "XGBoost") + f"""
      F1-Score: {max(lr_metrics['f1'], xgb_metrics['f1']):.4f}

⚠️  STATUS:
   """ + ("✅ TARGET REACHED (F1 ≥ 0.80)!" if max(lr_metrics['f1'], xgb_metrics['f1']) >= 0.80 else "⏳ Need to improve features or model tuning") + """

🚀 TO CONTINUE:
   python scripts/evaluate.py
        """)
        
        print("=" * 80)
        
        # Store splits for evaluation phase
        splits_path = self.features_path.parent / 'splits.json'
        splits_data = {
            'X_test_shape': X_test.shape,
            'y_test': [int(x) for x in y_test],
            'test_indices': list(range(len(X_test)))
        }
        with open(splits_path, 'w') as f:
            json.dump(splits_data, f)
        print(f"\n💾 Test set info saved: {splits_path}")
        
        return lr_model, xgb_model, lr_metrics, xgb_metrics


def main():
    """Main training function"""
    features_path = Path(__file__).parent.parent / 'data' / 'features.json'
    
    trainer = ModelTrainer(features_path)
    trainer.run()


if __name__ == "__main__":
    main()
