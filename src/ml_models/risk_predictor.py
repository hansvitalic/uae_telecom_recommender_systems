"""
Risk Predictor Module
Implements machine learning models for risk prediction and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import config

class RiskPredictor:
    """
    Implements various ML models for risk prediction and analysis
    """
    
    def __init__(self, ml_dataset_path: str = None):
        """Initialize with ML dataset"""
        self.dataset_path = ml_dataset_path or config.ML_DATASET_FILE
        self.ml_data = None
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
    
    def load_ml_dataset(self) -> pd.DataFrame:
        """Load the prepared ML dataset"""
        try:
            self.ml_data = pd.read_csv(self.dataset_path)
            print(f"✓ Loaded ML dataset with {len(self.ml_data)} samples and {self.ml_data.shape[1]} features")
            return self.ml_data
        except FileNotFoundError:
            print("❌ ML dataset not found. Run feature engineering first.")
            return None
    
    def prepare_datasets(self) -> Dict[str, Dict]:
        """Prepare datasets for different prediction tasks"""
        if self.ml_data is None:
            self.load_ml_dataset()
        
        # Separate features and targets
        feature_cols = [col for col in self.ml_data.columns if not col.startswith('target_') and col != 'Risk_ID']
        
        # Prepare features - encode categorical variables
        X_raw = self.ml_data[feature_cols].copy()
        
        # Identify categorical columns that need encoding
        categorical_cols = X_raw.select_dtypes(include=['object']).columns
        
        # Apply label encoding to categorical columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))
        
        # Fill any remaining NaN values
        X_raw = X_raw.fillna(0)
        
        datasets = {}
        
        # Binary classification datasets
        binary_targets = ['is_critical', 'is_high_risk', 'needs_immediate_attention']
        for target in binary_targets:
            target_col = f'target_{target}'
            if target_col in self.ml_data.columns:
                y = self.ml_data[target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_raw, y, test_size=0.2, random_state=42, stratify=y
                )
                
                datasets[target] = {
                    'X_train': X_train, 'X_test': X_test,
                    'y_train': y_train, 'y_test': y_test,
                    'task_type': 'classification'
                }
        
        # Multi-class classification
        if 'target_risk_level' in self.ml_data.columns:
            y = self.ml_data['target_risk_level']
            
            # Encode target labels
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            datasets['risk_level'] = {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'task_type': 'classification',
                'label_encoder': le_target
            }
        
        # Regression datasets
        regression_targets = ['risk_score', 'probability_numeric', 'impact_numeric']
        for target in regression_targets:
            target_col = f'target_{target}'
            if target_col in self.ml_data.columns:
                y = self.ml_data[target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_raw, y, test_size=0.2, random_state=42
                )
                
                datasets[target] = {
                    'X_train': X_train, 'X_test': X_test,
                    'y_train': y_train, 'y_test': y_test,
                    'task_type': 'regression'
                }
        
        print(f"✓ Prepared {len(datasets)} datasets for ML modeling")
        return datasets
    
    def train_classification_models(self, X_train: pd.DataFrame, y_train: pd.Series, task_name: str) -> Dict[str, Any]:
        """Train classification models for a specific task"""
        models = {}
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers[task_name] = scaler
        
        # Random Forest Classifier
        print(f"  🌲 Training Random Forest for {task_name}...")
        rf_model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10, min_samples_split=5
        )
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # Gradient Boosting Classifier
        print(f"  🚀 Training Gradient Boosting for {task_name}...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1
        )
        gb_model.fit(X_train, y_train)
        models['gradient_boosting'] = gb_model
        
        # Logistic Regression
        print(f"  📈 Training Logistic Regression for {task_name}...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        models['logistic_regression'] = lr_model
        
        # Support Vector Machine (if dataset is not too large)
        if len(X_train) < 1000:
            print(f"  🎯 Training SVM for {task_name}...")
            svm_model = SVC(random_state=42, probability=True)
            svm_model.fit(X_train_scaled, y_train)
            models['svm'] = svm_model
        
        return models
    
    def train_regression_models(self, X_train: pd.DataFrame, y_train: pd.Series, task_name: str) -> Dict[str, Any]:
        """Train regression models for a specific task"""
        models = {}
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers[task_name] = scaler
        
        # Random Forest Regressor
        print(f"  🌲 Training Random Forest Regressor for {task_name}...")
        rf_model = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10, min_samples_split=5
        )
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # Linear Regression
        print(f"  📈 Training Linear Regression for {task_name}...")
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        models['linear_regression'] = lr_model
        
        # Support Vector Regression
        if len(X_train) < 1000:
            print(f"  🎯 Training SVR for {task_name}...")
            svr_model = SVR()
            svr_model.fit(X_train_scaled, y_train)
            models['svr'] = svr_model
        
        return models
    
    def evaluate_classification_models(self, models: Dict, X_test: pd.DataFrame, y_test: pd.Series, task_name: str) -> Dict[str, Dict]:
        """Evaluate classification models"""
        evaluations = {}
        
        for model_name, model in models.items():
            print(f"    📊 Evaluating {model_name} for {task_name}...")
            
            # Prepare test data
            if model_name in ['logistic_regression', 'svm']:
                X_test_prepared = self.scalers[task_name].transform(X_test)
            else:
                X_test_prepared = X_test
            
            # Make predictions
            y_pred = model.predict(X_test_prepared)
            y_pred_proba = model.predict_proba(X_test_prepared)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            evaluation = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # Add AUC for binary classification
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                evaluation['auc'] = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_test_prepared, y_test, cv=3, scoring='accuracy')
            evaluation['cv_mean'] = cv_scores.mean()
            evaluation['cv_std'] = cv_scores.std()
            
            evaluations[model_name] = evaluation
        
        return evaluations
    
    def evaluate_regression_models(self, models: Dict, X_test: pd.DataFrame, y_test: pd.Series, task_name: str) -> Dict[str, Dict]:
        """Evaluate regression models"""
        evaluations = {}
        
        for model_name, model in models.items():
            print(f"    📊 Evaluating {model_name} for {task_name}...")
            
            # Prepare test data
            if model_name in ['linear_regression', 'svr']:
                X_test_prepared = self.scalers[task_name].transform(X_test)
            else:
                X_test_prepared = X_test
            
            # Make predictions
            y_pred = model.predict(X_test_prepared)
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error
            
            evaluation = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_test_prepared, y_test, cv=3, scoring='r2')
            evaluation['cv_mean'] = cv_scores.mean()
            evaluation['cv_std'] = cv_scores.std()
            
            evaluations[model_name] = evaluation
        
        return evaluations
    
    def extract_feature_importance(self, model: Any, feature_names: List[str], model_name: str) -> Dict[str, float]:
        """Extract feature importance from trained models"""
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance_dict = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # Linear models
            if len(model.coef_.shape) == 1:
                importance_dict = dict(zip(feature_names, np.abs(model.coef_)))
            else:
                # Multi-class case
                importance_dict = dict(zip(feature_names, np.abs(model.coef_).mean(axis=0)))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def train_all_models(self) -> Dict[str, Any]:
        """Train all ML models for all prediction tasks"""
        print("🤖 Starting ML model training...")
        
        # Prepare datasets
        datasets = self.prepare_datasets()
        
        all_results = {}
        
        for task_name, dataset in datasets.items():
            print(f"\n🎯 Training models for task: {task_name}")
            
            X_train, X_test = dataset['X_train'], dataset['X_test']
            y_train, y_test = dataset['y_train'], dataset['y_test']
            task_type = dataset['task_type']
            
            # Train models based on task type
            if task_type == 'classification':
                models = self.train_classification_models(X_train, y_train, task_name)
                evaluations = self.evaluate_classification_models(models, X_test, y_test, task_name)
            else:  # regression
                models = self.train_regression_models(X_train, y_train, task_name)
                evaluations = self.evaluate_regression_models(models, X_test, y_test, task_name)
            
            # Extract feature importance
            feature_importance = {}
            for model_name, model in models.items():
                importance = self.extract_feature_importance(model, X_train.columns.tolist(), model_name)
                feature_importance[model_name] = importance
            
            # Store results
            all_results[task_name] = {
                'models': models,
                'evaluations': evaluations,
                'feature_importance': feature_importance,
                'task_type': task_type,
                'best_model': self._identify_best_model(evaluations, task_type)
            }
            
            self.models[task_name] = models
            self.model_performance[task_name] = evaluations
            self.feature_importance[task_name] = feature_importance
        
        print(f"\n✓ Completed training for {len(all_results)} prediction tasks")
        return all_results
    
    def _identify_best_model(self, evaluations: Dict, task_type: str) -> str:
        """Identify the best performing model for a task"""
        if task_type == 'classification':
            # Use F1 score for classification
            best_model = max(evaluations.keys(), key=lambda x: evaluations[x].get('f1_score', 0))
        else:
            # Use R2 score for regression
            best_model = max(evaluations.keys(), key=lambda x: evaluations[x].get('r2', -np.inf))
        
        return best_model
    
    def make_predictions(self, risk_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Make predictions on new risk data"""
        if risk_data is None:
            # Use a subset of the training data for prediction
            feature_cols = [col for col in self.ml_data.columns if not col.startswith('target_') and col != 'Risk_ID']
            risk_data = self.ml_data[feature_cols + ['Risk_ID']].head(10)  # Use first 10 rows
        
        predictions = {}
        
        # Get feature columns (excluding targets and ID)
        feature_cols = [col for col in risk_data.columns if not col.startswith('target_') and col != 'Risk_ID']
        X = risk_data[feature_cols].copy()
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Fill NaN values
        X = X.fillna(0)
        
        for task_name, task_models in self.models.items():
            task_predictions = {}
            
            # Get best model for this task
            best_model_name = self._identify_best_model(self.model_performance[task_name], 
                                                       'classification' if 'is_' in task_name or task_name == 'risk_level' else 'regression')
            best_model = task_models[best_model_name]
            
            # Prepare data for prediction
            if best_model_name in ['logistic_regression', 'svm', 'linear_regression', 'svr']:
                X_prepared = self.scalers[task_name].transform(X)
            else:
                X_prepared = X
            
            # Make predictions
            pred = best_model.predict(X_prepared)
            task_predictions['predictions'] = pred.tolist()  # Convert to list for JSON serialization
            task_predictions['model_used'] = best_model_name
            
            # Add prediction probabilities for classification
            if hasattr(best_model, 'predict_proba'):
                proba = best_model.predict_proba(X_prepared)
                task_predictions['probabilities'] = proba.tolist()  # Convert to list
            
            predictions[task_name] = task_predictions
        
        return predictions
    
    def generate_risk_insights(self) -> Dict[str, Any]:
        """Generate insights from trained models"""
        insights = {
            'model_performance_summary': self._summarize_model_performance(),
            'feature_importance_analysis': self._analyze_feature_importance(),
            'prediction_patterns': self._analyze_prediction_patterns(),
            'risk_factors': self._identify_key_risk_factors()
        }
        
        return insights
    
    def _summarize_model_performance(self) -> Dict[str, Any]:
        """Summarize model performance across all tasks"""
        summary = {}
        
        for task_name, evaluations in self.model_performance.items():
            task_summary = {}
            
            if any('accuracy' in eval_metrics for eval_metrics in evaluations.values()):
                # Classification task
                best_accuracy = max(eval_metrics.get('accuracy', 0) for eval_metrics in evaluations.values())
                best_f1 = max(eval_metrics.get('f1_score', 0) for eval_metrics in evaluations.values())
                task_summary = {'best_accuracy': best_accuracy, 'best_f1_score': best_f1}
            else:
                # Regression task
                best_r2 = max(eval_metrics.get('r2', -np.inf) for eval_metrics in evaluations.values())
                best_rmse = min(eval_metrics.get('rmse', np.inf) for eval_metrics in evaluations.values())
                task_summary = {'best_r2': best_r2, 'best_rmse': best_rmse}
            
            summary[task_name] = task_summary
        
        return summary
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance across all models"""
        # Aggregate feature importance across all tasks and models
        feature_scores = {}
        
        for task_name, task_importance in self.feature_importance.items():
            for model_name, importance_dict in task_importance.items():
                for feature, score in importance_dict.items():
                    if feature not in feature_scores:
                        feature_scores[feature] = []
                    feature_scores[feature].append(score)
        
        # Calculate average importance
        avg_importance = {
            feature: np.mean(scores) 
            for feature, scores in feature_scores.items()
        }
        
        # Sort by importance
        top_features = dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:20])
        
        analysis = {
            'top_20_features': top_features,
            'feature_categories': self._categorize_important_features(top_features),
            'consistency_across_tasks': self._analyze_feature_consistency(feature_scores)
        }
        
        return analysis
    
    def _categorize_important_features(self, features: Dict[str, float]) -> Dict[str, List[str]]:
        """Categorize important features by type"""
        categories = {
            'risk_characteristics': [],
            'project_context': [],
            'mitigation_factors': [],
            'temporal_factors': [],
            'text_derived': []
        }
        
        for feature in features.keys():
            if any(keyword in feature.lower() for keyword in ['probability', 'impact', 'score', 'urgency']):
                categories['risk_characteristics'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['category', 'phase', 'sector', 'owner']):
                categories['project_context'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['mitigation', 'contingency', 'has_']):
                categories['mitigation_factors'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['week', 'lag', 'order']):
                categories['temporal_factors'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['length', 'word', 'count']):
                categories['text_derived'].append(feature)
        
        return categories
    
    def _analyze_feature_consistency(self, feature_scores: Dict[str, List[float]]) -> Dict[str, float]:
        """Analyze consistency of feature importance across models"""
        consistency = {}
        
        for feature, scores in feature_scores.items():
            if len(scores) > 1:
                # Calculate coefficient of variation
                cv = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
                consistency[feature] = 1 - cv  # Higher value = more consistent
            else:
                consistency[feature] = 1.0  # Single score is perfectly consistent
        
        return consistency
    
    def _analyze_prediction_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in model predictions"""
        if not self.models:
            return {"message": "No models trained yet"}
        
        # Make predictions on the dataset
        predictions = self.make_predictions()
        
        patterns = {
            'prediction_distributions': {},
            'correlation_analysis': {},
            'risk_segmentation': {}
        }
        
        # Analyze prediction distributions
        for task_name, task_pred in predictions.items():
            pred_values = task_pred['predictions']
            
            if isinstance(pred_values[0], (int, float)):
                # Numerical predictions
                patterns['prediction_distributions'][task_name] = {
                    'mean': np.mean(pred_values),
                    'std': np.std(pred_values),
                    'min': np.min(pred_values),
                    'max': np.max(pred_values)
                }
            else:
                # Categorical predictions
                unique, counts = np.unique(pred_values, return_counts=True)
                patterns['prediction_distributions'][task_name] = dict(zip(unique, counts))
        
        return patterns
    
    def _identify_key_risk_factors(self) -> Dict[str, Any]:
        """Identify key risk factors from model analysis"""
        # Get top features
        feature_analysis = self._analyze_feature_importance()
        top_features = list(feature_analysis['top_20_features'].keys())[:10]
        
        risk_factors = {
            'most_predictive_features': top_features,
            'risk_indicators': [],
            'protective_factors': [],
            'recommendations': []
        }
        
        # Analyze feature patterns
        for feature in top_features:
            if any(keyword in feature.lower() for keyword in ['probability', 'impact', 'urgent', 'critical']):
                risk_factors['risk_indicators'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['mitigation', 'contingency', 'has_']):
                risk_factors['protective_factors'].append(feature)
        
        # Generate recommendations
        risk_factors['recommendations'] = [
            "Focus on the most predictive features for risk assessment",
            "Strengthen protective factors to reduce risk exposure",
            "Monitor risk indicators closely for early warning",
            "Use model insights to improve risk management processes"
        ]
        
        return risk_factors
    
    def save_models(self, model_dir: str = None) -> Dict[str, str]:
        """Save trained models to disk"""
        if model_dir is None:
            model_dir = config.OUTPUT_DIR / "models"
        
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        saved_models = {}
        
        for task_name, task_models in self.models.items():
            for model_name, model in task_models.items():
                filename = f"{task_name}_{model_name}_model.joblib"
                filepath = os.path.join(model_dir, filename)
                joblib.dump(model, filepath)
                saved_models[f"{task_name}_{model_name}"] = filepath
        
        # Save scalers
        for task_name, scaler in self.scalers.items():
            filename = f"{task_name}_scaler.joblib"
            filepath = os.path.join(model_dir, filename)
            joblib.dump(scaler, filepath)
            saved_models[f"{task_name}_scaler"] = filepath
        
        print(f"✓ Saved {len(saved_models)} models and scalers to {model_dir}")
        return saved_models
    
    def generate_ml_report(self) -> Dict[str, Any]:
        """Generate comprehensive ML analysis report"""
        # Train all models if not already done
        if not self.models:
            training_results = self.train_all_models()
        
        # Generate insights
        insights = self.generate_risk_insights()
        
        # Save models
        saved_models = self.save_models()
        
        report = {
            "ml_summary": {
                "total_models_trained": sum(len(task_models) for task_models in self.models.values()),
                "prediction_tasks": list(self.models.keys()),
                "dataset_size": len(self.ml_data) if self.ml_data is not None else 0,
                "completion_status": "ML Analysis Complete"
            },
            "model_performance": self.model_performance,
            "feature_importance": self.feature_importance,
            "insights": insights,
            "saved_models": saved_models,
            "recommendations": self._generate_ml_recommendations(insights)
        }
        
        return report
    
    def _generate_ml_recommendations(self, insights: Dict) -> List[str]:
        """Generate ML-based recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        performance = insights['model_performance_summary']
        for task, metrics in performance.items():
            if 'best_accuracy' in metrics and metrics['best_accuracy'] < 0.8:
                recommendations.append(f"Improve data quality for {task} prediction task")
            if 'best_r2' in metrics and metrics['best_r2'] < 0.7:
                recommendations.append(f"Collect more features for {task} regression task")
        
        # Feature-based recommendations
        top_features = insights['feature_importance_analysis']['top_20_features']
        if any('mitigation' in feature.lower() for feature in top_features):
            recommendations.append("Mitigation strategies are key predictors - ensure comprehensive documentation")
        
        recommendations.extend([
            "Use model predictions to prioritize risk management efforts",
            "Implement real-time risk scoring using trained models",
            "Regular model retraining with new data is recommended"
        ])
        
        return recommendations

def main():
    """Main execution function for ML risk prediction"""
    predictor = RiskPredictor()
    
    print("🤖 Starting ML Risk Prediction...")
    print("=" * 50)
    
    # Generate ML report
    report = predictor.generate_ml_report()
    
    print(f"✓ Trained {report['ml_summary']['total_models_trained']} ML models")
    print(f"✓ Covered {len(report['ml_summary']['prediction_tasks'])} prediction tasks")
    print(f"✓ Processed {report['ml_summary']['dataset_size']} risk samples")
    
    return report

if __name__ == "__main__":
    main()