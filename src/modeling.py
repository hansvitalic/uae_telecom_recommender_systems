"""
Modeling Module for UAE Telecom Risk Recommender System

This module provides functionality to train RandomForest models using
engineered features for risk prediction and recommendation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class RiskModelingPipeline:
    """Pipeline for training and evaluating risk prediction models."""
    
    def __init__(self, model_type: str = 'classification', random_state: int = 42):
        """
        Initialize the modeling pipeline.
        
        Args:
            model_type (str): Type of model - 'classification' or 'regression'
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model: Optional[Union[RandomForestClassifier, RandomForestRegressor]] = None
        self.feature_names: Optional[List[str]] = None
        self.target_name: Optional[str] = None
        self.model_performance: Optional[Dict] = None
        
        # Initialize model based on type
        if model_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state
            )
        elif model_type == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state
            )
        else:
            raise ValueError("model_type must be 'classification' or 'regression'")
    
    def prepare_model_data(self, data: pd.DataFrame, target_column: str, 
                          feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            data (pd.DataFrame): Engineered features data
            target_column (str): Name of target column
            feature_columns (Optional[List[str]]): Specific features to use. Auto-select if None.
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target data
        """
        logger.info("Preparing model data")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Auto-select features if not specified
        if feature_columns is None:
            # Exclude non-feature columns
            exclude_columns = [
                'Risk_ID', 'Risk_Title', 'Risk_Description', 'cleaned_description',
                'Date_Identified', 'Review_Date', 'Documentation_Link',
                'Qualitative_Notes', 'Quantitative_Notes', 'Lessons_Learned',
                target_column
            ]
            
            # Include engineered features
            feature_columns = [col for col in data.columns if col not in exclude_columns]
            
            # Prioritize certain feature types
            priority_features = []
            
            # Add sentiment features
            priority_features.extend([col for col in feature_columns if 'sentiment' in col.lower()])
            
            # Add TF-IDF features (limit to top features)
            tfidf_features = [col for col in feature_columns if 'tfidf' in col.lower()][:20]  # Top 20
            priority_features.extend(tfidf_features)
            
            # Add encoded categorical features
            priority_features.extend([col for col in feature_columns if any(cat in col for cat in 
                                    ['Risk_Category_', 'Sector_', 'Project_Phase_', '_encoded'])])
            
            # Add scaled numerical features
            priority_features.extend([col for col in feature_columns if '_scaled' in col])
            
            # Add risk score features
            priority_features.extend([col for col in feature_columns if 'risk_score' in col.lower() or 
                                    'risk_level' in col.lower() or '_numeric' in col])
            
            feature_columns = list(set(priority_features))  # Remove duplicates
        
        # Filter for existing columns
        feature_columns = [col for col in feature_columns if col in data.columns]
        
        logger.info(f"Selected {len(feature_columns)} features for modeling")
        
        # Prepare features and target
        X = data[feature_columns].copy()
        y = data[target_column].copy()
        
        # Handle missing values in features
        X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        # Handle missing values in target
        if self.model_type == 'classification':
            y = y.fillna(y.mode()[0] if not y.mode().empty else 'Unknown')
        else:
            y = y.fillna(y.median())
        
        self.feature_names = feature_columns
        self.target_name = target_column
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, perform_cv: bool = True) -> Dict:
        """
        Train the RandomForest model.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            test_size (float): Test set proportion
            perform_cv (bool): Whether to perform cross-validation
            
        Returns:
            Dict: Training results and performance metrics
        """
        logger.info("Training RandomForest model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y if self.model_type == 'classification' else None
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate performance metrics
        performance = {
            'model_type': self.model_type,
            'n_features': len(self.feature_names),
            'train_size': len(X_train),
            'test_size': len(X_test),
        }
        
        if self.model_type == 'classification':
            # Classification metrics
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            performance.update({
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
            })
            
            logger.info(f"Classification Accuracy - Train: {train_score:.3f}, Test: {test_score:.3f}")
            
        else:
            # Regression metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            performance.update({
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': np.sqrt(train_mse),
                'test_rmse': np.sqrt(test_mse)
            })
            
            logger.info(f"Regression R² - Train: {train_r2:.3f}, Test: {test_r2:.3f}")
        
        # Cross-validation
        if perform_cv:
            logger.info("Performing cross-validation")
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, 
                                      scoring='accuracy' if self.model_type == 'classification' else 'r2')
            performance.update({
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            })
            logger.info(f"Cross-validation score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        performance['feature_importance'] = feature_importance.to_dict('records')
        
        self.model_performance = performance
        
        return performance
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                            param_grid: Optional[Dict] = None, cv: int = 3) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            param_grid (Optional[Dict]): Parameter grid for tuning
            cv (int): Cross-validation folds
            
        Returns:
            Dict: Best parameters and performance
        """
        logger.info("Performing hyperparameter tuning")
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        scoring = 'accuracy' if self.model_type == 'classification' else 'r2'
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring=scoring, 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.3f}")
        
        return tuning_results
    
    def predict_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make risk predictions on new data.
        
        Args:
            data (pd.DataFrame): New data for prediction
            
        Returns:
            pd.DataFrame: Data with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train model first.")
        
        logger.info("Making risk predictions")
        
        # Prepare features (same columns as training)
        X = data[self.feature_names].copy()
        X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        # Make predictions
        predictions = self.model.predict(X)
        prediction_proba = None
        
        if self.model_type == 'classification' and hasattr(self.model, 'predict_proba'):
            prediction_proba = self.model.predict_proba(X)
        
        # Add predictions to data
        result_data = data.copy()
        result_data[f'{self.target_name}_predicted'] = predictions
        
        if prediction_proba is not None:
            # Add probability columns
            classes = self.model.classes_
            for i, class_name in enumerate(classes):
                result_data[f'{self.target_name}_prob_{class_name}'] = prediction_proba[:, i]
        
        return result_data
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance ranking.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance ranking
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train model first.")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n)
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_type': self.model_type,
            'performance': self.model_performance
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from file.
        
        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.model_type = model_data['model_type']
        self.model_performance = model_data.get('performance')
        
        logger.info(f"Model loaded from {filepath}")


def create_risk_recommender_models(data: pd.DataFrame) -> Dict:
    """
    Create multiple models for different risk prediction tasks.
    
    Args:
        data (pd.DataFrame): Engineered features data
        
    Returns:
        Dict: Dictionary of trained models
    """
    logger.info("Creating risk recommender models")
    
    models = {}
    
    # Model 1: Risk Level Classification (High/Medium/Low)
    if 'risk_level' in data.columns:
        logger.info("Training Risk Level Classification Model")
        risk_level_model = RiskModelingPipeline(model_type='classification')
        X, y = risk_level_model.prepare_model_data(data, 'risk_level')
        performance = risk_level_model.train_model(X, y)
        models['risk_level_classifier'] = risk_level_model
    
    # Model 2: Risk Score Regression
    if 'Risk_Score' in data.columns:
        logger.info("Training Risk Score Regression Model")
        risk_score_model = RiskModelingPipeline(model_type='regression')
        X, y = risk_score_model.prepare_model_data(data, 'Risk_Score')
        performance = risk_score_model.train_model(X, y)
        models['risk_score_regressor'] = risk_score_model
    
    # Model 3: Risk Status Classification
    if 'Risk_Status' in data.columns:
        logger.info("Training Risk Status Classification Model")
        risk_status_model = RiskModelingPipeline(model_type='classification')
        X, y = risk_status_model.prepare_model_data(data, 'Risk_Status')
        performance = risk_status_model.train_model(X, y)
        models['risk_status_classifier'] = risk_status_model
    
    return models


def main():
    """Main function to demonstrate modeling pipeline."""
    # This would typically be called after feature engineering
    print("Modeling Pipeline - Ready for integration with feature engineering")
    
    # Example of initializing the pipeline
    pipeline = RiskModelingPipeline(model_type='classification')
    logger.info("Modeling pipeline initialized")


if __name__ == "__main__":
    main()