#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM Model for Seizure Onset Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib


class LightGBMOnsetDetector:
    """LightGBM-based seizure onset channel detector."""
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        num_leaves: int = 31,
        random_state: int = 42
    ):
        """
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            num_leaves: Maximum number of leaves
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.num_leaves = num_leaves
        self.random_state = random_state
        
        self.model: Optional[lgb.LGBMClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> 'LightGBMOnsetDetector':
        """
        Train the LightGBM model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Binary labels of shape (n_samples,)
            feature_names: Optional list of feature names
            validation_data: Optional (X_val, y_val) for early stopping
        
        Returns:
            Self
        """
        # Store feature names
        self.feature_names = feature_names
        
        # Optional: Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate class weights for imbalanced data
        n_positive = np.sum(y)
        n_negative = len(y) - n_positive
        if n_positive > 0 and n_negative > 0:
            scale_pos_weight = n_negative / n_positive
        else:
            scale_pos_weight = 1.0
        
        # Create LightGBM classifier with class weights
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            verbose=-1,
            scale_pos_weight=scale_pos_weight  # Handle class imbalance
        )
        
        # Prepare validation set for early stopping
        eval_set = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
        
        # Train
        if eval_set is not None:
            self.model.fit(
                X_scaled, y,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
        else:
            self.model.fit(X_scaled, y)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        importance = self.model.feature_importances_
        
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importance))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importance)}
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'num_leaves': self.num_leaves,
                'random_state': self.random_state
            }
        }
        
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'LightGBMOnsetDetector':
        """Load model from file."""
        model_data = joblib.load(filepath)
        
        instance = cls(**model_data['params'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        
        return instance

