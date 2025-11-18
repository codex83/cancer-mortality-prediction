"""
Model training and evaluation module for cancer mortality prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os


class CancerMortalityModel:
    """Handles model training, evaluation, and prediction."""
    
    def __init__(self, model_type='gradient_boosting', random_state=42):
        """
        Initialize the model.
        
        Args:
            model_type: Type of model to use ('gradient_boosting', 'random_forest', 'ridge')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.training_metrics = {}
        self.test_metrics = {}
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on model_type."""
        if self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                verbose=0
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            )
        elif self.model_type == 'ridge':
            self.model = Ridge(
                alpha=1.0,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        print(f"\nTraining {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        self.training_metrics = self._calculate_metrics(y_train, y_train_pred, "Training")
        
        print(f"Training complete!")
        
        return self
    
    def evaluate(self, X_test, y_test, dataset_name="Test"):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            dataset_name: Name of the dataset for display
            
        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating model on {dataset_name} data...")
        y_pred = self.model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, dataset_name)
        
        if dataset_name == "Test":
            self.test_metrics = metrics
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, dataset_name):
        """
        Calculate and display regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            dataset_name: Name of the dataset for display
            
        Returns:
            Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        print(f"\n{dataset_name} Metrics:")
        print(f"  RMSE:  {rmse:.4f}")
        print(f"  MAE:   {mae:.4f}")
        print(f"  R²:    {r2:.4f}")
        print(f"  MAPE:  {mape:.2f}%")
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.model.predict(X)
    
    def save_model(self, filepath='models/cancer_model.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"\nModel saved to {filepath}")
    
    @staticmethod
    def load_model(filepath='models/cancer_model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
        return model
    
    def get_feature_importance(self, feature_names, top_n=20):
        """
        Get feature importance for tree-based models.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.model_type in ['gradient_boosting', 'random_forest']:
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop {top_n} Most Important Features:")
            print("="*60)
            for idx, row in feature_importance.head(top_n).iterrows():
                print(f"{row['feature']:30s} {row['importance']:.6f}")
            
            return feature_importance
        else:
            print(f"Feature importance not available for {self.model_type}")
            return None
    
    def compare_predictions(self, y_true, y_pred, num_samples=10):
        """
        Display comparison of actual vs predicted values.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            num_samples: Number of samples to display
        """
        comparison = pd.DataFrame({
            'Actual': y_true.values[:num_samples] if isinstance(y_true, pd.Series) else y_true[:num_samples],
            'Predicted': y_pred[:num_samples],
            'Difference': np.abs(y_true.values[:num_samples] - y_pred[:num_samples]) if isinstance(y_true, pd.Series) else np.abs(y_true[:num_samples] - y_pred[:num_samples])
        })
        
        print(f"\nSample Predictions (first {num_samples}):")
        print("="*60)
        print(comparison.to_string(index=False))
        print("="*60)


def create_predictions_dataframe(X_test, y_test, y_pred, feature_columns):
    """
    Create a comprehensive dataframe with features, actual and predicted values.
    
    Args:
        X_test: Test features
        y_test: Actual target values
        y_pred: Predicted target values
        feature_columns: List of feature column names
        
    Returns:
        DataFrame with predictions
    """
    # Create base dataframe with features
    if isinstance(X_test, pd.DataFrame):
        predictions_df = X_test.copy()
    else:
        predictions_df = pd.DataFrame(X_test, columns=feature_columns)
    
    # Add actual and predicted values
    predictions_df['actual_deathRate'] = y_test.values if isinstance(y_test, pd.Series) else y_test
    predictions_df['predicted_deathRate'] = y_pred
    predictions_df['prediction_error'] = np.abs(predictions_df['actual_deathRate'] - predictions_df['predicted_deathRate'])
    
    return predictions_df


if __name__ == "__main__":
    print("This module should be imported and used with prepared data.")
    print("Run main.py to execute the complete pipeline.")

