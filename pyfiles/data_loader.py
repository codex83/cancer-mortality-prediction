"""
Data loading and preprocessing module for cancer mortality prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class CancerDataLoader:
    """Handles data loading, preprocessing, and feature engineering."""
    
    def __init__(self, data_path='data/cancer_reg.csv', test_size=0.2, random_state=42):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to the cancer dataset CSV file
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'TARGET_deathRate'
        
    def load_data(self):
        """Load the cancer dataset from CSV."""
        print(f"Loading data from {self.data_path}...")
        # Try different encodings to handle various CSV formats
        try:
            df = pd.read_csv(self.data_path)
        except UnicodeDecodeError:
            print("  UTF-8 encoding failed, trying latin-1...")
            df = pd.read_csv(self.data_path, encoding='latin-1')
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset: handle missing values, select features.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Preprocessed dataframe
        """
        print("\nPreprocessing data...")
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Drop non-numeric columns that aren't useful for modeling
        columns_to_drop = ['Geography', 'binnedInc']
        df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')
        
        # Handle missing values
        print(f"Missing values before imputation: {df_processed.isnull().sum().sum()}")
        
        # Fill numeric columns with median
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        print(f"Missing values after imputation: {df_processed.isnull().sum().sum()}")
        
        # Ensure target column exists
        if self.target_column not in df_processed.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        
        return df_processed
    
    def split_data(self, df):
        """
        Split data into training and testing sets.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            X_train, X_test, y_train, y_test, feature_columns
        """
        print("\nSplitting data into train and test sets...")
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Number of features: {len(X_train.columns)}")
        
        return X_train, X_test, y_train, y_test, self.feature_columns
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            X_train_scaled, X_test_scaled
        """
        print("\nScaling features...")
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        return X_train_scaled, X_test_scaled
    
    def prepare_data(self):
        """
        Complete data preparation pipeline.
        
        Returns:
            Dictionary containing all prepared data
        """
        # Load data
        df = self.load_data()
        
        # Preprocess
        df_processed = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test, feature_columns = self.split_data(df_processed)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Return all data components
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': feature_columns,
            'scaler': self.scaler,
            'X_test_unscaled': X_test  # Keep unscaled version for monitoring
        }


def apply_scenario_changes(X_test, scenario='A'):
    """
    Apply data changes according to assignment scenarios.
    
    Args:
        X_test: Test dataset
        scenario: 'A', 'AB', or 'ABC'
        
    Returns:
        Modified test dataset
    """
    X_modified = X_test.copy()
    
    if scenario in ['A', 'AB', 'ABC']:
        # Scenario A: Decrease medIncome by 40,000
        if 'medIncome' in X_modified.columns:
            print(f"\nApplying Scenario A: Decreasing medIncome by 40,000")
            X_modified['medIncome'] = X_modified['medIncome'] - 40000
            # Ensure no negative values
            X_modified['medIncome'] = X_modified['medIncome'].clip(lower=0)
    
    if scenario in ['AB', 'ABC']:
        # Scenario B: Increase povertyPercent by 20 points
        if 'povertyPercent' in X_modified.columns:
            print(f"Applying Scenario B: Increasing povertyPercent by 20 points")
            X_modified['povertyPercent'] = X_modified['povertyPercent'] + 20
            # Cap at 100 (percentage)
            X_modified['povertyPercent'] = X_modified['povertyPercent'].clip(upper=100)
    
    if scenario == 'ABC':
        # Scenario C: Increase AvgHouseholdSize by 2
        if 'AvgHouseholdSize' in X_modified.columns:
            print(f"Applying Scenario C: Increasing AvgHouseholdSize by 2")
            X_modified['AvgHouseholdSize'] = X_modified['AvgHouseholdSize'] + 2
    
    return X_modified


if __name__ == "__main__":
    # Test the data loader
    loader = CancerDataLoader()
    data = loader.prepare_data()
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Training features shape: {data['X_train'].shape}")
    print(f"Test features shape: {data['X_test'].shape}")
    print(f"Training target shape: {data['y_train'].shape}")
    print(f"Test target shape: {data['y_test'].shape}")
    print(f"\nFeatures ({len(data['feature_columns'])}):")
    for i, col in enumerate(data['feature_columns'], 1):
        print(f"{i}. {col}")

