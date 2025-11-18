"""
Model monitoring module using Evidently AI for detecting data drift and model degradation.
"""

import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
    RegressionQualityMetric,
    RegressionPredictedVsActualScatter,
    RegressionErrorDistribution
)
import os
from datetime import datetime


class ModelMonitor:
    """Handles model monitoring with Evidently AI."""
    
    def __init__(self, feature_columns, target_column='TARGET_deathRate', prediction_column='prediction'):
        """
        Initialize the monitoring system.
        
        Args:
            feature_columns: List of feature column names
            target_column: Name of the target column
            prediction_column: Name of the prediction column
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.prediction_column = prediction_column
        
        # Set up column mapping for Evidently
        self.column_mapping = ColumnMapping(
            target=target_column,
            prediction=prediction_column,
            numerical_features=feature_columns
        )
        
        # Create reports directory
        self.reports_dir = 'evidently_reports'
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def prepare_reference_data(self, X_train, y_train, y_train_pred):
        """
        Prepare reference dataset (baseline) for comparison.
        
        Args:
            X_train: Training features
            y_train: Training target
            y_train_pred: Training predictions
            
        Returns:
            Reference DataFrame
        """
        # Create reference dataframe
        if isinstance(X_train, pd.DataFrame):
            reference_df = X_train.copy()
        else:
            reference_df = pd.DataFrame(X_train, columns=self.feature_columns)
        
        reference_df[self.target_column] = y_train.values if isinstance(y_train, pd.Series) else y_train
        reference_df[self.prediction_column] = y_train_pred
        
        return reference_df
    
    def prepare_current_data(self, X_test, y_test, y_test_pred):
        """
        Prepare current dataset for comparison.
        
        Args:
            X_test: Test features
            y_test: Test target
            y_test_pred: Test predictions
            
        Returns:
            Current DataFrame
        """
        # Create current dataframe
        if isinstance(X_test, pd.DataFrame):
            current_df = X_test.copy()
        else:
            current_df = pd.DataFrame(X_test, columns=self.feature_columns)
        
        current_df[self.target_column] = y_test.values if isinstance(y_test, pd.Series) else y_test
        current_df[self.prediction_column] = y_test_pred
        
        return current_df
    
    def generate_data_drift_report(self, reference_data, current_data, scenario_name="baseline"):
        """
        Generate data drift report using Evidently.
        
        Args:
            reference_data: Reference (baseline) dataset
            current_data: Current dataset to compare
            scenario_name: Name of the scenario for report naming
            
        Returns:
            Path to the saved report
        """
        print(f"\nGenerating Data Drift Report for {scenario_name}...")
        
        # Create report with data drift preset
        report = Report(metrics=[
            DataDriftPreset(),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric()
        ])
        
        # Run the report
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.reports_dir, 
            f'data_drift_report_{scenario_name}_{timestamp}.html'
        )
        report.save_html(report_path)
        
        print(f"Data Drift Report saved to: {report_path}")
        
        return report_path
    
    def generate_model_performance_report(self, reference_data, current_data, scenario_name="baseline"):
        """
        Generate model performance report using Evidently.
        
        Args:
            reference_data: Reference (baseline) dataset
            current_data: Current dataset to compare
            scenario_name: Name of the scenario for report naming
            
        Returns:
            Path to the saved report
        """
        print(f"\nGenerating Model Performance Report for {scenario_name}...")
        
        # Create report with regression preset
        report = Report(metrics=[
            RegressionPreset(),
            RegressionQualityMetric(),
            RegressionPredictedVsActualScatter(),
            RegressionErrorDistribution()
        ])
        
        # Run the report
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.reports_dir,
            f'model_performance_report_{scenario_name}_{timestamp}.html'
        )
        report.save_html(report_path)
        
        print(f"Model Performance Report saved to: {report_path}")
        
        return report_path
    
    def generate_feature_drift_report(self, reference_data, current_data, 
                                     monitored_features, scenario_name="baseline"):
        """
        Generate detailed feature drift report for specific features.
        
        Args:
            reference_data: Reference (baseline) dataset
            current_data: Current dataset to compare
            monitored_features: List of features to monitor closely
            scenario_name: Name of the scenario for report naming
            
        Returns:
            Path to the saved report
        """
        print(f"\nGenerating Feature Drift Report for {scenario_name}...")
        
        # Create metrics for each monitored feature
        feature_metrics = [ColumnDriftMetric(column_name=feature) for feature in monitored_features]
        
        # Create report
        report = Report(metrics=[
            DatasetDriftMetric(),
            *feature_metrics
        ])
        
        # Run the report
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.reports_dir,
            f'feature_drift_report_{scenario_name}_{timestamp}.html'
        )
        report.save_html(report_path)
        
        print(f"Feature Drift Report saved to: {report_path}")
        
        return report_path
    
    def generate_comprehensive_report(self, reference_data, current_data, 
                                     monitored_features=None, scenario_name="baseline"):
        """
        Generate comprehensive monitoring report including all aspects.
        
        Args:
            reference_data: Reference (baseline) dataset
            current_data: Current dataset to compare
            monitored_features: Optional list of features to monitor closely
            scenario_name: Name of the scenario for report naming
            
        Returns:
            Dictionary with paths to all generated reports
        """
        print(f"\n{'='*60}")
        print(f"GENERATING COMPREHENSIVE MONITORING REPORTS - {scenario_name.upper()}")
        print(f"{'='*60}")
        
        # Generate all reports
        drift_report = self.generate_data_drift_report(reference_data, current_data, scenario_name)
        performance_report = self.generate_model_performance_report(reference_data, current_data, scenario_name)
        
        reports = {
            'data_drift': drift_report,
            'model_performance': performance_report
        }
        
        # Generate feature drift report if specific features are provided
        if monitored_features:
            feature_report = self.generate_feature_drift_report(
                reference_data, current_data, monitored_features, scenario_name
            )
            reports['feature_drift'] = feature_report
        
        print(f"\n{'='*60}")
        print(f"ALL REPORTS GENERATED SUCCESSFULLY FOR {scenario_name.upper()}")
        print(f"{'='*60}")
        
        return reports
    
    def print_drift_summary(self, reference_data, current_data, scenario_name="baseline"):
        """
        Print a summary of data drift detected.
        
        Args:
            reference_data: Reference (baseline) dataset
            current_data: Current dataset to compare
            scenario_name: Name of the scenario
        """
        print(f"\n{'='*60}")
        print(f"DRIFT SUMMARY - {scenario_name.upper()}")
        print(f"{'='*60}")
        
        # Calculate basic statistics for key features
        key_features = ['medIncome', 'povertyPercent', 'AvgHouseholdSize']
        
        for feature in key_features:
            if feature in reference_data.columns and feature in current_data.columns:
                ref_mean = reference_data[feature].mean()
                curr_mean = current_data[feature].mean()
                ref_std = reference_data[feature].std()
                curr_std = current_data[feature].std()
                
                mean_change = curr_mean - ref_mean
                mean_change_pct = (mean_change / ref_mean) * 100 if ref_mean != 0 else 0
                
                print(f"\n{feature}:")
                print(f"  Reference Mean: {ref_mean:.2f} (±{ref_std:.2f})")
                print(f"  Current Mean:   {curr_mean:.2f} (±{curr_std:.2f})")
                print(f"  Change:         {mean_change:+.2f} ({mean_change_pct:+.2f}%)")
        
        # Compare predictions
        if self.prediction_column in reference_data.columns and self.prediction_column in current_data.columns:
            ref_pred_mean = reference_data[self.prediction_column].mean()
            curr_pred_mean = current_data[self.prediction_column].mean()
            pred_change = curr_pred_mean - ref_pred_mean
            pred_change_pct = (pred_change / ref_pred_mean) * 100 if ref_pred_mean != 0 else 0
            
            print(f"\nPredicted Death Rate:")
            print(f"  Reference Mean: {ref_pred_mean:.2f}")
            print(f"  Current Mean:   {curr_pred_mean:.2f}")
            print(f"  Change:         {pred_change:+.2f} ({pred_change_pct:+.2f}%)")
        
        print(f"\n{'='*60}")


if __name__ == "__main__":
    print("This module should be imported and used with trained model data.")
    print("Run main.py to execute the complete pipeline.")

