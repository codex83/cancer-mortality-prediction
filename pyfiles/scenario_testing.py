"""
Scenario testing module for analyzing model behavior under data drift conditions.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import apply_scenario_changes
from model_monitoring import ModelMonitor
import os


class ScenarioTester:
    """Handles testing different data drift scenarios."""
    
    def __init__(self, model, scaler, feature_columns):
        """
        Initialize scenario tester.
        
        Args:
            model: Trained model instance
            scaler: Fitted StandardScaler instance
            feature_columns: List of feature column names
        """
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.results = {}
    
    def test_scenario(self, X_test_original, y_test, scenario_name, 
                     reference_data=None, monitor=None):
        """
        Test a specific scenario with modified data.
        
        Args:
            X_test_original: Original unscaled test data
            y_test: Test target values
            scenario_name: Name of the scenario ('A', 'AB', 'ABC')
            reference_data: Reference dataset for monitoring
            monitor: ModelMonitor instance
            
        Returns:
            Dictionary with scenario results
        """
        print(f"\n{'='*60}")
        print(f"TESTING SCENARIO: {scenario_name}")
        print(f"{'='*60}")
        
        # Apply scenario changes to unscaled data
        X_test_modified = apply_scenario_changes(X_test_original, scenario_name)
        
        # Scale the modified data using the original scaler
        X_test_scaled = self.scaler.transform(X_test_modified)
        X_test_scaled_df = pd.DataFrame(
            X_test_scaled, 
            columns=self.feature_columns,
            index=X_test_modified.index
        )
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled_df)
        
        # Evaluate model
        print(f"\nEvaluating model performance on Scenario {scenario_name}...")
        metrics = self.model.evaluate(X_test_scaled_df, y_test, f"Scenario {scenario_name}")
        
        # Prepare current data for monitoring
        if monitor is not None and reference_data is not None:
            current_data = monitor.prepare_current_data(
                X_test_modified,  # Use unscaled data for monitoring
                y_test,
                y_pred
            )
            
            # Print drift summary
            monitor.print_drift_summary(reference_data, current_data, scenario_name)
            
            # Generate comprehensive monitoring reports
            monitored_features = ['medIncome', 'povertyPercent', 'AvgHouseholdSize']
            reports = monitor.generate_comprehensive_report(
                reference_data,
                current_data,
                monitored_features=monitored_features,
                scenario_name=scenario_name
            )
        else:
            reports = None
        
        # Store results
        results = {
            'scenario': scenario_name,
            'metrics': metrics,
            'predictions': y_pred,
            'modified_data': X_test_modified,
            'scaled_data': X_test_scaled_df,
            'reports': reports
        }
        
        self.results[scenario_name] = results
        
        return results
    
    def run_all_scenarios(self, X_test_original, y_test, reference_data, monitor):
        """
        Run all three scenarios (A, AB, ABC) and generate reports.
        
        Args:
            X_test_original: Original unscaled test data
            y_test: Test target values
            reference_data: Reference dataset for monitoring
            monitor: ModelMonitor instance
            
        Returns:
            Dictionary with all scenario results
        """
        print(f"\n{'#'*60}")
        print(f"# RUNNING ALL SCENARIO TESTS")
        print(f"{'#'*60}")
        
        scenarios = ['A', 'AB', 'ABC']
        
        for scenario in scenarios:
            self.test_scenario(
                X_test_original,
                y_test,
                scenario,
                reference_data,
                monitor
            )
        
        # Generate comparison summary
        self.generate_comparison_summary()
        
        return self.results
    
    def generate_comparison_summary(self):
        """Generate a comparison summary of all scenarios."""
        print(f"\n{'='*60}")
        print(f"SCENARIO COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        # Create comparison table
        comparison_data = []
        
        for scenario_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Scenario': scenario_name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R²': metrics['R2'],
                'MAPE (%)': metrics['MAPE']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\nMetrics Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Calculate percentage changes from baseline (assuming 'A' is first)
        if 'A' in self.results:
            baseline_rmse = self.results['A']['metrics']['RMSE']
            baseline_r2 = self.results['A']['metrics']['R2']
            
            print(f"\nChanges from Scenario A (Baseline):")
            for scenario_name in ['AB', 'ABC']:
                if scenario_name in self.results:
                    rmse = self.results[scenario_name]['metrics']['RMSE']
                    r2 = self.results[scenario_name]['metrics']['R2']
                    
                    rmse_change = ((rmse - baseline_rmse) / baseline_rmse) * 100
                    r2_change = ((r2 - baseline_r2) / baseline_r2) * 100
                    
                    print(f"  Scenario {scenario_name}:")
                    print(f"    RMSE change: {rmse_change:+.2f}%")
                    print(f"    R² change:   {r2_change:+.2f}%")
        
        print(f"\n{'='*60}")
        
        # Save comparison to file
        comparison_path = os.path.join('evidently_reports', 'scenario_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison table saved to: {comparison_path}")
    
    def save_predictions(self, output_dir='predictions'):
        """
        Save predictions for all scenarios to CSV files.
        
        Args:
            output_dir: Directory to save prediction files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving predictions to {output_dir}/...")
        
        for scenario_name, result in self.results.items():
            predictions_df = result['modified_data'].copy()
            predictions_df['predictions'] = result['predictions']
            
            output_path = os.path.join(output_dir, f'predictions_scenario_{scenario_name}.csv')
            predictions_df.to_csv(output_path, index=False)
            print(f"  Saved: {output_path}")
        
        print("All predictions saved successfully!")


def create_baseline_evaluation(model, X_test_scaled, X_test_unscaled, y_test, monitor):
    """
    Create baseline evaluation using the original test set.
    
    Args:
        model: Trained model instance
        X_test_scaled: Scaled test features
        X_test_unscaled: Unscaled test features
        y_test: Test target values
        monitor: ModelMonitor instance
        
    Returns:
        Baseline results dictionary
    """
    print(f"\n{'='*60}")
    print(f"BASELINE EVALUATION (Original Test Set)")
    print(f"{'='*60}")
    
    # Make predictions on original test set
    y_pred_baseline = model.predict(X_test_scaled)
    
    # Evaluate model
    metrics = model.evaluate(X_test_scaled, y_test, "Baseline Test")
    
    # Show sample predictions
    model.compare_predictions(y_test, y_pred_baseline, num_samples=10)
    
    # Prepare baseline data for monitoring (as reference)
    baseline_data = monitor.prepare_current_data(
        X_test_unscaled,
        y_test,
        y_pred_baseline
    )
    
    baseline_results = {
        'metrics': metrics,
        'predictions': y_pred_baseline,
        'data': baseline_data
    }
    
    return baseline_results


if __name__ == "__main__":
    print("This module should be imported and used with a trained model.")
    print("Run main.py to execute the complete pipeline.")

