"""
Main execution script for Cancer Mortality Prediction with Model Monitoring.

This script implements the complete pipeline:
1. Load and preprocess data
2. Train machine learning model
3. Evaluate on test dataset
4. Test three data drift scenarios (A, AB, ABC)
5. Generate Evidently AI monitoring reports for each scenario
"""

import warnings
warnings.filterwarnings('ignore')

from data_loader import CancerDataLoader
from model_trainer import CancerMortalityModel
from model_monitoring import ModelMonitor
from scenario_testing import ScenarioTester, create_baseline_evaluation
import pandas as pd
import numpy as np
from datetime import datetime
import os


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'#'*80}")
    print(f"# {text.center(76)} #")
    print(f"{'#'*80}\n")


def main():
    """Main execution function."""
    start_time = datetime.now()
    
    print_header("CANCER MORTALITY PREDICTION WITH MODEL MONITORING")
    print(f"Execution started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # STEP 1: DATA PREPARATION
    # ========================================================================
    print_header("STEP 1: DATA PREPARATION")
    
    loader = CancerDataLoader(
        data_path='data/cancer_reg.csv',
        test_size=0.2,
        random_state=42
    )
    
    data = loader.prepare_data()
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_columns = data['feature_columns']
    scaler = data['scaler']
    X_test_unscaled = data['X_test_unscaled']
    
    print(f"\nData preparation complete!")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Number of features: {len(feature_columns)}")
    
    # ========================================================================
    # STEP 2: MODEL TRAINING
    # ========================================================================
    print_header("STEP 2: MODEL TRAINING")
    
    model = CancerMortalityModel(
        model_type='gradient_boosting',
        random_state=42
    )
    
    model.train(X_train, y_train)
    
    # Get training predictions for reference data
    y_train_pred = model.predict(X_train)
    
    # Show feature importance
    model.get_feature_importance(feature_columns, top_n=15)
    
    # Save model
    model.save_model('models/cancer_model.pkl')
    
    # ========================================================================
    # STEP 3: BASELINE EVALUATION
    # ========================================================================
    print_header("STEP 3: BASELINE EVALUATION ON TEST SET")
    
    # Initialize monitoring
    monitor = ModelMonitor(
        feature_columns=feature_columns,
        target_column='TARGET_deathRate',
        prediction_column='prediction'
    )
    
    # Create baseline evaluation
    baseline_results = create_baseline_evaluation(
        model, X_test, X_test_unscaled, y_test, monitor
    )
    
    # Prepare reference data (training set) for monitoring
    reference_data = monitor.prepare_reference_data(
        X_train, y_train, y_train_pred
    )
    
    print(f"\nBaseline evaluation complete!")
    print(f"  Test RMSE: {baseline_results['metrics']['RMSE']:.4f}")
    print(f"  Test R²: {baseline_results['metrics']['R2']:.4f}")
    
    # ========================================================================
    # STEP 4: SCENARIO TESTING
    # ========================================================================
    print_header("STEP 4: SCENARIO TESTING WITH DATA DRIFT")
    
    print("\nScenario Descriptions:")
    print("  A:     Decrease medIncome by 40,000")
    print("  AB:    A + Increase povertyPercent by 20 points")
    print("  ABC:   AB + Increase AvgHouseholdSize by 2")
    
    # Initialize scenario tester
    tester = ScenarioTester(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns
    )
    
    # Run all scenarios
    scenario_results = tester.run_all_scenarios(
        X_test_unscaled,
        y_test,
        reference_data,
        monitor
    )
    
    # Save predictions for all scenarios
    tester.save_predictions(output_dir='predictions')
    
    # ========================================================================
    # STEP 5: FINAL SUMMARY
    # ========================================================================
    print_header("EXECUTION SUMMARY")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"Execution completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {duration}")
    
    print("\n" + "="*80)
    print("DELIVERABLES GENERATED:")
    print("="*80)
    print("\n1. TRAINED MODEL:")
    print("   - models/cancer_model.pkl")
    
    print("\n2. PREDICTIONS:")
    print("   - predictions/predictions_scenario_A.csv")
    print("   - predictions/predictions_scenario_AB.csv")
    print("   - predictions/predictions_scenario_ABC.csv")
    
    print("\n3. EVIDENTLY AI MONITORING REPORTS:")
    print("   Location: evidently_reports/")
    print("   - Data Drift Reports (3 scenarios)")
    print("   - Model Performance Reports (3 scenarios)")
    print("   - Feature Drift Reports (3 scenarios)")
    print("   - Scenario Comparison Table (CSV)")
    
    print("\n4. PERFORMANCE METRICS:")
    print(f"   Baseline Test Set:")
    print(f"     - RMSE: {baseline_results['metrics']['RMSE']:.4f}")
    print(f"     - R²:   {baseline_results['metrics']['R2']:.4f}")
    print(f"     - MAE:  {baseline_results['metrics']['MAE']:.4f}")
    
    for scenario_name in ['A', 'AB', 'ABC']:
        metrics = scenario_results[scenario_name]['metrics']
        print(f"\n   Scenario {scenario_name}:")
        print(f"     - RMSE: {metrics['RMSE']:.4f}")
        print(f"     - R²:   {metrics['R2']:.4f}")
        print(f"     - MAE:  {metrics['MAE']:.4f}")
    
    print("\n" + "="*80)
    print("To view the Evidently AI reports, open the HTML files in:")
    print("  evidently_reports/")
    print("="*80)
    
    print("\n✓ All tasks completed successfully!")
    
    return {
        'model': model,
        'baseline_results': baseline_results,
        'scenario_results': scenario_results,
        'monitor': monitor
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "="*80)
        print("SUCCESS: Pipeline executed successfully!")
        print("="*80)
    except Exception as e:
        print("\n" + "="*80)
        print(f"ERROR: {str(e)}")
        print("="*80)
        raise

