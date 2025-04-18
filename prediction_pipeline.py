#!/usr/bin/env python3
"""
Valorant Match Prediction Pipeline

This script provides a streamlined workflow for the entire prediction process:
1. Make predictions for upcoming matches
2. Record actual match results
3. Analyze prediction performance
4. Retrain the model with new data

Each component is modular and can be run independently.
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

# Determine the Python executable to use
PYTHON_EXECUTABLE = sys.executable  # Use the same Python that's running this script

def run_prediction(team1, team2, team1_region=None, team2_region=None, advanced=False, output_dir=None):
    """Run the prediction script for a match."""
    print(f"\n{'='*40}")
    print(f"PREDICTING: {team1} vs {team2}")
    print(f"{'='*40}")
    
    # Build the command exactly as it works directly
    cmd = [PYTHON_EXECUTABLE, "main.py"]
    
    if team1:
        cmd.extend(["--team1", team1])
    
    if team1_region:
        cmd.extend(["--team1_region", team1_region])
    
    if team2:
        cmd.extend(["--team2", team2])
    
    if team2_region:
        cmd.extend(["--team2_region", team2_region])
    
    if advanced:
        cmd.append("--advanced")
    
    print(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Extract prediction file path from output
        prediction_file = None
        for line in result.stdout.split("\n"):
            if "Prediction saved to" in line:
                potential_path = line.split("Prediction saved to")[1].strip()
                if os.path.exists(potential_path):
                    prediction_file = potential_path
                    break
        
        if not prediction_file:
            # Try to find the prediction file by looking in the predictions directory
            pred_dir = "predictions"
            if os.path.exists(pred_dir):
                json_files = [f for f in os.listdir(pred_dir) if f.endswith(".json") and f"{team1}_vs_{team2}" in f]
                if json_files:
                    # Get the most recent file
                    json_files.sort(reverse=True)
                    prediction_file = os.path.join(pred_dir, json_files[0])
        
        # If output directory is specified, copy the prediction file there
        if prediction_file and output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create a custom name for the copied file
            filename = os.path.basename(prediction_file)
            output_file = os.path.join(output_dir, filename)
            
            # Copy file contents
            with open(prediction_file, 'r') as source:
                prediction_data = json.load(source)
            
            with open(output_file, 'w') as dest:
                json.dump(prediction_data, dest, indent=4)
            
            print(f"\nPrediction file copied to: {output_file}")
        
        return prediction_file
    
    except subprocess.CalledProcessError as e:
        print(f"Error running prediction: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return None

def record_result(prediction_file, actual_winner, match_date=None, notes=None, model_version="1.0", use_api=True):
    """Record the actual result of a match."""
    print(f"\n{'='*40}")
    print(f"RECORDING RESULT: Winner = {actual_winner}")
    print(f"{'='*40}")
    
    cmd = [PYTHON_EXECUTABLE, "result_tracker_enhanced.py", "record", prediction_file, actual_winner]
    
    if match_date:
        cmd.extend(["--match_date", match_date])
    
    if notes:
        cmd.extend(["--notes", notes])
    
    if model_version:
        cmd.extend(["--model_version", model_version])
    
    if not use_api:
        cmd.append("--no_fetch")
    
    print(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error recording result: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def analyze_performance(period=None, model_version=None, generate_report=False):
    """Analyze prediction performance."""
    print(f"\n{'='*40}")
    print(f"ANALYZING PERFORMANCE")
    print(f"{'='*40}")
    
    if generate_report:
        cmd = [PYTHON_EXECUTABLE, "performance_analyzer.py", "report"]
    else:
        cmd = [PYTHON_EXECUTABLE, "performance_analyzer.py", "metrics"]
    
    if period:
        cmd.extend(["--period", period])
    
    if model_version:
        cmd.extend(["--model_version", model_version])
    
    print(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing performance: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def retrain_model(weighting="recency", model_type="ensemble", recency_days=90, skip_feature_selection=False, 
                skip_hyperparameter_tuning=False, use_smote=True, use_api=True, model_name=None):
    """Retrain the prediction model."""
    print(f"\n{'='*40}")
    print(f"RETRAINING MODEL")
    print(f"{'='*40}")
    
    cmd = [PYTHON_EXECUTABLE, "model_retrainer.py", "train", "--weighting", weighting, "--model_type", model_type, 
          "--recency_days", str(recency_days)]
    
    if skip_feature_selection:
        cmd.append("--no_feature_selection")
    
    if skip_hyperparameter_tuning:
        cmd.append("--no_hyperparameter_tuning")
    
    if not use_smote:
        cmd.append("--no_smote")
    
    if not use_api:
        cmd.append("--no_api_data")
    
    if model_name:
        cmd.extend(["--model_name", model_name])
    
    print(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Extract model name from output
        model_name = None
        for line in result.stdout.split("\n"):
            if "Model Name:" in line:
                model_name = line.split("Model Name:")[1].strip()
                break
        
        return model_name
    
    except subprocess.CalledProcessError as e:
        print(f"Error retraining model: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return None

def analyze_errors(max_history=50, output_file=None):
    """Analyze patterns in prediction errors."""
    print(f"\n{'='*40}")
    print(f"ANALYZING ERROR PATTERNS")
    print(f"{'='*40}")
    
    cmd = [PYTHON_EXECUTABLE, "model_retrainer.py", "analyze", "--max_history", str(max_history)]
    
    if output_file:
        cmd.extend(["--output", output_file])
    
    print(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing errors: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def suggest_improvements():
    """Get suggestions for model improvements."""
    print(f"\n{'='*40}")
    print(f"SUGGESTING IMPROVEMENTS")
    print(f"{'='*40}")
    
    cmd = [PYTHON_EXECUTABLE, "model_retrainer.py", "suggest"]
    
    print(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error suggesting improvements: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def run_full_pipeline(team1, team2, actual_winner=None, team1_region=None, team2_region=None, 
                    advanced=False, match_date=None, notes=None, analyze=True, retrain=False, 
                    weighting="recency", model_type="ensemble"):
    """Run the full prediction pipeline."""
    print(f"\n{'='*50}")
    print(f"STARTING FULL PIPELINE: {team1} vs {team2}")
    print(f"{'='*50}")
    
    # Step 1: Make prediction
    prediction_file = run_prediction(team1, team2, team1_region, team2_region, advanced)
    
    if not prediction_file:
        print("Failed to generate prediction. Pipeline stopped.")
        return False
    
    # Step 2: Record result (if provided)
    if actual_winner:
        success = record_result(prediction_file, actual_winner, match_date, notes)
        
        if not success:
            print("Failed to record result. Pipeline stopped.")
            return False
    
    # Step 3: Analyze performance (if requested)
    if analyze:
        success = analyze_performance(generate_report=True)
        
        if not success:
            print("Failed to analyze performance. Pipeline continuing...")
        
        # Also analyze error patterns
        success = analyze_errors()
        
        if not success:
            print("Failed to analyze error patterns. Pipeline continuing...")
    
    # Step 4: Retrain model (if requested)
    if retrain:
        model_name = retrain_model(weighting=weighting, model_type=model_type)
        
        if not model_name:
            print("Failed to retrain model. Pipeline stopped.")
            return False
        
        # Get suggestions for further improvements
        suggest_improvements()
    
    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*50}")
    
    return True

def main():
    """Command line interface for the prediction pipeline."""
    parser = argparse.ArgumentParser(description='Valorant Match Prediction Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make prediction for a match')
    predict_parser.add_argument('--team1', required=True, help='Name of the first team')
    predict_parser.add_argument('--team2', required=True, help='Name of the second team')
    predict_parser.add_argument('--team1_region', help='Region of the first team (na, eu, etc.)')
    predict_parser.add_argument('--team2_region', help='Region of the second team (na, eu, etc.)')
    predict_parser.add_argument('--advanced', action='store_true', help='Use advanced prediction model')
    predict_parser.add_argument('--output_dir', help='Directory to store prediction output')
    
    # Record command
    record_parser = subparsers.add_parser('record', help='Record match result')
    record_parser.add_argument('--prediction_file', required=True, help='Path to prediction JSON file')
    record_parser.add_argument('--winner', required=True, help='Name of the team that won')
    record_parser.add_argument('--match_date', help='Date of the match (YYYY-MM-DD)')
    record_parser.add_argument('--notes', help='Additional notes about the match')
    record_parser.add_argument('--model_version', default="1.0", help='Version of the model used')
    record_parser.add_argument('--no_api', action='store_true', help='Skip fetching data from API')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze prediction performance')
    analyze_parser.add_argument('--period', choices=['1m', '3m', '6m', '1y'], help='Time period to analyze')
    analyze_parser.add_argument('--model_version', help='Model version to analyze')
    analyze_parser.add_argument('--report', action='store_true', help='Generate comprehensive report')
    
    # Retrain command
    retrain_parser = subparsers.add_parser('retrain', help='Retrain prediction model')
    retrain_parser.add_argument('--weighting', choices=['recency', 'prediction_error', 'both', 'none'], 
                               default='recency', help='Sample weighting scheme')
    retrain_parser.add_argument('--model_type', choices=['rf', 'gbm', 'ensemble'], 
                               default='ensemble', help='Type of model to train')
    retrain_parser.add_argument('--recency_days', type=int, default=90, 
                               help='Number of days to consider as "recent"')
    retrain_parser.add_argument('--skip_feature_selection', action='store_true', 
                               help='Skip feature selection')
    retrain_parser.add_argument('--skip_hyperparameter_tuning', action='store_true', 
                               help='Skip hyperparameter tuning')
    retrain_parser.add_argument('--no_smote', action='store_true', 
                               help='Skip SMOTE for class imbalance')
    retrain_parser.add_argument('--no_api', action='store_true', 
                               help='Skip fetching data from API')
    retrain_parser.add_argument('--model_name', help='Custom name for the model')
    
    # Errors command
    errors_parser = subparsers.add_parser('errors', help='Analyze prediction errors')
    errors_parser.add_argument('--max_history', type=int, default=50, 
                              help='Maximum number of recent results to analyze')
    errors_parser.add_argument('--output', help='Output JSON file for analysis')
    
    # Improve command
    improve_parser = subparsers.add_parser('improve', help='Get model improvement suggestions')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full prediction pipeline')
    pipeline_parser.add_argument('--team1', required=True, help='Name of the first team')
    pipeline_parser.add_argument('--team2', required=True, help='Name of the second team')
    pipeline_parser.add_argument('--winner', help='Name of the team that won (if known)')
    pipeline_parser.add_argument('--team1_region', help='Region of the first team (na, eu, etc.)')
    pipeline_parser.add_argument('--team2_region', help='Region of the second team (na, eu, etc.)')
    pipeline_parser.add_argument('--advanced', action='store_true', help='Use advanced prediction model')
    pipeline_parser.add_argument('--match_date', help='Date of the match (YYYY-MM-DD)')
    pipeline_parser.add_argument('--notes', help='Additional notes about the match')
    pipeline_parser.add_argument('--no_analyze', action='store_true', help='Skip performance analysis')
    pipeline_parser.add_argument('--retrain', action='store_true', help='Retrain model with new data')
    pipeline_parser.add_argument('--weighting', choices=['recency', 'prediction_error', 'both', 'none'], 
                                default='recency', help='Sample weighting scheme for retraining')
    pipeline_parser.add_argument('--model_type', choices=['rf', 'gbm', 'ensemble'], 
                                default='ensemble', help='Model type for retraining')
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        run_prediction(args.team1, args.team2, args.team1_region, args.team2_region, args.advanced, args.output_dir)
    
    elif args.command == 'record':
        record_result(args.prediction_file, args.winner, args.match_date, args.notes, 
                     args.model_version, not args.no_api)
    
    elif args.command == 'analyze':
        analyze_performance(args.period, args.model_version, args.report)
    
    elif args.command == 'retrain':
        retrain_model(args.weighting, args.model_type, args.recency_days, 
                     args.skip_feature_selection, args.skip_hyperparameter_tuning, 
                     not args.no_smote, not args.no_api, args.model_name)
    
    elif args.command == 'errors':
        analyze_errors(args.max_history, args.output)
    
    elif args.command == 'improve':
        suggest_improvements()
    
    elif args.command == 'pipeline':
        run_full_pipeline(args.team1, args.team2, args.winner, args.team1_region, args.team2_region,
                         args.advanced, args.match_date, args.notes, not args.no_analyze,
                         args.retrain, args.weighting, args.model_type)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()