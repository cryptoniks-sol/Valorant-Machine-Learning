import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns
from datetime import datetime, timedelta
import argparse

class PerformanceAnalyzer:
    """
    Analyzes the performance of the match prediction model.
    """
    def __init__(self, history_file='prediction_results/prediction_history.csv', 
                 results_dir='prediction_results', output_dir='performance_analysis'):
        """
        Initialize the performance analyzer.
        
        Args:
            history_file: CSV file with prediction history
            results_dir: Directory where prediction results are stored
            output_dir: Directory where analysis results will be stored
        """
        self.history_file = history_file
        self.results_dir = results_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load prediction history
        if not os.path.exists(self.history_file):
            raise FileNotFoundError(f"History file not found: {self.history_file}")
        
        self.history_df = pd.read_csv(self.history_file)
        
        # Convert dates to datetime
        self.history_df['match_date'] = pd.to_datetime(self.history_df['match_date'])
        if 'prediction_date' in self.history_df.columns:
            self.history_df['prediction_date'] = pd.to_datetime(self.history_df['prediction_date'])
    
    def calculate_metrics(self, period=None, model_version=None):
        """
        Calculate various performance metrics for the model.
        
        Args:
            period: Optional time period (e.g., '1m', '3m', '6m', '1y')
            model_version: Optional model version to filter by
            
        Returns:
            dict: Dictionary of performance metrics
        """
        # Filter data based on period and model version
        df = self._filter_data(self.history_df, period, model_version)
        
        if len(df) == 0:
            return {"error": "No data available for the specified filters"}
        
        # Extract actual and predicted values
        y_true = df['prediction_correct'].astype(bool).values
        y_pred = np.ones(len(df), dtype=bool)  # Model's prediction (always expects to be correct)
        y_prob = df['win_probability'].values  # Model's confidence
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'total_predictions': len(df),
            'correct_predictions': sum(y_true),
            'time_period': period or 'all',
            'model_version': model_version or 'all'
        }
        
        # Calculate calibration metrics (reliability)
        calibration_data = self._calculate_calibration(df)
        metrics['calibration'] = calibration_data
        
        # Identify patterns in incorrect predictions
        error_patterns = self._analyze_errors(df)
        metrics['error_patterns'] = error_patterns
        
        # Calculate feature importance analysis if possible
        feature_importance = self._analyze_feature_importance(df)
        metrics['feature_importance'] = feature_importance
        
        # Add map analysis if available
        if 'map' in df.columns and df['map'].notna().any():
            map_analysis = self._analyze_map_performance(df)
            metrics['map_analysis'] = map_analysis
        
        # Add score analysis if available
        if all(col in df.columns for col in ['team1_score', 'team2_score']):
            score_analysis = self._analyze_score_metrics(df)
            metrics['score_analysis'] = score_analysis
        
        return metrics
    
    def _filter_data(self, df, period=None, model_version=None):
        """
        Filter data based on time period and model version.
        
        Args:
            df: DataFrame to filter
            period: Time period (e.g., '1m', '3m', '6m', '1y')
            model_version: Model version to filter by
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Filter by model version
        if model_version:
            filtered_df = filtered_df[filtered_df['model_version'] == model_version]
        
        # Filter by time period
        if period:
            end_date = datetime.now()
            
            if period == '1m':
                start_date = end_date - timedelta(days=30)
            elif period == '3m':
                start_date = end_date - timedelta(days=90)
            elif period == '6m':
                start_date = end_date - timedelta(days=180)
            elif period == '1y':
                start_date = end_date - timedelta(days=365)
            else:
                # Invalid period, use all data
                return filtered_df
            
            filtered_df = filtered_df[(filtered_df['match_date'] >= start_date) & 
                                     (filtered_df['match_date'] <= end_date)]
        
        return filtered_df
    
    def _calculate_calibration(self, df):
        """
        Calculate calibration metrics (reliability).
        
        Args:
            df: DataFrame with prediction data
            
        Returns:
            dict: Calibration metrics
        """
        # Create probability bins
        bins = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
        df['prob_bin'] = pd.cut(df['win_probability'], bins, labels=bins[:-1] + 0.05)
        
        # Calculate actual win rate in each bin
        calibration = df.groupby('prob_bin').agg(
            count=('match_id', 'count'),
            wins=('prediction_correct', 'sum')
        )
        calibration['actual_win_rate'] = calibration['wins'] / calibration['count']
        calibration = calibration.reset_index()
        
        # Calculate calibration error
        bin_probs = calibration['prob_bin'].astype(float).values
        actual_probs = calibration['actual_win_rate'].values
        calibration['error'] = bin_probs - actual_probs
        
        # Calculate overall calibration error metrics
        valid_rows = ~np.isnan(actual_probs)
        if sum(valid_rows) > 0:
            mce = np.mean(np.abs(bin_probs[valid_rows] - actual_probs[valid_rows]))  # Mean Calibration Error
        else:
            mce = None
        
        return {
            'bin_data': calibration.to_dict(orient='records'),
            'mean_calibration_error': mce
        }
    
    def _analyze_errors(self, df):
        """
        Analyze patterns in incorrect predictions.
        
        Args:
            df: DataFrame with prediction data
            
        Returns:
            dict: Error analysis
        """
        # Filter incorrect predictions
        incorrect_df = df[~df['prediction_correct']]
        
        if len(incorrect_df) == 0:
            return {"no_errors": True}
        
        # Analyze by confidence level
        incorrect_df['confidence_level'] = pd.cut(
            incorrect_df['win_probability'], 
            bins=[0, 0.55, 0.65, 0.75, 0.85, 1.0],
            labels=['Very Low (≤55%)', 'Low (56-65%)', 'Medium (66-75%)', 'High (76-85%)', 'Very High (>85%)']
        )
        
        confidence_distribution = incorrect_df['confidence_level'].value_counts().to_dict()
        
        # Find teams that most often cause prediction errors
        team_error_counts = {}
        for _, row in incorrect_df.iterrows():
            predicted_winner = row['predicted_winner']
            actual_winner = row['actual_winner']
            
            # Increment count for team that was incorrectly predicted to win
            team_error_counts[predicted_winner] = team_error_counts.get(predicted_winner, 0) + 1
            
            # Also track the unexpected winner
            team_error_counts[f"{actual_winner}_upset"] = team_error_counts.get(f"{actual_winner}_upset", 0) + 1
        
        # Sort teams by error count
        team_error_counts = {k: v for k, v in sorted(team_error_counts.items(), 
                                                    key=lambda item: item[1], reverse=True)}
        
        # Try to load detailed match data for feature analysis
        feature_issues = {}
        feature_counts = {}
        
        for match_id in incorrect_df['match_id']:
            result_file = os.path.join(self.results_dir, f"{match_id}_result.json")
            
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        match_data = json.load(f)
                    
                    # If feature importance is available, analyze which features led to errors
                    if 'feature_importance' in match_data:
                        for feature, importance in match_data['feature_importance'].items():
                            if isinstance(importance, dict) and 'Importance' in importance:
                                importance_value = importance['Importance']
                            else:
                                importance_value = importance
                                
                            feature_counts[feature] = feature_counts.get(feature, 0) + float(importance_value)
                except Exception as e:
                    print(f"Error analyzing match {match_id}: {e}")
        
        # Sort features by importance in incorrect predictions
        if feature_counts:
            feature_counts = {k: v for k, v in sorted(feature_counts.items(), 
                                                     key=lambda item: item[1], reverse=True)}
            # Take top 5 features
            feature_issues = dict(list(feature_counts.items())[:5])
        
        return {
            'total_errors': len(incorrect_df),
            'error_rate': len(incorrect_df) / len(df),
            'confidence_distribution': confidence_distribution,
            'problematic_teams': dict(list(team_error_counts.items())[:5]),
            'misleading_features': feature_issues
        }
    
    def _analyze_feature_importance(self, df):
        """
        Analyze feature importance across predictions.
        
        Args:
            df: DataFrame with prediction data
            
        Returns:
            dict: Feature importance analysis
        """
        # Try to load detailed match data for feature analysis
        feature_importance = {}
        feature_counts = {}
        
        for match_id in df['match_id']:
            result_file = os.path.join(self.results_dir, f"{match_id}_result.json")
            
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        match_data = json.load(f)
                    
                    # If feature importance is available, analyze
                    if 'feature_importance' in match_data:
                        for feature, importance in match_data['feature_importance'].items():
                            if isinstance(importance, dict) and 'Importance' in importance:
                                importance_value = importance['Importance']
                            else:
                                importance_value = importance
                                
                            feature_importance[feature] = feature_importance.get(feature, 0) + float(importance_value)
                            feature_counts[feature] = feature_counts.get(feature, 0) + 1
                except Exception as e:
                    print(f"Error analyzing match {match_id}: {e}")
        
        # Calculate average importance
        avg_importance = {}
        for feature, total in feature_importance.items():
            count = feature_counts.get(feature, 1)  # Avoid division by zero
            avg_importance[feature] = total / count
        
        # Sort features by average importance
        avg_importance = {k: v for k, v in sorted(avg_importance.items(), 
                                                 key=lambda item: item[1], reverse=True)}
        
        # Analyze feature importance in correct vs incorrect predictions
        correct_importance = {}
        incorrect_importance = {}
        
        for match_id, correct in zip(df['match_id'], df['prediction_correct']):
            result_file = os.path.join(self.results_dir, f"{match_id}_result.json")
            
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        match_data = json.load(f)
                    
                    if 'feature_importance' in match_data:
                        for feature, importance in match_data['feature_importance'].items():
                            if isinstance(importance, dict) and 'Importance' in importance:
                                importance_value = importance['Importance']
                            else:
                                importance_value = importance
                                
                            if correct:
                                correct_importance[feature] = correct_importance.get(feature, 0) + float(importance_value)
                            else:
                                incorrect_importance[feature] = incorrect_importance.get(feature, 0) + float(importance_value)
                except Exception as e:
                    print(f"Error analyzing match {match_id}: {e}")
        
        # Identify potentially misleading features
        misleading_features = {}
        for feature in incorrect_importance:
            if feature in correct_importance:
                correct_value = correct_importance[feature] / sum(df['prediction_correct'])
                incorrect_value = incorrect_importance[feature] / (len(df) - sum(df['prediction_correct']))
                
                # If feature is more important in incorrect predictions
                if incorrect_value > correct_value:
                    misleading_features[feature] = incorrect_value / correct_value
        
        # Sort misleading features
        misleading_features = {k: v for k, v in sorted(misleading_features.items(), 
                                                      key=lambda item: item[1], reverse=True)}
        
        return {
            'overall_importance': dict(list(avg_importance.items())[:10]),
            'misleading_features': dict(list(misleading_features.items())[:5])
        }
    
    def _analyze_map_performance(self, df):
        """
        Analyze prediction performance by map.
        
        Args:
            df: DataFrame with prediction data
            
        Returns:
            dict: Map performance analysis
        """
        if 'map' not in df.columns or df['map'].isna().all():
            return {"error": "No map data available"}
        
        # Group by map
        map_performance = df.groupby('map').agg(
            total=('match_id', 'count'),
            correct=('prediction_correct', 'sum')
        )
        map_performance['accuracy'] = map_performance['correct'] / map_performance['total']
        
        # Sort by accuracy
        map_performance = map_performance.sort_values('accuracy', ascending=False)
        
        return map_performance.to_dict()
    
    def _analyze_score_metrics(self, df):
        """
        Analyze match score metrics to identify patterns.
        
        Args:
            df: DataFrame with prediction data
            
        Returns:
            dict: Score analysis
        """
        if not all(col in df.columns for col in ['team1_score', 'team2_score']):
            return {"error": "Score data not available"}
        
        # Calculate score differential for each match
        df['score_differential'] = abs(df['team1_score'] - df['team2_score'])
        
        # Group by prediction correctness
        score_by_correctness = df.groupby('prediction_correct').agg(
            avg_differential=('score_differential', 'mean'),
            min_differential=('score_differential', 'min'),
            max_differential=('score_differential', 'max'),
            count=('match_id', 'count')
        )
        
        # Create bins for score differential
        df['diff_bin'] = pd.cut(
            df['score_differential'],
            bins=[0, 3, 6, 9, 12, float('inf')],
            labels=['0-3', '4-6', '7-9', '10-12', '13+']
        )
        
        # Calculate accuracy by score differential bin
        diff_performance = df.groupby('diff_bin').agg(
            total=('match_id', 'count'),
            correct=('prediction_correct', 'sum')
        )
        diff_performance['accuracy'] = diff_performance['correct'] / diff_performance['total']
        
        return {
            'by_correctness': score_by_correctness.to_dict(),
            'by_differential': diff_performance.to_dict()
        }
    
    def generate_reports(self, period=None, model_version=None):
        """
        Generate comprehensive performance reports.
        
        Args:
            period: Optional time period (e.g., '1m', '3m', '6m', '1y')
            model_version: Optional model version to filter by
            
        Returns:
            str: Path to the generated report
        """
        # Calculate metrics
        metrics = self.calculate_metrics(period, model_version)
        
        if 'error' in metrics:
            return {"error": metrics['error']}
        
        # Create timestamp for report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        period_str = period or 'all_time'
        model_str = model_version or 'all_models'
        report_name = f"performance_report_{period_str}_{model_str}_{timestamp}"
        
        # Create report directory
        report_dir = os.path.join(self.output_dir, report_name)
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate visualizations
        self._generate_visualizations(metrics, report_dir)
        
        # Save metrics to JSON
        report_file = os.path.join(report_dir, "metrics.json")
        with open(report_file, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        
        # Generate HTML report
        html_report = self._generate_html_report(metrics, report_dir)
        
        return {
            "report_directory": report_dir,
            "metrics_file": report_file,
            "html_report": html_report
        }
    
    def _generate_visualizations(self, metrics, report_dir):
        """
        Generate visualizations for the performance report.
        
        Args:
            metrics: Dictionary of performance metrics
            report_dir: Directory to save visualizations
        """
        # Filter data
        df = self._filter_data(self.history_df, metrics['time_period'], metrics['model_version'])
        
        # Plot accuracy over time
        plt.figure(figsize=(12, 6))
        df['month'] = df['match_date'].dt.strftime('%Y-%m')
        monthly_acc = df.groupby('month').agg(
            total=('match_id', 'count'),
            correct=('prediction_correct', 'sum')
        )
        monthly_acc['accuracy'] = monthly_acc['correct'] / monthly_acc['total']
        
        plt.plot(monthly_acc.index, monthly_acc['accuracy'], 'o-', linewidth=2)
        plt.axhline(y=metrics['accuracy'], color='r', linestyle='--', alpha=0.7, 
                    label=f"Overall Accuracy: {metrics['accuracy']:.2%}")
        plt.title('Prediction Accuracy Over Time')
        plt.xlabel('Month')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "accuracy_over_time.png"))
        plt.close()
        
        # Plot calibration curve
        plt.figure(figsize=(10, 8))
        calibration_data = pd.DataFrame(metrics['calibration']['bin_data'])
        
        # Add perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Plot actual calibration
        plt.scatter(calibration_data['prob_bin'], calibration_data['actual_win_rate'], 
                   s=calibration_data['count']*5, alpha=0.7, c='blue')
        
        # Connect dots
        plt.plot(calibration_data['prob_bin'], calibration_data['actual_win_rate'], 'b-', alpha=0.5)
        
        # Add count labels
        for _, row in calibration_data.iterrows():
            if not np.isnan(row['actual_win_rate']) and row['count'] > 0:
                plt.annotate(f"{row['count']}", 
                            (row['prob_bin'], row['actual_win_rate']),
                            textcoords="offset points",
                            xytext=(0,10),
                            ha='center')
        
        plt.title('Calibration Curve (Predicted vs Actual Win Rate)')
        plt.xlabel('Predicted Win Probability')
        plt.ylabel('Actual Win Rate')
        plt.grid(alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(os.path.join(report_dir, "calibration_curve.png"))
        plt.close()
        
        # Plot confidence distribution
        plt.figure(figsize=(10, 6))
        df['confidence_level'] = pd.cut(
            df['win_probability'], 
            bins=[0, 0.55, 0.65, 0.75, 0.85, 1.0],
            labels=['Very Low (≤55%)', 'Low (56-65%)', 'Medium (66-75%)', 'High (76-85%)', 'Very High (>85%)']
        )
        
        confidence_results = df.groupby('confidence_level').agg(
            total=('match_id', 'count'),
            correct=('prediction_correct', 'sum')
        )
        confidence_results['accuracy'] = confidence_results['correct'] / confidence_results['total']
        
        # Create bar chart for accuracy by confidence level
        ax = confidence_results['accuracy'].plot(kind='bar', figsize=(10, 6), color='skyblue')
        
        # Add count labels
        for i, (_, row) in enumerate(confidence_results.iterrows()):
            if not np.isnan(row['accuracy']):
                plt.text(i, row['accuracy'] + 0.02, f"{row['total']}", ha='center')
        
        plt.title('Prediction Accuracy by Confidence Level')
        plt.xlabel('Confidence Level')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "accuracy_by_confidence.png"))
        plt.close()
        
        # Plot feature importance
        if metrics['feature_importance'] and metrics['feature_importance']['overall_importance']:
            plt.figure(figsize=(12, 8))
            importance_df = pd.DataFrame(list(metrics['feature_importance']['overall_importance'].items()), 
                                        columns=['Feature', 'Importance'])
            importance_df = importance_df.sort_values('Importance', ascending=True)
            
            plt.barh(importance_df['Feature'].values, importance_df['Importance'].values)
            plt.title('Most Important Features')
            plt.xlabel('Average Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "feature_importance.png"))
            plt.close()
            
            # Plot misleading features
            if metrics['feature_importance']['misleading_features']:
                plt.figure(figsize=(12, 8))
                misleading_df = pd.DataFrame(list(metrics['feature_importance']['misleading_features'].items()), 
                                           columns=['Feature', 'Error_Ratio'])
                misleading_df = misleading_df.sort_values('Error_Ratio', ascending=True)
                
                plt.barh(misleading_df['Feature'].values, misleading_df['Error_Ratio'].values, color='red')
                plt.title('Potentially Misleading Features')
                plt.xlabel('Error Importance Ratio')
                plt.ylabel('Feature')
                plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(report_dir, "misleading_features.png"))
                plt.close()
        
        # Plot map performance if available
        if 'map_analysis' in metrics and 'error' not in metrics['map_analysis']:
            map_data = pd.DataFrame({
                'Accuracy': metrics['map_analysis']['accuracy'],
                'Count': metrics['map_analysis']['total']
            })
            
            plt.figure(figsize=(12, 6))
            ax = map_data['Accuracy'].plot(kind='bar', figsize=(12, 6), color='lightgreen')
            
            # Add count labels
            for i, (_, row) in enumerate(map_data.iterrows()):
                plt.text(i, row['Accuracy'] + 0.02, f"{row['Count']}", ha='center')
            
            plt.title('Prediction Accuracy by Map')
            plt.xlabel('Map')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.1)
            plt.grid(alpha=0.3, axis='y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "accuracy_by_map.png"))
            plt.close()
        
        # Plot score differential analysis if available
        if 'score_analysis' in metrics and 'error' not in metrics['score_analysis']:
            if 'by_differential' in metrics['score_analysis']:
                diff_data = pd.DataFrame({
                    'Accuracy': metrics['score_analysis']['by_differential']['accuracy'],
                    'Count': metrics['score_analysis']['by_differential']['total']
                })
                
                plt.figure(figsize=(12, 6))
                ax = diff_data['Accuracy'].plot(kind='bar', figsize=(12, 6), color='orange')
                
                # Add count labels
                for i, (_, row) in enumerate(diff_data.iterrows()):
                    if not np.isnan(row['Accuracy']):
                        plt.text(i, row['Accuracy'] + 0.02, f"{row['Count']}", ha='center')
                
                plt.title('Prediction Accuracy by Score Differential')
                plt.xlabel('Score Differential')
                plt.ylabel('Accuracy')
                plt.ylim(0, 1.1)
                plt.grid(alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(report_dir, "accuracy_by_score_diff.png"))
                plt.close()
    
    def _generate_html_report(self, metrics, report_dir):
        """
        Generate an HTML report from the metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            report_dir: Directory to save the report
            
        Returns:
            str: Path to the HTML report
        """
        # Basic HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Model Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric-card {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .metric-title {{ font-size: 14px; color: #7f8c8d; text-transform: uppercase; }}
                .good {{ color: #27ae60; }}
                .bad {{ color: #e74c3c; }}
                .warning {{ color: #f39c12; }}
                .flex-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .flex-card {{ flex: 1; min-width: 200px; }}
                .img-container {{ text-align: center; margin: 20px 0; }}
                img {{ max-width: 100%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Prediction Model Performance Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Period: {metrics['time_period']}</p>
            <p>Model Version: {metrics['model_version']}</p>
            
            <h2>Overview</h2>
            <div class="flex-container">
                <div class="flex-card metric-card">
                    <div class="metric-title">Accuracy</div>
                    <div class="metric-value {'good' if metrics['accuracy'] >= 0.7 else 'warning' if metrics['accuracy'] >= 0.5 else 'bad'}">
                        {metrics['accuracy']:.2%}
                    </div>
                </div>
                <div class="flex-card metric-card">
                    <div class="metric-title">Precision</div>
                    <div class="metric-value {'good' if metrics['precision'] >= 0.7 else 'warning' if metrics['precision'] >= 0.5 else 'bad'}">
                        {metrics['precision']:.2%}
                    </div>
                </div>
                <div class="flex-card metric-card">
                    <div class="metric-title">Recall</div>
                    <div class="metric-value {'good' if metrics['recall'] >= 0.7 else 'warning' if metrics['recall'] >= 0.5 else 'bad'}">
                        {metrics['recall']:.2%}
                    </div>
                </div>
                <div class="flex-card metric-card">
                    <div class="metric-title">F1 Score</div>
                    <div class="metric-value {'good' if metrics['f1_score'] >= 0.7 else 'warning' if metrics['f1_score'] >= 0.5 else 'bad'}">
                        {metrics['f1_score']:.2%}
                    </div>
                </div>
                <div class="flex-card metric-card">
                    <div class="metric-title">Total Predictions</div>
                    <div class="metric-value">
                        {metrics['total_predictions']}
                    </div>
                </div>
                <div class="flex-card metric-card">
                    <div class="metric-title">Correct Predictions</div>
                    <div class="metric-value {'good' if metrics['accuracy'] >= 0.7 else 'warning' if metrics['accuracy'] >= 0.5 else 'bad'}">
                        {metrics['correct_predictions']} ({metrics['accuracy']:.2%})
                    </div>
                </div>
            </div>
            
            <h2>Visualizations</h2>
            
            <div class="flex-container">
                <div class="flex-card">
                    <h3>Accuracy Over Time</h3>
                    <div class="img-container">
                        <img src="accuracy_over_time.png" alt="Accuracy Over Time">
                    </div>
                </div>
                <div class="flex-card">
                    <h3>Calibration Curve</h3>
                    <div class="img-container">
                        <img src="calibration_curve.png" alt="Calibration Curve">
                    </div>
                </div>
            </div>
            
            <div class="flex-container">
                <div class="flex-card">
                    <h3>Accuracy by Confidence Level</h3>
                    <div class="img-container">
                        <img src="accuracy_by_confidence.png" alt="Accuracy by Confidence Level">
                    </div>
                </div>
                <div class="flex-card">
                    <h3>Feature Importance</h3>
                    <div class="img-container">
                        <img src="feature_importance.png" alt="Feature Importance">
                    </div>
                </div>
            </div>
            
            {f'''
            <div class="flex-container">
                <div class="flex-card">
                    <h3>Misleading Features</h3>
                    <div class="img-container">
                        <img src="misleading_features.png" alt="Misleading Features">
                    </div>
                </div>
                <div class="flex-card">
                    <h3>Accuracy by Map</h3>
                    <div class="img-container">
                        <img src="accuracy_by_map.png" alt="Accuracy by Map">
                    </div>
                </div>
            </div>
            ''' if 'map_analysis' in metrics and 'error' not in metrics['map_analysis'] and metrics['feature_importance']['misleading_features'] else ''}
            
            {f'''
            <div class="flex-container">
                <div class="flex-card">
                    <h3>Accuracy by Score Differential</h3>
                    <div class="img-container">
                        <img src="accuracy_by_score_diff.png" alt="Accuracy by Score Differential">
                    </div>
                </div>
            </div>
            ''' if 'score_analysis' in metrics and 'error' not in metrics['score_analysis'] else ''}
            
            <h2>Error Analysis</h2>
            
            <div class="metric-card">
                <h3>Error Patterns</h3>
                <p>Total errors: {metrics['error_patterns']['total_errors']} ({metrics['error_patterns']['error_rate']:.2%} error rate)</p>
                
                <h4>Errors by Confidence Level</h4>
                <table>
                    <tr>
                        <th>Confidence Level</th>
                        <th>Error Count</th>
                    </tr>
                    {''''.join(f'''
                    <tr>
                        <td>{level}</td>
                        <td>{count}</td>
                    </tr>
                    ''' for level, count in metrics['error_patterns'].get('confidence_distribution', {}).items())}
                </table>
                
                <h4>Most Problematic Teams</h4>
                <table>
                    <tr>
                        <th>Team</th>
                        <th>Error Count</th>
                    </tr>
                    {''''.join(f'''
                    <tr>
                        <td>{team.replace('_upset', ' (upset)')}</td>
                        <td>{count}</td>
                    </tr>
                    ''' for team, count in metrics['error_patterns'].get('problematic_teams', {}).items())}
                </table>
                
                {f'''
                <h4>Potentially Misleading Features</h4>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Importance in Errors</th>
                    </tr>
                    {''''.join(f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{importance:.4f}</td>
                    </tr>
                    """ for feature, importance in metrics['error_patterns'].get('misleading_features', {}).items())}
                </table>
                ''' if metrics['error_patterns'].get('misleading_features') else ''}
            </div>
            
            <h2>Calibration Analysis</h2>
            
            <div class="metric-card">
                <h3>Model Calibration</h3>
                <p>Mean Calibration Error: {metrics['calibration']['mean_calibration_error']:.4f if metrics['calibration']['mean_calibration_error'] is not None else 'N/A'}</p>
                
                <table>
                    <tr>
                        <th>Predicted Probability</th>
                        <th>Actual Win Rate</th>
                        <th>Sample Count</th>
                        <th>Calibration Error</th>
                    </tr>
                    {''''.join(f'''
                    <tr>
                        <td>{bin_data['prob_bin']:.2f}</td>
                        <td>{bin_data.get('actual_win_rate', 'N/A')}</td>
                        <td>{bin_data['count']}</td>
                        <td>{bin_data.get('error', 'N/A')}</td>
                    </tr>
                    ''' for bin_data in metrics['calibration']['bin_data'] if bin_data['count'] > 0)}
                </table>
            </div>
            
            {f'''
            <h2>Map Analysis</h2>
            
            <div class="metric-card">
                <h3>Prediction Accuracy by Map</h3>
                <table>
                    <tr>
                        <th>Map</th>
                        <th>Accuracy</th>
                        <th>Sample Count</th>
                    </tr>
                    {''''.join(f"""
                    <tr>
                        <td>{map_name}</td>
                        <td>{metrics['map_analysis']['accuracy'][map_name]:.2%}</td>
                        <td>{metrics['map_analysis']['total'][map_name]}</td>
                    </tr>
                    """ for map_name in metrics['map_analysis']['accuracy'])}
                </table>
            </div>
            ''' if 'map_analysis' in metrics and 'error' not in metrics['map_analysis'] else ''}
            
            {f'''
            <h2>Score Analysis</h2>
            
            <div class="metric-card">
                <h3>Score Differential Analysis</h3>
                
                <h4>By Prediction Correctness</h4>
                <table>
                    <tr>
                        <th>Prediction Result</th>
                        <th>Avg Score Differential</th>
                        <th>Min Score Differential</th>
                        <th>Max Score Differential</th>
                        <th>Count</th>
                    </tr>
                    <tr>
                        <td>Correct Predictions</td>
                        <td>{metrics['score_analysis']['by_correctness']['avg_differential'].get(True, 'N/A'):.2f}</td>
                        <td>{metrics['score_analysis']['by_correctness']['min_differential'].get(True, 'N/A')}</td>
                        <td>{metrics['score_analysis']['by_correctness']['max_differential'].get(True, 'N/A')}</td>
                        <td>{metrics['score_analysis']['by_correctness']['count'].get(True, 0)}</td>
                    </tr>
                    <tr>
                        <td>Incorrect Predictions</td>
                        <td>{metrics['score_analysis']['by_correctness']['avg_differential'].get(False, 'N/A'):.2f}</td>
                        <td>{metrics['score_analysis']['by_correctness']['min_differential'].get(False, 'N/A')}</td>
                        <td>{metrics['score_analysis']['by_correctness']['max_differential'].get(False, 'N/A')}</td>
                        <td>{metrics['score_analysis']['by_correctness']['count'].get(False, 0)}</td>
                    </tr>
                </table>
                
                <h4>Accuracy by Score Differential</h4>
                <table>
                    <tr>
                        <th>Score Differential</th>
                        <th>Accuracy</th>
                        <th>Sample Count</th>
                    </tr>
                    {''''.join(f"""
                    <tr>
                        <td>{diff_bin}</td>
                        <td>{metrics['score_analysis']['by_differential']['accuracy'][diff_bin]:.2%}</td>
                        <td>{metrics['score_analysis']['by_differential']['total'][diff_bin]}</td>
                    </tr>
                    """ for diff_bin in metrics['score_analysis']['by_differential']['accuracy'])}
                </table>
            </div>
            ''' if 'score_analysis' in metrics and 'error' not in metrics['score_analysis'] else ''}
            
            <h2>Feature Importance Analysis</h2>
            
            <div class="metric-card">
                <h3>Most Important Features</h3>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
                    {''''.join(f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{importance:.4f}</td>
                    </tr>
                    """ for feature, importance in metrics['feature_importance']['overall_importance'].items())}
                </table>
                
                {f'''
                <h3>Potentially Misleading Features</h3>
                <p>These features may be contributing to incorrect predictions.</p>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Error Importance Ratio</th>
                    </tr>
                    {''''.join(f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{ratio:.2f}</td>
                    </tr>
                    """ for feature, ratio in metrics['feature_importance']['misleading_features'].items())}
                </table>
                ''' if metrics['feature_importance']['misleading_features'] else ''}
            </div>
            
            <div class="metric-card">
                <h3>Recommendations for Model Improvement</h3>
                <ul>
                    {f"<li>The model is {'well' if metrics['calibration']['mean_calibration_error'] and metrics['calibration']['mean_calibration_error'] < 0.1 else 'poorly'} calibrated. {'Consider recalibrating the model to improve reliability.' if metrics['calibration']['mean_calibration_error'] and metrics['calibration']['mean_calibration_error'] >= 0.1 else ''}</li>" if metrics['calibration']['mean_calibration_error'] is not None else ""}
                    
                    {f"<li>The model performs {'well' if metrics['accuracy'] >= 0.7 else 'adequately' if metrics['accuracy'] >= 0.5 else 'poorly'} overall with an accuracy of {metrics['accuracy']:.2%}.</li>"}
                    
                    {f"<li>Consider reviewing feature importance and possibly downweighting misleading features.</li>" if metrics['feature_importance'].get('misleading_features') else ""}
                    
                    {f"<li>Review predictions for these problematic teams: {', '.join(team.replace('_upset', ' (upset)') for team in list(metrics['error_patterns']['problematic_teams'].keys())[:3])}.</li>" if metrics['error_patterns'].get('problematic_teams') else ""}
                    
                    {f"<li>The model is overconfident in the {max(metrics['error_patterns'].get('confidence_distribution', {}).items(), key=lambda x: x[1])[0]} confidence range.</li>" if metrics['error_patterns'].get('confidence_distribution') else ""}
                    
                    {f"<li>Consider weighting more recent matches more heavily during training.</li>" if 'month' in locals() and monthly_acc.index.size > 1 else ""}
                    
                    {f"<li>The model performs best on maps: {', '.join(list(metrics['map_analysis']['accuracy'].keys())[:2])} but struggles with {', '.join(list(metrics['map_analysis']['accuracy'].keys())[-2:])}. Consider creating map-specific features or training separate models per map.</li>" if 'map_analysis' in metrics and 'error' not in metrics['map_analysis'] and len(metrics['map_analysis']['accuracy']) >= 4 else ""}
                    
                    {f"<li>Consider adding more score-related features, as score differential seems to be a good predictor of model accuracy.</li>" if 'score_analysis' in metrics and 'error' not in metrics['score_analysis'] else ""}
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        html_file = os.path.join(report_dir, "report.html")
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        return html_file

def main():
    """Command line interface for the performance analyzer."""
    parser = argparse.ArgumentParser(description='Analyze prediction model performance')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Calculate metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Calculate performance metrics')
    metrics_parser.add_argument('--period', choices=['1m', '3m', '6m', '1y'], help='Time period to analyze')
    metrics_parser.add_argument('--model_version', help='Model version to analyze')
    metrics_parser.add_argument('--output', help='Output JSON file for metrics')
    
    # Generate report command
    report_parser = subparsers.add_parser('report', help='Generate performance report')
    report_parser.add_argument('--period', choices=['1m', '3m', '6m', '1y'], help='Time period to analyze')
    report_parser.add_argument('--model_version', help='Model version to analyze')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze specific aspects of performance')
    analyze_parser.add_argument('--aspect', choices=['errors', 'features', 'maps', 'scores', 'calibration'], 
                               required=True, help='Aspect to analyze')
    analyze_parser.add_argument('--period', choices=['1m', '3m', '6m', '1y'], help='Time period to analyze')
    analyze_parser.add_argument('--model_version', help='Model version to analyze')
    analyze_parser.add_argument('--output', help='Output JSON file for analysis')
    
    args = parser.parse_args()
    
    try:
        analyzer = PerformanceAnalyzer()
        
        if args.command == 'metrics':
            metrics = analyzer.calculate_metrics(args.period, args.model_version)
            
            if 'error' in metrics:
                print(f"Error: {metrics['error']}")
                return
            
            print("\n===== Performance Metrics =====")
            print(f"Period: {metrics['time_period']}")
            print(f"Model Version: {metrics['model_version']}")
            print(f"Total Predictions: {metrics['total_predictions']}")
            print(f"Correct Predictions: {metrics['correct_predictions']}")
            print(f"Accuracy: {metrics['accuracy']:.2%}")
            print(f"Precision: {metrics['precision']:.2%}")
            print(f"Recall: {metrics['recall']:.2%}")
            print(f"F1 Score: {metrics['f1_score']:.2%}")
            
            if args.output:
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                with open(args.output, 'w') as f:
                    json.dump(metrics, f, indent=4, default=str)
                print(f"\nMetrics saved to {args.output}")
        
        elif args.command == 'report':
            result = analyzer.generate_reports(args.period, args.model_version)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                return
            
            print("\n===== Report Generated =====")
            print(f"Report Directory: {result['report_directory']}")
            print(f"Metrics File: {result['metrics_file']}")
            print(f"HTML Report: {result['html_report']}")
            print("\nOpen the HTML report in a web browser to view the complete analysis.")
        
        elif args.command == 'analyze':
            metrics = analyzer.calculate_metrics(args.period, args.model_version)
            
            if 'error' in metrics:
                print(f"Error: {metrics['error']}")
                return
            
            if args.aspect == 'errors':
                print("\n===== Error Analysis =====")
                print(f"Total errors: {metrics['error_patterns']['total_errors']} ({metrics['error_patterns']['error_rate']:.2%} error rate)")
                
                print("\nErrors by confidence level:")
                for level, count in metrics['error_patterns'].get('confidence_distribution', {}).items():
                    print(f"  {level}: {count}")
                
                print("\nMost problematic teams:")
                for team, count in metrics['error_patterns'].get('problematic_teams', {}).items():
                    print(f"  {team.replace('_upset', ' (upset)')}: {count}")
                
                if metrics['error_patterns'].get('misleading_features'):
                    print("\nPotentially misleading features:")
                    for feature, importance in metrics['error_patterns'].get('misleading_features', {}).items():
                        print(f"  {feature}: {importance:.4f}")
            
            elif args.aspect == 'features':
                print("\n===== Feature Importance Analysis =====")
                print("Most important features:")
                for feature, importance in metrics['feature_importance']['overall_importance'].items():
                    print(f"  {feature}: {importance:.4f}")
                
                if metrics['feature_importance']['misleading_features']:
                    print("\nPotentially misleading features:")
                    for feature, ratio in metrics['feature_importance']['misleading_features'].items():
                        print(f"  {feature}: {ratio:.2f}")
            
            elif args.aspect == 'maps' and 'map_analysis' in metrics and 'error' not in metrics['map_analysis']:
                print("\n===== Map Performance Analysis =====")
                print("Prediction accuracy by map:")
                for map_name in metrics['map_analysis']['accuracy']:
                    accuracy = metrics['map_analysis']['accuracy'][map_name]
                    total = metrics['map_analysis']['total'][map_name]
                    print(f"  {map_name}: {accuracy:.2%} ({total} matches)")
            
            elif args.aspect == 'scores' and 'score_analysis' in metrics and 'error' not in metrics['score_analysis']:
                print("\n===== Score Analysis =====")
                print("Score differential by prediction correctness:")
                for correct, avg_diff in metrics['score_analysis']['by_correctness']['avg_differential'].items():
                    count = metrics['score_analysis']['by_correctness']['count'][correct]
                    result = "Correct" if correct else "Incorrect"
                    print(f"  {result} predictions (n={count}): {avg_diff:.2f} average score differential")
                
                print("\nAccuracy by score differential:")
                for diff_bin in metrics['score_analysis']['by_differential']['accuracy']:
                    accuracy = metrics['score_analysis']['by_differential']['accuracy'][diff_bin]
                    total = metrics['score_analysis']['by_differential']['total'][diff_bin]
                    print(f"  {diff_bin}: {accuracy:.2%} ({total} matches)")
            
            elif args.aspect == 'calibration':
                print("\n===== Calibration Analysis =====")
                print(f"Mean Calibration Error: {metrics['calibration']['mean_calibration_error']:.4f if metrics['calibration']['mean_calibration_error'] is not None else 'N/A'}")
                
                print("\nCalibration by probability bin:")
                for bin_data in metrics['calibration']['bin_data']:
                    if bin_data['count'] > 0:
                        print(f"  Predicted: {bin_data['prob_bin']:.2f}, Actual: {bin_data.get('actual_win_rate', 'N/A'):.2f}, Count: {bin_data['count']}, Error: {bin_data.get('error', 'N/A'):.2f}")
            
            if args.output:
                output_data = metrics.get(args.aspect, metrics)
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=4, default=str)
                print(f"\nAnalysis saved to {args.output}")
        
        else:
            parser.print_help()
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()