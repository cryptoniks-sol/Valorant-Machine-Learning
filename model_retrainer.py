import os
import json
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import time
import requests

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

# API URL (same as in the main script)
API_URL = "http://localhost:5000/api/v1"

class ModelRetrainer:
    """
    Retrains the prediction model using past match data and prediction results.
    Incorporates time-based weighting to prioritize recent matches.
    """
    def __init__(self, history_file='prediction_results/prediction_history.csv', 
                 results_dir='prediction_results', models_dir='models',
                 api_url=API_URL):
        """
        Initialize the model retrainer.
        
        Args:
            history_file: CSV file with prediction history
            results_dir: Directory where prediction results are stored
            models_dir: Directory where models will be stored
            api_url: URL for the Valorant API
        """
        self.history_file = history_file
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.api_url = api_url
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load prediction history
        if not os.path.exists(self.history_file):
            raise FileNotFoundError(f"History file not found: {self.history_file}")
        
        self.history_df = pd.read_csv(self.history_file)
        
        # Convert dates to datetime
        self.history_df['match_date'] = pd.to_datetime(self.history_df['match_date'])
    
    def get_team_id(self, team_name, region=None):
        """
        Search for a team ID by name, optionally filtering by region.
        
        Args:
            team_name (str): The name of the team to search for
            region (str, optional): Region code like 'na', 'eu', 'kr', etc.
        
        Returns:
            str: Team ID if found, None otherwise
        """
        print(f"Searching for team ID for '{team_name}'...")
        
        # Build the API URL with optional region filter
        url = f"{self.api_url}/teams?limit=300"
        if region:
            url += f"&region={region}"
            print(f"Filtering by region: {region}")
        
        # Fetch teams
        try:
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"Error fetching teams: {response.status_code}")
                return None
            
            teams_data = response.json()
            
            if 'data' not in teams_data:
                print("No 'data' field found in the response")
                return None
            
            # Try to find an exact match first
            for team in teams_data['data']:
                if team['name'].lower() == team_name.lower():
                    print(f"Found exact match: {team['name']} (ID: {team['id']})")
                    return team['id']
            
            # If no exact match, try partial match
            for team in teams_data['data']:
                if team_name.lower() in team['name'].lower() or team['name'].lower() in team_name.lower():
                    print(f"Found partial match: {team['name']} (ID: {team['id']})")
                    return team['id']
        
        except Exception as e:
            print(f"Error searching for team ID: {e}")
        
        print(f"No team ID found for '{team_name}'")
        return None
    
    def collect_training_data(self, weighting_scheme='recency', recency_days=90, include_api_data=True):
        """
        Collect and prepare training data from prediction history and match results.
        
        Args:
            weighting_scheme: How to weight training samples ('recency', 'prediction_error', 'both', or 'none')
            recency_days: Number of days to consider as "recent" for weighting
            include_api_data: Whether to fetch additional data from the API
            
        Returns:
            pd.DataFrame: Training data with features and target
        """
        print("Collecting training data...")
        
        # Get result files with match data
        result_files = [f for f in os.listdir(self.results_dir) if f.endswith('_result.json')]
        
        if not result_files:
            raise ValueError("No result files found. Need match data for training.")
        
        # Collect training samples
        training_data = []
        
        for result_file in result_files:
            try:
                with open(os.path.join(self.results_dir, result_file), 'r') as f:
                    match_data = json.load(f)
                
                # Skip if missing essential data
                if 'match' not in match_data or 'actual_winner' not in match_data:
                    continue
                
                # Extract teams
                match_str = match_data.get('match', '')
                if not match_str or ' vs ' not in match_str:
                    continue
                    
                team1_name, team2_name = match_str.split(' vs ')
                
                # Get prediction result
                predicted_winner = match_data.get('predicted_winner', '')
                actual_winner = match_data.get('actual_winner', '')
                prediction_correct = match_data.get('prediction_correct', False)
                
                # Calculate features
                if 'feature_importance' in match_data:
                    # Get features that were used for this prediction
                    features = {}
                    
                    # Extract feature values from team stats or betting indicators if available
                    if 'team1_stats' in match_data and 'team2_stats' in match_data:
                        # Basic team stats
                        team1_stats = match_data['team1_stats']
                        team2_stats = match_data['team2_stats']
                        
                        features.update({
                            'team1_win_rate': team1_stats.get('win_rate', 0.5),
                            'team2_win_rate': team2_stats.get('win_rate', 0.5),
                            'team1_score_diff': team1_stats.get('score_differential', 0),
                            'team2_score_diff': team2_stats.get('score_differential', 0),
                            'team1_recent_form': team1_stats.get('recent_form', 0.5),
                            'team2_recent_form': team2_stats.get('recent_form', 0.5),
                            'win_rate_diff': team1_stats.get('win_rate', 0.5) - team2_stats.get('win_rate', 0.5),
                            'ranking_diff': team1_stats.get('ranking', 50) - team2_stats.get('ranking', 50),
                            'rating_diff': team1_stats.get('rating', 1500) - team2_stats.get('rating', 1500),
                        })
                        
                        # H2H stats if available
                        if 'h2h_stats' in match_data:
                            h2h_stats = match_data['h2h_stats']
                            features.update({
                                'h2h_matches': h2h_stats.get('total_h2h_matches', 0),
                                'team1_h2h_win_rate': h2h_stats.get('team1_h2h_win_rate', 0.5),
                                'team1_h2h_wins': h2h_stats.get('team1_h2h_wins', 0)
                            })
                        
                        # Performance trends if available
                        if 'performance_trends' in match_data:
                            trends = match_data['performance_trends']
                            if 'team1' in trends and 'team2' in trends:
                                team1_trends = trends['team1']
                                team2_trends = trends['team2']
                                
                                features.update({
                                    'team1_recent_form_5': team1_trends.get('last_5_win_rate', 0.5),
                                    'team2_recent_form_5': team2_trends.get('last_5_win_rate', 0.5),
                                    'team1_form_trajectory': team1_trends.get('5_vs_10', 0),
                                    'team2_form_trajectory': team2_trends.get('5_vs_10', 0),
                                })
                    
                    # Add match performance data if available (from actual match result)
                    if 'match_performance' in match_data:
                        perf_data = match_data['match_performance']
                        valid_perf_data = {k: v for k, v in perf_data.items() if v is not None}
                        features.update(valid_perf_data)
                    
                    # Add actual features from the original prediction
                    for feature_name, importance in match_data['feature_importance'].items():
                        if feature_name not in features:
                            if isinstance(importance, dict) and 'Feature' in importance:
                                feature_value = importance.get('Feature', 0)
                            else:
                                # Use a default value since we don't have the actual value
                                feature_value = 0
                            features[feature_name] = feature_value
                    
                    # Determine target (1 = team1 wins, 0 = team2 wins)
                    target = 1 if actual_winner == team1_name else 0
                    
                    # Calculate sample weight
                    weight = 1.0
                    
                    if weighting_scheme != 'none':
                        # Recency weighting: More recent matches get higher weight
                        if weighting_scheme in ['recency', 'both']:
                            match_date = datetime.strptime(match_data.get('match_date', ''), "%Y-%m-%d") if isinstance(match_data.get('match_date', ''), str) else datetime.now()
                            days_ago = (datetime.now() - match_date).days
                            
                            if days_ago <= recency_days:
                                # Exponential decay weighting: w = exp(-days_ago/half_life)
                                half_life = recency_days / 2
                                recency_weight = np.exp(-days_ago / half_life)
                                weight *= recency_weight
                            else:
                                weight *= 0.5  # Older matches get half weight
                        
                        # Error-based weighting: Incorrect predictions influence model more
                        if weighting_scheme in ['prediction_error', 'both'] and not prediction_correct:
                            weight *= 1.5  # Errors get 50% more weight
                    
                    # Add sample with its weight
                    features['sample_weight'] = weight
                    features['target'] = target
                    features['team1'] = team1_name
                    features['team2'] = team2_name
                    features['match_date'] = match_data.get('match_date', '')
                    
                    training_data.append(features)
            
            except Exception as e:
                print(f"Error processing {result_file}: {e}")
        
        if not training_data:
            raise ValueError("No valid training data found in result files.")
        
        # Convert to DataFrame
        training_df = pd.DataFrame(training_data)
        
        # Fill missing values
        training_df = training_df.fillna(0)
        
        # Fetch additional data from API if requested
        if include_api_data:
            # Get unique teams
            all_teams = set(training_df['team1'].unique()).union(set(training_df['team2'].unique()))
            
            # Enrich with API data
            self._enrich_with_api_data(training_df, all_teams)
        
        return training_df
    
    def _enrich_with_api_data(self, training_df, teams):
        """
        Enrich training data with additional information from the API.
        
        Args:
            training_df: Training DataFrame to enrich
            teams: Set of team names to fetch data for
        """
        print("Enriching training data with API data...")
        
        team_data = {}
        
        for team_name in teams:
            team_id = self.get_team_id(team_name)
            
            if team_id:
                # Fetch team details
                try:
                    response = requests.get(f"{self.api_url}/teams/{team_id}")
                    
                    if response.status_code == 200:
                        team_details = response.json()
                        
                        # Extract useful data
                        team_data[team_name] = {
                            'ranking': team_details.get('ranking', 9999),
                            'rating': team_details.get('rating', 1500),
                            'region': team_details.get('region', ''),
                            'country': team_details.get('country', '')
                        }
                    
                    # Be nice to the API
                    time.sleep(0.5)
                
                except Exception as e:
                    print(f"Error fetching team details for {team_name}: {e}")
        
        # Add team data to training DataFrame
        for idx, row in training_df.iterrows():
            team1_name = row['team1']
            team2_name = row['team2']
            
            if team1_name in team_data:
                for key, value in team_data[team1_name].items():
                    training_df.at[idx, f'team1_{key}'] = value
            
            if team2_name in team_data:
                for key, value in team_data[team2_name].items():
                    training_df.at[idx, f'team2_{key}'] = value
        
        # Calculate additional features
        if 'team1_ranking' in training_df.columns and 'team2_ranking' in training_df.columns:
            training_df['ranking_diff'] = training_df['team1_ranking'] - training_df['team2_ranking']
        
        if 'team1_rating' in training_df.columns and 'team2_rating' in training_df.columns:
            training_df['rating_diff'] = training_df['team1_rating'] - training_df['team2_rating']
    
    def train_model(self, training_df, feature_selection=True, hyperparameter_tuning=True, 
                    model_type='ensemble', use_smote=True, model_name=None):
        """
        Train a new prediction model with the collected training data.
        
        Args:
            training_df: DataFrame with training data
            feature_selection: Whether to perform feature selection
            hyperparameter_tuning: Whether to tune hyperparameters
            model_type: Type of model to train ('rf', 'gbm', or 'ensemble')
            use_smote: Whether to use SMOTE for class imbalance
            model_name: Optional name for the model
            
        Returns:
            dict: Training results and model info
        """
        print("Training new prediction model...")
        
        if 'target' not in training_df.columns:
            raise ValueError("Training data must include 'target' column.")
        
        # Extract features and target
        X = training_df.drop(['target', 'team1', 'team2', 'match_date', 'sample_weight'], axis=1, errors='ignore')
        y = training_df['target']
        
        # Get sample weights if available
        sample_weights = None
        if 'sample_weight' in training_df.columns:
            sample_weights = training_df['sample_weight'].values
            X = X.drop('sample_weight', axis=1, errors='ignore')
        
        # Print class distribution
        class_counts = np.bincount(y)
        print(f"Class distribution - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Feature selection
        if feature_selection:
            print("Performing feature selection...")
            selector = SelectFromModel(RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), 
                                     threshold='median')
            X_selected = selector.fit_transform(X_scaled, y, sample_weight=sample_weights)
            selected_indices = selector.get_support()
            selected_features = [X.columns[i] for i, selected in enumerate(selected_indices) if selected]
            print(f"Selected {len(selected_features)} features: {selected_features}")
        else:
            X_selected = X_scaled
            selector = None
            selected_features = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test, sw_train, sw_test = self._split_data(X_selected, y, sample_weights)
        
        # Handle class imbalance
        if use_smote and len(class_counts) > 1 and min(class_counts) / max(class_counts) < 0.3:
            print("Applying SMOTE to handle class imbalance...")
            try:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                sw_train = np.ones(len(y_train))  # Reset weights after SMOTE
                print(f"After SMOTE - Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)}")
            except Exception as e:
                print(f"Error applying SMOTE: {e}. Using class weights instead.")
                use_smote = False
        
        # Create cross-validation strategy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Base model based on type
        if model_type == 'rf':
            base_model = RandomForestClassifier(class_weight='balanced' if not use_smote else None, random_state=42)
        elif model_type == 'gbm':
            base_model = GradientBoostingClassifier(random_state=42)
        else:  # ensemble
            rf = RandomForestClassifier(class_weight='balanced' if not use_smote else None, random_state=42)
            gbm = GradientBoostingClassifier(random_state=42)
            base_model = VotingClassifier(estimators=[('rf', rf), ('gbm', gbm)], voting='soft')
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            print("Tuning hyperparameters...")
            
            if model_type == 'rf':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            elif model_type == 'gbm':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 0.9, 1.0]
                }
            else:  # ensemble
                param_grid = {
                    'rf__n_estimators': [100, 200],
                    'rf__max_depth': [None, 10],
                    'gbm__n_estimators': [100, 200],
                    'gbm__learning_rate': [0.05, 0.1]
                }
            
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=10,
                cv=cv,
                scoring='f1',
                random_state=42,
                n_jobs=-1
            )
            
            search.fit(X_train, y_train, sample_weight=sw_train)
            best_params = search.best_params_
            print(f"Best parameters: {best_params}")
            model = search.best_estimator_
        else:
            model = base_model
            model.fit(X_train, y_train, sample_weight=sw_train)
            best_params = {}
        
        # Calibrate probabilities
        print("Calibrating probabilities...")
        calibrated_model = CalibratedClassifierCV(model, cv=5)
        calibrated_model.fit(X_selected, y, sample_weight=sample_weights)
        
        # Evaluate model
        y_pred = calibrated_model.predict(X_test)
        y_prob = calibrated_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        if len(set(y_test)) > 1:  # Only calculate AUC if there are both classes in test set
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
        print("\nModel Evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Calculate feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_[0])
        elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
            feature_importance = model.estimators_[0].feature_importances_
        else:
            # For ensemble or other models without direct feature importance
            feature_importance = np.ones(len(selected_features))
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(importance_df.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        importance_df = importance_df.sort_values('Importance', ascending=True)
        plt.barh(importance_df['Feature'].values, importance_df['Importance'].values)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Generate model name if not provided
        if not model_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_model_{timestamp}"
        
        # Save feature importance plot
        os.makedirs(self.models_dir, exist_ok=True)
        plt.savefig(os.path.join(self.models_dir, f"{model_name}_importance.png"))
        plt.close()
        
        # Save model info
        model_info = {
            'model_name': model_name,
            'model_type': model_type,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics': metrics,
            'feature_names': selected_features,
            'feature_importance': importance_df.to_dict(),
            'best_params': best_params,
            'weighting_scheme': training_df['sample_weight'].unique().tolist() if 'sample_weight' in training_df.columns else None,
            'class_distribution': {
                'class_0': int(class_counts[0]),
                'class_1': int(class_counts[1])
            },
            'smote_applied': use_smote
        }
        
        # Save models
        with open(os.path.join(self.models_dir, f"{model_name}.pkl"), 'wb') as f:
            pickle.dump(calibrated_model, f)
        
        if selector:
            with open(os.path.join(self.models_dir, f"{model_name}_selector.pkl"), 'wb') as f:
                pickle.dump(selector, f)
        
        with open(os.path.join(self.models_dir, f"{model_name}_scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(os.path.join(self.models_dir, f"{model_name}_info.json"), 'w') as f:
            json.dump(model_info, f, indent=4, default=str)
        
        print(f"\nModel saved as {model_name}")
        return model_info
    
    def _split_data(self, X, y, sample_weights=None):
        """
        Split data into train and test sets while preserving sample weights.
        
        Args:
            X: Features
            y: Target
            sample_weights: Optional sample weights
            
        Returns:
            Tuple with split data
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        if sample_weights is not None:
            sw_train, sw_test = train_test_split(sample_weights, test_size=0.2, random_state=42, stratify=y)
        else:
            sw_train = None
            sw_test = None
        
        return X_train, X_test, y_train, y_test, sw_train, sw_test
    
    def analyze_past_errors(self, max_history_size=50):
        """
        Analyze past prediction errors to identify problematic features and patterns.
        
        Args:
            max_history_size: Maximum number of recent results to analyze
            
        Returns:
            dict: Analysis results
        """
        print("Analyzing past prediction errors...")
        
        # Get most recent results
        recent_history = self.history_df.sort_values('match_date', ascending=False).head(max_history_size)
        
        # Count errors by team and confidence level
        error_df = recent_history[~recent_history['prediction_correct']]
        
        if len(error_df) == 0:
            return {
                "error_count": 0,
                "message": "No errors found in recent prediction history."
            }
        
        # Extract error patterns
        error_patterns = {}
        
        # Count errors by team
        team1_counts = error_df['team1'].value_counts().to_dict()
        team2_counts = error_df['team2'].value_counts().to_dict()
        
        # Combine team counts
        team_counts = {}
        for team, count in team1_counts.items():
            team_counts[team] = team_counts.get(team, 0) + count
        for team, count in team2_counts.items():
            team_counts[team] = team_counts.get(team, 0) + count
        
        team_counts = {k: v for k, v in sorted(team_counts.items(), key=lambda item: item[1], reverse=True)}
        
        # Count errors by predicted winner
        predicted_counts = error_df['predicted_winner'].value_counts().to_dict()
        
        # Count errors by actual winner
        actual_counts = error_df['actual_winner'].value_counts().to_dict()
        
        # Find upsets (predicted high-ranked team lost to lower-ranked team)
        upsets = []
        for _, row in error_df.iterrows():
            if 'team1_ranking' in row and 'team2_ranking' in row:
                predicted = row['predicted_winner']
                actual = row['actual_winner']
                
                if predicted == row['team1'] and row['team1_ranking'] < row['team2_ranking']:
                    upsets.append((predicted, actual, row['win_probability']))
                elif predicted == row['team2'] and row['team2_ranking'] < row['team1_ranking']:
                    upsets.append((predicted, actual, row['win_probability']))
        
        error_patterns = {
            'total_errors': len(error_df),
            'error_rate': len(error_df) / len(recent_history),
            'problematic_teams': dict(list(team_counts.items())[:5]),
            'incorrectly_predicted_winners': dict(list(predicted_counts.items())[:5]),
            'unexpected_winners': dict(list(actual_counts.items())[:5]),
            'upsets': upsets[:5]
        }
        
        # Analyze prediction confidence
        error_df['confidence_level'] = pd.cut(
            error_df['win_probability'], 
            bins=[0, 0.55, 0.65, 0.75, 0.85, 1.0],
            labels=['Very Low (≤55%)', 'Low (56-65%)', 'Medium (66-75%)', 'High (76-85%)', 'Very High (>85%)']
        )
        
        confidence_distribution = error_df['confidence_level'].value_counts().to_dict()
        error_patterns['confidence_distribution'] = confidence_distribution
        
        # Try to load feature importance from result files
        feature_frequency = {}
        
        for _, row in error_df.iterrows():
            match_id = row['match_id']
            result_file = os.path.join(self.results_dir, f"{match_id}_result.json")
            
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        match_data = json.load(f)
                    
                    # Extract top features
                    if 'feature_importance' in match_data:
                        for feature, importance in match_data['feature_importance'].items():
                            if isinstance(importance, dict) and 'Importance' in importance:
                                importance_value = importance['Importance']
                            else:
                                importance_value = importance
                            
                            feature_frequency[feature] = feature_frequency.get(feature, 0) + 1
                except Exception as e:
                    print(f"Error analyzing match {match_id}: {e}")
        
        # Sort features by frequency in errors
        feature_frequency = {k: v for k, v in sorted(feature_frequency.items(), 
                                                    key=lambda item: item[1], reverse=True)}
        
        error_patterns['common_features'] = dict(list(feature_frequency.items())[:10])
        
        # Analyze model versions if available
        if 'model_version' in error_df.columns:
            version_error_rates = {}
            
            for version in recent_history['model_version'].unique():
                version_df = recent_history[recent_history['model_version'] == version]
                correct = version_df['prediction_correct'].sum()
                total = len(version_df)
                
                if total > 0:
                    version_error_rates[version] = 1 - (correct / total)
            
            error_patterns['model_version_error_rates'] = version_error_rates
        
        return error_patterns
    
    def optimize_hyperparameters(self, training_df, param_grid=None, n_iter=20):
        """
        Perform extensive hyperparameter optimization.
        
        Args:
            training_df: DataFrame with training data
            param_grid: Optional param grid dictionary
            n_iter: Number of iterations for randomized search
            
        Returns:
            dict: Best parameters
        """
        print("Performing extensive hyperparameter optimization...")
        
        if 'target' not in training_df.columns:
            raise ValueError("Training data must include 'target' column.")
        
        # Extract features and target
        X = training_df.drop(['target', 'team1', 'team2', 'match_date', 'sample_weight'], axis=1, errors='ignore')
        y = training_df['target']
        
        # Get sample weights if available
        sample_weights = None
        if 'sample_weight' in training_df.columns:
            sample_weights = training_df['sample_weight'].values
            X = X.drop('sample_weight', axis=1, errors='ignore')
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [None, 5, 10, 15, 20, 25],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 6, 8],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
        
        # Create cross-validation strategy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Random search with cross-validation
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='f1',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_scaled, y, sample_weight=sample_weights)
        
        best_params = search.best_params_
        best_score = search.best_score_
        
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation score: {best_score:.4f}")
        
        # Get results DataFrame
        results_df = pd.DataFrame(search.cv_results_)
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.models_dir, f"param_optimization_results_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'results_file': results_file
        }
    
    def suggest_improvements(self, error_analysis=None):
        """
        Suggest improvements to the prediction model based on error analysis.
        
        Args:
            error_analysis: Optional pre-computed error analysis
            
        Returns:
            list: Suggested improvements
        """
        if not error_analysis:
            error_analysis = self.analyze_past_errors()
        
        suggestions = []
        
        # Threshold for considering a team problematic
        prob_team_threshold = 2
        
        # Suggest improvements based on problematic teams
        if 'problematic_teams' in error_analysis:
            problematic_teams = []
            for team, count in error_analysis['problematic_teams'].items():
                if count >= prob_team_threshold:
                    problematic_teams.append(team)
            
            if problematic_teams:
                suggestions.append(
                    f"Consider gathering more detailed data for these teams that frequently cause prediction errors: {', '.join(problematic_teams)}"
                )
        
        # Suggest improvements based on confidence distribution
        if 'confidence_distribution' in error_analysis:
            conf_dist = error_analysis['confidence_distribution']
            if conf_dist:
                max_conf_level = max(conf_dist.items(), key=lambda x: x[1])[0]
                suggestions.append(
                    f"The model appears overconfident in the {max_conf_level} range. Consider recalibration or adding a calibration layer."
                )
        
        # Suggest improvements based on common features
        if 'common_features' in error_analysis and error_analysis['common_features']:
            common_features = list(error_analysis['common_features'].keys())[:3]
            suggestions.append(
                f"Consider reviewing or potentially downweighting these features that appear frequently in prediction errors: {', '.join(common_features)}"
            )
        
        # Suggest improvements based on upsets
        if 'upsets' in error_analysis and error_analysis['upsets']:
            suggestions.append(
                "The model struggles with predicting upsets (lower-ranked teams defeating higher-ranked teams). Consider adding features related to team form, head-to-head history, or recent performance."
            )
        
        # Suggest improvements based on model version error rates
        if 'model_version_error_rates' in error_analysis:
            version_rates = error_analysis['model_version_error_rates']
            if version_rates:
                best_version = min(version_rates.items(), key=lambda x: x[1])[0]
                suggestions.append(
                    f"Model version {best_version} has the lowest error rate. Consider using its configuration as a starting point."
                )
        
        # Add general suggestions
        suggestions.extend([
            "Consider adding more match-specific features like map selections, player performance data, or recent team composition changes.",
            "Explore different weighting schemes to prioritize recent matches.",
            "Experiment with different model architectures or ensemble methods.",
            "Consider developing specialized models for different tournament tiers or regions."
        ])
        
        return suggestions

def main():
    """Command line interface for the model retrainer."""
    parser = argparse.ArgumentParser(description='Retrain prediction models with past results')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train a new prediction model')
    train_parser.add_argument('--weighting', choices=['recency', 'prediction_error', 'both', 'none'], 
                            default='recency', help='Sample weighting scheme')
    train_parser.add_argument('--recency_days', type=int, default=90, 
                            help='Number of days to consider as "recent" for weighting')
    train_parser.add_argument('--model_type', choices=['rf', 'gbm', 'ensemble'], 
                            default='ensemble', help='Type of model to train')
    train_parser.add_argument('--no_feature_selection', action='store_true', 
                            help='Skip feature selection')
    train_parser.add_argument('--no_hyperparameter_tuning', action='store_true', 
                            help='Skip hyperparameter tuning')
    train_parser.add_argument('--no_smote', action='store_true', 
                            help='Skip SMOTE for class imbalance')
    train_parser.add_argument('--no_api_data', action='store_true', 
                            help='Skip fetching additional data from API')
    train_parser.add_argument('--model_name', help='Custom name for the model')
    
    # Analyze errors command
    error_parser = subparsers.add_parser('analyze', help='Analyze past prediction errors')
    error_parser.add_argument('--max_history', type=int, default=50, 
                            help='Maximum number of recent results to analyze')
    error_parser.add_argument('--output', help='Output JSON file for analysis')
    
    # Optimize hyperparameters command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize model hyperparameters')
    optimize_parser.add_argument('--weighting', choices=['recency', 'prediction_error', 'both', 'none'], 
                               default='recency', help='Sample weighting scheme')
    optimize_parser.add_argument('--n_iter', type=int, default=20, 
                               help='Number of iterations for randomized search')
    
    # Suggest improvements command
    suggest_parser = subparsers.add_parser('suggest', help='Suggest model improvements')
    
    args = parser.parse_args()
    
    try:
        retrainer = ModelRetrainer()
        
        if args.command == 'train':
            # Collect training data
            training_df = retrainer.collect_training_data(
                weighting_scheme=args.weighting,
                recency_days=args.recency_days,
                include_api_data=not args.no_api_data
            )
            
            # Train new model
            model_info = retrainer.train_model(
                training_df,
                feature_selection=not args.no_feature_selection,
                hyperparameter_tuning=not args.no_hyperparameter_tuning,
                model_type=args.model_type,
                use_smote=not args.no_smote,
                model_name=args.model_name
            )
            
            print("\n===== Model Training Completed =====")
            print(f"Model Name: {model_info['model_name']}")
            print(f"Model Type: {model_info['model_type']}")
            print(f"Accuracy: {model_info['metrics']['accuracy']:.4f}")
            print(f"F1 Score: {model_info['metrics']['f1_score']:.4f}")
            print(f"Top Features: {', '.join(list(model_info['feature_importance']['Feature'].values())[:5])}")
            print(f"\nTo use this model in predictions, update the model path in your prediction script.")
        
        elif args.command == 'analyze':
            # Analyze past errors
            error_analysis = retrainer.analyze_past_errors(args.max_history)
            
            print("\n===== Error Analysis =====")
            print(f"Total Errors: {error_analysis['total_errors']}")
            print(f"Error Rate: {error_analysis['error_rate']:.2%}")
            
            if 'problematic_teams' in error_analysis and error_analysis['problematic_teams']:
                print("\nMost Problematic Teams:")
                for team, count in error_analysis['problematic_teams'].items():
                    print(f"  {team}: {count} errors")
            
            if 'confidence_distribution' in error_analysis and error_analysis['confidence_distribution']:
                print("\nErrors by Confidence Level:")
                for level, count in error_analysis['confidence_distribution'].items():
                    print(f"  {level}: {count} errors")
            
            if 'common_features' in error_analysis and error_analysis['common_features']:
                print("\nMost Common Features in Errors:")
                for feature, count in error_analysis['common_features'].items():
                    print(f"  {feature}: {count} occurrences")
            
            if 'upsets' in error_analysis and error_analysis['upsets']:
                print("\nUpsets (High-ranked teams lost to lower-ranked teams):")
                for predicted, actual, probability in error_analysis['upsets']:
                    print(f"  Predicted: {predicted}, Actual: {actual}, Probability: {probability:.2%}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(error_analysis, f, indent=4, default=str)
                print(f"\nAnalysis saved to {args.output}")
        
        elif args.command == 'optimize':
            # Collect training data
            training_df = retrainer.collect_training_data(
                weighting_scheme=args.weighting,
                include_api_data=True
            )
            
            # Optimize hyperparameters
            result = retrainer.optimize_hyperparameters(
                training_df,
                n_iter=args.n_iter
            )
            
            print("\n===== Hyperparameter Optimization Completed =====")
            print(f"Best Score: {result['best_score']:.4f}")
            print("\nBest Parameters:")
            for param, value in result['best_params'].items():
                print(f"  {param}: {value}")
            print(f"\nDetailed results saved to {result['results_file']}")
        
        elif args.command == 'suggest':
            # Analyze errors
            error_analysis = retrainer.analyze_past_errors()
            
            # Get suggestions
            suggestions = retrainer.suggest_improvements(error_analysis)
            
            print("\n===== Suggested Model Improvements =====")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()