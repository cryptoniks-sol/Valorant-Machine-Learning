"""
Valorant Match Prediction System

This script creates a machine learning system to predict Valorant esports match outcomes.
It includes components for data collection, feature engineering, model training and evaluation,
and prediction output with confidence scores.
"""

import pandas as pd
import numpy as np
import requests
import json
import datetime
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import warnings
import logging
from datetime import timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("valorant_predictor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("valorant_predictor")

# Suppress warnings
warnings.filterwarnings('ignore')

# Define base API URL
BASE_URL = "http://localhost:5000/api/v1"

class DataCollector:
    """
    Class to handle data collection from the Valorant API endpoints.
    """
    
    def __init__(self, base_url: str = BASE_URL):
        """
        Initialize the DataCollector.
        
        Args:
            base_url: Base URL for the API
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str) -> Dict:
        """
        Make a request to the API.
        
        Args:
            endpoint: API endpoint to request
            
        Returns:
            JSON response as a dictionary
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {e}")
            return {}
    
    def get_all_teams(self, region: str = "all", limit: int = 100) -> List[Dict]:
        """
        Get information about all teams.
        
        Args:
            region: Filter by region
            limit: Limit of results per page
            
        Returns:
            List of team information dictionaries
        """
        endpoint = f"/teams?region={region}&limit={limit}"
        response = self._make_request(endpoint)
        return response.get("data", [])
    
    def get_team_info(self, team_id: str) -> Dict:
        """
        Get detailed information about a specific team.
        
        Args:
            team_id: ID of the team
            
        Returns:
            Team information dictionary
        """
        endpoint = f"/teams/{team_id}"
        return self._make_request(endpoint)
    
    def get_match_history(self, team_id: str) -> List[Dict]:
        """
        Get match history for a specific team.
        
        Args:
            team_id: ID of the team
            
        Returns:
            List of match history dictionaries
        """
        endpoint = f"/match-history/{team_id}"
        response = self._make_request(endpoint)
        return response.get("data", [])
    
    def get_match_details(self, match_id: str) -> Dict:
        """
        Get details about a specific match.
        
        Args:
            match_id: ID of the match
            
        Returns:
            Match details dictionary
        """
        endpoint = f"/match-details/{match_id}"
        return self._make_request(endpoint)
    
    def get_player_info(self, player_id: str, agent: str = "all") -> Dict:
        """
        Get information about a specific player.
        
        Args:
            player_id: ID of the player
            agent: Filter by agent
            
        Returns:
            Player information dictionary
        """
        endpoint = f"/players/{player_id}?agent={agent}"
        return self._make_request(endpoint)
    
    def get_upcoming_matches(self) -> List[Dict]:
        """
        Get information about upcoming matches.
        
        Returns:
            List of upcoming match dictionaries
        """
        endpoint = "/matches"
        response = self._make_request(endpoint)
        return response.get("data", [])


class FeatureEngineering:
    """
    Class to handle feature engineering from raw team and match data.
    """
    
    def __init__(self, data_collector: DataCollector):
        """
        Initialize the FeatureEngineering.
        
        Args:
            data_collector: DataCollector instance for retrieving data
        """
        self.data_collector = data_collector
    
    def calculate_win_rate(self, team_id: str, window: Optional[int] = None) -> float:
        """
        Calculate win rate for a team, optionally within a time window.
        
        Args:
            team_id: ID of the team
            window: Number of recent matches to consider (None for all matches)
            
        Returns:
            Win rate as a float between 0 and 1
        """
        team_info = self.data_collector.get_team_info(team_id)
        
        if not team_info:
            return 0.0
        
        total_matches = team_info.get("stats", {}).get("matches", 0)
        
        if window is None or window >= total_matches:
            # Use overall stats
            wins = team_info.get("stats", {}).get("wins", 0)
            if total_matches == 0:
                return 0.0
            return wins / total_matches
        else:
            # Calculate from recent matches
            match_history = self.data_collector.get_match_history(team_id)
            
            if not match_history:
                return 0.0
                
            recent_matches = match_history[:window]
            wins = sum(1 for match in recent_matches if match.get("result") == "win")
            
            return wins / len(recent_matches) if recent_matches else 0.0
    
    def calculate_average_score(self, team_id: str, window: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Calculate average score and opponent score for a team.
        
        Args:
            team_id: ID of the team
            window: Number of recent matches to consider (None for all matches)
            
        Returns:
            Tuple of (avg_score, avg_opponent_score, score_differential)
        """
        match_history = self.data_collector.get_match_history(team_id)
        
        if not match_history:
            return 0.0, 0.0, 0.0
            
        if window is not None:
            match_history = match_history[:window]
            
        team_scores = []
        opponent_scores = []
        
        for match in match_history:
            # Extract scores from match data
            team_score = match.get("team_score", 0)
            opponent_score = match.get("opponent_score", 0)
            
            team_scores.append(team_score)
            opponent_scores.append(opponent_score)
        
        avg_score = sum(team_scores) / len(team_scores) if team_scores else 0.0
        avg_opponent_score = sum(opponent_scores) / len(opponent_scores) if opponent_scores else 0.0
        score_differential = avg_score - avg_opponent_score
        
        return avg_score, avg_opponent_score, score_differential
    
    def calculate_recent_form(self, team_id: str, window: int = 5) -> float:
        """
        Calculate recent form as a weighted score.
        More recent matches have higher weight.
        
        Args:
            team_id: ID of the team
            window: Number of recent matches to consider
            
        Returns:
            Form score as a float between 0 and 1
        """
        match_history = self.data_collector.get_match_history(team_id)
        
        if not match_history:
            return 0.0
            
        recent_matches = match_history[:window]
        
        if not recent_matches:
            return 0.0
            
        # Weights decrease with match age
        weights = [1.0 * (0.8 ** i) for i in range(len(recent_matches))]
        weight_sum = sum(weights)
        
        form_score = sum(weight * (1.0 if match.get("result") == "win" else 0.0) 
                         for weight, match in zip(weights, recent_matches))
        
        return form_score / weight_sum if weight_sum > 0 else 0.0
    
    def get_map_performance(self, team_id: str, map_name: str) -> Dict:
        """
        Get map-specific performance for a team.
        
        Args:
            team_id: ID of the team
            map_name: Name of the map
            
        Returns:
            Dictionary with map performance metrics
        """
        match_history = self.data_collector.get_match_history(team_id)
        
        if not match_history:
            return {"win_rate": 0.0, "matches_played": 0, "avg_score": 0.0}
            
        map_matches = [match for match in match_history if match.get("map") == map_name]
        
        if not map_matches:
            return {"win_rate": 0.0, "matches_played": 0, "avg_score": 0.0}
            
        wins = sum(1 for match in map_matches if match.get("result") == "win")
        total = len(map_matches)
        
        team_scores = [match.get("team_score", 0) for match in map_matches]
        avg_score = sum(team_scores) / total if total > 0 else 0.0
        
        return {
            "win_rate": wins / total if total > 0 else 0.0,
            "matches_played": total,
            "avg_score": avg_score
        }
    
    def get_head_to_head(self, team1_id: str, team2_id: str) -> Dict:
        """
        Calculate head-to-head statistics between two teams.
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            
        Returns:
            Dictionary with head-to-head metrics
        """
        team1_matches = self.data_collector.get_match_history(team1_id)
        
        if not team1_matches:
            return {
                "team1_wins": 0,
                "team2_wins": 0,
                "total_matches": 0,
                "team1_win_rate": 0.0
            }
            
        # Find matches between these two teams
        h2h_matches = [match for match in team1_matches 
                      if match.get("opponent_id") == team2_id]
        
        if not h2h_matches:
            return {
                "team1_wins": 0,
                "team2_wins": 0,
                "total_matches": 0,
                "team1_win_rate": 0.0
            }
            
        team1_wins = sum(1 for match in h2h_matches if match.get("result") == "win")
        team2_wins = len(h2h_matches) - team1_wins
        
        return {
            "team1_wins": team1_wins,
            "team2_wins": team2_wins,
            "total_matches": len(h2h_matches),
            "team1_win_rate": team1_wins / len(h2h_matches) if h2h_matches else 0.0
        }
    
    def get_player_stats(self, team_id: str) -> Dict:
        """
        Get aggregated player statistics for a team.
        
        Args:
            team_id: ID of the team
            
        Returns:
            Dictionary with aggregated player metrics
        """
        team_info = self.data_collector.get_team_info(team_id)
        
        if not team_info or "players" not in team_info:
            return {
                "avg_acs": 0.0,
                "avg_kd": 0.0,
                "max_acs": 0.0,
                "max_kd": 0.0
            }
            
        players = team_info.get("players", [])
        
        if not players:
            return {
                "avg_acs": 0.0,
                "avg_kd": 0.0,
                "max_acs": 0.0,
                "max_kd": 0.0
            }
        
        player_stats = []
        
        for player in players:
            player_id = player.get("id")
            if player_id:
                player_info = self.data_collector.get_player_info(player_id)
                if player_info:
                    acs = player_info.get("stats", {}).get("acs", 0.0)
                    kd = player_info.get("stats", {}).get("kd", 0.0)
                    player_stats.append({"acs": acs, "kd": kd})
        
        if not player_stats:
            return {
                "avg_acs": 0.0,
                "avg_kd": 0.0,
                "max_acs": 0.0,
                "max_kd": 0.0
            }
            
        avg_acs = sum(player["acs"] for player in player_stats) / len(player_stats)
        avg_kd = sum(player["kd"] for player in player_stats) / len(player_stats)
        max_acs = max(player["acs"] for player in player_stats)
        max_kd = max(player["kd"] for player in player_stats)
        
        return {
            "avg_acs": avg_acs,
            "avg_kd": avg_kd,
            "max_acs": max_acs,
            "max_kd": max_kd
        }
    
    def is_lan_event(self, match_id: str) -> bool:
        """
        Determine if a match is played on LAN.
        
        Args:
            match_id: ID of the match
            
        Returns:
            Boolean indicating if the match is on LAN
        """
        match_details = self.data_collector.get_match_details(match_id)
        
        if not match_details:
            return False
            
        # Look for LAN indicators in event type or name
        event_type = match_details.get("event", {}).get("type", "").lower()
        event_name = match_details.get("event", {}).get("name", "").lower()
        
        lan_keywords = ["lan", "major", "champions", "masters", "international"]
        
        return any(keyword in event_type or keyword in event_name for keyword in lan_keywords)
    
    def extract_features_for_match(
            self, team1_id: str, team2_id: str, map_name: Optional[str] = None) -> Dict:
        """
        Extract comprehensive features for a match between two teams.
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            map_name: Name of the map (optional)
            
        Returns:
            Dictionary with features for the match
        """
        # Get team statistics
        team1_info = self.data_collector.get_team_info(team1_id)
        team2_info = self.data_collector.get_team_info(team2_id)
        
        if not team1_info or not team2_info:
            logger.warning(f"Missing team info for {team1_id} or {team2_id}")
            return {}
        
        # Overall win rates
        team1_win_rate = self.calculate_win_rate(team1_id)
        team2_win_rate = self.calculate_win_rate(team2_id)
        
        # Recent win rates (last 5, 10, 20 matches)
        team1_win_rate_5 = self.calculate_win_rate(team1_id, 5)
        team1_win_rate_10 = self.calculate_win_rate(team1_id, 10)
        team1_win_rate_20 = self.calculate_win_rate(team1_id, 20)
        
        team2_win_rate_5 = self.calculate_win_rate(team2_id, 5)
        team2_win_rate_10 = self.calculate_win_rate(team2_id, 10)
        team2_win_rate_20 = self.calculate_win_rate(team2_id, 20)
        
        # Average scores
        team1_avg_score, team1_avg_opp_score, team1_score_diff = self.calculate_average_score(team1_id)
        team2_avg_score, team2_avg_opp_score, team2_score_diff = self.calculate_average_score(team2_id)
        
        # Recent form
        team1_form = self.calculate_recent_form(team1_id)
        team2_form = self.calculate_recent_form(team2_id)
        
        # Map performance (if specified)
        team1_map_perf = self.get_map_performance(team1_id, map_name) if map_name else {}
        team2_map_perf = self.get_map_performance(team2_id, map_name) if map_name else {}
        
        # Head-to-head statistics
        h2h_stats = self.get_head_to_head(team1_id, team2_id)
        
        # Player statistics
        team1_player_stats = self.get_player_stats(team1_id)
        team2_player_stats = self.get_player_stats(team2_id)
        
        # Team rankings and ratings
        team1_ranking = team1_info.get("ranking", 0)
        team2_ranking = team2_info.get("ranking", 0)
        team1_rating = team1_info.get("rating", 0.0)
        team2_rating = team2_info.get("rating", 0.0)
        
        # Assemble features
        features = {
            # Team IDs (for reference)
            "team1_id": team1_id,
            "team2_id": team2_id,
            
            # Overall performance
            "team1_win_rate": team1_win_rate,
            "team2_win_rate": team2_win_rate,
            "team1_matches_played": team1_info.get("stats", {}).get("matches", 0),
            "team2_matches_played": team2_info.get("stats", {}).get("matches", 0),
            
            # Recent performance
            "team1_win_rate_5": team1_win_rate_5,
            "team1_win_rate_10": team1_win_rate_10,
            "team1_win_rate_20": team1_win_rate_20,
            "team2_win_rate_5": team2_win_rate_5,
            "team2_win_rate_10": team2_win_rate_10,
            "team2_win_rate_20": team2_win_rate_20,
            
            # Score statistics
            "team1_avg_score": team1_avg_score,
            "team1_avg_opp_score": team1_avg_opp_score,
            "team1_score_diff": team1_score_diff,
            "team2_avg_score": team2_avg_score,
            "team2_avg_opp_score": team2_avg_opp_score,
            "team2_score_diff": team2_score_diff,
            
            # Form
            "team1_form": team1_form,
            "team2_form": team2_form,
            
            # Map performance (if specified)
            "team1_map_win_rate": team1_map_perf.get("win_rate", 0.0) if map_name else 0.0,
            "team2_map_win_rate": team2_map_perf.get("win_rate", 0.0) if map_name else 0.0,
            "team1_map_matches": team1_map_perf.get("matches_played", 0) if map_name else 0,
            "team2_map_matches": team2_map_perf.get("matches_played", 0) if map_name else 0,
            
            # Head-to-head
            "h2h_team1_wins": h2h_stats.get("team1_wins", 0),
            "h2h_team2_wins": h2h_stats.get("team2_wins", 0),
            "h2h_total_matches": h2h_stats.get("total_matches", 0),
            "h2h_team1_win_rate": h2h_stats.get("team1_win_rate", 0.0),
            
            # Player statistics
            "team1_avg_acs": team1_player_stats.get("avg_acs", 0.0),
            "team1_avg_kd": team1_player_stats.get("avg_kd", 0.0),
            "team1_max_acs": team1_player_stats.get("max_acs", 0.0),
            "team1_max_kd": team1_player_stats.get("max_kd", 0.0),
            "team2_avg_acs": team2_player_stats.get("avg_acs", 0.0),
            "team2_avg_kd": team2_player_stats.get("avg_kd", 0.0),
            "team2_max_acs": team2_player_stats.get("max_acs", 0.0),
            "team2_max_kd": team2_player_stats.get("max_kd", 0.0),
            
            # Rankings and ratings
            "team1_ranking": team1_ranking,
            "team2_ranking": team2_ranking,
            "team1_rating": team1_rating,
            "team2_rating": team2_rating,
            
            # Comparative features
            "ranking_diff": team1_ranking - team2_ranking,
            "rating_diff": team1_rating - team2_rating,
            "win_rate_diff": team1_win_rate - team2_win_rate,
            "recent_form_diff": team1_form - team2_form,
            "score_diff_diff": team1_score_diff - team2_score_diff,
            "map_win_rate_diff": (team1_map_perf.get("win_rate", 0.0) - 
                                 team2_map_perf.get("win_rate", 0.0)) if map_name else 0.0,
        }
        
        return features
    
    def create_match_dataset(self, from_date: Optional[str] = None, 
                           to_date: Optional[str] = None) -> pd.DataFrame:
        """
        Create a dataset of historical matches for model training.
        
        Args:
            from_date: Start date for matches (YYYY-MM-DD format)
            to_date: End date for matches (YYYY-MM-DD format)
            
        Returns:
            DataFrame with match features and outcomes
        """
        # Get all teams
        teams = self.data_collector.get_all_teams()
        
        if not teams:
            logger.error("No teams found")
            return pd.DataFrame()
            
        matches_data = []
        
        # For each team, get their match history
        for team in tqdm(teams, desc="Processing teams"):
            team_id = team.get("id")
            
            if not team_id:
                continue
                
            match_history = self.data_collector.get_match_history(team_id)
            
            if not match_history:
                continue
                
            # Filter by date if specified
            if from_date or to_date:
                filtered_matches = []
                
                for match in match_history:
                    match_date = match.get("date")
                    
                    if not match_date:
                        continue
                        
                    match_datetime = datetime.datetime.strptime(match_date, "%Y-%m-%d")
                    
                    if from_date:
                        from_datetime = datetime.datetime.strptime(from_date, "%Y-%m-%d")
                        if match_datetime < from_datetime:
                            continue
                            
                    if to_date:
                        to_datetime = datetime.datetime.strptime(to_date, "%Y-%m-%d")
                        if match_datetime > to_datetime:
                            continue
                            
                    filtered_matches.append(match)
                    
                match_history = filtered_matches
            
            # Process each match
            for match in match_history:
                match_id = match.get("id")
                opponent_id = match.get("opponent_id")
                map_name = match.get("map")
                
                if not match_id or not opponent_id:
                    continue
                    
                # Extract outcome (team1 win = 1, team1 loss = 0)
                team1_win = 1 if match.get("result") == "win" else 0
                
                # Extract features for this match
                try:
                    features = self.extract_features_for_match(team_id, opponent_id, map_name)
                    
                    if not features:
                        continue
                        
                    # Add outcome
                    features["team1_win"] = team1_win
                    
                    # Add map name
                    features["map"] = map_name if map_name else "unknown"
                    
                    matches_data.append(features)
                except Exception as e:
                    logger.error(f"Error extracting features for match {match_id}: {e}")
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(matches_data)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df


class ModelTrainer:
    """
    Class to handle machine learning model training and evaluation.
    """
    
    def __init__(self, feature_cols: Optional[List[str]] = None):
        """
        Initialize the ModelTrainer.
        
        Args:
            feature_cols: List of feature column names to use
        """
        self.feature_cols = feature_cols
        self.models = {}
        self.pipelines = {}
        self.feature_importances = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training.
        
        Args:
            data: DataFrame with match data
            
        Returns:
            Tuple of (X, y) with features and labels
        """
        if self.feature_cols is None:
            # Exclude non-feature columns
            exclude_cols = ["team1_id", "team2_id", "team1_win", "map"]
            self.feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[self.feature_cols].values
        y = data["team1_win"].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def train_models(self, data: pd.DataFrame) -> Dict:
        """
        Train multiple models on the given data.
        
        Args:
            data: DataFrame with match data
            
        Returns:
            Dictionary with model evaluation metrics
        """
        X, y = self.preprocess_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize models
        models = {
            "random_forest": RandomForestClassifier(random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "xgboost": xgb.XGBClassifier(random_state=42),
            "neural_network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name} model...")
            
            # Create pipeline
            pipeline = Pipeline([
                ("classifier", model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Save model
            self.models[name] = model
            self.pipelines[name] = pipeline
            
            # Predict
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred)
            }
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X, y, cv=5)
            metrics["cv_mean"] = cv_scores.mean()
            metrics["cv_std"] = cv_scores.std()
            
            results[name] = metrics
            
            # Get feature importances (for tree-based models)
            if hasattr(model, "feature_importances_"):
                self.feature_importances[name] = {
                    feature: importance 
                    for feature, importance in zip(self.feature_cols, model.feature_importances_)
                }
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda name: results[name]["accuracy"])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
        
        return results
    
    def tune_hyperparameters(self, data: pd.DataFrame, model_name: str) -> Dict:
        """
        Tune hyperparameters for a specific model.
        
        Args:
            data: DataFrame with match data
            model_name: Name of the model to tune
            
        Returns:
            Dictionary with best parameters and scores
        """
        X, y = self.preprocess_data(data)
        
        # Define parameter grid for different models
        param_grids = {
            "random_forest": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [None, 10, 20, 30],
                "classifier__min_samples_split": [2, 5, 10]
            },
            "gradient_boosting": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
                "classifier__max_depth": [3, 5, 7]
            },
            "xgboost": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
                "classifier__max_depth": [3, 5, 7],
                "classifier__gamma": [0, 0.1, 0.2]
            },
            "neural_network": {
                "classifier__hidden_layer_sizes": [(50,), (100,), (100, 50), (200, 100)],
                "classifier__alpha": [0.0001, 0.001, 0.01],
                "classifier__learning_rate": ["constant", "adaptive"]
            }
        }
        
        if model_name not in param_grids:
            logger.error(f"Model {model_name} not supported for hyperparameter tuning")
            return {}
            
        # Get pipeline and parameter grid
        pipeline = self.pipelines.get(model_name)
        param_grid = param_grids[model_name]
        
        if not pipeline:
            logger.error(f"Model {model_name} not trained yet")
            return {}
            
        # Create grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1
        )
        
        # Train grid search
        logger.info(f"Tuning hyperparameters for {model_name}...")
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.pipelines[model_name] = grid_search.best_estimator_
        self.models[model_name] = grid_search.best_estimator_.named_steps["classifier"]
        
        if model_name == self.best_model_name:
            self.best_model = self.models[model_name]
        
        # Get feature importances (for tree-based models)
        if hasattr(self.models[model_name], "feature_importances_"):
            self.feature_importances[model_name] = {
                feature: importance 
                for feature, importance in zip(self.feature_cols, self.models[model_name].feature_importances_)
            }
        
        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Dict:
        """
        Get feature importance for a specific model.
        
        Args:
            model_name: Name of the model (None for best model)
            
        Returns:
            Dictionary with feature importances
        """
        if model_name is None:
            model_name = self.best_model_name
            
        return self.feature_importances.get(model_name, {})
    
    def save_models(self, directory: str = "models") -> None:
        """
        Save trained models to disk.
        
        Args:
            directory: Directory to save models
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save each model
        for name, pipeline in self.pipelines.items():
            model_path = os.path.join(directory, f"{name}.joblib")
            joblib.dump(pipeline, model_path)
            logger.info(f"Saved {name} model to {model_path}")
            
        # Save feature columns and scaler
        feature_cols_path = os.path.join(directory, "feature_cols.joblib")
        scaler_path = os.path.join(directory, "scaler.joblib")
        
        joblib.dump(self.feature_cols, feature_cols_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Saved feature columns to {feature_cols_path}")
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save best model name
        with open(os.path.join(directory, "best_model.txt"), "w") as f:
            f.write(self.best_model_name)
    
    def load_models(self, directory: str = "models") -> bool:
        """
        Load trained models from disk.
        
        Args:
            directory: Directory to load models from
            
        Returns:
            Boolean indicating success
        """
        try:
            # Load feature columns and scaler
            feature_cols_path = os.path.join(directory, "feature_cols.joblib")
            scaler_path = os.path.join(directory, "scaler.joblib")
            
            self.feature_cols = joblib.load(feature_cols_path)
            self.scaler = joblib.load(scaler_path)
            
            # Load best model name
            with open(os.path.join(directory, "best_model.txt"), "r") as f:
                self.best_model_name = f.read().strip()
            
            # Load each model
            model_names = ["random_forest", "gradient_boosting", "xgboost", "neural_network"]
            
            for name in model_names:
                model_path = os.path.join(directory, f"{name}.joblib")
                
                if os.path.exists(model_path):
                    self.pipelines[name] = joblib.load(model_path)
                    self.models[name] = self.pipelines[name].named_steps["classifier"]
                    
                    # Get feature importances (for tree-based models)
                    if hasattr(self.models[name], "feature_importances_"):
                        self.feature_importances[name] = {
                            feature: importance 
                            for feature, importance in zip(self.feature_cols, self.models[name].feature_importances_)
                        }
            
            # Set best model
            self.best_model = self.models.get(self.best_model_name)
            
            logger.info(f"Loaded models from {directory}")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict_match(
            self, features: Dict, model_name: Optional[str] = None) -> Dict:
        """
        Predict match outcome based on features.
        
        Args:
            features: Dictionary with match features
            model_name: Name of the model to use (None for best model)
            
        Returns:
            Dictionary with prediction results
        """
        if model_name is None:
            model_name = self.best_model_name
            
        pipeline = self.pipelines.get(model_name)
        
        if not pipeline:
            logger.error(f"Model {model_name} not found")
            return {"error": f"Model {model_name} not found"}
            
        # Extract feature values
        X = np.array([features.get(feature, 0.0) for feature in self.feature_cols]).reshape(1, -1)
        
        # Scale features
        X = self.scaler.transform(X)
        
        # Predict probability
        proba = pipeline.predict_proba(X)[0]
        
        # Get predicted class and probability
        team1_win_prob = proba[1]
        team2_win_prob = proba[0]
        predicted_winner = "team1" if team1_win_prob > team2_win_prob else "team2"
        
        # Get confidence level
        confidence = max(team1_win_prob, team2_win_prob)
        
        # Categorize confidence
        if confidence > 0.8:
            confidence_level = "Very High"
        elif confidence > 0.7:
            confidence_level = "High"
        elif confidence > 0.6:
            confidence_level = "Moderate"
        elif confidence > 0.55:
            confidence_level = "Slight"
        else:
            confidence_level = "Uncertain"
            
        # Get top feature importances
        if model_name in self.feature_importances:
            importances = self.feature_importances[model_name]
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            top_features = []
            
        return {
            "predicted_winner": predicted_winner,
            "team1_win_probability": team1_win_prob,
            "team2_win_probability": team2_win_prob,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "model_used": model_name,
            "top_features": top_features
        }


class BettingAnalyzer:
    """
    Class to handle betting analysis and recommendations.
    """
    
    def __init__(self):
        """Initialize the BettingAnalyzer."""
        pass
    
    def calculate_implied_probability(self, odds: float) -> float:
        """
        Calculate implied probability from decimal odds.
        
        Args:
            odds: Decimal odds
            
        Returns:
            Implied probability (0-1)
        """
        return 1.0 / odds
    
    def identify_value_bets(
            self, prediction: Dict, team1_odds: float, team2_odds: float) -> Dict:
        """
        Identify value bets by comparing model predictions with market odds.
        
        Args:
            prediction: Match prediction dictionary
            team1_odds: Decimal odds for team1
            team2_odds: Decimal odds for team2
            
        Returns:
            Dictionary with value bet recommendations
        """
        # Calculate implied probabilities from odds
        team1_implied_prob = self.calculate_implied_probability(team1_odds)
        team2_implied_prob = self.calculate_implied_probability(team2_odds)
        
        # Get model probabilities
        team1_model_prob = prediction.get("team1_win_probability", 0.0)
        team2_model_prob = prediction.get("team2_win_probability", 0.0)
        
        # Calculate edges
        team1_edge = team1_model_prob - team1_implied_prob
        team2_edge = team2_model_prob - team2_implied_prob
        
        # Determine value bets
        value_bets = []
        
        if team1_edge > 0.05:
            value_bets.append({
                "team": "team1",
                "edge": team1_edge,
                "model_probability": team1_model_prob,
                "implied_probability": team1_implied_prob,
                "odds": team1_odds
            })
            
        if team2_edge > 0.05:
            value_bets.append({
                "team": "team2",
                "edge": team2_edge,
                "model_probability": team2_model_prob,
                "implied_probability": team2_implied_prob,
                "odds": team2_odds
            })
            
        return {
            "value_bets": value_bets,
            "team1_edge": team1_edge,
            "team2_edge": team2_edge
        }
    
    def calculate_kelly_criterion(
            self, edge: float, win_probability: float, odds: float) -> float:
        """
        Calculate Kelly Criterion for optimal bet sizing.
        
        Args:
            edge: Edge (model probability - implied probability)
            win_probability: Model win probability
            odds: Decimal odds
            
        Returns:
            Kelly Criterion bet size (as fraction of bankroll)
        """
        # Calculate Kelly fraction
        q = 1.0 - win_probability  # Probability of losing
        p = win_probability  # Probability of winning
        b = odds - 1.0  # Decimal odds - 1
        
        # Kelly formula: f* = (bp - q) / b
        kelly = (b * p - q) / b
        
        # Cap at 0.1 (10% of bankroll) for safety
        return min(max(kelly, 0.0), 0.1)
    
    def generate_betting_advice(
            self, prediction: Dict, team1_odds: float, team2_odds: float,
            handicap_odds: Optional[Dict] = None, over_under_odds: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive betting advice for a match.
        
        Args:
            prediction: Match prediction dictionary
            team1_odds: Decimal odds for team1
            team2_odds: Decimal odds for team2
            handicap_odds: Dictionary with handicap odds
            over_under_odds: Dictionary with over/under odds
            
        Returns:
            Dictionary with betting advice
        """
        # Identify value bets
        value_bets = self.identify_value_bets(prediction, team1_odds, team2_odds)
        
        # Generate betting recommendations
        recommendations = []
        
        for bet in value_bets.get("value_bets", []):
            team = bet.get("team")
            edge = bet.get("edge")
            win_probability = bet.get("model_probability")
            odds = bet.get("odds")
            
            kelly = self.calculate_kelly_criterion(edge, win_probability, odds)
            
            recommendations.append({
                "bet_type": "Match Winner",
                "selection": team,
                "odds": odds,
                "edge": edge,
                "kelly_criterion": kelly,
                "recommendation": f"Bet {kelly:.1%} of bankroll on {team} to win"
            })
            
        # Analyze handicap bets if provided
        if handicap_odds:
            team1_plus = handicap_odds.get("team1_plus", 0.0)
            team1_minus = handicap_odds.get("team1_minus", 0.0)
            
            # Simple heuristic for handicap bets based on win probabilities
            team1_win_prob = prediction.get("team1_win_probability", 0.0)
            
            if team1_win_prob > 0.75 and team1_minus > 1.5:
                implied_prob = self.calculate_implied_probability(team1_minus)
                edge = team1_win_prob - implied_prob
                
                if edge > 0.05:
                    kelly = self.calculate_kelly_criterion(edge, team1_win_prob, team1_minus)
                    
                    recommendations.append({
                        "bet_type": "Handicap",
                        "selection": "team1 -1.5",
                        "odds": team1_minus,
                        "edge": edge,
                        "kelly_criterion": kelly,
                        "recommendation": f"Bet {kelly:.1%} of bankroll on team1 -1.5"
                    })
                    
            elif team1_win_prob < 0.4 and team1_plus > 1.5:
                # For underdog with +1.5 handicap
                # Estimate probability of winning at least one map
                win_at_least_one_prob = 0.7 if team1_win_prob > 0.25 else 0.5
                
                implied_prob = self.calculate_implied_probability(team1_plus)
                edge = win_at_least_one_prob - implied_prob
                
                if edge > 0.05:
                    kelly = self.calculate_kelly_criterion(edge, win_at_least_one_prob, team1_plus)
                    
                    recommendations.append({
                        "bet_type": "Handicap",
                        "selection": "team1 +1.5",
                        "odds": team1_plus,
                        "edge": edge,
                        "kelly_criterion": kelly,
                        "recommendation": f"Bet {kelly:.1%} of bankroll on team1 +1.5"
                    })
                    
        # Analyze over/under bets if provided
        if over_under_odds:
            over_odds = over_under_odds.get("over", 0.0)
            under_odds = over_under_odds.get("under", 0.0)
            
            # Simplified heuristic for over/under bets
            # If teams are closely matched, maps are more likely to go the distance
            team1_win_prob = prediction.get("team1_win_probability", 0.0)
            team2_win_prob = prediction.get("team2_win_probability", 0.0)
            
            if abs(team1_win_prob - team2_win_prob) < 0.1 and over_odds > 1.8:
                # Close match, bet on over
                over_prob = 0.65  # Estimated probability based on heuristic
                
                implied_prob = self.calculate_implied_probability(over_odds)
                edge = over_prob - implied_prob
                
                if edge > 0.05:
                    kelly = self.calculate_kelly_criterion(edge, over_prob, over_odds)
                    
                    recommendations.append({
                        "bet_type": "Over/Under",
                        "selection": "Over",
                        "odds": over_odds,
                        "edge": edge,
                        "kelly_criterion": kelly,
                        "recommendation": f"Bet {kelly:.1%} of bankroll on Over"
                    })
                    
            elif abs(team1_win_prob - team2_win_prob) > 0.25 and under_odds > 1.8:
                # Lopsided match, bet on under
                under_prob = 0.65  # Estimated probability based on heuristic
                
                implied_prob = self.calculate_implied_probability(under_odds)
                edge = under_prob - implied_prob
                
                if edge > 0.05:
                    kelly = self.calculate_kelly_criterion(edge, under_prob, under_odds)
                    
                    recommendations.append({
                        "bet_type": "Over/Under",
                        "selection": "Under",
                        "odds": under_odds,
                        "edge": edge,
                        "kelly_criterion": kelly,
                        "recommendation": f"Bet {kelly:.1%} of bankroll on Under"
                    })
                    
        return {
            "value_bets": value_bets,
            "recommendations": recommendations,
            "match_prediction": prediction
        }


class ValorantPredictorSystem:
    """
    Main class to integrate all components of the Valorant Predictor System.
    """
    
    def __init__(self, base_url: str = BASE_URL, models_dir: str = "models"):
        """
        Initialize the ValorantPredictorSystem.
        
        Args:
            base_url: Base URL for the API
            models_dir: Directory to save/load models
        """
        self.data_collector = DataCollector(base_url)
        self.feature_engineering = FeatureEngineering(self.data_collector)
        self.model_trainer = ModelTrainer()
        self.betting_analyzer = BettingAnalyzer()
        self.models_dir = models_dir
        
        # Try to load pre-trained models
        if os.path.exists(models_dir):
            self.model_trainer.load_models(models_dir)
    
    def collect_and_train(
            self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> Dict:
        """
        Collect data and train models.
        
        Args:
            from_date: Start date for matches (YYYY-MM-DD format)
            to_date: End date for matches (YYYY-MM-DD format)
            
        Returns:
            Dictionary with training results
        """
        # Create dataset
        logger.info("Creating match dataset...")
        df = self.feature_engineering.create_match_dataset(from_date, to_date)
        
        if df.empty:
            logger.error("No matches found for training")
            return {"error": "No matches found for training"}
            
        logger.info(f"Created dataset with {len(df)} matches")
        
        # Train models
        logger.info("Training models...")
        training_results = self.model_trainer.train_models(df)
        
        # Save models
        logger.info("Saving models...")
        self.model_trainer.save_models(self.models_dir)
        
        return {
            "dataset_size": len(df),
            "training_results": training_results,
            "feature_importance": self.model_trainer.get_feature_importance()
        }
    
    def retrain_model(
            self, model_name: str, from_date: Optional[str] = None,
            to_date: Optional[str] = None) -> Dict:
        """
        Retrain a specific model.
        
        Args:
            model_name: Name of the model to retrain
            from_date: Start date for matches (YYYY-MM-DD format)
            to_date: End date for matches (YYYY-MM-DD format)
            
        Returns:
            Dictionary with training results
        """
        # Create dataset
        logger.info(f"Retraining {model_name} model...")
        df = self.feature_engineering.create_match_dataset(from_date, to_date)
        
        if df.empty:
            logger.error("No matches found for training")
            return {"error": "No matches found for training"}
            
        logger.info(f"Created dataset with {len(df)} matches")
        
        # Tune hyperparameters
        tuning_results = self.model_trainer.tune_hyperparameters(df, model_name)
        
        # Save models
        logger.info("Saving models...")
        self.model_trainer.save_models(self.models_dir)
        
        return {
            "dataset_size": len(df),
            "tuning_results": tuning_results,
            "feature_importance": self.model_trainer.get_feature_importance(model_name)
        }
    
    def predict_upcoming_matches(self) -> List[Dict]:
        """
        Predict outcomes for upcoming matches.
        
        Returns:
            List of prediction dictionaries for upcoming matches
        """
        # Get upcoming matches
        upcoming_matches = self.data_collector.get_upcoming_matches()
        
        if not upcoming_matches:
            logger.warning("No upcoming matches found")
            return []
            
        predictions = []
        
        for match in upcoming_matches:
            team1_id = match.get("team1_id")
            team2_id = match.get("team2_id")
            match_id = match.get("id")
            
            if not team1_id or not team2_id:
                continue
                
            # Extract features
            features = self.feature_engineering.extract_features_for_match(team1_id, team2_id)
            
            if not features:
                logger.warning(f"Could not extract features for match {match_id}")
                continue
                
            # Predict outcome
            prediction = self.model_trainer.predict_match(features)
            
            # Add match details
            prediction["match_id"] = match_id
            prediction["team1_id"] = team1_id
            prediction["team2_id"] = team2_id
            prediction["team1_name"] = match.get("team1_name", "Team 1")
            prediction["team2_name"] = match.get("team2_name", "Team 2")
            prediction["date"] = match.get("date")
            prediction["event"] = match.get("event", {}).get("name")
            
            predictions.append(prediction)
            
        return predictions
    
    def predict_specific_match(
            self, team1_id: str, team2_id: str, map_name: Optional[str] = None,
            team1_odds: Optional[float] = None, team2_odds: Optional[float] = None,
            handicap_odds: Optional[Dict] = None, 
            over_under_odds: Optional[Dict] = None) -> Dict:
        """
        Predict outcome for a specific match.
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            map_name: Name of the map (optional)
            team1_odds: Decimal odds for team1 (optional)
            team2_odds: Decimal odds for team2 (optional)
            handicap_odds: Dictionary with handicap odds (optional)
            over_under_odds: Dictionary with over/under odds (optional)
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.feature_engineering.extract_features_for_match(team1_id, team2_id, map_name)
        
        if not features:
            logger.error(f"Could not extract features for match between {team1_id} and {team2_id}")
            return {"error": "Could not extract features for match"}
            
        # Predict outcome
        prediction = self.model_trainer.predict_match(features)
        
        # Add team info
        team1_info = self.data_collector.get_team_info(team1_id)
        team2_info = self.data_collector.get_team_info(team2_id)
        
        prediction["team1_id"] = team1_id
        prediction["team2_id"] = team2_id
        prediction["team1_name"] = team1_info.get("name", "Team 1")
        prediction["team2_name"] = team2_info.get("name", "Team 2")
        
        if map_name:
            prediction["map"] = map_name
            
        # Add betting analysis if odds are provided
        if team1_odds and team2_odds:
            betting_advice = self.betting_analyzer.generate_betting_advice(
                prediction, team1_odds, team2_odds, handicap_odds, over_under_odds
            )
            
            prediction["betting_advice"] = betting_advice
            
        return prediction
    
    def backtest_model(
            self, from_date: str, to_date: str, 
            model_name: Optional[str] = None) -> Dict:
        """
        Backtest model against historical matches.
        
        Args:
            from_date: Start date for matches (YYYY-MM-DD format)
            to_date: End date for matches (YYYY-MM-DD format)
            model_name: Name of the model to backtest (None for best model)
            
        Returns:
            Dictionary with backtesting results
        """
        # Create dataset
        logger.info("Creating match dataset for backtesting...")
        df = self.feature_engineering.create_match_dataset(from_date, to_date)
        
        if df.empty:
            logger.error("No matches found for backtesting")
            return {"error": "No matches found for backtesting"}
            
        logger.info(f"Created dataset with {len(df)} matches")
        
        # Get model
        if model_name is None:
            model_name = self.model_trainer.best_model_name
            
        pipeline = self.model_trainer.pipelines.get(model_name)
        
        if not pipeline:
            logger.error(f"Model {model_name} not found")
            return {"error": f"Model {model_name} not found"}
            
        # Prepare features and labels
        X, y = self.model_trainer.preprocess_data(df)
        
        # Make predictions
        y_pred = pipeline.predict(X)
        y_proba = pipeline.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        
        # Analyze predictions
        team1_ids = df["team1_id"].values
        team2_ids = df["team2_id"].values
        
        matches = []
        
        for i in range(len(y)):
            team1_id = team1_ids[i]
            team2_id = team2_ids[i]
            
            # Get team info
            team1_info = self.data_collector.get_team_info(team1_id)
            team2_info = self.data_collector.get_team_info(team2_id)
            
            team1_name = team1_info.get("name", "Team 1")
            team2_name = team2_info.get("name", "Team 2")
            
            # Get actual and predicted outcomes
            actual = "team1" if y[i] == 1 else "team2"
            predicted = "team1" if y_pred[i] == 1 else "team2"
            
            # Get probabilities
            team1_prob = y_proba[i][1]
            team2_prob = y_proba[i][0]
            
            matches.append({
                "team1_id": team1_id,
                "team2_id": team2_id,
                "team1_name": team1_name,
                "team2_name": team2_name,
                "actual_winner": actual,
                "predicted_winner": predicted,
                "correct": actual == predicted,
                "team1_probability": team1_prob,
                "team2_probability": team2_prob,
                "confidence": max(team1_prob, team2_prob)
            })
            
        # Calculate performance by confidence level
        confidence_bins = {
            "Very High (>80%)": [],
            "High (70-80%)": [],
            "Moderate (60-70%)": [],
            "Slight (55-60%)": [],
            "Uncertain (<55%)": []
        }
        
        for match in matches:
            confidence = match["confidence"]
            
            if confidence > 0.8:
                confidence_bins["Very High (>80%)"].append(match)
            elif confidence > 0.7:
                confidence_bins["High (70-80%)"].append(match)
            elif confidence > 0.6:
                confidence_bins["Moderate (60-70%)"].append(match)
            elif confidence > 0.55:
                confidence_bins["Slight (55-60%)"].append(match)
            else:
                confidence_bins["Uncertain (<55%)"].append(match)
                
        confidence_accuracy = {}
        
        for bin_name, bin_matches in confidence_bins.items():
            if bin_matches:
                bin_accuracy = sum(1 for match in bin_matches if match["correct"]) / len(bin_matches)
                confidence_accuracy[bin_name] = {
                    "accuracy": bin_accuracy,
                    "match_count": len(bin_matches),
                    "correct_count": sum(1 for match in bin_matches if match["correct"])
                }
            else:
                confidence_accuracy[bin_name] = {
                    "accuracy": 0.0,
                    "match_count": 0,
                    "correct_count": 0
                }
                
        return {
            "matches_analyzed": len(df),
            "overall_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            },
            "confidence_accuracy": confidence_accuracy,
            "match_predictions": matches[:100]  # Limit to 100 for brevity
        }
    
    def learn_from_mistakes(
            self, from_date: str, to_date: str,
            model_name: Optional[str] = None) -> Dict:
        """
        Analyze incorrect predictions and retrain model.
        
        Args:
            from_date: Start date for matches (YYYY-MM-DD format)
            to_date: End date for matches (YYYY-MM-DD format)
            model_name: Name of the model to learn from (None for best model)
            
        Returns:
            Dictionary with learning results
        """
        # Backtest to find mistakes
        backtest_results = self.backtest_model(from_date, to_date, model_name)
        
        if "error" in backtest_results:
            return backtest_results
            
        matches = backtest_results.get("match_predictions", [])
        
        # Identify incorrect predictions
        incorrect_matches = [match for match in matches if not match.get("correct")]
        
        if not incorrect_matches:
            logger.info("No incorrect predictions found")
            return {"message": "No incorrect predictions found to learn from"}
            
        logger.info(f"Found {len(incorrect_matches)} incorrect predictions")
        
        # Analyze patterns in incorrect predictions
        team_loss_count = {}
        map_loss_count = {}
        confidence_loss_count = {
            "Very High (>80%)": 0,
            "High (70-80%)": 0,
            "Moderate (60-70%)": 0,
            "Slight (55-60%)": 0,
            "Uncertain (<55%)": 0
        }
        
        for match in incorrect_matches:
            team1_id = match.get("team1_id")
            team2_id = match.get("team2_id")
            team1_name = match.get("team1_name")
            team2_name = match.get("team2_name")
            
            # Count losses by team
            if match["predicted_winner"] == "team1":
                # Predicted team1 to win but they lost
                team_loss_count[team1_name] = team_loss_count.get(team1_name, 0) + 1
            else:
                # Predicted team2 to win but they lost
                team_loss_count[team2_name] = team_loss_count.get(team2_name, 0) + 1
                
            # Count by confidence
            confidence = match["confidence"]
            
            if confidence > 0.8:
                confidence_loss_count["Very High (>80%)"] += 1
            elif confidence > 0.7:
                confidence_loss_count["High (70-80%)"] += 1
            elif confidence > 0.6:
                confidence_loss_count["Moderate (60-70%)"] += 1
            elif confidence > 0.55:
                confidence_loss_count["Slight (55-60%)"] += 1
            else:
                confidence_loss_count["Uncertain (<55%)"] += 1
                
        # Retrain model with updated dataset
        if model_name is None:
            model_name = self.model_trainer.best_model_name
            
        retraining_results = self.retrain_model(model_name, from_date, to_date)
        
        return {
            "incorrect_predictions": len(incorrect_matches),
            "team_loss_patterns": sorted(team_loss_count.items(), key=lambda x: x[1], reverse=True)[:10],
            "confidence_loss_patterns": confidence_loss_count,
            "retraining_results": retraining_results
        }
    
    def visualize_feature_importance(self, model_name: Optional[str] = None) -> Dict:
        """
        Create visualization of feature importance.
        
        Args:
            model_name: Name of the model (None for best model)
            
        Returns:
            Dictionary with visualization data
        """
        # Get feature importance
        importance = self.model_trainer.get_feature_importance(model_name)
        
        if not importance:
            return {"error": "No feature importance available"}
            
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Create visualization data
        features = [feature for feature, _ in sorted_features[:15]]
        values = [value for _, value in sorted_features[:15]]
        
        return {
            "features": features,
            "values": values,
            "model": model_name or self.model_trainer.best_model_name
        }
    
    def plot_confidence_accuracy(self, backtest_results: Dict) -> Dict:
        """
        Create plot data for confidence vs. accuracy.
        
        Args:
            backtest_results: Results from backtesting
            
        Returns:
            Dictionary with plot data
        """
        confidence_accuracy = backtest_results.get("confidence_accuracy", {})
        
        if not confidence_accuracy:
            return {"error": "No confidence accuracy data available"}
            
        # Create plot data
        bins = []
        accuracies = []
        counts = []
        
        for bin_name, data in confidence_accuracy.items():
            bins.append(bin_name)
            accuracies.append(data.get("accuracy", 0.0))
            counts.append(data.get("match_count", 0))
            
        return {
            "bins": bins,
            "accuracies": accuracies,
            "counts": counts
        }
    
    def predict_match_score(
            self, team1_id: str, team2_id: str, 
            format_type: str = "bo3") -> Dict:
        """
        Predict match score for a specific match format.
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            format_type: Match format (bo1, bo3, bo5)
            
        Returns:
            Dictionary with score prediction
        """
        # Get maps for Valorant
        valorant_maps = [
            "Ascent", "Bind", "Haven", "Split", "Icebox", 
            "Breeze", "Fracture", "Pearl", "Lotus", "Sunset"
        ]
        
        # Number of maps to be played
        if format_type == "bo1":
            max_maps = 1
        elif format_type == "bo3":
            max_maps = 3
        elif format_type == "bo5":
            max_maps = 5
        else:
            return {"error": f"Unsupported format: {format_type}"}
            
        # Get team map preferences
        team1_map_perfs = {map_name: self.feature_engineering.get_map_performance(team1_id, map_name)
                          for map_name in valorant_maps}
        team2_map_perfs = {map_name: self.feature_engineering.get_map_performance(team2_id, map_name)
                          for map_name in valorant_maps}
        
        # Simulate map picks based on preferences
        team1_pick = max(team1_map_perfs.items(), key=lambda x: x[1].get("win_rate", 0.0))[0]
        team2_pick = max(team2_map_perfs.items(), key=lambda x: x[1].get("win_rate", 0.0))[0]
        
        # For bo3/bo5, select additional maps
        if format_type in ["bo3", "bo5"]:
            # Get remaining maps sorted by win rate for each team
            team1_maps = sorted(team1_map_perfs.items(), key=lambda x: x[1].get("win_rate", 0.0), reverse=True)
            team2_maps = sorted(team2_map_perfs.items(), key=lambda x: x[1].get("win_rate", 0.0), reverse=True)
            
            # Select additional maps (simplified pick/ban)
            maps_to_play = [team1_pick, team2_pick]
            
            for i in range(max_maps - 2):
                # Alternate picks
                if i % 2 == 0:
                    # Team 1 pick
                    for map_name, _ in team1_maps:
                        if map_name not in maps_to_play:
                            maps_to_play.append(map_name)
                            break
                else:
                    # Team 2 pick
                    for map_name, _ in team2_maps:
                        if map_name not in maps_to_play:
                            maps_to_play.append(map_name)
                            break
                            
            maps_to_play = maps_to_play[:max_maps]
        else:
            # For bo1, just use team1's pick (arbitrary)
            maps_to_play = [team1_pick]
            
        # Predict each map
        map_predictions = []
        team1_wins = 0
        team2_wins = 0
        
        for map_name in maps_to_play:
            features = self.feature_engineering.extract_features_for_match(team1_id, team2_id, map_name)
            prediction = self.model_trainer.predict_match(features)
            
            winner = prediction.get("predicted_winner")
            
            if winner == "team1":
                team1_wins += 1
            else:
                team2_wins += 1
                
            map_predictions.append({
                "map": map_name,
                "predicted_winner": winner,
                "team1_win_probability": prediction.get("team1_win_probability"),
                "team2_win_probability": prediction.get("team2_win_probability"),
                "confidence": prediction.get("confidence")
            })
            
            # Check if match is already decided
            if team1_wins > max_maps // 2 or team2_wins > max_maps // 2:
                break
                
        # Determine overall match winner and score
        match_winner = "team1" if team1_wins > team2_wins else "team2"
        match_score = f"{team1_wins}-{team2_wins}"
        
        # Get team info
        team1_info = self.data_collector.get_team_info(team1_id)
        team2_info = self.data_collector.get_team_info(team2_id)
        
        team1_name = team1_info.get("name", "Team 1")
        team2_name = team2_info.get("name", "Team 2")
        
        return {
            "team1_id": team1_id,
            "team2_id": team2_id,
            "team1_name": team1_name,
            "team2_name": team2_name,
            "format": format_type,
            "maps_to_play": maps_to_play,
            "map_predictions": map_predictions,
            "match_winner": match_winner,
            "match_score": match_score,
            "win_margin": abs(team1_wins - team2_wins)
        }
    
    def track_prediction_accuracy(self, save_to_file: bool = True) -> Dict:
        """
        Track prediction accuracy over time.
        
        Args:
            save_to_file: Whether to save results to a file
            
        Returns:
            Dictionary with tracking results
        """
        # Define date ranges
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date_30d = (datetime.datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        start_date_90d = (datetime.datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        # Backtest for different time periods
        results_30d = self.backtest_model(start_date_30d, end_date)
        results_90d = self.backtest_model(start_date_90d, end_date)
        
        # Extract metrics
        metrics_30d = results_30d.get("overall_metrics", {})
        metrics_90d = results_90d.get("overall_metrics", {})
        
        tracking_results = {
            "last_30_days": {
                "matches_analyzed": results_30d.get("matches_analyzed", 0),
                "accuracy": metrics_30d.get("accuracy", 0.0),
                "precision": metrics_30d.get("precision", 0.0),
                "recall": metrics_30d.get("recall", 0.0),
                "f1": metrics_30d.get("f1", 0.0)
            },
            "last_90_days": {
                "matches_analyzed": results_90d.get("matches_analyzed", 0),
                "accuracy": metrics_90d.get("accuracy", 0.0),
                "precision": metrics_90d.get("precision", 0.0),
                "recall": metrics_90d.get("recall", 0.0),
                "f1": metrics_90d.get("f1", 0.0)
            },
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to file
        if save_to_file:
            tracking_file = "prediction_accuracy_tracking.json"
            
            existing_data = []
            
            if os.path.exists(tracking_file):
                try:
                    with open(tracking_file, "r") as f:
                        existing_data = json.load(f)
                except Exception as e:
                    logger.error(f"Error reading tracking file: {e}")
            
            existing_data.append(tracking_results)
            
            try:
                with open(tracking_file, "w") as f:
                    json.dump(existing_data, f, indent=2)
                    
                logger.info(f"Saved tracking results to {tracking_file}")
            except Exception as e:
                logger.error(f"Error saving tracking file: {e}")
                
        return tracking_results


def main():
    """
    Main function to demonstrate the Valorant Predictor System.
    """
    # Create system
    system = ValorantPredictorSystem()
    
    # Check if models exist
    models_exist = os.path.exists("models") and os.path.exists(os.path.join("models", "best_model.txt"))
    
    if not models_exist:
        # Collect data and train models
        logger.info("Training models for the first time...")
        system.collect_and_train()
    else:
        logger.info("Using existing models")
        
    # Predict upcoming matches
    upcoming_predictions = system.predict_upcoming_matches()
    
    if upcoming_predictions:
        logger.info(f"Predicted {len(upcoming_predictions)} upcoming matches")
        
        # Print sample prediction
        sample = upcoming_predictions[0]
        logger.info(f"Sample prediction: {sample['team1_name']} vs {sample['team2_name']}")
        logger.info(f"Predicted winner: {sample['predicted_winner']} with {sample['confidence']:.2f} confidence")
    else:
        logger.info("No upcoming matches to predict")
        
    # Backtest model
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    logger.info(f"Backtesting model from {start_date} to {end_date}...")
    backtest_results = system.backtest_model(start_date, end_date)
    
    # Print backtest metrics
    metrics = backtest_results.get("overall_metrics", {})
    logger.info(f"Backtest accuracy: {metrics.get('accuracy', 0.0):.4f}")
    
    # Learn from mistakes
    logger.info("Learning from incorrect predictions...")
    learning_results = system.learn_from_mistakes(start_date, end_date)
    
    # Track prediction accuracy
    logger.info("Tracking prediction accuracy...")
    tracking_results = system.track_prediction_accuracy()
    
    # Example specific match prediction
    logger.info("Example specific match prediction:")
    
    # Get sample teams
    teams = system.data_collector.get_all_teams(limit=10)
    
    if len(teams) >= 2:
        team1_id = teams[0].get("id")
        team2_id = teams[1].get("id")
        
        if team1_id and team2_id:
            # Predict match
            match_prediction = system.predict_specific_match(
                team1_id, team2_id,
                team1_odds=1.8, team2_odds=2.2,
                handicap_odds={"team1_plus": 1.5, "team1_minus": 2.5},
                over_under_odds={"over": 1.9, "under": 1.9}
            )
            
            # Print prediction
            logger.info(f"Match prediction: {match_prediction['team1_name']} vs {match_prediction['team2_name']}")
            logger.info(f"Predicted winner: {match_prediction['predicted_winner']} with {match_prediction['confidence']:.2f} confidence")
            
            if "betting_advice" in match_prediction:
                recommendations = match_prediction["betting_advice"].get("recommendations", [])
                
                if recommendations:
                    logger.info("Betting recommendations:")
                    
                    for rec in recommendations:
                        logger.info(f"- {rec['recommendation']}")
                        
            # Predict match score (Bo3)
            score_prediction = system.predict_match_score(team1_id, team2_id, "bo3")
            
            logger.info(f"Score prediction: {score_prediction['match_score']}")
            
    logger.info("Done!")


if __name__ == "__main__":
    main()
