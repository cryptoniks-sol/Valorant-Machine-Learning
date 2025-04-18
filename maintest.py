# Valorant Match Prediction System
# A comprehensive ML-based prediction tool for Valorant esports matches

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

class ValorantDataCollector:
    """
    Handles all API interactions and data collection for Valorant match data
    """
    
    def __init__(self, base_url: str = "https://api.example.com/valorant"):
        """
        Initialize the data collector with API endpoints
        
        Args:
            base_url: Base URL for the Valorant API
        """
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "ValorantPredictionScript/1.0"
        }
    
    def get_team_data(self, team_id: str) -> Dict:
        """
        Collect basic team statistics
        
        Args:
            team_id: Unique identifier for the team
            
        Returns:
            Dictionary containing team statistics
        """
        endpoint = f"{self.base_url}/api/v1/teams/{team_id}"
        response = requests.get(endpoint, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching team data: {response.status_code}")
            return {}
    
    def get_match_history(self, team_id: str, limit: int = 50) -> List[Dict]:
        """
        Collect match history for a specific team
        
        Args:
            team_id: Unique identifier for the team
            limit: Maximum number of matches to retrieve
            
        Returns:
            List of match history dictionaries
        """
        endpoint = f"{self.base_url}/api/v1/match-history/{team_id}?limit={limit}"
        response = requests.get(endpoint, headers=self.headers)
        
        if response.status_code == 200:
            return response.json().get("matches", [])
        else:
            print(f"Error fetching match history: {response.status_code}")
            return []
    
    def get_match_details(self, match_id: str) -> Dict:
        """
        Collect detailed statistics for a specific match
        
        Args:
            match_id: Unique identifier for the match
            
        Returns:
            Dictionary containing detailed match information
        """
        endpoint = f"{self.base_url}/api/v1/match-details/{match_id}"
        response = requests.get(endpoint, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching match details: {response.status_code}")
            return {}
    
    def get_player_data(self, player_id: str) -> Dict:
        """
        Collect player statistics
        
        Args:
            player_id: Unique identifier for the player
            
        Returns:
            Dictionary containing player statistics
        """
        endpoint = f"{self.base_url}/api/v1/players/{player_id}"
        response = requests.get(endpoint, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching player data: {response.status_code}")
            return {}
    
    def get_player_agent_stats(self, player_id: str, agent: str) -> Dict:
        """
        Collect player statistics for a specific agent
        
        Args:
            player_id: Unique identifier for the player
            agent: Agent name to filter stats by
            
        Returns:
            Dictionary containing player-agent specific statistics
        """
        endpoint = f"{self.base_url}/api/v1/players/{player_id}?agent={agent}"
        response = requests.get(endpoint, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching player-agent data: {response.status_code}")
            return {}


class DataProcessor:
    """
    Processes raw API data into features for machine learning
    """
    
    def __init__(self, data_collector: ValorantDataCollector):
        """
        Initialize the data processor
        
        Args:
            data_collector: Instance of ValorantDataCollector for API interactions
        """
        self.data_collector = data_collector
        
    def calculate_win_rate(self, wins: int, total_matches: int) -> float:
        """
        Calculate win rate percentage
        
        Args:
            wins: Number of wins
            total_matches: Total number of matches played
            
        Returns:
            Win rate as a float between 0 and 1
        """
        if total_matches == 0:
            return 0.0
        return wins / total_matches
    
    def calculate_average_score(self, match_history: List[Dict]) -> Tuple[float, float]:
        """
        Calculate average score and opponent score from match history
        
        Args:
            match_history: List of match dictionaries
            
        Returns:
            Tuple of (average team score, average opponent score)
        """
        if not match_history:
            return 0.0, 0.0
            
        team_scores = []
        opponent_scores = []
        
        for match in match_history:
            team_scores.append(match.get("team_score", 0))
            opponent_scores.append(match.get("opponent_score", 0))
        
        return np.mean(team_scores), np.mean(opponent_scores)
    
    def calculate_recent_form(self, match_history: List[Dict], num_matches: int = 5) -> float:
        """
        Calculate recent form based on latest matches
        
        Args:
            match_history: List of match dictionaries
            num_matches: Number of recent matches to consider
            
        Returns:
            Recent form as a win percentage
        """
        if not match_history:
            return 0.0
            
        recent_matches = match_history[:min(num_matches, len(match_history))]
        wins = sum(1 for match in recent_matches if match.get("result") == "win")
        
        return self.calculate_win_rate(wins, len(recent_matches))
    
    def calculate_map_win_rates(self, match_history: List[Dict]) -> Dict[str, float]:
        """
        Calculate win rates for each map
        
        Args:
            match_history: List of match dictionaries
            
        Returns:
            Dictionary mapping map names to win rates
        """
        map_stats = {}
        
        for match in match_history:
            map_name = match.get("map")
            result = match.get("result")
            
            if map_name not in map_stats:
                map_stats[map_name] = {"wins": 0, "total": 0}
                
            map_stats[map_name]["total"] += 1
            if result == "win":
                map_stats[map_name]["wins"] += 1
        
        # Convert to win rates
        return {
            map_name: self.calculate_win_rate(stats["wins"], stats["total"])
            for map_name, stats in map_stats.items()
        }
    
    def get_head_to_head_stats(self, team1_id: str, team2_id: str) -> Dict:
        """
        Calculate head-to-head statistics between two teams
        
        Args:
            team1_id: ID of first team
            team2_id: ID of second team
            
        Returns:
            Dictionary with head-to-head statistics
        """
        team1_matches = self.data_collector.get_match_history(team1_id)
        
        # Filter matches against team2
        h2h_matches = [
            match for match in team1_matches
            if match.get("opponent_id") == team2_id
        ]
        
        if not h2h_matches:
            return {
                "total_matches": 0,
                "team1_wins": 0,
                "team2_wins": 0,
                "team1_win_rate": 0.0,
                "team2_win_rate": 0.0,
                "map_stats": {}
            }
        
        team1_wins = sum(1 for match in h2h_matches if match.get("result") == "win")
        team2_wins = len(h2h_matches) - team1_wins
        
        # Map statistics
        map_stats = {}
        for match in h2h_matches:
            map_name = match.get("map")
            if map_name not in map_stats:
                map_stats[map_name] = {"team1_wins": 0, "team2_wins": 0, "total": 0}
                
            map_stats[map_name]["total"] += 1
            if match.get("result") == "win":
                map_stats[map_name]["team1_wins"] += 1
            else:
                map_stats[map_name]["team2_wins"] += 1
        
        return {
            "total_matches": len(h2h_matches),
            "team1_wins": team1_wins,
            "team2_wins": team2_wins,
            "team1_win_rate": self.calculate_win_rate(team1_wins, len(h2h_matches)),
            "team2_win_rate": self.calculate_win_rate(team2_wins, len(h2h_matches)),
            "map_stats": map_stats
        }
    
    def get_time_window_stats(self, match_history: List[Dict], window_size: int) -> Dict:
        """
        Calculate statistics for a specific time window
        
        Args:
            match_history: List of match dictionaries
            window_size: Number of matches to include in the window
            
        Returns:
            Dictionary with time window statistics
        """
        if not match_history:
            return {
                "win_rate": 0.0,
                "avg_score": 0.0,
                "avg_opponent_score": 0.0,
                "score_differential": 0.0
            }
            
        window = match_history[:min(window_size, len(match_history))]
        wins = sum(1 for match in window if match.get("result") == "win")
        
        avg_score, avg_opp_score = self.calculate_average_score(window)
        
        return {
            "win_rate": self.calculate_win_rate(wins, len(window)),
            "avg_score": avg_score,
            "avg_opponent_score": avg_opp_score,
            "score_differential": avg_score - avg_opp_score
        }
    
    def calculate_recency_weighted_stats(self, match_history: List[Dict], decay_factor: float = 0.85) -> Dict:
        """
        Calculate recency-weighted performance metrics
        
        Args:
            match_history: List of match dictionaries
            decay_factor: Factor by which to decay importance (0-1)
            
        Returns:
            Dictionary with recency-weighted statistics
        """
        if not match_history:
            return {
                "weighted_win_rate": 0.0,
                "weighted_score_diff": 0.0
            }
        
        total_weight = 0
        weighted_wins = 0
        weighted_score_diff = 0
        
        for i, match in enumerate(match_history):
            weight = decay_factor ** i
            total_weight += weight
            
            if match.get("result") == "win":
                weighted_wins += weight
            
            score_diff = match.get("team_score", 0) - match.get("opponent_score", 0)
            weighted_score_diff += score_diff * weight
        
        return {
            "weighted_win_rate": weighted_wins / total_weight if total_weight > 0 else 0.0,
            "weighted_score_diff": weighted_score_diff / total_weight if total_weight > 0 else 0.0
        }
    
    def extract_attack_defense_stats(self, match_id: str) -> Dict:
        """
        Extract attack/defense round statistics from match details
        
        Args:
            match_id: ID of the match to analyze
            
        Returns:
            Dictionary with attack/defense statistics
        """
        match_details = self.data_collector.get_match_details(match_id)
        
        if not match_details:
            return {
                "attack_rounds_won": 0,
                "attack_rounds_played": 0,
                "defense_rounds_won": 0,
                "defense_rounds_played": 0,
                "attack_win_rate": 0.0,
                "defense_win_rate": 0.0
            }
        
        # Extract round data - this structure will depend on the actual API format
        rounds = match_details.get("rounds", [])
        
        attack_rounds_played = sum(1 for round in rounds if round.get("side") == "attack")
        attack_rounds_won = sum(1 for round in rounds if round.get("side") == "attack" and round.get("won"))
        
        defense_rounds_played = sum(1 for round in rounds if round.get("side") == "defense")
        defense_rounds_won = sum(1 for round in rounds if round.get("side") == "defense" and round.get("won"))
        
        return {
            "attack_rounds_won": attack_rounds_won,
            "attack_rounds_played": attack_rounds_played,
            "defense_rounds_won": defense_rounds_won,
            "defense_rounds_played": defense_rounds_played,
            "attack_win_rate": self.calculate_win_rate(attack_rounds_won, attack_rounds_played),
            "defense_win_rate": self.calculate_win_rate(defense_rounds_won, defense_rounds_played)
        }
    
    def get_team_player_stats(self, team_id: str) -> Dict[str, Dict]:
        """
        Get aggregated player statistics for a team
        
        Args:
            team_id: ID of the team
            
        Returns:
            Dictionary mapping player IDs to their statistics
        """
        team_data = self.data_collector.get_team_data(team_id)
        player_ids = team_data.get("player_ids", [])
        
        player_stats = {}
        for player_id in player_ids:
            player_data = self.data_collector.get_player_data(player_id)
            
            if player_data:
                player_stats[player_id] = {
                    "acs": player_data.get("acs", 0),
                    "kd_ratio": player_data.get("kd_ratio", 0),
                    "headshot_percentage": player_data.get("headshot_percentage", 0),
                    "agents": self._get_player_agents(player_id)
                }
        
        return player_stats
    
    def _get_player_agents(self, player_id: str) -> Dict[str, Dict]:
        """
        Get agent-specific statistics for a player
        
        Args:
            player_id: ID of the player
            
        Returns:
            Dictionary mapping agent names to performance statistics
        """
        # This method would need to determine which agents a player uses
        # and then fetch agent-specific statistics
        # The implementation depends on how the API structures this data
        
        # For now, we'll return a dummy structure
        return {
            "Jett": {"matches": 0, "win_rate": 0.0, "acs": 0},
            "Reyna": {"matches": 0, "win_rate": 0.0, "acs": 0},
            # Other agents would be populated here
        }
    
    def calculate_roster_stability(self, team_id: str, match_history: List[Dict]) -> float:
        """
        Calculate roster stability based on consistent lineups
        
        Args:
            team_id: ID of the team
            match_history: List of match dictionaries
            
        Returns:
            Roster stability score between 0 and 1
        """
        if not match_history or len(match_history) < 2:
            return 1.0  # Default to stable if not enough history
            
        # This implementation would depend on match_history containing lineup information
        # For now, we'll use a placeholder implementation
        
        unique_lineups = set()
        for match in match_history[:5]:  # Look at last 5 matches
            lineup = tuple(sorted(match.get("lineup", [])))
            unique_lineups.add(lineup)
        
        # More unique lineups = less stability
        stability = 1 - ((len(unique_lineups) - 1) / 4)  # Normalize to 0-1
        return max(0, min(1, stability))  # Clamp between 0 and 1
    
    def create_match_feature_vector(self, team1_id: str, team2_id: str) -> Dict:
        """
        Create a complete feature vector for a match between two teams
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            
        Returns:
            Dictionary containing all features for the match
        """
        # Get basic team data
        team1_data = self.data_collector.get_team_data(team1_id)
        team2_data = self.data_collector.get_team_data(team2_id)
        
        # Get match histories
        team1_history = self.data_collector.get_match_history(team1_id)
        team2_history = self.data_collector.get_match_history(team2_id)
        
        # Calculate basic stats
        team1_wins = team1_data.get("wins", 0)
        team1_losses = team1_data.get("losses", 0)
        team1_total = team1_wins + team1_losses
        
        team2_wins = team2_data.get("wins", 0)
        team2_losses = team2_data.get("losses", 0)
        team2_total = team2_wins + team2_losses
        
        # Win rates
        team1_win_rate = self.calculate_win_rate(team1_wins, team1_total)
        team2_win_rate = self.calculate_win_rate(team2_wins, team2_total)
        
        # Average scores
        team1_avg_score, team1_avg_opp_score = self.calculate_average_score(team1_history)
        team2_avg_score, team2_avg_opp_score = self.calculate_average_score(team2_history)
        
        # Recent form
        team1_recent_form = self.calculate_recent_form(team1_history)
        team2_recent_form = self.calculate_recent_form(team2_history)
        
        # Map win rates
        team1_map_rates = self.calculate_map_win_rates(team1_history)
        team2_map_rates = self.calculate_map_win_rates(team2_history)
        
        # Head-to-head stats
        h2h_stats = self.get_head_to_head_stats(team1_id, team2_id)
        
        # Time window stats
        team1_5_match_stats = self.get_time_window_stats(team1_history, 5)
        team1_10_match_stats = self.get_time_window_stats(team1_history, 10)
        team1_20_match_stats = self.get_time_window_stats(team1_history, 20)
        
        team2_5_match_stats = self.get_time_window_stats(team2_history, 5)
        team2_10_match_stats = self.get_time_window_stats(team2_history, 10)
        team2_20_match_stats = self.get_time_window_stats(team2_history, 20)
        
        # Recency weighted stats
        team1_recency_stats = self.calculate_recency_weighted_stats(team1_history)
        team2_recency_stats = self.calculate_recency_weighted_stats(team2_history)
        
        # Player stats
        team1_player_stats = self.get_team_player_stats(team1_id)
        team2_player_stats = self.get_team_player_stats(team2_id)
        
        # Calculate average player statistics
        team1_avg_acs = np.mean([stats.get("acs", 0) for stats in team1_player_stats.values()])
        team1_avg_kd = np.mean([stats.get("kd_ratio", 0) for stats in team1_player_stats.values()])
        
        team2_avg_acs = np.mean([stats.get("acs", 0) for stats in team2_player_stats.values()])
        team2_avg_kd = np.mean([stats.get("kd_ratio", 0) for stats in team2_player_stats.values()])
        
        # Roster stability
        team1_stability = self.calculate_roster_stability(team1_id, team1_history)
        team2_stability = self.calculate_roster_stability(team2_id, team2_history)
        
        # Compile all features
        features = {
            # Basic stats
            "team1_win_rate": team1_win_rate,
            "team2_win_rate": team2_win_rate,
            "team1_total_matches": team1_total,
            "team2_total_matches": team2_total,
            
            # Score stats
            "team1_avg_score": team1_avg_score,
            "team1_avg_opponent_score": team1_avg_opp_score,
            "team1_score_differential": team1_avg_score - team1_avg_opp_score,
            
            "team2_avg_score": team2_avg_score,
            "team2_avg_opponent_score": team2_avg_opp_score,
            "team2_score_differential": team2_avg_score - team2_avg_opp_score,
            
            # Form
            "team1_recent_form": team1_recent_form,
            "team2_recent_form": team2_recent_form,
            
            # Rankings/Ratings (if available)
            "team1_ranking": team1_data.get("ranking", 0),
            "team2_ranking": team2_data.get("ranking", 0),
            "team1_rating": team1_data.get("rating", 0),
            "team2_rating": team2_data.get("rating", 0),
            
            # Head-to-head
            "h2h_total_matches": h2h_stats["total_matches"],
            "h2h_team1_win_rate": h2h_stats["team1_win_rate"],
            "h2h_team2_win_rate": h2h_stats["team2_win_rate"],
            
            # Time window stats - 5 matches
            "team1_5m_win_rate": team1_5_match_stats["win_rate"],
            "team1_5m_score_diff": team1_5_match_stats["score_differential"],
            "team2_5m_win_rate": team2_5_match_stats["win_rate"],
            "team2_5m_score_diff": team2_5_match_stats["score_differential"],
            
            # Time window stats - 10 matches
            "team1_10m_win_rate": team1_10_match_stats["win_rate"],
            "team1_10m_score_diff": team1_10_match_stats["score_differential"],
            "team2_10m_win_rate": team2_10_match_stats["win_rate"],
            "team2_10m_score_diff": team2_10_match_stats["score_differential"],
            
            # Time window stats - 20 matches
            "team1_20m_win_rate": team1_20_match_stats["win_rate"],
            "team1_20m_score_diff": team1_20_match_stats["score_differential"],
            "team2_20m_win_rate": team2_20_match_stats["win_rate"],
            "team2_20m_score_diff": team2_20_match_stats["score_differential"],
            
            # Recency weighted stats
            "team1_weighted_win_rate": team1_recency_stats["weighted_win_rate"],
            "team1_weighted_score_diff": team1_recency_stats["weighted_score_diff"],
            "team2_weighted_win_rate": team2_recency_stats["weighted_win_rate"],
            "team2_weighted_score_diff": team2_recency_stats["weighted_score_diff"],
            
            # Player stats
            "team1_avg_acs": team1_avg_acs,
            "team1_avg_kd": team1_avg_kd,
            "team2_avg_acs": team2_avg_acs,
            "team2_avg_kd": team2_avg_kd,
            
            # Roster stability
            "team1_roster_stability": team1_stability,
            "team2_roster_stability": team2_stability,
            
            # Map win rates would be included per map
            # We'll add a few example maps
            "team1_win_rate_haven": team1_map_rates.get("Haven", 0.0),
            "team1_win_rate_bind": team1_map_rates.get("Bind", 0.0),
            "team1_win_rate_ascent": team1_map_rates.get("Ascent", 0.0),
            
            "team2_win_rate_haven": team2_map_rates.get("Haven", 0.0),
            "team2_win_rate_bind": team2_map_rates.get("Bind", 0.0),
            "team2_win_rate_ascent": team2_map_rates.get("Ascent", 0.0),
        }
        
        return features


class MatchPredictor:
    """
    ML model implementation for predicting match outcomes
    """
    
    def __init__(self, data_processor: DataProcessor):
        """
        Initialize the match predictor
        
        Args:
            data_processor: Instance of DataProcessor
        """
        self.data_processor = data_processor
        self.feature_scaler = StandardScaler()
        self.rf_model = None
        self.xgb_model = None
        self.nn_model = None
        self.ensemble_weights = [0.4, 0.4, 0.2]  # Default weights for ensemble
        
    def prepare_training_data(self, match_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training
        
        Args:
            match_data: List of dictionaries with match data and outcomes
            
        Returns:
            Tuple of (feature matrix, label vector)
        """
        features = []
        labels = []
        
        for match in match_data:
            team1_id = match["team1_id"]
            team2_id = match["team2_id"]
            
            # Generate features for this match
            match_features = self.data_processor.create_match_feature_vector(team1_id, team2_id)
            
            # Convert to list in a consistent order
            feature_values = [match_features[key] for key in sorted(match_features.keys())]
            features.append(feature_values)
            
            # Get the outcome (1 for team1 win, 0 for team2 win)
            labels.append(1 if match["winner_id"] == team1_id else 0)
        
        return np.array(features), np.array(labels)
    
    def preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess features with scaling
        
        Args:
            X: Feature matrix
            
        Returns:
            Preprocessed feature matrix
        """
        return self.feature_scaler.transform(X)
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train multiple ML models on the provided data
        
        Args:
            X: Feature matrix
            y: Label vector
        """
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # 1. Random Forest
        print("Training Random Forest model...")
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=5,
            scoring='accuracy'
        )
        rf_grid.fit(X_train, y_train)
        self.rf_model = rf_grid.best_estimator_
        
        rf_val_preds = self.rf_model.predict(X_val)
        rf_accuracy = accuracy_score(y_val, rf_val_preds)
        print(f"Random Forest validation accuracy: {rf_accuracy:.4f}")
        
        # 2. XGBoost
        print("Training XGBoost model...")
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=42),
            xgb_params,
            cv=5,
            scoring='accuracy'
        )
        xgb_grid.fit(X_train, y_train)
        self.xgb_model = xgb_grid.best_estimator_
        
        xgb_val_preds = self.xgb_model.predict(X_val)
        xgb_accuracy = accuracy_score(y_val, xgb_val_preds)
        print(f"XGBoost validation accuracy: {xgb_accuracy:.4f}")
        
        # 3. Neural Network
        print("Training Neural Network model...")
        input_dim = X_train.shape[1]
        
        self.nn_model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.nn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossent',
            metrics=['accuracy']
        )
        
        # Train with early stopping
        self.nn_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        
        nn_val_preds = (self.nn_model.predict(X_val) > 0.5).astype(int).flatten()
        nn_accuracy = accuracy_score(y_val, nn_val_preds)
        print(f"Neural Network validation accuracy: {nn_accuracy:.4f}")
        
        # Adjust ensemble weights based on validation performance
        total_accuracy = rf_accuracy + xgb_accuracy + nn_accuracy
        self.ensemble_weights = [
            rf_accuracy / total_accuracy,
            xgb_accuracy / total_accuracy,
            nn_accuracy / total_accuracy
        ]
        print(f"Ensemble weights set to: {self.ensemble_weights}")
    
    def predict_match(self, team1_id: str, team2_id: str) -> Dict:
        """
        Predict the outcome of a match between two teams
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            
        Returns:
            Dictionary with prediction results
        """
        # Generate features for this match
        match_features = self.data_processor.create_match_feature_vector(team1_id, team2_id)
        
        # Convert to appropriate format
        feature_values = [match_features[key] for key in sorted(match_features.keys())]
        X = np.array([feature_values])
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Make predictions with each model
        rf_prob = self.rf_model.predict_proba(X_scaled)[0, 1]
        xgb_prob = self.xgb_model.predict_proba(X_scaled)[0, 1]
        nn_prob = self.nn_model.predict(X_scaled)[0, 0]
        
        # Ensemble prediction
        ensemble_prob = (
            rf_prob * self.ensemble_weights[0] +
            xgb_prob * self.ensemble_weights[1] +
            nn_prob * self.ensemble_weights[2]
        )
        
        # Calculate feature importance (from Random Forest)
        feature_names = sorted(match_features.keys())
        importances = self.rf_model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        
        top_features = [
            {"name": feature_names[i], "importance": float(importances[i])}
            for i in sorted_indices[:10]  # Top 10 features
        ]
        
        # Determine winner
        winner_id = team1_id if ensemble_prob >= 0.5 else team2_id
        
        # Generate confidence level
        confidence_level = "Low"
        if abs(ensemble_prob - 0.5) > 0.3:
            confidence_level = "High"
        elif abs(ensemble_prob - 0.5) > 0.15:
            confidence_level = "Medium"
        
        # Predict map count (for best-of-3 series)
        # Simplistic approach: higher win probability = higher chance of 2-0
        if ensemble_prob >= 0.7 or ensemble_prob <= 0.3:
            map_count = "2-0"
        else:
            map_count = "2-1"
        
        team1_win_prob = ensemble_prob
        team2_win_prob = 1 - ensemble_prob
        
        return {
            "team1_id": team1_id,
            "team2_id": team2_id,
            "team1_win_probability": float(team1_win_prob),
            "team2_win_probability": float(team2_win_prob),
            "predicted_winner": winner_id,
            "confidence_level": confidence_level,
            "predicted_map_count": map_count,
            "model_probabilities": {
                "random_forest": float(rf_prob),
                "xgboost": float(xgb_prob),
                "neural_network": float(nn_prob),
                "ensemble": float(ensemble_prob)
            },
            "top_features": top_features
        }
    
    def predict_map_outcomes(self, team1_id: str, team2_id: str, maps: List[str]) -> List[Dict]:
        """
        Predict outcomes for specific maps
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            maps: List of map names to predict
            
        Returns:
            List of dictionaries with map prediction results
        """
        # This would ideally use map-specific models
        # For now, we'll use the main model with map win rates as a weighting factor
        
        match_features = self.data_processor.create_match_feature_vector(team1_id, team2_id)
        map_predictions = []
        
        for map_name in maps:
            # Get map-specific win rates
            team1_map_rate = match_features.get(f"team1_win_rate_{map_name.lower()}", 0.5)
            team2_map_rate = match_features.get(f"team2_win_rate_{map_name.lower()}", 0.5)
            
            # Make a base prediction
            base_prediction = self.predict_match(team1_id, team2_id)
            
            # Adjust based on map win rates
            map_factor = (team1_map_rate - team2_map_rate) * 0.2  # Scale the influence
            
            adjusted_prob = min(1.0, max(0.0, base_prediction["team1_win_probability"] + map_factor))
            
            map_predictions.append({
                "map": map_name,
                "team1_win_probability": adjusted_prob,
                "team2_win_probability": 1 - adjusted_prob,
                "predicted_winner": team1_id if adjusted_prob >= 0.5 else team2_id
            })
        
        return map_predictions
    
    def save_models(self, filename: str) -> None:
        """
        Save trained models to file
        
        Args:
            filename: Base filename to save models
        """
        # Save Random Forest
        with open(f"{filename}_rf.pkl", "wb") as f:
            pickle.dump(self.rf_model, f)
        
        # Save XGBoost
        with open(f"{filename}_xgb.pkl", "wb") as f:
            pickle.dump(self.xgb_model, f)
        
        # Save Neural Network
        self.nn_model.save(f"{filename}_nn.h5")
        
        # Save scaler
        with open(f"{filename}_scaler.pkl", "wb") as f:
            pickle.dump(self.feature_scaler, f)
        
        # Save ensemble weights
        with open(f"{filename}_ensemble.pkl", "wb") as f:
            pickle.dump(self.ensemble_weights, f)
        
        print(f"All models saved to {filename}_*.pkl/h5")
    
    def load_models(self, filename: str) -> None:
        """
        Load trained models from file
        
        Args:
            filename: Base filename to load models from
        """
        # Load Random Forest
        with open(f"{filename}_rf.pkl", "rb") as f:
            self.rf_model = pickle.load(f)
        
        # Load XGBoost
        with open(f"{filename}_xgb.pkl", "rb") as f:
            self.xgb_model = pickle.load(f)
        
        # Load Neural Network
        self.nn_model = keras.models.load_model(f"{filename}_nn.h5")
        
        # Load scaler
        with open(f"{filename}_scaler.pkl", "rb") as f:
            self.feature_scaler = pickle.load(f)
        
        # Load ensemble weights
        with open(f"{filename}_ensemble.pkl", "rb") as f:
            self.ensemble_weights = pickle.load(f)
        
        print(f"All models loaded from {filename}_*.pkl/h5")


class BettingAnalyzer:
    """
    Analyzes betting opportunities based on model predictions
    """
    
    def __init__(self, match_predictor: MatchPredictor):
        """
        Initialize the betting analyzer
        
        Args:
            match_predictor: Instance of MatchPredictor
        """
        self.match_predictor = match_predictor
    
    def calculate_implied_probability(self, odds: float) -> float:
        """
        Calculate implied probability from decimal odds
        
        Args:
            odds: Decimal odds (e.g., 2.50)
            
        Returns:
            Implied probability as a float between 0 and 1
        """
        return 1 / odds
    
    def identify_value_bets(self, 
                           team1_id: str, 
                           team2_id: str, 
                           team1_odds: float, 
                           team2_odds: float, 
                           team1_handicap_odds: float = None,
                           team2_handicap_odds: float = None,
                           over_maps_odds: float = None,
                           under_maps_odds: float = None) -> Dict:
        """
        Identify value betting opportunities
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            team1_odds: Decimal odds for team1 to win
            team2_odds: Decimal odds for team2 to win
            team1_handicap_odds: Decimal odds for team1 +1.5 maps (Bo3)
            team2_handicap_odds: Decimal odds for team2 +1.5 maps (Bo3)
            over_maps_odds: Decimal odds for over 2.5 maps (Bo3)
            under_maps_odds: Decimal odds for under 2.5 maps (Bo3)
            
        Returns:
            Dictionary with value betting analysis
        """
        # Get model prediction
        prediction = self.match_predictor.predict_match(team1_id, team2_id)
        
        # Calculate implied probabilities from odds
        team1_implied_prob = self.calculate_implied_probability(team1_odds)
        team2_implied_prob = self.calculate_implied_probability(team2_odds)
        
        # Calculate value
        team1_value = prediction["team1_win_probability"] - team1_implied_prob
        team2_value = prediction["team2_win_probability"] - team2_implied_prob
        
        # Map count analysis
        map_count_value = {}
        if prediction["predicted_map_count"] == "2-0":
            # Under 2.5 maps is likely a good bet
            if under_maps_odds:
                under_implied_prob = self.calculate_implied_probability(under_maps_odds)
                # Probability of 2-0 is higher when win probability is extreme
                under_model_prob = 0.7 if abs(prediction["team1_win_probability"] - 0.5) > 0.2 else 0.5
                map_count_value["under_2.5_maps"] = {
                    "value": under_model_prob - under_implied_prob,
                    "model_probability": under_model_prob,
                    "implied_probability": under_implied_prob,
                    "recommendation": "Bet" if under_model_prob > under_implied_prob else "No Bet"
                }
        else:  # 2-1 prediction
            # Over 2.5 maps is likely a good bet
            if over_maps_odds:
                over_implied_prob = self.calculate_implied_probability(over_maps_odds)
                # Probability of 2-1 is higher when win probability is close to 50%
                over_model_prob = 0.7 if abs(prediction["team1_win_probability"] - 0.5) < 0.2 else 0.5
                map_count_value["over_2.5_maps"] = {
                    "value": over_model_prob - over_implied_prob,
                    "model_probability": over_model_prob,
                    "implied_probability": over_implied_prob,
                    "recommendation": "Bet" if over_model_prob > over_implied_prob else "No Bet"
                }
        
        # Handicap analysis
        handicap_value = {}
        if team1_handicap_odds and team2_handicap_odds:
            team1_handicap_implied = self.calculate_implied_probability(team1_handicap_odds)
            team2_handicap_implied = self.calculate_implied_probability(team2_handicap_odds)
            
            # Calculate handicap probabilities from the model
            # For team1 +1.5, they need to win at least one map
            team1_plus_prob = prediction["team1_win_probability"] + (
                prediction["team2_win_probability"] * 0.4  # 40% chance of winning a map if losing match
            )
            
            # For team2 +1.5, they need to win at least one map
            team2_plus_prob = prediction["team2_win_probability"] + (
                prediction["team1_win_probability"] * 0.4  # 40% chance of winning a map if losing match
            )
            
            handicap_value["team1_plus_1.5"] = {
                "value": team1_plus_prob - team1_handicap_implied,
                "model_probability": team1_plus_prob,
                "implied_probability": team1_handicap_implied,
                "recommendation": "Bet" if team1_plus_prob > team1_handicap_implied else "No Bet"
            }
            
            handicap_value["team2_plus_1.5"] = {
                "value": team2_plus_prob - team2_handicap_implied,
                "model_probability": team2_plus_prob,
                "implied_probability": team2_handicap_implied,
                "recommendation": "Bet" if team2_plus_prob > team2_handicap_implied else "No Bet"
            }
        
        return {
            "match_winner": {
                "team1": {
                    "odds": team1_odds,
                    "implied_probability": team1_implied_prob,
                    "model_probability": prediction["team1_win_probability"],
                    "value": team1_value,
                    "recommendation": "Bet" if team1_value > 0.05 else "No Bet"
                },
                "team2": {
                    "odds": team2_odds,
                    "implied_probability": team2_implied_prob,
                    "model_probability": prediction["team2_win_probability"],
                    "value": team2_value,
                    "recommendation": "Bet" if team2_value > 0.05 else "No Bet"
                }
            },
            "handicap": handicap_value,
            "map_count": map_count_value
        }
    
    def calculate_kelly_criterion(self, probability: float, odds: float, fraction: float = 1.0) -> float:
        """
        Calculate Kelly Criterion bet size
        
        Args:
            probability: Probability of winning
            odds: Decimal odds offered
            fraction: Fraction of full Kelly to use (typically 0.1-0.5)
            
        Returns:
            Recommended bet size as percentage of bankroll
        """
        b = odds - 1  # Convert decimal odds to b odds
        q = 1 - probability  # Probability of losing
        
        # Kelly formula: f* = (bp - q) / b
        if b * probability > q:
            f = (b * probability - q) / b
            return f * fraction
        else:
            return 0.0
    
    def generate_betting_report(self, 
                              team1_id: str, 
                              team2_id: str, 
                              team1_odds: float, 
                              team2_odds: float,
                              team1_handicap_odds: float = None,
                              team2_handicap_odds: float = None,
                              over_maps_odds: float = None,
                              under_maps_odds: float = None) -> Dict:
        """
        Generate a comprehensive betting report
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            team1_odds: Decimal odds for team1 to win
            team2_odds: Decimal odds for team2 to win
            team1_handicap_odds: Decimal odds for team1 +1.5 maps (Bo3)
            team2_handicap_odds: Decimal odds for team2 +1.5 maps (Bo3)
            over_maps_odds: Decimal odds for over 2.5 maps (Bo3)
            under_maps_odds: Decimal odds for under 2.5 maps (Bo3)
            
        Returns:
            Dictionary with comprehensive betting report
        """
        # Get value bets
        value_analysis = self.identify_value_bets(
            team1_id, team2_id, 
            team1_odds, team2_odds,
            team1_handicap_odds, team2_handicap_odds,
            over_maps_odds, under_maps_odds
        )
        
        # Get model prediction
        prediction = self.match_predictor.predict_match(team1_id, team2_id)
        
        # Calculate Kelly bets
        kelly_bets = {}
        
        # Team 1 winner
        team1_value = value_analysis["match_winner"]["team1"]
        if team1_value["recommendation"] == "Bet":
            kelly_bets["team1_winner"] = {
                "kelly_percentage": self.calculate_kelly_criterion(
                    prediction["team1_win_probability"], team1_odds, 0.25
                ) * 100,  # Convert to percentage
                "odds": team1_odds
            }
        
        # Team 2 winner
        team2_value = value_analysis["match_winner"]["team2"]
        if team2_value["recommendation"] == "Bet":
            kelly_bets["team2_winner"] = {
                "kelly_percentage": self.calculate_kelly_criterion(
                    prediction["team2_win_probability"], team2_odds, 0.25
                ) * 100,  # Convert to percentage
                "odds": team2_odds
            }
        
        # Handicap and map count bets
        for bet_type in ["handicap", "map_count"]:
            for bet_name, bet_info in value_analysis.get(bet_type, {}).items():
                if bet_info["recommendation"] == "Bet":
                    odds = None
                    if bet_name == "team1_plus_1.5" and team1_handicap_odds:
                        odds = team1_handicap_odds
                    elif bet_name == "team2_plus_1.5" and team2_handicap_odds:
                        odds = team2_handicap_odds
                    elif bet_name == "over_2.5_maps" and over_maps_odds:
                        odds = over_maps_odds
                    elif bet_name == "under_2.5_maps" and under_maps_odds:
                        odds = under_maps_odds
                    
                    if odds:
                        kelly_bets[bet_name] = {
                            "kelly_percentage": self.calculate_kelly_criterion(
                                bet_info["model_probability"], odds, 0.25
                            ) * 100,  # Convert to percentage
                            "odds": odds
                        }
        
        # Find the best bet
        best_bet = None
        max_value = -1
        
        for bet_type in ["match_winner", "handicap", "map_count"]:
            for bet_name, bet_info in value_analysis.get(bet_type, {}).items():
                if isinstance(bet_info, dict) and "value" in bet_info:
                    value = bet_info["value"]
                    if value > max_value and bet_info.get("recommendation") == "Bet":
                        max_value = value
                        if bet_type == "match_winner":
                            best_bet = f"{bet_name} to win"
                        else:
                            best_bet = bet_name.replace("_", " ")
        
        return {
            "prediction": prediction,
            "value_analysis": value_analysis,
            "kelly_bets": kelly_bets,
            "best_bet": best_bet,
            "confidence": prediction["confidence_level"]
        }


class ValorantPredictionSystem:
    """
    Main class for the Valorant prediction system
    """
    
    def __init__(self, api_base_url: str = "https://api.example.com/valorant"):
        """
        Initialize the prediction system
        
        Args:
            api_base_url: Base URL for the Valorant API
        """
        self.data_collector = ValorantDataCollector(api_base_url)
        self.data_processor = DataProcessor(self.data_collector)
        self.match_predictor = MatchPredictor(self.data_processor)
        self.betting_analyzer = BettingAnalyzer(self.match_predictor)
        
    def train_system(self, match_data_path: str) -> None:
        """
        Train the prediction system with historical match data
        
        Args:
            match_data_path: Path to JSON file with historical match data
        """
        # Load match data
        with open(match_data_path, 'r') as f:
            match_data = json.load(f)
        
        # Prepare data for training
        X, y = self.match_predictor.prepare_training_data(match_data)
        
        # Train models
        self.match_predictor.train_models(X, y)
        
        print("System training complete")
    
    def save_system(self, base_path: str) -> None:
        """
        Save the trained system
        
        Args:
            base_path: Base path to save models and data
        """
        self.match_predictor.save_models(base_path)
    
    def load_system(self, base_path: str) -> None:
        """
        Load a trained system
        
        Args:
            base_path: Base path to load models and data from
        """
        self.match_predictor.load_models(base_path)
    
    def predict_match(self, team1_id: str, team2_id: str) -> Dict:
        """
        Predict the outcome of a match
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            
        Returns:
            Dictionary with prediction results
        """
        return self.match_predictor.predict_match(team1_id, team2_id)
    
    def analyze_betting_opportunities(self, 
                                    team1_id: str, 
                                    team2_id: str, 
                                    team1_odds: float, 
                                    team2_odds: float,
                                    team1_handicap_odds: float = None,
                                    team2_handicap_odds: float = None,
                                    over_maps_odds: float = None,
                                    under_maps_odds: float = None) -> Dict:
        """
        Analyze betting opportunities for a match
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            team1_odds: Decimal odds for team1 to win
            team2_odds: Decimal odds for team2 to win
            team1_handicap_odds: Decimal odds for team1 +1.5 maps (Bo3)
            team2_handicap_odds: Decimal odds for team2 +1.5 maps (Bo3)
            over_maps_odds: Decimal odds for over 2.5 maps (Bo3)
            under_maps_odds: Decimal odds for under 2.5 maps (Bo3)
            
        Returns:
            Dictionary with betting analysis
        """
        return self.betting_analyzer.generate_betting_report(
            team1_id, team2_id,
            team1_odds, team2_odds,
            team1_handicap_odds, team2_handicap_odds,
            over_maps_odds, under_maps_odds
        )


def visualize_prediction(prediction: Dict) -> None:
    """
    Visualize prediction results
    
    Args:
        prediction: Dictionary with prediction results
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot win probabilities
    plt.subplot(2, 2, 1)
    probs = [prediction["team1_win_probability"], prediction["team2_win_probability"]]
    plt.bar(["Team 1", "Team 2"], probs, color=['blue', 'orange'])
    plt.title("Win Probability")
    plt.ylim(0, 1)
    
    # Plot model probabilities
    plt.subplot(2, 2, 2)
    model_probs = prediction["model_probabilities"]
    plt.bar(model_probs.keys(), model_probs.values())
    plt.title("Model Predictions")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Plot feature importance
    plt.subplot(2, 1, 2)
    top_features = prediction["top_features"]
    feature_names = [f["name"] for f in top_features]
    importances = [f["importance"] for f in top_features]
    
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, importances, align='center')
    plt.yticks(y_pos, feature_names)
    plt.title("Feature Importance")
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to demonstrate the Valorant prediction system
    """
    print("Initializing Valorant Match Prediction System...")
    
    # Initialize system
    system = ValorantPredictionSystem()
    
    # Example usage
    print("\n1. Train the system with historical data")
    print("2. Load a pre-trained system")
    print("3. Make a prediction for a specific match")
    print("4. Analyze betting opportunities")
    
    choice = input("Select an option (1-4): ")
    
    if choice == "1":
        data_path = input("Enter path to historical match data JSON: ")
        system.train_system(data_path)
        
        save_choice = input("Save trained system? (y/n): ")
        if save_choice.lower() == "y":
            save_path = input("Enter base path to save system: ")
            system.save_system(save_path)
    
    elif choice == "2":
        load_path = input("Enter base path to load system from: ")
        system.load_system(load_path)
    
    elif choice == "3":
        team1_id = input("Enter ID for team 1: ")
        team2_id = input("Enter ID for team 2: ")
        
        prediction = system.predict_match(team1_id, team2_id)
        
        print("\nPrediction Results:")
        print(f"Team 1 Win Probability: {prediction['team1_win_probability']:.4f}")
        print(f"Team 2 Win Probability: {prediction['team2_win_probability']:.4f}")
        print(f"Predicted Winner: {'Team 1' if prediction['predicted_winner'] == team1_id else 'Team 2'}")
        print(f"Confidence Level: {prediction['confidence_level']}")
        print(f"Predicted Map Count: {prediction['predicted_map_count']}")
        
        vis_choice = input("Visualize prediction? (y/n): ")
        if vis_choice.lower() == "y":
            visualize_prediction(prediction)
    
    elif choice == "4":
        team1_id = input("Enter ID for team 1: ")
        team2_id = input("Enter ID for team 2: ")
        
        team1_odds = float(input("Enter decimal odds for team 1 to win: "))
        team2_odds = float(input("Enter decimal odds for team 2 to win: "))
        
        handicap_choice = input("Include handicap odds? (y/n): ")
        team1_handicap_odds = None
        team2_handicap_odds = None
        over_maps_odds = None
        under_maps_odds = None
        
        if handicap_choice.lower() == "y":
            team1_handicap_odds = float(input("Enter decimal odds for team 1 +1.5 maps: "))
            team2_handicap_odds = float(input("Enter decimal odds for team 2 +1.5 maps: "))
            over_maps_odds = float(input("Enter decimal odds for over 2.5 maps: "))
            under_maps_odds = float(input("Enter decimal odds for under 2.5 maps: "))
        
        betting_report = system.analyze_betting_opportunities(
            team1_id, team2_id,
            team1_odds, team2_odds,
            team1_handicap_odds, team2_handicap_odds,
            over_maps_odds, under_maps_odds
        )
        
        print("\nBetting Analysis:")
        print(f"Match Prediction: {'Team 1' if betting_report['prediction']['predicted_winner'] == team1_id else 'Team 2'} to win")
        print(f"Confidence: {betting_report['confidence']}")
        
        if betting_report["best_bet"]:
            print(f"Best Betting Opportunity: {betting_report['best_bet']}")
        else:
            print("No value bets found")
        
        print("\nKelly Criterion Bet Sizing:")
        for bet_name, bet_info in betting_report["kelly_bets"].items():
            print(f"- {bet_name}: {bet_info['kelly_percentage']:.2f}% of bankroll @ {bet_info['odds']:.2f} odds")


if __name__ == "__main__":
    main()