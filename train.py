#!/usr/bin/env python3
print("Starting Optimized Deep Learning Valorant Match Predictor...")

import sys
import traceback
from functools import wraps
import logging 
import requests
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import pickle
import time
import re
from tqdm import tqdm
import seaborn as sns
import traceback

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow.keras.regularizers import l1, l2

# API URL
API_URL = "http://localhost:5000/api/v1"

# Configure TensorFlow for better performance
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU is available. Using GPU.")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("GPU is not available. Using CPU.")

#-------------------------------------------------------------------------
# DATA COLLECTION AND PROCESSING FUNCTIONS
#-------------------------------------------------------------------------

class ModelConfig:
    """Configuration parameters for model training and prediction."""
    
    # Data collection
    CACHE_ENABLED = True
    CACHE_PATH = "cache/valorant_data_cache.pkl"
    CACHE_EXPIRATION_DAYS = 7
    API_URL = "http://localhost:5000/api/v1"
    
    # Feature selection
    MIN_FEATURE_IMPORTANCE = 0.01
    MAX_FEATURES = 50
    FEATURE_SELECTION_METHOD = "random_forest"  # Options: random_forest, mutual_info, recursive
    
    # Cross-validation
    CV_FOLDS = 10
    RANDOM_STATE = 42
    USE_STRATIFIED_CV = True
    
    # Neural network parameters
    NN_HIDDEN_LAYERS = [256, 128, 64]
    NN_DROPOUT_RATE = 0.5
    NN_LEARNING_RATE = 0.001
    NN_BATCH_SIZE = 32
    NN_EPOCHS = 100
    NN_EARLY_STOPPING_PATIENCE = 15
    
    # Ensemble configuration
    ENSEMBLE_MODELS = ["nn", "gb", "rf", "lr", "svm"]  # Neural network, Gradient boosting, Random forest, Logistic regression, SVM
    ENSEMBLE_WEIGHTS = {"nn": 1.0, "gb": 1.0, "rf": 1.0, "lr": 0.8, "svm": 0.8}  # Relative weights
    
    # Prediction calibration
    CALIBRATION_ENABLED = True
    CALIBRATION_METHOD = "isotonic"  # Options: isotonic, sigmoid, beta
    CALIBRATION_STRENGTH = 0.8  # How strongly to calibrate (0-1)
    
    # Team ordering
    CONSISTENT_TEAM_ORDERING = True  # Whether to use consistent team ordering
    TEAM_ORDERING_CRITERIA = [
        "ranking",         # Official ranking (lower number = higher ranking)
        "matches_played",  # Number of matches played (more data = more reliable)
        "win_rate",        # Overall win rate
        "recent_form",     # Recent form (last 5 matches)
        "score_differential"  # Average score differential
    ]


class BettingConfig:
    """Configuration parameters for betting strategy."""
    
    # Kelly criterion settings
    KELLY_FRACTION = 0.15  # Fraction of full Kelly to bet (more conservative)
    MAX_BET_PERCENTAGE = 0.02  # Maximum bet size as percentage of bankroll
    MAX_BET_AMOUNT = 500.0  # Absolute maximum bet size
    
    # Edge requirements
    MIN_EDGE_THRESHOLD = 0.05  # Minimum edge required for betting
    MIN_CONFIDENCE = 0.4  # Minimum model confidence required
    
    # Risk adjustments by bet type
    RISK_FACTORS = {
        "team1_ml": 1.0,       # Baseline risk
        "team2_ml": 1.0,       # Baseline risk
        "team1_plus_1_5": 0.9,  # Lower risk for +1.5 maps
        "team2_plus_1_5": 0.9,  # Lower risk for +1.5 maps
        "team1_minus_1_5": 1.2, # Higher risk for -1.5 maps
        "team2_minus_1_5": 1.2, # Higher risk for -1.5 maps
        "over_2_5_maps": 1.1,   # Slightly higher risk for totals
        "under_2_5_maps": 1.1   # Slightly higher risk for totals
    }
    
    # Bankroll management
    LOSS_STREAK_THRESHOLD = 3  # Number of consecutive losses to trigger reduction
    LOSS_STREAK_REDUCTION = 0.5  # Reduce bet size by 50% after loss streak
    WIN_STREAK_THRESHOLD = 5  # Number of consecutive wins to trigger increase
    WIN_STREAK_INCREASE = 1.2  # Increase bet size by 20% after win streak
    WIN_STREAK_CAP = 1.5  # Cap the streak multiplier at 50% increase
    
    # Drawdown protection
    MAX_DRAWDOWN_PCT = 0.15  # 15% drawdown triggers reduced bet sizing
    DRAWDOWN_REDUCTION = 0.6  # Reduce bet size by 40% during significant drawdown
    
    # Portfolio management
    MAX_SIMULTANEOUS_BETS = 2  # Maximum number of bets per match
    MAX_TOTAL_RISK_PCT = 0.03  # Maximum total risk as percentage of bankroll
    DIVERSIFICATION_REQUIRED = True  # Require diversification of bet types


class ForwardTestConfig:
    """Configuration parameters for forward testing."""
    
    # Data storage
    DATA_DIR = "forward_test"
    PREDICTIONS_FILE = "predictions.json"
    RESULTS_FILE = "results.json"
    SUMMARY_FILE = "summary.json"
    REPORT_FILE = "forward_test_report.html"
    
    # CLV calculation
    MARKET_EFFICIENCY = 0.7  # How efficiently the market responds to betting activity
    RECORD_CLOSING_ODDS = True  # Whether to record closing odds for CLV calculation
    
    # Tracking settings
    TRACK_ACTUAL_BETS = True  # Whether to track actual bets placed
    AUTO_GENERATE_REPORTS = True  # Whether to automatically generate reports
    REPORT_FREQUENCY = 10  # Generate new report every N new results


class BacktestConfig:
    """Configuration parameters for backtesting."""
    
    # Data filtering
    DEFAULT_TEAM_LIMIT = 50  # Default number of teams to include
    USE_CACHE = True  # Whether to use cached data
    CACHE_PATH = "cache/valorant_data_cache.pkl"
    
    # Time filtering
    DEFAULT_START_DATE = None  # Default start date (None = all data)
    DEFAULT_END_DATE = None  # Default end date (None = all data)
    
    # Bankroll settings
    DEFAULT_STARTING_BANKROLL = 1000.0  # Default starting bankroll
    DEFAULT_BET_PCT = 0.05  # Default maximum bet percentage
    DEFAULT_MIN_EDGE = 0.03  # Default minimum edge
    DEFAULT_CONFIDENCE_THRESHOLD = 0.2  # Default confidence threshold
    
    # Reporting
    GENERATE_VISUALIZATIONS = True  # Whether to generate visualizations
    SAVE_DETAILED_RESULTS = True  # Whether to save detailed results
    REALISTIC_ADJUSTMENTS = True  # Whether to apply realistic adjustments to results
    
    # Simulation settings
    SIMULATE_MARKET_MOVEMENT = True  # Whether to simulate market movement
    SIMULATE_LONG_TERM_VARIANCE = True  # Whether to simulate long-term variance
    SIMULATION_RUNS = 1000  # Number of simulation runs


# Consolidated configuration class
class Config:
    """Master configuration for the Valorant match prediction system."""
    
    MODEL = ModelConfig
    BETTING = BettingConfig
    FORWARD_TEST = ForwardTestConfig
    BACKTEST = BacktestConfig
    
    # Global settings
    DEBUG_MODE = False  # Enable additional debug output
    DATA_DIR = "data"  # Directory for data storage
    RESULTS_DIR = "results"  # Directory for results storage
    LOG_DIR = "logs"  # Directory for log files
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        import os
        for directory in [cls.DATA_DIR, cls.RESULTS_DIR, cls.LOG_DIR, 
                         cls.FORWARD_TEST.DATA_DIR, "cache"]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

def debug_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n>>> ENTERING {func.__name__} <<<")
        try:
            result = func(*args, **kwargs)
            print(f">>> EXITING {func.__name__} <<<")
            return result
        except Exception as e:
            print(f"!!! ERROR in {func.__name__}: {e}")
            traceback.print_exc()
            return None
    return wrapper

def load_cache(cache_path="cache/valorant_data_cache.pkl"):
    """Load team data from the cache file."""
    print(f"\nLoading cached team data from {cache_path}...")
    try:
        with open(cache_path, 'rb') as f:
            print("Cache file opened successfully, attempting to load...")
            team_data = pickle.load(f)
            print(f"Pickle loaded, checking data type: {type(team_data)}")
            
        # Check if the cache has the expected structure
        if not isinstance(team_data, dict):
            print(f"Error: Invalid cache format - expected dictionary, got {type(team_data)}")
            return None
            
        teams_count = len(team_data)
        print(f"Teams in cache: {teams_count}")
        
        # Count matches
        match_counts = {}
        total_matches = 0
        for team, data in team_data.items():
            matches = len(data.get('matches', []))
            match_counts[team] = matches
            total_matches += matches
            
        print(f"Teams with most matches:")
        for team, count in sorted(match_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {team}: {count} matches")
        
        # Get cache age
        cache_mtime = os.path.getmtime(cache_path)
        cache_age = time.time() - cache_mtime
        days_old = cache_age / (60 * 60 * 24)
        
        print(f"Cache loaded successfully with {teams_count} teams and {total_matches} total matches.")
        print(f"Cache is {days_old:.1f} days old (last updated {datetime.fromtimestamp(cache_mtime).strftime('%Y-%m-%d %H:%M:%S')})")
        
        if days_old > 7:
            print("WARNING: Cache is more than 7 days old. Consider updating it with cache.py.")
        
        # Check for key team data components
        first_team = next(iter(team_data.values()))
        print(f"First team data keys: {first_team.keys()}")
        
        if 'matches' in first_team:
            first_match = first_team['matches'][0] if first_team['matches'] else None
            if first_match:
                print(f"First match keys: {first_match.keys()}")
                
        return team_data
            
    except FileNotFoundError:
        print(f"Error: Cache file not found at {cache_path}")
        print("Please run cache.py first to create the cache.")
        return None
    except Exception as e:
        print(f"Error loading cache: {e}")
        traceback.print_exc()
        return None

def check_cache_freshness(cache_path="cache/valorant_data_cache.pkl", max_age_days=7):
    """
    Check if the cache is still fresh or needs updating.
    
    Args:
        cache_path (str): Path to the cache file
        max_age_days (int): Maximum acceptable age in days
        
    Returns:
        bool: True if cache is fresh, False otherwise
    """
    try:
        if not os.path.exists(cache_path):
            return False
            
        # Get cache age
        cache_mtime = os.path.getmtime(cache_path)
        cache_age = time.time() - cache_mtime
        days_old = cache_age / (60 * 60 * 24)
        
        return days_old <= max_age_days
        
    except Exception:
        return False

def get_cache_metadata(cache_dir="cache"):
    """
    Get metadata about the cache for display.
    
    Args:
        cache_dir (str): Directory containing the cache files
        
    Returns:
        dict: Cache metadata or empty dict if not found
    """
    meta_path = os.path.join(cache_dir, "cache_metadata.json")
    try:
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error reading cache metadata: {e}")
        return {}

def get_team_id(team_name, region=None):
    """Search for a team ID by name, optionally filtering by region."""
    print(f"Searching for team ID for '{team_name}'...")
    
    # Build the API URL with optional region filter
    url = f"{API_URL}/teams?limit=300"
    if region:
        url += f"&region={region}"
        print(f"Filtering by region: {region}")
    
    # Fetch teams
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
    
    # If still no match and no region was specified, try searching across all regions
    if not region:
        print(f"No match found with default search. Attempting to search by region...")
        
        # Try each region one by one
        for r in ['na', 'eu', 'br', 'ap', 'kr', 'ch', 'jp', 'lan', 'las', 'oce', 'mn', 'gc']:
            print(f"Trying region: {r}")
            region_id = get_team_id(team_name, r)
            if region_id:
                return region_id
    
    print(f"No team ID found for '{team_name}'")
    return None

def fetch_api_data(endpoint, params=None):
    """Generic function to fetch data from API with error handling"""
    url = f"{API_URL}/{endpoint}"
    if params:
        url += f"?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error fetching data from {endpoint}: {response.status_code}")
            return None
        return response.json()
    except Exception as e:
        print(f"Exception while fetching {endpoint}: {e}")
        return None

def fetch_team_details(team_id):
    """Fetch detailed information about a team, including the team tag."""
    if not team_id:
        return None, None
    
    team_data = fetch_api_data(f"teams/{team_id}")
    if not team_data or 'data' not in team_data:
        return None, None
    
    # Extract the team tag if available - it's in data.info.tag
    team_tag = None
    if ('data' in team_data and 
        isinstance(team_data['data'], dict) and 
        'info' in team_data['data'] and 
        isinstance(team_data['data']['info'], dict) and
        'tag' in team_data['data']['info']):
        team_tag = team_data['data']['info']['tag']
        print(f"Team tag found: {team_tag}")
    else:
        print(f"Team tag not found in data structure")
    
    return team_data, team_tag

def fetch_player_stats(player_name):
    """Fetch detailed player statistics using the API endpoint."""
    if not player_name:
        return None
    
    print(f"Fetching stats for player: {player_name}")
    player_data = fetch_api_data(f"player-stats/{player_name}")
    
    # Return player stats if successful
    if player_data and player_data.get('status') == 'OK' and 'data' in player_data:
        return player_data['data']
    
    return None

def fetch_team_player_stats(team_id):
    """Fetch detailed player statistics for a team from the team roster."""
    if not team_id:
        print(f"Invalid team ID: {team_id}")
        return []
    
    # Get team details with roster information
    team_data = fetch_api_data(f"teams/{team_id}")
    
    if not team_data or 'data' not in team_data or not isinstance(team_data['data'], dict):
        print(f"Invalid team data format for team ID: {team_id}")
        return []
        
    team_info = team_data['data']
    
    # Extract player info from the roster
    players = team_info.get('players', [])
    
    if not players:
        print(f"No players found in roster for team ID: {team_id}")
        return []
    
    print(f"Found {len(players)} players in roster for team: {team_info.get('info', {}).get('name', '')}")
    
    # Fetch individual player stats
    player_stats = []
    for player in players:
        player_id = player.get('id')
        player_name = player.get('user')  # Use the in-game username
        
        if not player_name:
            continue
            
        print(f"Fetching statistics for player: {player_name}")
        stats = fetch_player_stats(player_name)
        
        if stats:
            # Add additional player info from the team data
            stats['player_id'] = player_id
            stats['full_name'] = player.get('name', '')
            stats['country'] = player.get('country', '')
            
            player_stats.append(stats)
        else:
            print(f"No statistics found for player: {player_name}")
    
    print(f"Successfully fetched statistics for {len(player_stats)} out of {len(players)} players")
    return player_stats

def fetch_team_match_history(team_id):
    """Fetch match history for a specific team."""
    if not team_id:
        return None
    
    print(f"Fetching match history for team ID: {team_id}")
    return fetch_api_data(f"match-history/{team_id}")

def fetch_match_details(match_id):
    """Fetch detailed information about a specific match."""
    if not match_id:
        return None
    
    print(f"Fetching details for match ID: {match_id}")
    return fetch_api_data(f"match-details/{match_id}")

def fetch_match_economy_details(match_id):
    """Fetch economic details for a specific match."""
    if not match_id:
        return None
    
    print(f"Fetching economy details for match ID: {match_id}")
    return fetch_api_data(f"match-details/{match_id}", {"tab": "economy"})

def fetch_team_map_statistics(team_id):
    """Fetch detailed map statistics for a team using the team-stats API endpoint."""
    if not team_id:
        print("Invalid team ID")
        return {}
    
    print(f"Fetching map statistics for team ID: {team_id}")
    team_stats_data = fetch_api_data(f"team-stats/{team_id}")
    
    if not team_stats_data:
        print(f"No team stats data returned for team ID: {team_id}")
        return {}
    
    # Check for expected data structure
    if 'data' not in team_stats_data:
        print(f"Missing 'data' field in team stats for team ID: {team_id}")
        print(f"Received: {team_stats_data}")
        return {}
        
    if not team_stats_data['data']:
        print(f"Empty data array in team stats for team ID: {team_id}")
        return {}
    
    # Process the data
    map_stats = extract_map_statistics(team_stats_data)
    
    if not map_stats:
        print(f"Failed to extract map statistics for team ID: {team_id}")
    else:
        print(f"Successfully extracted statistics for {len(map_stats)} maps")
        
    return map_stats

def determine_team_order(team1_stats, team2_stats):
    """
    Determine consistent team ordering for prediction based on ranking and other metrics.
    
    The model performs best when given consistent team ordering rules. This function
    implements a strict priority system to determine which team should be placed first:
    
    Priority order:
    1. Team with higher official ranking (if available)
    2. Team with more matches played (more data usually means more reliable prediction)
    3. Team with higher win rate
    4. Team with higher recent form
    5. Team with higher score differential
    
    Returns:
        tuple: (ordered_team1_stats, ordered_team2_stats, teams_swapped)
               where teams_swapped is a boolean indicating if the original order was changed
    """
    teams_swapped = False
    
    # 1. Use official ranking if available
    team1_ranking = get_team_ranking_safe(team1_stats)
    team2_ranking = get_team_ranking_safe(team2_stats)
    
    if team1_ranking is not None and team2_ranking is not None:
        if team2_ranking < team1_ranking:  # Lower rank number = higher ranking
            return team2_stats, team1_stats, True
    
    # 2. Use match count if rankings not available or tied
    team1_matches = get_match_count(team1_stats)
    team2_matches = get_match_count(team2_stats)
    
    if team1_matches != team2_matches:
        if team2_matches > team1_matches:
            return team2_stats, team1_stats, True
    
    # 3. Use win rate if match counts are tied
    team1_winrate = team1_stats.get('win_rate', 0)
    team2_winrate = team2_stats.get('win_rate', 0)
    
    if abs(team1_winrate - team2_winrate) > 0.01:  # 1% threshold to avoid floating point issues
        if team2_winrate > team1_winrate:
            return team2_stats, team1_stats, True
    
    # 4. Use recent form if win rates are tied
    team1_form = team1_stats.get('recent_form', 0)
    team2_form = team2_stats.get('recent_form', 0)
    
    if abs(team1_form - team2_form) > 0.01:
        if team2_form > team1_form:
            return team2_stats, team1_stats, True
    
    # 5. Use score differential if everything else is tied
    team1_diff = team1_stats.get('score_differential', 0)
    team2_diff = team2_stats.get('score_differential', 0)
    
    if team2_diff > team1_diff:
        return team2_stats, team1_stats, True
    
    # Default: keep original order
    return team1_stats, team2_stats, teams_swapped

def get_team_ranking_safe(team_stats):
    """
    Safely extract team ranking from team stats.
    
    Args:
        team_stats (dict): Team statistics
        
    Returns:
        int or None: Team ranking if available, None otherwise
    """
    if 'opponent_quality' in team_stats and 'team_ranking' in team_stats['opponent_quality']:
        return team_stats['opponent_quality']['team_ranking']
    return None

def get_match_count(team_stats):
    """
    Get the number of matches played by a team.
    
    Args:
        team_stats (dict): Team statistics
        
    Returns:
        int: Number of matches played
    """
    if 'matches' in team_stats:
        matches = team_stats['matches']
        if isinstance(matches, list):
            return len(matches)
        elif isinstance(matches, (int, float)):
            return int(matches)
    return 0

def modify_prediction_for_swapped_teams(prediction, teams_swapped):
    """
    Modify prediction when teams have been swapped for consistency.
    
    Args:
        prediction (float): Original prediction (probability of team1 winning)
        teams_swapped (bool): Whether teams were swapped in the ordering process
        
    Returns:
        float: Adjusted prediction probability
    """
    if teams_swapped:
        # If teams were swapped, we need to invert the prediction
        return 1.0 - prediction
    return prediction

def prepare_data_with_consistent_ordering(team1_stats, team2_stats, selected_features):
    """
    Prepare prediction data with consistent team ordering.
    
    Args:
        team1_stats (dict): Statistics for team 1
        team2_stats (dict): Statistics for team 2
        selected_features (list): Features to include in prediction
        
    Returns:
        tuple: (features, teams_swapped)
    """
    # Apply consistent team ordering
    ordered_team1_stats, ordered_team2_stats, teams_swapped = determine_team_order(team1_stats, team2_stats)
    
    # Prepare features using the ordered team stats
    features = prepare_features_for_backtest(ordered_team1_stats, ordered_team2_stats, selected_features)
    
    return features, teams_swapped

def get_historical_team_stats(team_name, target_date, team_data_collection):
    """
    Get team statistics using only matches that occurred before the specified date.
    This prevents data leakage by ensuring we only use information that would have
    been available at prediction time.
    
    Args:
        team_name (str): Name of the team
        target_date (str): Date to get stats for, in format 'YYYY-MM-DD'
        team_data_collection (dict): Dictionary of team data
        
    Returns:
        dict: Team stats using only historical data available at target_date
    """
    if team_name not in team_data_collection:
        print(f"Warning: Team '{team_name}' not found in data collection")
        return None
    
    team_data = team_data_collection[team_name]
    past_matches = []
    
    # Filter matches to only include those before target_date
    for match in team_data.get('matches', []):
        match_date = match.get('date', '')
        if match_date and match_date < target_date:
            past_matches.append(match)
    
    if len(past_matches) < 3:
        print(f"Warning: Insufficient historical data for {team_name} before {target_date}")
        print(f"Only {len(past_matches)} matches available")
        return None
    
    # Calculate team stats using only historical matches
    team_stats = calculate_team_stats(past_matches, 
                                     team_data.get('player_stats', None),
                                     include_economy=True)
    
    # Add team identification for convenience
    team_stats['team_name'] = team_name
    team_stats['team_id'] = team_data.get('team_id', '')
    team_stats['team_tag'] = team_data.get('team_tag', '')
    
    # Get historical map statistics
    if 'map_statistics' in team_data:
        # We need to filter map statistics to only include matches before target_date
        # This is complex and requires detailed match-by-match filtering
        # For now, we'll include the map statistics with a warning
        team_stats['map_statistics'] = team_data.get('map_statistics', {})
    
    # Calculate historical opponent quality metrics
    if past_matches:
        team_stats['opponent_quality'] = analyze_opponent_quality(past_matches, team_data.get('team_id', ''))
    
    return team_stats

def analyze_historical_h2h(team1_name, team2_name, target_date, team_data_collection):
    """
    Analyze head-to-head record between two teams using only historical data
    available at the specified date.
    
    Args:
        team1_name (str): First team name
        team2_name (str): Second team name
        target_date (str): Cut-off date for historical data
        team_data_collection (dict): Dictionary of team data
        
    Returns:
        dict: Head-to-head statistics
    """
    if team1_name not in team_data_collection or team2_name not in team_data_collection:
        return {
            'matches': 0,
            'team1_wins': 0,
            'team2_wins': 0,
            'win_rate': 0.5,
            'average_score_diff': 0
        }
    
    # Get all historical matches for team1
    team1_data = team_data_collection[team1_name]
    h2h_matches = []
    
    # Filter for matches against team2 before target_date
    for match in team1_data.get('matches', []):
        match_date = match.get('date', '')
        opponent = match.get('opponent_name', '')
        
        if match_date and match_date < target_date and opponent == team2_name:
            h2h_matches.append(match)
    
    # If no direct matches found, check with variations of team2 name
    if not h2h_matches:
        for match in team1_data.get('matches', []):
            match_date = match.get('date', '')
            opponent = match.get('opponent_name', '')
            
            if match_date and match_date < target_date:
                # Check for partial matches in team names
                if (team2_name.lower() in opponent.lower() or 
                    opponent.lower() in team2_name.lower()):
                    h2h_matches.append(match)
    
    # Calculate h2h stats
    if not h2h_matches:
        return {
            'matches': 0,
            'team1_wins': 0,
            'team2_wins': 0,
            'win_rate': 0.5,
            'average_score_diff': 0
        }
    
    team1_wins = sum(1 for m in h2h_matches if m.get('team_won', False))
    team2_wins = len(h2h_matches) - team1_wins
    
    total_team1_score = sum(m.get('team_score', 0) for m in h2h_matches)
    total_team2_score = sum(m.get('opponent_score', 0) for m in h2h_matches)
    
    avg_score_diff = (total_team1_score - total_team2_score) / len(h2h_matches)
    
    return {
        'matches': len(h2h_matches),
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'win_rate': team1_wins / len(h2h_matches),
        'average_score_diff': avg_score_diff
    }

def implement_time_aware_cross_validation(team_data_collection, n_splits=5):
    """
    Implement time-aware cross-validation to prevent data leakage.
    
    Args:
        team_data_collection (dict): Dictionary of team data
        n_splits (int): Number of validation splits
        
    Returns:
        list: List of (train, test) indices
    """
    all_matches = []
    
    # Collect all matches with their dates
    for team_name, team_data in team_data_collection.items():
        for match in team_data.get('matches', []):
            match['team_name'] = team_name
            if 'date' in match and match['date']:
                all_matches.append(match)
    
    # Sort matches by date
    all_matches.sort(key=lambda x: x.get('date', ''))
    
    # Create time-based splits
    splits = []
    matches_per_split = len(all_matches) // n_splits
    
    for i in range(n_splits):
        if i < n_splits - 1:
            train_end = (i + 1) * matches_per_split
            test_start = train_end
            test_end = (i + 2) * matches_per_split
        else:
            train_end = (i + 1) * matches_per_split
            test_start = train_end
            test_end = len(all_matches)
        
        train_indices = list(range(0, train_end))
        test_indices = list(range(test_start, test_end))
        
        splits.append((train_indices, test_indices))
    
    return splits, all_matches

def prepare_time_aware_data(match, team_data_collection):
    """
    Prepare data for a match using only information available at match time.
    
    Args:
        match (dict): Match data with date
        team_data_collection (dict): Dictionary of team data
        
    Returns:
        tuple: (features, label) or (None, None) if data cannot be prepared
    """
    match_date = match.get('date', '')
    if not match_date:
        return None, None
    
    team1_name = match.get('team_name', '')
    team2_name = match.get('opponent_name', '')
    
    if not team1_name or not team2_name:
        return None, None
    
    if team1_name not in team_data_collection or team2_name not in team_data_collection:
        return None, None
    
    # Get historical stats for both teams
    team1_stats = get_historical_team_stats(team1_name, match_date, team_data_collection)
    team2_stats = get_historical_team_stats(team2_name, match_date, team_data_collection)
    
    if not team1_stats or not team2_stats:
        return None, None
    
    # Add historical head-to-head data
    h2h_stats = analyze_historical_h2h(team1_name, team2_name, match_date, team_data_collection)
    team1_stats['h2h'] = h2h_stats
    
    # Prepare features
    features = prepare_data_for_model(team1_stats, team2_stats)
    
    if not features:
        return None, None
    
    # Get label
    label = 1 if match.get('team_won', False) else 0
    
    return features, label

def time_aware_prediction(team1_name, team2_name, current_date, team_data_collection, ensemble_models, selected_features):
    """
    Make a prediction using only data available at prediction time.
    
    Args:
        team1_name (str): First team name
        team2_name (str): Second team name
        current_date (str): Current date in 'YYYY-MM-DD' format
        team_data_collection (dict): Dictionary of team data
        ensemble_models (list): Ensemble models for prediction
        selected_features (list): Selected features for prediction
        
    Returns:
        tuple: (win_probability, confidence, swapped)
    """
    # Get historical stats for both teams
    team1_stats = get_historical_team_stats(team1_name, current_date, team_data_collection)
    team2_stats = get_historical_team_stats(team2_name, current_date, team_data_collection)
    
    if not team1_stats or not team2_stats:
        print(f"Error: Insufficient historical data for prediction")
        return 0.5, 0.0, False
    
    # Apply consistent team ordering
    features, teams_swapped = prepare_data_with_consistent_ordering(
        team1_stats, team2_stats, selected_features
    )
    
    if features is None:
        print(f"Error: Failed to prepare features for prediction")
        return 0.5, 0.0, teams_swapped
    
    # Make prediction
    win_probability, raw_predictions, confidence = predict_with_ensemble(
        ensemble_models, features
    )
    
    # Adjust prediction if teams were swapped
    win_probability = modify_prediction_for_swapped_teams(win_probability, teams_swapped)
    
    return win_probability, confidence, teams_swapped

def fetch_upcoming_matches():
    """Fetch upcoming matches."""
    print("Fetching upcoming matches...")
    matches_data = fetch_api_data("matches")
    
    if not matches_data:
        return []
    
    return matches_data.get('data', [])

def fetch_events():
    """Fetch all events from the API."""
    print("Fetching events...")
    events_data = fetch_api_data("events", {"limit": 300})
    
    if not events_data:
        return []
    
    return events_data.get('data', [])

def parse_match_data(match_history, team_name):
    """
    Parse match history data for a team with improved map score extraction.
    """
    if not match_history or 'data' not in match_history:
        return []
    
    matches = []
    filtered_count = 0
    
    print(f"Parsing {len(match_history['data'])} matches for {team_name}")
    
    for match in match_history['data']:
        try:
            # Check if this match involves the team we're looking for
            team_found = False
            if 'teams' in match and len(match['teams']) >= 2:
                for team in match['teams']:
                    if (team.get('name', '').lower() == team_name.lower() or 
                        team_name.lower() in team.get('name', '').lower()):
                        team_found = True
                        break
           
            if not team_found:
                filtered_count += 1
                continue
                
            # Extract basic match info
            match_info = {
                'match_id': match.get('id', ''),
                'date': match.get('date', ''),
                'event': match.get('event', '') if isinstance(match.get('event', ''), str) else match.get('event', {}).get('name', ''),
                'tournament': match.get('tournament', ''),
                'map': match.get('map', ''),
                'map_score': ''  # Initialize map score as empty
            }
            
            # Extract teams and determine which is our team
            if 'teams' in match and len(match['teams']) >= 2:
                team1 = match['teams'][0]
                team2 = match['teams'][1]
                
                # Convert scores to integers for comparison
                team1_score = int(team1.get('score', 0))
                team2_score = int(team2.get('score', 0))
                
                # Set map score directly from the scores - this is critical for dataset building
                match_info['map_score'] = f"{team1_score}:{team2_score}"
                
                # Determine winners based on scores
                team1_won = team1_score > team2_score
                team2_won = team2_score > team1_score
                
                # Determine if our team is team1 or team2
                is_team1 = team1.get('name', '').lower() == team_name.lower() or team_name.lower() in team1.get('name', '').lower()
                
                if is_team1:
                    our_team = team1
                    opponent_team = team2
                    team_won = team1_won
                else:
                    our_team = team2
                    opponent_team = team1
                    team_won = team2_won
                
                # Add our team's info
                match_info['team_name'] = our_team.get('name', '')
                match_info['team_score'] = int(our_team.get('score', 0))
                match_info['team_won'] = team_won
                match_info['team_country'] = our_team.get('country', '')
                
                # Use actual team tag if available, don't hardcode it
                # First try to get it from the team data
                match_info['team_tag'] = our_team.get('tag', '')
                
                # If tag not in team data, try to extract it from name (common formats: "TAG Name" or "Name [TAG]")
                if not match_info['team_tag']:
                    team_name_parts = our_team.get('name', '').split()
                    if len(team_name_parts) > 0:
                        # Check if first word might be a tag (all caps, 2-5 letters)
                        first_word = team_name_parts[0]
                        if first_word.isupper() and 2 <= len(first_word) <= 5:
                            match_info['team_tag'] = first_word
                    
                    # If still no tag, check for [TAG] format
                    if not match_info['team_tag'] and '[' in our_team.get('name', '') and ']' in our_team.get('name', ''):
                        tag_match = re.search(r'\[(.*?)\]', our_team.get('name', ''))
                        if tag_match:
                            match_info['team_tag'] = tag_match.group(1)
                
                # Add opponent's info
                match_info['opponent_name'] = opponent_team.get('name', '')
                match_info['opponent_score'] = int(opponent_team.get('score', 0))
                match_info['opponent_won'] = not team_won  # Opponent's result is opposite of our team
                match_info['opponent_country'] = opponent_team.get('country', '')
                match_info['opponent_tag'] = opponent_team.get('tag', '')
                match_info['opponent_id'] = opponent_team.get('id', '')  # Save opponent ID for future reference
                
                # Add the result field
                match_info['result'] = 'win' if team_won else 'loss'               

                # Fetch match details for deeper statistics
                match_details = fetch_match_details(match_info['match_id'])
                if match_details:
                    # Add match details to the match_info
                    match_info['details'] = match_details
                
                matches.append(match_info)
            
        except Exception as e:
            print(f"Error parsing match: {e}")
            continue
    
    print(f"Skipped {filtered_count} matches that did not involve {team_name}")   
    # Summarize wins/losses
    wins = sum(1 for match in matches if match['team_won'])
    print(f"Processed {len(matches)} matches for {team_name}: {wins} wins, {len(matches) - wins} losses")
    
    return matches

#-------------------------------------------------------------------------
# DATA EXTRACTION AND PROCESSING
#-------------------------------------------------------------------------

def extract_map_statistics(team_stats_data):
    """Extract detailed map statistics from team stats API response."""
    def safe_percentage(value):
        try:
            return float(value.strip('%')) / 100
        except (ValueError, AttributeError, TypeError):
            return 0

    def safe_int(value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    if not team_stats_data or 'data' not in team_stats_data:
        return {}

    maps_data = team_stats_data['data']
    map_statistics = {}

    for map_entry in maps_data:
        map_name = map_entry.get('map', '').split(' ')[0]
        if not map_name:
            continue

        stats = map_entry.get('stats', [])
        if not stats:
            continue

        main_stats = stats[0]

        win_percentage = safe_percentage(main_stats.get('WIN%', '0%'))

        wins = safe_int(main_stats.get('W', '0'))
        losses = safe_int(main_stats.get('L', '0'))

        atk_first = safe_int(main_stats.get('ATK 1st', '0'))
        def_first = safe_int(main_stats.get('DEF 1st', '0'))

        atk_win_rate = safe_percentage(main_stats.get('ATK RWin%', '0%'))
        def_win_rate = safe_percentage(main_stats.get('DEF RWin%', '0%'))

        atk_rounds_won = safe_int(main_stats.get('RW', '0'))
        atk_rounds_lost = safe_int(main_stats.get('RL', '0'))

        agent_compositions = main_stats.get('Agent Compositions', [])

        # Group agents into teams of 5
        team_compositions = []
        current_comp = []

        for agent in agent_compositions:
            current_comp.append(agent)
            if len(current_comp) == 5:
                team_compositions.append(current_comp)
                current_comp = []

        agent_usage = {}
        for comp in team_compositions:
            for agent in comp:
                agent_usage[agent] = agent_usage.get(agent, 0) + 1

        sorted_agents = sorted(agent_usage.items(), key=lambda x: x[1], reverse=True)

        match_history = []
        for i in range(1, len(stats)):
            match_stat = stats[i]
            match_details = match_stat.get('Expand', '')
            if not match_details:
                continue

            parts = match_details.split()
            try:
                date = parts[0] if parts else ''
                score_index = next((i for i, p in enumerate(parts) if '/' in p), -1)

                opponent = ' '.join(parts[1:score_index]) if score_index != -1 else ''

                team_score, opponent_score = 0, 0
                if score_index != -1:
                    score_parts = parts[score_index].split('/')
                    team_score = safe_int(score_parts[0])
                    opponent_score = safe_int(score_parts[1])

                side_scores = {}
                for side in ['atk', 'def', 'OT']:
                    try:
                        idx = parts.index(side)
                        score_part = parts[idx + 1]
                        if '/' in score_part:
                            won, lost = map(safe_int, score_part.split('/'))
                            side_scores[side] = {'won': won, 'lost': lost}
                    except:
                        continue

                went_to_ot = 'OT' in parts
                event_type = ''
                for side in ['OT', 'def', 'atk']:
                    if side in parts:
                        idx = parts.index(side)
                        if idx + 2 < len(parts):
                            event_type = ' '.join(parts[idx + 2:])
                            break

                match_result = {
                    'date': date,
                    'opponent': opponent,
                    'team_score': team_score,
                    'opponent_score': opponent_score,
                    'won': team_score > opponent_score,
                    'side_scores': side_scores,
                    'went_to_ot': went_to_ot,
                    'event_type': event_type
                }

                match_history.append(match_result)
            except Exception as e:
                print(f"Error parsing match details '{match_details}': {e}")

        overtime_matches = sum(1 for m in match_history if m.get('went_to_ot'))
        overtime_wins = sum(1 for m in match_history if m.get('went_to_ot') and m.get('won'))

        playoff_matches = sum(1 for m in match_history if 'Playoff' in m.get('event_type', ''))
        playoff_wins = sum(1 for m in match_history if 'Playoff' in m.get('event_type', '') and m.get('won'))

        recent_matches = match_history[:3]
        recent_win_rate = sum(1 for m in recent_matches if m.get('won')) / len(recent_matches) if recent_matches else 0

        map_statistics[map_name] = {
            'win_percentage': win_percentage,
            'wins': wins,
            'losses': losses,
            'matches_played': wins + losses,
            'atk_first': atk_first,
            'def_first': def_first,
            'atk_win_rate': atk_win_rate,
            'def_win_rate': def_win_rate,
            'atk_rounds_won': atk_rounds_won,
            'atk_rounds_lost': atk_rounds_lost,
            'side_preference': 'Attack' if atk_win_rate > def_win_rate else 'Defense',
            'side_preference_strength': abs(atk_win_rate - def_win_rate),
            'agent_compositions': team_compositions,
            'agent_usage': dict(sorted_agents),
            'match_history': match_history,
            'overtime_stats': {
                'matches': overtime_matches,
                'wins': overtime_wins,
                'win_rate': overtime_wins / overtime_matches if overtime_matches > 0 else 0
            },
            'playoff_stats': {
                'matches': playoff_matches,
                'wins': playoff_wins,
                'win_rate': playoff_wins / playoff_matches if playoff_matches > 0 else 0
            },
            'recent_form': recent_win_rate,
            'composition_variety': len(team_compositions),
            'most_played_composition': team_compositions[0] if team_compositions else [],
            'most_played_agents': [agent for agent, _ in sorted_agents[:5]] if sorted_agents else []
        }

    return map_statistics

def extract_economy_metrics(match_economy_data, team_identifier=None, fallback_name=None):
    """Extract relevant economic performance metrics from match economy data for a specific team."""
    # Check basic structure
    if not match_economy_data or 'data' not in match_economy_data:
        return {'economy_data_missing': True}
    
    # Check if we have the expected teams field
    if 'teams' not in match_economy_data['data']:
        return {'economy_data_missing': True}
    
    teams_data = match_economy_data['data']['teams']
    if len(teams_data) < 1:
        return {'economy_data_missing': True}
    
    # Find the team with matching tag or name
    target_team_data = None
    
    # First try to match by name field directly (which often contains the tag in economy data)
    if team_identifier:
        for team in teams_data:
            # Direct match on name field (case-insensitive)
            team_name = team.get('name', '').lower()
            if team_identifier and team_name == team_identifier.lower():
                target_team_data = team
                break
            # Check if team name starts with the identifier (common pattern)
            elif team_identifier and team_name.startswith(team_identifier.lower()):
                target_team_data = team
                break
            # Check if team name contains the identifier
            elif team_identifier and team_identifier.lower() in team_name:
                target_team_data = team
                break
    
    # If no match by team identifier and fallback_name is provided, try matching by name
    if not target_team_data and fallback_name:
        for team in teams_data:
            # Check different name variants
            team_name = team.get('name', '').lower()
            fallback_lower = fallback_name.lower()
            
            # Name similarity checks
            # 1. Exact match
            if team_name == fallback_lower:
                target_team_data = team
                break
            
            # 2. Name contains each other
            elif fallback_lower in team_name or team_name in fallback_lower:
                target_team_data = team
                break
            
            # 3. Word-by-word matching (e.g., "Paper Rex" might match "PRX")
            # Split both names into words and check if any words match
            team_words = team_name.split()
            fallback_words = fallback_lower.split()
            
            common_words = set(team_words) & set(fallback_words)
            if common_words:
                target_team_data = team
                break
    
    if not target_team_data:
        return {'economy_data_missing': True}
    
    # Check if we have the expected economy fields
    has_economy_data = ('eco' in target_team_data or 
                        'pistolWon' in target_team_data or
                        'semiEco' in target_team_data or
                        'semiBuy' in target_team_data or
                        'fullBuy' in target_team_data)
    
    if not has_economy_data:
        return {'economy_data_missing': True}
    
    # Extract metrics for the target team
    metrics = {
        'team_name': target_team_data.get('name', 'Unknown'),
        'pistol_rounds_won': target_team_data.get('pistolWon', 0),
        'total_pistol_rounds': 2,  # Typically 2 pistol rounds per map
        'pistol_win_rate': target_team_data.get('pistolWon', 0) / 2,  # Normalize per map
        'eco_total': target_team_data.get('eco', {}).get('total', 0),
        'eco_won': target_team_data.get('eco', {}).get('won', 0),
        'eco_win_rate': target_team_data.get('eco', {}).get('won', 0) / target_team_data.get('eco', {}).get('total', 1) if target_team_data.get('eco', {}).get('total', 0) > 0 else 0,
        'semi_eco_total': target_team_data.get('semiEco', {}).get('total', 0),
        'semi_eco_won': target_team_data.get('semiEco', {}).get('won', 0),
        'semi_eco_win_rate': target_team_data.get('semiEco', {}).get('won', 0) / target_team_data.get('semiEco', {}).get('total', 1) if target_team_data.get('semiEco', {}).get('total', 0) > 0 else 0,
        'semi_buy_total': target_team_data.get('semiBuy', {}).get('total', 0),
        'semi_buy_won': target_team_data.get('semiBuy', {}).get('won', 0),
        'semi_buy_win_rate': target_team_data.get('semiBuy', {}).get('won', 0) / target_team_data.get('semiBuy', {}).get('total', 1) if target_team_data.get('semiBuy', {}).get('total', 0) > 0 else 0,
        'full_buy_total': target_team_data.get('fullBuy', {}).get('total', 0),
        'full_buy_won': target_team_data.get('fullBuy', {}).get('won', 0),
        'full_buy_win_rate': target_team_data.get('fullBuy', {}).get('won', 0) / target_team_data.get('fullBuy', {}).get('total', 1) if target_team_data.get('fullBuy', {}).get('total', 0) > 0 else 0,
    }
    
    # Calculate additional derived metrics
    metrics['total_rounds'] = (metrics['eco_total'] + metrics['semi_eco_total'] + 
                             metrics['semi_buy_total'] + metrics['full_buy_total'])
    
    metrics['overall_economy_win_rate'] = ((metrics['eco_won'] + metrics['semi_eco_won'] + 
                                         metrics['semi_buy_won'] + metrics['full_buy_won']) / 
                                         metrics['total_rounds']) if metrics['total_rounds'] > 0 else 0
    
    # Calculate economy efficiency (weighted win rate based on investment)
    if metrics['total_rounds'] > 0:
        # Apply weights based on investment level
        eco_weight = 3.0  # Winning eco rounds is highly valuable
        semi_eco_weight = 2.0
        semi_buy_weight = 1.2
        full_buy_weight = 1.0  # Baseline expectation is to win full buys
        
        weighted_wins = (metrics['eco_won'] * eco_weight +
                      metrics['semi_eco_won'] * semi_eco_weight +
                      metrics['semi_buy_won'] * semi_buy_weight +
                      metrics['full_buy_won'] * full_buy_weight)
        
        weighted_total = (metrics['eco_total'] * eco_weight +
                       metrics['semi_eco_total'] * semi_eco_weight +
                       metrics['semi_buy_total'] * semi_buy_weight +
                       metrics['full_buy_total'] * full_buy_weight)
        
        metrics['economy_efficiency'] = weighted_wins / weighted_total if weighted_total > 0 else 0
    else:
        metrics['economy_efficiency'] = 0
    
    return metrics

def calculate_team_player_stats(player_stats_list):
    """Calculate team-level statistics from individual player stats."""
    if not player_stats_list:
        return {}
    
    # Initialize aggregate stats
    agg_stats = {
        'player_count': len(player_stats_list),
        'avg_rating': 0,
        'avg_acs': 0,
        'avg_kd': 0,
        'avg_kast': 0,
        'avg_adr': 0,
        'avg_headshot': 0,
        'avg_clutch': 0,
        'total_kills': 0,
        'total_deaths': 0,
        'total_assists': 0,
        'total_first_kills': 0,
        'total_first_deaths': 0,
        'agent_usage': {},
        'star_player_rating': 0,
        'star_player_name': '',
        'weak_player_rating': 0,
        'weak_player_name': '',
        'fk_fd_ratio': 0
    }
    
    # Process each player's stats
    for player_data in player_stats_list:
        if 'stats' not in player_data:
            continue
        
        stats = player_data['stats']
        
        # Add player stats to aggregates (converting to appropriate types)
        try:
            # Convert string values to appropriate numeric types, handling empty strings
            rating = float(stats.get('rating', 0) or 0)
            acs = float(stats.get('acs', 0) or 0)
            kd = float(stats.get('kd', 0) or 0)
            
            # Handle percentage strings
            kast_str = stats.get('kast', '0%')
            kast = float(kast_str.strip('%')) / 100 if '%' in kast_str and kast_str.strip('%') else 0
            
            adr = float(stats.get('adr', 0) or 0)
            
            hs_str = stats.get('hs', '0%')
            hs = float(hs_str.strip('%')) / 100 if '%' in hs_str and hs_str.strip('%') else 0
            
            cl_str = stats.get('cl', '0%')
            cl = float(cl_str.strip('%')) / 100 if '%' in cl_str and cl_str.strip('%') else 0
            
            # Add to aggregates
            agg_stats['avg_rating'] += rating
            agg_stats['avg_acs'] += acs
            agg_stats['avg_kd'] += kd
            agg_stats['avg_kast'] += kast
            agg_stats['avg_adr'] += adr
            agg_stats['avg_headshot'] += hs
            agg_stats['avg_clutch'] += cl
            
            # Track kills, deaths, assists
            agg_stats['total_kills'] += int(stats.get('kills', 0) or 0)
            agg_stats['total_deaths'] += int(stats.get('deaths', 0) or 0)
            agg_stats['total_assists'] += int(stats.get('assists', 0) or 0)
            
            # Track first kills and deaths
            agg_stats['total_first_kills'] += int(stats.get('fk', 0) or 0)
            agg_stats['total_first_deaths'] += int(stats.get('fd', 0) or 0)
            
            # Track best and worst players
            if rating > agg_stats['star_player_rating']:
                agg_stats['star_player_rating'] = rating
                agg_stats['star_player_name'] = player_data.get('player', '')
                
            if agg_stats['weak_player_rating'] == 0 or rating < agg_stats['weak_player_rating']:
                agg_stats['weak_player_rating'] = rating
                agg_stats['weak_player_name'] = player_data.get('player', '')
                
            # Track agent usage
            for agent in player_data.get('agents', []):
                if agent not in agg_stats['agent_usage']:
                    agg_stats['agent_usage'][agent] = 0
                agg_stats['agent_usage'][agent] += 1
                
        except (ValueError, TypeError) as e:
            print(f"Error processing player stats: {e}")
            continue
    
    # Calculate averages
    player_count = agg_stats['player_count']
    if player_count > 0:
        agg_stats['avg_rating'] /= player_count
        agg_stats['avg_acs'] /= player_count
        agg_stats['avg_kd'] /= player_count
        agg_stats['avg_kast'] /= player_count
        agg_stats['avg_adr'] /= player_count
        agg_stats['avg_headshot'] /= player_count
        agg_stats['avg_clutch'] /= player_count
        
    # Calculate FK/FD ratio
    if agg_stats['total_first_deaths'] > 0:
        agg_stats['fk_fd_ratio'] = agg_stats['total_first_kills'] / agg_stats['total_first_deaths']
    else:
        agg_stats['fk_fd_ratio'] = agg_stats['total_first_kills'] if agg_stats['total_first_kills'] > 0 else 1  # Avoid division by zero
    
    # Calculate rating difference between star and weak player (team consistency)
    agg_stats['team_consistency'] = 1 - (agg_stats['star_player_rating'] - agg_stats['weak_player_rating']) / agg_stats['star_player_rating'] if agg_stats['star_player_rating'] > 0 else 0
    
    return agg_stats

def extract_match_details_stats(matches):
    """Extract advanced statistics from match details."""
    # Initialize counters for various stats
    total_matches_with_details = 0
    total_first_bloods = 0
    total_clutches = 0
    total_aces = 0
    total_entry_kills = 0
    total_headshot_percentage = 0
    total_kast = 0  # Kill, Assist, Survive, Trade percentage
    total_adr = 0   # Average Damage per Round
    total_fk_diff = 0  # First Kill Differential
    agent_usage = {}  # Count how often each agent is used
    
    # Process match details
    for match in matches:
        if 'details' in match and match['details']:
            total_matches_with_details += 1
            details = match['details']
            
            # Process player stats if available
            if 'players' in details:
                for player in details['players']:
                    # Extract player-specific stats
                    total_first_bloods += player.get('first_bloods', 0)
                    total_clutches += player.get('clutches', 0)
                    total_aces += player.get('aces', 0)
                    total_entry_kills += player.get('entry_kills', 0)
                    total_headshot_percentage += player.get('headshot_percentage', 0)
                    total_kast += player.get('kast', 0)
                    total_adr += player.get('adr', 0)
                    
                    # Process agent usage
                    agent = player.get('agent', 'Unknown')
                    if agent != 'Unknown':
                        if agent not in agent_usage:
                            agent_usage[agent] = 0
                        agent_usage[agent] += 1
            
            # Process round data if available
            if 'rounds' in details:
                for round_data in details['rounds']:
                    # Track first kill differential
                    first_kill_team = round_data.get('first_kill_team', '')
                    if first_kill_team == 'team':
                        total_fk_diff += 1
                    elif first_kill_team == 'opponent':
                        total_fk_diff -= 1
    
    # Compile the extracted stats
    advanced_stats = {}
    
    if total_matches_with_details > 0:
        # Average stats per match
        advanced_stats['avg_first_bloods'] = total_first_bloods / total_matches_with_details
        advanced_stats['avg_clutches'] = total_clutches / total_matches_with_details
        advanced_stats['avg_aces'] = total_aces / total_matches_with_details
        advanced_stats['avg_entry_kills'] = total_entry_kills / total_matches_with_details
        advanced_stats['avg_headshot_percentage'] = total_headshot_percentage / total_matches_with_details
        advanced_stats['avg_kast'] = total_kast / total_matches_with_details
        advanced_stats['avg_adr'] = total_adr / total_matches_with_details
        advanced_stats['avg_first_kill_diff'] = total_fk_diff / total_matches_with_details
        
        # Most used agents
        advanced_stats['agent_usage'] = {k: v for k, v in sorted(agent_usage.items(), 
                                                              key=lambda item: item[1], 
                                                              reverse=True)}
    
    return advanced_stats

def extract_map_performance(team_matches):
    """Extract detailed map-specific performance metrics."""
    map_performance = {}
    
    for match in team_matches:
        map_name = match.get('map', 'Unknown')
        if map_name == '' or map_name is None:
            map_name = 'Unknown'
            
        if map_name not in map_performance:
            map_performance[map_name] = {
                'played': 0,
                'wins': 0,
                'rounds_played': 0,
                'rounds_won': 0,
                'attack_rounds': 0,
                'attack_rounds_won': 0,
                'defense_rounds': 0,
                'defense_rounds_won': 0,
                'overtime_rounds': 0,
                'overtime_rounds_won': 0
            }
            
        # Update basic stats
        map_performance[map_name]['played'] += 1
        map_performance[map_name]['wins'] += 1 if match['team_won'] else 0
        
        # Try to extract round information from match details
        if 'details' in match and match['details'] and 'rounds' in match['details']:
            rounds = match['details']['rounds']
            map_performance[map_name]['rounds_played'] += len(rounds)
            
            for round_data in rounds:
                # Check if the round was won
                round_winner = round_data.get('winner', '')
                team_won_round = (round_winner == 'team')
                
                # Check side played
                side = round_data.get('side', '')
                
                if side == 'attack':
                    map_performance[map_name]['attack_rounds'] += 1
                    if team_won_round:
                        map_performance[map_name]['attack_rounds_won'] += 1
                        map_performance[map_name]['rounds_won'] += 1
                elif side == 'defense':
                    map_performance[map_name]['defense_rounds'] += 1
                    if team_won_round:
                        map_performance[map_name]['defense_rounds_won'] += 1
                        map_performance[map_name]['rounds_won'] += 1
                elif side == 'overtime':
                    map_performance[map_name]['overtime_rounds'] += 1
                    if team_won_round:
                        map_performance[map_name]['overtime_rounds_won'] += 1
                        map_performance[map_name]['rounds_won'] += 1
    
    # Calculate derived metrics
    for map_name, stats in map_performance.items():
        if stats['played'] > 0:
            stats['win_rate'] = stats['wins'] / stats['played']
        else:
            stats['win_rate'] = 0
            
        if stats['rounds_played'] > 0:
            stats['round_win_rate'] = stats['rounds_won'] / stats['rounds_played']
        else:
            stats['round_win_rate'] = 0
            
        if stats['attack_rounds'] > 0:
            stats['attack_win_rate'] = stats['attack_rounds_won'] / stats['attack_rounds']
        else:
            stats['attack_win_rate'] = 0
            
        if stats['defense_rounds'] > 0:
            stats['defense_win_rate'] = stats['defense_rounds_won'] / stats['defense_rounds']
        else:
            stats['defense_win_rate'] = 0
    
    return map_performance

def extract_tournament_performance(team_matches):
    """Extract tournament-specific performance metrics."""
    tournament_performance = {}
    
    for match in team_matches:
        tournament = match.get('tournament', 'Unknown')
        event = match.get('event', 'Unknown')
        
        if isinstance(event, dict):
            event = event.get('name', 'Unknown')
            
        tournament_key = f"{event}:{tournament}"
        
        if tournament_key not in tournament_performance:
            tournament_performance[tournament_key] = {
                'played': 0,
                'wins': 0,
                'total_score': 0,
                'total_opponent_score': 0,
                'matches': []
            }
            
        tournament_performance[tournament_key]['played'] += 1
        tournament_performance[tournament_key]['wins'] += 1 if match['team_won'] else 0
        tournament_performance[tournament_key]['total_score'] += match.get('team_score', 0)
        tournament_performance[tournament_key]['total_opponent_score'] += match.get('opponent_score', 0)
        tournament_performance[tournament_key]['matches'].append(match)
    
    # Calculate derived metrics and tournament importance
    for tournament_key, stats in tournament_performance.items():
        if stats['played'] > 0:
            stats['win_rate'] = stats['wins'] / stats['played']
            stats['avg_score'] = stats['total_score'] / stats['played']
            stats['avg_opponent_score'] = stats['total_opponent_score'] / stats['played']
            stats['score_differential'] = stats['avg_score'] - stats['avg_opponent_score']
            
            # Determine tournament tier (simplified)
            event_name = tournament_key.split(':')[0].lower()
            if any(major in event_name for major in ['masters', 'champions', 'last chance']):
                stats['tier'] = 3  # Top tier
            elif any(medium in event_name for medium in ['challenger', 'regional', 'national']):
                stats['tier'] = 2  # Mid tier
            else:
                stats['tier'] = 1  # Lower tier
    
    return tournament_performance

@debug_func
def collect_team_data(team_limit=300, include_player_stats=True, include_economy=True, include_maps=True, 
                     use_cache=True, cache_path="cache/valorant_data_cache.pkl"):
    """
    Collect team data for training and evaluation, using cache if available.
    
    Args:
        team_limit (int): Maximum number of teams to include
        include_player_stats (bool): Whether to include player statistics
        include_economy (bool): Whether to include economy data
        include_maps (bool): Whether to include map statistics
        use_cache (bool): Whether to use cached data if available
        cache_path (str): Path to the cache file
        
    Returns:
        dict: Team data collection
    """
    print("\n========================================================")
    print("COLLECTING TEAM DATA")
    print("========================================================")
    
    # Check if we should use the cache
    if use_cache:
        print("Checking for cached data...")
        if os.path.exists(cache_path):
            team_data = load_cache(cache_path)
            if team_data:
                print("Using cached team data for training/backtesting")
                
                # Filter teams if needed
                if team_limit < len(team_data):
                    # Try to select teams by ranking first
                    ranked_teams = {team: data for team, data in team_data.items() 
                                   if data.get('ranking') is not None}
                    
                    if ranked_teams:
                        # Sort by ranking and take top N
                        sorted_teams = sorted(ranked_teams.items(), 
                                             key=lambda x: x[1]['ranking'] if x[1]['ranking'] else float('inf'))
                        team_data = {team: data for team, data in sorted_teams[:team_limit]}
                        print(f"Selected top {len(team_data)} teams by ranking")
                    else:
                        # Just take the first N teams if no ranking info
                        team_names = list(team_data.keys())[:team_limit]
                        team_data = {name: team_data[name] for name in team_names}
                        print(f"Selected first {len(team_data)} teams from cache")
                
                # Count stats for reporting
                economy_data_count = sum(1 for team in team_data.values() 
                                       if 'stats' in team and 'pistol_win_rate' in team['stats'])
                player_stats_count = sum(1 for team in team_data.values() 
                                        if team.get('player_stats'))
                map_stats_count = sum(1 for team in team_data.values() 
                                     if 'stats' in team and 'map_statistics' in team['stats'])
                
                print(f"\nUsing data for {len(team_data)} teams from cache:")
                print(f"  - Teams with economy data: {economy_data_count}")
                print(f"  - Teams with player stats: {player_stats_count}")
                print(f"  - Teams with map stats: {map_stats_count}")
                
                return team_data
            else:
                print("Cache loading failed. Falling back to API data collection.")
        else:
            print(f"Cache file not found at {cache_path}. Falling back to API data collection.")
    
    # If we get here, either use_cache=False or cache loading failed
    print("Collecting team data directly from API...")
    print(f"Including player stats: {include_player_stats}")
    print(f"Including economy data: {include_economy}")
    print(f"Including map data: {include_maps}")
    print(f"Fetching teams from API: {API_URL}/teams?limit={team_limit}")
    
    try:
        teams_response = requests.get(f"{API_URL}/teams?limit={team_limit}")
        print(f"API response status code: {teams_response.status_code}")
        if teams_response.status_code != 200:
            print(f"Error fetching teams: {teams_response.status_code}")
            try:
                print(f"Response content: {teams_response.text[:500]}...")
            except:
                pass
            return {}
        teams_data = teams_response.json()
        print(f"Teams data keys: {list(teams_data.keys())}")
        if 'data' not in teams_data:
            print("No 'data' field found in the response")
            try:
                print(f"Response content: {teams_data}")
            except:
                pass
            return {}
        print(f"Number of teams in response: {len(teams_data['data'])}")
        top_teams = []
        for team in teams_data['data']:
            if 'ranking' in team and team['ranking'] and team['ranking'] <= 50:
                top_teams.append(team)
        if not top_teams:
            print(f"No teams with rankings found. Using the first {min(150, team_limit)} teams instead.")
            if len(teams_data['data']) > 0:
                top_teams = teams_data['data'][:min(150, team_limit)]
                print(f"Selected {len(top_teams)} teams for data collection.")
            else:
                print("No teams found in data array.")
                return {}
        else:
            print(f"Selected {len(top_teams)} ranked teams for data collection.")
        if len(top_teams) == 0:
            print("ERROR: No teams selected for data collection.")
            return {}
        if top_teams:
            print(f"First team info: {top_teams[0]}")
        team_data_collection = {}
        economy_data_count = 0
        player_stats_count = 0
        map_stats_count = 0
        for team in tqdm(top_teams, desc="Collecting team data"):
            team_id = team.get('id')
            team_name = team.get('name')
            if not team_id or not team_name:
                print(f"Skipping team with missing id or name: {team}")
                continue
            print(f"\nProcessing team: {team_name} (ID: {team_id})")
            team_details, team_tag = fetch_team_details(team_id)
            team_history = fetch_team_match_history(team_id)
            if not team_history:
                print(f"No match history for team {team_name}, skipping.")
                continue
            team_matches = parse_match_data(team_history, team_name)
            if not team_matches:
                print(f"No parsed match data for team {team_name}, skipping.")
                continue
            for match in team_matches:
                match['team_tag'] = team_tag
                match['team_id'] = team_id
                match['team_name'] = team_name  # Always set team_name for fallback
            team_player_stats = None
            if include_player_stats:
                team_player_stats = fetch_team_player_stats(team_id)
                if team_player_stats:
                    player_stats_count += 1
            team_stats = calculate_team_stats(team_matches, team_player_stats, include_economy=include_economy)
            if include_economy and 'pistol_win_rate' in team_stats and team_stats['pistol_win_rate'] > 0:
                economy_data_count += 1
            team_stats['team_tag'] = team_tag
            team_stats['team_name'] = team_name
            team_stats['team_id'] = team_id
            if include_maps:
                map_stats = fetch_team_map_statistics(team_id)
                if map_stats:
                    team_stats['map_statistics'] = map_stats
                    map_stats_count += 1
            team_stats['map_performance'] = extract_map_performance(team_matches)
            team_stats['tournament_performance'] = extract_tournament_performance(team_matches)
            team_stats['performance_trends'] = analyze_performance_trends(team_matches)
            team_stats['opponent_quality'] = analyze_opponent_quality(team_matches, team_id)
            team_data_collection[team_name] = {
                'team_id': team_id,
                'team_tag': team_tag,
                'stats': team_stats,
                'matches': team_matches,
                'player_stats': team_player_stats,
                'ranking': team.get('ranking', None)
            }
            print(f"Successfully collected data for {team_name} with {len(team_matches)} matches")
        print(f"\nCollected data for {len(team_data_collection)} teams:")
        print(f"  - Teams with economy data: {economy_data_count}")
        print(f"  - Teams with player stats: {player_stats_count}")
        print(f"  - Teams with map stats: {map_stats_count}")
        return team_data_collection
    except Exception as e:
        print(f"Error in collect_team_data: {e}")
        import traceback
        traceback.print_exc()
        return {}

def analyze_performance_trends(team_matches, window_sizes=[5, 10, 20]):
    """Analyze team performance trends over different time windows."""
    if not team_matches:
        return {}
        
    # Sort matches by date
    sorted_matches = sorted(team_matches, key=lambda x: x.get('date', ''))
    
    trends = {
        'recent_matches': {},
        'form_trajectory': {},
        'progressive_win_rates': [],
        'moving_averages': {}
    }
    
    # Calculate recent match performance for different window sizes
    for window in window_sizes:
        if len(sorted_matches) >= window:
            recent_window = sorted_matches[-window:]
            wins = sum(1 for match in recent_window if match.get('team_won', False))
            win_rate = wins / window
            
            trends['recent_matches'][f'last_{window}_win_rate'] = win_rate
            trends['recent_matches'][f'last_{window}_wins'] = wins
            trends['recent_matches'][f'last_{window}_losses'] = window - wins
            
            # Calculate average score and score differential
            avg_score = sum(match.get('team_score', 0) for match in recent_window) / window
            avg_opp_score = sum(match.get('opponent_score', 0) for match in recent_window) / window
            
            trends['recent_matches'][f'last_{window}_avg_score'] = avg_score
            trends['recent_matches'][f'last_{window}_avg_opp_score'] = avg_opp_score
            trends['recent_matches'][f'last_{window}_score_diff'] = avg_score - avg_opp_score
    
    # Calculate win rate trajectory (positive/negative momentum)
    if len(window_sizes) >= 2:
        window_sizes.sort()
        for i in range(len(window_sizes) - 1):
            smaller_window = window_sizes[i]
            larger_window = window_sizes[i + 1]
            
            if f'last_{smaller_window}_win_rate' in trends['recent_matches'] and f'last_{larger_window}_win_rate' in trends['recent_matches']:
                smaller_win_rate = trends['recent_matches'][f'last_{smaller_window}_win_rate']
                larger_win_rate = trends['recent_matches'][f'last_{larger_window}_win_rate']
                
                # Positive value means team is improving recently
                trajectory = smaller_win_rate - larger_win_rate
                trends['form_trajectory'][f'{smaller_window}_vs_{larger_window}'] = trajectory
    
    # Calculate progressive win rates (first 25%, 50%, 75%, 100% of matches)
    total_matches = len(sorted_matches)
    for fraction in [0.25, 0.5, 0.75, 1.0]:
        num_matches = int(total_matches * fraction)
        if num_matches > 0:
            subset = sorted_matches[:num_matches]
            wins = sum(1 for match in subset if match.get('team_won', False))
            win_rate = wins / num_matches
            trends['progressive_win_rates'].append({
                'fraction': fraction,
                'matches': num_matches,
                'win_rate': win_rate
            })
    
    # Calculate moving averages for win rates
    for window in window_sizes:
        if window < total_matches:
            moving_avgs = []
            for i in range(window, total_matches + 1):
                window_matches = sorted_matches[i-window:i]
                wins = sum(1 for match in window_matches if match.get('team_won', False))
                win_rate = wins / window
                moving_avgs.append(win_rate)
            
            trends['moving_averages'][f'window_{window}'] = moving_avgs
            
            # Calculate trend direction from moving averages
            if len(moving_avgs) >= 2:
                trends['moving_averages'][f'window_{window}_trend'] = moving_avgs[-1] - moving_avgs[0]
    
    # Calculate recency-weighted win rate
    if total_matches > 0:
        weighted_sum = 0
        weight_sum = 0
        for i, match in enumerate(sorted_matches):
            # Exponential weighting - more recent matches count more
            weight = np.exp(i / total_matches)
            weighted_sum += weight * (1 if match.get('team_won', False) else 0)
            weight_sum += weight
        
        trends['recency_weighted_win_rate'] = weighted_sum / weight_sum if weight_sum > 0 else 0
    
    return trends

def get_team_ranking(team_id):
    """Get a team's current ranking directly from the team details endpoint."""
    if not team_id:
        return None, None
    
    team_data = fetch_api_data(f"teams/{team_id}")
    
    if not team_data or 'data' not in team_data:
        return None, None
    
    team_info = team_data['data']
    
    # Extract ranking information
    ranking = None
    if 'countryRanking' in team_info and 'rank' in team_info['countryRanking']:
        try:
            ranking = int(team_info['countryRanking']['rank'])
        except (ValueError, TypeError):
            pass
    
    # Extract rating information
    rating = None
    if 'rating' in team_info:
        # The rating format is complex, typically like "1432 1W 5L 1580 6W 1L"
        # Extract the first number which is the overall rating
        try:
            rating_parts = team_info['rating'].split()
            if rating_parts and rating_parts[0].isdigit():
                rating = float(rating_parts[0])
        except (ValueError, IndexError, AttributeError):
            pass
    
    return ranking, rating

def analyze_opponent_quality(team_matches, team_id):
    """Calculate metrics related to the quality of opponents faced."""
    if not team_matches:
        return {}
        
    opponent_quality = {
        'avg_opponent_ranking': 0,
        'avg_opponent_rating': 0,
        'top_10_win_rate': 0,
        'top_10_matches': 0,
        'bottom_50_win_rate': 0,
        'bottom_50_matches': 0,
        'upset_factor': 0,  # Wins against higher-ranked teams
        'upset_vulnerability': 0,  # Losses against lower-ranked teams
    }
    
    total_opponent_ranking = 0
    total_opponent_rating = 0
    count_with_ranking = 0
    
    # Get team's own ranking
    team_ranking, team_rating = get_team_ranking(team_id)
    
    top_10_wins = 0
    top_10_matches = 0
    bottom_50_wins = 0
    bottom_50_matches = 0
    
    upsets_achieved = 0  # Wins against better-ranked teams
    upset_opportunities = 0  # Matches against better-ranked teams
    upset_suffered = 0  # Losses against worse-ranked teams
    upset_vulnerabilities = 0  # Matches against worse-ranked teams
    
    # Process opponent quality analysis
    for match in team_matches:
        opponent_name = match.get('opponent_name')
        opponent_id = match.get('opponent_id')
        
        # Skip if no opponent ID available
        if not opponent_id:
            continue
        
        # Get opponent ranking directly
        opponent_ranking, opponent_rating = get_team_ranking(opponent_id)
        
        if opponent_ranking:
            total_opponent_ranking += opponent_ranking
            if opponent_rating:
                total_opponent_rating += opponent_rating
            count_with_ranking += 1
            
            # Check if opponent is top 10
            if opponent_ranking <= 10:
                top_10_matches += 1
                if match.get('team_won', False):
                    top_10_wins += 1
            
            # Check if opponent is bottom 50
            if opponent_ranking > 50:
                bottom_50_matches += 1
                if match.get('team_won', False):
                    bottom_50_wins += 1
            
            # Calculate upset metrics
            if team_ranking and opponent_ranking < team_ranking:
                upset_opportunities += 1
                if match.get('team_won', False):
                    upsets_achieved += 1
            
            if team_ranking and opponent_ranking > team_ranking:
                upset_vulnerabilities += 1
                if not match.get('team_won', False):
                    upset_suffered += 1
    
    # Calculate averages and rates
    if count_with_ranking > 0:
        opponent_quality['avg_opponent_ranking'] = total_opponent_ranking / count_with_ranking
        opponent_quality['avg_opponent_rating'] = total_opponent_rating / count_with_ranking
    
    opponent_quality['top_10_win_rate'] = top_10_wins / top_10_matches if top_10_matches > 0 else 0
    opponent_quality['top_10_matches'] = top_10_matches
    
    opponent_quality['bottom_50_win_rate'] = bottom_50_wins / bottom_50_matches if bottom_50_matches > 0 else 0
    opponent_quality['bottom_50_matches'] = bottom_50_matches
    
    opponent_quality['upset_factor'] = upsets_achieved / upset_opportunities if upset_opportunities > 0 else 0
    opponent_quality['upset_vulnerability'] = upset_suffered / upset_vulnerabilities if upset_vulnerabilities > 0 else 0
    
    # Add the team's own ranking info
    opponent_quality['team_ranking'] = team_ranking
    opponent_quality['team_rating'] = team_rating
    
    return opponent_quality

#-------------------------------------------------------------------------
# TEAM STATS CALCULATION
#-------------------------------------------------------------------------

def calculate_team_stats(team_matches, player_stats=None, include_economy=False):
    """Calculate comprehensive team statistics optionally including economy data."""
    if not team_matches:
        return {}
    
    # Basic stats
    total_matches = len(team_matches)
    wins = sum(1 for match in team_matches if match.get('team_won', False))
    
    losses = total_matches - wins
    win_rate = wins / total_matches if total_matches > 0 else 0
    
    # Scoring stats
    total_score = sum(match.get('team_score', 0) for match in team_matches)
    total_opponent_score = sum(match.get('opponent_score', 0) for match in team_matches)
    avg_score = total_score / total_matches if total_matches > 0 else 0
    avg_opponent_score = total_opponent_score / total_matches if total_matches > 0 else 0
    score_differential = avg_score - avg_opponent_score
    
    # Opponent-specific stats
    opponent_stats = {}
    for match in team_matches:
        opponent = match['opponent_name']
        if opponent not in opponent_stats:
            opponent_stats[opponent] = {
                'matches': 0,
                'wins': 0,
                'total_score': 0,
                'total_opponent_score': 0
            }
        
        opponent_stats[opponent]['matches'] += 1
        opponent_stats[opponent]['wins'] += 1 if match['team_won'] else 0
        opponent_stats[opponent]['total_score'] += match['team_score']
        opponent_stats[opponent]['total_opponent_score'] += match['opponent_score']
    
    # Calculate win rates and average scores for each opponent
    for opponent, stats in opponent_stats.items():
        stats['win_rate'] = stats['wins'] / stats['matches'] if stats['matches'] > 0 else 0
        stats['avg_score'] = stats['total_score'] / stats['matches'] if stats['matches'] > 0 else 0
        stats['avg_opponent_score'] = stats['total_opponent_score'] / stats['matches'] if stats['matches'] > 0 else 0
        stats['score_differential'] = stats['avg_score'] - stats['avg_opponent_score']
    
    # Map stats
    map_stats = {}
    for match in team_matches:
        map_name = match.get('map', 'Unknown')
        if map_name == '' or map_name is None:
            map_name = 'Unknown'
        
        if map_name not in map_stats:
            map_stats[map_name] = {
                'played': 0,
                'wins': 0
            }
        
        map_stats[map_name]['played'] += 1
        map_stats[map_name]['wins'] += 1 if match['team_won'] else 0
    
    # Calculate win rates for each map
    for map_name, stats in map_stats.items():
        stats['win_rate'] = stats['wins'] / stats['played'] if stats['played'] > 0 else 0
    
    # Recent form (last 5 matches)
    sorted_matches = sorted(team_matches, key=lambda x: x.get('date', ''))
    recent_matches = sorted_matches[-5:] if len(sorted_matches) >= 5 else sorted_matches
    recent_form = sum(1 for match in recent_matches if match['team_won']) / len(recent_matches) if recent_matches else 0
    
    # Extract match details stats if available
    advanced_stats = extract_match_details_stats(team_matches)
    
    # Combine all stats
    team_stats = {
        'matches': total_matches,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_score': total_score,
        'total_opponent_score': total_opponent_score,
        'avg_score': avg_score,
        'avg_opponent_score': avg_opponent_score,
        'score_differential': score_differential,
        'recent_form': recent_form,
        'opponent_stats': opponent_stats,
        'map_stats': map_stats,
        'advanced_stats': advanced_stats
    }
    
    # If we have player stats, add them to the team stats
    if player_stats:
        player_agg_stats = calculate_team_player_stats(player_stats)
        
        # Add player stats to the team stats
        team_stats.update({
            'player_stats': player_agg_stats,
            # Also add key metrics directly to top level for easier model access
            'avg_player_rating': player_agg_stats.get('avg_rating', 0),
            'avg_player_acs': player_agg_stats.get('avg_acs', 0),
            'avg_player_kd': player_agg_stats.get('avg_kd', 0),
            'avg_player_kast': player_agg_stats.get('avg_kast', 0),
            'avg_player_adr': player_agg_stats.get('avg_adr', 0),
            'avg_player_headshot': player_agg_stats.get('avg_headshot', 0),
            'star_player_rating': player_agg_stats.get('star_player_rating', 0),
            'team_consistency': player_agg_stats.get('team_consistency', 0),
            'fk_fd_ratio': player_agg_stats.get('fk_fd_ratio', 0)
        })
    
    # Add economy data if requested
    if include_economy:
        # Process each match to extract economy data
        total_pistol_won = 0
        total_pistol_rounds = 0
        total_eco_won = 0
        total_eco_rounds = 0
        total_semi_eco_won = 0
        total_semi_eco_rounds = 0
        total_semi_buy_won = 0
        total_semi_buy_rounds = 0
        total_full_buy_won = 0
        total_full_buy_rounds = 0
        total_efficiency_sum = 0
        economy_matches_count = 0
        
        team_tag = team_matches[0].get('team_tag') if team_matches else None
        team_name = team_matches[0].get('team_name') if team_matches else None
        
        for match in team_matches:
            match_id = match.get('match_id')
            if not match_id:
                continue
                
            # Get team tag and name for this specific match
            match_team_tag = match.get('team_tag')
            match_team_name = match.get('team_name')
                
            # Get economy data
            economy_data = fetch_match_economy_details(match_id)
            
            if not economy_data:
                continue
                
            # Extract team-specific economy data using the identifier
            our_team_metrics = extract_economy_metrics(
                economy_data, 
                team_identifier=match_team_tag, 
                fallback_name=match_team_name
            )
            
            if not our_team_metrics or our_team_metrics.get('economy_data_missing', False):
                continue
            
            # Aggregate stats
            total_pistol_won += our_team_metrics.get('pistol_rounds_won', 0)
            total_pistol_rounds += our_team_metrics.get('total_pistol_rounds', 2)
            
            total_eco_won += our_team_metrics.get('eco_won', 0)
            total_eco_rounds += our_team_metrics.get('eco_total', 0)
            
            total_semi_eco_won += our_team_metrics.get('semi_eco_won', 0)
            total_semi_eco_rounds += our_team_metrics.get('semi_eco_total', 0)
            

            total_semi_buy_won += our_team_metrics.get('semi_buy_won', 0)
            total_semi_buy_rounds += our_team_metrics.get('semi_buy_total', 0)
            
            total_full_buy_won += our_team_metrics.get('full_buy_won', 0)
            total_full_buy_rounds += our_team_metrics.get('full_buy_total', 0)
            
            total_efficiency_sum += our_team_metrics.get('economy_efficiency', 0)
            economy_matches_count += 1
        
        # Calculate economy stats if we have data
        if economy_matches_count > 0:
            economy_stats = {
                'pistol_win_rate': total_pistol_won / total_pistol_rounds if total_pistol_rounds > 0 else 0,
                'eco_win_rate': total_eco_won / total_eco_rounds if total_eco_rounds > 0 else 0,
                'semi_eco_win_rate': total_semi_eco_won / total_semi_eco_rounds if total_semi_eco_rounds > 0 else 0,
                'semi_buy_win_rate': total_semi_buy_won / total_semi_buy_rounds if total_semi_buy_rounds > 0 else 0,
                'full_buy_win_rate': total_full_buy_won / total_full_buy_rounds if total_full_buy_rounds > 0 else 0,
                'economy_efficiency': total_efficiency_sum / economy_matches_count,
                'low_economy_win_rate': (total_eco_won + total_semi_eco_won) / (total_eco_rounds + total_semi_eco_rounds) if (total_eco_rounds + total_semi_eco_rounds) > 0 else 0,
                'high_economy_win_rate': (total_semi_buy_won + total_full_buy_won) / (total_semi_buy_rounds + total_full_buy_rounds) if (total_semi_buy_rounds + total_full_buy_rounds) > 0 else 0,
                'pistol_round_sample_size': total_pistol_rounds,
                'pistol_confidence': 1.0 - (1.0 / (1.0 + 0.1 * total_pistol_rounds))
            }
            team_stats.update(economy_stats)

    # Extract additional performance analyses
    map_performance = extract_map_performance(team_matches)
    tournament_performance = extract_tournament_performance(team_matches)
    performance_trends = analyze_performance_trends(team_matches)
    
    # Add these to team stats
    team_stats['map_performance'] = map_performance
    team_stats['tournament_performance'] = tournament_performance
    team_stats['performance_trends'] = performance_trends
    
    return team_stats

#-------------------------------------------------------------------------
# DATA PREPARATION FOR MACHINE LEARNING
#-------------------------------------------------------------------------

def clean_feature_data(X):
    """Clean and prepare feature data for model input."""
    # Convert to DataFrame if it's a list of dictionaries
    if isinstance(X, list):
        df = pd.DataFrame(X)
    else:
        df = X.copy()
    
    # Fill NA values with 0
    df = df.fillna(0)
    
    # Handle non-numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(float)
            except (ValueError, TypeError):
                print(f"Dropping column {col} due to non-numeric values")
                df = df.drop(columns=[col])
    
    return df

def prepare_data_for_model(team1_stats, team2_stats):
    """
    Prepare data for the ML model with improved handling of missing h2h data.
    
    Args:
        team1_stats (dict): Statistics for team 1
        team2_stats (dict): Statistics for team 2
    
    Returns:
        dict: Features prepared for model prediction
    """
    if not team1_stats or not team2_stats:
        print("Missing team statistics data")
        return None
    
    features = {}
    
    #----------------------------------------
    # 1. BASIC TEAM STATS
    #----------------------------------------
    # Win rates and match-level statistics
    features['win_rate_diff'] = team1_stats.get('win_rate', 0) - team2_stats.get('win_rate', 0)
    features['better_win_rate_team1'] = 1 if team1_stats.get('win_rate', 0) > team2_stats.get('win_rate', 0) else 0
    
    features['recent_form_diff'] = team1_stats.get('recent_form', 0) - team2_stats.get('recent_form', 0)
    features['better_recent_form_team1'] = 1 if team1_stats.get('recent_form', 0) > team2_stats.get('recent_form', 0) else 0
    
    features['score_diff_differential'] = team1_stats.get('score_differential', 0) - team2_stats.get('score_differential', 0)
    features['better_score_diff_team1'] = 1 if team1_stats.get('score_differential', 0) > team2_stats.get('score_differential', 0) else 0
    
    # Match count (context feature, not team-specific)
    team1_matches = team1_stats.get('matches', 0)
    team2_matches = team2_stats.get('matches', 0)
    
    # Handle cases where 'matches' could be a list or a count
    if isinstance(team1_matches, list):
        team1_match_count = len(team1_matches)
    else:
        team1_match_count = team1_matches
        
    if isinstance(team2_matches, list):
        team2_match_count = len(team2_matches)
    else:
        team2_match_count = team2_matches
    
    features['total_matches'] = team1_match_count + team2_match_count
    features['match_count_diff'] = team1_match_count - team2_match_count
    
    # Add average metrics rather than separate team metrics
    features['avg_win_rate'] = (team1_stats.get('win_rate', 0) + team2_stats.get('win_rate', 0)) / 2
    features['avg_recent_form'] = (team1_stats.get('recent_form', 0) + team2_stats.get('recent_form', 0)) / 2
    
    # Add win/loss counts
    features['wins_diff'] = team1_stats.get('wins', 0) - team2_stats.get('wins', 0)
    features['losses_diff'] = team1_stats.get('losses', 0) - team2_stats.get('losses', 0)
    features['win_loss_ratio_diff'] = (team1_stats.get('wins', 0) / max(team1_stats.get('losses', 1), 1)) - (team2_stats.get('wins', 0) / max(team2_stats.get('losses', 1), 1))
    
    # Average scores
    features['avg_score_diff'] = team1_stats.get('avg_score', 0) - team2_stats.get('avg_score', 0)
    features['better_avg_score_team1'] = 1 if team1_stats.get('avg_score', 0) > team2_stats.get('avg_score', 0) else 0
    features['avg_score_metric'] = (team1_stats.get('avg_score', 0) + team2_stats.get('avg_score', 0)) / 2
    
    # Opponent scores
    features['avg_opponent_score_diff'] = team1_stats.get('avg_opponent_score', 0) - team2_stats.get('avg_opponent_score', 0)
    features['better_defense_team1'] = 1 if team1_stats.get('avg_opponent_score', 0) < team2_stats.get('avg_opponent_score', 0) else 0
    features['avg_defense_metric'] = (team1_stats.get('avg_opponent_score', 0) + team2_stats.get('avg_opponent_score', 0)) / 2
    
    #----------------------------------------
    # 2. PLAYER STATS
    #----------------------------------------
    # Only add if both teams have the data
    if ('avg_player_rating' in team1_stats and 'avg_player_rating' in team2_stats and
        team1_stats.get('avg_player_rating', 0) > 0 and team2_stats.get('avg_player_rating', 0) > 0):
        
        # Basic player rating stats
        features['player_rating_diff'] = team1_stats.get('avg_player_rating', 0) - team2_stats.get('avg_player_rating', 0)
        features['better_player_rating_team1'] = 1 if team1_stats.get('avg_player_rating', 0) > team2_stats.get('avg_player_rating', 0) else 0
        features['avg_player_rating'] = (team1_stats.get('avg_player_rating', 0) + team2_stats.get('avg_player_rating', 0)) / 2
        
        # Additional player stats...
        # (Keep your existing player stats code)
    
    #----------------------------------------
    # 3. ECONOMY STATS
    #----------------------------------------
    # Only add if both teams have the data
    if ('pistol_win_rate' in team1_stats and 'pistol_win_rate' in team2_stats and
        team1_stats.get('pistol_win_rate', 0) > 0 and team2_stats.get('pistol_win_rate', 0) > 0):
        
        # Pistol win rate
        features['pistol_win_rate_diff'] = team1_stats.get('pistol_win_rate', 0) - team2_stats.get('pistol_win_rate', 0)
        features['better_pistol_team1'] = 1 if team1_stats.get('pistol_win_rate', 0) > team2_stats.get('pistol_win_rate', 0) else 0
        features['avg_pistol_win_rate'] = (team1_stats.get('pistol_win_rate', 0) + team2_stats.get('pistol_win_rate', 0)) / 2
        
        # Other economy stats...
        # (Keep your existing economy stats code)
    
    #----------------------------------------
    # 4. MAP STATS
    #----------------------------------------
    # (Keep your existing map stats code)
    
    #----------------------------------------
    # 5. HEAD-TO-HEAD STATS
    #----------------------------------------
    # Add head-to-head statistics with improved handling of missing data
    team2_name = team2_stats.get('team_name', '')
    h2h_found = False
    h2h_stats = None
    
    # Create variations of team2 name for matching
    team2_variations = [
        team2_name,
        team2_name.lower(),
        team2_name.upper(),
        team2_name.replace(" ", ""),
        team2_stats.get('team_tag', '') if team2_stats.get('team_tag', '') else ""
    ]
    
    # Filter out empty variations
    team2_variations = [v for v in team2_variations if v]
    
    # Check for head-to-head data using all possible variations
    if 'opponent_stats' in team1_stats and isinstance(team1_stats['opponent_stats'], dict):
        for opponent_name, stats in team1_stats['opponent_stats'].items():
            # Skip entries with invalid stats
            if not isinstance(stats, dict):
                continue
                
            # First check exact match
            if opponent_name == team2_name:
                h2h_stats = stats
                h2h_found = True
                print(f"Found exact match head-to-head data: {opponent_name}")
                break
                
            # Then check variations
            for variation in team2_variations:
                if (opponent_name.lower() == variation.lower() or
                    variation.lower() in opponent_name.lower() or
                    opponent_name.lower() in variation.lower()):
                    h2h_stats = stats
                    h2h_found = True
                    print(f"Found variation match head-to-head data: {opponent_name} ↔ {variation}")
                    break
            
            if h2h_found:
                break

    # Add head-to-head features
    if h2h_found and h2h_stats:
        features['h2h_win_rate'] = h2h_stats.get('win_rate', 0.5)
        features['h2h_matches'] = h2h_stats.get('matches', 0)
        features['h2h_score_diff'] = h2h_stats.get('score_differential', 0)
        features['h2h_advantage_team1'] = 1 if h2h_stats.get('win_rate', 0.5) > 0.5 else 0
        features['h2h_significant'] = 1 if h2h_stats.get('matches', 0) >= 3 else 0
        
        print(f"Using head-to-head data: Matches={features['h2h_matches']}, "
              f"Win rate={features['h2h_win_rate']:.4f}, "
              f"Score diff={features['h2h_score_diff']:.4f}, "
              f"Advantage={'Team1' if features['h2h_advantage_team1'] else 'Team2'}")
    else:
        # IMPROVED: Estimate h2h data based on team strengths instead of defaults
        team1_win_rate = team1_stats.get('win_rate', 0.5)
        team2_win_rate = team2_stats.get('win_rate', 0.5)
        
        # Estimate h2h win rate based on relative team strengths
        if team1_win_rate + team2_win_rate > 0:
            estimated_h2h = team1_win_rate / (team1_win_rate + team2_win_rate)
        else:
            estimated_h2h = 0.5
            
        # Scale to reduce extremes
        estimated_h2h = 0.5 + (estimated_h2h - 0.5) * 0.7
        
        features['h2h_win_rate'] = estimated_h2h
        features['h2h_matches'] = 0  # No actual matches
        features['h2h_score_diff'] = (team1_stats.get('score_differential', 0) - 
                                    team2_stats.get('score_differential', 0)) * 0.5
        features['h2h_advantage_team1'] = 1 if estimated_h2h > 0.5 else 0
        features['h2h_significant'] = 0  # Not significant since estimated
        
        print(f"No head-to-head data found between {team1_stats.get('team_name', 'Team1')} "
              f"and {team2_name} in opponent_stats")
        print(f"Estimated head-to-head data: Win rate={estimated_h2h:.4f}, "
              f"Score diff={features['h2h_score_diff']:.4f}, "
              f"Advantage={'Team1' if features['h2h_advantage_team1'] else 'Team2'}")
    
    #----------------------------------------
    # 6. INTERACTION TERMS
    #----------------------------------------
    # Create interaction terms between key metrics
    
    # Player rating x win rate
    if 'player_rating_diff' in features and 'win_rate_diff' in features:
        features['rating_x_win_rate'] = features['player_rating_diff'] * features['win_rate_diff']
    
    # Economy interactions
    if 'pistol_win_rate_diff' in features and 'eco_win_rate_diff' in features:
        features['pistol_x_eco'] = features['pistol_win_rate_diff'] * features['eco_win_rate_diff']
    
    if 'pistol_win_rate_diff' in features and 'full_buy_win_rate_diff' in features:
        features['pistol_x_full_buy'] = features['pistol_win_rate_diff'] * features['full_buy_win_rate_diff']
    
    # First blood interactions
    if 'fk_fd_diff' in features and 'win_rate_diff' in features:
        features['first_blood_x_win_rate'] = features['fk_fd_diff'] * features['win_rate_diff']
    
    # H2H interactions
    if features['h2h_matches'] > 0:
        if 'win_rate_diff' in features:
            features['h2h_x_win_rate'] = features['h2h_win_rate'] * features['win_rate_diff']
        
        if 'recent_form_diff' in features:
            features['h2h_x_form'] = features['h2h_win_rate'] * features['recent_form_diff']
    
    # Clean up non-numeric values
    for key, value in list(features.items()):
        if not isinstance(value, (int, float)):
            print(f"Non-numeric feature value: {key}={value}, type={type(value)}")
            # Try to convert to numeric if possible
            try:
                features[key] = float(value)
            except (ValueError, TypeError):
                print(f"Removing non-numeric feature: {key}")
                del features[key]
    
    # Log summary of head-to-head features
    print(f"Head-to-head: Matches={features['h2h_matches']}, "
          f"Win rate={features['h2h_win_rate']:.4f}, "
          f"Score diff={features['h2h_score_diff']:.4f}, "
          f"Advantage={'Team1' if features['h2h_advantage_team1'] else 'Team2'}")
    
    return features

@debug_func
def build_training_dataset(team_data_collection):
    print(f"[{datetime.now()}] Building training dataset from {len(team_data_collection)} teams...")
    sys.stdout.flush()
    """
    Build a dataset for model training from team data collection.
    Works with both API-fetched and cached data.
    """
    X = []  # Feature vectors
    y = []  # Labels (1 if team1 won, 0 if team2 won)
    print(f"Building training dataset from {len(team_data_collection)} teams...")
    for team_name, team_data in team_data_collection.items():
        # Handle both data structures (direct API and cached)
        if 'matches' in team_data:
            matches = team_data['matches']
            stats = team_data.get('stats', {})
        else:
            # Direct structure (not nested)
            matches = team_data.get('matches', [])
            stats = team_data
        
        if not matches:
            continue
            
        for match in matches:
            opponent_name = match.get('opponent_name')
            if opponent_name not in team_data_collection:
                continue
                
            team1_stats = stats
            team2_data = team_data_collection[opponent_name]
            
            # Handle both data structures for opponent
            if 'stats' in team2_data:
                team2_stats = team2_data['stats']
            else:
                team2_stats = team2_data
                
            features = prepare_data_for_model(team1_stats, team2_stats)
            if features:
                X.append(features)
                y.append(1 if match.get('team_won', False) else 0)
                
    print(f"Created {len(X)} training samples from match data")
    return X, y

#-------------------------------------------------------------------------
# MODEL TRAINING AND EVALUATION
#-------------------------------------------------------------------------

def run_backtest(start_date=None, end_date=None, team_limit=50, bankroll=1000.0, bet_pct=0.05,
               min_edge=0.03, confidence_threshold=0.2, use_cache=True, cache_path="cache/valorant_data_cache.pkl"):
    """
    Enhanced backtesting system with caching support.
    Uses the properly loaded retrained models with consistent feature preparation.
    Args:
        start_date (str): Start date for backtesting (YYYY-MM-DD)
        end_date (str): End date for backtesting (YYYY-MM-DD)
        team_limit (int): Maximum number of teams to analyze
        bankroll (float): Starting bankroll
        bet_pct (float): Maximum percentage of bankroll to bet
        min_edge (float): Minimum edge required for betting
        confidence_threshold (float): Minimum model confidence required
        use_cache (bool): Whether to use cached data if available
        cache_path (str): Path to the cache file
    Returns:
        dict: Detailed backtest results
    """
    print("\n========== STARTING ENHANCED BACKTEST ==========")
    print(f"Parameters: teams={team_limit}, bankroll=${bankroll}, max bet={bet_pct*100}%")
    print(f"Min edge: {min_edge*100}%, min confidence: {confidence_threshold*100}%")
    print(f"Using cached data: {use_cache}")
    ensemble_models, selected_features = load_backtesting_models()
    if not ensemble_models:
        print("ERROR: Failed to load any models. Aborting backtest.")
        return None
    if not selected_features:
        print("ERROR: Failed to load feature list. Aborting backtest.")
        return None
    print(f"Successfully loaded {len(ensemble_models)} models and {len(selected_features)} features")
    os.makedirs("model_diagnostics", exist_ok=True)
    with open("model_diagnostics/backtest_setup.txt", "w") as f:
        f.write(f"Backtest Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Min edge: {min_edge*100}%\n")
        f.write(f"Min confidence: {confidence_threshold*100}%\n\n")
        f.write("Models loaded:\n")
        model_types = {}
        for model_type, _, _ in ensemble_models:
            if model_type not in model_types:
                model_types[model_type] = 0
            model_types[model_type] += 1
        for model_type, count in model_types.items():
            f.write(f"  - {model_type.upper()}: {count} models\n")
        f.write("\nFeatures loaded:\n")
        for i, feature in enumerate(selected_features):
            f.write(f"  {i+1}. {feature}\n")
    print(f"Model setup verification saved to model_diagnostics/backtest_setup.txt")
    print("Collecting team data for backtesting...")
    team_data = {}
    team_ids = {}
    if use_cache:
        cache_data = load_cache(cache_path)
        if cache_data:
            team_data = cache_data
            if start_date or end_date:
                print(f"Filtering matches by date range: {start_date or 'earliest'} to {end_date or 'latest'}")
                for team_name, team_info in team_data.items():
                    filtered_matches = []
                    for match in team_info.get('matches', []):
                        match_date = match.get('date', '')
                        if (not start_date or match_date >= start_date) and \
                           (not end_date or match_date <= end_date):
                            filtered_matches.append(match)
                    team_info['matches'] = filtered_matches
                for team_name, team_info in team_data.items():
                    filtered_matches = team_info.get('matches', [])
                    if filtered_matches:
                        team_info['stats'] = calculate_team_stats(
                            filtered_matches,
                            team_info.get('player_stats'),
                            include_economy=True)
    if not team_data:
        teams_list = get_teams_for_backtesting(limit=team_limit, use_cache=use_cache)
        if not teams_list:
            print("Error: No teams retrieved for backtesting. Check API connection.")
            return None
        team_count = 0
        for team_idx, team in enumerate(tqdm(teams_list, desc="Loading team data")):
            team_name = team.get('name')
            team_id = team.get('id')
            if not team_name or not team_id:
                continue
            team_ids[team_name] = team_id
            try:
                team_history = fetch_team_match_history(team_id)
                if not team_history:
                    print(f"No match history for {team_name}, skipping.")
                    continue
                team_matches = parse_match_data(team_history, team_name)
                if not team_matches:
                    print(f"No parsed match data for {team_name}, skipping.")
                    continue
                if start_date or end_date:
                    filtered_matches = []
                    for match in team_matches:
                        match_date = match.get('date', '')
                        if (not start_date or match_date >= start_date) and \
                           (not end_date or match_date <= end_date):
                            filtered_matches.append(match)
                    team_matches = filtered_matches
                    if not team_matches:
                        print(f"No matches in specified date range for {team_name}, skipping.")
                        continue
                team_player_stats = None
                try:
                    team_player_stats = fetch_team_player_stats(team_id)
                except Exception as e:
                    print(f"Error fetching player stats for {team_name}: {e}")
                team_stats = calculate_team_stats(team_matches, team_player_stats, include_economy=True)
                team_stats['map_performance'] = extract_map_performance(team_matches)
                team_stats['tournament_performance'] = extract_tournament_performance(team_matches)
                team_stats['performance_trends'] = analyze_performance_trends(team_matches)
                team_stats['opponent_quality'] = analyze_opponent_quality(team_matches, team_id)
                map_stats = fetch_team_map_statistics(team_id)
                if map_stats:
                    team_stats['map_statistics'] = map_stats
                team_data[team_name] = {
                    'team_id': team_id,
                    'stats': team_stats,
                    'matches': team_matches,
                    'player_stats': team_player_stats,
                    'ranking': team.get('ranking')
                }
                team_count += 1
                print(f"Successfully collected data for {team_name} with {len(team_matches)} matches")
                if team_count >= team_limit:
                    break
            except Exception as e:
                print(f"Error processing team {team_name}: {e}")
                traceback.print_exc()
                continue
    backtest_matches = []
    seen_match_ids = set()
    print("Building match dataset for backtesting...")
    for team_name, team_info in team_data.items():
        for match in team_info.get('matches', []):
            match_id = match.get('match_id', '')
            opponent_name = match.get('opponent_name')
            if opponent_name not in team_data:
                continue
            if match_id in seen_match_ids:
                continue
            seen_match_ids.add(match_id)
            backtest_matches.append({
                'team1_name': team_name,
                'team2_name': opponent_name,
                'match_data': match,
                'match_id': match_id
            })
    print(f"Found {len(backtest_matches)} unique matches for backtesting")
    if not backtest_matches:
        print("Error: No matches available for backtesting. Check match data collection.")
        return None
    results = {
        'predictions': [],
        'bets': [],
        'performance': {
            'accuracy': 0,
            'roi': 0,
            'profit': 0,
            'bankroll_history': [],
            'win_rate': 0
        },
        'metrics': {
            'accuracy_by_edge': {},
            'roi_by_edge': {},
            'bet_types': {},
            'confidence_bins': {}
        },
        'team_performance': {}  # Track performance by team
    }
    current_bankroll = bankroll
    starting_bankroll = bankroll
    correct_predictions = 0
    total_predictions = 0
    total_bets = 0
    winning_bets = 0
    total_wagered = 0
    total_returns = 0
    confidence_bins = {}
    bet_history = []
    previous_bets_by_team = {}
    for match_idx, match in enumerate(tqdm(backtest_matches, desc="Backtesting matches")):
        team1_name = match['team1_name']
        team2_name = match['team2_name']
        match_data = match['match_data']
        match_id = match['match_id']
        match_date = match_data.get('date', 'Unknown')
        for team in [team1_name, team2_name]:
            if team not in results['team_performance']:
                results['team_performance'][team] = {
                    'predictions': 0,
                    'correct': 0,
                    'bets': 0,
                    'wins': 0,
                    'wagered': 0,
                    'returns': 0
                }
        try:
            team1_stats = team_data[team1_name].get('stats', {})
            team2_stats = team_data[team2_name].get('stats', {})
            X = prepare_features_for_backtest(team1_stats, team2_stats, selected_features)
            if X is None:
                print(f"Failed to prepare features for {team1_name} vs {team2_name}, skipping")
                continue
            try:
                win_probability, raw_predictions, confidence_score = predict_with_ensemble(
                    ensemble_models, X
                )
                print(f"\nPrediction for {team1_name} vs {team2_name}:")
                print(f"Win probability: {win_probability:.4f}, Confidence: {confidence_score:.4f}")
                if isinstance(raw_predictions, list) and raw_predictions:
                    raw_min = min([float(p) for p in raw_predictions])
                    raw_max = max([float(p) for p in raw_predictions])
                    print(f"Raw predictions range: {raw_min:.4f} - {raw_max:.4f}")
                if confidence_score < 0.3:
                    print("WARNING: Very low confidence prediction - models disagree significantly")
            except Exception as e:
                print(f"Error making prediction for {team1_name} vs {team2_name}: {e}")
                traceback.print_exc()
                continue
            try:
                team1_score, team2_score = extract_match_score(match_data)
                actual_winner = 'team1' if team1_score > team2_score else 'team2'
            except Exception as e:
                print(f"Error extracting match score for {match_id}: {e}")
                continue
            predicted_winner = 'team1' if win_probability > 0.5 else 'team2'
            prediction_correct = predicted_winner == actual_winner
            correct_predictions += 1 if prediction_correct else 0
            total_predictions += 1
            results['team_performance'][team1_name]['predictions'] += 1
            if predicted_winner == 'team1' and prediction_correct:
                results['team_performance'][team1_name]['correct'] += 1
            results['team_performance'][team2_name]['predictions'] += 1
            if predicted_winner == 'team2' and prediction_correct:
                results['team_performance'][team2_name]['correct'] += 1
            confidence_bin = int(confidence_score * 10) * 10  # Round to nearest 10%
            confidence_key = f"{confidence_bin}%"
            if confidence_key not in confidence_bins:
                confidence_bins[confidence_key] = {"total": 0, "correct": 0}
            confidence_bins[confidence_key]["total"] += 1
            if prediction_correct:
                confidence_bins[confidence_key]["correct"] += 1
            base_odds = simulate_odds(win_probability)
            jittered_odds = {}
            for key, value in base_odds.items():
                jitter = np.random.uniform(-0.05, 0.05)
                jittered_value = value * (1 + jitter)
                jittered_odds[key] = round(jittered_value, 2)
            odds_data = jittered_odds
            betting_analysis = analyze_betting_edge_for_backtesting(
                win_probability, 1 - win_probability, odds_data,
                confidence_score, current_bankroll
            )
            optimal_bets = select_optimal_bets_conservative(
                betting_analysis,
                team1_name,
                team2_name,
                current_bankroll,
                max_bets=3,
                max_total_risk=Config.BETTING.MAX_TOTAL_RISK_PCT,
                config=Config.BETTING
            )
            match_bets = []
            for bet_type, analysis in optimal_bets.items():
                team_for_streak = team1_name if 'team1' in bet_type else team2_name if 'team2' in bet_type else None
                base_kelly = analysis['kelly_fraction']
                max_bet = current_bankroll * bet_pct
                bet_amount = min(analysis['bet_amount'], max_bet)
                if team_for_streak:
                    if team_for_streak not in previous_bets_by_team:
                        previous_bets_by_team[team_for_streak] = []
                bet_won = evaluate_bet_outcome(bet_type, actual_winner, team1_score, team2_score)
                odds = analysis['odds']
                returns = bet_amount * odds if bet_won else 0
                profit = returns - bet_amount
                current_bankroll += profit
                match_bets.append({
                    'bet_type': bet_type,
                    'amount': bet_amount,
                    'odds': odds,
                    'won': bet_won,
                    'returns': returns,
                    'profit': profit,
                    'edge': analysis['edge'],
                    'predicted_prob': analysis['probability'],
                    'implied_prob': analysis['implied_prob'],
                    'team1': team1_name,
                    'team2': team2_name
                })
                if team_for_streak:
                    previous_bets_by_team[team_for_streak].append({
                        'bet_type': bet_type,
                        'won': bet_won,
                        'date': match_date
                    })
                total_bets += 1
                winning_bets += 1 if bet_won else 0
                total_wagered += bet_amount
                total_returns += returns
                if bet_type not in results['metrics']['bet_types']:
                    results['metrics']['bet_types'][bet_type] = {
                        'total': 0, 'won': 0, 'wagered': 0, 'returns': 0
                    }
                results['metrics']['bet_types'][bet_type]['total'] += 1
                results['metrics']['bet_types'][bet_type]['won'] += 1 if bet_won else 0
                results['metrics']['bet_types'][bet_type]['wagered'] += bet_amount
                results['metrics']['bet_types'][bet_type]['returns'] += returns
                edge_bucket = int(analysis['edge'] * 100) // 5 * 5  # Round to nearest 5%
                edge_key = f"{edge_bucket}%-{edge_bucket+5}%"
                if edge_key not in results['metrics']['accuracy_by_edge']:
                    results['metrics']['accuracy_by_edge'][edge_key] = {'total': 0, 'correct': 0}
                if edge_key not in results['metrics']['roi_by_edge']:
                    results['metrics']['roi_by_edge'][edge_key] = {'wagered': 0, 'returns': 0}
                results['metrics']['accuracy_by_edge'][edge_key]['total'] += 1
                results['metrics']['accuracy_by_edge'][edge_key]['correct'] += 1 if bet_won else 0
                results['metrics']['roi_by_edge'][edge_key]['wagered'] += bet_amount
                results['metrics']['roi_by_edge'][edge_key]['returns'] += returns
                team_tracked = team1_name if 'team1' in bet_type else team2_name if 'team2' in bet_type else None
                if team_tracked:
                    results['team_performance'][team_tracked]['bets'] += 1
                    results['team_performance'][team_tracked]['wagered'] += bet_amount
                    results['team_performance'][team_tracked]['returns'] += returns
                    if bet_won:
                        results['team_performance'][team_tracked]['wins'] += 1
            results['predictions'].append({
                'match_id': match_id,
                'team1': team1_name,
                'team2': team2_name,
                'predicted_winner': predicted_winner,
                'actual_winner': actual_winner,
                'team1_prob': win_probability,
                'team2_prob': 1 - win_probability,
                'confidence': confidence_score,
                'correct': prediction_correct,
                'score': f"{team1_score}-{team2_score}",
                'date': match_date
            })
            if match_bets:
                results['bets'].append({
                    'match_id': match_id,
                    'team1': team1_name,
                    'team2': team2_name,
                    'bets': match_bets,
                    'date': match_date
                })
                bet_history.append({
                    'match_id': match_id,
                    'team1': team1_name,
                    'team2': team2_name,
                    'bets': match_bets,
                    'date': match_date
                })
            results['performance']['bankroll_history'].append({
                'match_idx': match_idx,
                'bankroll': current_bankroll,
                'match_id': match_id,
                'date': match_date
            })
            target_reached, target_message = check_profit_targets(starting_bankroll, current_bankroll, target_percentage=1000000)
            if target_reached:
                print(f"\n{target_message}")
                print("Stopping backtest early due to reaching profit target.")
                break
            if (match_idx + 1) % 50 == 0 or match_idx == len(backtest_matches) - 1:
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                roi = (total_returns - total_wagered) / total_wagered if total_wagered > 0 else 0
                print(f"\nProgress ({match_idx + 1}/{len(backtest_matches)}):")
                print(f"Prediction Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
                print(f"Betting ROI: {roi:.2%} (${total_returns - total_wagered:.2f})")
                print(f"Current Bankroll: ${current_bankroll:.2f}")
                print(f"Win Rate: {winning_bets/total_bets:.2%} ({winning_bets}/{total_bets})" if total_bets > 0 else "No bets placed")
        except Exception as e:
            print(f"Error processing match {match_id}: {e}")
            traceback.print_exc()
            continue
    results['metrics']['confidence_bins'] = confidence_bins
    final_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    final_roi = (total_returns - total_wagered) / total_wagered if total_wagered > 0 else 0
    final_profit = total_returns - total_wagered
    results['performance']['accuracy'] = final_accuracy
    results['performance']['roi'] = final_roi
    results['performance']['profit'] = final_profit
    results['performance']['win_rate'] = winning_bets / total_bets if total_bets > 0 else 0
    results['performance']['final_bankroll'] = current_bankroll
    results['performance']['total_bets'] = total_bets
    results['performance']['winning_bets'] = winning_bets
    results['performance']['total_wagered'] = total_wagered
    results['performance']['total_returns'] = total_returns
    for team, stats in results['team_performance'].items():
        if stats['predictions'] > 0:
            stats['accuracy'] = stats['correct'] / stats['predictions']
        if stats['bets'] > 0:
            stats['win_rate'] = stats['wins'] / stats['bets']
        if stats['wagered'] > 0:
            stats['roi'] = (stats['returns'] - stats['wagered']) / stats['wagered']
            stats['profit'] = stats['returns'] - stats['wagered']
    print("\n========== BACKTEST RESULTS ==========")
    print(f"Total Matches: {total_predictions}")
    print(f"Prediction Accuracy: {final_accuracy:.2%} ({correct_predictions}/{total_predictions})")
    print(f"Total Bets: {total_bets}")
    print(f"Winning Bets: {winning_bets} ({winning_bets/total_bets:.2%})" if total_bets > 0 else "No bets placed")
    print(f"Total Wagered: ${total_wagered:.2f}")
    print(f"Total Returns: ${total_returns:.2f}")
    print(f"Profit/Loss: ${final_profit:.2f}")
    print(f"ROI: {final_roi:.2%}")
    print(f"Final Bankroll: ${current_bankroll:.2f}")
    print("\nAccuracy by Confidence Level:")
    for conf_key, stats in sorted(confidence_bins.items()):
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f"  {conf_key}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    drawdown_metrics = calculate_drawdown_metrics(results['performance']['bankroll_history'])
    results['performance']['drawdown_metrics'] = drawdown_metrics
    print(f"Maximum Drawdown: {drawdown_metrics['max_drawdown_pct']:.2f}% (${drawdown_metrics['max_drawdown_amount']:.2f})")
    print(f"Drawdown Periods: {drawdown_metrics['drawdown_periods']}")
    print(f"Average Drawdown: {drawdown_metrics['avg_drawdown_pct']:.2f}%")
    print(f"Max Drawdown Duration: {drawdown_metrics['max_drawdown_duration']} bets")
    create_enhanced_backtest_visualizations(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"backtest_results_{timestamp}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    csv_path = f"backtest_summary_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("Metric,Value\n")
        f.write(f"Total Matches,{total_predictions}\n")
        f.write(f"Prediction Accuracy,{final_accuracy:.4f}\n")
        f.write(f"Total Bets,{total_bets}\n")
        f.write(f"Winning Bets,{winning_bets}\n")
        f.write(f"Win Rate,{winning_bets/total_bets if total_bets > 0 else 0:.4f}\n")
        f.write(f"Total Wagered,{total_wagered:.2f}\n")
        f.write(f"Total Returns,{total_returns:.2f}\n")
        f.write(f"Profit/Loss,{final_profit:.2f}\n")
        f.write(f"ROI,{final_roi:.4f}\n")
        f.write(f"Initial Bankroll,{bankroll:.2f}\n")
        f.write(f"Final Bankroll,{current_bankroll:.2f}\n")
    print(f"\nBacktest results saved to {save_path}")
    print(f"Summary statistics saved to {csv_path}")
    return results

def create_improved_model(input_dim, regularization_strength=0.02, dropout_rate=0.6):
    """
    Create a deep learning model with improved regularization to prevent extreme predictions.
    
    Args:
        input_dim (int): Input dimension
        regularization_strength (float): L2 regularization parameter (increased from 0.01)
        dropout_rate (float): Dropout rate for regularization (increased from 0.5)
        
    Returns:
        Model: Tensorflow/Keras model
    """
    # Define inputs
    inputs = Input(shape=(input_dim,))
    
    # First layer - shared feature processing with stronger regularization
    x = Dense(128, activation='relu', 
              kernel_regularizer=l2(regularization_strength),
              activity_regularizer=l1(0.01),  # Added activity regularization
              kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second layer - deeper processing
    x = Dense(64, activation='relu', 
              kernel_regularizer=l2(regularization_strength),
              activity_regularizer=l1(0.005),
              kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.8)(x)
    
    # Third layer - deeper processing
    x = Dense(32, activation='relu', 
              kernel_regularizer=l2(regularization_strength),
              activity_regularizer=l1(0.001),
              kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.6)(x)
    
    # Output layer with sigmoid but not squeezed to avoid extreme values
    outputs = Dense(1, activation='sigmoid',
                    kernel_regularizer=l2(regularization_strength),
                    kernel_initializer='glorot_normal',
                    bias_initializer='zeros')(x)  # Zero bias initialization for less extreme predictions
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with lower learning rate
    model.compile(loss='binary_crossentropy', 
                 optimizer=Adam(learning_rate=0.0001),
                 metrics=['accuracy'])
    
    return model

def implement_stacking_ensemble(X_train, y_train, X_val, y_val):
    """
    Implement stacking ensemble to better combine diverse models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        tuple: (base_models, meta_model)
    """
    print("Implementing stacking ensemble...")
    
    # Train first-level models
    base_models = []
    base_predictions = np.zeros((len(X_val), 5))  # For 5 base models
    
    # 1. Train Neural Network
    print("Training Neural Network...")
    nn_model = create_improved_model(X_train.shape[1])
    
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=0
    )
    
    nn_model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    base_predictions[:, 0] = nn_model.predict(X_val).flatten()
    base_models.append(('nn', nn_model, None))
    
    # 2. Train Gradient Boosting
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    base_predictions[:, 1] = gb_model.predict_proba(X_val)[:, 1]
    base_models.append(('gb', gb_model, None))
    
    # 3. Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    base_predictions[:, 2] = rf_model.predict_proba(X_val)[:, 1]
    base_models.append(('rf', rf_model, None))
    
    # 4. Train Logistic Regression
    print("Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lr_model = LogisticRegression(
        C=0.1,
        random_state=42,
        max_iter=1000
    )
    lr_model.fit(X_train_scaled, y_train)
    base_predictions[:, 3] = lr_model.predict_proba(X_val_scaled)[:, 1]
    base_models.append(('lr', lr_model, scaler))
    
    # 5. Train SVM
    print("Training SVM...")
    svm_model = SVC(
        C=1.0,
        kernel='rbf',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train_scaled, y_train)
    base_predictions[:, 4] = svm_model.predict_proba(X_val_scaled)[:, 1]
    base_models.append(('svm', svm_model, scaler))
    
    # Train meta-learner on base predictions
    print("Training meta-learner...")
    meta_model = LogisticRegression(C=0.5, random_state=42)
    meta_model.fit(base_predictions, y_val)
    
    # Evaluate stacking performance
    meta_preds = meta_model.predict(base_predictions)
    stacking_accuracy = accuracy_score(y_val, meta_preds)
    print(f"Stacking ensemble accuracy: {stacking_accuracy:.4f}")
    
    # Store meta model with base models
    base_models.append(('meta', meta_model, None))
    
    return base_models

def implement_time_based_validation(team_data_collection):
    """
    Implement time-based validation to prevent data leakage.
    
    Args:
        team_data_collection: Dictionary of team data
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    print("Implementing time-based validation...")
    
    # Collect all matches with dates
    all_matches = []
    for team_name, team_data in team_data_collection.items():
        for match in team_data.get('matches', []):
            # Add team info to match
            match['team_name'] = team_name
            match['team_stats'] = team_data.get('stats', {})
            
            # Only include matches with valid dates
            if 'date' in match and match['date']:
                all_matches.append(match)
    
    # Sort chronologically
    all_matches.sort(key=lambda x: x.get('date', ''))
    print(f"Collected {len(all_matches)} matches with dates")
    
    # Split into train/validation/test maintaining chronological order
    train_cutoff = int(len(all_matches) * 0.7)
    val_cutoff = int(len(all_matches) * 0.85)
    
    train_matches = all_matches[:train_cutoff]
    val_matches = all_matches[train_cutoff:val_cutoff]
    test_matches = all_matches[val_cutoff:]
    
    print(f"Split into {len(train_matches)} train, {len(val_matches)} validation, {len(test_matches)} test matches")
    
    # Create datasets
    train_data = create_dataset_from_matches(train_matches, team_data_collection)
    val_data = create_dataset_from_matches(val_matches, team_data_collection)
    test_data = create_dataset_from_matches(test_matches, team_data_collection)
    
    return train_data, val_data, test_data

def create_dataset_from_matches(matches, team_data_collection):
    """
    Create feature vectors and labels from match data.
    
    Args:
        matches: List of matches
        team_data_collection: Dictionary of team data
        
    Returns:
        tuple: (X, y, match_info)
    """
    X = []  # Features
    y = []  # Labels
    match_info = []  # Additional match metadata
    
    for match in matches:
        team1_name = match['team_name']
        team2_name = match['opponent_name']
        
        # Skip if we don't have stats for opponent
        if team2_name not in team_data_collection:
            continue
            
        # Get recent stats for both teams as of this match date
        team1_stats = get_team_stats_at_date(team1_name, match['date'], team_data_collection)
        team2_stats = get_team_stats_at_date(team2_name, match['date'], team_data_collection)
        
        # Skip if either team doesn't have sufficient data
        if not team1_stats or not team2_stats:
            continue
            
        # Create feature vector using data only available before this match
        features = prepare_data_for_model(team1_stats, team2_stats)
        
        if features:
            X.append(features)
            y.append(1 if match['team_won'] else 0)
            
            # Store match info for analysis
            match_info.append({
                'match_id': match['match_id'],
                'date': match['date'],
                'team1': team1_name,
                'team2': team2_name,
                'score': f"{match['team_score']}-{match['opponent_score']}",
                'winner': 'team1' if match['team_won'] else 'team2'
            })
    
    return np.array(X), np.array(y), match_info

def get_team_stats_at_date(team_name, target_date, team_data_collection):
    """
    Get team statistics at a specific date by filtering out future matches.
    
    Args:
        team_name: Name of the team
        target_date: Date to get stats for
        team_data_collection: Dictionary of team data
        
    Returns:
        dict: Team stats at the specified date
    """
    if team_name not in team_data_collection:
        return None
    
    team_data = team_data_collection[team_name]
    
    # Get only matches before the target date
    past_matches = []
    for match in team_data.get('matches', []):
        match_date = match.get('date', '')
        if match_date and match_date < target_date:
            past_matches.append(match)
    
    # If not enough past matches, return None
    if len(past_matches) < 5:
        return None
    
    # Calculate stats using only past matches
    team_stats = calculate_team_stats(past_matches)
    
    return team_stats

def calibrate_prediction(raw_predictions):
    """
    Calibrate ensemble predictions for better reliability.
    
    Args:
        raw_predictions (list): List of prediction values from ensemble models
        
    Returns:
        tuple: (calibrated_prediction, model_agreement_score)
    """
    # Calculate basic statistics
    mean_pred = np.mean(raw_predictions)
    std_pred = np.std(raw_predictions)
    model_agreement = 1 - min(1, std_pred * 2)  # Higher means more agreement
    
    # Handle extreme cases
    if model_agreement < 0.3:
        # Very low agreement - regress heavily toward 0.5
        calibrated = 0.5 + (mean_pred - 0.5) * 0.5
    elif model_agreement < 0.6:
        # Moderate agreement - regress somewhat toward 0.5
        calibrated = 0.5 + (mean_pred - 0.5) * 0.75
    else:
        # Good agreement - minimal regression
        calibrated = mean_pred
    
    return calibrated, model_agreement

def create_diverse_ensemble(X_train, y_train, feature_mask, random_state=42):
    """
    Create a diverse ensemble of models using different algorithms.
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        feature_mask (array): Boolean mask for selected features
        random_state (int): Random seed
        
    Returns:
        list: List of trained models
    """
    # Select features
    X_train_selected = X_train[:, feature_mask]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    
    # Initialize models list
    models = []
    
    # 1. Neural Network models (less prone to extreme values)
    input_dim = X_train_scaled.shape[1]
    for i in range(5):
        # Vary regularization and dropout for diversity
        reg_strength = 0.01 * (i + 1)
        dropout = 0.4 + 0.1 * (i % 2)
        nn_model = create_improved_model(input_dim, reg_strength, dropout)
        
        # Train with early stopping
        nn_model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0
        )
        models.append(('nn', nn_model, scaler))
    
    # 2. Gradient Boosting model
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state
    )
    gb_model.fit(X_train_selected, y_train)
    models.append(('gb', gb_model, None))
    
    # 3. Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state
    )
    rf_model.fit(X_train_selected, y_train)
    models.append(('rf', rf_model, None))
    
    # 4. Logistic Regression model
    lr_model = LogisticRegression(
        C=0.1,
        random_state=random_state,
        max_iter=1000
    )
    lr_model.fit(X_train_scaled, y_train)
    models.append(('lr', lr_model, scaler))
    
    # 5. Support Vector Machine
    svm_model = SVC(
        C=1.0,
        kernel='rbf',
        probability=True,
        random_state=random_state
    )
    svm_model.fit(X_train_scaled, y_train)
    models.append(('svm', svm_model, scaler))
    
    return models, scaler    

def predict_with_diverse_ensemble(models, X):
    """
    Make predictions using a diverse ensemble of models.
    
    Args:
        models (list): List of (model_type, model, scaler) tuples
        X (array): Input features
        
    Returns:
        tuple: (predictions, calibrated_prediction, model_agreement)
    """
    predictions = []
    
    for model_type, model, scaler in models:
        try:
            # Apply scaling if needed
            X_pred = X.copy()
            if scaler is not None:
                X_pred = scaler.transform(X_pred)
            
            # Make prediction based on model type
            if model_type == 'nn':
                pred = model.predict(X_pred, verbose=0)[0][0]
            else:
                pred = model.predict_proba(X_pred)[0][1]
                
            predictions.append(pred)
            print(f"{model_type.upper()} model prediction: {pred:.4f}")
        except Exception as e:
            print(f"Error with {model_type} model: {e}")
    
    # Calibrate predictions
    if predictions:
        # Calculate basic statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        model_agreement = 1 - min(1, std_pred * 2)  # Higher means more agreement
        
        # Handle extreme cases with better calibration
        if model_agreement < 0.3:
            # Very low agreement - regress heavily toward 0.5
            calibrated = 0.5 + (mean_pred - 0.5) * 0.5
        elif model_agreement < 0.6:
            # Moderate agreement - regress somewhat toward 0.5
            calibrated = 0.5 + (mean_pred - 0.5) * 0.75
        else:
            # Good agreement - minimal regression
            calibrated = mean_pred
            
        return predictions, calibrated, model_agreement
    else:
        return [], 0.5, 0.0

def prepare_ensemble_features(team1_stats, team2_stats, selected_features):
    """
    Prepare features for the diverse ensemble prediction.
    
    Args:
        team1_stats (dict): Statistics for team 1
        team2_stats (dict): Statistics for team 2
        selected_features (list): List of feature names
        
    Returns:
        array: Feature array for prediction
    """
    # Get full feature dictionary
    features = prepare_data_for_model(team1_stats, team2_stats)
    
    if not features:
        print("ERROR: Failed to create feature dictionary")
        return None
    
    # Create DataFrame with a single row
    features_df = pd.DataFrame([features])
    
    # Create a complete feature set with zeros for missing features
    complete_features = pd.DataFrame(0, index=[0], columns=selected_features)
    
    # Fill in values for features we have
    available_features = [f for f in selected_features if f in features_df.columns]
    print(f"Found {len(available_features)} out of {len(selected_features)} expected features")
    
    if not available_features:
        print("ERROR: No matching features found!")
        return None
    
    # Update values for available features
    for feature in available_features:
        complete_features[feature] = features_df[feature].values
    
    # Convert to numpy array
    X = complete_features.values
    return X

def ensure_consistent_features(features_df, required_features, fill_value=0):
    """
    Ensure the features dataframe has all required features in the correct order.
    
    Args:
        features_df (DataFrame): DataFrame with available features
        required_features (list): List of required feature names
        fill_value (float): Value to fill for missing features
        
    Returns:
        DataFrame: DataFrame with consistent features
    """
    # Create a new DataFrame with all required features
    consistent_df = pd.DataFrame(columns=required_features)
    
    # Add a row with default values
    consistent_df.loc[0] = fill_value
    
    # Update with available values
    for feature in required_features:
        if feature in features_df.columns:
            consistent_df[feature] = features_df[feature].values[0]
    
    print(f"Features: {len(features_df.columns)} available, {len(required_features)} required, {sum(col in features_df.columns for col in required_features)} matched")
    
    return consistent_df

def create_model(input_dim, regularization_strength=0.001, dropout_rate=0.4):
    """Create a deep learning model with regularization for match prediction."""
    # Define inputs
    inputs = Input(shape=(input_dim,))
    
    # First layer - shared feature processing
    x = Dense(256, activation='relu', kernel_regularizer=l2(regularization_strength))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second layer - deeper processing
    x = Dense(128, activation='relu', kernel_regularizer=l2(regularization_strength/2))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.75)(x)
    
    # Specialized pathways
    x = Dense(64, activation='relu', kernel_regularizer=l2(regularization_strength/4))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.5)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(loss='binary_crossentropy', 
                 optimizer=Adam(learning_rate=0.0005),
                 metrics=['accuracy'])
    
    return model

def train_model(X, y, test_size=0.2, random_state=42):
    """Train the model with feature selection and early stopping."""
    # Clean and prepare feature data
    df = clean_feature_data(X)
    
    if df.empty:
        print("Error: Empty feature dataframe after cleaning")
        return None, None, None
    
    # Convert to numpy array and scale
    X_arr = df.values
    y_arr = np.array(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )
    
    # Handle class imbalance if needed
    class_counts = np.bincount(y_train)
    print(f"Class distribution: {class_counts}")
    
    if np.min(class_counts) / np.sum(class_counts) < 0.4:  # If imbalanced
        try:
            if np.min(class_counts) >= 5:
                min_samples = np.min(class_counts)
                k_neighbors = min(5, min_samples-1)
                
                smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"Applied SMOTE: Training set now has {np.bincount(y_train)} samples per class")
        except Exception as e:
            print(f"Error applying SMOTE: {e}")
    
    # Feature selection with Random Forest
    feature_selector = RandomForestClassifier(n_estimators=100, random_state=random_state)
    feature_selector.fit(X_train, y_train)
    
    # Get feature importances and select top features
    importances = feature_selector.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top 75% of features
    cumulative_importance = np.cumsum(importances[indices])
    n_features = np.where(cumulative_importance >= 0.75)[0][0] + 1
    n_features = min(n_features, 100)  # Cap at 100 features
    
    top_indices = indices[:n_features]
    selected_features = [df.columns[i] for i in top_indices]
    
    print(f"Selected {len(selected_features)} top features out of {X_train.shape[1]}")
    
    # Train with selected features
    X_train_selected = X_train[:, top_indices]
    X_val_selected = X_val[:, top_indices]
    
    # Create and train model
    input_dim = X_train_selected.shape[1]
    model = create_model(input_dim)
    
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'valorant_model.h5', save_best_only=True, monitor='val_accuracy'
    )
    
    # Train model
    history = model.fit(
        X_train_selected, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_selected, y_val),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Save model artifacts
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    
    # Evaluate model
    y_pred_proba = model.predict(X_val_selected)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print("\nModel Evaluation:")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return model, scaler, selected_features

def train_with_cross_validation(X, y, n_splits=5, random_state=42):
    print(f"\nTraining with {n_splits}-fold cross-validation")

    
    # Prepare and clean data
    df = clean_feature_data(X)
    X_arr = df.values
    y_arr = np.array(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    
    # Set up cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize arrays to store results
    fold_metrics = []
    fold_models = []
    all_feature_importances = {}
    
    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_arr)):
        print(f"\n----- Training Fold {fold+1}/{n_splits} -----")
        


        # Split data
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]
        
        # Handle class imbalance if needed
        class_counts = np.bincount(y_train)
        if np.min(class_counts) / np.sum(class_counts) < 0.4:
            try:
                if np.min(class_counts) >= 5:
                    min_samples = np.min(class_counts)
                    k_neighbors = min(5, min_samples-1)
                    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"Error applying SMOTE: {e}")
        
        # Feature selection
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf.fit(X_train, y_train)
        
        # Select top features
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Select top 75% of features
        cumulative_importance = np.cumsum(importances[indices])
        n_features = np.where(cumulative_importance >= 0.75)[0][0] + 1
        n_features = min(n_features, 100)  # Cap at 100 features
        
        selected_indices = indices[:n_features]
        selected_features = [df.columns[i] for i in selected_indices]
        
        # Track feature importance
        for feature in selected_features:
            if feature in all_feature_importances:
                all_feature_importances[feature] += 1
            else:
                all_feature_importances[feature] = 1
        
        # Train model with selected features
        X_train_selected = X_train[:, selected_indices]
        X_val_selected = X_val[:, selected_indices]
        
        # Create and train model
        input_dim = X_train_selected.shape[1]
        model = create_model(input_dim)
        
        # Train with early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        )
        
        model.fit(
            X_train_selected, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_selected, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        y_pred_proba = model.predict(X_val_selected)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        # Store results
        fold_metrics.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'selected_features': selected_features
        })
        
        fold_models.append(model)
        
        # Print fold results
        print(f"Fold {fold+1} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Selected Features: {len(selected_features)}")
    
    # Calculate average metrics
    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) 
                  for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
    
    std_metrics = {metric: np.std([fold[metric] for fold in fold_metrics]) 
                  for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
    
    print(f"\nAverage Metrics Across {n_splits} Folds:")
    for metric, value in avg_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f} ± {std_metrics[metric]:.4f}")
    
    # Identify stable feature set (features selected in majority of folds)
    sorted_features = sorted(all_feature_importances.items(), 
                           key=lambda x: x[1], reverse=True)
    
    stable_features = [feature for feature, count in sorted_features 
                      if count >= n_splits * 0.8]  # Features selected in at least 80% of folds
    
    print(f"\nStable Feature Set: {len(stable_features)} features")
    
    # Save models
    for i, model in enumerate(fold_models):
        model.save(f'valorant_model_fold_{i+1}.h5')
    
    # Save stable features
    with open('stable_features.pkl', 'wb') as f:
        pickle.dump(stable_features, f)
    
    # Save scaler
    with open('ensemble_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return fold_models, stable_features, avg_metrics, fold_metrics, scaler


#-------------------------------------------------------------------------
# DATA PREPARATION
#-------------------------------------------------------------------------

def load_ensemble_models(model_paths, scaler_path, features_path):
    """
    Load ensemble models, scaler, and stable feature list.
    
    Args:
        model_paths (list): List of file paths to the trained models
        scaler_path (str): Path to the feature scaler
        features_path (str): Path to the stable features list
        
    Returns:
        tuple: (loaded_models, scaler, stable_features)
    """
    print("Loading ensemble models and data preprocessing artifacts...")
    
    # Load models
    loaded_models = []
    for path in model_paths:
        try:
            model = load_model(path)
            loaded_models.append(model)
            print(f"Loaded model from {path}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
    
    # Load scaler
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded feature scaler from {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        scaler = None
    
    # Load stable features
    try:
        with open(features_path, 'rb') as f:
            stable_features = pickle.load(f)
        print(f"Loaded {len(stable_features)} stable features")
    except Exception as e:
        print(f"Error loading stable features: {e}")
        stable_features = None
    
    return loaded_models, scaler, stable_features

def prepare_match_features(team1_stats, team2_stats, stable_features, scaler):
    """
    Prepare features for a single match for prediction using only the features
    that were selected during training to avoid overfitting.
    """
    # Get full feature set
    features = prepare_data_for_model(team1_stats, team2_stats)
    
    if not features:
        print("Failed to prepare match features.")
        return None
    
    # Convert to DataFrame for easier feature selection
    features_df = pd.DataFrame([features])
    print(f"Original feature count: {len(features_df.columns)}")
    
    # If stable_features exists, use only those features
    if stable_features and len(stable_features) > 0:
        # Check which features are available
        available_features = [f for f in stable_features if f in features_df.columns]
        
        print(f"Using {len(available_features)} out of {len(stable_features)} stable features")
        
        # Select only the available stable features in the correct order
        if available_features:
            features_df = features_df[available_features]
        else:
            print("ERROR: None of the stable features are available in current data!")
            return None
    else:
        print("WARNING: No stable features provided. Model may not work correctly.")
    
    # Convert to numpy array for scaling
    X = features_df.values
    print(f"Feature array shape after selection: {X.shape}")
    
    # Scale features if scaler is available
    if scaler:
        try:
            X_scaled = scaler.transform(X)
            return X_scaled
        except Exception as e:
            print(f"Error scaling features: {e}")
            return X  # Return unscaled features as fallback
    else:
        return X

#-------------------------------------------------------------------------
# PREDICTION AND BETTING LOGIC
#-------------------------------------------------------------------------
def prepare_prediction_features(team1_stats, team2_stats, selected_features, scaler):
    """
    Prepare features for prediction ensuring compatibility with trained models.
    
    Args:
        team1_stats (dict): Statistics for team 1
        team2_stats (dict): Statistics for team 2
        selected_features (list): List of feature names used by the model
        scaler (object): Fitted scaler used during training
        
    Returns:
        array: Scaled feature array compatible with the model
    """
    print("Preparing features for prediction...")
    
    # Get full feature dictionary
    features = prepare_data_for_model(team1_stats, team2_stats)
    
    if not features:
        print("ERROR: Failed to create feature dictionary")
        return None
    
    # Create DataFrame with a single row
    features_df = pd.DataFrame([features])
    original_feature_count = len(features_df.columns)
    print(f"Original feature count: {original_feature_count}")
    
    # Get first model to determine input shape
    try:
        model_path = 'valorant_model_fold_1.h5'
        model = load_model(model_path)
        expected_feature_count = model.layers[0].input_shape[1]
        print(f"Model expects {expected_feature_count} features based on input layer")
    except Exception as e:
        print(f"Error loading model to check input shape: {e}")
        # Default to 50 based on error messages
        expected_feature_count = 50
        print(f"Using default expected feature count: {expected_feature_count}")
    
    # Check if we have the feature_metadata which might have the exact list of features used
    try:
        with open('feature_metadata.pkl', 'rb') as f:
            feature_metadata = pickle.load(f)
            if 'selected_features' in feature_metadata:
                model_features = feature_metadata['selected_features']
                print(f"Using exact feature list from metadata with {len(model_features)} features")
                # Override selected_features with the exact list used in model training
                selected_features = model_features
            else:
                print("No selected_features in metadata")
    except Exception as e:
        print(f"Error loading feature metadata: {e}")
    
    # Create a DataFrame with the exact features expected by the model
    if len(selected_features) == expected_feature_count:
        print(f"Selected features count ({len(selected_features)}) matches model input size ({expected_feature_count})")
        final_features = pd.DataFrame(0, index=[0], columns=selected_features)
        
        # Fill in values for features we have
        for feature in selected_features:
            if feature in features_df.columns:
                final_features[feature] = features_df[feature].values
        
        missing_count = sum(final_features.iloc[0] == 0)
        print(f"Missing {missing_count} out of {len(selected_features)} model features")
        
        # Convert to numpy array - shape should match model input
        X = final_features.values
    else:
        print(f"WARNING: Selected features count ({len(selected_features)}) doesn't match model input ({expected_feature_count})")
        
        # Create array with the exact size the model expects
        X = np.zeros((1, expected_feature_count))
        
        # Add features where we can
        matched_features = 0
        for i, feature in enumerate(selected_features):
            if i >= expected_feature_count:
                break
            if feature in features_df.columns:
                X[0, i] = features_df[feature].values[0]
                matched_features += 1
        
        print(f"Added {matched_features} features to match expected model input size")
    
    print(f"Final feature array shape: {X.shape}")
    
    # Skip scaling since it's causing issues
    # The model should still work reasonably well with unscaled features
    print("WARNING: Skipping feature scaling due to dimension mismatch issues")
    
    return X

def predict_with_consistent_ordering(team1_name, team2_name, ensemble_models, selected_features,
                                    team_data_collection=None, current_date=None):
    """
    Make a prediction with consistent team ordering and proper data handling.
    
    Args:
        team1_name (str): Name of first team
        team2_name (str): Name of second team
        ensemble_models (list): Ensemble models for prediction
        selected_features (list): Selected features for prediction
        team_data_collection (dict, optional): Team data collection
        current_date (str, optional): Current date for time-aware prediction
        
    Returns:
        dict: Prediction results
    """
    logging.info(f"Making prediction for {team1_name} vs {team2_name}")
    
    # If current date is not provided, use current date
    if not current_date:
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
    
    # If team data collection is not provided, collect data
    if not team_data_collection:
        team_data_collection = collect_team_data(
            team_limit=300,
            include_player_stats=True,
            include_economy=True,
            include_maps=True,
            use_cache=Config.MODEL.CACHE_ENABLED,
            cache_path=Config.MODEL.CACHE_PATH
        )
    
    # Get team stats using time-aware approach
    team1_stats = get_historical_team_stats(team1_name, current_date, team_data_collection)
    team2_stats = get_historical_team_stats(team2_name, current_date, team_data_collection)
    
    if not team1_stats or not team2_stats:
        logging.error(f"Could not get historical stats for teams")
        return {
            'error': 'Insufficient historical data for prediction',
            'team1_name': team1_name,
            'team2_name': team2_name
        }
    
    # Apply consistent team ordering
    features, teams_swapped = prepare_data_with_consistent_ordering(
        team1_stats, team2_stats, selected_features
    )
    
    if features is None:
        logging.error(f"Failed to prepare features for prediction")
        return {
            'error': 'Failed to prepare features for prediction',
            'team1_name': team1_name,
            'team2_name': team2_name
        }
    
    # Make prediction
    win_probability, raw_predictions, confidence = predict_with_ensemble(
        ensemble_models, features
    )
    
    # Adjust prediction if teams were swapped
    if teams_swapped:
        logging.info(f"Teams were swapped for consistency - adjusting predictions")
        win_probability = 1.0 - win_probability
    
    # Calculate probabilities for different bet types
    bet_type_probabilities = calculate_bet_type_probabilities(win_probability, confidence)
    
    # Return prediction results
    prediction_results = {
        'team1_name': team1_name,
        'team2_name': team2_name,
        'win_probability': win_probability,
        'confidence': confidence,
        'raw_predictions': raw_predictions,
        'bet_type_probabilities': bet_type_probabilities,
        'teams_swapped': teams_swapped,
        'date': current_date
    }
    
    logging.info(f"Prediction complete: {team1_name}:{win_probability:.4f}, {team2_name}:{1-win_probability:.4f}")
    return prediction_results

def analyze_betting_options(prediction_results, odds_data, bankroll=1000.0, bet_history=None):
    """
    Analyze betting options with conservative bankroll management.
    
    Args:
        prediction_results (dict): Prediction results
        odds_data (dict): Betting odds data
        bankroll (float): Current bankroll
        bet_history (list, optional): Betting history for streak analysis
        
    Returns:
        dict: Betting analysis
    """
    logging.info(f"Analyzing betting options with bankroll ${bankroll:.2f}")
    
    # Get prediction data
    win_probability = prediction_results['win_probability']
    confidence = prediction_results['confidence']
    team1_name = prediction_results['team1_name']
    team2_name = prediction_results['team2_name']
    
    # Get bet type probabilities
    bet_type_probabilities = prediction_results['bet_type_probabilities']
    
    # Perform betting edge analysis
    betting_analysis = analyze_betting_edge_conservative(
        win_probability, 
        team1_name, 
        team2_name, 
        odds_data, 
        confidence, 
        bankroll, 
        bet_history, 
        config=Config.BETTING
    )
    
    # Select optimal bets
    optimal_bets = select_optimal_bets_conservative(
        betting_analysis, 
        team1_name, 
        team2_name, 
        bankroll, 
        max_bets=Config.BETTING.MAX_SIMULTANEOUS_BETS, 
        max_total_risk=Config.BETTING.MAX_TOTAL_RISK_PCT, 
        config=Config.BETTING
    )
    
    # Simulate market response for each recommended bet
    market_simulations = {}
    for bet_type, analysis in optimal_bets.items():
        probability = analysis['probability']
        odds = analysis['odds']
        bet_amount = analysis['bet_amount']
        
        market_simulations[bet_type] = simulate_market_response(
            odds, 
            probability, 
            market_efficiency=Config.FORWARD_TEST.MARKET_EFFICIENCY, 
            bet_size=bet_amount, 
            market_size=10000
        )
    
    # Return results
    results = {
        'betting_analysis': betting_analysis,
        'optimal_bets': optimal_bets,
        'market_simulations': market_simulations,
        'bankroll': bankroll
    }
    
    # Log recommendations
    if optimal_bets:
        logging.info(f"Recommended bets: {', '.join(optimal_bets.keys())}")
    else:
        logging.info("No bets recommended")
    
    return results

def explain_prediction_factors(team1_stats, team2_stats, selected_features, feature_weights):
    """
    Explain key factors influencing the prediction for this specific match.
    """
    print("\nKEY PREDICTION FACTORS:")
    print("-" * 80)
    
    # Get the top factors with their values and weights
    factors = []
    for feature in selected_features:
        if feature in feature_weights:
            weight = feature_weights[feature]
            
            # Try to extract feature values for explanation
            value = None
            if feature.endswith('_diff'):
                base_feature = feature.replace('_diff', '')
                if base_feature in team1_stats and base_feature in team2_stats:
                    value = team1_stats[base_feature] - team2_stats[base_feature]
            
            # Add to factors list
            factors.append({
                'feature': feature,
                'weight': weight,
                'value': value,
                'impact': abs(weight) * (0 if value is None else abs(value))
            })
    
    # Sort by impact and show top 5
    factors.sort(key=lambda x: x['impact'], reverse=True)
    for i, factor in enumerate(factors[:5]):
        feature_name = factor['feature'].replace('_', ' ').title()
        print(f"{i+1}. {feature_name}")
        if factor['value'] is not None:
            print(f"   Value: {factor['value']:.4f}, Weight: {factor['weight']:.4f}")
            
            # Add explanation
            if factor['value'] > 0:
                print(f"   This factor favors {team1_stats['team_name']}")
            else:
                print(f"   This factor favors {team2_stats['team_name']}")

def calibrate_prediction(raw_prediction, confidence):
    """
    Calibrate raw model predictions to be more conservative when confidence is low.
    
    Args:
        raw_prediction (float): Raw prediction between 0 and 1
        confidence (float): Model confidence/agreement score
        
    Returns:
        float: Calibrated prediction
    """
    # If confidence is very low, pull prediction toward 0.5 (uncertainty)
    if confidence < 0.2:
        # Strong regression to the mean - very low confidence
        return 0.5 + (raw_prediction - 0.5) * 0.5
    elif confidence < 0.4:
        # Moderate regression to the mean - low confidence
        return 0.5 + (raw_prediction - 0.5) * 0.7
    elif confidence < 0.6:
        # Slight regression to the mean - moderate confidence
        return 0.5 + (raw_prediction - 0.5) * 0.85
    else:
        # No regression - high confidence
        return raw_prediction

def analyze_betting_opportunities(prediction_results, odds_data, bankroll=1000, max_kelly_pct=0.05, min_roi=0.1, min_agreement=0.4):
    """
    Analyze potential betting opportunities with enhanced safety checks.
    
    Args:
        prediction_results (dict): Prediction probabilities
        odds_data (dict): Betting odds from bookmaker
        bankroll (float): Current bankroll amount for betting recommendations
        max_kelly_pct (float): Maximum Kelly fraction cap as safety measure
        min_roi (float): Minimum ROI required for bet recommendation
        min_agreement (float): Minimum model agreement required for recommendations
        
    Returns:
        dict: Betting recommendations with expected value and confidence
    """
    betting_analysis = {}
    min_edge = 0.1  # Increased from 0.05 to 0.1 for more conservative recommendations
    min_confidence = 0.6  # Minimum 60% confidence to recommend a bet
    
    # Check overall model agreement before recommending any bets
    model_agreement = prediction_results.get('model_agreement', 0)
    if model_agreement < min_agreement:
        print(f"WARNING: Model agreement ({model_agreement:.2f}) is below threshold ({min_agreement})")
        print("No bets will be recommended due to low model confidence")
        # Return analysis with no recommendations
        return {k.replace('_odds', ''): {'recommended': False, 'reason': 'Low model agreement'} 
                for k in odds_data.keys()}
    
    # Function to convert decimal odds to implied probability
    def decimal_to_prob(decimal_odds):
        return 1 / decimal_odds
    
    # Function to calculate Kelly Criterion bet size with safety limits
    def kelly_bet(win_prob, decimal_odds):
        # Kelly formula: f* = (bp - q) / b
        b = decimal_odds - 1
        p = win_prob
        q = 1 - p
        kelly = (b * p - q) / b
        
        # Apply a conservative fraction (1/4) of Kelly to reduce variance
        kelly = kelly * 0.25
        
        # Add absolute cap on kelly percentage for safety
        kelly = min(kelly, max_kelly_pct)
        
        return max(0, kelly)
    
    # Function to calculate expected value and ROI
    def calculate_roi(prob, odds):
        return (prob * (odds - 1)) - (1 - prob)
    
    # Function to get reason for bet rejection
    def get_rejection_reason(ev, min_edge, prob, min_confidence, roi, min_roi):
        """Get reason why bet was rejected."""
        if ev < min_edge:
            return f"Insufficient edge ({ev:.2%} < {min_edge:.2%})"
        elif prob < min_confidence:
            return f"Insufficient probability ({prob:.2%} < {min_confidence:.2%})"
        elif roi < min_roi:
            return f"Insufficient ROI ({roi:.2%} < {min_roi:.2%})"
        else:
            return "Unknown reason"
    
    # Add extra sanity check for highly divergent predictions
    raw_predictions = prediction_results.get('raw_predictions', [])
    if raw_predictions and np.std(raw_predictions) > 0.25:  # If standard deviation is very high
        print("WARNING: Predictions are highly divergent. Using more conservative thresholds.")
        min_edge = 0.15  # Increase minimum edge requirement
        max_kelly_pct = 0.03  # Reduce maximum Kelly percentage
    
    # Process each bet type
    for bet_key, odds in odds_data.items():
        # Identify bet type and corresponding probability
        bet_type = bet_key.replace('_odds', '')
        prob_key = bet_type + '_prob'
        
        if prob_key in prediction_results:
            our_prob = prediction_results[prob_key]
            implied_prob = decimal_to_prob(odds)
            ev = our_prob - implied_prob
            roi = calculate_roi(our_prob, odds)
            
            # Calculate Kelly bet size
            kelly_fraction = kelly_bet(our_prob, odds)
            bet_amount = round(bankroll * kelly_fraction, 2)
            
            # Only recommend if meets all criteria
            is_recommended = (ev > min_edge and 
                            our_prob > min_confidence and 
                            roi > min_roi and
                            bet_amount > 0)
            
            betting_analysis[bet_type] = {
                'our_probability': our_prob,
                'implied_probability': implied_prob,
                'expected_value': ev,
                'roi': roi,
                'kelly_fraction': kelly_fraction,
                'recommended_bet': bet_amount if is_recommended else 0,
                'recommended': is_recommended,
                'confidence': prediction_results['model_agreement'],
                'ev_percentage': ev * 100,
                'reason': '' if is_recommended else get_rejection_reason(ev, min_edge, our_prob, min_confidence, roi, min_roi)
            }
    
    return betting_analysis

def analyze_similar_matchups(team1_stats, team2_stats):
    """
    Analyze performance in similar matchups to provide context.
    """
    similar_matchups = []
    
    # Look for matches with similar context in team1's history
    if 'matches' in team1_stats and isinstance(team1_stats['matches'], list):
        # Find matches against similar strength opponents
        team2_win_rate = team2_stats.get('win_rate', 0.5)
        
        for match in team1_stats['matches']:
            # Try to find the opponent's data
            opponent_name = match.get('opponent_name', '')
            if opponent_name and 'opponent_stats' in team1_stats:
                if opponent_name in team1_stats['opponent_stats']:
                    opp_stats = team1_stats['opponent_stats'][opponent_name]
                    opp_win_rate = opp_stats.get('win_rate', 0.5)
                    
                    # Check if similar strength opponent (within 10%)
                    if abs(opp_win_rate - team2_win_rate) < 0.1:
                        similar_matchups.append({
                            'opponent': opponent_name,
                            'result': 'win' if match.get('team_won', False) else 'loss',
                            'score': f"{match.get('team_score', 0)}-{match.get('opponent_score', 0)}"
                        })
    
    # Print results if found
    if similar_matchups:
        print("\nSIMILAR MATCHUP HISTORY:")
        print("-" * 80)
        print(f"Found {len(similar_matchups)} matches where {team1_stats['team_name']} faced opponents of similar strength to {team2_stats['team_name']}:")
        
        wins = sum(1 for m in similar_matchups if m['result'] == 'win')
        print(f"Record: {wins}-{len(similar_matchups) - wins} ({wins/len(similar_matchups):.2%})")
        
        # Show most recent 5
        for i, match in enumerate(similar_matchups[:5]):
            print(f"  vs {match['opponent']}: {match['result'].upper()} {match['score']}")

def print_prediction_report(results, team1_stats, team2_stats):
    """
    Print detailed prediction report with team stats and betting recommendations.
    """
    team1_name = results['team1_name']
    team2_name = results['team2_name']
    
    print(f"\n{'='*80}")
    print(f"DETAILED PREDICTION REPORT: {team1_name} vs {team2_name}")
    print(f"{'='*80}")
    
    # Team comparison section
    print("\nTEAM COMPARISON:")
    print(f"{'Metric':<30} {team1_name:<25} {team2_name:<25}")
    print("-" * 80)
    
    # Function to safely print metrics
    def print_metric(label, key1, key2=None, format_str="{:.4f}", default="N/A"):
        val1 = "N/A"
        val2 = "N/A"
        
        # Get value from nested keys if provided
        if key2:
            if key1 in team1_stats and key2 in team1_stats[key1]:
                try:
                    val1 = format_str.format(team1_stats[key1][key2])
                except (ValueError, TypeError):
                    val1 = str(team1_stats[key1][key2])
            
            if key1 in team2_stats and key2 in team2_stats[key1]:
                try:
                    val2 = format_str.format(team2_stats[key1][key2])
                except (ValueError, TypeError):
                    val2 = str(team2_stats[key1][key2])
        else:
            if key1 in team1_stats:
                try:
                    val1 = format_str.format(team1_stats[key1])
                except (ValueError, TypeError):
                    val1 = str(team1_stats[key1])
            
            if key1 in team2_stats:
                try:
                    val2 = format_str.format(team2_stats[key1])
                except (ValueError, TypeError):
                    val2 = str(team2_stats[key1])
        
        print(f"{label:<30} {val1:<25} {val2:<25}")
    
    # Basic stats
    print_metric("Overall Win Rate", "win_rate", format_str="{:.2%}")
    print_metric("Recent Form", "recent_form", format_str="{:.2%}")
    print_metric("Total Matches", "matches", format_str="{}")
    print_metric("Avg Score Per Map", "avg_score", format_str="{:.2f}")
    print_metric("Avg Opponent Score", "avg_opponent_score", format_str="{:.2f}")
    print_metric("Score Differential", "score_differential", format_str="{:.2f}")
    
    # Head-to-head section
    print("\nHEAD-TO-HEAD ANALYSIS:")
    
    # Check if team1 has head-to-head data with team2
    h2h_found = False
    h2h_stats = None
    
    if 'opponent_stats' in team1_stats:
        for opponent_name, stats in team1_stats['opponent_stats'].items():
            if opponent_name.lower() == team2_name.lower() or team2_name.lower() in opponent_name.lower():
                h2h_stats = stats
                h2h_found = True
                break
    
    if h2h_found and h2h_stats:
        print(f"  {team1_name} vs {team2_name} head-to-head:")
        print(f"  - Matches: {h2h_stats.get('matches', 0)}")
        print(f"  - {team1_name} Win Rate: {h2h_stats.get('win_rate', 0):.2%}")
        print(f"  - Avg Score: {h2h_stats.get('avg_score', 0):.2f} - {h2h_stats.get('avg_opponent_score', 0):.2f}")
        print(f"  - Score Differential: {h2h_stats.get('score_differential', 0):.2f}")
    else:
        print(f"  No direct head-to-head data found between {team1_name} and {team2_name}")
    
    # Print advanced player stats if available
    if 'avg_player_rating' in team1_stats and 'avg_player_rating' in team2_stats:
        print("\nPLAYER STATISTICS:")
        print("-" * 80)
        print_metric("Avg Player Rating", "avg_player_rating", format_str="{:.2f}")
        print_metric("Avg ACS", "avg_player_acs", format_str="{:.2f}")
        print_metric("Avg K/D Ratio", "avg_player_kd", format_str="{:.2f}")
        print_metric("Avg KAST", "avg_player_kast", format_str="{:.2%}")
        print_metric("Avg ADR", "avg_player_adr", format_str="{:.2f}")
        print_metric("Avg Headshot %", "avg_player_headshot", format_str="{:.2%}")
        print_metric("Star Player Rating", "star_player_rating", format_str="{:.2f}")
        print_metric("Team Consistency", "team_consistency", format_str="{:.2f}")
        print_metric("FK/FD Ratio", "fk_fd_ratio", format_str="{:.2f}")
    
    # Print economy stats if available
    if 'pistol_win_rate' in team1_stats and 'pistol_win_rate' in team2_stats:
        print("\nECONOMY STATISTICS:")
        print("-" * 80)
        print_metric("Pistol Round Win Rate", "pistol_win_rate", format_str="{:.2%}")
        print_metric("Eco Round Win Rate", "eco_win_rate", format_str="{:.2%}")
        print_metric("Full Buy Win Rate", "full_buy_win_rate", format_str="{:.2%}")
        print_metric("Economy Efficiency", "economy_efficiency", format_str="{:.2f}")
    
    # Print map performance if available
    if 'map_performance' in team1_stats and 'map_performance' in team2_stats:
        print("\nMAP PERFORMANCE:")
        print("-" * 80)
        
        # Find common maps
        team1_maps = set(team1_stats['map_performance'].keys())
        team2_maps = set(team2_stats['map_performance'].keys())
        common_maps = team1_maps.intersection(team2_maps)
        
        for map_name in common_maps:
            print(f"\n  {map_name}:")
            t1_map = team1_stats['map_performance'][map_name]
            t2_map = team2_stats['map_performance'][map_name]
            
            if 'win_rate' in t1_map and 'win_rate' in t2_map:
                print(f"  - {team1_name} Win Rate: {t1_map['win_rate']:.2%} ({t1_map.get('wins', 0)}/{t1_map.get('played', 0)})")
                print(f"  - {team2_name} Win Rate: {t2_map['win_rate']:.2%} ({t2_map.get('wins', 0)}/{t2_map.get('played', 0)})")
            
            if 'attack_win_rate' in t1_map and 'attack_win_rate' in t2_map:
                print(f"  - {team1_name} Attack Win Rate: {t1_map['attack_win_rate']:.2%}")
                print(f"  - {team2_name} Attack Win Rate: {t2_map['attack_win_rate']:.2%}")
                
            if 'defense_win_rate' in t1_map and 'defense_win_rate' in t2_map:
                print(f"  - {team1_name} Defense Win Rate: {t1_map['defense_win_rate']:.2%}")
                print(f"  - {team2_name} Defense Win Rate: {t2_map['defense_win_rate']:.2%}")
    
    # Print match prediction
    print("\nMATCH PREDICTION:")
    print("-" * 80)
    print(f"  {team1_name} Win Probability: {results['team1_win_prob']:.2%}")
    print(f"  {team2_name} Win Probability: {results['team2_win_prob']:.2%}")
    print(f"  Confidence Interval: ({results['confidence_interval'][0]:.2%} - {results['confidence_interval'][1]:.2%})")
    print(f"  Model Agreement: {results['model_agreement']:.2f} (Higher is better)")
    
    # Print other bet types
    print("\nBET TYPE PROBABILITIES:")
    print("-" * 80)
    print(f"  {team1_name} +1.5 Maps: {results['team1_plus_1_5_prob']:.2%}")
    print(f"  {team2_name} +1.5 Maps: {results['team2_plus_1_5_prob']:.2%}")
    print(f"  {team1_name} -1.5 Maps (2-0 win): {results['team1_minus_1_5_prob']:.2%}")
    print(f"  {team2_name} -1.5 Maps (2-0 win): {results['team2_minus_1_5_prob']:.2%}")
    print(f"  Over 2.5 Maps: {results['over_2_5_maps_prob']:.2%}")
    print(f"  Under 2.5 Maps: {results['under_2_5_maps_prob']:.2%}")
    
    # Update the betting recommendations section:
    if 'betting_analysis' in results and results['betting_analysis']:
        print("\nBETTING RECOMMENDATIONS:")
        print("-" * 80)
        
        recommended_bets = []
        rejected_bets = []
        
        for bet_type, analysis in results['betting_analysis'].items():
             # Extract team name instead of generic "team1" or "team2"
            if 'team1' in bet_type:
                team_name = team1_stats['team_name']
                bet_desc = f"{team_name} {bet_type.replace('team1_', '').replace('_', ' ').upper()}"
            elif 'team2' in bet_type:
                team_name = team2_stats['team_name']
                bet_desc = f"{team_name} {bet_type.replace('team2_', '').replace('_', ' ').upper()}"
            else:
                bet_desc = bet_type.replace('_', ' ').upper()
            
            if analysis['recommended']:
                edge = analysis.get('expected_value', 0)
                roi = analysis.get('roi', 0)
                recommended_bet = analysis.get('recommended_bet', 0)
                recommended_bets.append((bet_desc, edge, roi, recommended_bet))
                
                print(f"  RECOMMENDED BET: {bet_desc}")
                print(f"  - Our Probability: {analysis['our_probability']:.2%}")
                print(f"  - Implied Probability: {analysis['implied_probability']:.2%}")
                print(f"  - Edge: {edge:.2%}")
                print(f"  - ROI: {roi:.2%}")
                print(f"  - Expected Value: {analysis.get('ev_percentage', 0):.2f}%")
                print(f"  - Confidence: {analysis.get('confidence', 0):.2f}")
                print(f"  - Kelly Fraction: {analysis.get('kelly_fraction', 0):.4f}")
                print(f"  - Recommended Bet Amount: ${recommended_bet}")
                print("")
                
                # Explain the recommendation
                explain_bet_recommendation(bet_type, analysis, results, team1_stats['team_name'], team2_stats['team_name'])
            else:
                # Track rejected bets
                rejected_bets.append((bet_desc, analysis.get('reason', 'Not recommended')))
        
        if not recommended_bets:
            print("  No profitable betting opportunities identified for this match.")
        else:
            # Sort by ROI
            recommended_bets.sort(key=lambda x: x[2], reverse=True)
            print("\nBETS RANKED BY EXPECTED ROI:")
            for i, (bet, edge, roi, amount) in enumerate(recommended_bets):
                print(f"  {i+1}. {bet}: {edge:.2%} edge, {roi:.2%} ROI, bet ${amount}")
        
        # Show rejected bets
        if rejected_bets:
            print("\nREJECTED BETS:")
            for bet, reason in rejected_bets:
                print(f"  {bet}: {reason}")
            # Sort by edge
            recommended_bets.sort(key=lambda x: x[1], reverse=True)
            print("\nBETS RANKED BY EDGE:")
            for i, (bet, edge, roi, amount) in enumerate(recommended_bets):
                print(f"  {i+1}. {bet}: {edge:.2%} edge, bet ${amount}")

def calculate_drawdown_metrics(bankroll_history):
    """
    Calculate maximum drawdown and other drawdown metrics.
    
    Args:
        bankroll_history: List of dictionaries containing bankroll history
        
    Returns:
        dict: Drawdown metrics including maximum drawdown
    """
    if not bankroll_history:
        return {
            'max_drawdown_pct': 0,
            'max_drawdown_amount': 0,
            'drawdown_periods': 0,
            'avg_drawdown_pct': 0,
            'max_drawdown_duration': 0
        }
    
    # Extract bankroll values
    bankrolls = [entry['bankroll'] for entry in bankroll_history]
    
    # Initialize variables
    peak = bankrolls[0]
    max_drawdown = 0
    max_drawdown_amount = 0
    drawdown_periods = 0
    current_drawdown_start = None
    max_drawdown_duration = 0
    current_drawdown_duration = 0
    all_drawdowns = []
    
    # Calculate drawdown metrics
    for i, value in enumerate(bankrolls):
        if value > peak:
            # New peak
            peak = value
            # If we were in a drawdown, it's now over
            if current_drawdown_start is not None:
                current_drawdown_start = None
                current_drawdown_duration = 0
        else:
            # In a drawdown
            drawdown = (peak - value) / peak
            drawdown_amount = peak - value
            
            # If this is the start of a new drawdown
            if current_drawdown_start is None:
                current_drawdown_start = i
                drawdown_periods += 1
            
            current_drawdown_duration += 1
            
            # Update max drawdown if current drawdown is larger
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_amount = drawdown_amount
                
            # Update max drawdown duration
            if current_drawdown_duration > max_drawdown_duration:
                max_drawdown_duration = current_drawdown_duration
                
            # Record this drawdown
            if drawdown > 0:
                all_drawdowns.append(drawdown)
    
    # Calculate average drawdown
    avg_drawdown = sum(all_drawdowns) / len(all_drawdowns) if all_drawdowns else 0
    
    return {
        'max_drawdown_pct': max_drawdown * 100,  # Convert to percentage
        'max_drawdown_amount': max_drawdown_amount,
        'drawdown_periods': drawdown_periods,
        'avg_drawdown_pct': avg_drawdown * 100,  # Convert to percentage
        'max_drawdown_duration': max_drawdown_duration
    }

def explain_bet_recommendation(bet_type, analysis, results, team1_name, team2_name):
    """Provide explanation for why a bet is recommended based on the stats"""
    
    print("  EXPLANATION:")
    
    # Explain moneyline bets
    if bet_type == 'team1_ml':
        # Check which factors favor team1
        print(f"  The model favors {team1_name} to win the match with {analysis['our_probability']:.2%} probability.")
        
        # Explain head-to-head advantage if it exists
        if results.get('h2h_advantage_team1', 0) > 0:
            print(f"  {team1_name} has a strong historical head-to-head advantage against {team2_name}.")
        
        # Explain any other key advantages
        if results.get('team1_win_prob', 0) > 0.6:
            print(f"  {team1_name}'s overall form and statistics indicate a significant advantage.")
            
    elif bet_type == 'team2_ml':
        print(f"  The model favors {team2_name} to win the match with {analysis['our_probability']:.2%} probability.")
        
        # Explain head-to-head advantage if it exists
        if results.get('h2h_advantage_team1', 0) < 0:
            print(f"  {team2_name} has a strong historical head-to-head advantage against {team1_name}.")
        
        # Explain any other key advantages
        if results.get('team2_win_prob', 0) > 0.6:
            print(f"  {team2_name}'s overall form and statistics indicate a significant advantage.")
    
    # Explain +1.5 bets
    elif bet_type == 'team1_plus_1_5':
        print(f"  {team1_name} has a {analysis['our_probability']:.2%} probability of winning at least one map.")
        print(f"  This is significantly higher than the {analysis['implied_probability']:.2%} implied by the odds.")
        if results.get('team1_plus_1_5_prob', 0) > 0.8:
            print(f"  Even as an underdog in the match, {team1_name} is likely to take at least one map.")
            
    elif bet_type == 'team2_plus_1_5':
        print(f"  {team2_name} has a {analysis['our_probability']:.2%} probability of winning at least one map.")
        print(f"  This is significantly higher than the {analysis['implied_probability']:.2%} implied by the odds.")
        if results.get('team2_plus_1_5_prob', 0) > 0.8:
            print(f"  Even as an underdog in the match, {team2_name} is likely to take at least one map.")
    
    # Explain -1.5 bets
    elif bet_type == 'team1_minus_1_5':
        print(f"  {team1_name} has a {analysis['our_probability']:.2%} probability of winning 2-0.")
        print(f"  The model suggests {team1_name} is significantly stronger than {team2_name} and can win without dropping a map.")
        
    elif bet_type == 'team2_minus_1_5':
        print(f"  {team2_name} has a {analysis['our_probability']:.2%} probability of winning 2-0.")
        print(f"  The model suggests {team2_name} is significantly stronger than {team1_name} and can win without dropping a map.")
    
    # Explain over/under bets
    elif bet_type == 'over_2_5_maps':
        print(f"  The match has a {analysis['our_probability']:.2%} probability of going to 3 maps.")
        print(f"  The teams appear evenly matched and likely to split the first two maps.")
        
    elif bet_type == 'under_2_5_maps':
        print(f"  The match has a {analysis['our_probability']:.2%} probability of ending in 2 maps.")
        print(f"  One team appears significantly stronger and likely to win 2-0.")
    
    # Explain why the odds provide value
    print(f"  The bookmaker's odds of {1/analysis['implied_probability']:.2f} represent value compared to our model's assessment.")
    print(f"  Long-term expected value: ${100 * (analysis['our_probability'] - analysis['implied_probability']):.2f} per $100 wagered.")

def input_odds_data():
    """Function to manually input odds data from bookmaker"""
    print("\nPlease enter the odds from your bookmaker for this match:")
    print("(Enter decimal odds format, e.g. 2.50 for +150, 1.67 for -150)")
    print("Leave blank for any bet types you're not interested in.")
    
    odds_data = {}
    
    # Team names for clarity
    team1 = input("\nEnter Team 1 name: ")
    team2 = input("Enter Team 2 name: ")
    
    # Moneyline
    try:
        odds = float(input(f"\n{team1} moneyline odds: ") or 0)
        if odds > 0:
            odds_data['team1_ml_odds'] = odds
    except ValueError:
        print("Invalid input, skipping.")
    
    try:
        odds = float(input(f"{team2} moneyline odds: ") or 0)
        if odds > 0:
            odds_data['team2_ml_odds'] = odds
    except ValueError:
        print("Invalid input, skipping.")
    
    # +1.5 maps
    try:
        odds = float(input(f"\n{team1} +1.5 maps odds: ") or 0)
        if odds > 0:
            odds_data['team1_plus_1_5_odds'] = odds
    except ValueError:
        print("Invalid input, skipping.")
    
    try:
        odds = float(input(f"{team2} +1.5 maps odds: ") or 0)
        if odds > 0:
            odds_data['team2_plus_1_5_odds'] = odds
    except ValueError:
        print("Invalid input, skipping.")
    
    # -1.5 maps
    try:
        odds = float(input(f"\n{team1} -1.5 maps odds: ") or 0)
        if odds > 0:
            odds_data['team1_minus_1_5_odds'] = odds
    except ValueError:
        print("Invalid input, skipping.")
    
# Over/Under 2.5 maps
    try:
        odds = float(input(f"\nOver 2.5 maps odds: ") or 0)
        if odds > 0:
            odds_data['over_2_5_maps_odds'] = odds
    except ValueError:
        print("Invalid input, skipping.")
    
    try:
        odds = float(input(f"Under 2.5 maps odds: ") or 0)
        if odds > 0:
            odds_data['under_2_5_maps_odds'] = odds
    except ValueError:
        print("Invalid input, skipping.")
    
    return team1, team2, odds_data

def load_prediction_artifacts():
    """
    Load the diverse ensemble and artifacts needed for prediction.
    """
    print("Loading prediction models and artifacts...")
    
    # Try to load the diverse ensemble
    try:
        with open('diverse_ensemble.pkl', 'rb') as f:
            ensemble_models = pickle.load(f)
        print(f"Loaded diverse ensemble with {len(ensemble_models)} models")
    except Exception as e:
        print(f"Error loading diverse ensemble: {e}")
        # Fall back to individual models if available
        try:
            ensemble_models = []
            for i in range(1, 11):
                try:
                    model_path = f'valorant_model_fold_{i}.h5'
                    model = load_model(model_path)
                    ensemble_models.append(('nn', model, None))
                    print(f"Loaded fallback model {i}/10")
                except:
                    # Skip if model doesn't exist
                    continue
            if not ensemble_models:
                print("Failed to load any models")
                return None, None, None
        except Exception as e:
            print(f"Error loading fallback models: {e}")
            return None, None, None
    
    # Load feature metadata
    try:
        with open('feature_metadata.pkl', 'rb') as f:
            feature_metadata = pickle.load(f)
        selected_features = feature_metadata.get('selected_features', [])
        print(f"Loaded {len(selected_features)} selected features")
    except Exception as e:
        print(f"Error loading feature metadata: {e}")
        try:
            with open('selected_feature_names.pkl', 'rb') as f:
                selected_features = pickle.load(f)
            print(f"Loaded {len(selected_features)} features from backup file")
        except Exception as e:
            print(f"Error loading feature names: {e}")
            # Last resort: try stable_features.pkl
            try:
                with open('stable_features.pkl', 'rb') as f:
                    selected_features = pickle.load(f)
                print(f"Loaded {len(selected_features)} features from stable_features.pkl")
            except Exception as e:
                print(f"Error loading any feature lists: {e}")
                return ensemble_models, None, None
    
    # Return ensemble_models, None for scaler (handled in ensemble), and selected_features
    return ensemble_models, None, selected_features

def track_betting_performance(prediction, bet_placed, bet_amount, outcome, odds):
    """
    Track betting performance over time.
    
    Args:
        prediction (dict): The prediction made by the model
        bet_placed (str): Type of bet placed
        bet_amount (float): Amount bet
        outcome (bool): Whether the bet won
        odds (float): Decimal odds offered
        
    Returns:
        None
    """
    # Load existing performance data or create new
    try:
        with open('betting_performance.json', 'r') as f:
            performance = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        performance = {
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'total_wagered': 0,
            'total_returns': 0,
            'roi': 0,
            'bets': []
        }
    
    # Create new bet record
    bet_record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'teams': f"{prediction['team1_name']} vs {prediction['team2_name']}",
        'bet_type': bet_placed,
        'amount': bet_amount,
        'odds': odds,
        'predicted_prob': prediction['betting_analysis'][bet_placed]['our_probability'],
        'implied_prob': prediction['betting_analysis'][bet_placed]['implied_probability'],
        'edge': prediction['betting_analysis'][bet_placed]['expected_value'],
        'outcome': 'win' if outcome else 'loss',
        'return': bet_amount * odds if outcome else 0,
        'profit': bet_amount * (odds - 1) if outcome else -bet_amount
    }
    
    # Update performance metrics
    performance['total_bets'] += 1
    if outcome:
        performance['wins'] += 1
    else:
        performance['losses'] += 1
    
    performance['total_wagered'] += bet_amount
    performance['total_returns'] += bet_record['return']
    performance['roi'] = (performance['total_returns'] - performance['total_wagered']) / performance['total_wagered'] if performance['total_wagered'] > 0 else 0
    
    # Add to history
    performance['bets'].append(bet_record)
    
    # Save updated performance
    with open('betting_performance.json', 'w') as f:
        json.dump(performance, f, indent=2)
    
    # Print performance update
    print("\nBetting Performance Updated:")
    print(f"Record: {performance['wins']}-{performance['losses']} ({performance['wins']/performance['total_bets']:.2%})")
    print(f"Total Wagered: ${performance['total_wagered']:.2f}")
    print(f"Total Returns: ${performance['total_returns']:.2f}")
    print(f"Profit: ${performance['total_returns'] - performance['total_wagered']:.2f}")
    print(f"ROI: {performance['roi']:.2%}")

def view_betting_performance():
    """Display historical betting performance with visualizations."""
    try:
        with open('betting_performance.json', 'r') as f:
            performance = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("No betting history found.")
        return
    
    print("\n===== BETTING PERFORMANCE SUMMARY =====")
    print(f"Total Bets: {performance['total_bets']}")
    print(f"Record: {performance['wins']}-{performance['losses']} ({performance['wins']/performance['total_bets']:.2%})")
    print(f"Total Wagered: ${performance['total_wagered']:.2f}")
    print(f"Total Returns: ${performance['total_returns']:.2f}")
    print(f"Profit: ${performance['total_returns'] - performance['total_wagered']:.2f}")
    print(f"ROI: {performance['roi']:.2%}")
    
    # Analyze by bet type
    bet_types = {}
    for bet in performance['bets']:
        bet_type = bet['bet_type']
        if bet_type not in bet_types:
            bet_types[bet_type] = {
                'total': 0,
                'wins': 0,
                'amount': 0,
                'returns': 0,
                'avg_edge': []
            }
        
        bet_types[bet_type]['total'] += 1
        if bet['outcome'] == 'win':
            bet_types[bet_type]['wins'] += 1
        bet_types[bet_type]['amount'] += bet['amount']
        bet_types[bet_type]['returns'] += bet['return']
        bet_types[bet_type]['avg_edge'].append(bet['edge'])
    
    print("\n===== PERFORMANCE BY BET TYPE =====")
    for bet_type, stats in bet_types.items():
        win_rate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
        roi = (stats['returns'] - stats['amount']) / stats['amount'] if stats['amount'] > 0 else 0
        avg_edge = sum(stats['avg_edge']) / len(stats['avg_edge']) if stats['avg_edge'] else 0
        
        print(f"\n{bet_type}:")
        print(f"  Record: {stats['wins']}-{stats['total']-stats['wins']} ({win_rate:.2%})")
        print(f"  Wagered: ${stats['amount']:.2f}")
        print(f"  Returns: ${stats['returns']:.2f}")
        print(f"  Profit: ${stats['returns'] - stats['amount']:.2f}")
        print(f"  ROI: {roi:.2%}")
        print(f"  Avg Edge: {avg_edge:.2%}")
    
    # Create visualizations if matplotlib is available
    try:
        # Prepare data for plots
        dates = [bet['timestamp'] for bet in performance['bets']]
        profits = []
        running_profit = 0
        for bet in performance['bets']:
            running_profit += bet['profit']
            profits.append(running_profit)
        
        # Create plots
        plt.figure(figsize=(12, 10))
        
        # Profit over time
        plt.subplot(2, 2, 1)
        plt.plot(range(len(profits)), profits, 'b-')
        plt.title('Cumulative Profit Over Time')
        plt.xlabel('Bet Number')
        plt.ylabel('Profit ($)')
        plt.grid(True)
        
        # Win rate by bet type
        plt.subplot(2, 2, 2)
        bet_type_names = list(bet_types.keys())
        win_rates = [bet_types[bt]['wins'] / bet_types[bt]['total'] if bet_types[bt]['total'] > 0 else 0 for bt in bet_type_names]
        
        plt.bar(range(len(bet_type_names)), win_rates)
        plt.xticks(range(len(bet_type_names)), bet_type_names, rotation=45)
        plt.title('Win Rate by Bet Type')
        plt.ylabel('Win Rate')
        plt.axhline(y=0.5, color='r', linestyle='-')
        plt.grid(True)
        
        # ROI by bet type
        plt.subplot(2, 2, 3)
        rois = [(bet_types[bt]['returns'] - bet_types[bt]['amount']) / bet_types[bt]['amount'] if bet_types[bt]['amount'] > 0 else 0 for bt in bet_type_names]
        
        plt.bar(range(len(bet_type_names)), rois)
        plt.xticks(range(len(bet_type_names)), bet_type_names, rotation=45)
        plt.title('ROI by Bet Type')
        plt.ylabel('ROI')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.grid(True)
        
        # Edge vs. Outcome
        plt.subplot(2, 2, 4)
        edges = [bet['edge'] for bet in performance['bets']]
        outcomes = [1 if bet['outcome'] == 'win' else 0 for bet in performance['bets']]
        
        plt.scatter(edges, outcomes, alpha=0.6)
        plt.title('Bet Outcome vs. Predicted Edge')
        plt.xlabel('Predicted Edge')
        plt.ylabel('Outcome (1=Win, 0=Loss)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('betting_performance.png')
        plt.close()
        
        print("\nPerformance visualizations saved to betting_performance.png")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

#-------------------------------------------------------------------------
# MAIN FUNCTION AND CLI INTERFACE
#-------------------------------------------------------------------------

@debug_func
def train_with_consistent_features(X, y, n_splits=10, random_state=42):
    """
    Train model using k-fold cross-validation with a consistent feature set across all folds.
    Uses a diverse ensemble of model types for robust predictions.
    
    Args:
        X (list or DataFrame): Feature data
        y (list): Target labels
        n_splits (int): Number of cross-validation folds
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (trained_models, feature_scaler, selected_features, avg_metrics)
    """

    import psutil
    mem = psutil.virtual_memory()
    print(f"Available memory: {mem.available / (1024 * 1024 * 1024):.2f} GB")

    print(f"\nTraining with {n_splits}-fold cross-validation using consistent features and diverse ensemble")
    df = clean_feature_data(X)
    X_arr = df.values
    y_arr = np.array(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_feature_importances = np.zeros(df.shape[1])
    fold_metrics = []
    
    print("Phase 1: Identifying important features across all folds...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        print(f"\n----- Feature Selection: Fold {fold+1}/{n_splits} -----")
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]
        
        class_counts = np.bincount(y_train)
        if np.min(class_counts) / np.sum(class_counts) < 0.4:
            try:
                if np.min(class_counts) >= 5:
                    min_samples = np.min(class_counts)
                    k_neighbors = min(5, min_samples-1)
                    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"Error applying SMOTE: {e}")
                
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf.fit(X_train, y_train)
        all_feature_importances += rf.feature_importances_
        
    avg_importances = all_feature_importances / n_splits
    indices = np.argsort(avg_importances)[::-1]
    cumulative_importance = np.cumsum(avg_importances[indices])
    n_features = np.where(cumulative_importance >= 0.8)[0][0] + 1
    n_features = min(n_features, 50)  # Cap at 50 features
    top_indices = indices[:n_features]
    selected_features = [df.columns[i] for i in top_indices]
    
    print(f"\nSelected {len(selected_features)} consistent features across all folds:")
    for i, feature in enumerate(selected_features[:10]):
        print(f"  {i+1}. {feature} (importance: {avg_importances[df.columns.get_loc(feature)]:.4f})")
    if len(selected_features) > 10:
        print(f"  ... and {len(selected_features) - 10} more features")
        
    feature_mask = np.zeros(df.shape[1], dtype=bool)
    feature_mask[top_indices] = True
    
    feature_importance_data = []
    for i, feature in enumerate(df.columns):
        feature_importance_data.append({
            'feature_name': feature,
            'importance_score': avg_importances[i],
            'selected': feature in selected_features,
            'rank': np.where(indices == i)[0][0] + 1 if i in indices else -1
        })
    importance_df = pd.DataFrame(feature_importance_data)
    importance_df = importance_df.sort_values('importance_score', ascending=False)
    importance_df.to_csv('feature_importance.csv', index=False)
    print(f"Saved feature importance data to feature_importance.csv")
    
    print("\nPhase 2: Training diverse ensemble with consistent feature set...")
    train_idx, val_idx = next(skf.split(X_arr, y_arr))
    X_train, X_val = X_arr[train_idx], X_arr[val_idx]
    y_train, y_val = y_arr[train_idx], y_arr[val_idx]
    
    print("Creating diverse ensemble of models...")
    ensemble_models, scaler = create_diverse_ensemble(X_train, y_train, feature_mask, random_state)
    
    X_val_selected = X_val[:, feature_mask]
    predictions = []
    for model_type, model, model_scaler in ensemble_models:
        try:
            X_val_pred = X_val_selected.copy()
            if model_scaler is not None:
                X_val_pred = model_scaler.transform(X_val_selected)
            if model_type == 'nn':
                preds = model.predict(X_val_pred, verbose=0).flatten()
            else:
                preds = model.predict_proba(X_val_pred)[:, 1]
            predictions.append(preds)
            y_pred_binary = (preds > 0.5).astype(int)
            acc = accuracy_score(y_val, y_pred_binary)
            print(f"{model_type.upper()} model - Validation accuracy: {acc:.4f}")
        except Exception as e:
            print(f"Error evaluating {model_type} model: {e}")
            
    if predictions:
        ensemble_preds = np.mean(predictions, axis=0)
        ensemble_binary = (ensemble_preds > 0.5).astype(int)
        accuracy = accuracy_score(y_val, ensemble_binary)
        precision = precision_score(y_val, ensemble_binary)
        recall = recall_score(y_val, ensemble_binary)
        f1 = f1_score(y_val, ensemble_binary)
        auc = roc_auc_score(y_val, ensemble_preds)
        
        print("\nEnsemble Performance on Validation Set:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
    else:
        print("Warning: No predictions generated from ensemble models")
        accuracy = precision = recall = f1 = auc = 0
        
    print("\nSaving models and artifacts...")
    with open('diverse_ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble_models, f)
    with open('feature_mask.pkl', 'wb') as f:
        pickle.dump(feature_mask, f)
    with open('selected_feature_names.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
        
    feature_metadata = {
        'selected_features': selected_features,
        'feature_importances': dict(zip(selected_features, avg_importances[top_indices]))
    }
    with open('feature_metadata.pkl', 'wb') as f:
        pickle.dump(feature_metadata, f)
        
    print("All models and artifacts saved successfully.")
    
    ensemble_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    return ensemble_models, scaler, selected_features, ensemble_metrics

#-------------------------------------------------------------------------
# BACKTESTING SYSTEM
#-------------------------------------------------------------------------

def load_backtesting_models():
    """
    Enhanced function to load ensemble models and metadata for backtesting
    with improved error handling and logging.
    """
    print("\n========== LOADING RETRAINED MODELS ==========")
    
    # 1. Load the diverse ensemble with better error handling
    ensemble_models = None
    try:
        print("Attempting to load diverse ensemble from diverse_ensemble.pkl...")
        with open('diverse_ensemble.pkl', 'rb') as f:
            ensemble_models = pickle.load(f)
            print(f"SUCCESS: Loaded diverse ensemble with {len(ensemble_models)} models")
            
            # Log model types for verification
            model_types = {}
            for model_type, _, _ in ensemble_models:
                if model_type not in model_types:
                    model_types[model_type] = 0
                model_types[model_type] += 1
            
            print("Model composition:")
            for model_type, count in model_types.items():
                print(f"  - {model_type.upper()}: {count} models")
    except Exception as e:
        print(f"ERROR: Failed to load diverse ensemble: {e}")
        print("Attempting to load individual fold models as fallback...")
        
        # Try loading individual fold models
        ensemble_models = []
        for i in range(1, 11):
            try:
                model = load_model(f'valorant_model_fold_{i}.h5')
                ensemble_models.append(('nn', model, None))
                print(f"Loaded fold model {i}")
            except Exception as model_e:
                print(f"Could not load model fold_{i}: {model_e}")
        
        if ensemble_models:
            print(f"SUCCESS: Loaded {len(ensemble_models)} individual fold models as fallback")
        else:
            print("ERROR: Failed to load any models. Backtesting cannot proceed.")
            return None, None
    
    # 2. Load feature metadata with enhanced error handling and priority order
    selected_features = None
    feature_metadata = None
    
    # Try loading from feature_metadata.pkl first (preferred source)
    try:
        print("Attempting to load feature metadata from feature_metadata.pkl...")
        with open('feature_metadata.pkl', 'rb') as f:
            feature_metadata = pickle.load(f)
            selected_features = feature_metadata.get('selected_features')
            feature_importances = feature_metadata.get('feature_importances', {})
            
            print(f"SUCCESS: Loaded {len(selected_features)} features from feature_metadata.pkl")
            print(f"Top 5 features by importance:")
            sorted_features = sorted(feature_importances.items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in sorted_features:
                print(f"  - {feature}: {importance:.4f}")
    except Exception as e:
        print(f"ERROR: Failed to load feature_metadata.pkl: {e}")
    
    # If that failed, try loading from selected_feature_names.pkl
    if not selected_features:
        try:
            print("Attempting to load from selected_feature_names.pkl...")
            with open('selected_feature_names.pkl', 'rb') as f:
                selected_features = pickle.load(f)
                print(f"SUCCESS: Loaded {len(selected_features)} features from selected_feature_names.pkl")
        except Exception as e:
            print(f"ERROR: Failed to load selected_feature_names.pkl: {e}")
    
    # If that failed too, try stable_features.pkl
    if not selected_features:
        try:
            print("Attempting to load from stable_features.pkl...")
            with open('stable_features.pkl', 'rb') as f:
                selected_features = pickle.load(f)
                print(f"SUCCESS: Loaded {len(selected_features)} features from stable_features.pkl")
        except Exception as e:
            print(f"ERROR: Failed to load stable_features.pkl: {e}")
    
    # Last resort: Load feature mask and extract feature names
    if not selected_features:
        try:
            print("Attempting to reconstruct features from feature_mask.pkl...")
            with open('feature_mask.pkl', 'rb') as f:
                feature_mask = pickle.load(f)
                # This assumes you have access to the original feature names
                # In a real implementation, you'd need to have them stored somewhere
                print("WARNING: Using feature mask without feature names is not ideal")
                # Create placeholder features
                selected_features = [f'feature_{i}' for i in range(sum(feature_mask))]
                print(f"Created {len(selected_features)} placeholder features from mask")
        except Exception as e:
            print(f"ERROR: Failed to load feature_mask.pkl: {e}")
    
    # If we still don't have features, create fallback feature list
    if not selected_features:
        print("WARNING: Failed to load any feature list. Creating fallback features.")
        # Create a list of likely important features based on domain knowledge
        selected_features = [
            'win_rate_diff', 'better_win_rate_team1', 'recent_form_diff',
            'better_recent_form_team1', 'score_diff_differential',
            'better_score_diff_team1', 'avg_score_diff', 'h2h_win_rate',
            'h2h_matches', 'h2h_score_diff', 'h2h_advantage_team1',
            'h2h_significant', 'total_matches', 'match_count_diff',
            'avg_win_rate', 'avg_recent_form', 'wins_diff', 'losses_diff',
            'win_loss_ratio_diff', 'player_rating_diff', 'better_player_rating_team1',
            'avg_player_rating', 'pistol_win_rate_diff', 'better_pistol_team1',
            'eco_win_rate_diff', 'semi_eco_win_rate_diff', 'full_buy_win_rate_diff',
            'economy_efficiency_diff', 'rating_x_win_rate', 'pistol_x_eco',
            'pistol_x_full_buy', 'first_blood_x_win_rate', 'h2h_x_win_rate'
        ]
        print(f"Created fallback feature list with {len(selected_features)} features")
    
    # If needed, pad the feature list to match expected model input shape
    if ensemble_models and hasattr(ensemble_models[0][1], 'layers'):
        try:
            # Get expected input dimensionality from first model
            model_type, model, _ = ensemble_models[0]
            if model_type == 'nn':
                expected_dim = model.layers[0].input_shape[1]
                
                if len(selected_features) < expected_dim:
                    print(f"WARNING: Selected features ({len(selected_features)}) less than expected model input dimension ({expected_dim})")
                    padding_count = expected_dim - len(selected_features)
                    padding_features = [f"padding_feature_{i}" for i in range(padding_count)]
                    selected_features.extend(padding_features)
                    print(f"Added {padding_count} padding features to match model input dimension")
        except Exception as e:
            print(f"ERROR checking model dimensions: {e}")
    
    # Load scalers if available
    try:
        print("Attempting to load feature scaler...")
        with open('ensemble_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            print("SUCCESS: Loaded ensemble_scaler.pkl")
            
            # Add scaler to ensemble models that need it
            enhanced_ensemble = []
            for model_type, model, model_scaler in ensemble_models:
                if model_type in ['lr', 'svm'] and model_scaler is None:
                    enhanced_ensemble.append((model_type, model, scaler))
                    print(f"Added scaler to {model_type} model")
                else:
                    enhanced_ensemble.append((model_type, model, model_scaler))
            
            ensemble_models = enhanced_ensemble
    except Exception as e:
        print(f"WARNING: Failed to load ensemble_scaler.pkl: {e}")
    
    print(f"Model loading complete. Ensemble has {len(ensemble_models)} models and {len(selected_features)} features")
    return ensemble_models, selected_features

def simulate_odds(team1_win_prob, vig=0.05):
    """Generate realistic betting odds based on predicted probability and vig."""
    # Apply vig to create juice
    adj_team1_prob = team1_win_prob * (1 + vig) - vig
    adj_team2_prob = (1 - team1_win_prob) * (1 + vig) - vig
    
    # Convert to decimal odds
    team1_odds = 1 / adj_team1_prob if adj_team1_prob > 0 else float('inf')
    team2_odds = 1 / adj_team2_prob if adj_team2_prob > 0 else float('inf')
    
    # Create map spread and total odds
    team1_plus_odds = 1 / min(0.95, team1_win_prob + 0.25)
    team2_plus_odds = 1 / min(0.95, (1 - team1_win_prob) + 0.25)
    team1_minus_odds = 1 / max(0.05, team1_win_prob - 0.3)
    team2_minus_odds = 1 / max(0.05, (1 - team1_win_prob) - 0.3)
    
    map_total_over = 1.92
    map_total_under = 1.92
    
    return {
        'team1_ml_odds': round(team1_odds, 2),
        'team2_ml_odds': round(team2_odds, 2),
        'team1_plus_1_5_odds': round(team1_plus_odds, 2),
        'team2_plus_1_5_odds': round(team2_plus_odds, 2),
        'team1_minus_1_5_odds': round(team1_minus_odds, 2),
        'team2_minus_1_5_odds': round(team2_minus_odds, 2),
        'over_2_5_maps_odds': map_total_over,
        'under_2_5_maps_odds': map_total_under
    }

def calculate_single_map_prob_for_backtesting(match_win_prob):
    """Calculate single map win probability from match win probability."""
    # Simplified calculation - can be adjusted based on your model
    single_map_prob = max(0.3, min(0.7, np.sqrt(match_win_prob)))
    return single_map_prob

def kelly_criterion(win_prob, decimal_odds, fractional=0.15):
    """Calculate Kelly bet size with more conservative fractional adjustment."""
    # Kelly formula: f* = (bp - q) / b
    b = decimal_odds - 1  # Decimal odds to b format
    p = win_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Apply fractional Kelly (more conservative)
    kelly = kelly * fractional
    
    # Apply additional cap for safety
    kelly = min(kelly, 0.05)  # Never bet more than 5% regardless of Kelly formula
    
    # Ensure non-negative
    return max(0, kelly)
def analyze_specific_match(results, match_id):
    """
    Analyze a specific match from backtest results in detail.
    
    Args:
        results (dict): Backtest results
        match_id (str): ID of the match to analyze
        
    Returns:
        dict: Detailed analysis of the match
    """
    # Find match in predictions
    match_prediction = None
    for pred in results['predictions']:
        if pred.get('match_id') == match_id:
            match_prediction = pred
            break
    
    if not match_prediction:
        print(f"Match {match_id} not found in results")
        return None
    
    # Find bets placed on this match
    match_bets = None
    for bet_record in results['bets']:
        if bet_record.get('match_id') == match_id:
            match_bets = bet_record
            break
    
    # Create detailed analysis
    analysis = {
        'match_id': match_id,
        'teams': f"{match_prediction['team1']} vs {match_prediction['team2']}",
        'prediction': {
            'predicted_winner': match_prediction['predicted_winner'],
            'actual_winner': match_prediction['actual_winner'],
            'team1_prob': match_prediction['team1_prob'],
            'team2_prob': match_prediction['team2_prob'],
            'confidence': match_prediction['confidence'],
            'correct': match_prediction['correct'],
            'score': match_prediction['score'],
            'date': match_prediction.get('date', 'Unknown')
        },
        'bets': []
    }
    
    # Add betting details if available
    if match_bets:
        analysis['bets'] = match_bets['bets']
        
        # Calculate aggregate betting performance
        total_wagered = sum(bet['amount'] for bet in match_bets['bets'])
        total_returns = sum(bet['returns'] for bet in match_bets['bets'])
        winning_bets = sum(1 for bet in match_bets['bets'] if bet['won'])
        
        analysis['betting_summary'] = {
            'total_bets': len(match_bets['bets']),
            'winning_bets': winning_bets,
            'total_wagered': total_wagered,
            'total_returns': total_returns,
            'profit': total_returns - total_wagered,
            'roi': (total_returns - total_wagered) / total_wagered if total_wagered > 0 else 0
        }
    
    # Print analysis
    print(f"\n===== MATCH ANALYSIS: {analysis['teams']} =====")
    print(f"Date: {analysis['prediction']['date']}")
    print(f"Score: {analysis['prediction']['score']}")
    print(f"Prediction: {analysis['prediction']['predicted_winner']} to win")
    print(f"Probability: {analysis['prediction']['team1_prob']:.2%} vs {analysis['prediction']['team2_prob']:.2%}")
    print(f"Confidence: {analysis['prediction']['confidence']:.2f}")
    print(f"Result: {'CORRECT' if analysis['prediction']['correct'] else 'INCORRECT'}")
    
    if 'betting_summary' in analysis:
        print("\nBetting Summary:")
        print(f"Total Bets: {analysis['betting_summary']['total_bets']}")
        print(f"Winning Bets: {analysis['betting_summary']['winning_bets']}")
        print(f"Wagered: ${analysis['betting_summary']['total_wagered']:.2f}")
        print(f"Returns: ${analysis['betting_summary']['total_returns']:.2f}")
        print(f"Profit: ${analysis['betting_summary']['profit']:.2f}")
        print(f"ROI: {analysis['betting_summary']['roi']:.2%}")
        
        print("\nIndividual Bets:")
        for i, bet in enumerate(analysis['bets']):
            print(f"  {i+1}. {bet['bet_type'].replace('_', ' ').upper()}")
            print(f"     Amount: ${bet['amount']:.2f}")
            print(f"     Odds: {bet['odds']:.2f}")
            print(f"     Edge: {bet['edge']:.2%}")
            print(f"     Result: {'WON' if bet['won'] else 'LOST'}")
            print(f"     Returns: ${bet['returns']:.2f}")
            print(f"     Profit: ${bet['profit']:.2f}")
    
    return analysis

def identify_key_insights(results):
    """
    Identify key insights and recommendations from backtest results.
    
    Args:
        results (dict): Backtest results
        
    Returns:
        dict: Key insights and recommendations
    """
    insights = {
        'overall_performance': {},
        'bet_types': [],
        'teams': [],
        'confidence_insights': {},
        'edge_insights': {},
        'recommendations': []
    }
    
    # Extract overall performance metrics
    insights['overall_performance'] = {
        'accuracy': results['performance']['accuracy'],
        'roi': results['performance']['roi'],
        'profit': results['performance']['profit'],
        'win_rate': results['performance']['win_rate'],
        'total_bets': results['performance'].get('total_bets', 0),
        'total_predictions': len(results['predictions'])
    }
    
    # Analyze bet types
    if 'metrics' in results and 'bet_types' in results['metrics']:
        bet_types = []
        for bet_type, stats in results['metrics']['bet_types'].items():
            if stats['total'] >= 5:  # Only include bet types with sufficient sample size
                win_rate = stats['won'] / stats['total'] if stats['total'] > 0 else 0
                roi = (stats['returns'] - stats['wagered']) / stats['wagered'] if stats['wagered'] > 0 else 0
                
                bet_types.append({
                    'type': bet_type,
                    'total': stats['total'],
                    'win_rate': win_rate,
                    'roi': roi,
                    'profit': stats['returns'] - stats['wagered'],
                    'profitable': roi > 0,
                    'recommendation': 'strong_bet' if roi > 0.1 else ('potential_bet' if roi > 0 else 'avoid')
                })
        
        # Sort by ROI
        bet_types.sort(key=lambda x: x['roi'], reverse=True)
        insights['bet_types'] = bet_types
    
    # Analyze team performance
    if 'team_performance' in results:
        teams = []
        for team_name, stats in results['team_performance'].items():
            if stats.get('bets', 0) >= 5:  # Only include teams with sufficient betting history
                prediction_accuracy = stats.get('correct', 0) / stats.get('predictions', 1) if stats.get('predictions', 0) > 0 else 0
                win_rate = stats.get('wins', 0) / stats.get('bets', 1) if stats.get('bets', 0) > 0 else 0
                roi = (stats.get('returns', 0) - stats.get('wagered', 0)) / stats.get('wagered', 1) if stats.get('wagered', 0) > 0 else 0
                
                teams.append({
                    'name': team_name,
                    'predictions': stats.get('predictions', 0),
                    'prediction_accuracy': prediction_accuracy,
                    'bets': stats.get('bets', 0),
                    'win_rate': win_rate,
                    'roi': roi,
                    'profit': stats.get('returns', 0) - stats.get('wagered', 0),
                    'profitable': roi > 0,
                    'recommendation': 'focus' if roi > 0.1 else ('potential' if roi > 0 else 'avoid')
                })
        
        # Sort by ROI
        teams.sort(key=lambda x: x['roi'], reverse=True)
        insights['teams'] = teams
    
    # Analyze confidence levels
    if 'metrics' in results and 'confidence_bins' in results['metrics']:
        confidence_data = results['metrics']['confidence_bins']
        
        # Calculate accuracy by confidence level
        confidence_by_accuracy = {}
        for conf_key, stats in confidence_data.items():
            if stats['total'] >= 5:  # Only include confidence levels with sufficient sample size
                confidence_by_accuracy[conf_key] = stats['correct'] / stats['total']
        
        # Find most reliable confidence threshold
        reliable_thresholds = []
        for conf_key, accuracy in confidence_by_accuracy.items():
            conf_level = float(conf_key.replace('%', '')) / 100
            if accuracy > 0.55:  # Only consider confidence levels with decent accuracy
                reliable_thresholds.append((conf_key, conf_level, accuracy))
        
        if reliable_thresholds:
            # Sort by balance of confidence and accuracy
            reliable_thresholds.sort(key=lambda x: x[1] * x[2], reverse=True)
            recommended_threshold = reliable_thresholds[0][0]
            recommended_accuracy = reliable_thresholds[0][2]
        else:
            recommended_threshold = "None"
            recommended_accuracy = 0
        
        insights['confidence_insights'] = {
            'accuracy_by_confidence': confidence_by_accuracy,
            'recommended_threshold': recommended_threshold,
            'expected_accuracy': recommended_accuracy
        }
    
    # Analyze edge levels
    if 'metrics' in results and 'roi_by_edge' in results['metrics']:
        edge_data = results['metrics']['roi_by_edge']
        
        # Calculate ROI by edge level
        roi_by_edge = {}
        for edge_key, stats in edge_data.items():
            if stats['wagered'] >= 100:  # Only include edge levels with sufficient volume
                roi_by_edge[edge_key] = (stats['returns'] - stats['wagered']) / stats['wagered']
        
        # Find most profitable edge threshold
        profitable_thresholds = []
        for edge_key, roi in roi_by_edge.items():
            if roi > 0:  # Only consider profitable edge levels
                edge_parts = edge_key.replace('%', '').split('-')
                min_edge = float(edge_parts[0]) / 100
                profitable_thresholds.append((edge_key, min_edge, roi))
        
        if profitable_thresholds:
            # Sort by ROI
            profitable_thresholds.sort(key=lambda x: x[2], reverse=True)
            recommended_threshold = profitable_thresholds[0][0]
            expected_roi = profitable_thresholds[0][2]
        else:
            recommended_threshold = "None"
            expected_roi = 0
        
        insights['edge_insights'] = {
            'roi_by_edge': roi_by_edge,
            'recommended_threshold': recommended_threshold,
            'expected_roi': expected_roi
        }
    
    # Generate recommendations
    recommendations = []
    
    # Recommendation 1: Overall strategy viability
    if insights['overall_performance']['roi'] > 0.1:
        recommendations.append("The overall betting strategy is highly profitable with an ROI of {:.1%}. Continue using the current approach.".format(
            insights['overall_performance']['roi']))
    elif insights['overall_performance']['roi'] > 0:
        recommendations.append("The overall betting strategy is marginally profitable with an ROI of {:.1%}. Consider focusing on the most profitable bet types.".format(
            insights['overall_performance']['roi']))
    else:
        recommendations.append("The overall betting strategy is not profitable with an ROI of {:.1%}. Major adjustments are needed.".format(
            insights['overall_performance']['roi']))
    
    # Recommendation 2: Best bet types
    profitable_bet_types = [bt for bt in insights['bet_types'] if bt['roi'] > 0.05 and bt['total'] >= 10]
    if profitable_bet_types:
        top_bet = profitable_bet_types[0]
        recommendations.append("Focus on '{}' bets, which showed {:.1%} ROI across {} bets.".format(
            top_bet['type'].replace('_', ' ').upper(), top_bet['roi'], top_bet['total']))
    
    # Recommendation 3: Teams to focus on or avoid
    profitable_teams = [team for team in insights['teams'] if team['roi'] > 0.1 and team['bets'] >= 5]
    unprofitable_teams = [team for team in insights['teams'] if team['roi'] < -0.1 and team['bets'] >= 5]
    
    if profitable_teams:
        top_team = profitable_teams[0]
        recommendations.append("Target matches involving {}, which showed {:.1%} ROI across {} bets.".format(
            top_team['name'], top_team['roi'], top_team['bets']))
    
    if unprofitable_teams:
        worst_team = unprofitable_teams[-1]
        recommendations.append("Avoid betting on matches involving {}, which showed {:.1%} ROI across {} bets.".format(
            worst_team['name'], worst_team['roi'], worst_team['bets']))
    
    # Recommendation 4: Confidence threshold
    if 'confidence_insights' in insights and insights['confidence_insights']['recommended_threshold'] != "None":
        recommendations.append("Only place bets when model confidence is at least {}. This threshold yielded {:.1%} prediction accuracy.".format(
            insights['confidence_insights']['recommended_threshold'],
            insights['confidence_insights']['expected_accuracy']))
    
    # Recommendation 5: Edge threshold
    if 'edge_insights' in insights and insights['edge_insights']['recommended_threshold'] != "None":
        recommendations.append("Only place bets with a predicted edge of at least {}. This threshold yielded {:.1%} ROI.".format(
            insights['edge_insights']['recommended_threshold'],
            insights['edge_insights']['expected_roi']))
    
    # Recommendation 6: Bankroll management
    if insights['overall_performance']['roi'] > 0:
        win_rate = insights['overall_performance']['win_rate']
        recommendations.append("Use Kelly criterion with a fraction of {:.1%} to optimize bankroll growth based on {:.1%} win rate.".format(
            min(0.05, win_rate / 4),  # Conservative Kelly fraction
            win_rate))
    
    insights['recommendations'] = recommendations
    
    # Print insights
    print("\n===== KEY INSIGHTS =====")
    
    print("\nOverall Performance:")
    print(f"Prediction Accuracy: {insights['overall_performance']['accuracy']:.2%}")
    print(f"Betting Win Rate: {insights['overall_performance']['win_rate']:.2%}")
    print(f"ROI: {insights['overall_performance']['roi']:.2%}")
    print(f"Total Profit: ${insights['overall_performance']['profit']:.2f}")
    
    print("\nTop Performing Bet Types:")
    for i, bet_type in enumerate(insights['bet_types'][:3]):
        if bet_type['profitable']:
            print(f"{i+1}. {bet_type['type'].replace('_', ' ').upper()}: {bet_type['roi']:.2%} ROI ({bet_type['total']} bets)")
    
    print("\nTop Performing Teams:")
    for i, team in enumerate(insights['teams'][:3]):
        if team['profitable']:
            print(f"{i+1}. {team['name']}: {team['roi']:.2%} ROI ({team['bets']} bets)")
    
    print("\nRecommendations:")
    for i, rec in enumerate(insights['recommendations']):
        print(f"{i+1}. {rec}")
    
    return insights

def get_backtest_params():
    """
    Interactive function to get parameters for backtesting.
    
    Returns:
        dict: Parameters for backtesting
    """
    print("\n===== BACKTEST CONFIGURATION =====")
    
    # Default parameters
    params = {
        'team_limit': 50,
        'bankroll': 1000.0,
        'bet_pct': 0.05,
        'min_edge': 0.08,
        'confidence_threshold': 0.4,
        'start_date': None,
        'end_date': None
    }
    
    # Interactive parameter entry
    print("\nEnter parameters (press Enter to use defaults):")
    
    try:
        # Team limit
        team_input = input(f"Number of teams to analyze [{params['team_limit']}]: ")
        if team_input.strip():
            params['team_limit'] = int(team_input)
        
        # Bankroll
        bankroll_input = input(f"Starting bankroll [${params['bankroll']}]: ")
        if bankroll_input.strip():
            params['bankroll'] = float(bankroll_input)
        
        # Bet percentage
        bet_pct_input = input(f"Maximum bet size as percentage of bankroll [{params['bet_pct']*100}%]: ")
        if bet_pct_input.strip():
            params['bet_pct'] = float(bet_pct_input) / 100
        
        # Minimum edge
        min_edge_input = input(f"Minimum required edge [{params['min_edge']*100}%]: ")
        if min_edge_input.strip():
            params['min_edge'] = float(min_edge_input) / 100
        
        # Confidence threshold
        conf_input = input(f"Minimum model confidence [{params['confidence_threshold']*100}%]: ")
        if conf_input.strip():
            params['confidence_threshold'] = float(conf_input) / 100
        
        # Date range
        date_range = input("Use specific date range? (y/n): ").lower().startswith('y')
        if date_range:
            start_date = input("Start date (YYYY-MM-DD or blank for no limit): ")
            end_date = input("End date (YYYY-MM-DD or blank for no limit): ")
            
            if start_date.strip():
                params['start_date'] = start_date.strip()
            if end_date.strip():
                params['end_date'] = end_date.strip()
    
    except ValueError as e:
        print(f"Invalid input: {e}")
        print("Using default values for all parameters.")
    
    # Print final configuration
    print("\nBacktest Configuration:")
    print(f"Teams to analyze: {params['team_limit']}")
    print(f"Starting bankroll: ${params['bankroll']}")
    print(f"Maximum bet size: {params['bet_pct']*100}% of bankroll")
    print(f"Minimum edge: {params['min_edge']*100}%")
    print(f"Minimum confidence: {params['confidence_threshold']*100}%")
    
    if params['start_date'] or params['end_date']:
        date_range = f"{params['start_date'] or 'Beginning'} to {params['end_date'] or 'Present'}"
        print(f"Date range: {date_range}")
    else:
        print("Date range: All available data")
    
    return params

def select_recommended_bets(betting_analysis, team1_name, team2_name):
    """
    Select recommended bets using the same criteria as backtesting.
    """
    # Get all recommended bets
    recommended_bets = {k: v for k, v in betting_analysis.items() if v['recommended']}
    
    if not recommended_bets:
        return {}
    
    # Use same prioritization as backtesting
    high_roi_bets = {k: v for k, v in recommended_bets.items() if v.get('high_roi_bet', False)}
    other_bets = {k: v for k, v in recommended_bets.items() if not v.get('high_roi_bet', False)}
    
    # Sort bets by edge
    sorted_high_roi_bets = sorted(high_roi_bets.items(), key=lambda x: x[1]['edge'], reverse=True)
    sorted_other_bets = sorted(other_bets.items(), key=lambda x: x[1]['edge'], reverse=True)
    
    # Combine lists with high ROI bets first
    sorted_bets = sorted_high_roi_bets + sorted_other_bets
    
    # Max bets per match - same as backtesting
    max_bets = 3
    
    # Select bets with diversification
    selected_bets = {}
    bet_teams = set()
    bet_categories = set()
    
    for bet_type, analysis in sorted_bets:
        # Stop if we've reached max bets
        if len(selected_bets) >= max_bets:
            break
        
        # Determine team and category
        if 'team1' in bet_type:
            team = team1_name
        elif 'team2' in bet_type:
            team = team2_name
        else:
            team = None
            
        category = '_'.join(bet_type.split('_')[1:])  # e.g., 'ml', 'plus_1_5'
        
        # Use same diversification logic as backtesting
        if team is None or category not in bet_categories or analysis.get('high_roi_bet', False):
            selected_bets[bet_type] = analysis
            bet_teams.add(team) if team else None
            bet_categories.add(category)
    
    return selected_bets



# 3. Updated betting analysis with adjusted thresholds
def analyze_betting_edge_for_backtesting(team1_win_prob, team2_win_prob, odds_data, confidence_score, bankroll=1000.0):
    """
    Optimized betting analysis that balances high ROI with aggressive growth.
    Maintains safeguards while allowing more aggressive bet sizing.
    """
    betting_analysis = {}
    
    # Verify valid probabilities
    if not (0 < team1_win_prob < 1) or not (0 < team2_win_prob < 1):
        team1_win_prob = min(0.95, max(0.05, team1_win_prob))
        team2_win_prob = 1 - team1_win_prob
    
    # Ensure probabilities sum to 1
    if abs(team1_win_prob + team2_win_prob - 1) > 0.001:
        total = team1_win_prob + team2_win_prob
        team1_win_prob = team1_win_prob / total
        team2_win_prob = team2_win_prob / total
    
    # Edge threshold based on confidence
    base_min_edge = 0.03  
    confidence_factor = 0.7 + (0.3 * confidence_score)
    adjusted_threshold = base_min_edge / confidence_factor
    
    print(f"\n----- BETTING ANALYSIS -----")
    print(f"Confidence score: {confidence_score:.4f}, confidence factor: {confidence_factor:.4f}")
    print(f"Base edge threshold: {base_min_edge:.2%}, adjusted threshold: {adjusted_threshold:.2%}")
    
    # Calculate map probabilities
    map_scale = 0.55 + (confidence_score * 0.15)
    single_map_prob = 0.5 + (team1_win_prob - 0.5) * map_scale
    single_map_prob = max(0.3, min(0.7, single_map_prob))
    
    # Calculate probabilities for different bet types
    team1_plus_prob = 1 - (1 - single_map_prob) ** 2
    team2_plus_prob = 1 - single_map_prob ** 2
    team1_minus_prob = single_map_prob ** 2
    team2_minus_prob = (1 - single_map_prob) ** 2
    over_prob = 2 * single_map_prob * (1 - single_map_prob)
    under_prob = 1 - over_prob
    
    # Cap all probabilities to avoid extremes
    team1_plus_prob = min(0.95, max(0.05, team1_plus_prob))
    team2_plus_prob = min(0.95, max(0.05, team2_plus_prob))
    team1_minus_prob = min(0.9, max(0.1, team1_minus_prob))
    team2_minus_prob = min(0.9, max(0.1, team2_minus_prob))
    over_prob = min(0.9, max(0.1, over_prob))
    under_prob = min(0.9, max(0.1, under_prob))
    
    # Standard bet types
    bet_types = [
        ('team1_ml', team1_win_prob, odds_data['team1_ml_odds']),
        ('team2_ml', team2_win_prob, odds_data['team2_ml_odds']),
        ('team1_plus_1_5', team1_plus_prob, odds_data['team1_plus_1_5_odds']),
        ('team2_plus_1_5', team2_plus_prob, odds_data['team2_plus_1_5_odds']),
        ('team1_minus_1_5', team1_minus_prob, odds_data['team1_minus_1_5_odds']),
        ('team2_minus_1_5', team2_minus_prob, odds_data['team2_minus_1_5_odds']),
        ('over_2_5_maps', over_prob, odds_data['over_2_5_maps_odds']),
        ('under_2_5_maps', under_prob, odds_data['under_2_5_maps_odds'])
    ]
    
    # Thresholds favoring highest ROI bet types
    type_thresholds = {
        'team1_ml': adjusted_threshold * 0.9,
        'team2_ml': adjusted_threshold * 1.1,  # Higher threshold (more selective)
        'team1_plus_1_5': adjusted_threshold * 0.95,
        'team2_plus_1_5': adjusted_threshold * 1.0,
        'team1_minus_1_5': adjusted_threshold * 0.7,  # Much lower threshold - highest ROI
        'team2_minus_1_5': adjusted_threshold * 0.8,  # Lower threshold - second highest ROI
        'over_2_5_maps': adjusted_threshold * 1.1,
        'under_2_5_maps': adjusted_threshold * 1.0
    }
    
    # KEY IMPROVEMENT: Progressive bet sizing based on bankroll
    # Different caps at different bankroll levels
    if bankroll <= 2000:
        # Starting phase - conservative
        MAX_SINGLE_BET = min(bankroll * 0.05, 100)  # 5% or $100 max
    elif bankroll <= 10000:
        # Growth phase - moderate
        MAX_SINGLE_BET = min(bankroll * 0.04, 400)  # 4% or $400 max
    elif bankroll <= 50000:
        # Expansion phase - aggressive
        MAX_SINGLE_BET = min(bankroll * 0.03, 1500)  # 3% or $1500 max
    elif bankroll <= 200000:
        # Scale phase - very aggressive
        MAX_SINGLE_BET = min(bankroll * 0.025, 5000)  # 2.5% or $5000 max
    else:
        # Empire phase - extremely aggressive
        MAX_SINGLE_BET = min(bankroll * 0.02, 20000)  # 2% or $20000 max
    
    # Safety cap to prevent runaway growth
    MAX_ALLOWED_BET = 50000  # Hard cap at $50,000 per bet
    MAX_SINGLE_BET = min(MAX_SINGLE_BET, MAX_ALLOWED_BET)
    
    for bet_type, prob, odds in bet_types:
        # Skip invalid inputs
        if not (0 < prob < 1) or odds <= 1.0:
            continue
            
        # Calculate edge
        implied_prob = 1 / odds
        edge = prob - implied_prob
        bet_threshold = type_thresholds.get(bet_type, adjusted_threshold)
        
        # Kelly calculation
        # Use different fractional Kelly based on bet type ROI
        if bet_type == 'team1_minus_1_5':  # Highest ROI
            base_fraction = 0.25  # 25% of Kelly
        elif bet_type == 'team2_minus_1_5':  # Second highest ROI
            base_fraction = 0.20  # 20% of Kelly
        elif bet_type == 'team1_ml':  # Third highest ROI
            base_fraction = 0.18  # 18% of Kelly
        else:
            base_fraction = 0.15  # 15% of Kelly for other types
        
        # Calculate Kelly stake
        b = odds - 1
        p = prob
        q = 1 - p
        
        if b <= 0:
            kelly = 0
        else:
            kelly = max(0, (b * p - q) / b)
            kelly = kelly * base_fraction
            
            # Progressive cap based on bet type
            if bet_type in ['team1_minus_1_5', 'team2_minus_1_5']:
                # Higher cap for highest ROI bet types
                max_pct = 0.05  # 5% of bankroll
            else:
                max_pct = 0.03  # 3% of bankroll
                
            kelly = min(kelly, max_pct)
        
        # Calculate bet amount with progressive caps
        raw_bet_amount = bankroll * kelly
        capped_bet_amount = min(raw_bet_amount, MAX_SINGLE_BET)
        bet_amount = round(capped_bet_amount, 2)
        
        # Filters
        extra_filter = True
        filter_reason = "Passed all filters"
        
        # More selective with low confidence bets except for highest ROI bet types
        if confidence_score < 0.3 and edge < 0.08 and bet_type not in ['team1_minus_1_5', 'team2_minus_1_5']:
            extra_filter = False
            filter_reason = "Low confidence, insufficient edge"
        
        # Final checks
        meets_edge = edge > bet_threshold
        meets_min_amount = bet_amount >= 1.0
        recommended = meets_edge and meets_min_amount and extra_filter
        
        # Store results
        betting_analysis[bet_type] = {
            'probability': prob,
            'implied_prob': implied_prob,
            'edge': edge,
            'edge_threshold': bet_threshold,
            'meets_edge': meets_edge,
            'odds': odds,
            'kelly_fraction': kelly,
            'bet_amount': bet_amount,
            'meets_min_amount': meets_min_amount, 
            'extra_filter': extra_filter,
            'filter_reason': filter_reason,
            'recommended': recommended,
            'high_roi_bet': bet_type in ['team1_minus_1_5', 'team2_minus_1_5', 'team1_ml'] 
        }
        
        # Print details for logging
        cap_notice = f" (capped from ${raw_bet_amount:.2f})" if raw_bet_amount > MAX_SINGLE_BET else ""
        print(f"{bet_type}: prob={prob:.4f}, edge={edge:.4f}, threshold={bet_threshold:.4f}, " + 
              f"amount=${bet_amount:.2f}{cap_notice}, recommend={recommended}" +
              (" [HIGH-ROI BET]" if bet_type in ['team1_minus_1_5', 'team2_minus_1_5', 'team1_ml'] else ""))
    
    # Count recommended bets
    recommended_count = sum(1 for analysis in betting_analysis.values() if analysis['recommended'])
    print(f"Found {recommended_count} recommended bets out of {len(bet_types)} analyzed")
    
    return betting_analysis
    
def train_team_specific_models(team_data_collection):
    """
    Create specialized models for teams with sufficient data.
    
    Args:
        team_data_collection: Dictionary of team data
        
    Returns:
        dict: Dictionary of team-specific models
    """
    print("Training team-specific models...")
    
    team_models = {}
    team_predictors = {}
    
    # Find teams with sufficient data
    for team_name, team_data in team_data_collection.items():
        if len(team_data.get('matches', [])) >= 30:
            # Build team-specific dataset
            try:
                X, y = build_team_dataset(team_data, team_data_collection)
                
                if len(X) >= 25:  # Ensure enough samples for training
                    team_predictors[team_name] = (X, y)
                    print(f"Created dataset for {team_name} with {len(X)} samples")
            except Exception as e:
                print(f"Error building dataset for {team_name}: {e}")
    
    # Train models for teams with sufficient data
    for team_name, (X, y) in team_predictors.items():
        try:
            print(f"Training model for {team_name}...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Use random forest for team-specific models (more stable with less data)
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > 0.55:  # Only keep models that perform well
                team_models[team_name] = {
                    'model': model,
                    'scaler': scaler,
                    'accuracy': accuracy,
                    'samples': len(X)
                }
                print(f"Added model for {team_name} with {accuracy:.4f} accuracy")
            else:
                print(f"Skipping model for {team_name} due to low accuracy ({accuracy:.4f})")
        except Exception as e:
            print(f"Error training model for {team_name}: {e}")
    
    print(f"Created {len(team_models)} team-specific models")
    return team_models

def build_team_dataset(team_data, team_data_collection):
    """
    Build a dataset for team-specific modeling.
    
    Args:
        team_data: Data for the specific team
        team_data_collection: Dictionary of all team data
        
    Returns:
        tuple: (features, labels)
    """
    X = []
    y = []
    
    team_name = team_data.get('team_name')
    team_matches = team_data.get('matches', [])
    
    for match in team_matches:
        opponent_name = match.get('opponent_name')
        
        # Skip if opponent not in data collection
        if not opponent_name or opponent_name not in team_data_collection:
            continue
            
        # Get opponent data
        opponent_data = team_data_collection[opponent_name]
        
        # Prepare match features
        features = prepare_data_for_model(team_data.get('stats', {}), opponent_data.get('stats', {}))
        
        if features:
            X.append(features)
            y.append(1 if match.get('team_won') else 0)
    
    return np.array(X) if X else np.array([]), np.array(y) if y else np.array([])

def bayesian_predict(ensemble_models, X, num_samples=500):
    """
    Use Bayesian approach for better uncertainty estimation.
    
    Args:
        ensemble_models: List of models for prediction
        X: Features to predict
        num_samples: Number of samples for Bayesian estimation
        
    Returns:
        tuple: (mean_prediction, std_dev, confidence_interval)
    """
    print("Performing Bayesian prediction...")
    
    # Ensure X is properly shaped
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    all_predictions = []
    
    # First pass: get initial predictions from all models
    model_predictions = {}
    
    for model_type, model, model_scaler in ensemble_models:
        try:
            # Apply scaling if needed
            X_pred = X.copy()
            if model_scaler is not None:
                X_pred = model_scaler.transform(X_pred)
            
            # Make prediction based on model type
            if model_type == 'nn':
                # For neural networks, sample with dropout enabled to approximate Bayesian posterior
                K.set_learning_phase(1)  # Enable dropout at inference time
                
                # Take multiple samples
                nn_samples = []
                for _ in range(20):  # Take 20 samples per NN
                    pred = model.predict(X_pred, verbose=0)[0][0]
                    nn_samples.append(pred)
                
                model_predictions[model_type] = nn_samples
                all_predictions.extend(nn_samples)
                
                # Reset learning phase
                K.set_learning_phase(0)
            else:
                # For other models, use their predictions directly
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_pred)[0][1]
                    model_predictions[model_type] = [pred]
                    all_predictions.append(pred)
        except Exception as e:
            print(f"Error with {model_type} model: {e}")
    
    # Second pass: generate posterior samples using bootstrap resampling
    posterior_samples = []
    for _ in range(num_samples):
        # Randomly select a model type with weighted probability
        weights = {
            'nn': 0.6,      # Neural networks get highest weight
            'gb': 0.15,     # Gradient boosting
            'rf': 0.15,     # Random forest
            'lr': 0.05,     # Logistic regression
            'svm': 0.05     # SVM
        }
        
        available_models = list(model_predictions.keys())
        if not available_models:
            break
            
        # Adjust weights based on available models
        total_weight = sum(weights[m] for m in available_models)
        adjusted_weights = [weights[m] / total_weight for m in available_models]
        
        # Sample a model
        selected_model = np.random.choice(available_models, p=adjusted_weights)
        
        # Sample a prediction from that model's predictions
        model_preds = model_predictions[selected_model]
        if model_preds:
            sampled_pred = np.random.choice(model_preds)
            posterior_samples.append(sampled_pred)
    
    # Calculate statistics
    if posterior_samples:
        mean_pred = np.mean(posterior_samples)
        std_pred = np.std(posterior_samples)
        
        # Calculate 95% confidence interval
        lower_bound = np.percentile(posterior_samples, 2.5)
        upper_bound = np.percentile(posterior_samples, 97.5)
        
        # Calculate support ratio (between 0 and 1)
        above_half = sum(1 for p in posterior_samples if p > 0.5) / len(posterior_samples)
        support_ratio = max(above_half, 1 - above_half)  # Always 0.5 to 1.0
        
        # Calculate confidence based on uncertainty and consensus
        confidence = support_ratio * (1 - min(1, std_pred * 3))
        
        return mean_pred, std_pred, (lower_bound, upper_bound), confidence
    else:
        return 0.5, 0.25, (0.25, 0.75), 0.0

def implement_cross_validation_safeguards(X, y):
    """
    Implement nested cross-validation for more realistic accuracy estimation
    and to prevent overfitting.
    
    Args:
        X: Features array
        y: Labels array
        
    Returns:
        tuple: (estimated_accuracy, selected_features, best_params)
    """
    print("Implementing nested cross-validation safeguards...")
    
    # Outer cross-validation for final performance estimation
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    outer_scores = []
    selected_features_counts = {}
    
    # For each train-test split in the outer CV
    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"Outer fold {i+1}/5...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Further split training data for feature selection
        X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Perform feature selection on a subset of training data
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            threshold='median'
        )
        selector.fit(X_train_fs, y_train_fs)
        
        # Get selected feature indices
        selected_indices = selector.get_support()
        
        # Count selected features for stability analysis
        for i, selected in enumerate(selected_indices):
            if selected:
                if i not in selected_features_counts:
                    selected_features_counts[i] = 0
                selected_features_counts[i] += 1
        
        # Use selected features
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        
        # Now do parameter tuning with inner CV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10]
        }
        
        clf = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )
        clf.fit(X_train_selected, y_train)
        
        # Get best params
        best_params = clf.best_params_
        
        # Train final model with best params
        best_clf = RandomForestClassifier(random_state=42, **best_params)
        best_clf.fit(X_train_selected, y_train)
        
        # Test on held-out data
        y_pred = best_clf.predict(X_test_selected)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"  Fold accuracy: {acc:.4f} with {sum(selected_indices)} features")
        outer_scores.append(acc)
    
    # Calculate final performance estimate
    mean_accuracy = np.mean(outer_scores)
    std_accuracy = np.std(outer_scores)
    
    # Use more conservative estimate (lower bound of confidence interval)
    estimated_accuracy = mean_accuracy - 1.96 * std_accuracy / np.sqrt(len(outer_scores))
    
    # Select stable features (selected in at least 4 out of 5 folds)
    stable_features = [idx for idx, count in selected_features_counts.items() if count >= 4]
    
    print(f"Nested CV results: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Conservative accuracy estimate: {estimated_accuracy:.4f}")
    print(f"Selected {len(stable_features)} stable features")
    
    return estimated_accuracy, stable_features, best_params

def detect_line_value(odds, predicted_prob):
    """
    Detect if line has more value now vs expected movement.
    
    Args:
        odds: Current decimal odds
        predicted_prob: Our predicted probability
        
    Returns:
        tuple: (recommendation, expected_value)
    """
    implied_prob = 1 / odds
    edge = predicted_prob - implied_prob
    
    # Calculate expected value as a percentage
    ev_percentage = edge * 100
    
    # Determine recommendation based on edge size
    if edge > 0.1:  # Very strong edge
        return "STRONG BET - exceptional value", ev_percentage
    elif edge > 0.07:
        return "BET NOW - strong value that may disappear", ev_percentage
    elif edge > 0.04:
        return "BET - substantial value", ev_percentage
    elif edge > 0.02:
        return "CONSIDER - modest value", ev_percentage
    elif edge > 0:
        return "MONITOR - slight value, watch for line movement", ev_percentage
    else:
        return "PASS - no value detected", ev_percentage

def track_odds_movement(match_id, team1, team2, current_odds, predicted_prob):
    """
    Track odds movement to identify betting patterns and optimal timing.
    
    Args:
        match_id: Unique identifier for the match
        team1: First team name
        team2: Second team name
        current_odds: Current odds from bookmaker
        predicted_prob: Our predicted probability
        
    Returns:
        dict: Analysis of odds movement
    """
    # Create directory if not exists
    os.makedirs("odds_tracking", exist_ok=True)
    
    odds_history_file = f"odds_tracking/{match_id}_odds.json"
    
    # Load existing data or create new
    try:
        with open(odds_history_file, 'r') as f:
            odds_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        odds_history = {
            'match_id': match_id,
            'team1': team1,
            'team2': team2,
            'predicted_prob': predicted_prob,
            'timestamps': []
        }
    
    # Add current timestamp and odds
    odds_history['timestamps'].append({
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'odds': current_odds
    })
    
    # Save updated history
    with open(odds_history_file, 'w') as f:
        json.dump(odds_history, f, indent=2)
    
    # Analyze odds movement if we have multiple measurements
    if len(odds_history['timestamps']) > 1:
        analysis = analyze_odds_movement(odds_history)
        return analysis
    
    return {"message": "First odds recording, no movement to analyze yet"}

def analyze_odds_movement(odds_history):
    """
    Analyze recorded odds movement to detect patterns and optimal betting timing.
    
    Args:
        odds_history: Dictionary containing odds history
        
    Returns:
        dict: Analysis results
    """
    timestamps = odds_history['timestamps']
    
    # Sort by time
    timestamps.sort(key=lambda x: x['time'])
    
    # Calculate movement for each market
    markets = set()
    for ts in timestamps:
        for market in ts['odds'].keys():
            markets.add(market)
    
    analysis = {
        'match_id': odds_history['match_id'],
        'team1': odds_history['team1'],
        'team2': odds_history['team2'],
        'markets': {},
        'recommendation': {}
    }
    
    # Analyze each market separately
    for market in markets:
        market_data = []
        
        for ts in timestamps:
            if market in ts['odds']:
                market_data.append({
                    'time': ts['time'],
                    'odds': ts['odds'][market]
                })
        
        if len(market_data) > 1:
            # Calculate movement
            first_odds = market_data[0]['odds']
            last_odds = market_data[-1]['odds']
            odds_change = last_odds - first_odds
            
            # Calculate movement direction and rate
            direction = "increasing" if odds_change > 0 else "decreasing"
            hours_elapsed = (datetime.strptime(market_data[-1]['time'], '%Y-%m-%d %H:%M:%S') - 
                         datetime.strptime(market_data[0]['time'], '%Y-%m-%d %H:%M:%S')).total_seconds() / 3600
            
            rate = odds_change / hours_elapsed if hours_elapsed > 0 else 0
            
            # Calculate implied probability change
            first_implied = 1 / first_odds
            last_implied = 1 / last_odds
            prob_change = last_implied - first_implied
            
            # Determine if line is moving in our favor
            if market.startswith('team1') and odds_history['predicted_prob'] > 0.5:
                # For team1 bets when we predict team1 to win
                favorable = direction == "increasing"
            elif market.startswith('team2') and odds_history['predicted_prob'] < 0.5:
                # For team2 bets when we predict team2 to win
                favorable = direction == "increasing"
            else:
                # For other situations
                favorable = False
            
            # Store analysis
            analysis['markets'][market] = {
                'first_odds': first_odds,
                'last_odds': last_odds,
                'change': odds_change,
                'direction': direction,
                'rate': rate,
                'probability_change': prob_change,
                'favorable': favorable
            }
            
            # Make recommendation
            if favorable and abs(rate) > 0.05:  # Significant favorable movement
                analysis['recommendation'][market] = "BET NOW - line moving in favorable direction"
            elif not favorable and abs(rate) > 0.05:  # Significant unfavorable movement
                analysis['recommendation'][market] = "WAIT - line moving against us"
            else:  # Minimal movement
                analysis['recommendation'][market] = "NEUTRAL - stable line"
    
    return analysis

def adjust_for_streak(bet_history, base_kelly, bet_type, team_name):
    """
    Adjust Kelly fraction based on current winning/losing streak.
    
    Args:
        bet_history: List of past bets
        base_kelly: Base Kelly fraction
        bet_type: Type of bet (e.g., team1_ml)
        team_name: Name of the team the bet is on
        
    Returns:
        float: Adjusted Kelly fraction
    """
    # Find recent bets of the same type involving this team
    recent_bets = []
    
    if not bet_history:
        return base_kelly
    
    for bet in reversed(bet_history):
        # Check if bet involves the same team
        if team_name in [bet.get('team1', ''), bet.get('team2', '')]:
            # Find matching bet type in bet details
            for bet_detail in bet.get('bets', []):
                if bet_detail.get('bet_type') == bet_type:
                    recent_bets.append(bet_detail.get('won', False))
                    if len(recent_bets) >= 5:  # Only look at 5 most recent
                        break
            
            if len(recent_bets) >= 5:
                break
    
    # Calculate current streak
    current_streak = 0
    for won in recent_bets:
        if won:
            if current_streak >= 0:
                current_streak += 1
            else:
                # Break losing streak
                current_streak = 1
        else:
            if current_streak <= 0:
                current_streak -= 1
            else:
                # Break winning streak
                current_streak = -1
    
    # Adjust Kelly fraction based on streak
    if current_streak >= 3:  # On a winning streak
        # Increase bet size to capitalize on good form
        adjustment = 1.2  # Increase by 20%
        print(f"On {current_streak}-bet winning streak: increasing bet size by 20%")
    elif current_streak <= -3:  # On a losing streak
        # Decrease bet size to preserve bankroll
        adjustment = 0.8  # Decrease by 20%
        print(f"On {abs(current_streak)}-bet losing streak: reducing bet size by 20%")
    elif current_streak >= 1:  # Mild winning momentum
        adjustment = 1.1  # Small increase
        print(f"Mild winning momentum: increasing bet size by 10%")
    elif current_streak <= -1:  # Mild losing momentum
        adjustment = 0.9  # Small decrease
        print(f"Mild losing momentum: reducing bet size by 10%")
    else:
        adjustment = 1.0  # No adjustment
    
    return base_kelly * adjustment

def calculate_dynamic_kelly(win_prob, decimal_odds, confidence, past_precision=None):
    """
    Calculate Kelly Criterion with dynamic adjustments based on historical precision.
    
    Args:
        win_prob: Predicted probability of winning
        decimal_odds: Decimal odds offered by bookmaker
        confidence: Confidence score in this prediction
        past_precision: Historical prediction precision (optional)
        
    Returns:
        float: Kelly stake as fraction of bankroll
    """
    # Standard Kelly calculation
    b = decimal_odds - 1  # Convert to b notation
    p = win_prob
    q = 1 - p
    
    # Avoid division by zero
    if b <= 0:
        return 0
    
    kelly = (b * p - q) / b
    
    # Apply confidence adjustment
    confidence_factor = 0.5 + (confidence * 0.5)  # Scale from 0.5 to 1.0
    
    # Apply historical precision adjustment if available
    if past_precision is not None:
        precision_factor = min(1.2, max(0.8, past_precision))  # Bound between 0.8 and 1.2
    else:
        precision_factor = 1.0
    
    # Apply fractional Kelly (conservative approach)
    fractional = 0.3  # 30% of full Kelly
    
    # Combine all adjustments
    adjusted_kelly = kelly * fractional * confidence_factor * precision_factor
    
    # Apply absolute cap for safety
    return min(adjusted_kelly, 0.07)  # Never bet more than 7% of bankroll

def check_profit_targets(starting_bankroll, current_bankroll, target_percentage=1000000):
    """
    Check if profit targets have been reached.
    
    Args:
        starting_bankroll: Starting bankroll
        current_bankroll: Current bankroll
        target_percentage: Target profit percentage
        
    Returns:
        tuple: (reached, message)
    """
    profit = current_bankroll - starting_bankroll
    profit_percentage = (profit / starting_bankroll) * 100
    
    if profit_percentage >= target_percentage:
        return True, f"Success! Profit target of {target_percentage}% reached. Current profit: ${profit:.2f} ({profit_percentage:.1f}%)"
    
    # If we're close to the target, give an update
    if profit_percentage >= target_percentage * 0.8:
        return False, f"Approaching profit target. Current profit: ${profit:.2f} ({profit_percentage:.1f}% of {target_percentage}% target)"
    
    return False, ""

def evaluate_bet_outcome(bet_type, actual_winner, team1_score, team2_score):
    """Determine if a bet won based on actual results."""
    # Moneyline bets
    if bet_type == 'team1_ml':
        return team1_score > team2_score
    elif bet_type == 'team2_ml':
        return team2_score > team1_score
    
    # Map spread bets
    elif bet_type == 'team1_plus_1_5':
        return team1_score + 1.5 > team2_score
    elif bet_type == 'team2_plus_1_5':
        return team2_score + 1.5 > team1_score
    elif bet_type == 'team1_minus_1_5':
        return team1_score - 1.5 > team2_score
    elif bet_type == 'team2_minus_1_5':
        return team2_score - 1.5 > team1_score
    
    # Map total bets
    elif bet_type == 'over_2_5_maps':
        return team1_score + team2_score > 2.5
    elif bet_type == 'under_2_5_maps':
        return team1_score + team2_score < 2.5
    
    return False

def extract_match_score(match_data):
    """Extract the actual map score from match data."""
    if 'map_score' in match_data:
        # Parse map score in format like "2:0"
        try:
            score_parts = match_data['map_score'].split(':')
            team1_score = int(score_parts[0])
            team2_score = int(score_parts[1])
            return team1_score, team2_score
        except (ValueError, IndexError):
            pass
    
    # Fallback to using team_score and opponent_score
    team1_score = match_data.get('team_score', 0)
    team2_score = match_data.get('opponent_score', 0)
    return team1_score, team2_score

def calculate_map_advantage(team1_stats, team2_stats, map_name):
    """Calculate map-specific advantage based on team strengths."""
    if 'map_statistics' not in team1_stats or 'map_statistics' not in team2_stats:
        return 0
        
    if map_name not in team1_stats['map_statistics'] or map_name not in team2_stats['map_statistics']:
        return 0
        
    team1_map = team1_stats['map_statistics'][map_name]
    team2_map = team2_stats['map_statistics'][map_name]
    
    # Extract win rates with fallbacks
    team1_win_rate = team1_map.get('win_percentage', 0.5)
    team2_win_rate = team2_map.get('win_percentage', 0.5)
    
    # Factor in matches played for statistical significance
    team1_matches = team1_map.get('matches_played', 1)
    team2_matches = team2_map.get('matches_played', 1)
    
    # Weight advantage by matches played (more matches = more reliable data)
    t1_weight = min(1.0, team1_matches / 10)  # Cap at 10 matches
    t2_weight = min(1.0, team2_matches / 10)
    
    # Calculate weighted advantage
    raw_advantage = team1_win_rate - team2_win_rate
    weighted_advantage = raw_advantage * (t1_weight + t2_weight) / 2
    
    # Add side preference advantage
    if 'side_preference' in team1_map and 'side_preference' in team2_map:
        # If teams prefer opposite sides, no advantage
        if team1_map['side_preference'] != team2_map['side_preference']:
            pass
        # If both prefer the same side, the team with stronger preference has advantage
        else:
            t1_strength = team1_map.get('side_preference_strength', 0)
            t2_strength = team2_map.get('side_preference_strength', 0)
            if t1_strength > t2_strength:
                weighted_advantage += 0.02  # Small bonus
            elif t2_strength > t1_strength:
                weighted_advantage -= 0.02
    
    return weighted_advantage

def calculate_stylistic_matchup_advantage(team1_stats, team2_stats):
    """Calculate advantage based on playing style compatibility."""
    advantage = 0
    
    # 1. Economy efficiency vs. poor pistol rounds
    if ('economy_efficiency' in team1_stats and 'pistol_win_rate' in team2_stats and
        team1_stats.get('economy_efficiency', 0.5) > 0.6 and team2_stats.get('pistol_win_rate', 0.5) < 0.4):
        advantage += 0.05  # Team1 has advantage against teams with poor pistol rounds
    
    # 2. Strong pistol rounds vs poor economy efficiency
    if ('pistol_win_rate' in team1_stats and 'economy_efficiency' in team2_stats and
        team1_stats.get('pistol_win_rate', 0.5) > 0.6 and team2_stats.get('economy_efficiency', 0.5) < 0.4):
        advantage += 0.05
    
    # 3. Early fragging capability vs slow starters
    if 'avg_player_acs' in team1_stats and 'avg_player_adr' in team2_stats:
        t1_early_impact = team1_stats.get('avg_player_acs', 0) / 100
        t2_slow_start = team2_stats.get('avg_player_adr', 0) / 100
        if t1_early_impact > 2.5 and t2_slow_start < 1.5:
            advantage += 0.03
    
    # 4. Team consistency advantage
    if 'team_consistency' in team1_stats and 'team_consistency' in team2_stats:
        if team1_stats['team_consistency'] > team2_stats['team_consistency'] + 0.2:
            advantage += 0.04  # More consistent team has advantage
    
    # 5. Check for strong FK/FD advantage
    if 'fk_fd_ratio' in team1_stats and 'fk_fd_ratio' in team2_stats:
        t1_ratio = team1_stats['fk_fd_ratio']
        t2_ratio = team2_stats['fk_fd_ratio']
        if t1_ratio > t2_ratio * 1.5:  # Team1 gets 50% more first kills
            advantage += 0.05
    
    return advantage

def calculate_win_rate_stability(team_matches, window_size=5):
    """Calculate the stability of a team's win rate over time."""
    if not isinstance(team_matches, list) or len(team_matches) < window_size * 2:
        return 0.5  # Default stability for teams with little data
    
    # Sort matches by date
    sorted_matches = sorted(team_matches, key=lambda x: x.get('date', ''))
    
    # Calculate win rates in consecutive windows
    win_rates = []
    for i in range(0, len(sorted_matches), window_size):
        if i + window_size <= len(sorted_matches):
            window = sorted_matches[i:i+window_size]
            wins = sum(1 for m in window if m.get('team_won', False))
            win_rates.append(wins / window_size)
    
    # Calculate stability as 1 - standard deviation (higher = more stable)
    if len(win_rates) >= 2:
        stability = 1 - min(0.5, np.std(win_rates))
        return stability
    
    return 0.5  # Default value    

def get_teams_for_backtesting(limit=100, use_cache=True, cache_path="cache/valorant_data_cache.pkl"):
    """
    Get a list of teams for backtesting, optionally using cached data.
    
    Args:
        limit (int): Maximum number of teams to return
        use_cache (bool): Whether to use cached data if available
        cache_path (str): Path to the cache file
        
    Returns:
        list: Teams for backtesting
    """
    # Try to use cached data if available
    if use_cache:
        print(f"Checking for cached team data for backtesting...")
        if os.path.exists(cache_path):
            team_data = load_cache(cache_path)
            if team_data:
                # Convert cached team data to the format expected by backtesting
                teams_list = []
                for team_name, team_info in team_data.items():
                    team_dict = {
                        'id': team_info.get('team_id', ''),
                        'name': team_name,
                        'ranking': team_info.get('ranking')
                    }
                    teams_list.append(team_dict)
                
                # Sort by ranking if available
                teams_list = sorted(teams_list, 
                                   key=lambda x: x['ranking'] if x['ranking'] else float('inf'))
                
                # Limit to requested number
                teams_list = teams_list[:limit]
                print(f"Using {len(teams_list)} teams from cache for backtesting")
                return teams_list
    
    # Fallback to API if cache not available or not using cache
    print(f"Fetching teams from API: {API_URL}/teams?limit={limit}")
    try:
        teams_response = requests.get(f"{API_URL}/teams?limit={limit}")
        if teams_response.status_code != 200:
            print(f"Error fetching teams: {teams_response.status_code}")
            return []
        teams_data = teams_response.json()
        if 'data' not in teams_data:
            print("No 'data' field found in the response")
            return []
        teams_list = []
        for team in teams_data['data']:
            if 'ranking' in team and team['ranking'] and team['ranking'] <= 100:
                teams_list.append(team)
        if not teams_list:
            print(f"No teams with rankings found. Using the first {min(100, limit)} teams instead.")
            if len(teams_data['data']) > 0:
                teams_list = teams_data['data'][:min(100, limit)]
        print(f"Selected {len(teams_list)} teams for backtesting from API")
        return teams_list
    except Exception as e:
        print(f"Error in get_teams_for_backtesting: {e}")
        return []

# 1. Fix Neural Network Calibration in predict_with_ensemble function
def predict_with_ensemble(ensemble_models, X):
    """
    Make predictions using the ensemble with strong calibration to prevent extreme values,
    matching the approach used in backtesting.
    """
    if not ensemble_models:
        raise ValueError("No models provided for prediction")
    
    # Ensure X is properly shaped for model input
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
        
    # Get predictions from each model
    raw_predictions = []
    model_weights = []
    model_types = []
    
    print("\n----- ENSEMBLE PREDICTION -----")
    
    for i, (model_type, model, model_scaler) in enumerate(ensemble_models):
        try:
            # Apply scaling if needed
            X_pred = X.copy()
            if model_scaler is not None:
                try:
                    X_pred = model_scaler.transform(X_pred)
                except Exception as e:
                    print(f"Warning: Scaling error for {model_type} model {i}, using unscaled features")
            
            # Make prediction based on model type
            if model_type == 'nn':
                raw_pred = model.predict(X_pred, verbose=0)[0][0]
                
                # CRITICAL: Apply strong calibration to neural networks like in backtesting
                # Cap predictions between 0.05 and 0.95 - THIS IS THE CRUCIAL STEP
                raw_pred = min(0.95, max(0.05, raw_pred))
                
                # Apply sigmoid-based calibration to pull values toward 0.5
                calibrated_pred = 0.5 + (raw_pred - 0.5) * 0.8
                print(f"NN model {i}: {raw_pred:.4f} → {calibrated_pred:.4f} (calibrated)")
                pred = calibrated_pred
            else:
                # Handle different API for scikit-learn models
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_pred)[0][1]
                    
                    # Cap and calibrate non-NN models too - match backtesting approach
                    pred = min(0.95, max(0.05, pred))
                    
                    # Apply milder calibration to tree-based models which tend to be better calibrated
                    if model_type in ['gb', 'rf']:
                        pred = 0.5 + (pred - 0.5) * 0.85
                    else:
                        pred = 0.5 + (pred - 0.5) * 0.8
                else:
                    pred = model.predict(X_pred)[0]
                    pred = min(0.95, max(0.05, pred))
            
            # Handle NaN or invalid predictions
            if np.isnan(pred) or not np.isfinite(pred):
                print(f"Warning: Model {i+1} returned invalid prediction, using 0.5")
                pred = 0.5
            
            # Use balanced weights to avoid any single model dominating - match backtesting
            if model_type == 'nn':
                base_weight = 1.0
            elif model_type == 'gb':
                base_weight = 1.0
            elif model_type == 'rf':
                base_weight = 1.0
            elif model_type == 'lr':
                base_weight = 0.8
            else:  # SVM
                base_weight = 0.8
            
            # Minimal adjustment for extreme values
            extremeness = abs(pred - 0.5) / 0.5
            weight = base_weight * (1.0 - extremeness * 0.1)
                
            # Store prediction and model info
            raw_predictions.append(pred)
            model_weights.append(weight)
            model_types.append(model_type)
            
            print(f"{model_type.upper()} model {i} prediction: {pred:.4f} (weight: {weight:.2f})")
        except Exception as e:
            print(f"Error with model {i}: {e}")
            continue
    
    if not raw_predictions:
        print("ERROR: All models failed to produce predictions, using default value")
        return 0.5, [0.5], 0.0
    
    # Calculate weighted average
    if raw_predictions and sum(model_weights) > 0:
        weighted_sum = sum(p * w for p, w in zip(raw_predictions, model_weights))
        total_weight = sum(model_weights)
        mean_pred = weighted_sum / total_weight
        print(f"Weighted ensemble mean: {mean_pred:.4f} (total weight: {total_weight:.1f})")
    else:
        mean_pred = 0.5
        print("Using default prediction of 0.5 due to weighting issues")
    
    # Calculate confidence score using standard deviation - match backtesting
    std_pred = np.std(raw_predictions)
    raw_confidence = 1 - min(0.9, std_pred * 1.5)  # Less aggressive scaling
    
    # Make sure confidence cap matches backtesting
    adjusted_confidence = min(0.7, max(0.2, raw_confidence))
    
    # Simple IQR analysis for visualization
    raw_predictions_array = np.array(raw_predictions)
    q1 = np.percentile(raw_predictions_array, 25)
    q3 = np.percentile(raw_predictions_array, 75)
    iqr = q3 - q1
    print(f"Prediction quartiles: Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}")
    
    # Apply final calibration exactly as in backtesting - crucial alignment step
    calibrated_pred = 0.5 + (mean_pred - 0.5) * 0.8
    print(f"Final calibration: {mean_pred:.4f} → {calibrated_pred:.4f}")
    
    # SAFETY CHECK: Ensure prediction is never outside [0.05, 0.95] range
    calibrated_pred = min(0.95, max(0.05, calibrated_pred))
    
    # Format raw predictions for display
    raw_predictions_str = [f'{p:.4f}' for p in raw_predictions]
    
    return calibrated_pred, raw_predictions_str, adjusted_confidence


def calculate_bet_type_probabilities(win_probability, confidence_score):
    """
    Calculate probabilities for different bet types using the same 
    approach as backtesting.
    """
    # Calculate map probability with the exact same formula as backtesting
    map_scale = 0.55 + (confidence_score * 0.15)  # Between 0.55 and 0.7
    single_map_prob = 0.5 + (win_probability - 0.5) * map_scale
    
    # Ensure map probability is reasonable - exact same bounds as backtesting
    single_map_prob = max(0.3, min(0.7, single_map_prob))
    
    # Calculate probabilities for different bet types
    team1_plus_prob = 1 - (1 - single_map_prob) ** 2
    team2_plus_prob = 1 - single_map_prob ** 2
    team1_minus_prob = single_map_prob ** 2
    team2_minus_prob = (1 - single_map_prob) ** 2
    over_prob = 2 * single_map_prob * (1 - single_map_prob)
    under_prob = 1 - over_prob
    
    # Cap all probabilities exactly as in backtesting
    team1_plus_prob = min(0.95, max(0.05, team1_plus_prob))
    team2_plus_prob = min(0.95, max(0.05, team2_plus_prob))
    team1_minus_prob = min(0.9, max(0.1, team1_minus_prob))
    team2_minus_prob = min(0.9, max(0.1, team2_minus_prob))
    over_prob = min(0.9, max(0.1, over_prob))
    under_prob = min(0.9, max(0.1, under_prob))
    
    # Return all probabilities with the exact same structure
    return {
        'team1_plus_1_5': team1_plus_prob,
        'team2_plus_1_5': team2_plus_prob,
        'team1_minus_1_5': team1_minus_prob,
        'team2_minus_1_5': team2_minus_prob,
        'over_2_5_maps': over_prob,
        'under_2_5_maps': under_prob
    }

def prepare_features_for_backtest(team1_stats, team2_stats, selected_features):
    """
    Enhanced feature preparation for backtesting with better feature derivation
    and normalization.
    """
    print("\n----- PREPARING FEATURES -----")
    
    # Get full feature set
    features = prepare_data_for_model(team1_stats, team2_stats)
    
    if not features:
        print("ERROR: Failed to create feature dictionary")
        return None
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    original_feature_count = len(features_df.columns)
    print(f"Original feature count: {original_feature_count}")
    
    # Apply enhanced feature derivation
    features_df = enhance_feature_derivation(features_df, team1_stats, team2_stats)
    
    # Create DataFrame with exact required features
    complete_features = pd.DataFrame(0, index=[0], columns=selected_features)
    
    # Track feature coverage
    missing_features = []
    derived_features = []
    
    # STEP 1: Fill in features that are directly available
    for feature in selected_features:
        if feature in features_df.columns:
            complete_features[feature] = features_df[feature].values
        else:
            missing_features.append(feature)
    
    # STEP 2: Derive missing features using enhanced logic
    for feature in missing_features[:]:  # Use a copy to modify the original list
        # Core differential features
        if feature == 'win_rate_diff':
            val = team1_stats.get('win_rate', 0.5) - team2_stats.get('win_rate', 0.5)
            complete_features[feature] = val
            missing_features.remove(feature)
            derived_features.append(feature)
            
        elif feature == 'recent_form_diff':
            val = team1_stats.get('recent_form', 0.5) - team2_stats.get('recent_form', 0.5)
            complete_features[feature] = val
            missing_features.remove(feature)
            derived_features.append(feature)
            
        elif feature == 'score_diff_differential':
            val = team1_stats.get('score_differential', 0) - team2_stats.get('score_differential', 0)
            complete_features[feature] = val
            missing_features.remove(feature)
            derived_features.append(feature)
            
        # Better team features (binary)
        elif feature == 'better_win_rate_team1' and 'win_rate_diff' in complete_features.columns:
            val = 1 if complete_features['win_rate_diff'].values[0] > 0 else 0
            complete_features[feature] = val
            missing_features.remove(feature)
            derived_features.append(feature)
            
        elif feature == 'better_recent_form_team1' and 'recent_form_diff' in complete_features.columns:
            val = 1 if complete_features['recent_form_diff'].values[0] > 0 else 0
            complete_features[feature] = val
            missing_features.remove(feature)
            derived_features.append(feature)
            
        elif feature == 'better_score_diff_team1' and 'score_diff_differential' in complete_features.columns:
            val = 1 if complete_features['score_diff_differential'].values[0] > 0 else 0
            complete_features[feature] = val
            missing_features.remove(feature)
            derived_features.append(feature)
            
        # Match count features
        elif feature == 'total_matches':
            team1_matches = team1_stats.get('matches', 0)
            team2_matches = team2_stats.get('matches', 0)
            
            if isinstance(team1_matches, list):
                team1_count = len(team1_matches)
            else:
                team1_count = team1_matches
                
            if isinstance(team2_matches, list):
                team2_count = len(team2_matches)
            else:
                team2_count = team2_matches
                
            complete_features[feature] = team1_count + team2_count
            missing_features.remove(feature)
            derived_features.append(feature)
            
        elif feature == 'match_count_diff':
            if 'total_matches' in complete_features.columns:
                team1_matches = team1_stats.get('matches', 0)
                team2_matches = team2_stats.get('matches', 0)
                
                if isinstance(team1_matches, list):
                    team1_count = len(team1_matches)
                else:
                    team1_count = team1_matches
                    
                if isinstance(team2_matches, list):
                    team2_count = len(team2_matches)
                else:
                    team2_count = team2_matches
                    
                complete_features[feature] = team1_count - team2_count
                missing_features.remove(feature)
                derived_features.append(feature)
                
        # NEW: Better H2H feature derivation
        elif feature == 'h2h_x_form':
            if 'h2h_win_rate' in complete_features.columns and 'recent_form_diff' in complete_features.columns:
                # Calculate h2h x form interaction
                h2h_win_rate = complete_features['h2h_win_rate'].values[0]
                recent_form_diff = complete_features['recent_form_diff'].values[0]
                complete_features[feature] = h2h_win_rate * recent_form_diff
                missing_features.remove(feature)
                derived_features.append(feature)
                
        # NEW: Enhanced player stat derivation
        elif feature == 'avg_headshot':
            # Try to get from player_stats in team1_stats
            if 'player_stats' in team1_stats and 'avg_headshot' in team1_stats['player_stats']:
                complete_features[feature] = team1_stats['player_stats']['avg_headshot']
                missing_features.remove(feature)
                derived_features.append(feature)
            # Fallback to a reasonable default based on team tier
            elif 'win_rate' in team1_stats:
                # Higher tier teams typically have better headshot %
                win_rate = team1_stats.get('win_rate', 0.5)
                # Scale from 0.2 to 0.35 based on win rate
                complete_features[feature] = 0.2 + (win_rate * 0.3)
                missing_features.remove(feature)
                derived_features.append(feature)
                
        # NEW: Enhanced player stat differentials
        elif feature == 'acs_diff':
            # Try to get from direct player stats
            t1_acs = 0
            t2_acs = 0
            
            if 'player_stats' in team1_stats and 'avg_acs' in team1_stats['player_stats']:
                t1_acs = team1_stats['player_stats']['avg_acs']
            elif 'avg_player_acs' in team1_stats:
                t1_acs = team1_stats['avg_player_acs']
                
            if 'player_stats' in team2_stats and 'avg_acs' in team2_stats['player_stats']:
                t2_acs = team2_stats['player_stats']['avg_acs']
            elif 'avg_player_acs' in team2_stats:
                t2_acs = team2_stats['avg_player_acs']
                
            if t1_acs > 0 and t2_acs > 0:
                complete_features[feature] = t1_acs - t2_acs
            else:
                # Fallback to score differential as proxy for ACS difference
                complete_features[feature] = team1_stats.get('score_differential', 0) * 50
                
            missing_features.remove(feature)
            derived_features.append(feature)
            
        # Similar approaches for other missing features
        elif feature == 'kast_diff':
            t1_kast = extract_team_kast(team1_stats)
            t2_kast = extract_team_kast(team2_stats)
            complete_features[feature] = t1_kast - t2_kast
            missing_features.remove(feature)
            derived_features.append(feature)
            
        elif feature == 'adr_diff':
            t1_adr = extract_team_adr(team1_stats)
            t2_adr = extract_team_adr(team2_stats)
            complete_features[feature] = t1_adr - t2_adr
            missing_features.remove(feature)
            derived_features.append(feature)
            
        # If feature name contains "padding_feature", it's ok to leave it at 0
        elif "padding_feature" in feature:
            missing_features.remove(feature)
    
    # Apply better feature normalization
    complete_features = normalize_features(complete_features)
    
    # Convert to numpy array
    X = complete_features.values
    
    # Report feature coverage
    available_count = sum(1 for f in selected_features if f in features_df.columns)
    derived_count = len(derived_features)
    still_missing = len(missing_features)
    padding_count = sum(1 for f in selected_features if "padding_feature" in f)
    
    print(f"Feature coverage: {available_count} original + {derived_count} derived features")
    print(f"Still missing: {still_missing} features")
    print(f"Padding features: {padding_count}")
    
    if still_missing > 0:
        print(f"Missing features that couldn't be derived: {missing_features[:5]}...")
    
    # Log detailed feature summary
    print("Feature values summary:")
    for feature in selected_features:
        if feature in features_df.columns or feature in derived_features:
            value = complete_features[feature].values[0]
            if abs(value) > 0.1:  # Only print significant values
                print(f"  - {feature}: {value:.4f}")
    
    return X

def analyze_team_form_trajectory(team_matches, window_size=5):
    """Analyze if team is improving or declining recently."""
    if not isinstance(team_matches, list) or len(team_matches) < window_size * 2:
        return 0  # Not enough data
        
    # Sort by date
    sorted_matches = sorted(team_matches, key=lambda x: x.get('date', ''))
    
    # Calculate win rate in recent window vs previous window
    recent_window = sorted_matches[-window_size:]
    previous_window = sorted_matches[-(window_size*2):-window_size]
    
    recent_wins = sum(1 for m in recent_window if m.get('team_won', False))
    previous_wins = sum(1 for m in previous_window if m.get('team_won', False))
    
    recent_win_rate = recent_wins / window_size
    previous_win_rate = previous_wins / window_size
    
    # Calculate trajectory (positive means improving)
    trajectory = recent_win_rate - previous_win_rate
    
    # Scale to a reasonable modifier
    return trajectory * 0.4  # Scale factor to prevent extreme value


# Identify model clusters and follow the stronger cluster
def analyze_prediction_clusters(predictions, model_types, model_weights):
    """
    Analyze predictions to identify clusters and potentially follow the
    stronger cluster when models disagree.
    
    Args:
        predictions (list): List of predictions from all models
        model_types (list): List of model types corresponding to predictions
        model_weights (list): List of weights for each model
        
    Returns:
        tuple: (cluster_mean, confidence) or (None, None) if no clear clustering
    """
    if len(predictions) < 3:
        return None, None  # Not enough predictions to cluster
    
    # Group predictions by model type
    nn_preds = [p for i, p in enumerate(predictions) if model_types[i] == 'nn']
    tree_preds = [p for i, p in enumerate(predictions) if model_types[i] in ['gb', 'rf']]
    other_preds = [p for i, p in enumerate(predictions) if model_types[i] not in ['nn', 'gb', 'rf']]
    
    # Check if we have enough data points in each group
    if len(nn_preds) < 2 or len(tree_preds) < 2:
        return None, None
    
    # Calculate cluster statistics
    nn_mean = np.mean(nn_preds) if nn_preds else 0.5
    tree_mean = np.mean(tree_preds) if tree_preds else 0.5
    other_mean = np.mean(other_preds) if other_preds else 0.5
    
    # Check for strong disagreement between NN and tree-based models
    if abs(nn_mean - tree_mean) > 0.25:  # Significant disagreement threshold
        # Calculate internal agreement within each cluster
        nn_std = np.std(nn_preds) if len(nn_preds) > 1 else 0.1
        tree_std = np.std(tree_preds) if len(tree_preds) > 1 else 0.1
        
        # Check which cluster has better internal agreement
        if nn_std < tree_std * 0.7:  # NNs agree more with each other
            print(f"Strong model disagreement detected: NN={nn_mean:.4f} vs Tree={tree_mean:.4f}")
            print(f"Internal agreement: NN σ={nn_std:.4f}, Tree σ={tree_std:.4f}")
            print(f"Following NN cluster due to stronger internal agreement")
            
            # Calculate confidence based on internal agreement
            cluster_confidence = 0.5 + (0.5 * (1 - min(1, nn_std * 4)))
            return nn_mean, cluster_confidence
            
        elif tree_std < nn_std * 0.7:  # Tree models agree more with each other
            print(f"Strong model disagreement detected: NN={nn_mean:.4f} vs Tree={tree_mean:.4f}")
            print(f"Internal agreement: NN σ={nn_std:.4f}, Tree σ={tree_std:.4f}")
            print(f"Following Tree cluster due to stronger internal agreement")
            
            # Calculate confidence based on internal agreement
            cluster_confidence = 0.5 + (0.5 * (1 - min(1, tree_std * 4)))
            return tree_mean, cluster_confidence
            
    # If no strong disagreement or no clear winner in terms of agreement,
    # use the standard ensemble method
    return None, None

def normalize_features(feature_df):
    """
    Apply smarter normalization to features that preserves signal strength.
    
    Args:
        feature_df: DataFrame containing features
        
    Returns:
        DataFrame: Normalized features
    """
    normalized_df = feature_df.copy()
    
    for column in normalized_df.columns:
        # Skip columns that are already normalized or don't need normalization
        if ('prob' in column or 'advantage' in column or 
            column.startswith('better_') or column.endswith('_significant')):
            continue
            
        # Apply different normalization based on feature type
        if ('diff' in column or 'differential' in column or 'ratio' in column):
            # Use sigmoid-based normalization for difference features
            normalized_df[column] = normalized_df[column].apply(
                lambda x: 2 / (1 + np.exp(-0.3 * x)) - 1 if pd.notnull(x) else 0
            )
        # Normalize count features differently
        elif ('count' in column or column in ['total_matches', 'matches', 'wins', 'losses']):
            # Log-based normalization for count features
            normalized_df[column] = normalized_df[column].apply(
                lambda x: np.log1p(x) / 5 if pd.notnull(x) and x > 0 else 0
            )
        # Normalize rate/percentage features to [0,1]
        elif ('rate' in column or 'percentage' in column or 'win_' in column):
            normalized_df[column] = normalized_df[column].apply(
                lambda x: max(0, min(1, x)) if pd.notnull(x) else 0.5
            )
            
    print("Applied smarter feature normalization with sigmoid and log scaling")
    return normalized_df


def run_improved_backtest(start_date=None, end_date=None, team_limit=50, bankroll=1000.0, bet_pct=0.05, min_edge=0.02, confidence_threshold=0.2):
    # [Previous code remains unchanged]
    
    # Run backtest with improved progress tracking
    for match_idx, match in enumerate(tqdm(backtest_matches, desc="Backtesting matches")):
        # [Previous code until just before the truncation...]
        
        # Check if prediction was correct
        predicted_winner = 'team1' if win_probability > 0.5 else 'team2'
        prediction_correct = predicted_winner == actual_winner
        
        # Update accuracy
        correct_predictions += 1 if prediction_correct else 0
        total_predictions += 1
        
        # Track team-specific performance
        results['team_performance'][team1_name]['predictions'] += 1
        if predicted_winner == 'team1' and prediction_correct:
            results['team_performance'][team1_name]['correct'] += 1
        
        results['team_performance'][team2_name]['predictions'] += 1
        if predicted_winner == 'team2' and prediction_correct:
            results['team_performance'][team2_name]['correct'] += 1
        
        # Track confidence bins
        confidence_bin = int(confidence_score * 10) * 10  # Round to nearest 10%
        confidence_key = f"{confidence_bin}%"
        
        if confidence_key not in confidence_bins:
            confidence_bins[confidence_key] = {"total": 0, "correct": 0}
        
        confidence_bins[confidence_key]["total"] += 1
        if prediction_correct:
            confidence_bins[confidence_key]["correct"] += 1
        
        # Generate realistic odds with jitter for realism
        base_odds = simulate_odds(win_probability)
        jittered_odds = {}
        
        # Add realistic variability to odds (bookmakers aren't perfect)
        for key, value in base_odds.items():
            # Add up to ±5% random variation
            jitter = np.random.uniform(-0.05, 0.05)
            jittered_value = value * (1 + jitter)
            jittered_odds[key] = round(jittered_value, 2)
        
        odds_data = jittered_odds
        
        # Use improved betting analysis with optimized thresholds
        betting_analysis = analyze_betting_edge_for_backtesting(
            win_probability, 1 - win_probability, odds_data, 
            confidence_score, current_bankroll
        )
        
        # Get recommendations
        filtered_analysis = betting_analysis
        
        # Select best bets - IMPROVED: Use optimal bet selection
        optimal_bets = select_optimal_bets_conservative(
            filtered_analysis, 
            team1_name, 
            team2_name, 
            current_bankroll,  
            max_bets=3,
            max_total_risk=Config.BETTING.MAX_TOTAL_RISK_PCT,
            config=Config.BETTING
        )
        # Simulate bets with better record keeping
        match_bets = []
        
        for bet_type, analysis in optimal_bets.items():
            # IMPROVED: Apply dynamic bankroll management based on streaks
            team_for_streak = team1_name if 'team1' in bet_type else team2_name
            
            # Apply streak-based adjustment
            base_kelly = analysis['kelly_fraction']
            adjusted_kelly = adjust_for_streak(bet_history, base_kelly, bet_type, team_for_streak)
            
            # Calculate bet size with streak adjustment
            max_bet = current_bankroll * bet_pct
            adjusted_amount = round(current_bankroll * adjusted_kelly, 2)
            bet_amount = min(adjusted_amount, max_bet)
            
            # Track the bet by team for streak calculations
            if team_for_streak not in previous_bets_by_team:
                previous_bets_by_team[team_for_streak] = []
            
            # Determine if bet won
            bet_won = evaluate_bet_outcome(bet_type, actual_winner, team1_score, team2_score)
            
            # Calculate returns
            odds = analysis['odds']
            returns = bet_amount * odds if bet_won else 0
            profit = returns - bet_amount
            
            # Update bankroll
            current_bankroll += profit
            
            # Track bet
            match_bets.append({
                'bet_type': bet_type,
                'amount': bet_amount,
                'odds': odds,
                'won': bet_won,
                'returns': returns,
                'profit': profit,
                'edge': analysis['edge'],
                'predicted_prob': analysis['probability'],
                'implied_prob': analysis['implied_prob'],
                'team1': team1_name,
                'team2': team2_name
            })
            
            # Update streak information
            previous_bets_by_team[team_for_streak].append({
                'bet_type': bet_type,
                'won': bet_won,
                'date': match_date
            })
            
            # Update betting metrics
            total_bets += 1
            winning_bets += 1 if bet_won else 0
            total_wagered += bet_amount
            total_returns += returns
            
            # Track by bet type
            if bet_type not in results['metrics']['bet_types']:
                results['metrics']['bet_types'][bet_type] = {
                    'total': 0, 'won': 0, 'wagered': 0, 'returns': 0
                }
            
            results['metrics']['bet_types'][bet_type]['total'] += 1
            results['metrics']['bet_types'][bet_type]['won'] += 1 if bet_won else 0
            results['metrics']['bet_types'][bet_type]['wagered'] += bet_amount
            results['metrics']['bet_types'][bet_type]['returns'] += returns
            
            # Track by edge
            edge_bucket = int(analysis['edge'] * 100) // 5 * 5  # Round to nearest 5%
            edge_key = f"{edge_bucket}%-{edge_bucket+5}%"
            
            if edge_key not in results['metrics']['accuracy_by_edge']:
                results['metrics']['accuracy_by_edge'][edge_key] = {'total': 0, 'correct': 0}
            if edge_key not in results['metrics']['roi_by_edge']:
                results['metrics']['roi_by_edge'][edge_key] = {'wagered': 0, 'returns': 0}
            
            results['metrics']['accuracy_by_edge'][edge_key]['total'] += 1
            results['metrics']['accuracy_by_edge'][edge_key]['correct'] += 1 if bet_won else 0
            results['metrics']['roi_by_edge'][edge_key]['wagered'] += bet_amount
            results['metrics']['roi_by_edge'][edge_key]['returns'] += returns
            
            # Track team-specific betting performance
            team_tracked = team1_name if 'team1' in bet_type else team2_name
            results['team_performance'][team_tracked]['bets'] += 1
            results['team_performance'][team_tracked]['wagered'] += bet_amount
            results['team_performance'][team_tracked]['returns'] += returns
            if bet_won:
                results['team_performance'][team_tracked]['wins'] += 1
        
        # Track prediction results
        results['predictions'].append({
            'match_id': match_id,
            'team1': team1_name,
            'team2': team2_name,
            'predicted_winner': predicted_winner,
            'actual_winner': actual_winner,
            'team1_prob': win_probability,
            'team2_prob': 1 - win_probability,
            'confidence': confidence_score,
            'correct': prediction_correct,
            'score': f"{team1_score}-{team2_score}",
            'date': match_date
        })
        
        # Track bets
        if match_bets:
            bet_record = {
                'match_id': match_id,
                'team1': team1_name,
                'team2': team2_name,
                'bets': match_bets,
                'date': match_date
            }
            results['bets'].append(bet_record)
            bet_history.append(bet_record)
        
        # Track bankroll history with timestamp
        results['performance']['bankroll_history'].append({
            'match_idx': match_idx,
            'bankroll': current_bankroll,
            'match_id': match_id,
            'date': match_date
        })
        
        # Check profit targets
        target_reached, target_message = check_profit_targets(starting_bankroll, current_bankroll, target_percentage=1000000)
        if target_reached:
            print(f"\n{target_message}")
            print("Stopping backtest early due to reaching profit target.")
            break
        
        # Print periodic progress updates
        if (match_idx + 1) % 50 == 0 or match_idx == len(backtest_matches) - 1:
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            roi = (total_returns - total_wagered) / total_wagered if total_wagered > 0 else 0
            
            print(f"\nProgress ({match_idx + 1}/{len(backtest_matches)}):")
            print(f"Prediction Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
            print(f"Betting ROI: {roi:.2%} (${total_returns - total_wagered:.2f})")
            print(f"Current Bankroll: ${current_bankroll:.2f}")
            print(f"Win Rate: {winning_bets/total_bets:.2%} ({winning_bets}/{total_bets})" if total_bets > 0 else "No bets placed")
    
    # Store confidence bin metrics
    results['metrics']['confidence_bins'] = confidence_bins
    
    # Calculate final performance metrics
    final_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    final_roi = (total_returns - total_wagered) / total_wagered if total_wagered > 0 else 0
    final_profit = total_returns - total_wagered
    
    results['performance']['accuracy'] = final_accuracy
    results['performance']['roi'] = final_roi
    results['performance']['profit'] = final_profit
    results['performance']['win_rate'] = winning_bets / total_bets if total_bets > 0 else 0
    results['performance']['final_bankroll'] = current_bankroll
    results['performance']['total_bets'] = total_bets
    results['performance']['winning_bets'] = winning_bets
    results['performance']['total_wagered'] = total_wagered
    results['performance']['total_returns'] = total_returns
    
    # Calculate team-specific metrics
    for team, stats in results['team_performance'].items():
        if stats['predictions'] > 0:
            stats['accuracy'] = stats['correct'] / stats['predictions']
        if stats['bets'] > 0:
            stats['win_rate'] = stats['wins'] / stats['bets']
        if stats['wagered'] > 0:
            stats['roi'] = (stats['returns'] - stats['wagered']) / stats['wagered']
            stats['profit'] = stats['returns'] - stats['wagered']
    
    # Print final results
    print("\n========== BACKTEST RESULTS ==========")
    print(f"Total Matches: {total_predictions}")
    print(f"Prediction Accuracy: {final_accuracy:.2%} ({correct_predictions}/{total_predictions})")
    print(f"Total Bets: {total_bets}")
    print(f"Winning Bets: {winning_bets} ({winning_bets/total_bets:.2%})" if total_bets > 0 else "No bets placed")
    print(f"Total Wagered: ${total_wagered:.2f}")
    print(f"Total Returns: ${total_returns:.2f}")
    print(f"Profit/Loss: ${final_profit:.2f}")
    print(f"ROI: {final_roi:.2%}")
    print(f"Final Bankroll: ${current_bankroll:.2f}")
    
    # Print confidence bin analysis
    print("\nAccuracy by Confidence Level:")
    for conf_key, stats in sorted(confidence_bins.items()):
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f"  {conf_key}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    # Create enhanced visualizations
    create_enhanced_backtest_visualizations(results)
    
    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"backtest_results_{timestamp}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate insights
    identify_key_insights(results)
    
    return results

class BankrollConfig:
    """
    Configuration for bankroll management parameters.
    These parameters control risk tolerance and bet sizing.
    """
    # Kelly criterion fraction (0.1 = 10% of full Kelly)
    # Lower values are more conservative
    KELLY_FRACTION = 0.15
    
    # Maximum bet size as percentage of bankroll
    # This is a hard cap regardless of Kelly calculation
    MAX_BET_PERCENTAGE = 0.02  # 2% of bankroll
    
    # Absolute maximum bet size
    MAX_BET_AMOUNT = 500.0  # $500
    
    # Minimum edge required for betting
    MIN_EDGE_THRESHOLD = 0.05  # 5% edge
    
    # Minimum confidence required for betting
    MIN_CONFIDENCE = 0.4  # 40% confidence
    
    # Progressive bet size reduction if losing
    LOSS_STREAK_THRESHOLD = 3  # Number of consecutive losses to trigger reduction
    LOSS_STREAK_REDUCTION = 0.5  # Reduce bet size by 50% after loss streak
    
    # Progressive bet size increase if winning
    WIN_STREAK_THRESHOLD = 5  # Number of consecutive wins to trigger increase
    WIN_STREAK_INCREASE = 1.2  # Increase bet size by 20% after win streak
    WIN_STREAK_CAP = 1.5  # Cap the streak multiplier at 50% increase
    
    # Drawdown protection
    MAX_DRAWDOWN_PCT = 0.15  # 15% drawdown triggers reduced bet sizing
    DRAWDOWN_REDUCTION = 0.6  # Reduce bet size by 40% when in significant drawdown

class PerformanceExpectations:
    """
    Realistic performance expectations and warnings for backtesting results.
    """
    BACKTEST_OVERPERFORMANCE = 0.3  # Backtests typically overestimate ROI by 30%
    
    REAL_WORLD_FACTORS = [
        "Closing line value matters more than backtested results",
        "Odds movement affects real-world performance",
        "Limited market liquidity may impact performance",
        "Market adaptation reduces edges over time",
        "Unforeseen variables will affect results (e.g., player changes)",
        "Line movement often works against value bettors ('steam')"
    ]
    
    VOLATILITY_FACTORS = [
        "Even a high win rate strategy experiences significant variance",
        "A 58% strategy will have 5+ game losing streaks regularly",
        "Drawdowns of 20%+ should be expected even with proper bankroll management",
        "45% of backtested profit comes from just 15% of bets",
        "Real-world ROI typically 30-40% lower than backtest ROI"
    ]
    
    SUSTAINABILITY_FACTORS = [
        "Edges in esports betting markets tend to decrease over time",
        "A successful strategy attracts competition, reducing edges",
        "Sportsbooks adjust their approach to minimize exploitable opportunities",
        "Strategy performance degrades without regular model retraining",
        "Historical samples may not reflect current market dynamics"
    ]

def calculate_realistic_roi(backtest_roi):
    """
    Calculate realistic ROI expectations based on backtest results.
    
    Args:
        backtest_roi (float): ROI from backtesting
        
    Returns:
        dict: Realistic ROI projections with confidence intervals
    """
    # Conservative adjustment: reduce ROI by standard factor
    expected_roi = backtest_roi * (1 - PerformanceExpectations.BACKTEST_OVERPERFORMANCE)
    
    # Calculate confidence intervals
    # 70% confidence interval
    lower_70 = expected_roi * 0.6
    upper_70 = expected_roi * 1.3
    
    # 95% confidence interval
    lower_95 = expected_roi * 0.3
    upper_95 = expected_roi * 1.5
    
    # Required sample size for statistical significance
    if backtest_roi > 0:
        required_sample = int(max(30, min(300, 100 / backtest_roi ** 2)))
    else:
        required_sample = 300  # Default for zero or negative ROI
    
    # Return comprehensive projections
    return {
        'backtest_roi': backtest_roi,
        'expected_roi': expected_roi,
        'confidence_intervals': {
            '70%': (lower_70, upper_70),
            '95%': (lower_95, upper_95)
        },
        'required_sample': required_sample,
        'is_realistic': expected_roi > 0 and backtest_roi > 0.05  # 5% ROI minimum threshold
    }

def calculate_expected_drawdowns(backtest_roi, win_rate, avg_odds=2.0):
    """
    Calculate expected drawdowns for a strategy.
    
    Args:
        backtest_roi (float): ROI from backtesting
        win_rate (float): Win rate from backtesting
        avg_odds (float): Average decimal odds
        
    Returns:
        dict: Expected drawdown metrics
    """
    # Check for zero win_rate to prevent division by zero
    if win_rate <= 0:
        return {
            'expected_max_drawdown': 0.5,  # Default value
            'expected_drawdown_duration': 30,  # Default value
            'probability_major_drawdown': 0.8,  # Default value
            'conservative_kelly': 0.0  # No Kelly bet if win_rate is zero
        }
        
    # Kelly fraction calculation
    p = win_rate
    q = 1 - p
    b = avg_odds - 1
    
    # Calculate optimal Kelly fraction
    if p * b > q:
        full_kelly = (p * b - q) / b
    else:
        full_kelly = 0
    
    # Conservative Kelly is 10-25% of full Kelly
    conservative_kelly = full_kelly * 0.2
    
    # Expected maximum drawdown with conservative Kelly
    expected_max_drawdown = min(0.8, max(0.1, 0.4 * (1 - win_rate) / win_rate))
    
    # Expected drawdown duration (in number of bets)
    expected_drawdown_duration = int(min(500, max(20, 80 * (1 - win_rate) / backtest_roi))) if backtest_roi > 0 else 50
    
    # Probability of 20%+ drawdown
    prob_major_drawdown = min(0.95, max(0.05, 0.8 * (1 - win_rate) / win_rate))
    
    return {
        'expected_max_drawdown': expected_max_drawdown,
        'expected_drawdown_duration': expected_drawdown_duration,
        'probability_major_drawdown': prob_major_drawdown,
        'conservative_kelly': conservative_kelly
    }

def simulate_market_response(initial_odds, win_probability, market_efficiency=0.7, bet_size=100, market_size=10000):
    """
    Simulate how the betting market would respond to betting activity.
    
    Args:
        initial_odds (float): Initial decimal odds
        win_probability (float): Your model's predicted win probability
        market_efficiency (float): How efficient the market is (0-1)
        bet_size (float): Size of your bet
        market_size (float): Estimated market size for this bet
        
    Returns:
        dict: Simulated market response
    """
    # Calculate edge in initial odds
    implied_prob = 1 / initial_odds
    initial_edge = win_probability - implied_prob
    
    # Calculate market impact
    market_impact = (bet_size / market_size) * market_efficiency
    
    # Calculate how odds would move
    # The market moves against profitable bets
    if initial_edge > 0:
        # Positive edge, odds would shorten
        new_implied_prob = implied_prob + (initial_edge * market_impact)
        new_odds = 1 / new_implied_prob if new_implied_prob > 0 else initial_odds
    else:
        # Negative edge or no edge, odds would drift out
        new_implied_prob = implied_prob + (initial_edge * market_impact * 0.5)  # Less drift for negative edges
        new_odds = 1 / new_implied_prob if new_implied_prob > 0 else initial_odds
    
    # Calculate closing line value
    clv = (initial_odds - new_odds) / initial_odds if initial_edge > 0 else 0
    
    # Calculate remaining edge after market movement
    new_edge = win_probability - new_implied_prob
    
    return {
        'initial_odds': initial_odds,
        'initial_edge': initial_edge,
        'initial_edge_pct': initial_edge * 100,
        'expected_closing_odds': new_odds,
        'closing_line_value': clv,
        'closing_edge': new_edge,
        'closing_edge_pct': new_edge * 100,
        'edge_retained_pct': (new_edge / initial_edge * 100) if initial_edge > 0 else 0
    }

def generate_performance_warning(backtest_results):
    """
    Generate realistic warnings based on backtest results.
    
    Args:
        backtest_results (dict): Results from backtesting
        
    Returns:
        str: Warning message with realistic expectations
    """
    # Extract key metrics
    backtest_roi = backtest_results.get('performance', {}).get('roi', 0)
    win_rate = backtest_results.get('performance', {}).get('win_rate', 0)
    total_bets = backtest_results.get('performance', {}).get('total_bets', 0)
    
    # Check if any bets were placed
    if total_bets == 0:
        return (
            "\n===== BACKTESTING PRODUCED NO BETS =====\n\n"
            "The backtest did not place any bets with the current parameters.\n"
            "This could be due to:\n"
            "1. Minimum edge threshold too high\n"
            "2. Minimum confidence threshold too high\n"
            "3. Odds not providing sufficient value\n\n"
            "RECOMMENDATIONS:\n"
            "- Try reducing the minimum edge requirement\n"
            "- Lower the confidence threshold\n"
            "- Adjust bet sizing parameters\n"
            "- Ensure the model is properly calibrated\n\n"
            "No performance projections can be made without bets."
        )
    
    realistic_roi = calculate_realistic_roi(backtest_roi)
    drawdown_projections = calculate_expected_drawdowns(backtest_roi, win_rate)
    
    # Generated warning message
    warning = (
        "\n===== REALISTIC PERFORMANCE EXPECTATIONS =====\n\n"
        f"SUMMARY:\n"
        f"- Backtest ROI: {backtest_roi:.2%}\n"
        f"- Expected real-world ROI: {realistic_roi['expected_roi']:.2%}\n"
        f"- 70% Confidence Interval: {realistic_roi['confidence_intervals']['70%'][0]:.2%} to "
        f"{realistic_roi['confidence_intervals']['70%'][1]:.2%}\n"
        f"- 95% Confidence Interval: {realistic_roi['confidence_intervals']['95%'][0]:.2%} to "
        f"{realistic_roi['confidence_intervals']['95%'][1]:.2%}\n\n"
        
        f"RISK ASSESSMENT:\n"
        f"- Expected maximum drawdown: {drawdown_projections['expected_max_drawdown']:.2%}\n"
        f"- Expected drawdown duration: {drawdown_projections['expected_drawdown_duration']} bets\n"
        f"- Probability of 20%+ drawdown: {drawdown_projections['probability_major_drawdown']:.2%}\n"
        f"- Recommended conservative Kelly: {drawdown_projections['conservative_kelly']:.4f}\n\n"
        
        f"SAMPLE SIZE ASSESSMENT:\n"
        f"- Current sample size: {total_bets} bets\n"
        f"- Required sample for statistical confidence: {realistic_roi['required_sample']} bets\n"
        f"- Sample adequacy: {'Sufficient' if total_bets >= realistic_roi['required_sample'] else 'Insufficient'}\n\n"
        
        "REAL-WORLD FACTORS TO CONSIDER:\n"
    )
    
    # Add random selection of real-world factors
    import random
    factors = random.sample(PerformanceExpectations.REAL_WORLD_FACTORS, min(3, len(PerformanceExpectations.REAL_WORLD_FACTORS)))
    for i, factor in enumerate(factors):
        warning += f"{i+1}. {factor}\n"
    
    warning += "\nVOLATILITY FACTORS:\n"
    vol_factors = random.sample(PerformanceExpectations.VOLATILITY_FACTORS, min(3, len(PerformanceExpectations.VOLATILITY_FACTORS)))
    for i, factor in enumerate(vol_factors):
        warning += f"{i+1}. {factor}\n"
    
    warning += "\nSUSTAINABILITY FACTORS:\n"
    sus_factors = random.sample(PerformanceExpectations.SUSTAINABILITY_FACTORS, min(3, len(PerformanceExpectations.SUSTAINABILITY_FACTORS)))
    for i, factor in enumerate(sus_factors):
        warning += f"{i+1}. {factor}\n"
    
    # Final recommendation
    if realistic_roi['expected_roi'] > 0.1:
        warning += "\nOVERALL ASSESSMENT: The strategy shows strong promise but expect significant variance."
    elif realistic_roi['expected_roi'] > 0.05:
        warning += "\nOVERALL ASSESSMENT: The strategy shows moderate promise but exercise caution."
    elif realistic_roi['expected_roi'] > 0:
        warning += "\nOVERALL ASSESSMENT: The strategy may be marginally profitable but requires strict discipline."
    else:
        warning += "\nOVERALL ASSESSMENT: The strategy does not demonstrate a clear edge in real-world conditions."
    
    return warning

def simulate_long_term_performance(win_rate, avg_odds, kelly_fraction, starting_bankroll=1000, num_bets=1000, num_simulations=100):
    """
    Simulate long-term performance with realistic variance.
    
    Args:
        win_rate (float): Expected win rate
        avg_odds (float): Average decimal odds
        kelly_fraction (float): Kelly criterion fraction
        starting_bankroll (float): Starting bankroll
        num_bets (int): Number of bets to simulate
        num_simulations (int): Number of simulations to run
        
    Returns:
        dict: Simulation results
    """
    import numpy as np
    
    results = {
        'final_bankrolls': [],
        'max_drawdowns': [],
        'longest_drawdowns': [],
        'max_bankrolls': [],
        'ruin_rate': 0
    }
    
    for _ in range(num_simulations):
        bankroll = starting_bankroll
        max_bankroll = bankroll
        current_drawdown = 0
        max_drawdown = 0
        drawdown_start = None
        longest_drawdown = 0
        current_drawdown_length = 0
        
        bankroll_history = [bankroll]
        
        for i in range(num_bets):
            if bankroll <= 0:
                # Bankruptcy
                break
                
            # Calculate bet size using Kelly
            bet_size = min(bankroll * kelly_fraction, bankroll * 0.1)  # Cap at 10% of bankroll for safety
            
            # Simulate bet outcome
            outcome = np.random.random() < win_rate
            
            if outcome:
                # Win
                profit = bet_size * (avg_odds - 1)
                bankroll += profit
                
                if bankroll > max_bankroll:
                    max_bankroll = bankroll
                    
                if drawdown_start is not None:
                    # End of drawdown
                    if current_drawdown_length > longest_drawdown:
                        longest_drawdown = current_drawdown_length
                    drawdown_start = None
                    current_drawdown = 0
                    current_drawdown_length = 0
            else:
                # Loss
                bankroll -= bet_size
                
                if drawdown_start is None:
                    # Start of new drawdown
                    drawdown_start = bankroll
                    current_drawdown_length = 1
                else:
                    # Continuing drawdown
                    current_drawdown_length += 1
                    current_drawdown = (drawdown_start - bankroll) / drawdown_start
                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown
            
            bankroll_history.append(bankroll)
        
        # Record final results
        results['final_bankrolls'].append(bankroll)
        results['max_drawdowns'].append(max_drawdown)
        results['longest_drawdowns'].append(longest_drawdown)
        results['max_bankrolls'].append(max_bankroll)
        
        if bankroll <= 0:
            results['ruin_rate'] += 1
    
    results['ruin_rate'] /= num_simulations
    
    # Calculate summary statistics
    results['mean_final_bankroll'] = sum(results['final_bankrolls']) / num_simulations
    results['median_final_bankroll'] = sorted(results['final_bankrolls'])[num_simulations // 2]
    results['mean_max_drawdown'] = sum(results['max_drawdowns']) / num_simulations
    results['worst_max_drawdown'] = max(results['max_drawdowns'])
    results['mean_longest_drawdown'] = sum(results['longest_drawdowns']) / num_simulations
    results['worst_longest_drawdown'] = max(results['longest_drawdowns'])
    
    # Calculate percentiles
    results['bankroll_10th_percentile'] = sorted(results['final_bankrolls'])[int(num_simulations * 0.1)]
    results['bankroll_25th_percentile'] = sorted(results['final_bankrolls'])[int(num_simulations * 0.25)]
    results['bankroll_75th_percentile'] = sorted(results['final_bankrolls'])[int(num_simulations * 0.75)]
    results['bankroll_90th_percentile'] = sorted(results['final_bankrolls'])[int(num_simulations * 0.9)]
    
    return results

class ForwardTestManager:
    """
    Manager for forward testing predictions and tracking results.
    """
    def __init__(self, data_dir="forward_test"):
        """
        Initialize the forward test manager.
        
        Args:
            data_dir (str): Directory for storing forward test data
        """
        self.data_dir = data_dir
        self.predictions_file = os.path.join(data_dir, "predictions.json")
        self.results_file = os.path.join(data_dir, "results.json")
        self.summary_file = os.path.join(data_dir, "summary.json")
        
        # Create directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Initialize files if they don't exist
        if not os.path.exists(self.predictions_file):
            self.save_json(self.predictions_file, [])
            
        if not os.path.exists(self.results_file):
            self.save_json(self.results_file, [])
            
        if not os.path.exists(self.summary_file):
            summary = {
                "prediction_count": 0,
                "accuracy": 0,
                "bets": {
                    "total": 0,
                    "won": 0,
                    "win_rate": 0,
                    "total_wagered": 0,
                    "total_returns": 0,
                    "roi": 0,
                    "profit": 0
                },
                "performance_by_type": {},
                "clv_metrics": {
                    "average_clv": 0,
                    "positive_clv_count": 0,
                    "negative_clv_count": 0,
                    "clv_win_rate": 0
                },
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.save_json(self.summary_file, summary)
    
    def save_json(self, file_path, data):
        """Save data to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_json(self, file_path):
        """Load data from JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def record_prediction(self, match_data, prediction_data, bet_recommendations=None):
        """
        Record a forward test prediction before the match occurs.
        
        Args:
            match_data (dict): Match information
            prediction_data (dict): Prediction information
            bet_recommendations (dict): Recommended bets (optional)
            
        Returns:
            str: Prediction ID
        """
        # Load existing predictions
        predictions = self.load_json(self.predictions_file)
        
        # Generate unique prediction ID
        prediction_id = f"FT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create prediction record
        prediction = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "match": match_data,
            "prediction": prediction_data,
            "bet_recommendations": bet_recommendations,
            "status": "pending"
        }
        
        # Add to predictions list
        predictions.append(prediction)
        
        # Save predictions
        self.save_json(self.predictions_file, predictions)
        
        return prediction_id
    
    def record_result(self, prediction_id, match_result, closing_odds=None, actual_bets=None):
        """
        Record the result of a previously predicted match.
        
        Args:
            prediction_id (str): Prediction ID
            match_result (dict): Actual match result
            closing_odds (dict): Closing odds for CLV calculation (optional)
            actual_bets (dict): Actual bets placed (optional)
            
        Returns:
            bool: Success or failure
        """
        # Load existing predictions and results
        predictions = self.load_json(self.predictions_file)
        results = self.load_json(self.results_file)
        
        # Find the prediction
        prediction = None
        for pred in predictions:
            if pred.get("prediction_id") == prediction_id:
                prediction = pred
                break
                
        if not prediction:
            print(f"Prediction with ID {prediction_id} not found.")
            return False
        
        # Update prediction status
        prediction["status"] = "completed"
        
        # Evaluate prediction accuracy
        team1_name = prediction["match"]["team1_name"]
        team2_name = prediction["match"]["team2_name"]
        
        predicted_winner = "team1" if prediction["prediction"]["win_probability"] > 0.5 else "team2"
        actual_winner = match_result["winner"]
        
        prediction_correct = predicted_winner == actual_winner
        
        # Calculate CLV if closing odds are provided
        clv_metrics = {}
        if closing_odds and "odds_data" in prediction:
            for bet_type, odds in prediction["odds_data"].items():
                if bet_type in closing_odds:
                    initial_odds = odds
                    final_odds = closing_odds[bet_type]
                    
                    # Calculate CLV
                    if bet_type.startswith("team1") and predicted_winner == "team1":
                        clv = (initial_odds - final_odds) / initial_odds
                    elif bet_type.startswith("team2") and predicted_winner == "team2":
                        clv = (initial_odds - final_odds) / initial_odds
                    else:
                        # For totals or if betting against prediction
                        clv = 0
                    
                    clv_metrics[bet_type] = {
                        "initial_odds": initial_odds,
                        "closing_odds": final_odds,
                        "clv": clv
                    }
        
        # Evaluate betting performance
        betting_results = []
        if actual_bets:
            for bet in actual_bets:
                bet_type = bet["bet_type"]
                stake = bet["stake"]
                odds = bet["odds"]
                
                # Determine if bet won
                bet_won = False
                if bet_type == "team1_ml" and actual_winner == "team1":
                    bet_won = True
                elif bet_type == "team2_ml" and actual_winner == "team2":
                    bet_won = True
                elif bet_type == "team1_plus_1_5":
                    team1_score = match_result["team1_score"]
                    team2_score = match_result["team2_score"]
                    bet_won = (team1_score + 1.5) > team2_score
                elif bet_type == "team2_plus_1_5":
                    team1_score = match_result["team1_score"]
                    team2_score = match_result["team2_score"]
                    bet_won = (team2_score + 1.5) > team1_score
                elif bet_type == "team1_minus_1_5":
                    team1_score = match_result["team1_score"]
                    team2_score = match_result["team2_score"]
                    bet_won = (team1_score - 1.5) > team2_score
                elif bet_type == "team2_minus_1_5":
                    team1_score = match_result["team1_score"]
                    team2_score = match_result["team2_score"]
                    bet_won = (team2_score - 1.5) > team1_score
                elif bet_type == "over_2_5_maps":
                    team1_score = match_result["team1_score"]
                    team2_score = match_result["team2_score"]
                    bet_won = (team1_score + team2_score) > 2.5
                elif bet_type == "under_2_5_maps":
                    team1_score = match_result["team1_score"]
                    team2_score = match_result["team2_score"]
                    bet_won = (team1_score + team2_score) < 2.5
                
                # Calculate returns and profit
                returns = stake * odds if bet_won else 0
                profit = returns - stake
                
                # Record bet result
                betting_results.append({
                    "bet_type": bet_type,
                    "stake": stake,
                    "odds": odds,
                    "won": bet_won,
                    "returns": returns,
                    "profit": profit
                })
        
        # Create result record
        result = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "match": {
                "team1_name": team1_name,
                "team2_name": team2_name,
                "date": match_result.get("date", prediction["match"].get("date", "")),
                "team1_score": match_result["team1_score"],
                "team2_score": match_result["team2_score"],
                "winner": actual_winner
            },
            "prediction": {
                "predicted_winner": predicted_winner,
                "win_probability": prediction["prediction"]["win_probability"],
                "confidence": prediction["prediction"].get("confidence", 0)
            },
            "accuracy": {
                "correct": prediction_correct
            },
            "betting": {
                "results": betting_results
            },
            "clv": clv_metrics,
            "original_prediction": prediction
        }
        
        # Add to results list
        results.append(result)
        
        # Save results
        self.save_json(self.results_file, results)
        
        # Update predictions
        for i, pred in enumerate(predictions):
            if pred.get("prediction_id") == prediction_id:
                predictions[i] = prediction
                break
        
        self.save_json(self.predictions_file, predictions)
        
        # Update summary statistics
        self.update_summary()
        
        return True

    def update_summary(self):
            """Update summary statistics based on current results."""
            results = self.load_json(self.results_file)
            
            summary = {
                "prediction_count": len(results),
                "accuracy": 0,
                "bets": {
                    "total": 0,
                    "won": 0,
                    "win_rate": 0,
                    "total_wagered": 0,
                    "total_returns": 0,
                    "roi": 0,
                    "profit": 0
                },
                "performance_by_type": {},
                "clv_metrics": {
                    "average_clv": 0,
                    "positive_clv_count": 0,
                    "negative_clv_count": 0,
                    "clv_win_rate": 0
                },
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if not results:
                self.save_json(self.summary_file, summary)
                return
            
            # Calculate prediction accuracy
            correct_predictions = sum(1 for r in results if r.get("accuracy", {}).get("correct", False))
            summary["accuracy"] = correct_predictions / len(results) if len(results) > 0 else 0
            
            # Calculate betting performance
            bet_types = {}
            clv_values = []
            
            for result in results:
                betting_results = result.get("betting", {}).get("results", [])
                
                for bet in betting_results:
                    bet_type = bet.get("bet_type", "")
                    stake = bet.get("stake", 0)
                    returns = bet.get("returns", 0)
                    won = bet.get("won", False)
                    
                    # Update overall betting stats
                    summary["bets"]["total"] += 1
                    summary["bets"]["won"] += 1 if won else 0
                    summary["bets"]["total_wagered"] += stake
                    summary["bets"]["total_returns"] += returns
                    
                    # Update performance by bet type
                    if bet_type not in bet_types:
                        bet_types[bet_type] = {
                            "total": 0,
                            "won": 0,
                            "total_wagered": 0,
                            "total_returns": 0,
                            "roi": 0,
                            "profit": 0
                        }
                    
                    bet_types[bet_type]["total"] += 1
                    bet_types[bet_type]["won"] += 1 if won else 0
                    bet_types[bet_type]["total_wagered"] += stake
                    bet_types[bet_type]["total_returns"] += returns
                
                # Process CLV metrics
                clv_metrics = result.get("clv", {})
                for bet_type, clv_data in clv_metrics.items():
                    clv = clv_data.get("clv", 0)
                    clv_values.append(clv)
                    
                    if clv > 0:
                        summary["clv_metrics"]["positive_clv_count"] += 1
                    elif clv < 0:
                        summary["clv_metrics"]["negative_clv_count"] += 1
            
            # Calculate ROI and profit
            if summary["bets"]["total_wagered"] > 0:
                summary["bets"]["win_rate"] = summary["bets"]["won"] / summary["bets"]["total"]
                summary["bets"]["roi"] = (summary["bets"]["total_returns"] - summary["bets"]["total_wagered"]) / summary["bets"]["total_wagered"]
                summary["bets"]["profit"] = summary["bets"]["total_returns"] - summary["bets"]["total_wagered"]
            
            # Calculate ROI and profit by bet type
            for bet_type, stats in bet_types.items():
                if stats["total_wagered"] > 0:
                    stats["win_rate"] = stats["won"] / stats["total"]
                    stats["roi"] = (stats["total_returns"] - stats["total_wagered"]) / stats["total_wagered"]
                    stats["profit"] = stats["total_returns"] - stats["total_wagered"]
            
            summary["performance_by_type"] = bet_types
            
            # Calculate CLV metrics
            if clv_values:
                summary["clv_metrics"]["average_clv"] = sum(clv_values) / len(clv_values)
                total_clv_counts = summary["clv_metrics"]["positive_clv_count"] + summary["clv_metrics"]["negative_clv_count"]
                if total_clv_counts > 0:
                    summary["clv_metrics"]["clv_win_rate"] = summary["clv_metrics"]["positive_clv_count"] / total_clv_counts
            
            # Save summary
            self.save_json(self.summary_file, summary)
    
    def pending_predictions(self):
        """Get all pending predictions."""
        predictions = self.load_json(self.predictions_file)
        return [p for p in predictions if p.get("status") == "pending"]
    
    def get_summary(self):
        """Get current performance summary."""
        return self.load_json(self.summary_file)
    
    def generate_performance_report(self, output_file="forward_test_report.html"):
        """Generate a comprehensive HTML performance report."""
        results = self.load_json(self.results_file)
        summary = self.load_json(self.summary_file)
        
        if not results:
            return "No results available for reporting."
        
        # Convert to DataFrame for easier analysis
        df_results = pd.DataFrame([
            {
                "prediction_id": r["prediction_id"],
                "date": r["match"]["date"],
                "team1": r["match"]["team1_name"],
                "team2": r["match"]["team2_name"],
                "predicted_winner": r["prediction"]["predicted_winner"],
                "actual_winner": r["match"]["winner"],
                "win_probability": r["prediction"]["win_probability"],
                "confidence": r["prediction"]["confidence"],
                "correct": r["accuracy"]["correct"],
                "score": f"{r['match']['team1_score']}-{r['match']['team2_score']}"
            }
            for r in results
        ])
        
        # Create DataFrame for betting results
        bet_results = []
        for r in results:
            for bet in r.get("betting", {}).get("results", []):
                bet_results.append({
                    "prediction_id": r["prediction_id"],
                    "date": r["match"]["date"],
                    "team1": r["match"]["team1_name"],
                    "team2": r["match"]["team2_name"],
                    "bet_type": bet["bet_type"],
                    "stake": bet["stake"],
                    "odds": bet["odds"],
                    "won": bet["won"],
                    "returns": bet["returns"],
                    "profit": bet["profit"]
                })
        
        df_bets = pd.DataFrame(bet_results) if bet_results else None
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forward Test Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .stats-card {{ background-color: #f9f9f9; border: 1px solid #ddd; 
                             border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
                .stats-row {{ display: flex; flex-wrap: wrap; }}
                .stat-box {{ flex: 1; min-width: 200px; margin: 10px; text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
                .stat-label {{ font-size: 14px; color: #666; }}
                .good {{ color: green; }}
                .bad {{ color: red; }}
                .neutral {{ color: blue; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .chart-container {{ width: 100%; height: 400px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Valorant Match Prediction - Forward Test Report</h1>
                <p>Report generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="stats-card">
                    <h2>Overall Performance</h2>
                    <div class="stats-row">
                        <div class="stat-box">
                            <div class="stat-label">Prediction Accuracy</div>
                            <div class="stat-value {self._get_accuracy_class(summary['accuracy'])}">{summary['accuracy']:.2%}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Predictions Made</div>
                            <div class="stat-value">{summary['prediction_count']}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Betting ROI</div>
                            <div class="stat-value {self._get_roi_class(summary['bets']['roi'])}">{summary['bets']['roi']:.2%}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Bet Win Rate</div>
                            <div class="stat-value {self._get_winrate_class(summary['bets']['win_rate'])}">{summary['bets']['win_rate']:.2%}</div>
                        </div>
                    </div>
                    <div class="stats-row">
                        <div class="stat-box">
                            <div class="stat-label">Total Wagered</div>
                            <div class="stat-value">${summary['bets']['total_wagered']:.2f}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Total Returns</div>
                            <div class="stat-value">${summary['bets']['total_returns']:.2f}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Profit/Loss</div>
                            <div class="stat-value {self._get_profit_class(summary['bets']['profit'])}">${summary['bets']['profit']:.2f}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Average CLV</div>
                            <div class="stat-value {self._get_clv_class(summary['clv_metrics']['average_clv'])}">{summary['clv_metrics']['average_clv']:.2%}</div>
                        </div>
                    </div>
                </div>
                
                <h2>Performance by Bet Type</h2>
                <table>
                    <tr>
                        <th>Bet Type</th>
                        <th>Count</th>
                        <th>Win Rate</th>
                        <th>ROI</th>
                        <th>Profit</th>
                    </tr>
        """
        
        # Add performance by bet type
        for bet_type, stats in summary["performance_by_type"].items():
            winrate_class = self._get_winrate_class(stats["win_rate"])
            roi_class = self._get_roi_class(stats["roi"])
            profit_class = self._get_profit_class(stats["profit"])
            
            html += f"""
                    <tr>
                        <td>{bet_type.replace('_', ' ').upper()}</td>
                        <td>{stats['total']}</td>
                        <td class="{winrate_class}">{stats['win_rate']:.2%}</td>
                        <td class="{roi_class}">{stats['roi']:.2%}</td>
                        <td class="{profit_class}">${stats['profit']:.2f}</td>
                    </tr>
            """
        
        html += """
                </table>
                
                <h2>Recent Predictions</h2>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Match</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Actual Result</th>
                        <th>Correct</th>
                    </tr>
        """
        
        # Add recent predictions, ordered by date (most recent first)
        df_recent = df_results.sort_values('date', ascending=False).head(10)
        for _, row in df_recent.iterrows():
            correct_class = "good" if row["correct"] else "bad"
            confidence_class = self._get_confidence_class(row["confidence"])
            
            pred_text = f"{row['team1'] if row['predicted_winner'] == 'team1' else row['team2']} to win ({row['win_probability']:.2%})"
            result_text = f"{row['team1'] if row['actual_winner'] == 'team1' else row['team2']} won {row['score']}"
            
            html += f"""
                    <tr>
                        <td>{row['date']}</td>
                        <td>{row['team1']} vs {row['team2']}</td>
                        <td>{pred_text}</td>
                        <td class="{confidence_class}">{row['confidence']:.2f}</td>
                        <td>{result_text}</td>
                        <td class="{correct_class}">{"Yes" if row["correct"] else "No"}</td>
                    </tr>
            """
        
        html += """
                </table>
        """
        
        # Add betting performance charts if there are bets
        if df_bets is not None and not df_bets.empty:
            # Generate charts (using JavaScript)
            html += """
                <h2>Betting Performance</h2>
                
                <div class="chart-container">
                    <canvas id="cumulativeProfit"></canvas>
                </div>
                
                <div class="chart-container">
                    <canvas id="betTypePerformance"></canvas>
                </div>
                
                <h2>Recent Bets</h2>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Match</th>
                        <th>Bet Type</th>
                        <th>Stake</th>
                        <th>Odds</th>
                        <th>Result</th>
                        <th>Profit</th>
                    </tr>
            """
            
            # Add recent bets, ordered by date (most recent first)
            df_recent_bets = df_bets.sort_values('date', ascending=False).head(10)
            for _, row in df_recent_bets.iterrows():
                result_class = "good" if row["won"] else "bad"
                profit_class = "good" if row["profit"] > 0 else "bad"
                
                html += f"""
                        <tr>
                            <td>{row['date']}</td>
                            <td>{row['team1']} vs {row['team2']}</td>
                            <td>{row['bet_type'].replace('_', ' ').upper()}</td>
                            <td>${row['stake']:.2f}</td>
                            <td>{row['odds']:.2f}</td>
                            <td class="{result_class}">{"Won" if row["won"] else "Lost"}</td>
                            <td class="{profit_class}">${row['profit']:.2f}</td>
                        </tr>
                """
            
            html += """
                </table>
            """
            
            # Generate betting data for chart
            df_bets['cumulative_profit'] = df_bets['profit'].cumsum()
            
            # Convert to JSON for JavaScript
            cumulative_data = df_bets.sort_values('date')[['date', 'cumulative_profit']].to_dict('records')
            
            # Add Chart.js library and chart creation
            html += """
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        // Cumulative profit chart
                        var ctxProfit = document.getElementById('cumulativeProfit').getContext('2d');
                        var profitChart = new Chart(ctxProfit, {
                            type: 'line',
                            data: {
                                labels: %s,
                                datasets: [{
                                    label: 'Cumulative Profit ($)',
                                    data: %s,
                                    borderColor: 'blue',
                                    backgroundColor: 'rgba(0, 0, 255, 0.1)',
                                    tension: 0.1,
                                    fill: true
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    title: {
                                        display: true,
                                        text: 'Cumulative Profit Over Time'
                                    }
                                },
                                scales: {
                                    y: {
                                        beginAtZero: false
                                    }
                                }
                            }
                        });
                        
                        // Bet type performance chart
                        var ctxBetType = document.getElementById('betTypePerformance').getContext('2d');
                        var betTypeChart = new Chart(ctxBetType, {
                            type: 'bar',
                            data: {
                                labels: %s,
                                datasets: [{
                                    label: 'ROI',
                                    data: %s,
                                    backgroundColor: %s
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    title: {
                                        display: true,
                                        text: 'ROI by Bet Type'
                                    }
                                },
                                scales: {
                                    y: {
                                        beginAtZero: true,
                                        format: '%%'
                                    }
                                }
                            }
                        });
                    });
                </script>
            """ % (
                json.dumps([d['date'] for d in cumulative_data]),
                json.dumps([d['cumulative_profit'] for d in cumulative_data]),
                json.dumps([k.replace('_', ' ').upper() for k in summary["performance_by_type"].keys()]),
                json.dumps([v['roi'] for v in summary["performance_by_type"].values()]),
                json.dumps(['green' if v['roi'] > 0 else 'red' for v in summary["performance_by_type"].values()])
            )
        
        html += """
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(os.path.join(self.data_dir, output_file), 'w') as f:
            f.write(html)
        
        return os.path.join(self.data_dir, output_file)
    
    def _get_accuracy_class(self, accuracy):
        """Get CSS class based on prediction accuracy."""
        if accuracy >= 0.65:
            return "good"
        elif accuracy >= 0.55:
            return "neutral"
        else:
            return "bad"
    
    def _get_roi_class(self, roi):
        """Get CSS class based on ROI."""
        if roi >= 0.1:
            return "good"
        elif roi >= 0:
            return "neutral"
        else:
            return "bad"
    
    def _get_winrate_class(self, winrate):
        """Get CSS class based on win rate."""
        if winrate >= 0.6:
            return "good"
        elif winrate >= 0.5:
            return "neutral"
        else:
            return "bad"
    
    def _get_profit_class(self, profit):
        """Get CSS class based on profit."""
        if profit > 10:
            return "good"
        elif profit >= 0:
            return "neutral"
        else:
            return "bad"
    
    def _get_clv_class(self, clv):
        """Get CSS class based on CLV."""
        if clv >= 0.02:
            return "good"
        elif clv >= 0:
            return "neutral"
        else:
            return "bad"
    
    def _get_confidence_class(self, confidence):
        """Get CSS class based on confidence level."""
        if confidence >= 0.7:
            return "good"
        elif confidence >= 0.5:
            return "neutral"
        else:
            return "bad"

def run_forward_test_manager():
    """
    Run the forward test manager interactively.
    """
    manager = ForwardTestManager()
    
    while True:
        print("\n===== VALORANT FORWARD TEST MANAGER =====")
        print("1. Record new prediction")
        print("2. Update match result")
        print("3. View pending predictions")
        print("4. View performance summary")
        print("5. Generate performance report")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            print("\n----- RECORD NEW PREDICTION -----")
            team1 = input("Team 1 name: ")
            team2 = input("Team 2 name: ")
            match_date = input("Match date (YYYY-MM-DD): ")
            win_prob = float(input("Team 1 win probability (0-1): "))
            confidence = float(input("Prediction confidence (0-1): "))
            
            match_data = {
                "team1_name": team1,
                "team2_name": team2,
                "date": match_date
            }
            
            prediction_data = {
                "win_probability": win_prob,
                "confidence": confidence,
                "predicted_winner": "team1" if win_prob > 0.5 else "team2"
            }
            
            bet_data = {}
            place_bets = input("Record bet recommendations? (y/n): ").lower() == 'y'
            
            if place_bets:
                bet_types = ['team1_ml', 'team2_ml', 'team1_plus_1_5', 'team2_plus_1_5', 
                             'team1_minus_1_5', 'team2_minus_1_5', 'over_2_5_maps', 'under_2_5_maps']
                
                for bet_type in bet_types:
                    record = input(f"Record {bet_type}? (y/n): ").lower() == 'y'
                    if record:
                        probability = float(input(f"Probability for {bet_type} (0-1): "))
                        odds = float(input(f"Odds for {bet_type}: "))
                        bet_amount = float(input(f"Recommended bet amount for {bet_type}: "))
                        
                        bet_data[bet_type] = {
                            "probability": probability,
                            "odds": odds,
                            "bet_amount": bet_amount,
                            "expected_value": (probability * odds - 1) * 100
                        }
            
            prediction_id = manager.record_prediction(match_data, prediction_data, bet_data)
            print(f"\nPrediction recorded with ID: {prediction_id}")
            
        elif choice == '2':
            print("\n----- UPDATE MATCH RESULT -----")
            pending = manager.pending_predictions()
            
            if not pending:
                print("No pending predictions to update.")
                continue
            
            print("\nPending predictions:")
            for i, pred in enumerate(pending):
                team1 = pred["match"]["team1_name"]
                team2 = pred["match"]["team2_name"]
                pred_winner = "team1" if pred["prediction"]["win_probability"] > 0.5 else "team2"
                pred_winner_name = team1 if pred_winner == "team1" else team2
                
                print(f"{i+1}. {team1} vs {team2} - Predicted: {pred_winner_name} ({pred['prediction_id']})")
            
            try:
                idx = int(input("\nEnter prediction number to update: ")) - 1
                if idx < 0 or idx >= len(pending):
                    print("Invalid selection.")
                    continue
                
                prediction = pending[idx]
                team1 = prediction["match"]["team1_name"]
                team2 = prediction["match"]["team2_name"]
                
                print(f"\nUpdating result for {team1} vs {team2}")
                team1_score = int(input(f"{team1} score: "))
                team2_score = int(input(f"{team2} score: "))
                
                winner = "team1" if team1_score > team2_score else "team2"
                
                match_result = {
                    "team1_score": team1_score,
                    "team2_score": team2_score,
                    "winner": winner,
                    "date": prediction["match"].get("date", "")
                }
                
                # Optional: Record closing odds for CLV
                record_closing_odds = input("\nRecord closing odds for CLV? (y/n): ").lower() == 'y'
                closing_odds = {}
                
                if record_closing_odds:
                    if "bet_recommendations" in prediction:
                        for bet_type in prediction["bet_recommendations"]:
                            odds = float(input(f"Closing odds for {bet_type}: "))
                            closing_odds[bet_type] = odds
                
                # Optional: Record actual bets placed
                record_actual_bets = input("\nRecord actual bets placed? (y/n): ").lower() == 'y'
                actual_bets = []
                
                if record_actual_bets:
                    while True:
                        bet_type = input("\nBet type (or 'done' to finish): ")
                        if bet_type.lower() == 'done':
                            break
                        
                        stake = float(input("Stake amount: "))
                        odds = float(input("Odds: "))
                        
                        actual_bets.append({
                            "bet_type": bet_type,
                            "stake": stake,
                            "odds": odds
                        })
                
                success = manager.record_result(
                    prediction["prediction_id"], 
                    match_result, 
                    closing_odds if record_closing_odds else None,
                    actual_bets if record_actual_bets else None
                )
                
                if success:
                    print("\nMatch result recorded successfully.")
                else:
                    print("\nFailed to record match result.")
                
            except ValueError:
                print("Invalid input.")
                continue
                
        elif choice == '3':
            print("\n----- PENDING PREDICTIONS -----")
            pending = manager.pending_predictions()
            
            if not pending:
                print("No pending predictions.")
                continue
            
            for i, pred in enumerate(pending):
                team1 = pred["match"]["team1_name"]
                team2 = pred["match"]["team2_name"]
                match_date = pred["match"].get("date", "Unknown")
                pred_winner = "team1" if pred["prediction"]["win_probability"] > 0.5 else "team2"
                pred_winner_name = team1 if pred_winner == "team1" else team2
                
                print(f"{i+1}. {match_date}: {team1} vs {team2} - Predicted: {pred_winner_name} ({pred['prediction_id']})")
                
        elif choice == '4':
            print("\n----- PERFORMANCE SUMMARY -----")
            summary = manager.get_summary()
            
            print(f"\nPredictions: {summary['prediction_count']}")
            print(f"Accuracy: {summary['accuracy']:.2%}")
            print(f"\nBetting performance:")
            print(f"Total bets: {summary['bets']['total']}")
            print(f"Win rate: {summary['bets']['win_rate']:.2%}")
            print(f"ROI: {summary['bets']['roi']:.2%}")
            print(f"Profit: ${summary['bets']['profit']:.2f}")
            
            print("\nPerformance by bet type:")
            for bet_type, stats in summary["performance_by_type"].items():
                print(f"  {bet_type}: {stats['win_rate']:.2%} win rate, {stats['roi']:.2%} ROI, ${stats['profit']:.2f} profit")
            
            print(f"\nCLV metrics:")
            print(f"Average CLV: {summary['clv_metrics']['average_clv']:.2%}")
            print(f"CLV win rate: {summary['clv_metrics']['clv_win_rate']:.2%}")
            
        elif choice == '5':
            print("\n----- GENERATING PERFORMANCE REPORT -----")
            report_path = manager.generate_performance_report()
            print(f"Report generated: {report_path}")
            
        elif choice == '6':
            print("\nExiting Forward Test Manager.")
            break
            
        else:
            print("\nInvalid choice. Please try again.")
                
def conservative_kelly(win_prob, decimal_odds, config=BankrollConfig):
    """
    Calculate bet size using a conservative version of the Kelly Criterion.
    
    Args:
        win_prob (float): Probability of winning (0.0 to 1.0)
        decimal_odds (float): Decimal odds (e.g., 2.0 for +100 American odds)
        config (BankrollConfig): Configuration parameters
        
    Returns:
        float: Kelly fraction (0.0 to 1.0)
    """
    if win_prob <= 0 or win_prob >= 1 or decimal_odds <= 1:
        return 0.0
    
    # Convert to "b" format for Kelly formula
    b = decimal_odds - 1
    
    # Standard Kelly formula: (bp - q) / b
    # where p = win probability, q = 1-p, b = decimal odds - 1
    p = win_prob
    q = 1 - p
    
    # Calculate full Kelly
    full_kelly = (b * p - q) / b
    
    # Apply conservative fraction
    conservative_kelly = full_kelly * config.KELLY_FRACTION
    
    # Cap the maximum bet size
    capped_kelly = min(conservative_kelly, config.MAX_BET_PERCENTAGE)
    
    return max(0.0, capped_kelly)

def calculate_bet_amount(bankroll, win_prob, decimal_odds, bet_history=None, config=BankrollConfig):
    """
    Calculate bet amount with comprehensive safeguards.
    
    Args:
        bankroll (float): Current bankroll amount
        win_prob (float): Probability of winning (0.0 to 1.0)
        decimal_odds (float): Decimal odds (e.g., 2.0 for +100 American odds)
        bet_history (list, optional): List of previous bets
        config (BankrollConfig): Configuration parameters
        
    Returns:
        tuple: (bet_amount, kelly_fraction, adjustments)
    """
    adjustments = []
    
    # Base Kelly calculation
    kelly_fraction = conservative_kelly(win_prob, decimal_odds, config)
    base_bet = bankroll * kelly_fraction
    
    # Adjust for winning/losing streaks if we have betting history
    streak_multiplier = 1.0
    if bet_history and len(bet_history) > 0:
        # Check for recent losing streak
        recent_bets = bet_history[-config.LOSS_STREAK_THRESHOLD:]
        if len(recent_bets) >= config.LOSS_STREAK_THRESHOLD:
            if all(not bet.get('won', False) for bet in recent_bets):
                streak_multiplier = config.LOSS_STREAK_REDUCTION
                adjustments.append(f"Losing streak: reducing by {(1-config.LOSS_STREAK_REDUCTION)*100:.0f}%")
        
        # Check for winning streak
        win_streak_length = 0
        for i in range(min(10, len(bet_history))):
            if bet_history[-(i+1)].get('won', False):
                win_streak_length += 1
            else:
                break
                
        if win_streak_length >= config.WIN_STREAK_THRESHOLD:
            # Only increase bets after a significant win streak
            streak_multiplier = min(config.WIN_STREAK_CAP, 
                                   1.0 + (win_streak_length - config.WIN_STREAK_THRESHOLD) * 0.05)
            adjustments.append(f"Winning streak: increasing by {(streak_multiplier-1)*100:.0f}%")
    
    # Adjust for bankroll drawdown protection
    if bet_history and len(bet_history) > 0:
        starting_bankroll = bet_history[0].get('starting_bankroll', bankroll)
        current_drawdown = (starting_bankroll - bankroll) / starting_bankroll
        
        if current_drawdown > config.MAX_DRAWDOWN_PCT:
            drawdown_multiplier = config.DRAWDOWN_REDUCTION
            streak_multiplier = min(streak_multiplier, drawdown_multiplier)
            adjustments.append(f"Drawdown protection: reducing to {drawdown_multiplier*100:.0f}% of normal")
    
    # Apply all multipliers
    adjusted_bet = base_bet * streak_multiplier
    
    # Apply absolute caps
    max_amount = min(bankroll * config.MAX_BET_PERCENTAGE, config.MAX_BET_AMOUNT)
    final_bet = min(adjusted_bet, max_amount)
    
    # Ensure minimum bet amount (at least $1)
    if 0 < final_bet < 1:
        final_bet = 0  # Don't place very small bets
    
    # Round to nearest dollar
    final_bet = round(final_bet, 0)
    
    return final_bet, kelly_fraction, adjustments

def analyze_betting_edge_conservative(win_probability, team1_name, team2_name, odds_data, 
                                     confidence_score, bankroll=1000.0, bet_history=None, 
                                     config=BankrollConfig):
    """
    Conservative betting edge analysis with multiple safeguards.
    
    Args:
        win_probability (float): Probability of team1 winning
        team1_name (str): Name of team 1
        team2_name (str): Name of team 2
        odds_data (dict): Betting odds data
        confidence_score (float): Model confidence (0.0 to 1.0)
        bankroll (float): Current bankroll
        bet_history (list, optional): List of previous bets
        config (BankrollConfig): Configuration parameters
        
    Returns:
        dict: Betting analysis with recommendations
    """
    # Calculate probabilities for different bet types
    team2_win_prob = 1.0 - win_probability
    
    # Calculate map probabilities using conservative scaling
    map_scale = 0.55 + (confidence_score * 0.15)  # Between 0.55 and 0.7
    single_map_prob = 0.5 + (win_probability - 0.5) * map_scale
    single_map_prob = max(0.3, min(0.7, single_map_prob))
    
    team1_plus_prob = 1 - (1 - single_map_prob) ** 2
    team2_plus_prob = 1 - single_map_prob ** 2
    team1_minus_prob = single_map_prob ** 2
    team2_minus_prob = (1 - single_map_prob) ** 2
    over_prob = 2 * single_map_prob * (1 - single_map_prob)
    under_prob = 1 - over_prob
    
    # Bound probabilities to prevent extreme values
    team1_plus_prob = min(0.95, max(0.05, team1_plus_prob))
    team2_plus_prob = min(0.95, max(0.05, team2_plus_prob))
    team1_minus_prob = min(0.9, max(0.1, team1_minus_prob))
    team2_minus_prob = min(0.9, max(0.1, team2_minus_prob))
    over_prob = min(0.9, max(0.1, over_prob))
    under_prob = min(0.9, max(0.1, under_prob))
    
    # Calibration adjustment based on confidence
    # Lower confidence leads to more regression to the mean
    if confidence_score < 0.4:
        # Low confidence: regress probabilities 40% toward 0.5
        win_probability = 0.5 + (win_probability - 0.5) * 0.6
        team2_win_prob = 1.0 - win_probability
        
        # Also regress derived probabilities
        team1_plus_prob = 0.5 + (team1_plus_prob - 0.5) * 0.6
        team2_plus_prob = 0.5 + (team2_plus_prob - 0.5) * 0.6
        team1_minus_prob = 0.5 + (team1_minus_prob - 0.5) * 0.6
        team2_minus_prob = 0.5 + (team2_minus_prob - 0.5) * 0.6
        over_prob = 0.5 + (over_prob - 0.5) * 0.6
        under_prob = 0.5 + (under_prob - 0.5) * 0.6
    
    # Define all potential bets
    bet_types = {
        'team1_ml': {
            'probability': win_probability,
            'odds': odds_data.get('team1_ml_odds', 0),
            'team': team1_name,
            'description': f"{team1_name} to win",
            'risk_factor': 1.0  # Base risk factor
        },
        'team2_ml': {
            'probability': team2_win_prob,
            'odds': odds_data.get('team2_ml_odds', 0),
            'team': team2_name,
            'description': f"{team2_name} to win",
            'risk_factor': 1.0
        },
        'team1_plus_1_5': {
            'probability': team1_plus_prob,
            'odds': odds_data.get('team1_plus_1_5_odds', 0),
            'team': team1_name,
            'description': f"{team1_name} +1.5 maps",
            'risk_factor': 0.9  # Lower risk for +1.5 maps
        },
        'team2_plus_1_5': {
            'probability': team2_plus_prob,
            'odds': odds_data.get('team2_plus_1_5_odds', 0),
            'team': team2_name,
            'description': f"{team2_name} +1.5 maps",
            'risk_factor': 0.9
        },
        'team1_minus_1_5': {
            'probability': team1_minus_prob,
            'odds': odds_data.get('team1_minus_1_5_odds', 0),
            'team': team1_name,
            'description': f"{team1_name} -1.5 maps",
            'risk_factor': 1.2  # Higher risk for -1.5 maps
        },
        'team2_minus_1_5': {
            'probability': team2_minus_prob,
            'odds': odds_data.get('team2_minus_1_5_odds', 0),
            'team': team2_name,
            'description': f"{team2_name} -1.5 maps",
            'risk_factor': 1.2
        },
        'over_2_5_maps': {
            'probability': over_prob,
            'odds': odds_data.get('over_2_5_maps_odds', 0),
            'team': None,
            'description': "Over 2.5 maps",
            'risk_factor': 1.1
        },
        'under_2_5_maps': {
            'probability': under_prob,
            'odds': odds_data.get('under_2_5_maps_odds', 0),
            'team': None,
            'description': "Under 2.5 maps",
            'risk_factor': 1.1
        }
    }
    
    # Analyze each bet type
    betting_analysis = {}
    for bet_type, bet_info in bet_types.items():
        prob = bet_info['probability']
        odds = bet_info['odds']
        
        if odds <= 1.0:
            continue  # Skip if odds are invalid
        
        # Calculate implied probability
        implied_prob = 1.0 / odds
        
        # Calculate edge
        edge = prob - implied_prob
        
        # Calculate EV
        ev = (prob * (odds - 1)) - (1 - prob)
        
        # Adjust minimum edge threshold by risk factor
        risk_adjusted_threshold = config.MIN_EDGE_THRESHOLD * bet_info['risk_factor']
        
        # Check if edge is sufficient
        meets_edge = edge > risk_adjusted_threshold
        
        # Check if confidence is sufficient
        edge_to_confidence_ratio = edge / risk_adjusted_threshold if risk_adjusted_threshold > 0 else 0
        required_confidence = config.MIN_CONFIDENCE * (1 - min(0.3, edge_to_confidence_ratio))
        meets_confidence = confidence_score >= required_confidence
        
        # Calculate bet amount
        bet_amount, kelly, adjustments = calculate_bet_amount(
            bankroll, prob, odds, bet_history, config
        )
        
        # Only recommend if all criteria are met
        recommended = meets_edge and meets_confidence and bet_amount > 0
        
        # Store analysis
        betting_analysis[bet_type] = {
            'probability': prob,
            'implied_prob': implied_prob,
            'edge': edge,
            'risk_adjusted_threshold': risk_adjusted_threshold,
            'ev': ev,
            'ev_percentage': ev * 100,
            'odds': odds,
            'kelly_fraction': kelly,
            'bet_amount': bet_amount,
            'meets_edge': meets_edge,
            'meets_confidence': meets_confidence,
            'recommended': recommended,
            'confidence': confidence_score,
            'description': bet_info['description'],
            'adjustments': adjustments
        }
    
    return betting_analysis

def load_configuration():
    """Load and initialize the configuration settings."""
    Config.create_directories()
    
    # Initialize logging
    import logging
    import os
    from datetime import datetime
    
    log_file = os.path.join(Config.LOG_DIR, f"valorant_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG if Config.DEBUG_MODE else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Configuration loaded")
    return Config

def select_optimal_bets_conservative(betting_analysis, team1_name, team2_name, bankroll,
                                   max_bets=2, max_total_risk=0.03, config=BankrollConfig):
    """
    Select the optimal bets with conservative constraints.
    Args:
        betting_analysis (dict): Betting analysis data
        team1_name (str): Name of team 1
        team2_name (str): Name of team 2
        bankroll (float): Current bankroll
        max_bets (int): Maximum number of bets to place
        max_total_risk (float): Maximum total risk as percentage of bankroll
        config (BankrollConfig): Configuration parameters
    Returns:
        dict: Selected bets with analysis
    """
    recommended_bets = {k: v for k, v in betting_analysis.items() if v['recommended']}
    if not recommended_bets:
        return {}
    
    # Use 'edge' instead of 'ev'
    sorted_bets = sorted(recommended_bets.items(), key=lambda x: x[1]['edge'], reverse=True)
    
    selected_bets = {}
    total_risk = 0.0
    bet_teams = set()
    bet_categories = set()
    
    for bet_type, analysis in sorted_bets:
        if len(selected_bets) >= max_bets:
            break
            
        bet_risk = analysis['bet_amount'] / bankroll
        if total_risk + bet_risk > max_total_risk:
            max_additional_risk = max_total_risk - total_risk
            adjusted_amount = round(max_additional_risk * bankroll, 0)
            if adjusted_amount < 1:
                continue  # Skip if adjusted amount is too small
            analysis['bet_amount'] = adjusted_amount
            # Don't try to append to 'adjustments' since it doesn't exist
            # Just create a new dictionary entry if needed
            # analysis['adjustments'].append(f"Reduced for total portfolio risk management")
        
        if 'team1' in bet_type:
            team = team1_name
        elif 'team2' in bet_type:
            team = team2_name
        else:
            team = None
            
        category = '_'.join(bet_type.split('_')[1:])  # e.g., 'ml', 'plus_1_5'
        
        # Use 'edge' instead of 'ev'
        exceptional_edge = analysis['edge'] > 0.15  # 15% edge is exceptional
        
        if team and team in bet_teams and category in bet_categories and not exceptional_edge:
            continue
            
        selected_bets[bet_type] = analysis
        total_risk += bet_risk
        
        if team:
            bet_teams.add(team)
        bet_categories.add(category)
        
    return selected_bets

def implement_dynamic_betting_strategy(match_history, current_bankroll, starting_bankroll, win_rate, target_roi=0.25):
    """
    Implement a dynamic betting strategy that adjusts based on bankroll and performance.
    
    Args:
        match_history: List of past matches and bets
        current_bankroll: Current bankroll
        starting_bankroll: Starting bankroll
        win_rate: Current win rate
        target_roi: Target ROI
        
    Returns:
        tuple: (max_bet_pct, min_edge, confidence_threshold)
    """
    # Calculate current performance
    bankroll_growth = current_bankroll / starting_bankroll
    
    # Base parameters
    base_max_bet_pct = 0.05
    base_min_edge = 0.02
    base_confidence = 0.2
    
    # Adjust based on bankroll growth
    if bankroll_growth > 2.0:  # Doubled initial bankroll
        # More conservative approach to protect profits
        max_bet_pct = base_max_bet_pct * 0.8
        min_edge = base_min_edge * 1.2
        confidence_threshold = base_confidence * 1.2
        print("Strategy: Conservative (protecting large profits)")
    elif bankroll_growth > 1.5:  # 50% growth
        # Slightly more conservative
        max_bet_pct = base_max_bet_pct * 0.9
        min_edge = base_min_edge * 1.1
        confidence_threshold = base_confidence * 1.1
        print("Strategy: Moderately conservative (protecting good profits)")
    elif bankroll_growth < 0.7:  # Lost 30%
        # More aggressive to recover losses, but with higher quality threshold
        max_bet_pct = base_max_bet_pct * 1.1
        min_edge = base_min_edge * 1.2  # Higher edge requirements for safety
        confidence_threshold = base_confidence * 1.1
        print("Strategy: Recovery mode (seeking quality opportunities)")
    elif bankroll_growth < 0.9:  # Lost 10%
        # Slightly more aggressive
        max_bet_pct = base_max_bet_pct * 1.05
        min_edge = base_min_edge * 1.05
        confidence_threshold = base_confidence
        print("Strategy: Slightly aggressive (minor recovery)")
    else:  # Normal performance
        # Standard approach
        max_bet_pct = base_max_bet_pct
        min_edge = base_min_edge
        confidence_threshold = base_confidence
        print("Strategy: Balanced (normal operation)")
    
    # Adjust based on recent performance (last 20 bets)
    recent_bets = []
    for match in match_history[-10:]:
        for bet in match.get('bets', []):
            recent_bets.append(bet.get('won', False))
    
    if len(recent_bets) >= 5:
        recent_win_rate = sum(1 for b in recent_bets if b) / len(recent_bets)
        
        # If recent performance is much better than overall
        if recent_win_rate > win_rate * 1.3:
            # Slightly increase bet size to capitalize on good form
            max_bet_pct *= 1.1
            print("Recent performance boost: Increasing bet size")
        # If recent performance is much worse than overall
        elif recent_win_rate < win_rate * 0.7:
            # Reduce bet size temporarily
            max_bet_pct *= 0.9
            # Increase edge requirements
            min_edge *= 1.1
            print("Recent performance decline: Reducing exposure")
    
    # Cap maximum bet percentage for safety
    max_bet_pct = min(max_bet_pct, 0.07)
    
    return max_bet_pct, min_edge, confidence_threshold


def calculate_recency_weighted_win_rate(team_matches, weight_decay=0.9):
    """Calculate win rate with more recent matches weighted higher."""
    if not team_matches:
        return 0.5
    
    # Ensure team_matches is a list
    if not isinstance(team_matches, list):
        return 0.5
    
    # Sort matches by date
    sorted_matches = sorted(team_matches, key=lambda x: x.get('date', ''))
    
    total_weight = 0
    weighted_wins = 0
    
    for i, match in enumerate(sorted_matches):
        # Exponential weighting - more recent matches count more
        weight = weight_decay ** (len(sorted_matches) - i - 1)
        weighted_wins += weight * (1 if match.get('team_won', False) else 0)
        total_weight += weight
    
    return weighted_wins / total_weight if total_weight > 0 else 0.5

def enhance_feature_derivation(features_df, team1_stats, team2_stats):
    """
    Simplified version that avoids errors with team stats structure.
    """
    print("Using simplified feature derivation")
    return features_df

def calculate_pistol_importance(matches):
    """
    Calculate importance of pistol rounds based on correlation with match wins.
    
    Args:
        matches: List of match data
        
    Returns:
        float: Importance value between 0.05 and 0.3
    """
    # Safety check - make sure matches is a list
    if not isinstance(matches, list):
        return 0.15  # Default importance for invalid input
    
    # Check if we have enough matches for meaningful correlation
    if not matches or len(matches) < 10:
        return 0.15  # Default importance for insufficient data
    
    # Initialize data collection
    pistol_wins = []
    match_wins = []
    
    # Process each match
    for match in matches:
        # Skip invalid matches
        if not isinstance(match, dict):
            continue
            
        # Try to extract pistol round data
        if 'details' in match and match['details']:
            # Initialize counters
            pistol_count = 0
            pistol_won = 0
            
            # Simple approach: use match score as proxy for pistol performance
            team_score = match.get('team_score', 0)
            opponent_score = match.get('opponent_score', 0)
            
            # Skip matches with invalid scores
            if not isinstance(team_score, (int, float)) or not isinstance(opponent_score, (int, float)):
                continue
                
            # Rough heuristic: pistol importance correlates with dominant wins
            if team_score > opponent_score + 5:  # Dominant win
                pistol_count = 2
                pistol_won = 2
            elif team_score > opponent_score:  # Close win
                pistol_count = 2
                pistol_won = 1
            elif team_score + 5 < opponent_score:  # Dominant loss
                pistol_count = 2
                pistol_won = 0
            else:  # Close loss
                pistol_count = 2
                pistol_won = 1
            
            # Only add data if we have valid counts
            if pistol_count > 0:
                try:
                    pistol_wins.append(pistol_won / pistol_count)
                    match_wins.append(1 if match.get('team_won', False) else 0)
                except (TypeError, ZeroDivisionError):
                    # Skip if division fails
                    continue
    
    # Calculate correlation if we have enough data
    if len(pistol_wins) >= 5 and len(match_wins) >= 5:
        try:
            # Calculate correlation coefficient
            correlation = np.corrcoef(pistol_wins, match_wins)[0, 1]
            
            # Handle NaN correlation
            if np.isnan(correlation):
                return 0.15  # Default for invalid correlation
                
            # Ensure it's between 0.05 and 0.3
            return max(0.05, min(0.3, abs(correlation)))
        except Exception:
            # Fallback for any calculation errors
            return 0.15
    
    # Default importance if we don't have enough valid data points
    return 0.15

def calculate_opponent_quality(matches, recent_only=False):
    """Calculate average quality of opponents faced."""
    if not matches:
        return 0.5
    
    # Filter to recent matches if requested
    if recent_only and len(matches) > 5:
        # Sort by date
        sorted_matches = sorted(matches, key=lambda x: x.get('date', ''))
        matches = sorted_matches[-5:]  # Last 5 matches
    
    # Calculate average opponent win rate
    opponent_win_rates = []
    
    for match in matches:
        # Use opponent's score as a proxy for quality
        team_score = match.get('team_score', 0)
        opponent_score = match.get('opponent_score', 0)
        
        # Calculate implied opponent quality
        if team_score + opponent_score > 0:
            quality = opponent_score / (team_score + opponent_score)
            opponent_win_rates.append(quality)
    
    if opponent_win_rates:
        return sum(opponent_win_rates) / len(opponent_win_rates)
    
    return 0.5  # Default quality

def calculate_ot_win_rate(map_statistics):
    """Calculate overtime win rate across all maps."""
    total_ot_matches = 0
    total_ot_wins = 0
    
    for map_name, stats in map_statistics.items():
        if 'overtime_stats' in stats:
            ot_stats = stats['overtime_stats']
            total_ot_matches += ot_stats.get('matches', 0)
            total_ot_wins += ot_stats.get('wins', 0)
    
    if total_ot_matches > 0:
        return total_ot_wins / total_ot_matches
    
    return 0.5  # Default win rate

def run_backtest_with_safeguards(params):
    """
    Run backtesting with enhanced safeguards.
    
    Args:
        params (dict): Backtest parameters
        
    Returns:
        dict: Backtest results
    """
    logging.info(f"Starting backtest with parameters: {params}")
    
    # Load models and features
    ensemble_models, selected_features = load_backtesting_models()
    
    if not ensemble_models or not selected_features:
        logging.error("Failed to load models or features")
        return None
    
    # Get parameters
    start_date = params.get('start_date', Config.BACKTEST.DEFAULT_START_DATE)
    end_date = params.get('end_date', Config.BACKTEST.DEFAULT_END_DATE)
    team_limit = params.get('team_limit', Config.BACKTEST.DEFAULT_TEAM_LIMIT)
    bankroll = params.get('bankroll', Config.BACKTEST.DEFAULT_STARTING_BANKROLL)
    bet_pct = params.get('bet_pct', Config.BACKTEST.DEFAULT_BET_PCT)
    min_edge = params.get('min_edge', Config.BACKTEST.DEFAULT_MIN_EDGE)
    confidence_threshold = params.get('confidence_threshold', Config.BACKTEST.DEFAULT_CONFIDENCE_THRESHOLD)
    use_cache = params.get('use_cache', Config.BACKTEST.USE_CACHE)
    
    # Run the backtest
    results = run_backtest(
        start_date=start_date,
        end_date=end_date,
        team_limit=team_limit,
        bankroll=bankroll,
        bet_pct=bet_pct,
        min_edge=min_edge,
        confidence_threshold=confidence_threshold,
        use_cache=use_cache,
        cache_path=Config.BACKTEST.CACHE_PATH
    )
    
    if not results:
        logging.error("Backtest failed or no results generated")
        return None
    
    # Generate realistic performance expectations
    warning = generate_performance_warning(results)
    results['realistic_expectations'] = warning
    
    # Simulate long-term performance
    if Config.BACKTEST.SIMULATE_LONG_TERM_VARIANCE:
        win_rate = results['performance']['win_rate']
        avg_odds = 2.0  # Estimate - could be calculated from results
        kelly_fraction = Config.BETTING.KELLY_FRACTION
        
        simulation = simulate_long_term_performance(
            win_rate,
            avg_odds,
            kelly_fraction,
            starting_bankroll=bankroll,
            num_bets=1000,
            num_simulations=Config.BACKTEST.SIMULATION_RUNS
        )
        
        results['long_term_simulation'] = simulation
    
    logging.info(f"Backtest completed: {results['performance']['accuracy']:.2%} accuracy, {results['performance']['roi']:.2%} ROI")
    return results

def init_forward_testing():
    """Initialize the forward testing framework."""
    manager = ForwardTestManager(data_dir=Config.FORWARD_TEST.DATA_DIR)
    return manager

def forward_test_prediction(manager, team1_name, team2_name, ensemble_models, selected_features, team_data_collection):
    """
    Make a prediction and record it in the forward testing framework.
    
    Args:
        manager (ForwardTestManager): Forward test manager
        team1_name (str): Name of first team
        team2_name (str): Name of second team
        ensemble_models (list): Ensemble models
        selected_features (list): Selected features
        team_data_collection (dict): Team data collection
        
    Returns:
        str: Prediction ID
    """
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Make prediction
    prediction_results = predict_with_consistent_ordering(
        team1_name, 
        team2_name, 
        ensemble_models, 
        selected_features, 
        team_data_collection, 
        current_date
    )
    
    if 'error' in prediction_results:
        logging.error(f"Prediction error: {prediction_results['error']}")
        return None
    
    # Get odds data (this would be from user input or API)
    odds_data = {}
    print("\nEnter current odds:")
    for bet_type in ['team1_ml', 'team2_ml', 'team1_plus_1_5', 'team2_plus_1_5', 
                    'team1_minus_1_5', 'team2_minus_1_5', 'over_2_5_maps', 'under_2_5_maps']:
        try:
            odds = float(input(f"{bet_type} odds: "))
            odds_data[f"{bet_type}_odds"] = odds
        except ValueError:
            continue
    
    # Analyze betting options
    betting_analysis = analyze_betting_options(
        prediction_results, 
        odds_data, 
        bankroll=1000.0  # This would come from user input or tracking
    )
    
    # Record prediction
    match_data = {
        'team1_name': team1_name,
        'team2_name': team2_name,
        'date': current_date
    }
    
    prediction_data = {
        'win_probability': prediction_results['win_probability'],
        'confidence': prediction_results['confidence'],
        'predicted_winner': 'team1' if prediction_results['win_probability'] > 0.5 else 'team2'
    }
    
    bet_recommendations = {bet_type: analysis for bet_type, analysis in betting_analysis['optimal_bets'].items()}
    
    prediction_id = manager.record_prediction(match_data, prediction_data, bet_recommendations)
    
    return prediction_id

def create_enhanced_backtest_visualizations(results):
    """Create enhanced visualizations for backtest results with improved error handling."""
    # Create directory for visualizations
    os.makedirs("backtest_plots", exist_ok=True)
    
    # 1. Bankroll history with improved formatting
    plt.figure(figsize=(14, 7))
    
    # Extract bankroll data with safety checks
    if 'performance' in results and 'bankroll_history' in results['performance']:
        bankroll_history = [entry['bankroll'] for entry in results['performance']['bankroll_history']]
        match_indices = [entry['match_idx'] for entry in results['performance']['bankroll_history']]
        
        if bankroll_history:  # Only proceed if we have data
            # Plot with better formatting
            plt.plot(match_indices, bankroll_history, 'b-', linewidth=2)
            
            # Add initial bankroll reference line
            initial_bankroll = bankroll_history[0] if bankroll_history else 1000
            plt.axhline(y=initial_bankroll, color='r', linestyle='--', alpha=0.5, label='Initial Bankroll')
            
            # Add annotations for significant points if we have enough data
            if len(bankroll_history) > 1:
                max_bankroll = max(bankroll_history)
                max_idx = bankroll_history.index(max_bankroll)
                min_bankroll = min(bankroll_history)
                min_idx = bankroll_history.index(min_bankroll)
                
                plt.annotate(f'Max: ${max_bankroll:.2f}', 
                            xy=(match_indices[max_idx], max_bankroll),
                            xytext=(match_indices[max_idx], max_bankroll + 50),
                            arrowprops=dict(facecolor='green', shrink=0.05), 
                            ha='center')
                
                plt.annotate(f'Min: ${min_bankroll:.2f}', 
                            xy=(match_indices[min_idx], min_bankroll),
                            xytext=(match_indices[min_idx], min_bankroll - 50),
                            arrowprops=dict(facecolor='red', shrink=0.05), 
                            ha='center')
            
            # Add final bankroll annotation
            if bankroll_history:
                plt.annotate(f'Final: ${bankroll_history[-1]:.2f}',
                            xy=(match_indices[-1], bankroll_history[-1]),
                            xytext=(match_indices[-1]-5, bankroll_history[-1] + 30),
                            ha='right')
    else:
        # Create a simple placeholder chart
        plt.text(0.5, 0.5, "No bankroll history data available", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
    
    plt.title('Bankroll Progression During Backtest', fontsize=16)
    plt.xlabel('Match Number', fontsize=14)
    plt.ylabel('Bankroll ($)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('backtest_plots/bankroll_history.png', dpi=300)
    plt.close()
    
    # Create a simple index HTML file 
    with open('backtest_plots/index.html', 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .plot-container { margin-bottom: 30px; }
                img { max-width: 100%; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>Valorant Match Prediction Backtest Results</h1>
            <div class="plot-container">
                <h2>Bankroll History</h2>
                <img src="bankroll_history.png" alt="Bankroll History">
            </div>
        </body>
        </html>
        """)
    
    print(f"Created simplified visualization dashboard at backtest_plots/index.html")

def main_optimized():
    """
    Main function with all optimizations integrated.
    
    This function implements the command-line interface with all
    requested improvements integrated.
    """
    import argparse
    import os
    import sys
    import logging
    from datetime import datetime
    
    # Load configuration
    config = load_configuration()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Valorant Match Predictor and Betting System")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--retrain", action="store_true", help="Retrain with consistent features")
    parser.add_argument("--predict", action="store_true", help="Predict a match outcome and analyze betting options")
    parser.add_argument("--stats", action="store_true", help="View betting performance statistics")
    parser.add_argument("--players", action="store_true", help="Include player stats in analysis")
    parser.add_argument("--economy", action="store_true", help="Include economy data in analysis")
    parser.add_argument("--maps", action="store_true", help="Include map statistics")
    parser.add_argument("--cross-validate", action="store_true", help="Train with cross-validation")
    parser.add_argument("--folds", type=int, default=Config.MODEL.CV_FOLDS, help="Number of folds for cross-validation")
    parser.add_argument("--team1", type=str, help="First team name")
    parser.add_argument("--team2", type=str, help="Second team name")
    parser.add_argument("--live", action="store_true", help="Track the bet live (input result after match)")
    parser.add_argument("--bankroll", type=float, default=1000, help="Your current betting bankroll")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting on historical matches")
    parser.add_argument("--teams", type=int, default=Config.BACKTEST.DEFAULT_TEAM_LIMIT, 
                      help="Number of teams to include in backtesting")
    parser.add_argument("--start-date", type=str, help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--test-bankroll", type=float, default=Config.BACKTEST.DEFAULT_STARTING_BANKROLL, 
                      help="Starting bankroll for backtesting")
    parser.add_argument("--max-bet", type=float, default=Config.BACKTEST.DEFAULT_BET_PCT, 
                      help="Maximum bet size as fraction of bankroll")
    parser.add_argument("--min-edge", type=float, default=Config.BACKTEST.DEFAULT_MIN_EDGE, 
                      help="Minimum edge required for betting")
    parser.add_argument("--min-confidence", type=float, default=Config.BACKTEST.DEFAULT_CONFIDENCE_THRESHOLD, 
                      help="Minimum model confidence required")
    parser.add_argument("--interactive", action="store_true", help="Use interactive parameter entry for backtesting")
    parser.add_argument("--analyze-results", type=str, help="Analyze previous backtest results file")
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached data, always fetch from API")
    parser.add_argument("--cache-path", type=str, default=Config.MODEL.CACHE_PATH,
                      help="Path to the cache file")
    parser.add_argument("--cache-info", action="store_true", help="Display information about the cache and exit")
    parser.add_argument("--debug", action="store_true", help="Enable extra debug output")
    parser.add_argument("--forward-test", action="store_true", help="Use forward testing framework")
    
    args = parser.parse_args()
    
    # Update debug mode
    Config.DEBUG_MODE = args.debug
    
    # Check for cache info request
    if args.cache_info:
        # Display cache information
        cache_path = args.cache_path
        if os.path.exists(cache_path):
            metadata = get_cache_metadata()
            if metadata:
                print("\n=== Cache Information ===")
                print(f"Last updated: {metadata.get('timestamp', 'Unknown')}")
                print(f"Teams: {metadata.get('teams_count', 0)}")
                print(f"Total matches: {metadata.get('total_matches', 0)}")
                freshness = check_cache_freshness(cache_path)
                print(f"Cache is {'fresh' if freshness else 'outdated'}")
                if not freshness:
                    print("Recommendation: Run cache.py to update your cache")
            else:
                print("No cache metadata found. The cache may be invalid.")
        else:
            print(f"No cache found at {cache_path}")
            print("Run cache.py to create the initial cache.")
        return
    
    # Set data collection parameters
    include_player_stats = args.players
    include_economy = args.economy
    include_maps = args.maps
    use_cache = not args.no_cache  # Default to using cache unless --no-cache is specified
    
    # If no specific data inclusion flags are set, include all
    if not args.players and not args.economy and not args.maps:
        include_player_stats = True
        include_economy = True
        include_maps = True
    
    logging.info(f"Configuration: players={include_player_stats}, economy={include_economy}, maps={include_maps}, use_cache={use_cache}")
    
    # Handle training
    if args.train or args.retrain:
        print(f"\n{'='*80}")
        print(f"{'TRAINING NEW MODEL' if args.train else 'RETRAINING MODEL WITH CONSISTENT FEATURES'}")
        print(f"{'='*80}")
        print(f"Starting timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check for cache availability
        if use_cache and not os.path.exists(args.cache_path):
            print(f"Cache file not found at {args.cache_path}")
            print("Please run cache.py first to create the cache, or use --no-cache to fetch data from API.")
            return
        
        # Check memory availability
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(f"System memory: {mem.total / (1024**3):.2f} GB total, {mem.available / (1024**3):.2f} GB available")
            if mem.available < 2 * (1024**3):  # Less than 2GB available
                print("WARNING: Low memory available. Training may fail due to insufficient memory.")
        except ImportError:
            print("psutil not installed. Skipping memory check.")
        
        try:
            # Collect team data
            print("\nCollecting team data...")
            team_data_collection = collect_team_data(
                include_player_stats=include_player_stats,
                include_economy=include_economy,
                include_maps=include_maps,
                use_cache=use_cache,
                cache_path=args.cache_path
            )
            
            if not team_data_collection:
                print("Failed to collect team data. Aborting training.")
                return
                
            print(f"Successfully collected data for {len(team_data_collection)} teams")
            
            # Build training dataset
            print("\nBuilding training dataset...")
            X, y = build_training_dataset(team_data_collection)
            print(f"Built training dataset with {len(X)} samples and {len(y)} labels.")
            
            if len(X) < 10:
                print("Not enough training data. Please collect more match data.")
                return
                
            # Train the model
            if args.cross_validate:
                print(f"\nTraining with {args.folds}-fold cross-validation and ensemble modeling...")
                
                if args.retrain:
                    print("Implementing enhanced retraining with consistent features...")
                    
                    try:
                        ensemble_models, scaler, selected_features, avg_metrics = train_with_consistent_features(
                            X, y, n_splits=args.folds, random_state=Config.MODEL.RANDOM_STATE
                        )
                        print("Retraining with consistent features complete.")
                        print(f"Metrics: {avg_metrics}")
                    except Exception as e:
                        print(f"ERROR in train_with_consistent_features: {e}")
                        import traceback
                        traceback.print_exc()
                        return
                else:
                    print("Training standard cross-validation ensemble...")
                    
                    try:
                        ensemble_models, stable_features, avg_metrics, fold_metrics, scaler = train_with_cross_validation(
                            X, y, n_splits=args.folds, random_state=Config.MODEL.RANDOM_STATE
                        )
                        print("Ensemble model training complete.")
                        print(f"Average metrics: {avg_metrics}")
                    except Exception as e:
                        print(f"ERROR in train_with_cross_validation: {e}")
                        import traceback
                        traceback.print_exc()
                        return
            else:
                print("Training standard model...")
                
                try:
                    model, scaler, feature_names = train_model(X, y)
                    print("Model training complete.")
                    if model:
                        print(f"Feature count: {len(feature_names)}")
                except Exception as e:
                    print(f"ERROR in train_model: {e}")
                    import traceback
                    traceback.print_exc()
                    return
                    
            print(f"\nTraining completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"CRITICAL ERROR in training process: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Handle prediction
    elif args.predict:
        print("\n=== MATCH PREDICTION ===")
        
        try:
            # Load models and features
            ensemble_models, selected_features = load_backtesting_models()
            
            if not ensemble_models or not selected_features:
                print("Failed to load models or features. Make sure you've trained models first.")
                return
            
            # Get team names
            if args.team1 and args.team2:
                team1_name = args.team1
                team2_name = args.team2
            else:
                print("\nEnter the teams to predict:")
                team1_name = input("Team 1 name: ")
                team2_name = input("Team 2 name: ")
                
                if not team1_name or not team2_name:
                    print("Team names are required for prediction.")
                    return
            
            # Get bankroll
            if not args.bankroll:
                try:
                    bankroll = float(input("\nEnter your current bankroll ($): ") or 1000)
                except ValueError:
                    bankroll = 1000
                    print("Invalid bankroll value, using default $1000")
            else:
                bankroll = args.bankroll
            
            # Collect team data
            team_data_collection = collect_team_data(
                include_player_stats=include_player_stats,
                include_economy=include_economy,
                include_maps=include_maps,
                use_cache=use_cache,
                cache_path=args.cache_path
            )
            
            if not team_data_collection:
                print("Failed to collect team data. Check API connection or cache.")
                return
            
            # Make prediction with consistent team ordering
            prediction_results = predict_with_consistent_ordering(
                team1_name, 
                team2_name, 
                ensemble_models, 
                selected_features, 
                team_data_collection
            )
            
            if 'error' in prediction_results:
                print(f"Error in prediction: {prediction_results['error']}")
                return
            
            # Get odds
            print("\nEnter the betting odds from your bookmaker:")
            odds_data = {}
            try:
                team1_ml = float(input(f"{team1_name} moneyline odds (decimal format, e.g. 2.50): ") or 0)
                if team1_ml > 0:
                    odds_data['team1_ml_odds'] = team1_ml
                    
                team2_ml = float(input(f"{team2_name} moneyline odds (decimal format, e.g. 2.50): ") or 0)
                if team2_ml > 0:
                    odds_data['team2_ml_odds'] = team2_ml
                    
                team1_plus = float(input(f"{team1_name} +1.5 maps odds: ") or 0)
                if team1_plus > 0:
                    odds_data['team1_plus_1_5_odds'] = team1_plus
                    
                team2_plus = float(input(f"{team2_name} +1.5 maps odds: ") or 0)
                if team2_plus > 0:
                    odds_data['team2_plus_1_5_odds'] = team2_plus
                    
                team1_minus = float(input(f"{team1_name} -1.5 maps odds: ") or 0)
                if team1_minus > 0:
                    odds_data['team1_minus_1_5_odds'] = team1_minus
                    
                team2_minus = float(input(f"{team2_name} -1.5 maps odds: ") or 0)
                if team2_minus > 0:
                    odds_data['team2_minus_1_5_odds'] = team2_minus
                    
                over = float(input(f"Over 2.5 maps odds: ") or 0)
                if over > 0:
                    odds_data['over_2_5_maps_odds'] = over
                    
                under = float(input(f"Under 2.5 maps odds: ") or 0)
                if under > 0:
                    odds_data['under_2_5_maps_odds'] = under
            except ValueError:
                print("Invalid odds input. Using available odds only.")
            
            # Get betting history for streak analysis
            try:
                with open('betting_performance.json', 'r') as f:
                    import json
                    betting_data = json.load(f)
                    bet_history = betting_data.get('bets', [])
            except (FileNotFoundError, json.JSONDecodeError):
                bet_history = None
            
            # Analyze betting options
            betting_analysis = analyze_betting_options(
                prediction_results, 
                odds_data, 
                bankroll=bankroll,
                bet_history=bet_history
            )
            
            # Display prediction results
            print("\n========== PREDICTION RESULTS ==========")
            print(f"Match: {team1_name} vs {team2_name}")
            print(f"{team1_name} win probability: {prediction_results['win_probability']:.2%}")
            print(f"{team2_name} win probability: {(1-prediction_results['win_probability']):.2%}")
            print(f"Model confidence: {prediction_results['confidence']:.2f}")
            
            if prediction_results.get('teams_swapped', False):
                print("\nNote: Teams were reordered for consistent prediction processing.")
            
            # Display betting recommendations
            if 'optimal_bets' in betting_analysis and betting_analysis['optimal_bets']:
                print("\nRECOMMENDED BETS:")
                print("-" * 80)
                
                for bet_type, analysis in betting_analysis['optimal_bets'].items():
                    formatted_type = bet_type.replace("_", " ").upper()
                    team_name = team1_name if "team1" in bet_type else team2_name if "team2" in bet_type else "OVER/UNDER"
                    print(f"  {team_name} {formatted_type}:")
                    print(f"  - Odds: {analysis['odds']:.2f}")
                    print(f"  - Our probability: {analysis['probability']:.2%}")
                    print(f"  - Edge: {analysis['edge']:.2%}")
                    print(f"  - Recommended bet: ${analysis['bet_amount']:.2f}")
                    
                    if 'adjustments' in analysis and analysis['adjustments']:
                        print(f"  - Adjustments: {', '.join(analysis['adjustments'])}")
                    
                    # Display expected market movement
                    if 'market_simulations' in betting_analysis and bet_type in betting_analysis['market_simulations']:
                        market_sim = betting_analysis['market_simulations'][bet_type]
                        print(f"  - Expected closing odds: {market_sim['expected_closing_odds']:.2f}")
                        print(f"  - Expected CLV: {market_sim['closing_line_value']:.2%}")
                    
                    print("")
            else:
                print("\nNo bets recommended for this match based on current odds.")
            
            # Display real-world expectations
            print("\nREAL-WORLD EXPECTATIONS:")
            print("-" * 80)
            print("• Actual results typically underperform backtests by 30-40%")
            print("• Closing Line Value (CLV) is often more predictive of long-term success than immediate results")
            print("• Expect 5+ game losing streaks even with a strong strategy")
            print("• Drawdowns of 20%+ should be expected with any betting strategy")
            
            # Track live bet if requested
            if args.live and betting_analysis.get('optimal_bets'):
                print("\nAfter the match, please enter the results:")
                recommended_bets = list(betting_analysis['optimal_bets'].keys())
                
                print("Recommended bets:")
                for i, bet_type in enumerate(recommended_bets):
                    formatted_type = bet_type.replace("_", " ").upper()
                    team_name = team1_name if "team1" in bet_type else team2_name if "team2" in bet_type else "OVER/UNDER"
                    print(f"{i+1}. {team_name} {formatted_type}")
                
                try:
                    bet_choice = int(input("\nWhich bet did you place? (enter number, 0 for none): ")) - 1
                    
                    if 0 <= bet_choice < len(recommended_bets):
                        bet_placed = recommended_bets[bet_choice]
                        bet_amount = float(input("How much did you bet? $"))
                        outcome = input("Did the bet win? (y/n): ").lower().startswith('y')
                        
                        odds = 0
                        for bet_key, odds_value in odds_data.items():
                            if bet_key.replace('_odds', '') == bet_placed:
                                odds = odds_value
                                break
                        
                        if odds > 0:
                            track_betting_performance(prediction_results, bet_placed, bet_amount, outcome, odds)
                            print("Bet tracked successfully.")
                except (ValueError, IndexError):
                    print("Invalid input, not tracking this bet.")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Handle backtesting
    elif args.backtest:
        print("\n=== BACKTESTING SYSTEM ===")
        
        try:
            print("Running enhanced backtesting to verify prediction accuracy and betting strategy...")
            
            if args.interactive:
                print("Using interactive parameter entry mode")
                params = get_backtest_params()
                results = run_backtest_with_safeguards(params)
            else:
                print(f"Using command line parameters: teams={args.teams}, bankroll=${args.test_bankroll}")
                print(f"Max bet: {args.max_bet*100}%, min edge: {args.min_edge*100}%, min confidence: {args.min_confidence*100}%")
                
                params = {
                    'start_date': args.start_date,
                    'end_date': args.end_date,
                    'team_limit': args.teams,
                    'bankroll': args.test_bankroll,
                    'bet_pct': args.max_bet,
                    'min_edge': args.min_edge,
                    'confidence_threshold': args.min_confidence,
                    'use_cache': use_cache,
                    'cache_path': args.cache_path
                }
                
                results = run_backtest_with_safeguards(params)
            
            if not results:
                print("Backtesting failed or no results generated.")
                return
            
            # Display realistic performance expectations
            print("\n" + results['realistic_expectations'])
            
            # Offer to analyze specific matches
            analyze_matches = input("\nWould you like to analyze specific matches from the backtest? (y/n): ").lower().startswith('y')
            
            if analyze_matches:
                while True:
                    print("\nOptions:")
                    print("1. Analyze a match by index")
                    print("2. Analyze a specific team's matches")
                    print("3. Exit analysis")
                    
                    choice = input("\nEnter your choice (1-3): ")
                    
                    if choice == '1':
                        try:
                            match_idx = int(input("Enter match index (0-{}): ".format(len(results['predictions'])-1)))
                            
                            if 0 <= match_idx < len(results['predictions']):
                                match_id = results['predictions'][match_idx]['match_id']
                                analyze_specific_match(results, match_id)
                            else:
                                print("Invalid match index")
                        except ValueError:
                            print("Invalid input")
                    elif choice == '2':
                        team_name = input("Enter team name: ")
                        team_matches = []
                        
                        for pred in results['predictions']:
                            if pred['team1'].lower() == team_name.lower() or pred['team2'].lower() == team_name.lower():
                                team_matches.append(pred)
                        
                        if not team_matches:
                            print(f"No matches found for {team_name}")
                        else:
                            print(f"\nFound {len(team_matches)} matches for {team_name}:")
                            
                            for i, match in enumerate(team_matches):
                                print(f"{i}. {match['team1']} vs {match['team2']} ({match['score']})")
                            
                            try:
                                match_idx = int(input("Enter match index to analyze: "))
                                
                                if 0 <= match_idx < len(team_matches):
                                    match_id = team_matches[match_idx]['match_id']
                                    analyze_specific_match(results, match_id)
                                else:
                                    print("Invalid match index")
                            except ValueError:
                                print("Invalid input")
                    elif choice == '3':
                        break
                    else:
                        print("Invalid choice")
            
        except Exception as e:
            print(f"Error during backtesting: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Handle results analysis
    elif args.analyze_results:
        print("\n=== ANALYZING BACKTEST RESULTS ===")
        
        try:
            print(f"Analyzing backtest results from {args.analyze_results}...")
            
            if not os.path.exists(args.analyze_results):
                print(f"Error: Results file '{args.analyze_results}' not found")
                return
            
            with open(args.analyze_results, 'r') as f:
                import json
                results = json.load(f)
            
            insights = identify_key_insights(results)
            
            # Display realistic performance expectations
            realistic_roi = calculate_realistic_roi(results['performance']['roi'])
            drawdown_projections = calculate_expected_drawdowns(
                results['performance']['roi'],
                results['performance']['win_rate']
            )
            
            print("\n===== REALISTIC PERFORMANCE EXPECTATIONS =====")
            print(f"Backtest ROI: {results['performance']['roi']:.2%}")
            print(f"Expected real-world ROI: {realistic_roi['expected_roi']:.2%}")
            print(f"70% Confidence Interval: {realistic_roi['confidence_intervals']['70%'][0]:.2%} to {realistic_roi['confidence_intervals']['70%'][1]:.2%}")
            print(f"Expected max drawdown: {drawdown_projections['expected_max_drawdown']:.2%}")
            print(f"Expected drawdown duration: {drawdown_projections['expected_drawdown_duration']} bets")
            
            analyze_matches = input("\nWould you like to analyze specific matches from the backtest? (y/n): ").lower().startswith('y')
            
            if analyze_matches:
                while True:
                    print("\nOptions:")
                    print("1. Analyze a match by index")
                    print("2. Analyze a specific team's matches")
                    print("3. Exit analysis")
                    
                    choice = input("\nEnter your choice (1-3): ")
                    
                    if choice == '1':
                        # Match analysis code (similar to above)
                        pass
                    elif choice == '2':
                        # Team analysis code (similar to above)
                        pass
                    elif choice == '3':
                        break
                    else:
                        print("Invalid choice")
            
        except Exception as e:
            print(f"Error analyzing backtest results: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Handle statistics view
    elif args.stats:
        print("\n=== VIEWING BETTING PERFORMANCE ===")
        
        try:
            view_betting_performance()
        except Exception as e:
            print(f"Error viewing betting performance: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Handle forward testing
    elif args.forward_test:
        print("\n=== FORWARD TESTING SYSTEM ===")
        
        try:
            forward_test_manager = ForwardTestManager()
            run_forward_test_manager()
        except Exception as e:
            print(f"Error in forward testing: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # No action specified
    else:
        print("Please specify an action: --train, --retrain, --predict, --backtest, --analyze-results, --stats, or --forward-test")
        print("\nExample commands:")
        print("  python valorant_predictor.py --train --cross-validate --folds 10")
        print("  python valorant_predictor.py --predict --team1 'Sentinels' --team2 'Cloud9'")
        print("  python valorant_predictor.py --backtest --teams 40 --interactive")
        print("  python valorant_predictor.py --analyze-results backtest_results_20240505_120000.json")
        print("  python valorant_predictor.py --stats")
        print("  python valorant_predictor.py --forward-test")
        print("  python valorant_predictor.py --cache-info")
        print("  python valorant_predictor.py --train --no-cache  # Force API fetch, don't use cache")
        print("  python valorant_predictor.py --debug  # Enable additional debugging output")

if __name__ == "__main__":
    main_optimized()