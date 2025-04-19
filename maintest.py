import requests
import json
import os
import pandas as pd
import numpy as np
import pickle
import time
import re
from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Advanced ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV, train_test_split, learning_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
API_URL = "http://localhost:5000/api/v1"
MODEL_DIR = "models"
PREDICTIONS_DIR = "predictions"
LOGS_DIR = "logs"
DATA_DIR = "data"
BACKTEST_DIR = "backtests"

# Ensure directories exist
for directory in [MODEL_DIR, PREDICTIONS_DIR, LOGS_DIR, DATA_DIR, BACKTEST_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'valorant_predictor.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ValorantPredictor")

class ValorantAPI:
    """Class to handle all API interactions"""
    
    def __init__(self, base_url=API_URL):
        self.base_url = base_url
        logger.info(f"Initialized API with base URL: {base_url}")
    
    def _make_request(self, endpoint, params=None):
        """Make an API request with error handling and retries"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/{endpoint}"
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limiting
                    retry_delay = int(response.headers.get('Retry-After', retry_delay * 2))
                    logger.warning(f"Rate limited, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"API error: {response.status_code} for endpoint {endpoint}")
                    time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Request failed: {e}")
                time.sleep(retry_delay)
            
            retry_delay *= 2
        
        logger.error(f"Failed to get data from {endpoint} after {max_retries} attempts")
        return None
    
    def get_team_id(self, team_name, region=None):
        """Search for a team ID by name, optionally filtering by region."""
        logger.info(f"Searching for team ID for '{team_name}' in region {region or 'all'}...")
        
        params = {"limit": 300}
        if region:
            params["region"] = region
        
        teams_data = self._make_request("teams", params)
        
        if not teams_data or 'data' not in teams_data:
            logger.error("No valid team data found in the response")
            return None
        
        # Try exact match first
        for team in teams_data['data']:
            if team['name'].lower() == team_name.lower():
                logger.info(f"Found exact match: {team['name']} (ID: {team['id']})")
                return team['id']
        
        # If no exact match, try partial match
        for team in teams_data['data']:
            if team_name.lower() in team['name'].lower() or team['name'].lower() in team_name.lower():
                logger.info(f"Found partial match: {team['name']} (ID: {team['id']})")
                return team['id']
        
        # If still no match and no region specified, try all regions
        if not region:
            logger.info(f"No match found. Searching across all regions...")
            for r in ['na', 'eu', 'br', 'ap', 'kr', 'ch', 'jp', 'lan', 'las', 'oce', 'mn', 'gc']:
                logger.info(f"Trying region: {r}")
                team_id = self.get_team_id(team_name, r)
                if team_id:
                    return team_id
        
        logger.warning(f"No team ID found for '{team_name}'")
        return None
    
    def get_team_details(self, team_id):
        """Fetch detailed information about a team."""
        if not team_id:
            return None
        
        logger.info(f"Fetching details for team ID: {team_id}")
        return self._make_request(f"teams/{team_id}")
    
    def get_team_match_history(self, team_id):
        """Fetch match history for a specific team."""
        if not team_id:
            return None
        
        logger.info(f"Fetching match history for team ID: {team_id}")
        return self._make_request(f"match-history/{team_id}")
    
    def get_team_player_stats(self, team_id):
        """Fetch player statistics for a team."""
        team_details = self.get_team_details(team_id)
        
        if not team_details or 'players' not in team_details:
            return []
        
        player_stats = []
        
        for player in team_details['players']:
            player_id = player.get('id')
            if not player_id:
                continue
                
            logger.info(f"Fetching stats for player: {player.get('name', 'Unknown')} (ID: {player_id})")
            player_data = self._make_request(f"players/{player_id}")
            
            if player_data:
                player_stats.append(player_data)
            
            # Be nice to the API
            time.sleep(0.5)
        
        return player_stats
    
    def get_match_details(self, match_id):
        """Fetch detailed information about a specific match including KD and other metrics."""
        if not match_id:
            return None
        
        logger.info(f"Fetching details for match ID: {match_id}")
        return self._make_request(f"match-details/{match_id}")
    
    def get_all_teams(self, limit=300):
        """Fetch information about all teams."""
        logger.info(f"Fetching list of all teams (limit: {limit})...")
        teams_data = self._make_request("teams", {"limit": limit})
        
        if not teams_data or 'data' not in teams_data:
            logger.error("No valid team data found in the response")
            return []
        
        return teams_data['data']


class MatchDataProcessor:
    """Class to process raw match data into structured format"""
    
    def __init__(self, api):
        self.api = api
    
    def parse_match_data(self, match_history, team_name):
        """Parse match history data for a team."""
        if not match_history or 'data' not in match_history:
            return []
        
        matches = []
        
        logger.info(f"Parsing {len(match_history['data'])} matches for {team_name}")
        
        for match in match_history['data']:
            try:
                # Extract basic match info
                match_info = {
                    'match_id': match.get('id', ''),
                    'date': match.get('date', ''),
                    'event': match.get('event', '') if isinstance(match.get('event', ''), str) else match.get('event', {}).get('name', ''),
                    'tournament': match.get('tournament', ''),
                    'map': match.get('map', ''),
                    'map_score': match.get('map_score', '')
                }
                
                # Extract teams and determine which is our team
                if 'teams' in match and len(match['teams']) >= 2:
                    team1 = match['teams'][0]
                    team2 = match['teams'][1]
                    
                    # Convert scores to integers for comparison
                    team1_score = int(team1.get('score', 0))
                    team2_score = int(team2.get('score', 0))
                    
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
                    
                    # Add opponent's info
                    match_info['opponent_name'] = opponent_team.get('name', '')
                    match_info['opponent_score'] = int(opponent_team.get('score', 0))
                    match_info['opponent_won'] = not team_won
                    match_info['opponent_country'] = opponent_team.get('country', '')
                    
                    # Fetch additional match details with KD data if available
                    match_details = self.api.get_match_details(match_info['match_id'])
                    if match_details:
                        self._add_player_stats_to_match(match_info, match_details, is_team1)
                    
                    matches.append(match_info)
                
            except Exception as e:
                logger.error(f"Error parsing match: {e}")
                continue
        
        # Summarize wins/losses
        wins = sum(1 for match in matches if match['team_won'])
        logger.info(f"Processed {len(matches)} matches for {team_name}: {wins} wins, {len(matches) - wins} losses")
        
        return matches
    
    def _add_player_stats_to_match(self, match_info, match_details, is_team1):
        """Add player statistics from match details to the match info dict."""
        try:
            if not match_details or 'players' not in match_details:
                return
            
            # Initialize team stats
            team_kd = 0
            team_player_count = 0
            opponent_kd = 0
            opponent_player_count = 0
            
            team_acs = 0
            opponent_acs = 0
            team_adr = 0
            opponent_adr = 0
            team_fb = 0
            opponent_fb = 0
            
            for player in match_details['players']:
                team_idx = 0 if is_team1 else 1
                player_team_idx = 0 if player.get('team_idx', 0) == 0 else 1
                
                # Determine if player is on our team or opponent team
                is_our_team = player_team_idx == team_idx
                
                # Add player stats to appropriate team
                if is_our_team:
                    team_kd += player.get('kd', 0)
                    team_acs += player.get('acs', 0)
                    team_adr += player.get('adr', 0)
                    team_fb += player.get('first_bloods', 0)
                    team_player_count += 1
                else:
                    opponent_kd += player.get('kd', 0)
                    opponent_acs += player.get('acs', 0)
                    opponent_adr += player.get('adr', 0)
                    opponent_fb += player.get('first_bloods', 0)
                    opponent_player_count += 1
            
            # Calculate team averages
            if team_player_count > 0:
                match_info['team_avg_kd'] = team_kd / team_player_count
                match_info['team_avg_acs'] = team_acs / team_player_count
                match_info['team_avg_adr'] = team_adr / team_player_count
                match_info['team_avg_fb'] = team_fb / team_player_count
            
            if opponent_player_count > 0:
                match_info['opponent_avg_kd'] = opponent_kd / opponent_player_count
                match_info['opponent_avg_acs'] = opponent_acs / opponent_player_count
                match_info['opponent_avg_adr'] = opponent_adr / opponent_player_count
                match_info['opponent_avg_fb'] = opponent_fb / opponent_player_count
            
            # Add KD differential
            if team_player_count > 0 and opponent_player_count > 0:
                match_info['kd_differential'] = match_info.get('team_avg_kd', 0) - match_info.get('opponent_avg_kd', 0)
                match_info['acs_differential'] = match_info.get('team_avg_acs', 0) - match_info.get('opponent_avg_acs', 0)
                match_info['adr_differential'] = match_info.get('team_avg_adr', 0) - match_info.get('opponent_avg_adr', 0)
                match_info['fb_differential'] = match_info.get('team_avg_fb', 0) - match_info.get('opponent_avg_fb', 0)
        
        except Exception as e:
            logger.error(f"Error adding player stats to match: {e}")
    
    def calculate_team_stats(self, matches):
        """Calculate comprehensive statistics for a team from its matches."""
        if not matches:
            return {}
        
        # Basic stats
        total_matches = len(matches)
        wins = sum(1 for match in matches if match.get('team_won', False))
        
        losses = total_matches - wins
        win_rate = wins / total_matches if total_matches > 0 else 0
        
        # Scoring stats
        total_score = sum(match.get('team_score', 0) for match in matches)
        total_opponent_score = sum(match.get('opponent_score', 0) for match in matches)
        avg_score = total_score / total_matches if total_matches > 0 else 0
        avg_opponent_score = total_opponent_score / total_matches if total_matches > 0 else 0
        score_differential = avg_score - avg_opponent_score
        
        # Performance stats from match details
        kd_stats = [match.get('team_avg_kd', 0) for match in matches if 'team_avg_kd' in match]
        acs_stats = [match.get('team_avg_acs', 0) for match in matches if 'team_avg_acs' in match]
        adr_stats = [match.get('team_avg_adr', 0) for match in matches if 'team_avg_adr' in match]
        fb_stats = [match.get('team_avg_fb', 0) for match in matches if 'team_avg_fb' in match]
        
        avg_kd = np.mean(kd_stats) if kd_stats else 0
        avg_acs = np.mean(acs_stats) if acs_stats else 0
        avg_adr = np.mean(adr_stats) if adr_stats else 0
        avg_fb = np.mean(fb_stats) if fb_stats else 0
        
        # Opponent-specific stats
        opponent_stats = {}
        for match in matches:
            opponent = match['opponent_name']
            if opponent not in opponent_stats:
                opponent_stats[opponent] = {
                    'matches': 0,
                    'wins': 0,
                    'total_score': 0,
                    'total_opponent_score': 0,
                    'kd_values': [],
                    'acs_values': [],
                    'adr_values': [],
                    'fb_values': []
                }
            
            opponent_stats[opponent]['matches'] += 1
            opponent_stats[opponent]['wins'] += 1 if match['team_won'] else 0
            opponent_stats[opponent]['total_score'] += match['team_score']
            opponent_stats[opponent]['total_opponent_score'] += match['opponent_score']
            
            # Add KD and other metrics if available
            if 'team_avg_kd' in match:
                opponent_stats[opponent]['kd_values'].append(match['team_avg_kd'])
            if 'team_avg_acs' in match:
                opponent_stats[opponent]['acs_values'].append(match['team_avg_acs'])
            if 'team_avg_adr' in match:
                opponent_stats[opponent]['adr_values'].append(match['team_avg_adr'])
            if 'team_avg_fb' in match:
                opponent_stats[opponent]['fb_values'].append(match['team_avg_fb'])
        
        # Calculate win rates and average scores for each opponent
        for opponent, stats in opponent_stats.items():
            stats['win_rate'] = stats['wins'] / stats['matches'] if stats['matches'] > 0 else 0
            stats['avg_score'] = stats['total_score'] / stats['matches'] if stats['matches'] > 0 else 0
            stats['avg_opponent_score'] = stats['total_opponent_score'] / stats['matches'] if stats['matches'] > 0 else 0
            stats['score_differential'] = stats['avg_score'] - stats['avg_opponent_score']
            
            # Add average KD and other metrics
            stats['avg_kd'] = np.mean(stats['kd_values']) if stats['kd_values'] else 0
            stats['avg_acs'] = np.mean(stats['acs_values']) if stats['acs_values'] else 0
            stats['avg_adr'] = np.mean(stats['adr_values']) if stats['adr_values'] else 0
            stats['avg_fb'] = np.mean(stats['fb_values']) if stats['fb_values'] else 0
        
        # Map stats
        map_stats = {}
        for match in matches:
            map_name = match.get('map', 'Unknown')
            if map_name == '' or map_name is None:
                map_name = 'Unknown'
            
            if map_name not in map_stats:
                map_stats[map_name] = {
                    'played': 0,
                    'wins': 0,
                    'kd_values': [],
                    'acs_values': [],
                    'adr_values': [],
                    'fb_values': []
                }
            
            map_stats[map_name]['played'] += 1
            map_stats[map_name]['wins'] += 1 if match['team_won'] else 0
            
            # Add KD and other metrics if available
            if 'team_avg_kd' in match:
                map_stats[map_name]['kd_values'].append(match['team_avg_kd'])
            if 'team_avg_acs' in match:
                map_stats[map_name]['acs_values'].append(match['team_avg_acs'])
            if 'team_avg_adr' in match:
                map_stats[map_name]['adr_values'].append(match['team_avg_adr'])
            if 'team_avg_fb' in match:
                map_stats[map_name]['fb_values'].append(match['team_avg_fb'])
        
        # Calculate win rates for each map
        for map_name, stats in map_stats.items():
            stats['win_rate'] = stats['wins'] / stats['played'] if stats['played'] > 0 else 0
            stats['avg_kd'] = np.mean(stats['kd_values']) if stats['kd_values'] else 0
            stats['avg_acs'] = np.mean(stats['acs_values']) if stats['acs_values'] else 0
            stats['avg_adr'] = np.mean(stats['adr_values']) if stats['adr_values'] else 0
            stats['avg_fb'] = np.mean(stats['fb_values']) if stats['fb_values'] else 0
        
        # Recent form (last 5 matches)
        sorted_matches = sorted(matches, key=lambda x: x.get('date', ''))
        recent_matches = sorted_matches[-5:] if len(sorted_matches) >= 5 else sorted_matches
        recent_form = sum(1 for match in recent_matches if match['team_won']) / len(recent_matches) if recent_matches else 0
        
        # KD and other metrics trends
        recent_kd = [match.get('team_avg_kd', 0) for match in recent_matches if 'team_avg_kd' in match]
        recent_acs = [match.get('team_avg_acs', 0) for match in recent_matches if 'team_avg_acs' in match]
        recent_adr = [match.get('team_avg_adr', 0) for match in recent_matches if 'team_avg_adr' in match]
        recent_fb = [match.get('team_avg_fb', 0) for match in recent_matches if 'team_avg_fb' in match]
        
        recent_avg_kd = np.mean(recent_kd) if recent_kd else 0
        recent_avg_acs = np.mean(recent_acs) if recent_acs else 0
        recent_avg_adr = np.mean(recent_adr) if recent_adr else 0
        recent_avg_fb = np.mean(recent_fb) if recent_fb else 0
        
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
            'avg_kd': avg_kd,
            'avg_acs': avg_acs,
            'avg_adr': avg_adr,
            'avg_fb': avg_fb,
            'recent_form': recent_form,
            'recent_avg_kd': recent_avg_kd,
            'recent_avg_acs': recent_avg_acs,
            'recent_avg_adr': recent_avg_adr,
            'recent_avg_fb': recent_avg_fb,
            'opponent_stats': opponent_stats,
            'map_stats': map_stats
        }
        
        return team_stats
    
    def extract_tournament_performance(self, team_matches):
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
                    'kd_values': [],
                    'acs_values': [],
                    'adr_values': [],
                    'fb_values': [],
                    'matches': []
                }
                
            tournament_performance[tournament_key]['played'] += 1
            tournament_performance[tournament_key]['wins'] += 1 if match['team_won'] else 0
            tournament_performance[tournament_key]['total_score'] += match.get('team_score', 0)
            tournament_performance[tournament_key]['total_opponent_score'] += match.get('opponent_score', 0)
            tournament_performance[tournament_key]['matches'].append(match)
            
            # Add KD and other metrics if available
            if 'team_avg_kd' in match:
                tournament_performance[tournament_key]['kd_values'].append(match['team_avg_kd'])
            if 'team_avg_acs' in match:
                tournament_performance[tournament_key]['acs_values'].append(match['team_avg_acs'])
            if 'team_avg_adr' in match:
                tournament_performance[tournament_key]['adr_values'].append(match['team_avg_adr'])
            if 'team_avg_fb' in match:
                tournament_performance[tournament_key]['fb_values'].append(match['team_avg_fb'])
        
        # Calculate derived metrics and tournament importance
        for tournament_key, stats in tournament_performance.items():
            if stats['played'] > 0:
                stats['win_rate'] = stats['wins'] / stats['played']
                stats['avg_score'] = stats['total_score'] / stats['played']
                stats['avg_opponent_score'] = stats['total_opponent_score'] / stats['played']
                stats['score_differential'] = stats['avg_score'] - stats['avg_opponent_score']
                stats['avg_kd'] = np.mean(stats['kd_values']) if stats['kd_values'] else 0
                stats['avg_acs'] = np.mean(stats['acs_values']) if stats['acs_values'] else 0
                stats['avg_adr'] = np.mean(stats['adr_values']) if stats['adr_values'] else 0
                stats['avg_fb'] = np.mean(stats['fb_values']) if stats['fb_values'] else 0
                
                # Determine tournament tier (simplified - you could create a more nuanced system)
                event_name = tournament_key.split(':')[0].lower()
                if any(major in event_name for major in ['masters', 'champions', 'last chance']):
                    stats['tier'] = 3  # Top tier
                elif any(medium in event_name for medium in ['challenger', 'regional', 'national']):
                    stats['tier'] = 2  # Mid tier
                else:
                    stats['tier'] = 1  # Lower tier
        
        return tournament_performance
    
    def analyze_performance_trends(self, team_matches, window_sizes=[5, 10, 20]):
        """Analyze team performance trends over different time windows."""
        if not team_matches:
            return {}
            
        # Sort matches by date
        sorted_matches = sorted(team_matches, key=lambda x: x.get('date', ''))
        
        trends = {
            'recent_matches': {},
            'form_trajectory': {},
            'progressive_win_rates': [],
            'moving_averages': {},
            'kd_trends': {},
            'acs_trends': {},
            'adr_trends': {},
            'fb_trends': {}
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
                
                # Calculate KD and other metrics trends
                kd_vals = [match.get('team_avg_kd', 0) for match in recent_window if 'team_avg_kd' in match]
                acs_vals = [match.get('team_avg_acs', 0) for match in recent_window if 'team_avg_acs' in match]
                adr_vals = [match.get('team_avg_adr', 0) for match in recent_window if 'team_avg_adr' in match]
                fb_vals = [match.get('team_avg_fb', 0) for match in recent_window if 'team_avg_fb' in match]
                
                trends['kd_trends'][f'last_{window}_avg'] = np.mean(kd_vals) if kd_vals else 0
                trends['acs_trends'][f'last_{window}_avg'] = np.mean(acs_vals) if acs_vals else 0
                trends['adr_trends'][f'last_{window}_avg'] = np.mean(adr_vals) if adr_vals else 0
                trends['fb_trends'][f'last_{window}_avg'] = np.mean(fb_vals) if fb_vals else 0
                
                # Calculate trend direction by comparing first half to second half
                if len(kd_vals) >= 2:
                    mid_point = len(kd_vals) // 2
                    first_half_kd = np.mean(kd_vals[:mid_point]) if kd_vals[:mid_point] else 0
                    second_half_kd = np.mean(kd_vals[mid_point:]) if kd_vals[mid_point:] else 0
                    trends['kd_trends'][f'last_{window}_trend'] = second_half_kd - first_half_kd
                
                if len(acs_vals) >= 2:
                    mid_point = len(acs_vals) // 2
                    first_half_acs = np.mean(acs_vals[:mid_point]) if acs_vals[:mid_point] else 0
                    second_half_acs = np.mean(acs_vals[mid_point:]) if acs_vals[mid_point:] else 0
                    trends['acs_trends'][f'last_{window}_trend'] = second_half_acs - first_half_acs
        
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
        
        # Calculate moving averages for win rates and KD
        for window in window_sizes:
            if window < total_matches:
                win_moving_avgs = []
                kd_moving_avgs = []
                acs_moving_avgs = []
                
                for i in range(window, total_matches + 1):
                    window_matches = sorted_matches[i-window:i]
                    # Win rate moving average
                    wins = sum(1 for match in window_matches if match.get('team_won', False))
                    win_rate = wins / window
                    win_moving_avgs.append(win_rate)
                    
                    # KD moving average
                    kd_vals = [match.get('team_avg_kd', 0) for match in window_matches if 'team_avg_kd' in match]
                    kd_avg = np.mean(kd_vals) if kd_vals else 0
                    kd_moving_avgs.append(kd_avg)
                    
                    # ACS moving average
                    acs_vals = [match.get('team_avg_acs', 0) for match in window_matches if 'team_avg_acs' in match]
                    acs_avg = np.mean(acs_vals) if acs_vals else 0
                    acs_moving_avgs.append(acs_avg)
                
                trends['moving_averages'][f'win_window_{window}'] = win_moving_avgs
                trends['moving_averages'][f'kd_window_{window}'] = kd_moving_avgs
                trends['moving_averages'][f'acs_window_{window}'] = acs_moving_avgs
                
                # Calculate trend direction from moving averages
                if len(win_moving_avgs) >= 2:
                    trends['moving_averages'][f'win_window_{window}_trend'] = win_moving_avgs[-1] - win_moving_avgs[0]
                if len(kd_moving_avgs) >= 2:
                    trends['moving_averages'][f'kd_window_{window}_trend'] = kd_moving_avgs[-1] - kd_moving_avgs[0]
                if len(acs_moving_avgs) >= 2:
                    trends['moving_averages'][f'acs_window_{window}_trend'] = acs_moving_avgs[-1] - acs_moving_avgs[0]
        
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
            
            # Also calculate recency-weighted KD and ACS
            kd_weighted_sum = 0
            kd_weight_sum = 0
            acs_weighted_sum = 0
            acs_weight_sum = 0
            
            for i, match in enumerate(sorted_matches):
                weight = np.exp(i / total_matches)
                
                if 'team_avg_kd' in match:
                    kd_weighted_sum += weight * match['team_avg_kd']
                    kd_weight_sum += weight
                
                if 'team_avg_acs' in match:
                    acs_weighted_sum += weight * match['team_avg_acs']
                    acs_weight_sum += weight
            
            trends['recency_weighted_kd'] = kd_weighted_sum / kd_weight_sum if kd_weight_sum > 0 else 0
            trends['recency_weighted_acs'] = acs_weighted_sum / acs_weight_sum if acs_weight_sum > 0 else 0
        
        return trends
    
    def analyze_opponent_quality(self, team_matches, all_teams):
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
        
        # Extract team's own ranking for comparison
        team_ranking = None
        team_name = None
        if team_matches and len(team_matches) > 0:
            team_name = team_matches[0].get('team_name')
            # Try to find team in all_teams_data
            for team in all_teams:
                if team.get('name') == team_name:
                    team_ranking = team.get('ranking')
                    break
        
        top_10_wins = 0
        top_10_matches = 0
        bottom_50_wins = 0
        bottom_50_matches = 0
        
        upsets_achieved = 0  # Wins against better-ranked teams
        upset_opportunities = 0  # Matches against better-ranked teams
        upset_suffered = 0  # Losses against worse-ranked teams
        upset_vulnerabilities = 0  # Matches against worse-ranked teams
        
        # KD against top-ranked opponents
        top_10_kd_values = []
        other_kd_values = []
        
        for match in team_matches:
            opponent_name = match.get('opponent_name')
            opponent_ranking = None
            opponent_rating = None
            
            # Find opponent in all_teams_data
            for team in all_teams:
                if team.get('name') == opponent_name:
                    opponent_ranking = team.get('ranking')
                    opponent_rating = team.get('rating')
                    break
            
            if opponent_ranking:
                total_opponent_ranking += opponent_ranking
                count_with_ranking += 1
                
                # Check if opponent is in top 10
                if opponent_ranking <= 10:
                    top_10_matches += 1
                    if match.get('team_won', False):
                        top_10_wins += 1
                    
                    # Track KD against top opponents
                    if 'team_avg_kd' in match:
                        top_10_kd_values.append(match['team_avg_kd'])
                else:
                    # Track KD against other opponents
                    if 'team_avg_kd' in match:
                        other_kd_values.append(match['team_avg_kd'])
                
                # Check if opponent is outside top 50
                if opponent_ranking > 50:
                    bottom_50_matches += 1
                    if match.get('team_won', False):
                        bottom_50_wins += 1
                
                # Calculate upset metrics if we know both team rankings
                if team_ranking:
                    if opponent_ranking < team_ranking:  # Opponent is better ranked
                        upset_opportunities += 1
                        if match.get('team_won', False):
                            upsets_achieved += 1
                    elif opponent_ranking > team_ranking:  # Opponent is worse ranked
                        upset_vulnerabilities += 1
                        if not match.get('team_won', False):
                            upset_suffered += 1
            
            if opponent_rating:
                total_opponent_rating += opponent_rating
        
        # Calculate averages and rates
        if count_with_ranking > 0:
            opponent_quality['avg_opponent_ranking'] = total_opponent_ranking / count_with_ranking
        
        if len(team_matches) > 0:
            opponent_quality['avg_opponent_rating'] = total_opponent_rating / len(team_matches)
        
        if top_10_matches > 0:
            opponent_quality['top_10_win_rate'] = top_10_wins / top_10_matches
        opponent_quality['top_10_matches'] = top_10_matches
        
        if bottom_50_matches > 0:
            opponent_quality['bottom_50_win_rate'] = bottom_50_wins / bottom_50_matches
        opponent_quality['bottom_50_matches'] = bottom_50_matches
        
        if upset_opportunities > 0:
            opponent_quality['upset_factor'] = upsets_achieved / upset_opportunities
        
        if upset_vulnerabilities > 0:
            opponent_quality['upset_vulnerability'] = upset_suffered / upset_vulnerabilities
        
        # Calculate opponent strength of schedule (lower is better)
        opponent_quality['strength_of_schedule'] = opponent_quality['avg_opponent_ranking']
        
        # Calculate KD differential between top opponents and others
        if top_10_kd_values and other_kd_values:
            top_10_avg_kd = np.mean(top_10_kd_values)
            other_avg_kd = np.mean(other_kd_values)
            opponent_quality['top_10_kd_differential'] = top_10_avg_kd - other_avg_kd
        
        return opponent_quality
    
    def analyze_event_performance(self, team_matches):
        """Analyze performance differences between online and LAN events."""
        event_performance = {
            'lan': {
                'matches': 0,
                'wins': 0,
                'win_rate': 0,
                'avg_score': 0,
                'avg_opponent_score': 0,
                'score_differential': 0,
                'avg_kd': 0,
                'avg_acs': 0
            },
            'online': {
                'matches': 0,
                'wins': 0,
                'win_rate': 0,
                'avg_score': 0,
                'avg_opponent_score': 0,
                'score_differential': 0,
                'avg_kd': 0,
                'avg_acs': 0
            }
        }
        
        # KD and ACS values by event type
        lan_kd_values = []
        online_kd_values = []
        lan_acs_values = []
        online_acs_values = []
        
        # Accumulate stats by event type
        for match in team_matches:
            event_info = match.get('event', {})
            event_name = event_info if isinstance(event_info, str) else event_info.get('name', '')
            
            # Determine if LAN or online based on event name
            # This is a simple heuristic - you might need a more robust method
            is_lan = any(lan_term in event_name.lower() for lan_term in ['lan', 'masters', 'champions', 'major'])
            
            event_type = 'lan' if is_lan else 'online'
            
            event_performance[event_type]['matches'] += 1
            event_performance[event_type]['wins'] += 1 if match.get('team_won', False) else 0
            event_performance[event_type]['avg_score'] += match.get('team_score', 0)
            event_performance[event_type]['avg_opponent_score'] += match.get('opponent_score', 0)
            
            # Track KD and ACS by event type
            if 'team_avg_kd' in match:
                if is_lan:
                    lan_kd_values.append(match['team_avg_kd'])
                else:
                    online_kd_values.append(match['team_avg_kd'])
            
            if 'team_avg_acs' in match:
                if is_lan:
                    lan_acs_values.append(match['team_avg_acs'])
                else:
                    online_acs_values.append(match['team_avg_acs'])
        
        # Calculate averages
        for event_type in ['lan', 'online']:
            stats = event_performance[event_type]
            if stats['matches'] > 0:
                stats['win_rate'] = stats['wins'] / stats['matches']
                stats['avg_score'] = stats['avg_score'] / stats['matches']
                stats['avg_opponent_score'] = stats['avg_opponent_score'] / stats['matches']
                stats['score_differential'] = stats['avg_score'] - stats['avg_opponent_score']
        
        # Add KD and ACS averages
        event_performance['lan']['avg_kd'] = np.mean(lan_kd_values) if lan_kd_values else 0
        event_performance['online']['avg_kd'] = np.mean(online_kd_values) if online_kd_values else 0
        event_performance['lan']['avg_acs'] = np.mean(lan_acs_values) if lan_acs_values else 0
        event_performance['online']['avg_acs'] = np.mean(online_acs_values) if online_acs_values else 0
        
        # Calculate LAN vs. online differentials
        if event_performance['lan']['matches'] > 0 and event_performance['online']['matches'] > 0:
            event_performance['lan_vs_online_win_rate_diff'] = event_performance['lan']['win_rate'] - event_performance['online']['win_rate']
            event_performance['lan_vs_online_kd_diff'] = event_performance['lan']['avg_kd'] - event_performance['online']['avg_kd']
            event_performance['lan_vs_online_acs_diff'] = event_performance['lan']['avg_acs'] - event_performance['online']['avg_acs']
        else:
            event_performance['lan_vs_online_win_rate_diff'] = 0
            event_performance['lan_vs_online_kd_diff'] = 0
            event_performance['lan_vs_online_acs_diff'] = 0
        
        return event_performance
    
    def aggregate_player_metrics(self, player_stats):
        """Aggregate individual player statistics to derive team-level metrics."""
        if not player_stats:
            return {}
        
        team_player_metrics = {
            'avg_team_rating': 0,
            'avg_team_acs': 0,
            'avg_team_kd_ratio': 0,
            'star_player_rating': 0,  # highest rated player
            'role_balance': 0,        # distribution of player roles
            'roster_stability': 0,    # how long current roster has been together
            'player_stats': []
        }
        
        # Calculate team averages
        total_rating = 0
        total_acs = 0
        total_kd = 0
        highest_rating = 0
        
        for player in player_stats:
            player_rating = player.get('rating', 0)
            player_acs = player.get('acs', 0)  # Average Combat Score
            player_kd = player.get('kd', 0)    # Kill/Death ratio
            
            total_rating += player_rating
            total_acs += player_acs
            total_kd += player_kd
            
            # Track highest rated player
            if player_rating > highest_rating:
                highest_rating = player_rating
                
            # Store individual player stats for role analysis
            team_player_metrics['player_stats'].append({
                'name': player.get('name', 'Unknown'),
                'rating': player_rating,
                'acs': player_acs,
                'kd': player_kd,
                'agents': player.get('agents', []),  # List of agents played
                'first_blood': player.get('first_blood', 0),  # Entry ability
                'clutch_success': player.get('clutch_success', 0)  # Clutch ability
            })
        
        # Calculate team averages
        num_players = len(player_stats)
        if num_players > 0:
            team_player_metrics['avg_team_rating'] = total_rating / num_players
            team_player_metrics['avg_team_acs'] = total_acs / num_players
            team_player_metrics['avg_team_kd_ratio'] = total_kd / num_players
            team_player_metrics['star_player_rating'] = highest_rating
        
        ## Calculate role balance based on agent distribution
        # This is a simple approach - you could develop more sophisticated metrics
        agent_roles = {
            'controller': ['astra', 'brimstone', 'omen', 'viper', 'harbor'],
            'duelist': ['jett', 'phoenix', 'raze', 'reyna', 'yoru', 'neon'],
            'initiator': ['breach', 'kayo', 'skye', 'sova', 'fade', 'gekko'],
            'sentinel': ['chamber', 'cypher', 'killjoy', 'sage', 'deadlock']
        }
        
        # Count agents by role
        role_counts = {role: 0 for role in agent_roles}
        
        for player in team_player_metrics['player_stats']:
            player_agents = player.get('agents', [])
            for agent in player_agents:
                for role, agents in agent_roles.items():
                    if agent.lower() in agents:
                        role_counts[role] += 1
                        break
        
        # Calculate role balance (standard deviation of role distribution)
        # Lower values indicate more balanced team composition
        if role_counts:
            values = list(role_counts.values())
            if values:
                team_player_metrics['role_balance'] = np.std(values)
        
        return team_player_metrics
    
    def calculate_head_to_head(self, team1_matches, team2_name, team2_matches, team1_name):
        """Calculate head-to-head statistics between two teams."""
        # Validate input
        if not team1_name or not team2_name:
            logger.error("Empty team name provided to head-to-head calculation")
            return {
                'team1_h2h_matches': 0,
                'team1_h2h_wins': 0,
                'team1_h2h_win_rate': 0,
                'team2_h2h_matches': 0,
                'team2_h2h_wins': 0,
                'team2_h2h_win_rate': 0,
                'total_h2h_matches': 0,
                'team1_h2h_avg_kd': 0,
                'team2_h2h_avg_kd': 0,
                'team1_h2h_avg_acs': 0,
                'team2_h2h_avg_acs': 0
            }
        
        logger.info(f"Calculating H2H for {team1_name} vs {team2_name}")
        
        # Find matches between team1 and team2 - EXACT MATCH ONLY
        team1_vs_team2 = [
            match for match in team1_matches 
            if match['opponent_name'].lower() == team2_name.lower()
        ]
        
        team2_vs_team1 = [
            match for match in team2_matches 
            if match['opponent_name'].lower() == team1_name.lower()
        ]
        
        # Create a set of unique match IDs to avoid double-counting
        unique_match_ids = set()
        for match in team1_vs_team2:
            unique_match_ids.add(match['match_id'])
        for match in team2_vs_team1:
            unique_match_ids.add(match['match_id'])
        
        # Calculate total unique matches
        total_h2h_matches = len(unique_match_ids)
        
        # Calculate stats for team1
        team1_h2h_wins = sum(1 for match in team1_vs_team2 if match['team_won'])
        team1_h2h_win_rate = team1_h2h_wins / len(team1_vs_team2) if len(team1_vs_team2) > 0 else 0
        
        # Calculate stats for team2
        team2_h2h_wins = sum(1 for match in team2_vs_team1 if match['team_won'])
        team2_h2h_win_rate = team2_h2h_wins / len(team2_vs_team1) if len(team2_vs_team1) > 0 else 0
        
        # Calculate KD and ACS averages in head-to-head matches
        team1_kd_values = [match.get('team_avg_kd', 0) for match in team1_vs_team2 if 'team_avg_kd' in match]
        team2_kd_values = [match.get('team_avg_kd', 0) for match in team2_vs_team1 if 'team_avg_kd' in match]
        team1_acs_values = [match.get('team_avg_acs', 0) for match in team1_vs_team2 if 'team_avg_acs' in match]
        team2_acs_values = [match.get('team_avg_acs', 0) for match in team2_vs_team1 if 'team_avg_acs' in match]
        
        team1_h2h_avg_kd = np.mean(team1_kd_values) if team1_kd_values else 0
        team2_h2h_avg_kd = np.mean(team2_kd_values) if team2_kd_values else 0
        team1_h2h_avg_acs = np.mean(team1_acs_values) if team1_acs_values else 0
        team2_h2h_avg_acs = np.mean(team2_acs_values) if team2_acs_values else 0
        
        # Combine all h2h stats
        h2h_stats = {
            'team1_h2h_matches': len(team1_vs_team2),
            'team1_h2h_wins': team1_h2h_wins,
            'team1_h2h_win_rate': team1_h2h_win_rate,
            'team2_h2h_matches': len(team2_vs_team1),
            'team2_h2h_wins': team2_h2h_wins,
            'team2_h2h_win_rate': team2_h2h_win_rate,
            'total_h2h_matches': total_h2h_matches,
            'team1_h2h_avg_kd': team1_h2h_avg_kd,
            'team2_h2h_avg_kd': team2_h2h_avg_kd,
            'team1_h2h_avg_acs': team1_h2h_avg_acs,
            'team2_h2h_avg_acs': team2_h2h_avg_acs,
            'kd_differential': team1_h2h_avg_kd - team2_h2h_avg_kd,
            'acs_differential': team1_h2h_avg_acs - team2_h2h_avg_acs
        }
        
        return h2h_stats


class FeatureEngineering:
    """Class to create and process features for machine learning models"""
    
    def __init__(self, data_processor):
        self.data_processor = data_processor
    
    def create_basic_features(self, team1_stats, team2_stats, h2h_stats, team1_details, team2_details):
        """Create a basic feature vector for prediction."""
        features = {
            # Basic team1 stats
            'team1_matches': team1_stats.get('matches', 0),
            'team1_wins': team1_stats.get('wins', 0),
            'team1_win_rate': team1_stats.get('win_rate', 0.5),
            'team1_avg_score': team1_stats.get('avg_score', 0),
            'team1_score_diff': team1_stats.get('score_differential', 0),
            'team1_recent_form': team1_stats.get('recent_form', 0.5),
            'team1_avg_kd': team1_stats.get('avg_kd', 1.0),
            'team1_recent_avg_kd': team1_stats.get('recent_avg_kd', 1.0),
            'team1_avg_acs': team1_stats.get('avg_acs', 0),
            'team1_recent_avg_acs': team1_stats.get('recent_avg_acs', 0),
            
            # Basic team2 stats
            'team2_matches': team2_stats.get('matches', 0),
            'team2_wins': team2_stats.get('wins', 0),
            'team2_win_rate': team2_stats.get('win_rate', 0.5),
            'team2_avg_score': team2_stats.get('avg_score', 0),
            'team2_score_diff': team2_stats.get('score_differential', 0),
            'team2_recent_form': team2_stats.get('recent_form', 0.5),
            'team2_avg_kd': team2_stats.get('avg_kd', 1.0),
            'team2_recent_avg_kd': team2_stats.get('recent_avg_kd', 1.0),
            'team2_avg_acs': team2_stats.get('avg_acs', 0),
            'team2_recent_avg_acs': team2_stats.get('recent_avg_acs', 0),
            
            # Team ranking and rating
            'team1_ranking': team1_details.get('ranking', 9999) if team1_details else 9999,
            'team2_ranking': team2_details.get('ranking', 9999) if team2_details else 9999,
            'team1_rating': team1_details.get('rating', 1500) if team1_details else 1500,
            'team2_rating': team2_details.get('rating', 1500) if team2_details else 1500,
            
            # Head-to-head stats
            'h2h_matches': h2h_stats.get('total_h2h_matches', 0),
            'team1_h2h_wins': h2h_stats.get('team1_h2h_wins', 0),
            'team1_h2h_win_rate': h2h_stats.get('team1_h2h_win_rate', 0.5),
            'team1_h2h_avg_kd': h2h_stats.get('team1_h2h_avg_kd', 1.0),
            'team2_h2h_avg_kd': h2h_stats.get('team2_h2h_avg_kd', 1.0),
            'h2h_kd_differential': h2h_stats.get('kd_differential', 0),
            
            # Relative strength indicators
            'ranking_diff': (team1_details.get('ranking', 9999) if team1_details else 9999) - 
                            (team2_details.get('ranking', 9999) if team2_details else 9999),
            'rating_diff': (team1_details.get('rating', 1500) if team1_details else 1500) - 
                           (team2_details.get('rating', 1500) if team2_details else 1500),
            'win_rate_diff': team1_stats.get('win_rate', 0.5) - team2_stats.get('win_rate', 0.5),
            'avg_score_diff': team1_stats.get('avg_score', 0) - team2_stats.get('avg_score', 0),
            'recent_form_diff': team1_stats.get('recent_form', 0.5) - team2_stats.get('recent_form', 0.5),
            'avg_kd_diff': team1_stats.get('avg_kd', 1.0) - team2_stats.get('avg_kd', 1.0),
            'recent_kd_diff': team1_stats.get('recent_avg_kd', 1.0) - team2_stats.get('recent_avg_kd', 1.0),
            'avg_acs_diff': team1_stats.get('avg_acs', 0) - team2_stats.get('avg_acs', 0),
            'recent_acs_diff': team1_stats.get('recent_avg_acs', 0) - team2_stats.get('recent_avg_acs', 0)
        }
        
        return features
    
    def create_comprehensive_features(self, team1_id, team2_id, api, all_teams):
        """Create a comprehensive feature set for match prediction."""
        # Fetch all necessary data
        team1_details = api.get_team_details(team1_id)
        team2_details = api.get_team_details(team2_id)
        
        team1_history = api.get_team_match_history(team1_id)
        team2_history = api.get_team_match_history(team2_id)
        
        team1_matches = self.data_processor.parse_match_data(team1_history, team1_details.get('name', ''))
        team2_matches = self.data_processor.parse_match_data(team2_history, team2_details.get('name', ''))
        
        # Fetch player statistics
        team1_player_stats = api.get_team_player_stats(team1_id)
        team2_player_stats = api.get_team_player_stats(team2_id)
        
        # Calculate all advanced statistics
        team1_stats = self.data_processor.calculate_team_stats(team1_matches)
        team2_stats = self.data_processor.calculate_team_stats(team2_matches)
        
        team1_trends = self.data_processor.analyze_performance_trends(team1_matches)
        team2_trends = self.data_processor.analyze_performance_trends(team2_matches)
        
        team1_opponent_quality = self.data_processor.analyze_opponent_quality(team1_matches, all_teams)
        team2_opponent_quality = self.data_processor.analyze_opponent_quality(team2_matches, all_teams)
        
        team1_event_performance = self.data_processor.analyze_event_performance(team1_matches)
        team2_event_performance = self.data_processor.analyze_event_performance(team2_matches)
        
        team1_player_metrics = self.data_processor.aggregate_player_metrics(team1_player_stats)
        team2_player_metrics = self.data_processor.aggregate_player_metrics(team2_player_stats)
        
        h2h_stats = self.data_processor.calculate_head_to_head(team1_matches, team2_details.get('name', ''), 
                                          team2_matches, team1_details.get('name', ''))
        
        # Create comprehensive feature dictionary
        features = {
            # Basic team stats
            'team1_matches': team1_stats.get('matches', 0),
            'team1_wins': team1_stats.get('wins', 0),
            'team1_win_rate': team1_stats.get('win_rate', 0.5),
            'team1_avg_score': team1_stats.get('avg_score', 0),
            'team1_score_diff': team1_stats.get('score_differential', 0),
            'team1_avg_kd': team1_stats.get('avg_kd', 1.0),
            'team1_avg_acs': team1_stats.get('avg_acs', 0),
            'team1_avg_adr': team1_stats.get('avg_adr', 0),
            'team1_avg_fb': team1_stats.get('avg_fb', 0),
            
            'team2_matches': team2_stats.get('matches', 0),
            'team2_wins': team2_stats.get('wins', 0),
            'team2_win_rate': team2_stats.get('win_rate', 0.5),
            'team2_avg_score': team2_stats.get('avg_score', 0),
            'team2_score_diff': team2_stats.get('score_differential', 0),
            'team2_avg_kd': team2_stats.get('avg_kd', 1.0),
            'team2_avg_acs': team2_stats.get('avg_acs', 0),
            'team2_avg_adr': team2_stats.get('avg_adr', 0),
            'team2_avg_fb': team2_stats.get('avg_fb', 0),
            
            # Team ranking and rating
            'team1_ranking': team1_details.get('ranking', 9999),
            'team2_ranking': team2_details.get('ranking', 9999),
            'team1_rating': team1_details.get('rating', 1500),
            'team2_rating': team2_details.get('rating', 1500),
            
            # Recent form and trend features
            'team1_recent_form_5': team1_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5),
            'team1_recent_form_10': team1_trends.get('recent_matches', {}).get('last_10_win_rate', 0.5),
            'team1_form_trajectory': team1_trends.get('form_trajectory', {}).get('5_vs_10', 0),
            'team1_recency_weighted_win_rate': team1_trends.get('recency_weighted_win_rate', 0.5),
            'team1_recency_weighted_kd': team1_trends.get('recency_weighted_kd', 1.0),
            'team1_recency_weighted_acs': team1_trends.get('recency_weighted_acs', 0),
            'team1_kd_trend_5': team1_trends.get('kd_trends', {}).get('last_5_trend', 0),
            'team1_acs_trend_5': team1_trends.get('acs_trends', {}).get('last_5_trend', 0),
            
            'team2_recent_form_5': team2_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5),
            'team2_recent_form_10': team2_trends.get('recent_matches', {}).get('last_10_win_rate', 0.5),
            'team2_form_trajectory': team2_trends.get('form_trajectory', {}).get('5_vs_10', 0),
            'team2_recency_weighted_win_rate': team2_trends.get('recency_weighted_win_rate', 0.5),
            'team2_recency_weighted_kd': team2_trends.get('recency_weighted_kd', 1.0),
            'team2_recency_weighted_acs': team2_trends.get('recency_weighted_acs', 0),
            'team2_kd_trend_5': team2_trends.get('kd_trends', {}).get('last_5_trend', 0),
            'team2_acs_trend_5': team2_trends.get('acs_trends', {}).get('last_5_trend', 0),
            
            # Event type performance
            'team1_lan_win_rate': team1_event_performance.get('lan', {}).get('win_rate', 0.5),
            'team1_online_win_rate': team1_event_performance.get('online', {}).get('win_rate', 0.5),
            'team1_lan_vs_online_diff': team1_event_performance.get('lan_vs_online_win_rate_diff', 0),
            'team1_lan_kd': team1_event_performance.get('lan', {}).get('avg_kd', 1.0),
            'team1_online_kd': team1_event_performance.get('online', {}).get('avg_kd', 1.0),
            'team1_lan_vs_online_kd_diff': team1_event_performance.get('lan_vs_online_kd_diff', 0),
            
            'team2_lan_win_rate': team2_event_performance.get('lan', {}).get('win_rate', 0.5),
            'team2_online_win_rate': team2_event_performance.get('online', {}).get('win_rate', 0.5),
            'team2_lan_vs_online_diff': team2_event_performance.get('lan_vs_online_win_rate_diff', 0),
            'team2_lan_kd': team2_event_performance.get('lan', {}).get('avg_kd', 1.0),
            'team2_online_kd': team2_event_performance.get('online', {}).get('avg_kd', 1.0),
            'team2_lan_vs_online_kd_diff': team2_event_performance.get('lan_vs_online_kd_diff', 0),
            
            # Opponent quality metrics
            'team1_avg_opponent_ranking': team1_opponent_quality.get('avg_opponent_ranking', 50),
            'team1_top_10_win_rate': team1_opponent_quality.get('top_10_win_rate', 0),
            'team1_upset_factor': team1_opponent_quality.get('upset_factor', 0),
            'team1_top_10_kd_differential': team1_opponent_quality.get('top_10_kd_differential', 0),
            
            'team2_avg_opponent_ranking': team2_opponent_quality.get('avg_opponent_ranking', 50),
            'team2_top_10_win_rate': team2_opponent_quality.get('top_10_win_rate', 0),
            'team2_upset_factor': team2_opponent_quality.get('upset_factor', 0),
            'team2_top_10_kd_differential': team2_opponent_quality.get('top_10_kd_differential', 0),
            
            # Player metrics
            'team1_avg_player_rating': team1_player_metrics.get('avg_team_rating', 1),
            'team1_star_player_rating': team1_player_metrics.get('star_player_rating', 1),
            'team1_role_balance': team1_player_metrics.get('role_balance', 1),
            'team1_avg_player_kd': team1_player_metrics.get('avg_team_kd_ratio', 1),
            
            'team2_avg_player_rating': team2_player_metrics.get('avg_team_rating', 1),
            'team2_star_player_rating': team2_player_metrics.get('star_player_rating', 1),
            'team2_role_balance': team2_player_metrics.get('role_balance', 1),
            'team2_avg_player_kd': team2_player_metrics.get('avg_team_kd_ratio', 1),
            
            # Head-to-head stats
            'h2h_matches': h2h_stats.get('total_h2h_matches', 0),
            'team1_h2h_wins': h2h_stats.get('team1_h2h_wins', 0),
            'team1_h2h_win_rate': h2h_stats.get('team1_h2h_win_rate', 0.5),
            'team1_h2h_avg_kd': h2h_stats.get('team1_h2h_avg_kd', 1.0),
            'team2_h2h_avg_kd': h2h_stats.get('team2_h2h_avg_kd', 1.0),
            'h2h_kd_differential': h2h_stats.get('kd_differential', 0),
            'team1_h2h_avg_acs': h2h_stats.get('team1_h2h_avg_acs', 0),
            'team2_h2h_avg_acs': h2h_stats.get('team2_h2h_avg_acs', 0),
            'h2h_acs_differential': h2h_stats.get('acs_differential', 0),
            
            # Relative strength indicators
            'ranking_diff': team1_details.get('ranking', 9999) - team2_details.get('ranking', 9999),
            'rating_diff': team1_details.get('rating', 1500) - team2_details.get('rating', 1500),
            'win_rate_diff': team1_stats.get('win_rate', 0.5) - team2_stats.get('win_rate', 0.5),
            'avg_score_diff': team1_stats.get('avg_score', 0) - team2_stats.get('avg_score', 0),
            'recent_form_diff': team1_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5) - 
                                team2_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5),
            'player_rating_diff': team1_player_metrics.get('avg_team_rating', 1) - 
                                 team2_player_metrics.get('avg_team_rating', 1),
            'star_player_diff': team1_player_metrics.get('star_player_rating', 1) - 
                               team2_player_metrics.get('star_player_rating', 1),
            'kd_diff': team1_stats.get('avg_kd', 1.0) - team2_stats.get('avg_kd', 1.0),
            'recent_kd_diff': team1_stats.get('recent_avg_kd', 1.0) - team2_stats.get('recent_avg_kd', 1.0),
            'acs_diff': team1_stats.get('avg_acs', 0) - team2_stats.get('avg_acs', 0),
            'recent_acs_diff': team1_stats.get('recent_avg_acs', 0) - team2_stats.get('recent_avg_acs', 0),
            'adr_diff': team1_stats.get('avg_adr', 0) - team2_stats.get('avg_adr', 0),
            'fb_diff': team1_stats.get('avg_fb', 0) - team2_stats.get('avg_fb', 0)
        }
        
        # Add symmetrical feature interactions (multiplication and ratios)
        features['win_rate_product'] = team1_stats.get('win_rate', 0.5) * team2_stats.get('win_rate', 0.5)
        features['win_rate_ratio'] = team1_stats.get('win_rate', 0.5) / team2_stats.get('win_rate', 0.5) if team2_stats.get('win_rate', 0.5) > 0 else 1.0
        features['kd_product'] = team1_stats.get('avg_kd', 1.0) * team2_stats.get('avg_kd', 1.0)
        features['kd_ratio'] = team1_stats.get('avg_kd', 1.0) / team2_stats.get('avg_kd', 1.0) if team2_stats.get('avg_kd', 1.0) > 0 else 1.0
        features['rating_product'] = team1_details.get('rating', 1500) * team2_details.get('rating', 1500)
        features['rating_ratio'] = team1_details.get('rating', 1500) / team2_details.get('rating', 1500) if team2_details.get('rating', 1500) > 0 else 1.0
        features['recent_form_product'] = team1_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5) * team2_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5)
        
        return features
    
    def create_training_data_from_match_history(self, matches, team_id, team_name, api, all_teams):
        """Create training data from a team's match history."""
        training_samples = []
        
        for match in matches:
            opponent_name = match.get('opponent_name', '')
            opponent_id = None
            
            # Find opponent ID
            for team in all_teams:
                if team.get('name') == opponent_name:
                    opponent_id = team.get('id')
                    break
            
            if not opponent_id:
                continue
            
            # Get opponent's match history
            opponent_history = api.get_team_match_history(opponent_id)
            if not opponent_history:
                continue
                
            # Parse opponent match data
            opponent_matches = self.data_processor.parse_match_data(opponent_history, opponent_name)
            if not opponent_matches:
                continue
            
            # Collect match data before this match
            match_date = match.get('date', '')
            if not match_date:
                continue
                
            try:
                match_datetime = datetime.strptime(match_date, '%Y-%m-%d')
            except:
                continue
            
            # Filter history to only include matches prior to this match
            team_prior_matches = [m for m in matches if m.get('date', '') and m.get('date', '') < match_date]
            opponent_prior_matches = [m for m in opponent_matches if m.get('date', '') and m.get('date', '') < match_date]
            
            if len(team_prior_matches) < 5 or len(opponent_prior_matches) < 5:
                continue  # Not enough prior matches for both teams
            
            # Calculate features based on prior match data
            team_stats = self.data_processor.calculate_team_stats(team_prior_matches)
            opponent_stats = self.data_processor.calculate_team_stats(opponent_prior_matches)
            
            team_details = api.get_team_details(team_id)
            opponent_details = api.get_team_details(opponent_id)
            
            h2h_stats = self.data_processor.calculate_head_to_head(
                team_prior_matches, opponent_name, 
                opponent_prior_matches, team_name
            )
            
            # Get actual match result (target variable)
            team_won = match.get('team_won', False)
            
            # Create feature vector
            features = self.create_basic_features(team_stats, opponent_stats, h2h_stats, team_details, opponent_details)
            
            # Add label
            features['team1_won'] = 1 if team_won else 0
            
            training_samples.append(features)
        
        return pd.DataFrame(training_samples) if training_samples else None
    
    def create_symmetrical_training_data(self, team1_matches, team2_matches, team1_name, team2_name, api, all_teams):
        """Create symmetrical training data by swapping team perspectives."""
        # Get training data from team1's perspective
        team1_data = self.create_training_data_from_match_history(team1_matches, team1_name, team1_name, api, all_teams)
        
        # Get training data from team2's perspective
        team2_data = self.create_training_data_from_match_history(team2_matches, team2_name, team2_name, api, all_teams)
        
        # Combine datasets
        combined_data = pd.concat([team1_data, team2_data], ignore_index=True) if team1_data is not None and team2_data is not None else None
        
        # Add mirror samples by swapping team1 and team2 features
        if combined_data is not None:
            mirror_samples = []
            
            for _, row in combined_data.iterrows():
                mirrored_row = {}
                
                # Swap team1 and team2 features
                for col in row.index:
                    if col.startswith('team1_'):
                        # Replace team1_ with team2_ in column name
                        new_col = col.replace('team1_', 'team2_')
                        mirrored_row[new_col] = row[col]
                    elif col.startswith('team2_'):
                        # Replace team2_ with team1_ in column name
                        new_col = col.replace('team2_', 'team1_')
                        mirrored_row[new_col] = row[col]
                    elif col == 'team1_won':
                        # Invert target variable
                        mirrored_row[col] = 1 - row[col]
                    else:
                        # For differential features, negate the value
                        if any(col.endswith(suffix) for suffix in ['_diff', '_differential']):
                            mirrored_row[col] = -row[col]
                        else:
                            mirrored_row[col] = row[col]
                
                mirror_samples.append(mirrored_row)
            
            # Add mirror samples to combined data
            mirror_df = pd.DataFrame(mirror_samples)
            enhanced_data = pd.concat([combined_data, mirror_df], ignore_index=True)
            
            return enhanced_data
        
        return combined_data
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # Fill missing values with defaults
        # For win rates and ratios, use 0.5
        # For K/D ratios, use 1.0
        # For other metrics, use 0
        
        win_rate_cols = [col for col in df.columns if 'win_rate' in col or col.endswith('_form')]
        kd_cols = [col for col in df.columns if 'kd' in col]
        other_cols = [col for col in df.columns if col not in win_rate_cols and col not in kd_cols]
        
        for col in win_rate_cols:
            df[col] = df[col].fillna(0.5)
        
        for col in kd_cols:
            df[col] = df[col].fillna(1.0)
            
        for col in other_cols:
            df[col] = df[col].fillna(0)
            
        return df
    
    def normalize_features(self, train_df, test_df=None):
        """Normalize features for better model performance."""
        scaler = StandardScaler()
        
        # Separate target
        if 'team1_won' in train_df.columns:
            X_train = train_df.drop('team1_won', axis=1)
            y_train = train_df['team1_won']
        else:
            X_train = train_df
            y_train = None
        
        # Fit and transform training data
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Create DataFrame with scaled values
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        # If test data is provided, transform it using the same scaler
        if test_df is not None:
            if 'team1_won' in test_df.columns:
                X_test = test_df.drop('team1_won', axis=1)
                y_test = test_df['team1_won']
            else:
                X_test = test_df
                y_test = None
                
            X_test_scaled = scaler.transform(X_test)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            
            # Combine with target if available
            if y_test is not None:
                X_test_scaled_df['team1_won'] = y_test
        else:
            X_test_scaled_df = None
        
        # Combine with target if available
        if y_train is not None:
            X_train_scaled_df['team1_won'] = y_train
        
        return X_train_scaled_df, X_test_scaled_df, scaler


class ModelTrainer:
    """Class to train and evaluate machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importances = None
        self.scaler = None
        self.selected_features = None
    
    def train_basic_model(self, training_data):
        """Train a basic Random Forest model."""
        if len(training_data) < 10:
            logger.warning(f"Only {len(training_data)} matches available for training. Model may be unreliable.")
        
        # Separate features and target
        X = training_data.drop('team1_won', axis=1)
        y = training_data['team1_won']
        
        # Check for balanced classes
        class_counts = np.bincount(y)
        if len(class_counts) == 1:
            logger.warning(f"All examples belong to class {class_counts[0]}. Using dummy classifier.")
            only_class = int(y.iloc[0])
            
            # Use a dummy classifier
            model = DummyClassifier(strategy='constant', constant=only_class)
            model.fit(X, y)
            
            # Create dummy feature importance
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': np.zeros(len(X.columns))
            })
        else:
            logger.info(f"Class distribution - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            )
            model.fit(X_scaled, y)
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
        
        logger.info("Top 10 most important features:")
        logger.info(feature_importance.head(10))
        
        self.models['random_forest'] = model
        self.best_model = model
        self.feature_importances = feature_importance
        self.scaler = scaler
        self.selected_features = X.columns.tolist()
        
        return model, scaler, X.columns.tolist(), feature_importance
    
    def train_ensemble_model(self, training_data):
        """Train an ensemble model with multiple algorithms."""
        if len(training_data) < 20:
            logger.warning(f"Only {len(training_data)} matches available for training. Model may be unreliable.")
            return self.train_basic_model(training_data)
        
        # Separate features and target
        X = training_data.drop('team1_won', axis=1)
        y = training_data['team1_won']
        
        # Check for balanced classes
        class_counts = np.bincount(y)
        if len(class_counts) == 1:
            logger.warning(f"All examples belong to class {class_counts[0]}. Using dummy classifier.")
            return self.train_basic_model(training_data)
        
        logger.info(f"Class distribution - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create cross-validation strategy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train different models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, 
                class_weight='balanced',
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"Training {name} model...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Save model
            self.models[name] = model
        
        # Create ensemble model
        logger.info("Training voting ensemble model...")
        ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            voting='soft'
        )
        ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate ensemble model
        y_pred = ensemble.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"Ensemble - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Save ensemble model
        self.models['ensemble'] = ensemble
        
        # Use Random Forest for feature importance (more interpretable)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': models['random_forest'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        logger.info("Top 10 most important features:")
        logger.info(feature_importance.head(10))
        
        # Find best model based on F1 score
        best_model_name = max(
            [(name, f1_score(y_test, model.predict(X_test_scaled))) for name, model in self.models.items()],
            key=lambda x: x[1]
        )[0]
        
        logger.info(f"Best model: {best_model_name}")
        
        self.best_model = self.models[best_model_name]
        self.feature_importances = feature_importance
        self.scaler = scaler
        self.selected_features = X.columns.tolist()
        
        return self.best_model, scaler, X.columns.tolist(), feature_importance
    
    def train_advanced_ml_model(self, training_data):
        """Train an advanced Random Forest model with cross-validation, hyperparameter tuning and feature selection."""
        if len(training_data) < 30:
            logger.warning(f"Only {len(training_data)} matches available for training. Using ensemble model instead.")
            return self.train_ensemble_model(training_data)
        
        # Separate features and target
        X = training_data.drop('team1_won', axis=1)
        y = training_data['team1_won']
        
        # Check for balanced classes
        class_counts = np.bincount(y)
        if len(class_counts) == 1:
            logger.warning(f"All examples belong to class {class_counts[0]}. Using dummy classifier.")
            return self.train_basic_model(training_data)
        
        logger.info(f"Class distribution - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create cross-validation strategy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # If classes are very imbalanced, use SMOTE
        if len(class_counts) > 1 and min(class_counts) / max(class_counts) < 0.3:
            logger.info("Applying SMOTE to handle class imbalance...")
            smote = SMOTE(random_state=42)
            X_scaled, y = smote.fit_resample(X_scaled, y)
            logger.info(f"After SMOTE - Class 0: {sum(y==0)}, Class 1: {sum(y==1)}")
        
        # Feature selection
        logger.info("Performing feature selection...")
        selector = SelectFromModel(RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), threshold='median')
        X_selected = selector.fit_transform(X_scaled, y)
        selected_indices = selector.get_support()
        selected_features = [X.columns[i] for i, selected in enumerate(selected_indices) if selected]
        logger.info(f"Selected {len(selected_features)} features")
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        
        # Hyperparameter tuning
        logger.info("Tuning hyperparameters...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        search = RandomizedSearchCV(
            RandomForestClassifier(class_weight='balanced', random_state=42),
            param_distributions=param_grid,
            n_iter=10,
            cv=cv,
            scoring='f1',
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        best_params = search.best_params_
        logger.info(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        logger.info("Training final model with best parameters...")
        model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model on test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info("\nModel Evaluation:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        if len(set(y_test)) > 1:  # Only calculate AUC if there are both classes in test set
            auc = roc_auc_score(y_test, y_prob)
            logger.info(f"ROC AUC: {auc:.4f}")
        
        # Get feature importance for selected features
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        logger.info("\nTop 10 most important features:")
        logger.info(feature_importance.head(10))
        
        # Create a more advanced ensemble model
        logger.info("\nTraining ensemble model with multiple algorithms...")
        rf = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
        gb = GradientBoostingClassifier(random_state=42)
        
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble model
        ens_y_pred = ensemble.predict(X_test)
        ens_y_prob = ensemble.predict_proba(X_test)[:, 1]
        
        ens_accuracy = accuracy_score(y_test, ens_y_pred)
        ens_f1 = f1_score(y_test, ens_y_pred)
        
        logger.info("\nEnsemble Model Evaluation:")
        logger.info(f"Accuracy: {ens_accuracy:.4f}")
        logger.info(f"F1 Score: {ens_f1:.4f}")
        
        # Calibrate probabilities for better prediction
        logger.info("\nCalibrating probabilities...")
        calibrated_model = CalibratedClassifierCV(ensemble, cv=5)
        calibrated_model.fit(X_selected, y)
        
        # Save models
        self.models['random_forest'] = model
        self.models['ensemble'] = ensemble
        self.models['calibrated_ensemble'] = calibrated_model
        
        # Use the calibrated ensemble as best model
        self.best_model = calibrated_model
        self.feature_importances = feature_importance
        self.scaler = scaler
        self.selector = selector
        self.selected_features = selected_features
        
        return calibrated_model, scaler, selector, selected_features, feature_importance
    
    def train_deep_learning_model(self, training_data):
        """Train a deep neural network model for match prediction."""
        if len(training_data) < 50:
            logger.warning(f"Only {len(training_data)} matches available for training. Using advanced ML model instead.")
            return self.train_advanced_ml_model(training_data)
        
        # Separate features and target
        X = training_data.drop('team1_won', axis=1)
        y = training_data['team1_won']
        
        # Check for balanced classes
        class_counts = np.bincount(y)
        if len(class_counts) == 1:
            logger.warning(f"All examples belong to class {class_counts[0]}. Using dummy classifier.")
            return self.train_basic_model(training_data)
        
        logger.info(f"Class distribution - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Define feature groups
        team1_cols = [col for col in X.columns if col.startswith('team1_')]
        team2_cols = [col for col in X.columns if col.startswith('team2_')]
        diff_cols = [col for col in X.columns if '_diff' in col or '_differential' in col]
        other_cols = [col for col in X.columns if col not in team1_cols + team2_cols + diff_cols]
        
        # Get indices for each feature group
        team1_indices = [i for i, col in enumerate(X.columns) if col in team1_cols]
        team2_indices = [i for i, col in enumerate(X.columns) if col in team2_cols]
        diff_indices = [i for i, col in enumerate(X.columns) if col in diff_cols]
        other_indices = [i for i, col in enumerate(X.columns) if col in other_cols]
        
        # Define model architecture with multiple inputs
        team1_input = Input(shape=(len(team1_indices),), name='team1_input')
        team2_input = Input(shape=(len(team2_indices),), name='team2_input')
        diff_input = Input(shape=(len(diff_indices),), name='diff_input')
        other_input = Input(shape=(len(other_indices),), name='other_input')
        
        # Process team1 stats
        team1_dense = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(team1_input)
        team1_dense = BatchNormalization()(team1_dense)
        team1_dense = Dropout(0.3)(team1_dense)
        
        # Process team2 stats
        team2_dense = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(team2_input)
        team2_dense = BatchNormalization()(team2_dense)
        team2_dense = Dropout(0.3)(team2_dense)
        
        # Process differential features
        diff_dense = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(diff_input)
        diff_dense = BatchNormalization()(diff_dense)
        diff_dense = Dropout(0.3)(diff_dense)
        
        # Process other features
        if len(other_indices) > 0:
            other_dense = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(other_input)
            other_dense = BatchNormalization()(other_dense)
            other_dense = Dropout(0.3)(other_dense)
            
            # Combine all features
            combined = Concatenate()([team1_dense, team2_dense, diff_dense, other_dense])
        else:
            # Combine without other features
            combined = Concatenate()([team1_dense, team2_dense, diff_dense])
        
        # Hidden layers
        hidden = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.3)(hidden)
        
        hidden = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.3)(hidden)
        
        # Output layer
        output = Dense(1, activation='sigmoid')(hidden)
        
        # Create model
        model = Model(
            inputs=[team1_input, team2_input, diff_input, other_input] if len(other_indices) > 0 else [team1_input, team2_input, diff_input],
            outputs=output
        )
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
            ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, 'best_nn_model.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Prepare input data
        X_train_team1 = X_train[:, team1_indices]
        X_train_team2 = X_train[:, team2_indices]
        X_train_diff = X_train[:, diff_indices]
        X_train_other = X_train[:, other_indices] if len(other_indices) > 0 else None
        
        X_test_team1 = X_test[:, team1_indices]
        X_test_team2 = X_test[:, team2_indices]
        X_test_diff = X_test[:, diff_indices]
        X_test_other = X_test[:, other_indices] if len(other_indices) > 0 else None
        
        # Train model
        logger.info("Training deep learning model...")
        train_inputs = [X_train_team1, X_train_team2, X_train_diff]
        test_inputs = [X_test_team1, X_test_team2, X_test_diff]
        
        if len(other_indices) > 0:
            train_inputs.append(X_train_other)
            test_inputs.append(X_test_other)
        
        history = model.fit(
            train_inputs,
            y_train,
            validation_data=(test_inputs, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=2
        )
        
        # Evaluate model
        y_pred_proba = model.predict(test_inputs)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info("\nDeep Learning Model Evaluation:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Generate feature importance using a traditional model
        # (neural networks don't provide feature importances directly)
        rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf_model.fit(X_train, y_train)
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Save model components
        self.models['deep_learning'] = model
        self.deep_learning_indices = {
            'team1_indices': team1_indices,
            'team2_indices': team2_indices,
            'diff_indices': diff_indices,
            'other_indices': other_indices
        }
        
        # Compare with traditional model
        logger.info("Training comparison Random Forest model...")
        rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf_model.fit(X_train, y_train)
        rf_y_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_y_pred)
        rf_f1 = f1_score(y_test, rf_y_pred)
        
        logger.info(f"Random Forest Accuracy: {rf_accuracy:.4f}, F1: {rf_f1:.4f}")
        
        # Choose best model
        if f1 >= rf_f1:
            logger.info("Deep Learning model performs better. Using it as the best model.")
            self.best_model = 'deep_learning'
        else:
            logger.info("Random Forest model performs better. Using it as the best model.")
            self.best_model = rf_model
            self.models['random_forest'] = rf_model
        
        self.feature_importances = feature_importance
        self.scaler = scaler
        self.selected_features = X.columns.tolist()
        
        return (model, rf_model, scaler, self.deep_learning_indices, 
                self.selected_features, feature_importance)
    
    def predict_match_outcome(self, features, model_type='best'):
        """Predict the outcome of a match between two teams."""
        if model_type == 'best':
            model = self.best_model
        elif model_type in self.models:
            model = self.models[model_type]
        else:
            logger.error(f"Model type '{model_type}' not found.")
            return None, None
        
        # Create DataFrame from features
        features_df = pd.DataFrame([features])
        
        # Check if using deep learning model
        if model_type == 'deep_learning' or (model_type == 'best' and self.best_model == 'deep_learning'):
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Extract feature groups
            team1_features = features_scaled[:, self.deep_learning_indices['team1_indices']]
            team2_features = features_scaled[:, self.deep_learning_indices['team2_indices']]
            diff_features = features_scaled[:, self.deep_learning_indices['diff_indices']]
            
            # Prepare input
            model_inputs = [team1_features, team2_features, diff_features]
            
            if len(self.deep_learning_indices['other_indices']) > 0:
                other_features = features_scaled[:, self.deep_learning_indices['other_indices']]
                model_inputs.append(other_features)
            
            # Make prediction
            prediction_proba = self.models['deep_learning'].predict(model_inputs)[0][0]
            prediction = 1 if prediction_proba > 0.5 else 0
            prediction_proba = np.array([1 - prediction_proba, prediction_proba])
        else:
            # Handle model with feature selection
            if hasattr(self, 'selector') and self.selector is not None:
                # Ensure features match what the model expects
                for feature in self.selected_features:
                    if feature not in features_df.columns:
                        features_df[feature] = 0
                
                # Scale features
                features_scaled = self.scaler.transform(features_df)
                
                # Apply feature selection
                features_selected = self.selector.transform(features_scaled)
                
                # Make prediction
                prediction = model.predict(features_selected)[0]
                prediction_proba = model.predict_proba(features_selected)[0]
            else:
                # For basic models without feature selection
                # Ensure features match what the model expects
                for feature in self.selected_features:
                    if feature not in features_df.columns:
                        features_df[feature] = 0
                
                features_df = features_df[self.selected_features]
                
                # Scale features
                features_scaled = self.scaler.transform(features_df)
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                prediction_proba = model.predict_proba(features_scaled)[0]
        
        return prediction, prediction_proba
    
    def save_model(self, filename_prefix='valorant_predictor'):
        """Save the trained model and related components."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save components
        model_info = {
            'timestamp': timestamp,
            'feature_names': self.selected_features,
            'feature_importance': self.feature_importances.to_dict() if self.feature_importances is not None else None,
            'deep_learning_indices': getattr(self, 'deep_learning_indices', None),
            'has_selector': hasattr(self, 'selector') and self.selector is not None
        }
        
        # Save model information
        with open(os.path.join(MODEL_DIR, f"{filename_prefix}_info_{timestamp}.json"), 'w') as f:
            json.dump(model_info, f, indent=4, default=str)
        
        # Save scaler
        with open(os.path.join(MODEL_DIR, f"{filename_prefix}_scaler_{timestamp}.pkl"), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save selector if available
        if hasattr(self, 'selector') and self.selector is not None:
            with open(os.path.join(MODEL_DIR, f"{filename_prefix}_selector_{timestamp}.pkl"), 'wb') as f:
                pickle.dump(self.selector, f)
        
        # Save each model
        for model_name, model in self.models.items():
            if model_name == 'deep_learning':
                # Save Keras model in h5 format
                model.save(os.path.join(MODEL_DIR, f"{filename_prefix}_{model_name}_{timestamp}.h5"))
            else:
                # Save scikit-learn model
                with open(os.path.join(MODEL_DIR, f"{filename_prefix}_{model_name}_{timestamp}.pkl"), 'wb') as f:
                    pickle.dump(model, f)
        
        logger.info(f"Model saved with prefix {filename_prefix}_{timestamp}")
        return f"{filename_prefix}_{timestamp}"
    
    def load_model(self, model_prefix):
        """Load a saved model and its components."""
        # Load model information
        info_path = os.path.join(MODEL_DIR, f"{model_prefix}_info.json")
        if not os.path.exists(info_path):
            # Try finding files with prefix
            info_files = [f for f in os.listdir(MODEL_DIR) if f.startswith(f"{model_prefix}_info")]
            if not info_files:
                logger.error(f"No model information found with prefix {model_prefix}")
                return False
            info_path = os.path.join(MODEL_DIR, info_files[0])
        
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        
        # Load selected features
        self.selected_features = model_info.get('feature_names')
        self.deep_learning_indices = model_info.get('deep_learning_indices')
        
        # Load feature importance if available
        if model_info.get('feature_importance'):
            # Convert back to DataFrame
            feature_importance_dict = model_info['feature_importance']
            self.feature_importances = pd.DataFrame({
                'Feature': feature_importance_dict.get('Feature', {}).values(),
                'Importance': feature_importance_dict.get('Importance', {}).values()
            })
        
        # Load scaler
        scaler_path = os.path.join(MODEL_DIR, f"{model_prefix}_scaler.pkl")
        if not os.path.exists(scaler_path):
            scaler_files = [f for f in os.listdir(MODEL_DIR) if f.startswith(f"{model_prefix}_scaler")]
            if not scaler_files:
                logger.error(f"No scaler found with prefix {model_prefix}")
                return False
            scaler_path = os.path.join(MODEL_DIR, scaler_files[0])
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load selector if it exists
        if model_info.get('has_selector'):
            selector_path = os.path.join(MODEL_DIR, f"{model_prefix}_selector.pkl")
            if not os.path.exists(selector_path):
                selector_files = [f for f in os.listdir(MODEL_DIR) if f.startswith(f"{model_prefix}_selector")]
                if selector_files:
                    selector_path = os.path.join(MODEL_DIR, selector_files[0])
                    with open(selector_path, 'rb') as f:
                        self.selector = pickle.load(f)
        
        # Load models
        self.models = {}
        
        # Check for deep learning model
        dl_path = os.path.join(MODEL_DIR, f"{model_prefix}_deep_learning.h5")
        if os.path.exists(dl_path) or any(f.startswith(f"{model_prefix}_deep_learning") for f in os.listdir(MODEL_DIR)):
            if not os.path.exists(dl_path):
                dl_files = [f for f in os.listdir(MODEL_DIR) if f.startswith(f"{model_prefix}_deep_learning")]
                dl_path = os.path.join(MODEL_DIR, dl_files[0])
            
            self.models['deep_learning'] = load_model(dl_path)
        
        # Load scikit-learn models
        model_types = ['random_forest', 'ensemble', 'calibrated_ensemble']
        for model_type in model_types:
            model_path = os.path.join(MODEL_DIR, f"{model_prefix}_{model_type}.pkl")
            if not os.path.exists(model_path):
                model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith(f"{model_prefix}_{model_type}")]
                if not model_files:
                    continue
                model_path = os.path.join(MODEL_DIR, model_files[0])
            
            with open(model_path, 'rb') as f:
                self.models[model_type] = pickle.load(f)
        
        # Set best model to calibrated_ensemble if available, otherwise ensemble or random_forest
        if 'calibrated_ensemble' in self.models:
            self.best_model = self.models['calibrated_ensemble']
        elif 'ensemble' in self.models:
            self.best_model = self.models['ensemble']
        elif 'random_forest' in self.models:
            self.best_model = self.models['random_forest']
        elif 'deep_learning' in self.models:
            self.best_model = 'deep_learning'
        else:
            logger.error(f"No models found with prefix {model_prefix}")
            return False
        
        logger.info(f"Successfully loaded model with prefix {model_prefix}")
        return True


class PredictionSystem:
    """Main class for managing the end-to-end prediction system"""
    
    def __init__(self, api_url=API_URL):
        self.api = ValorantAPI(api_url)
        self.data_processor = MatchDataProcessor(self.api)
        self.feature_engineering = FeatureEngineering(self.data_processor)
        self.model_trainer = ModelTrainer()
        self.prediction_history = []
        self.model_loaded = False
    
    def predict_match(self, team1_name, team2_name, team1_region=None, team2_region=None, 
                     save_results=True, model_type='best', is_lan=None, verbose=True):
        """Predict the outcome of a match between two teams."""
        # Get team IDs
        team1_id = self.api.get_team_id(team1_name, team1_region)
        team2_id = self.api.get_team_id(team2_name, team2_region)
        
        if not team1_id or not team2_id:
            logger.error("Could not find team IDs. Please check team names and try again.")
            return None
        
        # Get all teams for additional metrics
        all_teams = self.api.get_all_teams()
        
        # Create comprehensive feature set
        try:
            features = self.feature_engineering.create_comprehensive_features(team1_id, team2_id, self.api, all_teams)
            
            # Add LAN/online indicator if provided
            if is_lan is not None:
                features['is_lan'] = 1 if is_lan else 0
            
            # Make prediction
            prediction, prediction_proba = self.model_trainer.predict_match_outcome(features, model_type)
            
            # Determine winner and confidence
            if prediction == 1:
                winner = team1_name
                win_probability = prediction_proba[1]
            else:
                winner = team2_name
                win_probability = prediction_proba[0]
            
            if verbose:
                logger.info(f"\n============ MATCH PREDICTION ============")
                logger.info(f"Team 1: {team1_name}")            'team2_avg_acs': team2_stats.get('avg_acs', 0),
            'team2_recent_avg_acs': team2_stats.get('recent_avg_acs', 0),
            
            # Team ranking and rating
            'team1_ranking': team1_details.get('ranking', 9999) if team1_details else 9999,
            'team2_ranking': team2_details.get('ranking', 9999) if team2_details else 9999,
            'team1_rating': team1_details.get('rating', 1500) if team1_details else 1500,
            'team2_rating': team2_details.get('rating', 1500) if team2_details else 1500,
            
            # Head-to-head stats
            'h2h_matches': h2h_stats.get('total_h2h_matches', 0),
            'team1_h2h_wins': h2h_stats.get('team1_h2h_wins', 0),
            'team1_h2h_win_rate': h2h_stats.get('team1_h2h_win_rate', 0.5),
            'team1_h2h_avg_kd': h2h_stats.get('team1_h2h_avg_kd', 1.0),
            'team2_h2h_avg_kd': h2h_stats.get('team2_h2h_avg_kd', 1.0),
            'h2h_kd_differential': h2h_stats.get('kd_differential', 0),
            
            # Relative strength indicators
            'ranking_diff': (team1_details.get('ranking', 9999) if team1_details else 9999) - 
                            (team2_details.get('ranking', 9999) if team2_details else 9999),
            'rating_diff': (team1_details.get('rating', 1500) if team1_details else 1500) - 
                           (team2_details.get('rating', 1500) if team2_details else 1500),
            'win_rate_diff': team1_stats.get('win_rate', 0.5) - team2_stats.get('win_rate', 0.5),
            'avg_score_diff': team1_stats.get('avg_score', 0) - team2_stats.get('avg_score', 0),
            'recent_form_diff': team1_stats.get('recent_form', 0.5) - team2_stats.get('recent_form', 0.5),
            'avg_kd_diff': team1_stats.get('avg_kd', 1.0) - team2_stats.get('avg_kd', 1.0),
            'recent_kd_diff': team1_stats.get('recent_avg_kd', 1.0) - team2_stats.get('recent_avg_kd', 1.0),
            'avg_acs_diff': team1_stats.get('avg_acs', 0) - team2_stats.get('avg_acs', 0),
            'recent_acs_diff': team1_stats.get('recent_avg_acs', 0) - team2_stats.get('recent_avg_acs', 0)
        }
        
        return features
    
    def create_comprehensive_features(self, team1_id, team2_id, api, all_teams):
        """Create a comprehensive feature set for match prediction."""
        # Fetch all necessary data
        team1_details = api.get_team_details(team1_id)
        team2_details = api.get_team_details(team2_id)
        
        team1_history = api.get_team_match_history(team1_id)
        team2_history = api.get_team_match_history(team2_id)
        
        team1_matches = self.data_processor.parse_match_data(team1_history, team1_details.get('name', ''))
        team2_matches = self.data_processor.parse_match_data(team2_history, team2_details.get('name', ''))
        
        # Fetch player statistics
        team1_player_stats = api.get_team_player_stats(team1_id)
        team2_player_stats = api.get_team_player_stats(team2_id)
        
        # Calculate all advanced statistics
        team1_stats = self.data_processor.calculate_team_stats(team1_matches)
        team2_stats = self.data_processor.calculate_team_stats(team2_matches)
        
        team1_trends = self.data_processor.analyze_performance_trends(team1_matches)
        team2_trends = self.data_processor.analyze_performance_trends(team2_matches)
        
        team1_opponent_quality = self.data_processor.analyze_opponent_quality(team1_matches, all_teams)
        team2_opponent_quality = self.data_processor.analyze_opponent_quality(team2_matches, all_teams)
        
        team1_event_performance = self.data_processor.analyze_event_performance(team1_matches)
        team2_event_performance = self.data_processor.analyze_event_performance(team2_matches)
        
        team1_player_metrics = self.data_processor.aggregate_player_metrics(team1_player_stats)
        team2_player_metrics = self.data_processor.aggregate_player_metrics(team2_player_stats)
        
        h2h_stats = self.data_processor.calculate_head_to_head(team1_matches, team2_details.get('name', ''), 
                                          team2_matches, team1_details.get('name', ''))
                        logger.info(f"Team 1: {team1_name}")
                logger.info(f"Team 2: {team2_name}")
                logger.info(f"\nPredicted Winner: {winner}")
                logger.info(f"Win Probability: {win_probability:.2%}")
            
            # Create prediction result dictionary
            prediction_result = {
                'match': f"{team1_name} vs {team2_name}",
                'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'predicted_winner': winner,
                'win_probability': float(win_probability),
                'team1_name': team1_name,
                'team2_name': team2_name,
                'team1_id': team1_id,
                'team2_id': team2_id,
                'features': features,
                'is_lan': is_lan
            }
            
            # Add to prediction history
            self.prediction_history.append(prediction_result)
            
            # Save prediction
            if save_results:
                self._save_prediction(prediction_result)
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def _save_prediction(self, prediction_result):
        """Save prediction result to file."""
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        
        # Clean team names for filename
        team1_name_clean = re.sub(r'[^\w\s]', '', prediction_result['team1_name']).replace(' ', '_')
        team2_name_clean = re.sub(r'[^\w\s]', '', prediction_result['team2_name']).replace(' ', '_')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{team1_name_clean}_vs_{team2_name_clean}_{timestamp}.json"
        
        with open(os.path.join(PREDICTIONS_DIR, filename), 'w') as f:
            json.dump(prediction_result, f, indent=4, default=str)
        
        logger.info(f"Prediction saved to {filename}")
    
    def visualize_prediction(self, prediction_result, show_plots=True, save_plots=True):
        """Generate visualizations for the prediction results."""
        if not prediction_result:
            logger.error("No prediction result to visualize")
            return
        
        # Extract data from prediction result
        team1_name = prediction_result['team1_name']
        team2_name = prediction_result['team2_name']
        
        features = prediction_result['features']
        win_probability = prediction_result['win_probability']
        predicted_winner = prediction_result['predicted_winner']
        
        # Set up plot style
        plt.style.use('ggplot')
        sns.set_palette("deep")
        
        # Create figure and axes for plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Match Prediction: {team1_name} vs {team2_name}", fontsize=16)
        
        # 1. Win Probability Plot
        axes[0, 0].bar([team1_name, team2_name], 
                       [win_probability if predicted_winner == team1_name else 1-win_probability,
                        win_probability if predicted_winner == team2_name else 1-win_probability],
                       color=['#1f77b4', '#ff7f0e'])
        axes[0, 0].set_title('Win Probability')
        axes[0, 0].set_ylabel('Probability')
        axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        # Add percentage labels
        for i, team in enumerate([team1_name, team2_name]):
            prob = win_probability if team == predicted_winner else 1-win_probability
            axes[0, 0].text(i, prob + 0.02, f'{prob:.1%}', ha='center', va='bottom')
        
        axes[0, 0].set_ylim(0, 1.1)
        
        # 2. Team Stats Comparison
        stats_to_compare = [
            ('win_rate', 'Win Rate'), 
            ('recent_form_5', 'Recent Form'), 
            ('avg_kd', 'KD Ratio'), 
            ('avg_acs', 'ACS')
        ]
        
        team1_values = []
        team2_values = []
        labels = []
        
        for key, label in stats_to_compare:
            team1_key = f'team1_{key}'
            team2_key = f'team2_{key}'
            
            if team1_key in features and team2_key in features:
                team1_values.append(features[team1_key])
                team2_values.append(features[team2_key])
                labels.append(label)
        
        # Convert win_rate and recent_form to percentages
        if 'win_rate' in [item[0] for item in stats_to_compare]:
            idx = [item[0] for item in stats_to_compare].index('win_rate')
            team1_values[idx] *= 100
            team2_values[idx] *= 100
        
        if 'recent_form_5' in [item[0] for item in stats_to_compare]:
            idx = [item[0] for item in stats_to_compare].index('recent_form_5')
            team1_values[idx] *= 100
            team2_values[idx] *= 100
        
        x = np.arange(len(labels))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, team1_values, width, label=team1_name)
        axes[0, 1].bar(x + width/2, team2_values, width, label=team2_name)
        axes[0, 1].set_title('Team Stats Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(labels)
        axes[0, 1].legend()
        
        # Add value labels on the bars
        for i, (v1, v2) in enumerate(zip(team1_values, team2_values)):
            if labels[i] in ['Win Rate', 'Recent Form']:
                axes[0, 1].text(i - width/2, v1 + 1, f'{v1:.1f}%', ha='center')
                axes[0, 1].text(i + width/2, v2 + 1, f'{v2:.1f}%', ha='center')
            else:
                axes[0, 1].text(i - width/2, v1 + 0.1, f'{v1:.2f}', ha='center')
                axes[0, 1].text(i + width/2, v2 + 0.1, f'{v2:.2f}', ha='center')
        
        # 3. Feature Importance (top 10)
        try:
            top_features = self.model_trainer.feature_importances.head(10)
            axes[1, 0].barh(top_features['Feature'], top_features['Importance'])
            axes[1, 0].set_title('Top 10 Most Important Features')
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].invert_yaxis()  # Highest importance at the top
        except:
            axes[1, 0].text(0.5, 0.5, 'Feature importance not available', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Feature Importance')
        
        # 4. Performance Metrics
        performance_metrics = [
            ('kd_diff', 'KD Differential'),
            ('acs_diff', 'ACS Differential'),
            ('recent_form_diff', 'Recent Form Diff'),
            ('win_rate_diff', 'Win Rate Diff')
        ]
        
        metric_values = []
        metric_labels = []
        colors = []
        
        for key, label in performance_metrics:
            if key in features:
                metric_values.append(features[key])
                metric_labels.append(label)
                # Positive value favors team1, negative favors team2
                colors.append('#1f77b4' if features[key] > 0 else '#ff7f0e')
        
        # Scale win_rate and form differences to percentages
        if 'win_rate_diff' in [item[0] for item in performance_metrics]:
            idx = [item[0] for item in performance_metrics].index('win_rate_diff')
            metric_values[idx] *= 100
        
        if 'recent_form_diff' in [item[0] for item in performance_metrics]:
            idx = [item[0] for item in performance_metrics].index('recent_form_diff')
            metric_values[idx] *= 100
        
        axes[1, 1].bar(metric_labels, metric_values, color=colors)
        axes[1, 1].set_title('Performance Differentials (Team1 - Team2)')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(metric_values):
            if metric_labels[i] in ['Win Rate Diff', 'Recent Form Diff']:
                axes[1, 1].text(i, v + (1 if v > 0 else -1), f'{v:.1f}%', ha='center')
            else:
                axes[1, 1].text(i, v + (0.1 if v > 0 else -0.1), f'{v:.2f}', ha='center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the figure title
        
        # Save plots
        if save_plots:
            os.makedirs(os.path.join(PREDICTIONS_DIR, 'visualizations'), exist_ok=True)
            
            # Clean team names for filename
            team1_name_clean = re.sub(r'[^\w\s]', '', team1_name).replace(' ', '_')
            team2_name_clean = re.sub(r'[^\w\s]', '', team2_name).replace(' ', '_')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{team1_name_clean}_vs_{team2_name_clean}_{timestamp}.png"
            
            plt.savefig(os.path.join(PREDICTIONS_DIR, 'visualizations', filename), dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to visualizations/{filename}")
        
        # Show plots
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def train_model(self, team_list=None, num_teams=20, method='advanced', save_model=True):
        """Train a prediction model using data from multiple teams."""
        if team_list is None and num_teams <= 0:
            logger.error("Either provide a team list or specify a positive number of teams")
            return False
        
        # If no team list is provided, get the top N teams
        if team_list is None:
            all_teams = self.api.get_all_teams()
            
            # Sort teams by ranking
            all_teams = sorted(all_teams, key=lambda x: x.get('ranking', 9999))
            
            # Take the top N teams
            team_list = all_teams[:num_teams]
            team_names = [team.get('name') for team in team_list]
            
            logger.info(f"Using top {num_teams} teams for training: {', '.join(team_names)}")
        
        # Create training dataset
        logger.info("Creating training dataset...")
        training_samples = []
        
        for team in tqdm(team_list, desc="Processing teams"):
            team_id = team.get('id') if isinstance(team, dict) else team
            team_name = team.get('name') if isinstance(team, dict) else None
            
            if team_name is None:
                team_details = self.api.get_team_details(team_id)
                if team_details:
                    team_name = team_details.get('name')
                else:
                    continue
            
            # Get match history
            team_history = self.api.get_team_match_history(team_id)
            if not team_history:
                continue
            
            # Parse match data
            team_matches = self.data_processor.parse_match_data(team_history, team_name)
            if not team_matches:
                continue
            
            logger.info(f"Creating training samples from {len(team_matches)} matches for {team_name}")
            
            # Create training samples from this team's matches
            team_samples = self.feature_engineering.create_training_data_from_match_history(
                team_matches, team_id, team_name, self.api, team_list
            )
            
            if team_samples is not None and not team_samples.empty:
                training_samples.append(team_samples)
                logger.info(f"Added {len(team_samples)} samples from {team_name}")
        
        if not training_samples:
            logger.error("Failed to create any training samples")
            return False
        
        # Combine all training samples
        training_data = pd.concat(training_samples, ignore_index=True)
        
        # Handle missing values
        training_data = self.feature_engineering.handle_missing_values(training_data)
        
        logger.info(f"Created training dataset with {len(training_data)} samples")
        
        # Train model based on specified method
        logger.info(f"Training {method} model...")
        
        if method == 'basic':
            model, scaler, feature_names, feature_importance = self.model_trainer.train_basic_model(training_data)
        elif method == 'ensemble':
            model, scaler, feature_names, feature_importance = self.model_trainer.train_ensemble_model(training_data)
        elif method == 'advanced':
            result = self.model_trainer.train_advanced_ml_model(training_data)
            if len(result) == 5:
                model, scaler, selector, feature_names, feature_importance = result
            else:
                model, scaler, feature_names, feature_importance = result
        elif method == 'deep':
            result = self.model_trainer.train_deep_learning_model(training_data)
            if len(result) == 6:
                model, rf_model, scaler, dl_indices, feature_names, feature_importance = result
            else:
                model, scaler, feature_names, feature_importance = result
        else:
            logger.error(f"Unknown training method: {method}")
            return False
        
        logger.info("Model training complete")
        self.model_loaded = True
        
        # Save model
        if save_model:
            model_prefix = self.model_trainer.save_model(f"valorant_predictor_{method}")
            logger.info(f"Model saved with prefix {model_prefix}")
        
        return True
    
    def load_model(self, model_prefix):
        """Load a previously trained model."""
        success = self.model_trainer.load_model(model_prefix)
        if success:
            self.model_loaded = True
            logger.info(f"Successfully loaded model {model_prefix}")
        else:
            logger.error(f"Failed to load model {model_prefix}")
        
        return success
    
    def update_model(self, team_list=None, num_teams=10, method='advanced', save_model=True):
        """Update the current model with new match data."""
        if not self.model_loaded:
            logger.error("No model loaded to update")
            return False
        
        # TODO: Implement proper model updating
        # For now, just train a new model
        return self.train_model(team_list, num_teams, method, save_model)
    
    def evaluate_prediction(self, prediction_result, actual_winner):
        """Evaluate a prediction against the actual match outcome."""
        if not prediction_result:
            return False
        
        predicted_winner = prediction_result['predicted_winner']
        team1_name = prediction_result['team1_name']
        team2_name = prediction_result['team2_name']
        
        # Convert to normalized names
        predicted_winner_norm = predicted_winner.lower().strip()
        actual_winner_norm = actual_winner.lower().strip()
        team1_name_norm = team1_name.lower().strip()
        team2_name_norm = team2_name.lower().strip()
        
        # Check if actual winner is one of the teams
        if actual_winner_norm != team1_name_norm and actual_winner_norm != team2_name_norm:
            logger.error(f"Actual winner '{actual_winner}' is not one of the teams")
            return False
        
        # Check if prediction was correct
        correct = (predicted_winner_norm == actual_winner_norm)
        
        # Create evaluation result
        evaluation = {
            'match': f"{team1_name} vs {team2_name}",
            'prediction_time': prediction_result['prediction_time'],
            'predicted_winner': predicted_winner,
            'actual_winner': actual_winner,
            'win_probability': prediction_result['win_probability'],
            'correct': correct,
            'features': prediction_result['features']
        }
        
        # Save evaluation
        self._save_evaluation(evaluation)
        
        logger.info(f"Prediction for {team1_name} vs {team2_name}: {'CORRECT' if correct else 'INCORRECT'}")
        logger.info(f"Predicted {predicted_winner} with {prediction_result['win_probability']:.2%} probability")
        logger.info(f"Actual winner: {actual_winner}")
        
        return evaluation
    
    def _save_evaluation(self, evaluation):
        """Save prediction evaluation to file."""
        os.makedirs(os.path.join(PREDICTIONS_DIR, 'evaluations'), exist_ok=True)
        
        # Clean team names for filename
        team1_name_clean = re.sub(r'[^\w\s]', '', evaluation['match'].split(' vs ')[0]).replace(' ', '_')
        team2_name_clean = re.sub(r'[^\w\s]', '', evaluation['match'].split(' vs ')[1]).replace(' ', '_')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{team1_name_clean}_vs_{team2_name_clean}_eval_{timestamp}.json"
        
        with open(os.path.join(PREDICTIONS_DIR, 'evaluations', filename), 'w') as f:
            json.dump(evaluation, f, indent=4, default=str)
        
        logger.info(f"Evaluation saved to evaluations/{filename}")
    
    def learn_from_outcome(self, prediction_result, actual_winner, method='adaptive'):
        """Update model based on prediction outcomes."""
        if not self.model_loaded:
            logger.error("No model loaded to update")
            return False
        
        if not prediction_result:
            logger.error("No prediction result provided")
            return False
        
        # Get evaluation
        evaluation = self.evaluate_prediction(prediction_result, actual_winner)
        if not evaluation:
            return False
        
        # For now, just log this. In the future, implement adaptive learning
        logger.info(f"Learning from outcome: {'CORRECT' if evaluation['correct'] else 'INCORRECT'} prediction")
        
        # TODO: Implement proper learning from outcomes
        # This would involve retraining or fine-tuning the model with this new data point
        
        return True
    
    def predict_future_matches(self, upcoming_matches, save_results=True, visualize=False):
        """Predict outcomes for a list of upcoming matches."""
        if not self.model_loaded:
            logger.error("No model loaded. Please load or train a model first.")
            return []
        
        predictions = []
        
        for match in upcoming_matches:
            team1_name = match.get('team1')
            team2_name = match.get('team2')
            team1_region = match.get('team1_region')
            team2_region = match.get('team2_region')
            is_lan = match.get('is_lan')
            
            prediction = self.predict_match(
                team1_name, team2_name, team1_region, team2_region,
                save_results=save_results, is_lan=is_lan, verbose=False
            )
            
            if prediction:
                predictions.append(prediction)
                
                if visualize:
                    self.visualize_prediction(prediction, show_plots=False, save_plots=True)
        
        return predictions
    
    def backtest_model(self, match_data, visualize_results=True):
        """Backtest the prediction model on historical match data."""
        if not self.model_loaded:
            logger.error("No model loaded. Please load or train a model first.")
            return None
        
        if not match_data:
            logger.error("No match data provided for backtesting")
            return None
        
        results = {
            'total': 0,
            'correct': 0,
            'accuracy': 0,
            'avg_confidence': 0,
            'brier_score': 0,
            'predictions': []
        }
        
        confidence_bins = {
            '50-60%': {'total': 0, 'correct': 0, 'accuracy': 0},
            '60-70%': {'total': 0, 'correct': 0, 'accuracy': 0},
            '70-80%': {'total': 0, 'correct': 0, 'accuracy': 0},
            '80-90%': {'total': 0, 'correct': 0, 'accuracy': 0},
            '90-100%': {'total': 0, 'correct': 0, 'accuracy': 0}
        }
        
        # Process each match
        for match in tqdm(match_data, desc="Backtesting"):
            team1_name = match.get('team1')
            team2_name = match.get('team2')
            actual_winner = match.get('winner')
            is_lan = match.get('is_lan')
            
            # Skip if missing required data
            if not team1_name or not team2_name or not actual_winner:
                continue
            
            # Make prediction
            prediction = self.predict_match(
                team1_name, team2_name, 
                team1_region=match.get('team1_region'), 
                team2_region=match.get('team2_region'),
                save_results=False,
                is_lan=is_lan,
                verbose=False
            )
            
            if not prediction:
                continue
                
            # Check if prediction was correct
            predicted_winner = prediction['predicted_winner']
            win_probability = prediction['win_probability']
            
            correct = (predicted_winner.lower().strip() == actual_winner.lower().strip())
            
            # Calculate Brier score (prediction calibration metric)
            # Brier score = (f - o)² where f is forecast probability and o is outcome (1 if correct, 0 if not)
            p = win_probability if predicted_winner == actual_winner else 1 - win_probability
            brier_score = (p - 1) ** 2
            
            # Add to results
            prediction_result = {
                'match': f"{team1_name} vs {team2_name}",
                'predicted_winner': predicted_winner,
                'actual_winner': actual_winner,
                'win_probability': win_probability,
                'correct': correct,
                'brier_score': brier_score
            }
            
            results['predictions'].append(prediction_result)
            results['total'] += 1
            results['correct'] += 1 if correct else 0
            results['avg_confidence'] += win_probability
            results['brier_score'] += brier_score
            
            # Add to confidence bins
            conf_pct = win_probability * 100
            if 50 <= conf_pct < 60:
                bin_key = '50-60%'
            elif 60 <= conf_pct < 70:
                bin_key = '60-70%'
            elif 70 <= conf_pct < 80:
                bin_key = '70-80%'
            elif 80 <= conf_pct < 90:
                bin_key = '80-90%'
            else:
                bin_key = '90-100%'
            
            confidence_bins[bin_key]['total'] += 1
            confidence_bins[bin_key]['correct'] += 1 if correct else 0
        
        # Calculate final metrics
        if results['total'] > 0:
            results['accuracy'] = results['correct'] / results['total']
            results['avg_confidence'] /= results['total']
            results['brier_score'] /= results['total']
            
            # Calculate accuracy by confidence bin
            for bin_key, bin_data in confidence_bins.items():
                if bin_data['total'] > 0:
                    bin_data['accuracy'] = bin_data['correct'] / bin_data['total']
            
            results['confidence_bins'] = confidence_bins
        
        # Log results
        logger.info("\n============ BACKTEST RESULTS ============")
        logger.info(f"Total matches: {results['total']}")
        logger.info(f"Correct predictions: {results['correct']}")
        logger.info(f"Accuracy: {results['accuracy']:.2%}")
        logger.info(f"Average confidence: {results['avg_confidence']:.2%}")
        logger.info(f"Brier score: {results['brier_score']:.4f} (lower is better)")
        
        # Log accuracy by confidence bin
        logger.info("\nAccuracy by confidence bin:")
        for bin_key, bin_data in confidence_bins.items():
            if bin_data['total'] > 0:
                logger.info(f"{bin_key}: {bin_data['accuracy']:.2%} ({bin_data['correct']}/{bin_data['total']})")
        
        # Visualize results
        if visualize_results:
            self._visualize_backtest_results(results)
        
        # Save backtest results
        self._save_backtest_results(results)
        
        return results
    
    def _visualize_backtest_results(self, results):
        """Visualize backtest results with multiple plots."""
        if not results or results['total'] == 0:
            logger.error("No valid backtest results to visualize")
            return
        
        # Create directory for backtest visualizations
        os.makedirs(os.path.join(BACKTEST_DIR, 'visualizations'), exist_ok=True)
        
        # Set up plot style
        plt.style.use('ggplot')
        sns.set_palette("deep")
        
        # Create figure and axes for plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Backtest Results ({results['total']} matches)", fontsize=16)
        
        # 1. Overall Accuracy Plot
        axes[0, 0].bar(['Accuracy'], [results['accuracy']], color='blue')
        axes[0, 0].set_title('Overall Prediction Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].text(0, results['accuracy'] + 0.02, f"{results['accuracy']:.2%}", ha='center')
        
        # 2. Accuracy by Confidence Bin
        if 'confidence_bins' in results:
            bin_labels = []
            bin_accuracies = []
            bin_counts = []
            
            for bin_key, bin_data in sorted(results['confidence_bins'].items()):
                if bin_data['total'] > 0:
                    bin_labels.append(bin_key)
                    bin_accuracies.append(bin_data['accuracy'])
                    bin_counts.append(bin_data['total'])
            
            # Plot accuracy by bin
            bar_plot = axes[0, 1].bar(bin_labels, bin_accuracies, color='orange')
            axes[0, 1].set_title('Accuracy by Confidence Bin')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_ylim(0, 1)
            
            # Add count labels
            for i, count in enumerate(bin_counts):
                axes[0, 1].text(i, 0.03, f"n={count}", ha='center', color='black')
            
            # Add accuracy percentage labels
            for i, v in enumerate(bin_accuracies):
                axes[0, 1].text(i, v + 0.02, f"{v:.2%}", ha='center')
            
            # Add expected accuracy line
            bin_centers = [int(bin_key.split('-')[0]) + 5 for bin_key in bin_labels]
            axes[0, 1].plot(bin_labels, [x/100 for x in bin_centers], 'k--', alpha=0.5, label='Expected')
            axes[0, 1].legend()
        
        # 3. Prediction Distribution
        if 'predictions' in results:
            # Get confidence for each prediction
            confidences = [pred['win_probability'] for pred in results['predictions']]
            
            # Plot histogram
            axes[1, 0].hist(confidences, bins=10, range=(0.5, 1.0), color='green', alpha=0.7)
            axes[1, 0].set_title('Distribution of Prediction Confidence')
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_ylabel('Count')
        
        # 4. Brier Score (Calibration)
        if 'predictions' in results:
            correct_preds = [p for p in results['predictions'] if p['correct']]
            incorrect_preds = [p for p in results['predictions'] if not p['correct']]
            
            # Calculate average Brier score for correct and incorrect predictions
            correct_brier = np.mean([p['brier_score'] for p in correct_preds]) if correct_preds else 0
            incorrect_brier = np.mean([p['brier_score'] for p in incorrect_preds]) if incorrect_preds else 0
            overall_brier = results['brier_score']
            
            # Plot Brier scores
            axes[1, 1].bar(['Overall', 'Correct Predictions', 'Incorrect Predictions'], 
                          [overall_brier, correct_brier, incorrect_brier],
                          color=['blue', 'green', 'red'])
            axes[1, 1].set_title('Brier Score (Lower is Better)')
            axes[1, 1].set_ylabel('Score')
            
            # Add score labels
            for i, v in enumerate([overall_brier, correct_brier, incorrect_brier]):
                axes[1, 1].text(i, v + 0.01, f"{v:.4f}", ha='center')
        
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the figure title
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_results_{timestamp}.png"
        plt.savefig(os.path.join(BACKTEST_DIR, 'visualizations', filename), dpi=300, bbox_inches='tight')
        logger.info(f"Backtest visualization saved to {BACKTEST_DIR}/visualizations/{filename}")
        
        plt.close()
    
    def _save_backtest_results(self, results):
        """Save backtest results to file."""
        os.makedirs(BACKTEST_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_results_{timestamp}.json"
        
        # Remove large feature data from predictions to keep file size reasonable
        if 'predictions' in results:
            for pred in results['predictions']:
                if 'features' in pred:
                    del pred['features']
        
        with open(os.path.join(BACKTEST_DIR, filename), 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        logger.info(f"Backtest results saved to {filename}")
    
    def collect_historical_matches(self, team_list=None, num_teams=20, min_date=None, max_date=None):
        """Collect historical matches for backtesting."""
        if team_list is None and num_teams <= 0:
            logger.error("Either provide a team list or specify a positive number of teams")
            return []
        
        # If no team list is provided, get the top N teams
        if team_list is None:
            all_teams = self.api.get_all_teams()
            
            # Sort teams by ranking
            all_teams = sorted(all_teams, key=lambda x: x.get('ranking', 9999))
            
            # Take the top N teams
            team_list = all_teams[:num_teams]
            team_names = [team.get('name') for team in team_list]
            
            logger.info(f"Using top {num_teams} teams for historical data: {', '.join(team_names)}")
        
        # Process date range
        if min_date is not None:
            if isinstance(min_date, str):
                min_date = datetime.strptime(min_date, '%Y-%m-%d')
        else:
            # Default to 6 months ago
            min_date = datetime.now() - timedelta(days=180)
        
        if max_date is not None:
            if isinstance(max_date, str):
                max_date = datetime.strptime(max_date, '%Y-%m-%d')
        else:
            # Default to today
            max_date = datetime.now()
        
        logger.info(f"Collecting matches from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        
        # Collect historical matches
        historical_matches = []
        
        for team in tqdm(team_list, desc="Collecting historical matches"):
            team_id = team.get('id') if isinstance(team, dict) else team
            team_name = team.get('name') if isinstance(team, dict) else None
            
            if team_name is None:
                team_details = self.api.get_team_details(team_id)
                if team_details:
                    team_name = team_details.get('name')
                else:
                    continue
            
            # Get match history
            team_history = self.api.get_team_match_history(team_id)
            if not team_history:
                continue
            
            # Parse match data
            team_matches = self.data_processor.parse_match_data(team_history, team_name)
            if not team_matches:
                continue
            
            # Filter by date range
            for match in team_matches:
                match_date_str = match.get('date', '')
                if not match_date_str:
                    continue
                
                try:
                    match_date = datetime.strptime(match_date_str, '%Y-%m-%d')
                    if min_date <= match_date <= max_date:
                        # Create historical match entry
                        historical_match = {
                            'team1': team_name,
                            'team2': match.get('opponent_name', ''),
                            'winner': team_name if match.get('team_won', False) else match.get('opponent_name', ''),
                            'date': match_date_str,
                            'event': match.get('event', ''),
                            'is_lan': 'lan' in str(match.get('event', '')).lower() or 'masters' in str(match.get('event', '')).lower()
                        }
                        
                        # Only add if both teams are known
                        if historical_match['team1'] and historical_match['team2'] and historical_match['winner']:
                            historical_matches.append(historical_match)
                except ValueError:
                    continue
        
        # Remove duplicates (same match might appear in both teams' history)
        unique_matches = []
        seen_matches = set()
        
        for match in historical_matches:
            # Create a unique identifier for the match
            teams = sorted([match['team1'], match['team2']])
            match_id = f"{teams[0]}_{teams[1]}_{match['date']}"
            
            if match_id not in seen_matches:
                seen_matches.add(match_id)
                unique_matches.append(match)
        
        logger.info(f"Collected {len(unique_matches)} unique historical matches")
        
        # Save historical matches
        os.makedirs(os.path.join(DATA_DIR, 'historical_matches'), exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"historical_matches_{timestamp}.json"
        
        with open(os.path.join(DATA_DIR, 'historical_matches', filename), 'w') as f:
            json.dump(unique_matches, f, indent=4, default=str)
        
        logger.info(f"Historical matches saved to {DATA_DIR}/historical_matches/{filename}")
        
        return unique_matches


def main():
    parser = argparse.ArgumentParser(description='Enhanced Valorant Match Prediction System')
    
    # Command-specific arguments
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Predict match command
    predict_parser = subparsers.add_parser('predict', help='Predict the outcome of a match')
    predict_parser.add_argument('--team1', type=str, required=True, help='Name of the first team')
    predict_parser.add_argument('--team2', type=str, required=True, help='Name of the second team')
    predict_parser.add_argument('--team1-region', type=str, help='Region of the first team (na, eu, etc.)')
    predict_parser.add_argument('--team2-region', type=str, help='Region of the second team (na, eu, etc.)')
    predict_parser.add_argument('--model', type=str, default='best', help='Model type to use (best, random_forest, ensemble, deep_learning)')
    predict_parser.add_argument('--visualize', action='store_true', help='Generate visualization of the prediction')
    predict_parser.add_argument('--lan', action='store_true', help='Indicate if the match is played on LAN')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train a prediction model')
    train_parser.add_argument('--num-teams', type=int, default=20, help='Number of top teams to use for training')
    train_parser.add_argument('--method', type=str, default='advanced', choices=['basic', 'ensemble', 'advanced', 'deep'], help='Training method to use')
    train_parser.add_argument('--save', action='store_true', default=True, help='Save the trained model')
    
    # Load model command
    load_parser = subparsers.add_parser('load', help='Load a previously trained model')
    load_parser.add_argument('--model-prefix', type=str, required=True, help='Prefix of the model to load')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest the prediction model on historical matches')
    backtest_parser.add_argument('--num-teams', type=int, default=20, help='Number of top teams to use for collecting historical matches')
    backtest_parser.add_argument('--min-date', type=str, help='Minimum date for historical matches (YYYY-MM-DD)')
    backtest_parser.add_argument('--max-date', type=str, help='Maximum date for historical matches (YYYY-MM-DD)')
    backtest_parser.add_argument('--visualize', action='store_true', default=True, help='Generate visualization of backtest results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create prediction system
    system = PredictionSystem()
    
    # Execute command
    if args.command == 'predict':
        # Load model
        model_loaded = False
        
        # Try to load the latest model
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_info.json')]
        if model_files:
            # Sort by timestamp (newest first)
            model_files.sort(reverse=True)
            latest_model = model_files[0].replace('_info.json', '')
            model_loaded = system.load_model(latest_model)
        
        if not model_loaded:
            # Train a basic model
            logger.info("No model loaded. Training a basic model...")
            system.train_model(num_teams=10, method='basic')
        
        # Make prediction
        prediction = system.predict_match(
            args.team1, args.team2,
            team1_region=args.team1_region,
            team2_region=args.team2_region,
            model_type=args.model,
            is_lan=args.lan
        )
        
        if prediction and args.visualize:
            system.visualize_prediction(prediction)
    
    elif args.command == 'train':
        # Train model
        system.train_model(num_teams=args.num_teams, method=args.method, save_model=args.save)
    
    elif args.command == 'load':
        # Load model
        system.load_model(args.model_prefix)
    
    elif args.command == 'backtest':
        # Load model
        model_loaded = False
        
        # Try to load the latest model
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_info.json')]
        if model_files:
            # Sort by timestamp (newest first)
            model_files.sort(reverse=True)
            latest_model = model_files[0].replace('_info.json', '')
            model_loaded = system.load_model(latest_model)
        
        if not model_loaded:
            # Train a basic model
            logger.info("No model loaded. Training a basic model...")
            system.train_model(num_teams=10, method='basic')
        
        # Collect historical matches
        historical_matches = system.collect_historical_matches(
            num_teams=args.num_teams,
            min_date=args.min_date,
            max_date=args.max_date
        )
        
        # Run backtest
        system.backtest_model(historical_matches, visualize_results=args.visualize)
    
    else:
        # Interactive mode
        print("\nEnhanced Valorant Match Prediction System")
        print("========================================")
        
        # Try to load the latest model
        model_loaded = False
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_info.json')]
        if model_files:
            # Sort by timestamp (newest first)
            model_files.sort(reverse=True)
            latest_model = model_files[0].replace('_info.json', '')
            model_loaded = system.load_model(latest_model)
            
            if model_loaded:
                print(f"Loaded model: {latest_model}")
            else:
                print("Failed to load the latest model.")
        
        if not model_loaded:
            print("\nNo model loaded. You need to train a model first.")
            train_now = input("Would you like to train a model now? (y/n): ")
            
            if train_now.lower() == 'y':
                num_teams = int(input("Number of teams to use for training (recommended: 20): ") or "20")
                method_options = {
                    1: 'basic',
                    2: 'ensemble',
                    3: 'advanced',
                    4: 'deep'
                }
                
                print("\nTraining methods:")
                print("1. Basic - Simple Random Forest (fastest)")
                print("2. Ensemble - Multiple models ensemble")
                print("3. Advanced - Optimized ML pipeline (recommended)")
                print("4. Deep Learning - Neural network (requires more data)")
                
                method_choice = int(input("Select method (1-4): ") or "3")
                method = method_options.get(method_choice, 'advanced')
                
                print(f"\nTraining model using {method} method with {num_teams} teams...")
                system.train_model(num_teams=num_teams, method=method)
            else:
                print("Exiting. You need to train a model to use the system.")
                return
        
        # Main menu
        while True:
            print("\nMain Menu:")
            print("1. Predict a match")
            print("2. Train/update model")
            print("3. Backtest model")
            print("4. Exit")
            
            choice = input("\nSelect an option (1-4): ")
            
            if choice == '1':
                # Predict match
                team1 = input("\nEnter the name of the first team: ")
                team1_region = input("Enter region for the first team (optional): ") or None
                team2 = input("Enter the name of the second team: ")
                team2_region = input("Enter region for the second team (optional): ") or None
                is_lan = input("Is this a LAN match? (y/n): ").lower() == 'y'
                
                prediction = system.predict_match(
                    team1, team2,
                    team1_region=team1_region,
                    team2_region=team2_region,
                    is_lan=is_lan
                )
                
                if prediction:
                    visualize = input("Generate visualization? (y/n): ").lower() == 'y'
                    if visualize:
                        system.visualize_prediction(prediction)
            
            elif choice == '2':
                # Train/update model
                num_teams = int(input("Number of teams to use for training (recommended: 20): ") or "20")
                method_options = {
                    1: 'basic',
                    2: 'ensemble',
                    3: 'advanced',
                    4: 'deep'
                }
                
                print("\nTraining methods:")
                print("1. Basic - Simple Random Forest (fastest)")
                print("2. Ensemble - Multiple models ensemble")
                print("3. Advanced - Optimized ML pipeline (recommended)")
                print("4. Deep Learning - Neural network (requires more data)")
                
                method_choice = int(input("Select method (1-4): ") or "3")
                method = method_options.get(method_choice, 'advanced')
                
                print(f"\nTraining model using {method} method with {num_teams} teams...")
                system.train_model(num_teams=num_teams, method=method)
            
            elif choice == '3':
                # Backtest model
                num_teams = int(input("Number of teams to collect matches from (recommended: 20): ") or "20")
                min_date = input("Minimum date for historical matches (YYYY-MM-DD, leave empty for 6 months ago): ") or None
                max_date = input("Maximum date for historical matches (YYYY-MM-DD, leave empty for today): ") or None
                
                print(f"\nCollecting historical matches from {num_teams} teams...")
                historical_matches = system.collect_historical_matches(
                    num_teams=num_teams,
                    min_date=min_date,
                    max_date=max_date
                )
                
                if historical_matches:
                    print(f"Collected {len(historical_matches)} historical matches.")
                    print("\nRunning backtest...")
                    system.backtest_model(historical_matches)
                else:
                    print("No historical matches found.")
            
            elif choice == '4':
                # Exit
                print("Exiting. Thank you for using the Valorant Match Prediction System!")
                break
            
            else:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()                