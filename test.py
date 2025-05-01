import requests
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import pickle
import time
import re
from tqdm import tqdm
import seaborn as sns

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Add this import at the top
from tensorflow.keras.models import load_model

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
    # for team in teams_data['data']:
    #     if team_name.lower() in team['name'].lower() or team['name'].lower() in team_name.lower():
    #         print(f"Found partial match: {team['name']} (ID: {team['id']})")
    #         return team['id']
    
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
        return {}
    
    # Process the data
    return extract_map_statistics(team_stats_data)

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
    events_data = fetch_api_data("events", {"limit": 100})
    
    if not events_data:
        return []
    
    return events_data.get('data', [])

def parse_match_data(match_history, team_name):
    """Parse match history data for a team with correct team tag assignment."""
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

def collect_team_data(team_limit=100, include_player_stats=True, include_economy=True, include_maps=False):
    """Collect data for all teams to use in training and evaluation."""
    print("\n========================================================")
    print("COLLECTING TEAM DATA")
    print("========================================================")
    print(f"Including player stats: {include_player_stats}")
    print(f"Including economy data: {include_economy}")
    print(f"Including map data: {include_maps}")
    
    # Fetch all teams
    teams_response = requests.get(f"{API_URL}/teams?limit={team_limit}")
    if teams_response.status_code != 200:
        print(f"Error fetching teams: {teams_response.status_code}")
        return {}
    
    teams_data = teams_response.json()
    
    if 'data' not in teams_data:
        print("No teams data found.")
        return {}
    
    # Select teams based on ranking or use a limited sample
    top_teams = []
    for team in teams_data['data']:
        if 'ranking' in team and team['ranking'] and team['ranking'] <= 50:
            top_teams.append(team)
    
    # If no teams with rankings were found, just take the first N teams
    if not top_teams:
        print(f"No teams with rankings found. Using the first {min(150, team_limit)} teams instead.")
        top_teams = teams_data['data'][:min(150, team_limit)]
    
    print(f"Selected {len(top_teams)} teams for data collection.")
    
    # Collect match data for each team
    team_data_collection = {}
    
    # Track data availability counts
    economy_data_count = 0
    player_stats_count = 0
    map_stats_count = 0
    
    for team in tqdm(top_teams, desc="Collecting team data"):
        team_id = team['id']
        team_name = team['name']
        
        # Get team tag for economy data matching
        team_details, team_tag = fetch_team_details(team_id)
        
        team_history = fetch_team_match_history(team_id)
        if not team_history:
            continue
            
        team_matches = parse_match_data(team_history, team_name)
        
        # Skip teams with no match data
        if not team_matches:
            continue
        
        # Add team tag and ID to all matches for data extraction
        for match in team_matches:
            match['team_tag'] = team_tag
            match['team_id'] = team_id
            match['team_name'] = team_name  # Always set team_name for fallback
        
        # Fetch player stats if requested
        team_player_stats = None
        if include_player_stats:
            team_player_stats = fetch_team_player_stats(team_id)
            if team_player_stats:
                player_stats_count += 1
        
        # Calculate team stats with appropriate features
        team_stats = calculate_team_stats(team_matches, team_player_stats, include_economy=include_economy)
        
        # Check if we got economy data
        if include_economy and 'pistol_win_rate' in team_stats and team_stats['pistol_win_rate'] > 0:
            economy_data_count += 1
        
        # Store team tag and ID in the stats
        team_stats['team_tag'] = team_tag
        team_stats['team_name'] = team_name
        team_stats['team_id'] = team_id
        
        # Fetch and add map statistics if requested
        if include_maps:
            map_stats = fetch_team_map_statistics(team_id)
            if map_stats:
                team_stats['map_statistics'] = map_stats
                map_stats_count += 1
        
        # Extract additional analyses
        team_stats['map_performance'] = extract_map_performance(team_matches)
        team_stats['tournament_performance'] = extract_tournament_performance(team_matches)
        team_stats['performance_trends'] = analyze_performance_trends(team_matches)
        team_stats['opponent_quality'] = analyze_opponent_quality(team_matches, team_id)
        
        # Add team matches to stats object
        team_stats['matches'] = team_matches
        
        # Add to collection
        team_data_collection[team_name] = team_stats
    
    print(f"\nCollected data for {len(team_data_collection)} teams:")
    print(f"  - Teams with economy data: {economy_data_count}")
    print(f"  - Teams with player stats: {player_stats_count}")
    print(f"  - Teams with map stats: {map_stats_count}")
    
    return team_data_collection

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
            
            ''' PART2'''

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
    """Prepare data for the ML model by creating symmetrical feature vectors."""
    if not team1_stats or not team2_stats:
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
        
        # Team rating (might be different from avg player rating)
        if 'team_rating' in team1_stats and 'team_rating' in team2_stats:
            features['team_rating_diff'] = team1_stats.get('team_rating', 0) - team2_stats.get('team_rating', 0)
            features['better_team_rating_team1'] = 1 if team1_stats.get('team_rating', 0) > team2_stats.get('team_rating', 0) else 0
            features['avg_team_rating'] = (team1_stats.get('team_rating', 0) + team2_stats.get('team_rating', 0)) / 2
        
        # ACS (Average Combat Score)
        features['acs_diff'] = team1_stats.get('avg_player_acs', 0) - team2_stats.get('avg_player_acs', 0)
        features['better_acs_team1'] = 1 if team1_stats.get('avg_player_acs', 0) > team2_stats.get('avg_player_acs', 0) else 0
        features['avg_acs'] = (team1_stats.get('avg_player_acs', 0) + team2_stats.get('avg_player_acs', 0)) / 2
        
        # K/D Ratio
        features['kd_diff'] = team1_stats.get('avg_player_kd', 0) - team2_stats.get('avg_player_kd', 0)
        features['better_kd_team1'] = 1 if team1_stats.get('avg_player_kd', 0) > team2_stats.get('avg_player_kd', 0) else 0
        features['avg_kd'] = (team1_stats.get('avg_player_kd', 0) + team2_stats.get('avg_player_kd', 0)) / 2
        
        # KAST (Kill, Assist, Survive, Trade)
        features['kast_diff'] = team1_stats.get('avg_player_kast', 0) - team2_stats.get('avg_player_kast', 0)
        features['better_kast_team1'] = 1 if team1_stats.get('avg_player_kast', 0) > team2_stats.get('avg_player_kast', 0) else 0
        features['avg_kast'] = (team1_stats.get('avg_player_kast', 0) + team2_stats.get('avg_player_kast', 0)) / 2
        
        # ADR (Average Damage per Round)
        features['adr_diff'] = team1_stats.get('avg_player_adr', 0) - team2_stats.get('avg_player_adr', 0)
        features['better_adr_team1'] = 1 if team1_stats.get('avg_player_adr', 0) > team2_stats.get('avg_player_adr', 0) else 0
        features['avg_adr'] = (team1_stats.get('avg_player_adr', 0) + team2_stats.get('avg_player_adr', 0)) / 2
        
        # Headshot percentage
        features['headshot_diff'] = team1_stats.get('avg_player_headshot', 0) - team2_stats.get('avg_player_headshot', 0)
        features['better_headshot_team1'] = 1 if team1_stats.get('avg_player_headshot', 0) > team2_stats.get('avg_player_headshot', 0) else 0
        features['avg_headshot'] = (team1_stats.get('avg_player_headshot', 0) + team2_stats.get('avg_player_headshot', 0)) / 2
        
        # Star player rating
        features['star_player_diff'] = team1_stats.get('star_player_rating', 0) - team2_stats.get('star_player_rating', 0)
        features['better_star_player_team1'] = 1 if team1_stats.get('star_player_rating', 0) > team2_stats.get('star_player_rating', 0) else 0
        features['avg_star_player'] = (team1_stats.get('star_player_rating', 0) + team2_stats.get('star_player_rating', 0)) / 2
        
        # Team consistency (star player vs. worst player)
        features['consistency_diff'] = team1_stats.get('team_consistency', 0) - team2_stats.get('team_consistency', 0)
        features['better_consistency_team1'] = 1 if team1_stats.get('team_consistency', 0) > team2_stats.get('team_consistency', 0) else 0
        features['avg_consistency'] = (team1_stats.get('team_consistency', 0) + team2_stats.get('team_consistency', 0)) / 2
        
        # First Kill / First Death ratio
        features['fk_fd_diff'] = team1_stats.get('fk_fd_ratio', 0) - team2_stats.get('fk_fd_ratio', 0)
        features['better_fk_fd_team1'] = 1 if team1_stats.get('fk_fd_ratio', 0) > team2_stats.get('fk_fd_ratio', 0) else 0
        features['avg_fk_fd'] = (team1_stats.get('fk_fd_ratio', 0) + team2_stats.get('fk_fd_ratio', 0)) / 2
    
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
        
        # Eco win rate
        features['eco_win_rate_diff'] = team1_stats.get('eco_win_rate', 0) - team2_stats.get('eco_win_rate', 0)
        features['better_eco_team1'] = 1 if team1_stats.get('eco_win_rate', 0) > team2_stats.get('eco_win_rate', 0) else 0
        features['avg_eco_win_rate'] = (team1_stats.get('eco_win_rate', 0) + team2_stats.get('eco_win_rate', 0)) / 2
        
        # Full-buy win rate
        features['full_buy_win_rate_diff'] = team1_stats.get('full_buy_win_rate', 0) - team2_stats.get('full_buy_win_rate', 0)
        features['better_full_buy_team1'] = 1 if team1_stats.get('full_buy_win_rate', 0) > team2_stats.get('full_buy_win_rate', 0) else 0
        features['avg_full_buy_win_rate'] = (team1_stats.get('full_buy_win_rate', 0) + team2_stats.get('full_buy_win_rate', 0)) / 2
        
        # Economy efficiency
        features['economy_efficiency_diff'] = team1_stats.get('economy_efficiency', 0) - team2_stats.get('economy_efficiency', 0)
        features['better_economy_efficiency_team1'] = 1 if team1_stats.get('economy_efficiency', 0) > team2_stats.get('economy_efficiency', 0) else 0
        features['avg_economy_efficiency'] = (team1_stats.get('economy_efficiency', 0) + team2_stats.get('economy_efficiency', 0)) / 2
    
    #----------------------------------------
    # 4. MAP STATS
    #----------------------------------------
    if ('map_statistics' in team1_stats and 'map_statistics' in team2_stats and
        team1_stats['map_statistics'] and team2_stats['map_statistics']):
        
        # Find common maps
        team1_maps = set(team1_stats['map_statistics'].keys())
        team2_maps = set(team2_stats['map_statistics'].keys())
        common_maps = team1_maps.intersection(team2_maps)
        
        # Add general map statistics
        features['common_maps_count'] = len(common_maps)
        features['team1_map_pool_size'] = len(team1_maps)
        features['team2_map_pool_size'] = len(team2_maps)
        
        # Prepare map-specific comparisons
        for map_name in common_maps:
            t1_map = team1_stats['map_statistics'][map_name]
            t2_map = team2_stats['map_statistics'][map_name]
            
            # Clean map name for feature naming
            map_key = map_name.replace(' ', '_').lower()
            
            # Win percentage comparison
            features[f'{map_key}_win_rate_diff'] = t1_map['win_percentage'] - t2_map['win_percentage']
            features[f'better_{map_key}_team1'] = 1 if t1_map['win_percentage'] > t2_map['win_percentage'] else 0
            
            # Side performance comparison
            if 'side_preference' in t1_map and 'side_preference' in t2_map:
                features[f'{map_key}_side_pref_diff'] = (1 if t1_map['side_preference'] == 'Attack' else -1) - (1 if t2_map['side_preference'] == 'Attack' else -1)
                features[f'{map_key}_atk_win_rate_diff'] = t1_map['atk_win_rate'] - t2_map['atk_win_rate']
                features[f'{map_key}_def_win_rate_diff'] = t1_map['def_win_rate'] - t2_map['def_win_rate']
    
    #----------------------------------------
    # 5. HEAD-TO-HEAD STATS
    #----------------------------------------
    # Add head-to-head statistics if available
    team2_name = team2_stats.get('team_name', '')
    if 'opponent_stats' in team1_stats and team2_name in team1_stats['opponent_stats']:
        h2h_stats = team1_stats['opponent_stats'][team2_name]
        
        # Add head-to-head metrics
        features['h2h_win_rate'] = h2h_stats.get('win_rate', 0.5)  # From team1's perspective
        features['h2h_matches'] = h2h_stats.get('matches', 0)
        features['h2h_score_diff'] = h2h_stats.get('score_differential', 0)
        
        # Create binary indicators
        features['h2h_advantage_team1'] = 1 if features['h2h_win_rate'] > 0.5 else 0
        features['h2h_significant'] = 1 if features['h2h_matches'] >= 3 else 0
    else:
        # Default values if no H2H data
        features['h2h_win_rate'] = 0.5
        features['h2h_matches'] = 0
        features['h2h_score_diff'] = 0
        features['h2h_advantage_team1'] = 0
        features['h2h_significant'] = 0
    
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
    
    # Clean up non-numeric values
    for key, value in list(features.items()):
        if not isinstance(value, (int, float)):
            del features[key]
    
    return features

def build_training_dataset(team_data_collection):
    """Build a dataset for model training from team data collection."""
    X = []  # Feature vectors
    y = []  # Labels (1 if team1 won, 0 if team2 won)
    
    print(f"Building training dataset from {len(team_data_collection)} teams...")
    
    # Process all teams' matches
    for team_name, team_data in team_data_collection.items():
        matches = team_data.get('matches', [])
        if not matches:
            continue
            
        for match in matches:
            # Get opponent name
            opponent_name = match.get('opponent_name')
            
            # Skip if we don't have data for the opponent
            if opponent_name not in team_data_collection:
                continue
            
            # Get stats for both teams
            team1_stats = team_data
            team2_stats = team_data_collection[opponent_name]
            
            # Prepare feature vector
            features = prepare_data_for_model(team1_stats, team2_stats)
            
            if features:
                X.append(features)
                y.append(1 if match.get('team_won', False) else 0)
    
    print(f"Created {len(X)} training samples from match data")
    return X, y

#-------------------------------------------------------------------------
# MODEL TRAINING AND EVALUATION
#-------------------------------------------------------------------------

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
    """Train model using k-fold cross-validation for robust performance."""
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
        

        '''PART 3'''

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
# BETTING FUNCTIONS - NEW SECTION
#-------------------------------------------------------------------------

def decimal_to_american(decimal_odds):
    """Convert decimal odds to American format."""
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))

def american_to_decimal(american_odds):
    """Convert American odds to decimal format."""
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))

def odds_to_probability(decimal_odds):
    """Convert decimal odds to implied probability."""
    return 1 / decimal_odds

def probability_to_odds(probability):
    """Convert probability to fair decimal odds."""
    return 1 / probability

def calculate_expected_value(probability, decimal_odds):
    """Calculate the expected value of a bet."""
    return (probability * (decimal_odds - 1)) - (1 - probability)

def calculate_kelly_stake(probability, decimal_odds, kelly_fraction=1.0):
    """Calculate the optimal stake using the Kelly Criterion with a fraction for conservativeness."""
    # Extract probability implied by the odds
    implied_prob = odds_to_probability(decimal_odds)
    
    # Check for positive expected value
    edge = probability - implied_prob
    if edge <= 0:
        return 0.0
    
    # Calculate Kelly stake
    full_kelly = probability - ((1 - probability) / (decimal_odds - 1))
    
    # Apply fractional Kelly for more conservative betting
    adjusted_kelly = full_kelly * kelly_fraction
    
    # Return as a percentage of bankroll (capped at 25% to prevent extreme recommendations)
    return min(adjusted_kelly, 0.25)

def visualize_betting_analysis(analysis, save_path=None):
    """Create a visualization of the betting analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Betting Analysis: {analysis['matchup']}", fontsize=16)
    
    # Plot 1: Win Probability Comparison
    ax1 = axes[0, 0]
    labels = [analysis['team1_name'], analysis['team2_name']]
    model_probs = [analysis['team1_win_probability'], analysis['team2_win_probability']]
    market_probs = [odds_to_probability(analysis['odds']['ml_odds_team1']), 
                   odds_to_probability(analysis['odds']['ml_odds_team2'])]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, model_probs, width, label='Model Probability')
    ax1.bar(x + width/2, market_probs, width, label='Market Implied Probability')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1)
    ax1.set_title('Win Probability Comparison')
    ax1.set_ylabel('Probability')
    ax1.legend()
    
    # Add probability values on bars
    for i, v in enumerate(model_probs):
        ax1.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
    for i, v in enumerate(market_probs):
        ax1.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    # Plot 2: Expected Value Comparison
    ax2 = axes[0, 1]
    bet_types = []
    evs = []
    colors = []
    
    for bet in analysis['bets']:
        bet_types.append(f"{bet['type']} ({bet['selection']})")
        evs.append(bet['ev'])
        colors.append('green' if bet['ev'] > 0 else 'red')
    
    ax2.bar(bet_types, evs, color=colors)
    ax2.set_title('Expected Value by Bet Type')
    ax2.set_ylabel('Expected Value')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xticklabels(bet_types, rotation=45, ha='right')
    
    # Add EV values on bars
    for i, v in enumerate(evs):
        ax2.text(i, v + 0.01 if v >= 0 else v - 0.03, f'{v:.2f}', ha='center')
    
    # Plot 3: Kelly Criterion Recommendations
    ax3 = axes[1, 0]
    recommended_bets = []
    kelly_pcts = []
    
    for bet in analysis['bets']:
        if bet['kelly_fraction'] > 0:
            recommended_bets.append(f"{bet['type']} ({bet['selection']})")
            kelly_pcts.append(bet['kelly_fraction'] * 100)  # Convert to percentage
    
    if recommended_bets:
        ax3.bar(recommended_bets, kelly_pcts, color='purple')
        ax3.set_title('Recommended Bet Sizes (% of Bankroll)')
        ax3.set_ylabel('% of Bankroll')
        ax3.set_xticklabels(recommended_bets, rotation=45, ha='right')
        
        # Add percentage values on bars
        for i, v in enumerate(kelly_pcts):
            ax3.text(i, v + 0.5, f'{v:.1f}%', ha='center')
    else:
        ax3.text(0.5, 0.5, "No positive expected value bets found", 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Confidence Visualization
    ax4 = axes[1, 1]
    
    # Create a custom confidence visualization
    confidence_levels = []
    items = []
    
    # Overall prediction confidence
    ensemble_std = analysis.get('ensemble_std', 0.1)
    confidence_score = max(0, 1 - ensemble_std*5)  # Convert std to a confidence-like metric
    confidence_levels.append(confidence_score)
    items.append('Model Consensus')
    
    # Data quality
    data_quality = analysis.get('data_quality', 0.7)
    confidence_levels.append(data_quality)
    items.append('Data Quality')
    
    # Historical accuracy
    historical_accuracy = analysis.get('historical_accuracy', 0.68)
    confidence_levels.append(historical_accuracy)
    items.append('Historical Accuracy')
    
    # Sample size adequacy
    sample_adequacy = analysis.get('sample_adequacy', 0.8)
    confidence_levels.append(sample_adequacy)
    items.append('Sample Adequacy')
    
    ax4.barh(items, confidence_levels, color='skyblue')
    ax4.set_title('Confidence Metrics')
    ax4.set_xlim(0, 1)
    ax4.axvline(x=0.7, color='orange', linestyle='--', alpha=0.7, label='Confidence Threshold')
    ax4.legend()
    
    # Add confidence values on bars
    for i, v in enumerate(confidence_levels):
        ax4.text(v + 0.02, i, f'{v:.2f}', va='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def load_ensemble_models(num_folds=5):
    """Load the ensemble models from disk."""
    models = []
    for i in range(1, num_folds + 1):
        model_path = f'valorant_model_fold_{i}.h5'
        if os.path.exists(model_path):
            model = load_model(model_path)
            models.append(model)
        else:
            print(f"Warning: Model {model_path} not found.")
    
    if not models:
        print("Error: No ensemble models found. Please train models first.")
        return None
    
    print(f"Loaded {len(models)} ensemble models.")
    return models

def load_ensemble_components():
    """Load ensemble model components (scaler, features)."""
    scaler = None
    stable_features = None
    
    # Try to load ensemble scaler
    if os.path.exists('ensemble_scaler.pkl'):
        with open('ensemble_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    else:
        print("Warning: Ensemble scaler not found. Trying standard scaler.")
        if os.path.exists('feature_scaler.pkl'):
            with open('feature_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
    
    # Try to load stable features
    if os.path.exists('stable_features.pkl'):
        with open('stable_features.pkl', 'rb') as f:
            stable_features = pickle.load(f)
    else:
        print("Warning: Stable features not found. Trying standard features.")
        if os.path.exists('feature_names.pkl'):
            with open('feature_names.pkl', 'rb') as f:
                stable_features = pickle.load(f)
    
    if not scaler or not stable_features:
        print("Error: Required model components not found.")
        return None, None
    
    return scaler, stable_features

def predict_with_ensemble(team1_name, team2_name, num_folds=5, analyze_features=False):
    """Make a prediction using the ensemble model."""
    # Load models and components
    ensemble_models = load_ensemble_models(num_folds)
    scaler, stable_features = load_ensemble_components()
    
    if not ensemble_models or not scaler or not stable_features:
        return None
    
    # Get team IDs
    team1_id = get_team_id(team1_name)
    team2_id = get_team_id(team2_name)
    
    if not team1_id or not team2_id:
        print(f"Error: Could not find team IDs for {team1_name} and/or {team2_name}")
        return None
    
    # Fetch team details
    team1_details, team1_tag = fetch_team_details(team1_id)
    team2_details, team2_tag = fetch_team_details(team2_id)
    
    # Fetch team match history
    team1_history = fetch_team_match_history(team1_id)
    team2_history = fetch_team_match_history(team2_id)
    
    # Parse match data
    team1_matches = parse_match_data(team1_history, team1_name)
    team2_matches = parse_match_data(team2_history, team2_name)
    
    # Fetch player stats
    team1_player_stats = fetch_team_player_stats(team1_id)
    team2_player_stats = fetch_team_player_stats(team2_id)
    
    # Calculate team stats
    team1_stats = calculate_team_stats(team1_matches, team1_player_stats, include_economy=True)
    team2_stats = calculate_team_stats(team2_matches, team2_player_stats, include_economy=True)
    
    # Add team identifiers to stats
    team1_stats['team_tag'] = team1_tag
    team1_stats['team_name'] = team1_name
    team1_stats['team_id'] = team1_id
    
    team2_stats['team_tag'] = team2_tag
    team2_stats['team_name'] = team2_name
    team2_stats['team_id'] = team2_id
    
    # Prepare data for model
    features = prepare_data_for_model(team1_stats, team2_stats)
    
    if not features:
        print("Error: Could not prepare features for prediction.")
        return None
    
    # Create a DataFrame for feature selection
    df = pd.DataFrame([features])
    
    # Ensure all required features are present - ADD THIS CODE
    for feature in stable_features:
        if feature not in df.columns:
            df[feature] = 0  # Add missing features with default values
    
    # Select only stable features
    X = df[stable_features].values
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions with each model
    predictions = []
    for model in ensemble_models:
        pred = model.predict(X_scaled)[0][0]
        predictions.append(pred)
    
    # Calculate ensemble statistics
    ensemble_mean = np.mean(predictions)
    ensemble_std = np.std(predictions)
    ensemble_min = np.min(predictions)
    ensemble_max = np.max(predictions)
    
    # Calculate ensemble weighted prediction based on validation performance
    # This is a simplified approach - ideally weights would come from validation metrics
    team1_win_probability = ensemble_mean
    
    # Calculate feature importances
    feature_importance = {}
    if analyze_features and len(stable_features) > 0:
        # Train a quick RF model to get feature importances
        X_train = X_scaled
        y_train = np.array([1])  # Dummy target
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        try:
            rf.fit(X_train, y_train)
            importances = rf.feature_importances_
            
            for i, feature in enumerate(stable_features):
                feature_importance[feature] = importances[i]
        except:
            pass
    
    # Determine data quality based on number of matches
    team1_matches_count = len(team1_matches)
    team2_matches_count = len(team2_matches)
    min_matches = min(team1_matches_count, team2_matches_count)
    
    data_quality = 0.4  # Base quality
    if min_matches >= 5:
        data_quality = 0.6
    if min_matches >= 10:
        data_quality = 0.8
    if min_matches >= 20:
        data_quality = 1.0
        
    # Include both model-predicted and feature-based data quality
    sample_adequacy = data_quality
    
    # Create prediction result
    prediction_result = {
        'team1_name': team1_name,
        'team2_name': team2_name,
        'team1_win_probability': float(team1_win_probability),
        'team2_win_probability': 1 - float(team1_win_probability),
        'ensemble_mean': float(ensemble_mean),
        'ensemble_std': float(ensemble_std),
        'ensemble_min': float(ensemble_min),
        'ensemble_max': float(ensemble_max),
        'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'feature_importance': feature_importance if analyze_features else {},
        'data_quality': data_quality,
        'sample_adequacy': sample_adequacy,
        'team1_matches_count': team1_matches_count,
        'team2_matches_count': team2_matches_count,
        'historical_accuracy': 0.68  # Based on your mentioned model accuracy
    }
    
    return prediction_result

def analyze_match_betting(team1_name, team2_name, odds_data, bankroll=1000, min_ev=0.05, kelly_fraction=0.25):
    """Analyze betting opportunities for a match with manual odds input."""
    # Get prediction from ensemble model
    prediction = predict_with_ensemble(team1_name, team2_name)
    
    if not prediction:
        print(f"Error: Could not get prediction for {team1_name} vs {team2_name}")
        return None
    
    # Extract core probabilities
    team1_win_prob = prediction['team1_win_probability']
    team2_win_prob = prediction['team2_win_probability']
    
    # Calculate map score probabilities (simple model based on win probability)
    # These are approximations - ideally you'd have a dedicated model for this
    
    # Assume higher probability for 2-0 if win probability is higher
    team1_2_0_prob = team1_win_prob * (0.4 + team1_win_prob * 0.4)
    team1_2_1_prob = team1_win_prob - team1_2_0_prob
    
    team2_2_0_prob = team2_win_prob * (0.4 + team2_win_prob * 0.4)
    team2_2_1_prob = team2_win_prob - team2_2_0_prob
    
    # Calculate map total probabilities
    maps_2_prob = team1_2_0_prob + team2_2_0_prob
    maps_3_prob = team1_2_1_prob + team2_2_1_prob
    
    # Calculate +1.5 and -1.5 map probabilities
    team1_plus_1_5_prob = team1_win_prob + team1_2_1_prob  # Team 1 wins or loses 1-2
    team2_plus_1_5_prob = team2_win_prob + team2_2_1_prob  # Team 2 wins or loses 1-2
    
    team1_minus_1_5_prob = team1_2_0_prob  # Team 1 wins 2-0
    team2_minus_1_5_prob = team2_2_0_prob  # Team 2 wins 2-0
    
    # Calculate expected values for different bet types
    bets_analysis = []
    
    # Moneyline bets
    team1_ml_ev = calculate_expected_value(team1_win_prob, odds_data['ml_odds_team1'])
    team2_ml_ev = calculate_expected_value(team2_win_prob, odds_data['ml_odds_team2'])
    
    team1_ml_kelly = calculate_kelly_stake(team1_win_prob, odds_data['ml_odds_team1'], kelly_fraction)
    team2_ml_kelly = calculate_kelly_stake(team2_win_prob, odds_data['ml_odds_team2'], kelly_fraction)
    
    # Handicap bets
    team1_plus_1_5_ev = calculate_expected_value(team1_plus_1_5_prob, odds_data['team1_plus_1_5_odds'])
    team2_plus_1_5_ev = calculate_expected_value(team2_plus_1_5_prob, odds_data['team2_plus_1_5_odds'])
    
    team1_plus_1_5_kelly = calculate_kelly_stake(team1_plus_1_5_prob, odds_data['team1_plus_1_5_odds'], kelly_fraction)
    team2_plus_1_5_kelly = calculate_kelly_stake(team2_plus_1_5_prob, odds_data['team2_plus_1_5_odds'], kelly_fraction)
    
    team1_minus_1_5_ev = calculate_expected_value(team1_minus_1_5_prob, odds_data['team1_minus_1_5_odds'])
    team2_minus_1_5_ev = calculate_expected_value(team2_minus_1_5_prob, odds_data['team2_minus_1_5_odds'])
    
    team1_minus_1_5_kelly = calculate_kelly_stake(team1_minus_1_5_prob, odds_data['team1_minus_1_5_odds'], kelly_fraction)
    team2_minus_1_5_kelly = calculate_kelly_stake(team2_minus_1_5_prob, odds_data['team2_minus_1_5_odds'], kelly_fraction)
    
    # Total maps bets
    over_2_5_ev = calculate_expected_value(maps_3_prob, odds_data['over_2_5_odds'])
    under_2_5_ev = calculate_expected_value(maps_2_prob, odds_data['under_2_5_odds'])
    
    over_2_5_kelly = calculate_kelly_stake(maps_3_prob, odds_data['over_2_5_odds'], kelly_fraction)
    under_2_5_kelly = calculate_kelly_stake(maps_2_prob, odds_data['under_2_5_odds'], kelly_fraction)
    
    # Add all bets to analysis list
    bets_analysis = [
        {
            'type': 'Moneyline',
            'selection': team1_name,
            'model_prob': team1_win_prob,
            'implied_prob': odds_to_probability(odds_data['ml_odds_team1']),
            'edge': team1_win_prob - odds_to_probability(odds_data['ml_odds_team1']),
            'decimal_odds': odds_data['ml_odds_team1'],
            'american_odds': decimal_to_american(odds_data['ml_odds_team1']),
            'ev': team1_ml_ev,
            'kelly_fraction': team1_ml_kelly,
            'recommended_bet': team1_ml_kelly * bankroll if team1_ml_ev > min_ev else 0
        },
        {
            'type': 'Moneyline',
            'selection': team2_name,
            'model_prob': team2_win_prob,
            'implied_prob': odds_to_probability(odds_data['ml_odds_team2']),
            'edge': team2_win_prob - odds_to_probability(odds_data['ml_odds_team2']),
            'decimal_odds': odds_data['ml_odds_team2'],
            'american_odds': decimal_to_american(odds_data['ml_odds_team2']),
            'ev': team2_ml_ev,
            'kelly_fraction': team2_ml_kelly,
            'recommended_bet': team2_ml_kelly * bankroll if team2_ml_ev > min_ev else 0
        },
        {
            'type': 'Handicap +1.5',
            'selection': team1_name,
            'model_prob': team1_plus_1_5_prob,
            'implied_prob': odds_to_probability(odds_data['team1_plus_1_5_odds']),
            'edge': team1_plus_1_5_prob - odds_to_probability(odds_data['team1_plus_1_5_odds']),
            'decimal_odds': odds_data['team1_plus_1_5_odds'],
            'american_odds': decimal_to_american(odds_data['team1_plus_1_5_odds']),
            'ev': team1_plus_1_5_ev,
            'kelly_fraction': team1_plus_1_5_kelly,
            'recommended_bet': team1_plus_1_5_kelly * bankroll if team1_plus_1_5_ev > min_ev else 0
        },
        {
            'type': 'Handicap +1.5',
            'selection': team2_name,
            'model_prob': team2_plus_1_5_prob,
            'implied_prob': odds_to_probability(odds_data['team2_plus_1_5_odds']),
            'edge': team2_plus_1_5_prob - odds_to_probability(odds_data['team2_plus_1_5_odds']),
            'decimal_odds': odds_data['team2_plus_1_5_odds'],
            'american_odds': decimal_to_american(odds_data['team2_plus_1_5_odds']),
            'ev': team2_plus_1_5_ev,
            'kelly_fraction': team2_plus_1_5_kelly,
            'recommended_bet': team2_plus_1_5_kelly * bankroll if team2_plus_1_5_ev > min_ev else 0
        },
        {
            'type': 'Handicap -1.5',
            'selection': team1_name,
            'model_prob': team1_minus_1_5_prob,
            'implied_prob': odds_to_probability(odds_data['team1_minus_1_5_odds']),
            'edge': team1_minus_1_5_prob - odds_to_probability(odds_data['team1_minus_1_5_odds']),
            'decimal_odds': odds_data['team1_minus_1_5_odds'],
            'american_odds': decimal_to_american(odds_data['team1_minus_1_5_odds']),
            'ev': team1_minus_1_5_ev,
            'kelly_fraction': team1_minus_1_5_kelly,
            'recommended_bet': team1_minus_1_5_kelly * bankroll if team1_minus_1_5_ev > min_ev else 0
        },
        {
            'type': 'Handicap -1.5',
            'selection': team2_name,
            'model_prob': team2_minus_1_5_prob,
            'implied_prob': odds_to_probability(odds_data['team2_minus_1_5_odds']),
            'edge': team2_minus_1_5_prob - odds_to_probability(odds_data['team2_minus_1_5_odds']),
            'decimal_odds': odds_data['team2_minus_1_5_odds'],
            'american_odds': decimal_to_american(odds_data['team2_minus_1_5_odds']),
            'ev': team2_minus_1_5_ev,
            'kelly_fraction': team2_minus_1_5_kelly,
            'recommended_bet': team2_minus_1_5_kelly * bankroll if team2_minus_1_5_ev > min_ev else 0
        },
        {
            'type': 'Total Maps',
            'selection': 'Over 2.5',
            'model_prob': maps_3_prob,
            'implied_prob': odds_to_probability(odds_data['over_2_5_odds']),
            'edge': maps_3_prob - odds_to_probability(odds_data['over_2_5_odds']),
            'decimal_odds': odds_data['over_2_5_odds'],
            'american_odds': decimal_to_american(odds_data['over_2_5_odds']),
            'ev': over_2_5_ev,
            'kelly_fraction': over_2_5_kelly,
            'recommended_bet': over_2_5_kelly * bankroll if over_2_5_ev > min_ev else 0
        },
        {
            'type': 'Total Maps',
            'selection': 'Under 2.5',
            'model_prob': maps_2_prob,
            'implied_prob': odds_to_probability(odds_data['under_2_5_odds']),
            'edge': maps_2_prob - odds_to_probability(odds_data['under_2_5_odds']),
            'decimal_odds': odds_data['under_2_5_odds'],
            'american_odds': decimal_to_american(odds_data['under_2_5_odds']),
            'ev': under_2_5_ev,
            'kelly_fraction': under_2_5_kelly,
            'recommended_bet': under_2_5_kelly * bankroll if under_2_5_ev > min_ev else 0
        }
    ]
    
    # Sort by expected value
    bets_analysis.sort(key=lambda x: x['ev'], reverse=True)
    
    # Count recommended bets
    recommended_count = sum(1 for bet in bets_analysis if bet['recommended_bet'] > 0)
    
    # Create detailed analysis report
    analysis_report = {
        'matchup': f"{team1_name} vs {team2_name}",
        'prediction_time': prediction['prediction_time'],
        'team1_name': team1_name,
        'team2_name': team2_name,
        'team1_win_probability': team1_win_prob,
        'team2_win_probability': team2_win_prob,
        'team1_2_0_probability': team1_2_0_prob,
        'team1_2_1_probability': team1_2_1_prob,
        'team2_2_0_probability': team2_2_0_prob,
        'team2_2_1_probability': team2_2_1_prob,
        'maps_2_probability': maps_2_prob,
        'maps_3_probability': maps_3_prob,
        'odds': odds_data,
        'bets': bets_analysis,
        'recommended_count': recommended_count,
        'ensemble_std': prediction['ensemble_std'],
        'data_quality': prediction['data_quality'],
        'sample_adequacy': prediction['sample_adequacy'],
        'historical_accuracy': prediction['historical_accuracy'],
        'team1_matches_count': prediction['team1_matches_count'],
        'team2_matches_count': prediction['team2_matches_count'],
        'bankroll': bankroll,
        'min_ev_threshold': min_ev,
        'kelly_fraction': kelly_fraction
    }
    
    return analysis_report

def save_bet_to_history(bet_data, file_path='bet_history.csv'):
    """Save a bet to the historical record."""
    # Create a list of fields to save
    fields = [
        'timestamp', 'match_id', 'match_date', 'team1', 'team2', 
        'bet_type', 'selection', 'predicted_probability', 'implied_probability',
        'decimal_odds', 'bet_amount', 'expected_value', 'kelly_fraction',
        'outcome', 'profit_loss', 'notes'
    ]
    
    # Check if file exists
    file_exists = os.path.isfile(file_path)
    
    # Write data
    with open(file_path, 'a', newline='') as f:
        writer = pd.DataFrame([bet_data], columns=fields)
        writer.to_csv(f, header=not file_exists, index=False)
    
    print(f"Bet saved to {file_path}")

def display_prediction_results(prediction):
    """Display the results of a match prediction in a readable format."""
    if not prediction:
        print("No prediction available.")
        return
    
    print("\n" + "="*80)
    print(f"MATCH PREDICTION: {prediction['team1_name']} vs {prediction['team2_name']}")
    print("="*80)
    
    # Format probabilities as percentages
    team1_win_pct = prediction['team1_win_probability'] * 100
    team2_win_pct = prediction['team2_win_probability'] * 100
    
    print(f"\nWin Probability:")
    print(f"  {prediction['team1_name']}: {team1_win_pct:.1f}%")
    print(f"  {prediction['team2_name']}: {team2_win_pct:.1f}%")
    
    # Show ensemble model details if available
    if 'ensemble_mean' in prediction:
        print("\nEnsemble Model Details:")
        print(f"  Mean Probability: {prediction['ensemble_mean']:.4f}")
        print(f"  Standard Deviation: {prediction['ensemble_std']:.4f}")
        print(f"  Range: {prediction['ensemble_min']:.4f} - {prediction['ensemble_max']:.4f}")
    
    # Show data quality information
    print("\nData Confidence:")
    print(f"  Data Quality: {prediction['data_quality']:.2f}")
    print(f"  Sample Adequacy: {prediction['sample_adequacy']:.2f}")
    print(f"  Team 1 Matches: {prediction['team1_matches_count']}")
    print(f"  Team 2 Matches: {prediction['team2_matches_count']}")
    
    # Show top feature importances if available
    if prediction.get('feature_importance'):
        print("\nTop 5 Important Features:")
        sorted_features = sorted(prediction['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.4f}")
    
    print("\nPrediction made at:", prediction['prediction_time'])
    print("="*80)

def display_betting_analysis(analysis):
    """Display the results of betting analysis in a readable format."""
    if not analysis:
        print("No analysis available.")
        return
    
    print("\n" + "="*80)
    print(f"BETTING ANALYSIS: {analysis['matchup']}")
    print("="*80)
    
    # Format probabilities as percentages
    team1_win_pct = analysis['team1_win_probability'] * 100
    team2_win_pct = analysis['team2_win_probability'] * 100
    
    print(f"\nWin Probability:")
    print(f"  {analysis['team1_name']}: {team1_win_pct:.1f}%")
    print(f"  {analysis['team2_name']}: {team2_win_pct:.1f}%")
    
    # Score probabilities
    print(f"\nScore Probabilities:")
    print(f"  {analysis['team1_name']} 2-0: {analysis['team1_2_0_probability']*100:.1f}%")
    print(f"  {analysis['team1_name']} 2-1: {analysis['team1_2_1_probability']*100:.1f}%")
    print(f"  {analysis['team2_name']} 2-0: {analysis['team2_2_0_probability']*100:.1f}%")
    print(f"  {analysis['team2_name']} 2-1: {analysis['team2_2_1_probability']*100:.1f}%")
    print(f"  Maps Played = 2: {analysis['maps_2_probability']*100:.1f}%")
    print(f"  Maps Played = 3: {analysis['maps_3_probability']*100:.1f}%")
    
    # Odds data
    print("\nOdds:")
    print(f"  {analysis['team1_name']} ML: {analysis['odds']['ml_odds_team1']:.2f} (implied {odds_to_probability(analysis['odds']['ml_odds_team1'])*100:.1f}%)")
    print(f"  {analysis['team2_name']} ML: {analysis['odds']['ml_odds_team2']:.2f} (implied {odds_to_probability(analysis['odds']['ml_odds_team2'])*100:.1f}%)")
    print(f"  {analysis['team1_name']} +1.5: {analysis['odds']['team1_plus_1_5_odds']:.2f}")
    print(f"  {analysis['team2_name']} +1.5: {analysis['odds']['team2_plus_1_5_odds']:.2f}")
    print(f"  {analysis['team1_name']} -1.5: {analysis['odds']['team1_minus_1_5_odds']:.2f}")
    print(f"  {analysis['team2_name']} -1.5: {analysis['odds']['team2_minus_1_5_odds']:.2f}")
    print(f"  Over 2.5 Maps: {analysis['odds']['over_2_5_odds']:.2f}")
    print(f"  Under 2.5 Maps: {analysis['odds']['under_2_5_odds']:.2f}")
    
    # Data quality 
    print(f"\nModel Confidence Metrics:")
    print(f"  Model Consensus: {1-analysis['ensemble_std']*5:.2f}")
    print(f"  Data Quality: {analysis['data_quality']:.2f}")
    print(f"  Sample Adequacy: {analysis['sample_adequacy']:.2f}")
    print(f"  Historical Accuracy: {analysis['historical_accuracy']:.2f}")
    
    # Betting recommendations
    print("\nBetting Recommendations:")
    print(f"Bankroll: ${analysis['bankroll']:.2f}")
    print(f"Minimum EV Threshold: {analysis['min_ev_threshold']:.2f}")
    print(f"Kelly Fraction: {analysis['kelly_fraction']:.2f}")
    
    # Sort bets by EV
    sorted_bets = sorted(analysis['bets'], key=lambda x: x['ev'], reverse=True)
    
    # Display all bets with their EVs
    print("\nAll Bet Options (sorted by Expected Value):")
    print("-"*80)
    print(f"{'Type':<15} {'Selection':<15} {'Odds':<8} {'Model %':<8} {'Market %':<8} {'Edge':<8} {'EV':<8} {'Kelly %':<8} {'Bet':<10}")
    print("-"*80)
    
    for bet in sorted_bets:
        bet_type = bet['type']
        selection = bet['selection']
        odds = f"{bet['decimal_odds']:.2f}"
        model_prob = f"{bet['model_prob']*100:.1f}%"
        market_prob = f"{bet['implied_prob']*100:.1f}%"
        edge = f"{bet['edge']*100:+.1f}%"
        ev = f"{bet['ev']*100:+.1f}%"
        kelly = f"{bet['kelly_fraction']*100:.1f}%"
        rec_bet = f"${bet['recommended_bet']:.2f}" if bet['recommended_bet'] > 0 else "-"
        
        print(f"{bet_type:<15} {selection:<15} {odds:<8} {model_prob:<8} {market_prob:<8} {edge:<8} {ev:<8} {kelly:<8} {rec_bet:<10}")
    
    # Show only recommended bets
    recommended_bets = [bet for bet in analysis['bets'] if bet['recommended_bet'] > 0]
    
    if recommended_bets:
        print("\nRECOMMENDED BETS:")
        print("-"*80)
        total_recommended = sum(bet['recommended_bet'] for bet in recommended_bets)
        
        for bet in recommended_bets:
            bet_type = bet['type']
            selection = bet['selection']
            odds = f"{bet['decimal_odds']:.2f}"
            rec_bet = f"${bet['recommended_bet']:.2f}"
            ev = f"{bet['ev']*100:+.1f}%"
            percent_of_bankroll = bet['recommended_bet'] / analysis['bankroll'] * 100
            
            print(f"{bet_type} - {selection} @ {odds} - {rec_bet} ({percent_of_bankroll:.1f}% of bankroll) - EV: {ev}")
        
        print("-"*80)
        print(f"Total Recommended Bet: ${total_recommended:.2f} ({total_recommended/analysis['bankroll']*100:.1f}% of bankroll)")
    else:
        print("\nNo bets recommended with positive expected value above the threshold.")
    
    print("\nAnalysis created at:", analysis['prediction_time'])
    print("="*80)

def record_match_result(match_id, team1_name, team2_name, team1_score, team2_score, file_path='match_results.csv'):
    """Record the result of a match for backtesting purposes."""
    # Create match result data
    match_data = {
        'match_id': match_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'team1': team1_name,
        'team2': team2_name,
        'team1_score': team1_score,
        'team2_score': team2_score,
        'winner': team1_name if team1_score > team2_score else team2_name,
        'score_line': f"{team1_score}-{team2_score}",
        'maps_played': team1_score + team2_score
    }
    
    # Add result booleans for different bet types
    match_data['team1_ml_result'] = 1 if team1_score > team2_score else 0
    match_data['team2_ml_result'] = 1 if team2_score > team1_score else 0
    match_data['team1_plus_1_5_result'] = 1 if team1_score >= 1 else 0
    match_data['team2_plus_1_5_result'] = 1 if team2_score >= 1 else 0
    match_data['team1_minus_1_5_result'] = 1 if team1_score >= 2 and team2_score == 0 else 0
    match_data['team2_minus_1_5_result'] = 1 if team2_score >= 2 and team1_score == 0 else 0
    match_data['over_2_5_maps_result'] = 1 if (team1_score + team2_score) > 2 else 0
    match_data['under_2_5_maps_result'] = 1 if (team1_score + team2_score) <= 2 else 0
    
    # Check if file exists
    file_exists = os.path.isfile(file_path)
    
    # Write data
    with open(file_path, 'a', newline='') as f:
        fields = match_data.keys()
        writer = pd.DataFrame([match_data], columns=fields)
        writer.to_csv(f, header=not file_exists, index=False)
    
    print(f"Match result saved to {file_path}")
    return match_data

def calculate_bet_outcome(bet_data, match_result):
    """Calculate the outcome of a bet based on match result."""
    bet_type = bet_data['bet_type']
    selection = bet_data['selection']
    bet_amount = bet_data['bet_amount']
    decimal_odds = bet_data['decimal_odds']
    
    # Moneyline bets
    if bet_type == 'Moneyline':
        if selection == match_result['team1'] and match_result['team1_ml_result'] == 1:
            return bet_amount * (decimal_odds - 1)
        elif selection == match_result['team2'] and match_result['team2_ml_result'] == 1:
            return bet_amount * (decimal_odds - 1)
        else:
            return -bet_amount
    
    # Handicap bets
    elif bet_type == 'Handicap +1.5':
        if selection == match_result['team1'] and match_result['team1_plus_1_5_result'] == 1:
            return bet_amount * (decimal_odds - 1)
        elif selection == match_result['team2'] and match_result['team2_plus_1_5_result'] == 1:
            return bet_amount * (decimal_odds - 1)
        else:
            return -bet_amount
    
    elif bet_type == 'Handicap -1.5':
        if selection == match_result['team1'] and match_result['team1_minus_1_5_result'] == 1:
            return bet_amount * (decimal_odds - 1)
        elif selection == match_result['team2'] and match_result['team2_minus_1_5_result'] == 1:
            return bet_amount * (decimal_odds - 1)
        else:
            return -bet_amount
    
    # Total maps bets
    elif bet_type == 'Total Maps':
        if selection == 'Over 2.5' and match_result['over_2_5_maps_result'] == 1:
            return bet_amount * (decimal_odds - 1)
        elif selection == 'Under 2.5' and match_result['under_2_5_maps_result'] == 1:
            return bet_amount * (decimal_odds - 1)
        else:
            return -bet_amount
    
    # Default case - bet lost
    return -bet_amount

def update_bet_history_with_result(bet_id, match_result, file_path='bet_history.csv'):
    """Update a bet in the history with the actual result."""
    # Load bet history
    if not os.path.isfile(file_path):
        print(f"Error: Bet history file {file_path} not found.")
        return False
    
    bet_history = pd.read_csv(file_path)
    
    # Find the bet with the given ID
    if 'bet_id' not in bet_history.columns or bet_id not in bet_history['bet_id'].values:
        print(f"Error: Bet ID {bet_id} not found in history.")
        return False
    
    # Get the bet data
    bet_idx = bet_history[bet_history['bet_id'] == bet_id].index[0]
    bet_data = bet_history.iloc[bet_idx].to_dict()
    
    # Calculate bet outcome
    profit_loss = calculate_bet_outcome(bet_data, match_result)
    
    # Update the bet record
    bet_history.at[bet_idx, 'outcome'] = 'win' if profit_loss > 0 else 'loss'
    bet_history.at[bet_idx, 'profit_loss'] = profit_loss
    bet_history.at[bet_idx, 'result_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save updated history
    bet_history.to_csv(file_path, index=False)
    
    print(f"Bet {bet_id} updated with result: {bet_history.at[bet_idx, 'outcome']}, P/L: ${profit_loss:.2f}")
    return True

def backtest_model(cutoff_date, bet_amount=100, confidence_threshold=0.05, kelly_fraction=0.25, num_folds=5):
    """Backtest the model on historical matches after a given cutoff date."""
    print(f"Starting backtest from {cutoff_date} with {bet_amount:.2f} bet amount")
    
    # Load ensemble components
    ensemble_models = load_ensemble_models(num_folds)
    scaler, stable_features = load_ensemble_components()
    
    if not ensemble_models or not scaler or not stable_features:
        print("Error: Could not load model components for backtesting.")
        return None
    
    # Load match results if available
    results_file = 'match_results.csv'
    if not os.path.isfile(results_file):
        print(f"Error: Match results file {results_file} not found.")
        return None
    
    match_results = pd.read_csv(results_file)
    
    # Convert cutoff date to datetime
    cutoff_dt = datetime.strptime(cutoff_date, "%Y/%m/%d")
    
    # Filter matches after cutoff date
    match_results['timestamp'] = pd.to_datetime(match_results['timestamp'])
    test_matches = match_results[match_results['timestamp'] >= cutoff_dt]
    
    if len(test_matches) == 0:
        print(f"No matches found after {cutoff_date}")
        return None
    
    print(f"Found {len(test_matches)} matches for backtesting")
    
    # Initialize results tracking
    backtest_results = []
    correct_predictions = 0
    high_conf_correct = 0
    high_conf_total = 0
    total_profit = 0
    total_bets = 0
    
    # Process each match
    for idx, match in test_matches.iterrows():
        match_id = match['match_id']
        team1_name = match['team1']
        team2_name = match['team2']
        
        print(f"Backtesting match {match_id}: {team1_name} vs {team2_name}")
        
        # Get prediction for this match
        # We assume team1 and team2 are in their state before the match
        prediction = predict_with_ensemble(team1_name, team2_name)
        
        if not prediction:
            print(f"  Could not generate prediction for match {match_id}")
            continue
        
        # Determine predicted winner
        predicted_winner = team1_name if prediction['team1_win_probability'] > 0.5 else team2_name
        actual_winner = match['winner']
        
        # Check if prediction was correct
        was_correct = predicted_winner == actual_winner
        if was_correct:
            correct_predictions += 1
        
        # Check prediction confidence
        team1_win_prob = prediction['team1_win_probability']
        team2_win_prob = prediction['team2_win_probability']
        
        # Higher of the two probabilities represents confidence
        confidence = max(team1_win_prob, team2_win_prob)
        
        # Track high confidence predictions
        if abs(team1_win_prob - 0.5) >= confidence_threshold:
            high_conf_total += 1
            if was_correct:
                high_conf_correct += 1
        
        # Simulate betting outcomes
        match_profit = 0
        
        # Simple betting strategy - bet on team with higher probability if confidence exceeds threshold
        if team1_win_prob >= 0.5 + confidence_threshold:
            # Bet on team1
            total_bets += 1
            
            # Calculate Kelly stake
            implied_odds = 2.0  # Simplified assumption for backtesting
            
            # Apply Kelly formula
            kelly_stake = calculate_kelly_stake(team1_win_prob, implied_odds, kelly_fraction)
            
            # Calculate bet amount
            actual_bet = bet_amount * kelly_stake
            
            # Determine outcome
            if match['team1_ml_result'] == 1:
                profit = actual_bet * (implied_odds - 1)
                match_profit += profit
            else:
                profit = -actual_bet
                match_profit += profit
            
        elif team2_win_prob >= 0.5 + confidence_threshold:
            # Bet on team2
            total_bets += 1
            
            # Calculate Kelly stake
            implied_odds = 2.0  # Simplified assumption
            
            # Apply Kelly formula
            kelly_stake = calculate_kelly_stake(team2_win_prob, implied_odds, kelly_fraction)
            
            # Calculate bet amount
            actual_bet = bet_amount * kelly_stake
            
            # Determine outcome
            if match['team2_ml_result'] == 1:
                profit = actual_bet * (implied_odds - 1)
                match_profit += profit
            else:
                profit = -actual_bet
                match_profit += profit
        
        # Update total profit
        total_profit += match_profit
        
        # Record result for this match
        backtest_results.append({
            'match_id': match_id,
            'team1': team1_name,
            'team2': team2_name,
            'predicted_winner': predicted_winner,
            'actual_winner': actual_winner,
            'prediction_correct': was_correct,
            'confidence': confidence,
            'high_confidence': abs(team1_win_prob - 0.5) >= confidence_threshold,
            'match_profit': match_profit
        })
        
        print(f"  Prediction: {predicted_winner} (confidence: {confidence:.2f})")
        print(f"  Actual winner: {actual_winner}")
        print(f"  Correct: {was_correct}")
        print(f"  Match profit: ${match_profit:.2f}")
    
    # Calculate overall metrics
    if len(backtest_results) > 0:
        overall_accuracy = correct_predictions / len(backtest_results)
    else:
        overall_accuracy = 0
    
    if high_conf_total > 0:
        high_conf_accuracy = high_conf_correct / high_conf_total
    else:
        high_conf_accuracy = 0
    
    if total_bets > 0:
        roi = total_profit / (bet_amount * total_bets)
    else:
        roi = 0
    
    # Create summary report
    backtest_summary = {
        'cutoff_date': cutoff_date,
        'matches_tested': len(backtest_results),
        'overall_accuracy': overall_accuracy,
        'high_conf_accuracy': high_conf_accuracy,
        'high_conf_predictions': high_conf_total,
        'total_bets': total_bets,
        'total_profit': total_profit,
        'roi': roi,
        'bet_amount': bet_amount,
        'confidence_threshold': confidence_threshold,
        'kelly_fraction': kelly_fraction,
        'results': backtest_results
    }
    
    # Save backtest results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(backtest_results)
    results_df.to_csv(f'backtest_results_{timestamp}.csv', index=False)
    
    # Create visualization
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Backtest Results (Cutoff: {cutoff_date})', fontsize=16)
    
    # Plot accuracy
    axs[0, 0].bar(['Overall', 'High Confidence'], [overall_accuracy, high_conf_accuracy])
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_title('Prediction Accuracy')
    axs[0, 0].set_ylabel('Accuracy')
    
    # Add accuracy values
    axs[0, 0].text(0, overall_accuracy + 0.02, f'{overall_accuracy:.2f}', ha='center')
    axs[0, 0].text(1, high_conf_accuracy + 0.02, f'{high_conf_accuracy:.2f}', ha='center')
    
    # Plot profit over time
    if backtest_results:
        cumulative_profit = [0]
        for result in backtest_results:
            cumulative_profit.append(cumulative_profit[-1] + result['match_profit'])
        
        match_indices = range(len(cumulative_profit))
        axs[0, 1].plot(match_indices, cumulative_profit)
        axs[0, 1].set_title('Cumulative Profit Over Time')
        axs[0, 1].set_xlabel('Match Number')
        axs[0, 1].set_ylabel('Profit ($)')
        axs[0, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Plot correct vs incorrect predictions
    if backtest_results:
        results_df = pd.DataFrame(backtest_results)
        correct_count = results_df['prediction_correct'].sum()
        incorrect_count = len(results_df) - correct_count
        
        axs[1, 0].bar(['Correct', 'Incorrect'], [correct_count, incorrect_count])
        axs[1, 0].set_title('Prediction Results')
        axs[1, 0].set_ylabel('Count')
        
        # Add count values
        axs[1, 0].text(0, correct_count + 1, str(correct_count), ha='center')
        axs[1, 0].text(1, incorrect_count + 1, str(incorrect_count), ha='center')
    
    # Plot confidence distribution
    if backtest_results:
        confidence_values = results_df['confidence'].values
        axs[1, 1].hist(confidence_values, bins=10, range=(0.5, 1.0))
        axs[1, 1].set_title('Prediction Confidence Distribution')
        axs[1, 1].set_xlabel('Confidence')
        axs[1, 1].set_ylabel('Count')
        axs[1, 1].axvline(x=0.5 + confidence_threshold, color='r', linestyle='--', alpha=0.7,
                        label=f'Threshold ({0.5 + confidence_threshold:.2f})')
        axs[1, 1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'backtest_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    return backtest_summary

def train_score_model(team_data_collection):
    """Train a model specifically for predicting map scores (2-0 vs 2-1)."""
    print("Building dataset for score prediction model...")
    
    X = []
    y_score = []
    
    for team_name, team_data in team_data_collection.items():
        matches = team_data.get('matches', [])
        if not matches:
            continue
            
        for match in matches:
            # Get opponent name
            opponent_name = match.get('opponent_name')
            
            # Skip if we don't have data for the opponent
            if opponent_name not in team_data_collection:
                continue
            
            # Get team scores
            team_score = match.get('team_score', 0)
            opponent_score = match.get('opponent_score', 0)
            
            # Skip invalid scores
            if team_score + opponent_score <= 0:
                continue
                
            # Skip if not completed match (Bo3)
            if team_score + opponent_score > 3 or (team_score < 2 and opponent_score < 2):
                continue
            
            # Get stats for both teams
            team1_stats = team_data
            team2_stats = team_data_collection[opponent_name]
            
            # Prepare feature vector
            features = prepare_data_for_model(team1_stats, team2_stats)
            
            if features:
                # Determine score type: 0 = 2-0 win, 1 = 2-1 win, 2 = 1-2 loss, 3 = 0-2 loss
                score_type = -1
                if team_score > opponent_score:
                    # Team won
                    if team_score == 2 and opponent_score == 0:
                        score_type = 0  # 2-0 win
                    elif team_score == 2 and opponent_score == 1:
                        score_type = 1  # 2-1 win
                else:
                    # Team lost
                    if team_score == 1 and opponent_score == 2:
                        score_type = 2  # 1-2 loss
                    elif team_score == 0 and opponent_score == 2:
                        score_type = 3  # 0-2 loss
                
                if score_type >= 0:
                    X.append(features)
                    y_score.append(score_type)
    
    print(f"Created {len(X)} training samples for score prediction")
    
    if len(X) < 50:
        print("Not enough training data for score prediction model")
        return None
    
    # Clean and prepare feature data
    df = clean_feature_data(X)
    X_arr = df.values
    y_arr = np.array(y_score)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(4, activation='softmax')  # 4 score classes
    ])
    
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    # Train model
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"Score model accuracy: {accuracy:.4f}")
    
    # Save model components
    model.save('valorant_score_model.h5')
    with open('score_model_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('score_model_features.pkl', 'wb') as f:
        pickle.dump(df.columns.tolist(), f)
    
    return model, scaler, df.columns.tolist()

def predict_score_probabilities(team1_name, team2_name):
    """Predict detailed map score probabilities using the specialized score model."""
    # Check if score model exists
    if not os.path.exists('valorant_score_model.h5'):
        print("Score prediction model not found. Using approximation instead.")
        return None
    
    # Load score model and components
    try:
        score_model = load_model('valorant_score_model.h5')
        
        with open('score_model_scaler.pkl', 'rb') as f:
            score_scaler = pickle.load(f)
            
        with open('score_model_features.pkl', 'rb') as f:
            score_features = pickle.load(f)
    except Exception as e:
        print(f"Error loading score model components: {e}")
        return None
    
    # Get team data
    team1_id = get_team_id(team1_name)
    team2_id = get_team_id(team2_name)
    
    if not team1_id or not team2_id:
        print(f"Could not find team IDs for {team1_name} and/or {team2_name}")
        return None
    
    # Fetch team stats
    team1_matches = parse_match_data(fetch_team_match_history(team1_id), team1_name)
    team2_matches = parse_match_data(fetch_team_match_history(team2_id), team2_name)
    
    team1_player_stats = fetch_team_player_stats(team1_id)
    team2_player_stats = fetch_team_player_stats(team2_id)
    
    team1_stats = calculate_team_stats(team1_matches, team1_player_stats, include_economy=True)
    team2_stats = calculate_team_stats(team2_matches, team2_player_stats, include_economy=True)
    
    # Prepare features
    features = prepare_data_for_model(team1_stats, team2_stats)
    
    if not features:
        print("Could not prepare features for score prediction")
        return None
    
    # Create DataFrame for feature selection
    df = pd.DataFrame([features])
    
    # Ensure all required features are present
    for feature in score_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only model features
    X = df[score_features].values
    
    # Scale features
    X_scaled = score_scaler.transform(X)
    
    # Make prediction
    score_probs = score_model.predict(X_scaled)[0]
    
    # Map probabilities to score outcomes
    # 0 = team1 wins 2-0, 1 = team1 wins 2-1, 2 = team2 wins 2-1, 3 = team2 wins 2-0
    score_predictions = {
        f"{team1_name} 2-0": float(score_probs[0]),
        f"{team1_name} 2-1": float(score_probs[1]),
        f"{team2_name} 2-1": float(score_probs[2]),
        f"{team2_name} 2-0": float(score_probs[3])
    }
    
    # Validate probabilities sum to approximately 1
    prob_sum = sum(score_predictions.values())
    if not 0.99 <= prob_sum <= 1.01:
        print(f"Warning: Score probabilities sum to {prob_sum}, normalizing")
        for key in score_predictions:
            score_predictions[key] /= prob_sum
    
    # Calculate derived probabilities
    score_predictions['team1_win'] = score_predictions[f"{team1_name} 2-0"] + score_predictions[f"{team1_name} 2-1"]
    score_predictions['team2_win'] = score_predictions[f"{team2_name} 2-0"] + score_predictions[f"{team2_name} 2-1"]
    
    score_predictions['team1_plus_1_5'] = score_predictions[f"{team1_name} 2-0"] + score_predictions[f"{team1_name} 2-1"] + score_predictions[f"{team2_name} 2-1"]
    score_predictions['team2_plus_1_5'] = score_predictions[f"{team2_name} 2-0"] + score_predictions[f"{team2_name} 2-1"] + score_predictions[f"{team1_name} 2-1"]
    
    score_predictions['team1_minus_1_5'] = score_predictions[f"{team1_name} 2-0"]
    score_predictions['team2_minus_1_5'] = score_predictions[f"{team2_name} 2-0"]
    
    score_predictions['over_2_5_maps'] = score_predictions[f"{team1_name} 2-1"] + score_predictions[f"{team2_name} 2-1"]
    score_predictions['under_2_5_maps'] = score_predictions[f"{team1_name} 2-0"] + score_predictions[f"{team2_name} 2-0"]
    
    return score_predictions

def analyze_match_betting_with_score_model(team1_name, team2_name, odds_data, bankroll=1000, min_ev=0.05, kelly_fraction=0.25):
    """Enhanced betting analysis using the score prediction model for more accurate map handicap and totals."""
    # Get win prediction from ensemble model
    win_prediction = predict_with_ensemble(team1_name, team2_name)
    
    if not win_prediction:
        print(f"Error: Could not get win prediction for {team1_name} vs {team2_name}")
        return None
    
    # Get score prediction if available
    score_probs = predict_score_probabilities(team1_name, team2_name)
    
    # If score model is unavailable, use win probability to approximate
    if not score_probs:
        print("Using approximated score probabilities based on win prediction")
        
        # Extract core probabilities
        team1_win_prob = win_prediction['team1_win_probability']
        team2_win_prob = win_prediction['team2_win_probability']
        
        # Approximate map score probabilities
        team1_2_0_prob = team1_win_prob * (0.4 + team1_win_prob * 0.4)
        team1_2_1_prob = team1_win_prob - team1_2_0_prob
        
        team2_2_0_prob = team2_win_prob * (0.4 + team2_win_prob * 0.4)
        team2_2_1_prob = team2_win_prob - team2_2_0_prob
        
        # Create score probabilities dictionary
        score_probs = {
            f"{team1_name} 2-0": team1_2_0_prob,
            f"{team1_name} 2-1": team1_2_1_prob,
            f"{team2_name} 2-1": team2_2_1_prob,
            f"{team2_name} 2-0": team2_2_0_prob,
            'team1_win': team1_win_prob,
            'team2_win': team2_win_prob,
            'team1_plus_1_5': team1_win_prob + team1_2_1_prob,
            'team2_plus_1_5': team2_win_prob + team2_2_1_prob,
            'team1_minus_1_5': team1_2_0_prob,
            'team2_minus_1_5': team2_2_0_prob,
            'over_2_5_maps': team1_2_1_prob + team2_2_1_prob,
            'under_2_5_maps': team1_2_0_prob + team2_2_0_prob
        }
    
    # Calculate expected values for different bet types
    bets_analysis = []
    
    # Moneyline bets
    team1_ml_ev = calculate_expected_value(score_probs['team1_win'], odds_data['ml_odds_team1'])
    team2_ml_ev = calculate_expected_value(score_probs['team2_win'], odds_data['ml_odds_team2'])
    
    team1_ml_kelly = calculate_kelly_stake(score_probs['team1_win'], odds_data['ml_odds_team1'], kelly_fraction)
    team2_ml_kelly = calculate_kelly_stake(score_probs['team2_win'], odds_data['ml_odds_team2'], kelly_fraction)
    
    # Handicap bets
    team1_plus_1_5_ev = calculate_expected_value(score_probs['team1_plus_1_5'], odds_data['team1_plus_1_5_odds'])
    team2_plus_1_5_ev = calculate_expected_value(score_probs['team2_plus_1_5'], odds_data['team2_plus_1_5_odds'])
    
    team1_plus_1_5_kelly = calculate_kelly_stake(score_probs['team1_plus_1_5'], odds_data['team1_plus_1_5_odds'], kelly_fraction)
    team2_plus_1_5_kelly = calculate_kelly_stake(score_probs['team2_plus_1_5'], odds_data['team2_plus_1_5_odds'], kelly_fraction)
    
    team1_minus_1_5_ev = calculate_expected_value(score_probs['team1_minus_1_5'], odds_data['team1_minus_1_5_odds'])
    team2_minus_1_5_ev = calculate_expected_value(score_probs['team2_minus_1_5'], odds_data['team2_minus_1_5_odds'])
    
    team1_minus_1_5_kelly = calculate_kelly_stake(score_probs['team1_minus_1_5'], odds_data['team1_minus_1_5_odds'], kelly_fraction)
    team2_minus_1_5_kelly = calculate_kelly_stake(score_probs['team2_minus_1_5'], odds_data['team2_minus_1_5_odds'], kelly_fraction)
    
    # Total maps bets
    over_2_5_ev = calculate_expected_value(score_probs['over_2_5_maps'], odds_data['over_2_5_odds'])
    under_2_5_ev = calculate_expected_value(score_probs['under_2_5_maps'], odds_data['under_2_5_odds'])
    
    over_2_5_kelly = calculate_kelly_stake(score_probs['over_2_5_maps'], odds_data['over_2_5_odds'], kelly_fraction)
    under_2_5_kelly = calculate_kelly_stake(score_probs['under_2_5_maps'], odds_data['under_2_5_odds'], kelly_fraction)
    
    # Add all bets to analysis list
    bets_analysis = [
        {
            'type': 'Moneyline',
            'selection': team1_name,
            'model_prob': score_probs['team1_win'],
            'implied_prob': odds_to_probability(odds_data['ml_odds_team1']),
            'edge': score_probs['team1_win'] - odds_to_probability(odds_data['ml_odds_team1']),
            'decimal_odds': odds_data['ml_odds_team1'],
            'american_odds': decimal_to_american(odds_data['ml_odds_team1']),
            'ev': team1_ml_ev,
            'kelly_fraction': team1_ml_kelly,
            'recommended_bet': team1_ml_kelly * bankroll if team1_ml_ev > min_ev else 0
        },
        {
            'type': 'Moneyline',
            'selection': team2_name,
            'model_prob': score_probs['team2_win'],
            'implied_prob': odds_to_probability(odds_data['ml_odds_team2']),
            'edge': score_probs['team2_win'] - odds_to_probability(odds_data['ml_odds_team2']),
            'decimal_odds': odds_data['ml_odds_team2'],
            'american_odds': decimal_to_american(odds_data['ml_odds_team2']),
            'ev': team2_ml_ev,
            'kelly_fraction': team2_ml_kelly,
            'recommended_bet': team2_ml_kelly * bankroll if team2_ml_ev > min_ev else 0
        },
        {
            'type': 'Handicap +1.5',
            'selection': team1_name,
            'model_prob': score_probs['team1_plus_1_5'],
            'implied_prob': odds_to_probability(odds_data['team1_plus_1_5_odds']),
            'edge': score_probs['team1_plus_1_5'] - odds_to_probability(odds_data['team1_plus_1_5_odds']),
            'decimal_odds': odds_data['team1_plus_1_5_odds'],
            'american_odds': decimal_to_american(odds_data['team1_plus_1_5_odds']),
            'ev': team1_plus_1_5_ev,
            'kelly_fraction': team1_plus_1_5_kelly,
            'recommended_bet': team1_plus_1_5_kelly * bankroll if team1_plus_1_5_ev > min_ev else 0
        },
        {
            'type': 'Handicap +1.5',
            'selection': team2_name,
            'model_prob': score_probs['team2_plus_1_5'],
            'implied_prob': odds_to_probability(odds_data['team2_plus_1_5_odds']),
            'edge': score_probs['team2_plus_1_5'] - odds_to_probability(odds_data['team2_plus_1_5_odds']),
            'decimal_odds': odds_data['team2_plus_1_5_odds'],
            'american_odds': decimal_to_american(odds_data['team2_plus_1_5_odds']),
            'ev': team2_plus_1_5_ev,
            'kelly_fraction': team2_plus_1_5_kelly,
            'recommended_bet': team2_plus_1_5_kelly * bankroll if team2_plus_1_5_ev > min_ev else 0
        },
        {
            'type': 'Handicap -1.5',
            'selection': team1_name,
            'model_prob': score_probs['team1_minus_1_5'],
            'implied_prob': odds_to_probability(odds_data['team1_minus_1_5_odds']),
            'edge': score_probs['team1_minus_1_5'] - odds_to_probability(odds_data['team1_minus_1_5_odds']),
            'decimal_odds': odds_data['team1_minus_1_5_odds'],
            'american_odds': decimal_to_american(odds_data['team1_minus_1_5_odds']),
            'ev': team1_minus_1_5_ev,
            'kelly_fraction': team1_minus_1_5_kelly,
            'recommended_bet': team1_minus_1_5_kelly * bankroll if team1_minus_1_5_ev > min_ev else 0
        },
        {
            'type': 'Handicap -1.5',
            'selection': team2_name,
            'model_prob': score_probs['team2_minus_1_5'],
            'implied_prob': odds_to_probability(odds_data['team2_minus_1_5_odds']),
            'edge': score_probs['team2_minus_1_5'] - odds_to_probability(odds_data['team2_minus_1_5_odds']),
            'decimal_odds': odds_data['team2_minus_1_5_odds'],
            'american_odds': decimal_to_american(odds_data['team2_minus_1_5_odds']),
            'ev': team2_minus_1_5_ev,
            'kelly_fraction': team2_minus_1_5_kelly,
            'recommended_bet': team2_minus_1_5_kelly * bankroll if team2_minus_1_5_ev > min_ev else 0
        },
        {
            'type': 'Total Maps',
            'selection': 'Over 2.5',
            'model_prob': score_probs['over_2_5_maps'],
            'implied_prob': odds_to_probability(odds_data['over_2_5_odds']),
            'edge': score_probs['over_2_5_maps'] - odds_to_probability(odds_data['over_2_5_odds']),
            'decimal_odds': odds_data['over_2_5_odds'],
            'american_odds': decimal_to_american(odds_data['over_2_5_odds']),
            'ev': over_2_5_ev,
            'kelly_fraction': over_2_5_kelly,
            'recommended_bet': over_2_5_kelly * bankroll if over_2_5_ev > min_ev else 0
        },
        {
            'type': 'Total Maps',
            'selection': 'Under 2.5',
            'model_prob': score_probs['under_2_5_maps'],
            'implied_prob': odds_to_probability(odds_data['under_2_5_odds']),
            'edge': score_probs['under_2_5_maps'] - odds_to_probability(odds_data['under_2_5_odds']),
            'decimal_odds': odds_data['under_2_5_odds'],
            'american_odds': decimal_to_american(odds_data['under_2_5_odds']),
            'ev': under_2_5_ev,
            'kelly_fraction': under_2_5_kelly,
            'recommended_bet': under_2_5_kelly * bankroll if under_2_5_ev > min_ev else 0
        }
    ]
    
    # Sort by expected value
    bets_analysis.sort(key=lambda x: x['ev'], reverse=True)
    
    # Count recommended bets
    recommended_count = sum(1 for bet in bets_analysis if bet['recommended_bet'] > 0)
    
    # Create detailed analysis report
    analysis_report = {
        'matchup': f"{team1_name} vs {team2_name}",
        'prediction_time': win_prediction['prediction_time'],
        'team1_name': team1_name,
        'team2_name': team2_name,
        'team1_win_probability': score_probs['team1_win'],
        'team2_win_probability': score_probs['team2_win'],
        'team1_2_0_probability': score_probs[f"{team1_name} 2-0"],
        'team1_2_1_probability': score_probs[f"{team1_name} 2-1"],
        'team2_2_0_probability': score_probs[f"{team2_name} 2-0"],
        'team2_2_1_probability': score_probs[f"{team2_name} 2-1"],
        'maps_2_probability': score_probs['under_2_5_maps'],
        'maps_3_probability': score_probs['over_2_5_maps'],
        'odds': odds_data,
        'bets': bets_analysis,
        'recommended_count': recommended_count,
        'ensemble_std': win_prediction['ensemble_std'],
        'data_quality': win_prediction['data_quality'],
        'sample_adequacy': win_prediction['sample_adequacy'],
        'historical_accuracy': win_prediction['historical_accuracy'],
        'team1_matches_count': win_prediction['team1_matches_count'],
        'team2_matches_count': win_prediction['team2_matches_count'],
        'bankroll': bankroll,
        'min_ev_threshold': min_ev,
        'kelly_fraction': kelly_fraction,
        'used_score_model': score_probs is not None
    }
    
    return analysis_report

def find_value_bets(all_teams=None, min_ev=0.05, kelly_fraction=0.25, bankroll=1000):
    """Find value betting opportunities across multiple matchups."""
    if all_teams is None:
        # Fetch all teams
        print("Fetching all teams...")
        teams_response = requests.get(f"{API_URL}/teams?limit=300")
        if teams_response.status_code != 200:
            print(f"Error fetching teams: {teams_response.status_code}")
            return None
        
        teams_data = teams_response.json()
        
        if 'data' not in teams_data:
            print("No teams data found.")
            return None
        
        all_teams = teams_data['data']
    
    # Get top teams (by ranking)
    top_teams = []
    for team in all_teams:
        if 'ranking' in team and team['ranking'] and team['ranking'] <= 50:
            top_teams.append(team)
    
    # If no teams with rankings were found, just use the first 30 teams
    if not top_teams:
        print("No teams with rankings found. Using a subset of teams instead.")
        top_teams = all_teams[:30]
    
    print(f"Selected {len(top_teams)} teams for analysis.")
    
    # Fetch upcoming matches
    upcoming_matches = fetch_upcoming_matches()
    
    if not upcoming_matches:
        print("No upcoming matches found.")
        
        # Simulate matches between top teams
        print("Simulating potential matchups between top teams...")
        simulated_matchups = []
        
        for i in range(len(top_teams)):
            for j in range(i + 1, len(top_teams)):
                team1 = top_teams[i]
                team2 = top_teams[j]
                
                simulated_matchups.append({
                    'team1': {'id': team1['id'], 'name': team1['name']},
                    'team2': {'id': team2['id'], 'name': team2['name']},
                    'is_simulated': True
                })
        
        upcoming_matches = simulated_matchups[:20]  # Limit to 20 simulated matchups
    
    # Analyze each matchup
    value_bets = []
    
    for match in tqdm(upcoming_matches, desc="Analyzing matches for value bets"):
        try:
            team1_name = match.get('team1', {}).get('name', '')
            team2_name = match.get('team2', {}).get('name', '')
            
            if not team1_name or not team2_name:
                continue
            
            print(f"\nAnalyzing {team1_name} vs {team2_name}")
            
            # Simulate booking odds
            # In a real scenario, you would fetch these from an API
            # Here we'll use simulated odds based on reasonable market values
            
            # Get win prediction to estimate odds
            prediction = predict_with_ensemble(team1_name, team2_name)
            
            if not prediction:
                print(f"Could not generate prediction for {team1_name} vs {team2_name}")
                continue
            
            team1_prob = prediction['team1_win_probability']
            team2_prob = prediction['team2_win_probability']
            
            # Add a 5% margin to probabilities to simulate bookmaker edge
            margin = 0.05
            team1_implied = min(0.95, team1_prob * (1 + margin))
            team2_implied = min(0.95, team2_prob * (1 + margin))
            
            # Normalize implied probabilities to >100% total (bookmaker margin)
            total_implied = team1_implied + team2_implied
            if total_implied < 1.05:
                team1_implied = team1_implied * 1.05 / total_implied
                team2_implied = team2_implied * 1.05 / total_implied
            
            # Convert to odds
            sim_odds = {
                'ml_odds_team1': min(10.0, max(1.05, 1 / team1_implied)),
                'ml_odds_team2': min(10.0, max(1.05, 1 / team2_implied)),
                'team1_plus_1_5_odds': min(3.0, max(1.05, 1 / (0.65 + 0.25 * team1_prob))),
                'team2_plus_1_5_odds': min(3.0, max(1.05, 1 / (0.65 + 0.25 * team2_prob))),
                'team1_minus_1_5_odds': min(10.0, max(1.2, 1 / (0.4 * team1_prob))),
                'team2_minus_1_5_odds': min(10.0, max(1.2, 1 / (0.4 * team2_prob))),
                'over_2_5_odds': max(1.05, min(5.0, 1 / (0.3 + 0.4 * min(team1_prob, team2_prob)))),
                'under_2_5_odds': max(1.05, min(5.0, 1 / (0.6 - 0.2 * min(team1_prob, team2_prob))))
            }
            
            # Analyze match with simulated odds
            analysis = analyze_match_betting_with_score_model(
                team1_name, 
                team2_name, 
                sim_odds,
                bankroll=bankroll,
                min_ev=min_ev,
                kelly_fraction=kelly_fraction
            )
            
            if not analysis:
                continue
            
            # Filter for value bets
            value_found = False
            
            for bet in analysis['bets']:
                if bet['ev'] > min_ev:
                    value_found = True
                    value_bets.append({
                        'team1': team1_name,
                        'team2': team2_name,
                        'bet_type': bet['type'],
                        'selection': bet['selection'],
                        'model_prob': bet['model_prob'],
                        'decimal_odds': bet['decimal_odds'],
                        'american_odds': bet['american_odds'],
                        'expected_value': bet['ev'],
                        'kelly_stake': bet['kelly_fraction'],
                        'recommended_bet': bet['recommended_bet'],
                        'simulated_odds': 'is_simulated' in match,
                        'match_date': match.get('date', 'Unknown')
                    })
            
            if value_found:
                print(f"Found value bets for {team1_name} vs {team2_name}")
            
        except Exception as e:
            print(f"Error analyzing match: {e}")
            continue
    
    # Sort value bets by expected value
    value_bets.sort(key=lambda x: x['expected_value'], reverse=True)
    
    # Create a summary report
    value_report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_matches_analyzed': len(upcoming_matches),
        'value_bets_found': len(value_bets),
        'min_ev_threshold': min_ev,
        'kelly_fraction': kelly_fraction,
        'bankroll': bankroll,
        'value_bets': value_bets
    }
    
    # Save report to file
    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'value_bets_report_{report_time}.json', 'w') as f:
        json.dump(value_report, f, indent=2)
    
    # Create a readable CSV version too
    if value_bets:
        df = pd.DataFrame(value_bets)
        df.to_csv(f'value_bets_report_{report_time}.csv', index=False)
    
    return value_report

def build_betting_dashboard(bankroll=1000, min_ev=0.05, kelly_fraction=0.25):
    """Build a comprehensive betting dashboard with statistics and visualizations."""
    # Check for bet history
    history_file = 'bet_history.csv'
    results_file = 'match_results.csv'
    
    if not os.path.isfile(history_file):
        print(f"No bet history found at {history_file}")
        # Create empty history file with proper headers
        with open(history_file, 'w', newline='') as f:
            headers = ['bet_id', 'timestamp', 'match_id', 'match_date', 'team1', 'team2', 
                      'bet_type', 'selection', 'predicted_probability', 'implied_probability',
                      'decimal_odds', 'bet_amount', 'expected_value', 'kelly_fraction',
                      'outcome', 'profit_loss', 'result_timestamp', 'notes']
            writer = pd.DataFrame(columns=headers)
            writer.to_csv(f, index=False)
    
    # Create results file if needed
    if not os.path.isfile(results_file):
        print(f"No match results found at {results_file}")
        # Create empty results file with proper headers
        with open(results_file, 'w', newline='') as f:
            headers = ['match_id', 'timestamp', 'team1', 'team2', 'team1_score', 'team2_score',
                      'winner', 'score_line', 'maps_played', 'team1_ml_result', 'team2_ml_result',
                      'team1_plus_1_5_result', 'team2_plus_1_5_result', 'team1_minus_1_5_result',
                      'team2_minus_1_5_result', 'over_2_5_maps_result', 'under_2_5_maps_result']
            writer = pd.DataFrame(columns=headers)
            writer.to_csv(f, index=False)
    
    # Load betting history
    try:
        bet_history = pd.read_csv(history_file)
        has_history = len(bet_history) > 0
    except Exception as e:
        print(f"Error loading bet history: {e}")
        bet_history = pd.DataFrame()
        has_history = False
    
    # Calculate overall performance metrics
    performance_metrics = {}
    
    if has_history:
        # Filter for bets with known outcomes
        completed_bets = bet_history[bet_history['outcome'].notna()]
        
        if len(completed_bets) > 0:
            # Overall metrics
            total_bets = len(completed_bets)
            winning_bets = len(completed_bets[completed_bets['outcome'] == 'win'])
            total_wagered = completed_bets['bet_amount'].sum()
            total_profit = completed_bets['profit_loss'].sum()
            
            performance_metrics['total_bets'] = total_bets
            performance_metrics['winning_bets'] = winning_bets
            performance_metrics['win_rate'] = winning_bets / total_bets if total_bets > 0 else 0
            performance_metrics['total_wagered'] = total_wagered
            performance_metrics['total_profit'] = total_profit
            performance_metrics['roi'] = total_profit / total_wagered if total_wagered > 0 else 0
            
            # Performance by bet type
            bet_types = completed_bets['bet_type'].unique()
            performance_by_type = {}
            
            for bet_type in bet_types:
                type_bets = completed_bets[completed_bets['bet_type'] == bet_type]
                type_wins = len(type_bets[type_bets['outcome'] == 'win'])
                type_wagered = type_bets['bet_amount'].sum()
                type_profit = type_bets['profit_loss'].sum()
                
                performance_by_type[bet_type] = {
                    'bets': len(type_bets),
                    'wins': type_wins,
                    'win_rate': type_wins / len(type_bets) if len(type_bets) > 0 else 0,
                    'wagered': type_wagered,
                    'profit': type_profit,
                    'roi': type_profit / type_wagered if type_wagered > 0 else 0
                }
            
            performance_metrics['by_type'] = performance_by_type
            
            # Performance by EV range
            completed_bets['ev_range'] = pd.cut(
                completed_bets['expected_value'],
                bins=[-float('inf'), 0.05, 0.1, 0.15, 0.2, float('inf')],
                labels=['0-5%', '5-10%', '10-15%', '15-20%', '20%+']
            )
            
            performance_by_ev = {}
            for ev_range in completed_bets['ev_range'].unique():
                if pd.isna(ev_range):
                    continue
                    
                ev_bets = completed_bets[completed_bets['ev_range'] == ev_range]
                ev_wins = len(ev_bets[ev_bets['outcome'] == 'win'])
                ev_wagered = ev_bets['bet_amount'].sum()
                ev_profit = ev_bets['profit_loss'].sum()
                
                performance_by_ev[str(ev_range)] = {
                    'bets': len(ev_bets),
                    'wins': ev_wins,
                    'win_rate': ev_wins / len(ev_bets) if len(ev_bets) > 0 else 0,
                    'wagered': ev_wagered,
                    'profit': ev_profit,
                    'roi': ev_profit / ev_wagered if ev_wagered > 0 else 0
                }
            
            performance_metrics['by_ev'] = performance_by_ev
            
            # Calculate profit over time
            completed_bets = completed_bets.sort_values('timestamp')
            completed_bets['cumulative_profit'] = completed_bets['profit_loss'].cumsum()
            
            performance_metrics['profit_over_time'] = {
                'timestamps': completed_bets['timestamp'].tolist(),
                'cumulative_profit': completed_bets['cumulative_profit'].tolist()
            }
            
            # Model calibration - predicted probability vs. actual win rate
            completed_bets['prob_range'] = pd.cut(
                completed_bets['predicted_probability'],
                bins=[0, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
                labels=['0-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75-80%', '80-85%', '85-90%', '90-95%', '95-100%']
            )
            
            calibration_data = {}
            for prob_range in completed_bets['prob_range'].unique():
                if pd.isna(prob_range):
                    continue
                    
                range_bets = completed_bets[completed_bets['prob_range'] == prob_range]
                range_wins = len(range_bets[range_bets['outcome'] == 'win'])
                
                calibration_data[str(prob_range)] = {
                    'bets': len(range_bets),
                    'wins': range_wins,
                    'actual_win_rate': range_wins / len(range_bets) if len(range_bets) > 0 else 0,
                    'predicted_probability': range_bets['predicted_probability'].mean()
                }
            
            performance_metrics['calibration'] = calibration_data
    
    # Find value betting opportunities
    print("Looking for current value betting opportunities...")
    value_report = find_value_bets(min_ev=min_ev, kelly_fraction=kelly_fraction, bankroll=bankroll)
    
    # Create dashboard
    dashboard_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'current_bankroll': bankroll,
        'performance_metrics': performance_metrics,
        'value_bets': value_report['value_bets'] if value_report else [],
        'settings': {
            'min_ev': min_ev,
            'kelly_fraction': kelly_fraction
        }
    }
    
    # Save dashboard data
    dashboard_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'betting_dashboard_{dashboard_time}.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    # Create visualizations
    create_dashboard_visualizations(dashboard_data, f'betting_dashboard_{dashboard_time}.png')
    
    return dashboard_data

def create_dashboard_visualizations(dashboard_data, save_path):
    """Create visualizations for the betting dashboard."""
    has_performance_data = (
        'performance_metrics' in dashboard_data and 
        dashboard_data['performance_metrics'] and
        'total_bets' in dashboard_data['performance_metrics'] and
        dashboard_data['performance_metrics']['total_bets'] > 0
    )
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 16))
    
    # Define grid layout
    if has_performance_data:
        # Comprehensive dashboard with performance data
        gs = fig.add_gridspec(3, 3)
    else:
        # Simple dashboard with only value bets
        gs = fig.add_gridspec(1, 2)
    
    # Title and timestamp
    fig.suptitle(f"Valorant Betting Dashboard", fontsize=24, y=0.98)
    fig.text(0.5, 0.96, f"Generated on {dashboard_data['timestamp']}", fontsize=14, ha='center')
    
    if has_performance_data:
        # Performance metrics
        metrics = dashboard_data['performance_metrics']
        
        # 1. Overall metrics summary
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        ax1.set_title("Overall Performance", fontsize=18)
        
        summary_text = (
            f"Total Bets: {metrics['total_bets']}\n"
            f"Winning Bets: {metrics['winning_bets']}\n"
            f"Win Rate: {metrics['win_rate']:.2%}\n"
            f"Total Wagered: ${metrics['total_wagered']:.2f}\n"
            f"Total Profit: ${metrics['total_profit']:.2f}\n"
            f"Return on Investment: {metrics['roi']:.2%}"
        )
        
        ax1.text(0.5, 0.5, summary_text, fontsize=14, ha='center', va='center')
        
        # 2. Profit over time
        if 'profit_over_time' in metrics and metrics['profit_over_time']['timestamps']:
            ax2 = fig.add_subplot(gs[0, 1:])
            ax2.plot(metrics['profit_over_time']['cumulative_profit'], marker='o')
            ax2.set_title("Cumulative Profit Over Time", fontsize=18)
            ax2.set_ylabel("Profit ($)")
            ax2.set_xlabel("Bet Number")
            ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
        
        # 3. Performance by bet type
        if 'by_type' in metrics and metrics['by_type']:
            ax3 = fig.add_subplot(gs[1, 0])
            
            bet_types = list(metrics['by_type'].keys())
            bet_counts = [metrics['by_type'][bt]['bets'] for bt in bet_types]
            win_rates = [metrics['by_type'][bt]['win_rate'] for bt in bet_types]
            rois = [metrics['by_type'][bt]['roi'] for bt in bet_types]
            
            width = 0.35
            x = np.arange(len(bet_types))
            
            ax3.bar(x, win_rates, width, label='Win Rate')
            ax3.bar(x + width, rois, width, label='ROI')
            
            ax3.set_title("Performance by Bet Type", fontsize=18)
            ax3.set_xticks(x + width / 2)
            ax3.set_xticklabels(bet_types)
            ax3.legend()
            ax3.set_ylim(-0.5, 1.5)
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            
            # Add bet count annotations
            for i, count in enumerate(bet_counts):
                ax3.annotate(f"n={count}", xy=(i + width/2, 0.05), ha='center')
        
        # 4. Model calibration
        if 'calibration' in metrics and metrics['calibration']:
            ax4 = fig.add_subplot(gs[1, 1:])
            
            prob_ranges = list(metrics['calibration'].keys())
            pred_probs = [metrics['calibration'][pr]['predicted_probability'] for pr in prob_ranges]
            actual_rates = [metrics['calibration'][pr]['actual_win_rate'] for pr in prob_ranges]
            bet_counts = [metrics['calibration'][pr]['bets'] for pr in prob_ranges]
            
            # Size points by bet count
            sizes = [max(50, min(500, count * 10)) for count in bet_counts]
            
            # Perfect calibration line
            ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            # Actual calibration points
            ax4.scatter(pred_probs, actual_rates, s=sizes, alpha=0.7)
            
            # Annotate points with count
            for i, (x, y, count) in enumerate(zip(pred_probs, actual_rates, bet_counts)):
                ax4.annotate(f"n={count}", xy=(x, y), xytext=(5, 5), textcoords='offset points')
            
            ax4.set_title("Model Calibration: Predicted vs. Actual Win Rate", fontsize=18)
            ax4.set_xlabel("Predicted Probability")
            ax4.set_ylabel("Actual Win Rate")
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
        
        # 5. Performance by EV range
        if 'by_ev' in metrics and metrics['by_ev']:
            ax5 = fig.add_subplot(gs[2, 0:2])
            
            ev_ranges = list(metrics['by_ev'].keys())
            ev_win_rates = [metrics['by_ev'][ev]['win_rate'] for ev in ev_ranges]
            ev_rois = [metrics['by_ev'][ev]['roi'] for ev in ev_ranges]
            ev_bet_counts = [metrics['by_ev'][ev]['bets'] for ev in ev_ranges]
            
            x = np.arange(len(ev_ranges))
            width = 0.35
            
            bars1 = ax5.bar(x - width/2, ev_win_rates, width, label='Win Rate')
            bars2 = ax5.bar(x + width/2, ev_rois, width, label='ROI')
            
            ax5.set_title("Performance by Expected Value Range", fontsize=18)
            ax5.set_xticks(x)
            ax5.set_xticklabels(ev_ranges)
            ax5.legend()
            ax5.set_ylim(-0.5, 1.5)
            ax5.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            
            # Add bet count annotations
            for i, count in enumerate(ev_bet_counts):
                ax5.annotate(f"n={count}", xy=(i, -0.1), ha='center')
    
    # Value betting opportunities
    if 'value_bets' in dashboard_data and dashboard_data['value_bets']:
        value_bets = dashboard_data['value_bets']
        
        if has_performance_data:
            ax6 = fig.add_subplot(gs[2, 2])
        else:
            ax6 = fig.add_subplot(gs[0, 0:])
        
        ax6.axis('off')
        ax6.set_title("Top Value Betting Opportunities", fontsize=18)
        
        # Show top 5 value bets
        top_bets = sorted(value_bets, key=lambda x: x['expected_value'], reverse=True)[:5]
        
        table_data = []
        for i, bet in enumerate(top_bets):
            row = [
                f"{bet['team1']} vs {bet['team2']}",
                f"{bet['bet_type']} - {bet['selection']}",
                f"{bet['decimal_odds']:.2f}",
                f"{bet['expected_value']:.2%}",
                f"${bet['recommended_bet']:.2f}"
            ]
            table_data.append(row)
        
        if table_data:
            col_labels = ["Match", "Bet Type", "Odds", "EV", "Bet Size"]
            table = ax6.table(
                cellText=table_data,
                colLabels=col_labels,
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
        else:
            ax6.text(0.5, 0.5, "No value betting opportunities found", 
                    ha='center', va='center', fontsize=14)
    else:
        if has_performance_data:
            ax6 = fig.add_subplot(gs[2, 2])
        else:
            ax6 = fig.add_subplot(gs[0, 1])
            
        ax6.axis('off')
        ax6.set_title("Value Betting Opportunities", fontsize=18)
        ax6.text(0.5, 0.5, "No value betting opportunities found", 
                ha='center', va='center', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dashboard visualization saved to {save_path}")

#-------------------------------------------------------------------------
# CALIBRATION AND MODEL ADJUSTMENT FUNCTIONS
#-------------------------------------------------------------------------

def calibrate_winning_probabilities(model_probabilities, calibration_data=None):
    """Apply calibration adjustments to model probabilities based on historical performance."""
    if not calibration_data:
        # Load default calibration data if available
        if os.path.exists('probability_calibration.json'):
            with open('probability_calibration.json', 'r') as f:
                try:
                    calibration_data = json.load(f)
                except:
                    print("Error loading calibration data. Using raw probabilities.")
                    return model_probabilities
        else:
            # No calibration data available
            return model_probabilities
    
    calibrated_probs = model_probabilities.copy()
    
    # Apply calibration if we have data
    if 'probability_ranges' in calibration_data:
        # Find the appropriate calibration factor for this probability
        win_prob = model_probabilities.get('team1_win_probability', 0.5)
        
        # Find which range this probability falls into
        for range_data in calibration_data['probability_ranges']:
            min_prob = range_data.get('min_prob', 0)
            max_prob = range_data.get('max_prob', 1)
            
            if min_prob <= win_prob <= max_prob:
                # Apply the calibration adjustment
                adjustment = range_data.get('adjustment', 0)
                
                # Adjust the win probabilities
                calibrated_probs['team1_win_probability'] = min(1.0, max(0.0, win_prob + adjustment))
                calibrated_probs['team2_win_probability'] = 1.0 - calibrated_probs['team1_win_probability']
                
                # Log the adjustment
                print(f"Applied calibration adjustment of {adjustment:+.2%} to win probability")
                break
    
    # Apply score calibration if available
    if 'score_calibration' in calibration_data and 'team1_2_0_probability' in model_probabilities:
        # Adjust score distribution while preserving the total win probability
        team1_win_prob = calibrated_probs['team1_win_probability']
        team2_win_prob = calibrated_probs['team2_win_probability']
        
        # Get score distribution adjustments
        score_cal = calibration_data['score_calibration']
        
        # Adjust 2-0 vs 2-1 distribution for team1
        team1_2_0_adj = score_cal.get('team1_2_0_adjustment', 0)
        
        # Current 2-0 probability
        current_2_0 = model_probabilities.get('team1_2_0_probability', team1_win_prob * 0.6)
        
        # Calculate new 2-0 probability, ensuring it stays within team1's total win probability
        new_2_0 = min(team1_win_prob, max(0, current_2_0 + team1_2_0_adj))
        
        # Update 2-0 and 2-1 probabilities
        calibrated_probs['team1_2_0_probability'] = new_2_0
        calibrated_probs['team1_2_1_probability'] = team1_win_prob - new_2_0
        
        # Do the same for team2
        team2_2_0_adj = score_cal.get('team2_2_0_adjustment', 0)
        current_2_0 = model_probabilities.get('team2_2_0_probability', team2_win_prob * 0.6)
        new_2_0 = min(team2_win_prob, max(0, current_2_0 + team2_2_0_adj))
        
        calibrated_probs['team2_2_0_probability'] = new_2_0
        calibrated_probs['team2_2_1_probability'] = team2_win_prob - new_2_0
        
        # Update derived probabilities
        calibrated_probs['maps_2_probability'] = calibrated_probs['team1_2_0_probability'] + calibrated_probs['team2_2_0_probability']
        calibrated_probs['maps_3_probability'] = calibrated_probs['team1_2_1_probability'] + calibrated_probs['team2_2_1_probability']
        
        # Update handicap probabilities
        calibrated_probs['team1_plus_1_5'] = team1_win_prob + calibrated_probs['team2_2_1_probability']
        calibrated_probs['team2_plus_1_5'] = team2_win_prob + calibrated_probs['team1_2_1_probability']
        calibrated_probs['team1_minus_1_5'] = calibrated_probs['team1_2_0_probability']
        calibrated_probs['team2_minus_1_5'] = calibrated_probs['team2_2_0_probability']
        
        print(f"Applied score distribution calibration")
    
    return calibrated_probs

def generate_calibration_data(historical_bets, save_file='probability_calibration.json'):
    """Generate calibration data from historical betting results."""
    if len(historical_bets) < 20:
        print("Not enough historical data for meaningful calibration (minimum 20 bets required).")
        return None
    
    # Group bets by probability range
    historical_bets['prob_range'] = pd.cut(
        historical_bets['predicted_probability'],
        bins=[0, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
        labels=['0-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75-80%', '80-85%', '85-90%', '90-95%', '95-100%']
    )
    
    # Calculate the calibration data
    calibration_ranges = []
    
    for prob_range in historical_bets['prob_range'].unique():
        if pd.isna(prob_range):
            continue
            
        range_bets = historical_bets[historical_bets['prob_range'] == prob_range]
        
        if len(range_bets) < 5:
            # Skip ranges with too few samples
            continue
            
        range_wins = len(range_bets[range_bets['outcome'] == 'win'])
        actual_win_rate = range_wins / len(range_bets) if len(range_bets) > 0 else 0
        avg_predicted_prob = range_bets['predicted_probability'].mean()
        
        # Calculate needed adjustment
        adjustment = actual_win_rate - avg_predicted_prob
        
        # Parse the range string to get min/max
        range_str = str(prob_range)
        if '-' in range_str:
            min_str, max_str = range_str.split('-')
            min_prob = float(min_str.strip('%')) / 100
            max_prob = float(max_str.strip('%')) / 100
        else:
            # Handle edge cases
            min_prob = 0
            max_prob = 1
        
        calibration_ranges.append({
            'range': range_str,
            'min_prob': min_prob,
            'max_prob': max_prob,
            'bets': len(range_bets),
            'wins': range_wins,
            'actual_win_rate': actual_win_rate,
            'predicted_probability': avg_predicted_prob,
            'adjustment': adjustment
        })
    
    # Generate calibration for score distribution if we have that data
    score_calibration = {}
    
    # Check if we have score data in the historical results
    if 'actual_score' in historical_bets.columns:
        # Calculate 2-0 vs 2-1 distribution
        team1_wins = historical_bets[historical_bets['outcome'] == 'win']
        
        if len(team1_wins) > 10:
            team1_2_0_actual = len(team1_wins[team1_wins['actual_score'] == '2-0']) / len(team1_wins)
            team1_2_0_predicted = team1_wins['team1_2_0_probability'].mean() / team1_wins['predicted_probability'].mean()
            
            team1_2_0_adjustment = team1_2_0_actual - team1_2_0_predicted
            
            score_calibration['team1_2_0_adjustment'] = team1_2_0_adjustment
    
    # Compile the final calibration data
    calibration_data = {
        'generated_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'data_points': len(historical_bets),
        'probability_ranges': calibration_ranges,
        'score_calibration': score_calibration
    }
    
    # Save to file
    with open(save_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"Calibration data generated and saved to {save_file}")
    
    return calibration_data

def analyze_model_accuracy(historical_matches, prediction_records=None):
    """Analyze the model's prediction accuracy over time to check for degradation."""
    if not historical_matches:
        print("No historical match data available for analysis.")
        return None
    
    # If we have specific prediction records, use those
    # Otherwise, regenerate predictions for historical matches
    if not prediction_records:
        print("Generating predictions for historical matches...")
        predictions = []
        
        for match in tqdm(historical_matches):
            team1_name = match['team1']
            team2_name = match['team2']
            
            # Generate prediction
            prediction = predict_with_ensemble(team1_name, team2_name)
            
            if prediction:
                # Record prediction and actual result
                team1_score = match.get('team1_score', 0)
                team2_score = match.get('team2_score', 0)
                actual_winner = team1_name if team1_score > team2_score else team2_name
                predicted_winner = team1_name if prediction['team1_win_probability'] > 0.5 else team2_name
                
                predictions.append({
                    'match_id': match.get('match_id', ''),
                    'date': match.get('timestamp', ''),
                    'team1': team1_name,
                    'team2': team2_name,
                    'predicted_winner': predicted_winner,
                    'actual_winner': actual_winner,
                    'prediction_correct': predicted_winner == actual_winner,
                    'team1_win_probability': prediction['team1_win_probability'],
                    'team2_win_probability': prediction['team2_win_probability'],
                    'team1_score': team1_score,
                    'team2_score': team2_score
                })
    else:
        predictions = prediction_records
    
    if not predictions:
        print("No predictions available for analysis.")
        return None
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(predictions)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate overall accuracy
    overall_accuracy = df['prediction_correct'].mean()
    
    # Calculate accuracy over time (monthly)
    df['month'] = df['date'].dt.to_period('M')
    monthly_acc = df.groupby('month')['prediction_correct'].mean()
    
    # Calculate monthly count
    monthly_count = df.groupby('month').size()
    
    # Calculate monthly accuracy for different confidence levels
    df['confidence'] = df.apply(lambda row: max(row['team1_win_probability'], row['team2_win_probability']), axis=1)
    
    # High confidence predictions (>65%)
    high_conf = df[df['confidence'] > 0.65]
    high_conf_acc = high_conf['prediction_correct'].mean() if len(high_conf) > 0 else 0
    
    # Very high confidence predictions (>80%)
    very_high_conf = df[df['confidence'] > 0.8]
    very_high_conf_acc = very_high_conf['prediction_correct'].mean() if len(very_high_conf) > 0 else 0
    
    # Calculate recent accuracy (last 20 predictions)
    recent_acc = df.tail(20)['prediction_correct'].mean() if len(df) >= 20 else df['prediction_correct'].mean()
    
    # Check for model degradation
    recent_20pct = int(len(df) * 0.2)
    early_acc = df.head(recent_20pct)['prediction_correct'].mean() if recent_20pct > 0 else 0
    recent_acc_trend = df.tail(recent_20pct)['prediction_correct'].mean() if recent_20pct > 0 else 0
    
    degradation_detected = (recent_acc_trend < early_acc * 0.9) and (recent_20pct >= 10)
    
    # Compile results
    analysis_results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_predictions': len(df),
        'overall_accuracy': overall_accuracy,
        'high_confidence_accuracy': high_conf_acc,
        'very_high_confidence_accuracy': very_high_conf_acc,
        'recent_accuracy': recent_acc,
        'degradation_detected': degradation_detected,
        'early_accuracy': early_acc,
        'recent_trend_accuracy': recent_acc_trend,
        'monthly_accuracy': {str(month): acc for month, acc in monthly_acc.items()},
        'monthly_count': {str(month): count for month, count in monthly_count.items()}
    }
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot monthly accuracy with count overlay
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(range(len(monthly_acc)), monthly_acc.values, marker='o', linestyle='-', linewidth=2)
    ax1.set_title('Monthly Prediction Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    ax1.axhline(y=overall_accuracy, color='g', linestyle='--', alpha=0.5, label=f'Overall: {overall_accuracy:.2f}')
    ax1.grid(True, alpha=0.3)
    
    # Twin axis for count
    ax1b = ax1.twinx()
    ax1b.bar(range(len(monthly_count)), monthly_count.values, alpha=0.2, color='gray')
        # Calculate overall performance metrics
    performance_metrics = {}
    
    if has_history:
        # Filter for bets with known outcomes
        completed_bets = bet_history[bet_history['outcome'].notna()]
        
        if len(completed_bets) > 0:
            # Overall metrics
            total_bets = len(completed_bets)
            winning_bets = len(completed_bets[completed_bets['outcome'] == 'win'])
            total_wagered = completed_bets['bet_amount'].sum()
            total_profit = completed_bets['profit_loss'].sum()
            
            performance_metrics['total_bets'] = total_bets
            performance_metrics['winning_bets'] = winning_bets
            performance_metrics['win_rate'] = winning_bets / total_bets if total_bets > 0 else 0
            performance_metrics['total_wagered'] = total_wagered
            performance_metrics['total_profit'] = total_profit
            performance_metrics['roi'] = total_profit / total_wagered if total_wagered > 0 else 0
            
            # Performance by bet type
            bet_types = completed_bets['bet_type'].unique()
            performance_by_type = {}
            
            for bet_type in bet_types:
                type_bets = completed_bets[completed_bets['bet_type'] == bet_type]
                type_wins = len(type_bets[type_bets['outcome'] == 'win'])
                type_wagered = type_bets['bet_amount'].sum()
                type_profit = type_bets['profit_loss'].sum()
                
                performance_by_type[bet_type] = {
                    'bets': len(type_bets),
                    'wins': type_wins,
                    'win_rate': type_wins / len(type_bets) if len(type_bets) > 0 else 0,
                    'wagered': type_wagered,
                    'profit': type_profit,
                    'roi': type_profit / type_wagered if type_wagered > 0 else 0
                }
            
            performance_metrics['by_type'] = performance_by_type
            
            # Performance by EV range
            completed_bets['ev_range'] = pd.cut(
                completed_bets['expected_value'],
                bins=[-float('inf'), 0.05, 0.1, 0.15, 0.2, float('inf')],
                labels=['0-5%', '5-10%', '10-15%', '15-20%', '20%+']
            )
            
            performance_by_ev = {}
            for ev_range in completed_bets['ev_range'].unique():
                if pd.isna(ev_range):
                    continue
                    
                ev_bets = completed_bets[completed_bets['ev_range'] == ev_range]
                ev_wins = len(ev_bets[ev_bets['outcome'] == 'win'])
                ev_wagered = ev_bets['bet_amount'].sum()
                ev_profit = ev_bets['profit_loss'].sum()
                
                performance_by_ev[str(ev_range)] = {
                    'bets': len(ev_bets),
                    'wins': ev_wins,
                    'win_rate': ev_wins / len(ev_bets) if len(ev_bets) > 0 else 0,
                    'wagered': ev_wagered,
                    'profit': ev_profit,
                    'roi': ev_profit / ev_wagered if ev_wagered > 0 else 0
                }
            
            performance_metrics['by_ev'] = performance_by_ev
            
            # Calculate profit over time
            completed_bets = completed_bets.sort_values('timestamp')
            completed_bets['cumulative_profit'] = completed_bets['profit_loss'].cumsum()
            
            performance_metrics['profit_over_time'] = {
                'timestamps': completed_bets['timestamp'].tolist(),
                'cumulative_profit': completed_bets['cumulative_profit'].tolist()
            }
            
            # Model calibration - predicted probability vs. actual win rate
            completed_bets['prob_range'] = pd.cut(
                completed_bets['predicted_probability'],
                bins=[0, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
                labels=['0-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75-80%', '80-85%', '85-90%', '90-95%', '95-100%']
            )
            
            calibration_data = {}
            for prob_range in completed_bets['prob_range'].unique():
                if pd.isna(prob_range):
                    continue
                    
                range_bets = completed_bets[completed_bets['prob_range'] == prob_range]
                range_wins = len(range_bets[range_bets['outcome'] == 'win'])
                
                calibration_data[str(prob_range)] = {
                    'bets': len(range_bets),
                    'wins': range_wins,
                    'actual_win_rate': range_wins / len(range_bets) if len(range_bets) > 0 else 0,
                    'predicted_probability': range_bets['predicted_probability'].mean()
                }
            
            performance_metrics['calibration'] = calibration_data
    
    # Find value betting opportunities
    print("Looking for current value betting opportunities...")
    value_report = find_value_bets(min_ev=min_ev, kelly_fraction=kelly_fraction, bankroll=bankroll)
    
    # Create dashboard
    dashboard_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'current_bankroll': bankroll,
        'performance_metrics': performance_metrics,
        'value_bets': value_report['value_bets'] if value_report else [],
        'settings': {
            'min_ev': min_ev,
            'kelly_fraction': kelly_fraction
        }
    }
    
    # Save dashboard data
    dashboard_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'betting_dashboard_{dashboard_time}.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    # Create visualizations
    create_dashboard_visualizations(dashboard_data, f'betting_dashboard_{dashboard_time}.png')
    
    return dashboard_data

def create_dashboard_visualizations(dashboard_data, save_path):
    """Create visualizations for the betting dashboard."""
    has_performance_data = (
        'performance_metrics' in dashboard_data and 
        dashboard_data['performance_metrics'] and
        'total_bets' in dashboard_data['performance_metrics'] and
        dashboard_data['performance_metrics']['total_bets'] > 0
    )
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 16))
    
    # Define grid layout
    if has_performance_data:
        # Comprehensive dashboard with performance data
        gs = fig.add_gridspec(3, 3)
    else:
        # Simple dashboard with only value bets
        gs = fig.add_gridspec(1, 2)
    
    # Title and timestamp
    fig.suptitle(f"Valorant Betting Dashboard", fontsize=24, y=0.98)
    fig.text(0.5, 0.96, f"Generated on {dashboard_data['timestamp']}", fontsize=14, ha='center')
    
    if has_performance_data:
        # Performance metrics
        metrics = dashboard_data['performance_metrics']
        
        # 1. Overall metrics summary
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        ax1.set_title("Overall Performance", fontsize=18)
        
        summary_text = (
            f"Total Bets: {metrics['total_bets']}\n"
            f"Winning Bets: {metrics['winning_bets']}\n"
            f"Win Rate: {metrics['win_rate']:.2%}\n"
            f"Total Wagered: ${metrics['total_wagered']:.2f}\n"
            f"Total Profit: ${metrics['total_profit']:.2f}\n"
            f"Return on Investment: {metrics['roi']:.2%}"
        )
        
        ax1.text(0.5, 0.5, summary_text, fontsize=14, ha='center', va='center')
        
        # 2. Profit over time
        if 'profit_over_time' in metrics and metrics['profit_over_time']['timestamps']:
            ax2 = fig.add_subplot(gs[0, 1:])
            ax2.plot(metrics['profit_over_time']['cumulative_profit'], marker='o')
            ax2.set_title("Cumulative Profit Over Time", fontsize=18)
            ax2.set_ylabel("Profit ($)")
            ax2.set_xlabel("Bet Number")
            ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
        
        # 3. Performance by bet type
        if 'by_type' in metrics and metrics['by_type']:
            ax3 = fig.add_subplot(gs[1, 0])
            
            bet_types = list(metrics['by_type'].keys())
            bet_counts = [metrics['by_type'][bt]['bets'] for bt in bet_types]
            win_rates = [metrics['by_type'][bt]['win_rate'] for bt in bet_types]
            rois = [metrics['by_type'][bt]['roi'] for bt in bet_types]
            
            width = 0.35
            x = np.arange(len(bet_types))
            
            ax3.bar(x, win_rates, width, label='Win Rate')
            ax3.bar(x + width, rois, width, label='ROI')
            
            ax3.set_title("Performance by Bet Type", fontsize=18)
            ax3.set_xticks(x + width / 2)
            ax3.set_xticklabels(bet_types)
            ax3.legend()
            ax3.set_ylim(-0.5, 1.5)
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            
            # Add bet count annotations
            for i, count in enumerate(bet_counts):
                ax3.annotate(f"n={count}", xy=(i + width/2, 0.05), ha='center')
        
        # 4. Model calibration
        if 'calibration' in metrics and metrics['calibration']:
            ax4 = fig.add_subplot(gs[1, 1:])
            
            prob_ranges = list(metrics['calibration'].keys())
            pred_probs = [metrics['calibration'][pr]['predicted_probability'] for pr in prob_ranges]
            actual_rates = [metrics['calibration'][pr]['actual_win_rate'] for pr in prob_ranges]
            bet_counts = [metrics['calibration'][pr]['bets'] for pr in prob_ranges]
            
            # Size points by bet count
            sizes = [max(50, min(500, count * 10)) for count in bet_counts]
            
            # Perfect calibration line
            ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            # Actual calibration points
            ax4.scatter(pred_probs, actual_rates, s=sizes, alpha=0.7)
            
            # Annotate points with count
            for i, (x, y, count) in enumerate(zip(pred_probs, actual_rates, bet_counts)):
                ax4.annotate(f"n={count}", xy=(x, y), xytext=(5, 5), textcoords='offset points')
            
            ax4.set_title("Model Calibration: Predicted vs. Actual Win Rate", fontsize=18)
            ax4.set_xlabel("Predicted Probability")
            ax4.set_ylabel("Actual Win Rate")
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
        
        # 5. Performance by EV range
        if 'by_ev' in metrics and metrics['by_ev']:
            ax5 = fig.add_subplot(gs[2, 0:2])
            
            ev_ranges = list(metrics['by_ev'].keys())
            ev_win_rates = [metrics['by_ev'][ev]['win_rate'] for ev in ev_ranges]
            ev_rois = [metrics['by_ev'][ev]['roi'] for ev in ev_ranges]
            ev_bet_counts = [metrics['by_ev'][ev]['bets'] for ev in ev_ranges]
            
            x = np.arange(len(ev_ranges))
            width = 0.35
            
            bars1 = ax5.bar(x - width/2, ev_win_rates, width, label='Win Rate')
            bars2 = ax5.bar(x + width/2, ev_rois, width, label='ROI')
            
            ax5.set_title("Performance by Expected Value Range", fontsize=18)
            ax5.set_xticks(x)
            ax5.set_xticklabels(ev_ranges)
            ax5.legend()
            ax5.set_ylim(-0.5, 1.5)
            ax5.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            
            # Add bet count annotations
            for i, count in enumerate(ev_bet_counts):
                ax5.annotate(f"n={count}", xy=(i, -0.1), ha='center')
    
    # Value betting opportunities
    if 'value_bets' in dashboard_data and dashboard_data['value_bets']:
        value_bets = dashboard_data['value_bets']
        
        if has_performance_data:
            ax6 = fig.add_subplot(gs[2, 2])
        else:
            ax6 = fig.add_subplot(gs[0, 0:])
        
        ax6.axis('off')
        ax6.set_title("Top Value Betting Opportunities", fontsize=18)
        
        # Show top 5 value bets
        top_bets = sorted(value_bets, key=lambda x: x['expected_value'], reverse=True)[:5]
        
        table_data = []
        for i, bet in enumerate(top_bets):
            row = [
                f"{bet['team1']} vs {bet['team2']}",
                f"{bet['bet_type']} - {bet['selection']}",
                f"{bet['decimal_odds']:.2f}",
                f"{bet['expected_value']:.2%}",
                f"${bet['recommended_bet']:.2f}"
            ]
            table_data.append(row)
        
        if table_data:
            col_labels = ["Match", "Bet Type", "Odds", "EV", "Bet Size"]
            table = ax6.table(
                cellText=table_data,
                colLabels=col_labels,
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
        else:
            ax6.text(0.5, 0.5, "No value betting opportunities found", 
                    ha='center', va='center', fontsize=14)
    else:
        if has_performance_data:
            ax6 = fig.add_subplot(gs[2, 2])
        else:
            ax6 = fig.add_subplot(gs[0, 1])
            
        ax6.axis('off')
        ax6.set_title("Value Betting Opportunities", fontsize=18)
        ax6.text(0.5, 0.5, "No value betting opportunities found", 
                ha='center', va='center', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dashboard visualization saved to {save_path}")

def generate_betting_dashboard_report(dashboard_data, file_path=None):
    """Generate a detailed text report based on the betting dashboard data."""
    if not file_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f'betting_dashboard_report_{timestamp}.txt'
    
    with open(file_path, 'w') as f:
        # Title and timestamp
        f.write("=" * 80 + "\n")
        f.write(f"VALORANT BETTING DASHBOARD REPORT\n")
        f.write(f"Generated on {dashboard_data['timestamp']}\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall performance metrics
        has_performance_data = (
            'performance_metrics' in dashboard_data and 
            dashboard_data['performance_metrics'] and
            'total_bets' in dashboard_data['performance_metrics'] and
            dashboard_data['performance_metrics']['total_bets'] > 0
        )
        
        if has_performance_data:
            metrics = dashboard_data['performance_metrics']
            
            f.write("OVERALL PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Bets: {metrics['total_bets']}\n")
            f.write(f"Winning Bets: {metrics['winning_bets']}\n")
            f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
            f.write(f"Total Wagered: ${metrics['total_wagered']:.2f}\n")
            f.write(f"Total Profit: ${metrics['total_profit']:.2f}\n")
            f.write(f"Return on Investment: {metrics['roi']:.2%}\n\n")
            
            # Performance by bet type
            if 'by_type' in metrics and metrics['by_type']:
                f.write("PERFORMANCE BY BET TYPE\n")
                f.write("-" * 40 + "\n")
                
                for bet_type, stats in metrics['by_type'].items():
                    f.write(f"{bet_type}:\n")
                    f.write(f"  Bets: {stats['bets']}\n")
                    f.write(f"  Wins: {stats['wins']}\n")
                    f.write(f"  Win Rate: {stats['win_rate']:.2%}\n")
                    f.write(f"  Profit: ${stats['profit']:.2f}\n")
                    f.write(f"  ROI: {stats['roi']:.2%}\n\n")
            
            # Performance by EV range
            if 'by_ev' in metrics and metrics['by_ev']:
                f.write("PERFORMANCE BY EXPECTED VALUE RANGE\n")
                f.write("-" * 40 + "\n")
                
                for ev_range, stats in metrics['by_ev'].items():
                    f.write(f"EV {ev_range}:\n")
                    f.write(f"  Bets: {stats['bets']}\n")
                    f.write(f"  Wins: {stats['wins']}\n")
                    f.write(f"  Win Rate: {stats['win_rate']:.2%}\n")
                    f.write(f"  Profit: ${stats['profit']:.2f}\n")
                    f.write(f"  ROI: {stats['roi']:.2%}\n\n")
            
            # Model calibration
            if 'calibration' in metrics and metrics['calibration']:
                f.write("MODEL CALIBRATION - PREDICTED VS ACTUAL WIN RATE\n")
                f.write("-" * 40 + "\n")
                
                for prob_range, stats in metrics['calibration'].items():
                    f.write(f"Probability Range {prob_range}:\n")
                    f.write(f"  Bets: {stats['bets']}\n")
                    f.write(f"  Predicted Probability: {stats['predicted_probability']:.2%}\n")
                    f.write(f"  Actual Win Rate: {stats['actual_win_rate']:.2%}\n")
                    f.write(f"  Error: {(stats['actual_win_rate'] - stats['predicted_probability']):.2%}\n\n")
        else:
            f.write("No historical betting performance data available.\n\n")
        
        # Value betting opportunities
        f.write("VALUE BETTING OPPORTUNITIES\n")
        f.write("-" * 40 + "\n")
        
        if 'value_bets' in dashboard_data and dashboard_data['value_bets']:
            value_bets = dashboard_data['value_bets']
            
            # Sort by expected value
            value_bets = sorted(value_bets, key=lambda x: x['expected_value'], reverse=True)
            
            f.write(f"Found {len(value_bets)} value betting opportunities.\n")
            f.write(f"Current bankroll: ${dashboard_data['current_bankroll']:.2f}\n")
            f.write(f"Kelly fraction: {dashboard_data['settings']['kelly_fraction']:.2f}\n")
            f.write(f"Minimum EV threshold: {dashboard_data['settings']['min_ev']:.2%}\n\n")
            
            for i, bet in enumerate(value_bets):
                f.write(f"Bet #{i+1}: {bet['team1']} vs {bet['team2']}\n")
                f.write(f"  Type: {bet['bet_type']} - {bet['selection']}\n")
                f.write(f"  Odds: {bet['decimal_odds']:.2f} ({bet['american_odds']:+d})\n")
                f.write(f"  Model probability: {bet['model_prob']:.2%}\n")
                f.write(f"  Expected value: {bet['expected_value']:.2%}\n")
                f.write(f"  Kelly stake: {bet['kelly_stake']:.2%}\n")
                f.write(f"  Recommended bet: ${bet['recommended_bet']:.2f}\n")
                
                if bet.get('match_date'):
                    f.write(f"  Match date: {bet['match_date']}\n")
                    
                f.write("\n")
        else:
            f.write("No value betting opportunities found.\n\n")
        
        # Conclusion
        f.write("=" * 80 + "\n")
        f.write("BETTING STRATEGY RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        if has_performance_data:
            metrics = dashboard_data['performance_metrics']
            
            # Generate strategy recommendations based on performance
            if metrics['roi'] > 0.1:
                f.write("Your betting strategy is performing exceptionally well with a positive ROI.\n")
                f.write("Recommendations:\n")
                f.write("1. Continue with your current approach and Kelly stake sizing\n")
                f.write("2. Consider gradually increasing your bankroll allocation\n")
                f.write("3. Track model calibration closely to ensure predictions remain accurate\n")
            elif metrics['roi'] > 0:
                f.write("Your betting strategy is showing promising results with a positive ROI.\n")
                f.write("Recommendations:\n")
                f.write("1. Maintain current bankroll management discipline\n")
                f.write("2. Focus on bet types that show the strongest performance\n")
                f.write("3. Consider refining your model to improve prediction accuracy\n")
            elif metrics['roi'] > -0.1:
                f.write("Your betting strategy is close to break-even.\n")
                f.write("Recommendations:\n")
                f.write("1. Review bet types that are underperforming and consider adjustments\n")
                f.write("2. Be more selective with bets, possibly increasing the minimum EV threshold\n")
                f.write("3. Consider using a more conservative Kelly fraction (0.1-0.25)\n")
            else:
                f.write("Your betting strategy is currently underperforming.\n")
                f.write("Recommendations:\n")
                f.write("1. Temporarily reduce bet sizes while evaluating strategy\n")
                f.write("2. Focus only on highest EV opportunities (15%+ expected value)\n")
                f.write("3. Review model calibration and consider model retraining\n")
                f.write("4. Ensure odds are being accurately recorded\n")
        else:
            f.write("No historical betting data is available yet.\n")
            f.write("Recommendations:\n")
            f.write("1. Start with small bet sizes (25% of Kelly recommended amounts)\n")
            f.write("2. Focus on highest EV opportunities first\n")
            f.write("3. Record all bets and results carefully to build performance history\n")
            f.write("4. After accumulating 20+ bets, review performance and adjust strategy\n")
    
    print(f"Detailed betting dashboard report saved to {file_path}")
    return file_path

def calibrate_score_prediction_model(historical_matches_data, validate_accuracy=True):
    """Calibrate the score prediction model based on historical match data and validate its accuracy."""
    # This function would analyze the accuracy of our score predictions compared to actual results
    # and apply calibration adjustments if needed
    print("Calibrating score prediction model...")
    
    # Ensure we have historical data with scores
    if not historical_matches_data or len(historical_matches_data) < 10:
        print("Not enough historical match data for meaningful calibration.")
        return False
    
    # Extract actual vs. predicted data
    calibration_data = []
    
    for match in historical_matches_data:
        team1_name = match['team1']
        team2_name = match['team2']
        team1_score = match['team1_score']
        team2_score = match['team2_score']
        
        # Skip matches with invalid scores
        if team1_score + team2_score <= 0 or team1_score + team2_score > 3:
            continue
        
        # Get prediction for this match
        score_probs = predict_score_probabilities(team1_name, team2_name)
        
        if not score_probs:
            continue
        
        # Determine actual outcome
        actual_outcome = ""
        if team1_score > team2_score:
            if team1_score == 2 and team2_score == 0:
                actual_outcome = f"{team1_name} 2-0"
            elif team1_score == 2 and team2_score == 1:
                actual_outcome = f"{team1_name} 2-1"
        else:
            if team1_score == 0 and team2_score == 2:
                actual_outcome = f"{team2_name} 2-0"
            elif team1_score == 1 and team2_score == 2:
                actual_outcome = f"{team2_name} 2-1"
        
        # Skip if we couldn't determine the outcome
        if not actual_outcome:
            continue
        
        # Record this data point
        calibration_data.append({
            'match_id': match.get('match_id', ''),
            'team1': team1_name,
            'team2': team2_name,
            'actual_outcome': actual_outcome,
            'predicted_probs': {
                f"{team1_name} 2-0": score_probs.get(f"{team1_name} 2-0", 0),
                f"{team1_name} 2-1": score_probs.get(f"{team1_name} 2-1", 0),
                f"{team2_name} 2-1": score_probs.get(f"{team2_name} 2-1", 0),
                f"{team2_name} 2-0": score_probs.get(f"{team2_name} 2-0", 0)
            }
        })
    
    if len(calibration_data) < 10:
        print(f"Only found {len(calibration_data)} valid matches for calibration. Need at least 10.")
        return False
    
    print(f"Found {len(calibration_data)} matches for calibration analysis.")
    
    # Calculate calibration statistics
    outcomes = [f"{team1_name} 2-0", f"{team1_name} 2-1", f"{team2_name} 2-1", f"{team2_name} 2-0"]
    outcome_counts = {outcome: 0 for outcome in outcomes}
    outcome_predicted_probs = {outcome: [] for outcome in outcomes}
    
    for data_point in calibration_data:
        actual = data_point['actual_outcome']
        outcome_counts[actual] = outcome_counts.get(actual, 0) + 1
        
        # Record predicted probabilities for each outcome type
        for outcome in data_point['predicted_probs']:
            outcome_predicted_probs[outcome].append(data_point['predicted_probs'][outcome])
    
    # Calculate average predicted probability for each outcome
    avg_predicted_probs = {}
    for outcome in outcome_predicted_probs:
        if outcome_predicted_probs[outcome]:
            avg_predicted_probs[outcome] = sum(outcome_predicted_probs[outcome]) / len(outcome_predicted_probs[outcome])
        else:
            avg_predicted_probs[outcome] = 0
    
    # Calculate actual frequencies
    total_matches = len(calibration_data)
    actual_frequencies = {outcome: count / total_matches for outcome, count in outcome_counts.items()}
    
    # Calculate calibration errors
    calibration_errors = {
        outcome: avg_predicted_probs.get(outcome, 0) - actual_frequencies.get(outcome, 0)
        for outcome in outcomes
    }
    
    print("\nCalibration Analysis:")
    print("-" * 40)
    print(f"{'Outcome':<15} {'Predicted %':<15} {'Actual %':<15} {'Error':<15}")
    print("-" * 40)
    
    for outcome in outcomes:
        if outcome in avg_predicted_probs and outcome in actual_frequencies:
            print(f"{outcome:<15} {avg_predicted_probs[outcome]:.2%} {actual_frequencies[outcome]:.2%} {calibration_errors[outcome]:+.2%}")
    
    print("-" * 40)
    
    # Determine if calibration is needed
    needs_calibration = any(abs(error) > 0.05 for error in calibration_errors.values())
    
    if needs_calibration:
        print("\nCalibration needed. Saving calibration factors.")
        
        # Calculate calibration factors (simple approach)
        calibration_factors = {}
        
        for outcome in outcomes:
            if outcome in avg_predicted_probs and avg_predicted_probs[outcome] > 0:
                if outcome in actual_frequencies and actual_frequencies[outcome] > 0:
                    # Calculate adjustment factor
                    calibration_factors[outcome] = actual_frequencies[outcome] / avg_predicted_probs[outcome]
                else:
                    calibration_factors[outcome] = 1.0
    
    # Save calibration factors for future use
    with open('calibration_factors.json', 'w') as f:
        json.dump({
            'calibration_factors': calibration_factors,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sample_size': len(calibration_data)
        }, f, indent=2)
    
    print("Calibration factors saved.")
    return validate_accuracy if validate_accuracy else True

#-------------------------------------------------------------------------
# MAIN FUNCTION AND CLI INTERFACE UPDATES
#-------------------------------------------------------------------------

def main():
    """Main function to handle command line arguments and run the program."""
    parser = argparse.ArgumentParser(description="Valorant Match Predictor and Betting System")
    
    # Add command line arguments
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--optimize", action="store_true", help="Run model optimization")
    parser.add_argument("--predict", action="store_true", help="Predict a specific match")
    parser.add_argument("--team1", type=str, help="First team name")
    parser.add_argument("--team2", type=str, help="Second team name")
    parser.add_argument("--analyze", action="store_true", help="Analyze all upcoming matches")
    parser.add_argument("--players", action="store_true", help="Include player stats in analysis")
    parser.add_argument("--economy", action="store_true", help="Include economy data in analysis")
    parser.add_argument("--maps", action="store_true", help="Include map statistics")
    parser.add_argument("--team-limit", type=int, default=100, help="Limit number of teams to process")
    parser.add_argument("--cross-validate", action="store_true", help="Train with cross-validation")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for cross-validation")
    
    # New betting-related arguments
    parser.add_argument("--betting", action="store_true", help="Perform betting analysis")
    parser.add_argument("--manual-odds", action="store_true", help="Input odds manually")
    parser.add_argument("--bankroll", type=float, default=1000, help="Betting bankroll amount")
    parser.add_argument("--min-ev", type=float, default=0.05, help="Minimum expected value threshold")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly criterion fraction")
    parser.add_argument("--backtest", action="store_true", help="Perform model backtesting")
    parser.add_argument("--cutoff-date", type=str, help="Cutoff date for backtesting (YYYY/MM/DD)")
    parser.add_argument("--bet-amount", type=float, default=100, help="Fixed bet amount for backtesting")
    parser.add_argument("--confidence", type=float, default=0.05, help="Confidence threshold for backtesting")
    parser.add_argument("--find-value", action="store_true", help="Find value betting opportunities")
    parser.add_argument("--dashboard", action="store_true", help="Generate betting dashboard")
    parser.add_argument("--calibrate", action="store_true", help="Generate model calibration data")
    parser.add_argument("--train-score-model", action="store_true", help="Train a dedicated score prediction model")

    args = parser.parse_args()
    
    # Default to including all data types
    include_player_stats = args.players
    include_economy = args.economy
    include_maps = args.maps
    
    # If no data types specified, include all by default
    if not args.players and not args.economy and not args.maps:
        include_player_stats = True
        include_economy = True
        include_maps = False  # Maps off by default as it's most expensive
    
    if args.train:
        print("Training a new model...")
        
        # Collect team data
        team_data_collection = collect_team_data(
            team_limit=args.team_limit,
            include_player_stats=include_player_stats,
            include_economy=include_economy,
            include_maps=include_maps
        )
        
        if not team_data_collection:
            print("Failed to collect team data. Aborting training.")
            return
        
        # Build training dataset
        X, y = build_training_dataset(team_data_collection)
        
        print(f"Built training dataset with {len(X)} samples.")
        
        # Check if we have enough data to train
        if len(X) < 10:
            print("Not enough training data. Please collect more match data.")
            return
        
        # Train with appropriate method
        if args.cross_validate:
            print(f"Training with {args.folds}-fold cross-validation and ensemble modeling...")
            ensemble_models, stable_features, avg_metrics, fold_metrics, scaler = train_with_cross_validation(
                X, y, n_splits=args.folds, random_state=42
            )
            print("Ensemble model training complete.")
        else:
            # Train regular model
            print("Training standard model...")
            model, scaler, feature_names = train_model(X, y)
            print("Model training complete.")
            
        # Train score model if requested
        if args.train_score_model:
            print("Training score prediction model...")
            score_model, score_scaler, score_features = train_score_model(team_data_collection)
            print("Score prediction model training complete.")
    
    elif args.predict and args.team1 and args.team2:
        # Check if ensemble models exist
        ensemble_exists = any(os.path.exists(f'valorant_model_fold_{i+1}.h5') for i in range(5))
        
        if ensemble_exists:
            print(f"Predicting match between {args.team1} and {args.team2} using ensemble model...")
            prediction = predict_with_ensemble(args.team1, args.team2)
        else:
            print(f"Predicting match between {args.team1} and {args.team2} with standard model...")
            prediction = predict_match(args.team1, args.team2, include_maps=include_maps)
        
        if prediction:
            display_prediction_results(prediction)
            
            # If betting analysis requested
            if args.betting:
                if args.manual_odds:
                    print(f"Performing betting analysis with manual odds...")
                    
                    # Get manual odds input
                    try:
                        odds_data = {}
                        odds_data['ml_odds_team1'] = float(input(f"Moneyline odds for {args.team1} to win: "))
                        odds_data['ml_odds_team2'] = float(input(f"Moneyline odds for {args.team2} to win: "))
                        odds_data['team1_plus_1_5_odds'] = float(input(f"{args.team1} +1.5 maps odds: "))
                        odds_data['team2_plus_1_5_odds'] = float(input(f"{args.team2} +1.5 maps odds: "))
                        odds_data['team1_minus_1_5_odds'] = float(input(f"{args.team1} -1.5 maps odds: "))
                        odds_data['team2_minus_1_5_odds'] = float(input(f"{args.team2} -1.5 maps odds: "))
                        odds_data['over_2_5_odds'] = float(input("Over 2.5 maps odds: "))
                        odds_data['under_2_5_odds'] = float(input("Under 2.5 maps odds: "))
                        
                        # Check if score model exists
                        if os.path.exists('valorant_score_model.h5'):
                            analysis = analyze_match_betting_with_score_model(
                                args.team1,
                                args.team2,
                                odds_data,
                                bankroll=args.bankroll,
                                min_ev=args.min_ev,
                                kelly_fraction=args.kelly
                            )
                        else:
                            analysis = analyze_match_betting(
                                args.team1,
                                args.team2,
                                odds_data,
                                bankroll=args.bankroll,
                                min_ev=args.min_ev,
                                kelly_fraction=args.kelly
                            )
                        
                        if analysis:
                            display_betting_analysis(analysis)
                            # Create visualization
                            fig = visualize_betting_analysis(analysis, f"betting_analysis_{args.team1}_vs_{args.team2}.png")
                            plt.close(fig)
                        else:
                            print("Betting analysis failed.")
                            
                    except ValueError:
                        print("Invalid odds input. Please enter decimal odds (e.g. 1.85).")
                else:
                    print("Please use --manual-odds to input odds for betting analysis.")
        else:
            print(f"Could not generate prediction for {args.team1} vs {args.team2}")
            
    elif args.analyze:
        print("Analyzing upcoming matches...")
        analyze_upcoming_matches()
            
    elif args.backtest:
        if not args.cutoff_date:
            print("Please specify a cutoff date with --cutoff-date YYYY/MM/DD")
            return
        
        print(f"Performing backtesting with cutoff date: {args.cutoff_date}")
        print(f"Bet amount: ${args.bet_amount}, Confidence threshold: {args.confidence}")
        
        results = backtest_model(
            args.cutoff_date, 
            bet_amount=args.bet_amount, 
            confidence_threshold=args.confidence,
            kelly_fraction=args.kelly
        )
        
        if results:
            print("\nBacktesting Results:")
            print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
            print(f"High Confidence Accuracy: {results['high_conf_accuracy']:.4f}")
            print(f"ROI: {results['roi']:.4f}")
            print(f"Total profit: ${results['total_profit']:.2f}")
            print(f"Total bets: {results['total_bets']}")
            avg_profit = results['total_profit']/results['total_bets'] if results['total_bets'] > 0 else 0
            print(f"Average profit per bet: ${avg_profit:.2f}")
        else:
            print("Backtesting failed or returned no results.")
    
    elif args.find_value:
        print(f"Finding value betting opportunities...")
        print(f"Using bankroll: ${args.bankroll}, Minimum EV: {args.min_ev}%, Kelly fraction: {args.kelly}")
        
        value_report = find_value_bets(
            min_ev=args.min_ev,
            kelly_fraction=args.kelly,
            bankroll=args.bankroll
        )
        
        if value_report:
            print(f"\nFound {value_report['value_bets_found']} value betting opportunities.")
            print(f"Report saved to value_bets_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        else:
            print("Value bet analysis failed or returned no results.")
    
    elif args.dashboard:
        print(f"Generating betting dashboard...")
        print(f"Using bankroll: ${args.bankroll}, Minimum EV: {args.min_ev}%, Kelly fraction: {args.kelly}")
        
        dashboard_data = build_betting_dashboard(
            bankroll=args.bankroll,
            min_ev=args.min_ev,
            kelly_fraction=args.kelly
        )
        
        if dashboard_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = generate_betting_dashboard_report(dashboard_data)
            print(f"Dashboard generated and saved as betting_dashboard_{timestamp}.png")
            print(f"Detailed report saved to {report_path}")
        else:
            print("Dashboard generation failed.")
    
    elif args.calibrate:
        print("Generating model calibration data...")
        
        # Load bet history
        history_file = 'bet_history.csv'
        if not os.path.isfile(history_file):
            print(f"Error: No bet history found at {history_file}")
            return
        
        try:
            bet_history = pd.read_csv(history_file)
            if len(bet_history) < 20:
                print("Not enough betting history for calibration (minimum 20 bets required).")
                return
            
            calibration_data = generate_calibration_data(bet_history)
            if calibration_data:
                print("Calibration data generated successfully.")
        except Exception as e:
            print(f"Error generating calibration data: {e}")
    
    elif args.betting and args.team1 and args.team2 and args.manual_odds:
        print(f"Performing betting analysis for {args.team1} vs {args.team2}...")
        print(f"Using bankroll: ${args.bankroll}, Minimum EV: {args.min_ev}%, Kelly fraction: {args.kelly}")
        print("\nPlease enter the betting odds (decimal format, e.g. 1.85):")
        
        # Manual odds input
        odds_data = {}
        
        try:
            odds_data['ml_odds_team1'] = float(input(f"Moneyline odds for {args.team1} to win: "))
            odds_data['ml_odds_team2'] = float(input(f"Moneyline odds for {args.team2} to win: "))
            
            odds_data['team1_plus_1_5_odds'] = float(input(f"{args.team1} +1.5 maps odds: "))
            odds_data['team2_plus_1_5_odds'] = float(input(f"{args.team2} +1.5 maps odds: "))
            
            odds_data['team1_minus_1_5_odds'] = float(input(f"{args.team1} -1.5 maps odds: "))
            odds_data['team2_minus_1_5_odds'] = float(input(f"{args.team2} -1.5 maps odds: "))
            
            odds_data['over_2_5_odds'] = float(input("Over 2.5 maps odds: "))
            odds_data['under_2_5_odds'] = float(input("Under 2.5 maps odds: "))
            
            # Check if score model exists
            if os.path.exists('valorant_score_model.h5'):
                analysis = analyze_match_betting_with_score_model(
                    args.team1,
                    args.team2,
                    odds_data,
                    bankroll=args.bankroll,
                    min_ev=args.min_ev,
                    kelly_fraction=args.kelly
                )
            else:
                analysis = analyze_match_betting(
                    args.team1,
                    args.team2,
                    odds_data,
                    bankroll=args.bankroll,
                    min_ev=args.min_ev,
                    kelly_fraction=args.kelly
                )
            
            if analysis:
                display_betting_analysis(analysis)
                # Create visualization
                fig = visualize_betting_analysis(analysis, f"betting_analysis_{args.team1}_vs_{args.team2}.png")
                plt.close(fig)
            else:
                print("Analysis failed to produce results.")
        except ValueError as e:
            print(f"Error: Invalid odds input. Please enter valid decimal odds (e.g., 1.85).")
            print(f"Exception: {e}")
            return
    
    else:
        print("Please specify an action: --train, --predict, --analyze, --backtest, --betting, --find-value, or --dashboard")
        print("For predictions and betting analysis, specify --team1 and --team2")
        print("For betting analysis, use --betting --manual-odds, and consider using --bankroll, --min-ev, and --kelly")
        print("For backtesting, specify --cutoff-date YYYY/MM/DD")


if __name__ == "__main__":
    main()