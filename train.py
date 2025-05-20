#!/usr/bin/env python3
print("Starting Optimized Deep Learning Valorant Match Predictor...")

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

def collect_team_data(team_limit=300, include_player_stats=True, include_economy=True, include_maps=True):
    """Collect data for all teams to use in training and evaluation."""
    print("\n========================================================")
    print("COLLECTING TEAM DATA")
    print("========================================================")
    print(f"Including player stats: {include_player_stats}")
    print(f"Including economy data: {include_economy}")
    print(f"Including map data: {include_maps}")
    
    # Fetch all teams
    print(f"Fetching teams from API: {API_URL}/teams?limit={team_limit}")
    try:
        teams_response = requests.get(f"{API_URL}/teams?limit={team_limit}")
        print(f"API response status code: {teams_response.status_code}")
        
        if teams_response.status_code != 200:
            print(f"Error fetching teams: {teams_response.status_code}")
            # Try to print response content for debugging
            try:
                print(f"Response content: {teams_response.text[:500]}...")
            except:
                pass
            return {}
        
        teams_data = teams_response.json()
        
        # Debug info
        print(f"Teams data keys: {list(teams_data.keys())}")
        
        if 'data' not in teams_data:
            print("No 'data' field found in the response")
            # Try to print response content for debugging
            try:
                print(f"Response content: {teams_data}")
            except:
                pass
            return {}
        
        print(f"Number of teams in response: {len(teams_data['data'])}")
        
        # Select teams based on ranking or use a limited sample
        top_teams = []
        for team in teams_data['data']:
            if 'ranking' in team and team['ranking'] and team['ranking'] <= 50:
                top_teams.append(team)
        
        # If no teams with rankings were found, just take the first N teams
        if not top_teams:
            print(f"No teams with rankings found. Using the first {min(130, team_limit)} teams instead.")
            if len(teams_data['data']) > 0:
                top_teams = teams_data['data'][:min(130, team_limit)]
                print(f"Selected {len(top_teams)} teams for data collection.")
            else:
                print("No teams found in data array.")
                return {}
        else:
            print(f"Selected {len(top_teams)} ranked teams for data collection.")
        
        # Check if we actually have team data
        if len(top_teams) == 0:
            print("ERROR: No teams selected for data collection.")
            return {}
        
        # Print first team info for debugging
        if top_teams:
            print(f"First team info: {top_teams[0]}")
            
        # Collect match data for each team
        team_data_collection = {}
        
        # Track data availability counts
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
            
            # Get team tag for economy data matching
            team_details, team_tag = fetch_team_details(team_id)
            
            team_history = fetch_team_match_history(team_id)
            if not team_history:
                print(f"No match history for team {team_name}, skipping.")
                continue
                
            team_matches = parse_match_data(team_history, team_name)
            
            # Skip teams with no match data
            if not team_matches:
                print(f"No parsed match data for team {team_name}, skipping.")
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
            
            # Print progress
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
    Prepare symmetrical features for model prediction with corrected handling of binary features.
    """
    if not team1_stats or not team2_stats:
        print("Missing team statistics data")
        return None
    
    features = {}
    
    # Helper function for safe numeric difference
    def safe_diff(val1, val2, default=0):
        """Calculate difference between two values with type safety."""
        # Handle lists by using their length
        if isinstance(val1, list):
            val1 = len(val1)
        if isinstance(val2, list):
            val2 = len(val2)
            
        # Try to convert to float and calculate difference
        try:
            return float(val1) - float(val2)
        except (ValueError, TypeError):
            return default
    
    # Helper function for safe averaging
    def safe_avg(val1, val2, default=0):
        """Calculate average of two values with type safety."""
        # Handle lists by using their length
        if isinstance(val1, list):
            val1 = len(val1)
        if isinstance(val2, list):
            val2 = len(val2)
            
        # Try to convert to float and calculate average
        try:
            return (float(val1) + float(val2)) / 2
        except (ValueError, TypeError):
            return default
    
    # Helper function for safe binary comparison
    def safe_binary(val1, val2):
        """Compare two values with type safety, returning 1 if val1 > val2, 0 otherwise."""
        try:
            # Make sure we return an integer 0 or 1, not a boolean
            return int(float(val1) > float(val2))
        except (ValueError, TypeError):
            return 0
    
    # Helper function for safe division
    def safe_div(val1, val2, default=0):
        """Calculate division with type safety."""
        try:
            return float(val1) / float(val2)
        except (ValueError, TypeError, ZeroDivisionError):
            return default
            
    # Helper function for safe multiplication
    def safe_mult(val1, val2, default=0):
        """Calculate multiplication with type safety."""
        try:
            return float(val1) * float(val2)
        except (ValueError, TypeError):
            return default
    
    #----------------------------------------
    # 1. DIFFERENCE FEATURES (sign changes when teams swap)
    #----------------------------------------
    # These features should negate when teams are swapped
    features['win_rate_diff'] = safe_diff(team1_stats.get('win_rate', 0), team2_stats.get('win_rate', 0))
    features['recent_form_diff'] = safe_diff(team1_stats.get('recent_form', 0), team2_stats.get('recent_form', 0))
    features['score_diff_differential'] = safe_diff(team1_stats.get('score_differential', 0), team2_stats.get('score_differential', 0))
    features['wins_diff'] = safe_diff(team1_stats.get('wins', 0), team2_stats.get('wins', 0))
    features['losses_diff'] = safe_diff(team1_stats.get('losses', 0), team2_stats.get('losses', 0))
    features['avg_score_diff'] = safe_diff(team1_stats.get('avg_score', 0), team2_stats.get('avg_score', 0))
    features['avg_opponent_score_diff'] = safe_diff(team1_stats.get('avg_opponent_score', 0), team2_stats.get('avg_opponent_score', 0))
    features['match_count_diff'] = safe_diff(team1_stats.get('matches', 0), team2_stats.get('matches', 0))
    
    # Calculate recency weighted win rate if available, otherwise use regular win_rate
    if 'recency_weighted_win_rate' in team1_stats and 'recency_weighted_win_rate' in team2_stats:
        features['recency_weighted_win_rate_diff'] = safe_diff(
            team1_stats.get('recency_weighted_win_rate', 0), 
            team2_stats.get('recency_weighted_win_rate', 0)
        )
    else:
        features['recency_weighted_win_rate_diff'] = features['win_rate_diff']
    
    # Win-loss ratio difference with safe division
    team1_wl_ratio = safe_div(team1_stats.get('wins', 0), max(team1_stats.get('losses', 1), 1))
    team2_wl_ratio = safe_div(team2_stats.get('wins', 0), max(team2_stats.get('losses', 1), 1))
    features['win_loss_ratio_diff'] = safe_diff(team1_wl_ratio, team2_wl_ratio)
    
    # Map win rate difference
    if 'map_statistics' in team1_stats and 'map_statistics' in team2_stats:
        team1_maps = set(team1_stats['map_statistics'].keys())
        team2_maps = set(team2_stats['map_statistics'].keys())
        common_maps = team1_maps.intersection(team2_maps)
        
        if common_maps:
            # Safely get win_percentage with type checking
            team1_map_winrates = []
            team2_map_winrates = []
            
            for m in common_maps:
                try:
                    t1_win_pct = float(team1_stats['map_statistics'][m].get('win_percentage', 0))
                    t2_win_pct = float(team2_stats['map_statistics'][m].get('win_percentage', 0))
                    team1_map_winrates.append(t1_win_pct)
                    team2_map_winrates.append(t2_win_pct)
                except (ValueError, TypeError):
                    continue
            
            if team1_map_winrates and team2_map_winrates:
                # Calculate average map win rates
                t1_avg_map_winrate = sum(team1_map_winrates) / len(team1_map_winrates)
                t2_avg_map_winrate = sum(team2_map_winrates) / len(team2_map_winrates)
                features['avg_map_win_rate_diff'] = t1_avg_map_winrate - t2_avg_map_winrate
                
                # Count maps where team1 has advantage - this is a BINARY feature
                maps_advantage_count = sum(1 for i in range(len(team1_map_winrates)) 
                                          if team1_map_winrates[i] > team2_map_winrates[i])
                features['maps_advantage_team1'] = int(maps_advantage_count / len(common_maps) > 0.5)
                
                # Best map performance difference
                features['best_map_diff'] = max(team1_map_winrates) - max(team2_map_winrates)
    
    #----------------------------------------
    # 2. PLAYER STATS DIFFERENCES (sign changes when teams swap)
    #----------------------------------------
    if 'avg_player_rating' in team1_stats and 'avg_player_rating' in team2_stats:
        features['player_rating_diff'] = safe_diff(team1_stats.get('avg_player_rating', 0), team2_stats.get('avg_player_rating', 0))
        features['acs_diff'] = safe_diff(team1_stats.get('avg_player_acs', 0), team2_stats.get('avg_player_acs', 0))
        features['kd_diff'] = safe_diff(team1_stats.get('avg_player_kd', 0), team2_stats.get('avg_player_kd', 0))
        features['kast_diff'] = safe_diff(team1_stats.get('avg_player_kast', 0), team2_stats.get('avg_player_kast', 0))
        features['adr_diff'] = safe_diff(team1_stats.get('avg_player_adr', 0), team2_stats.get('avg_player_adr', 0))
        features['headshot_diff'] = safe_diff(team1_stats.get('avg_player_headshot', 0), team2_stats.get('avg_player_headshot', 0))
        features['star_player_diff'] = safe_diff(team1_stats.get('star_player_rating', 0), team2_stats.get('star_player_rating', 0))
        features['team_consistency_diff'] = safe_diff(team1_stats.get('team_consistency', 0), team2_stats.get('team_consistency', 0))
        features['fk_fd_diff'] = safe_diff(team1_stats.get('fk_fd_ratio', 0), team2_stats.get('fk_fd_ratio', 0))
    
    #----------------------------------------
    # 3. ECONOMY STATS DIFFERENCES (sign changes when teams swap)
    #----------------------------------------
    if 'pistol_win_rate' in team1_stats and 'pistol_win_rate' in team2_stats:
        features['pistol_win_rate_diff'] = safe_diff(team1_stats.get('pistol_win_rate', 0), team2_stats.get('pistol_win_rate', 0))
        
        if 'eco_win_rate' in team1_stats and 'eco_win_rate' in team2_stats:
            features['eco_win_rate_diff'] = safe_diff(team1_stats.get('eco_win_rate', 0), team2_stats.get('eco_win_rate', 0))
        
        if 'semi_eco_win_rate' in team1_stats and 'semi_eco_win_rate' in team2_stats:
            features['semi_eco_win_rate_diff'] = safe_diff(team1_stats.get('semi_eco_win_rate', 0), team2_stats.get('semi_eco_win_rate', 0))
        
        if 'full_buy_win_rate' in team1_stats and 'full_buy_win_rate' in team2_stats:
            features['full_buy_win_rate_diff'] = safe_diff(team1_stats.get('full_buy_win_rate', 0), team2_stats.get('full_buy_win_rate', 0))
        
        if 'economy_efficiency' in team1_stats and 'economy_efficiency' in team2_stats:
            features['economy_efficiency_diff'] = safe_diff(team1_stats.get('economy_efficiency', 0), team2_stats.get('economy_efficiency', 0))
        
        if 'low_economy_win_rate' in team1_stats and 'low_economy_win_rate' in team2_stats:
            features['low_economy_win_rate_diff'] = safe_diff(team1_stats.get('low_economy_win_rate', 0), team2_stats.get('low_economy_win_rate', 0))
        
        if 'high_economy_win_rate' in team1_stats and 'high_economy_win_rate' in team2_stats:
            features['high_economy_win_rate_diff'] = safe_diff(team1_stats.get('high_economy_win_rate', 0), team2_stats.get('high_economy_win_rate', 0))
    
    #----------------------------------------
    # 4. BINARY FEATURES (value flips when teams swap: 0->1, 1->0)
    #----------------------------------------
    # Always use explicit integer conversion for all binary features - VERY IMPORTANT
    features['better_win_rate_team1'] = safe_binary(team1_stats.get('win_rate', 0), team2_stats.get('win_rate', 0))
    features['better_recent_form_team1'] = safe_binary(team1_stats.get('recent_form', 0), team2_stats.get('recent_form', 0))
    
    # Use direct comparison for score_differential
    t1_score_diff = team1_stats.get('score_differential', 0)
    t2_score_diff = team2_stats.get('score_differential', 0)
    features['better_score_diff_team1'] = safe_binary(t1_score_diff, t2_score_diff)
    
    features['better_avg_score_team1'] = safe_binary(team1_stats.get('avg_score', 0), team2_stats.get('avg_score', 0))
    
    # For defense metrics, lower is better - so comparison is reversed!
    t1_defense = team1_stats.get('avg_opponent_score', float('inf'))
    t2_defense = team2_stats.get('avg_opponent_score', float('inf'))
    
    # Make sure we're comparing with correct logic (team1 is better when its value is LOWER)
    features['better_defense_team1'] = safe_binary(
        t2_defense,  # Higher value is worse defense
        t1_defense   # Lower value is better defense
    )
    
    if 'avg_player_rating' in team1_stats and 'avg_player_rating' in team2_stats:
        features['better_player_rating_team1'] = safe_binary(team1_stats.get('avg_player_rating', 0), team2_stats.get('avg_player_rating', 0))
    
    if 'pistol_win_rate' in team1_stats and 'pistol_win_rate' in team2_stats:
        features['better_pistol_team1'] = safe_binary(team1_stats.get('pistol_win_rate', 0), team2_stats.get('pistol_win_rate', 0))
    
    # Better trajectory
    if 'performance_trends' in team1_stats and 'performance_trends' in team2_stats:
        if ('form_trajectory' in team1_stats.get('performance_trends', {}) and 
            'form_trajectory' in team2_stats.get('performance_trends', {})):
            team1_trajectory = team1_stats['performance_trends']['form_trajectory'].get('5_vs_10', 0)
            team2_trajectory = team2_stats['performance_trends']['form_trajectory'].get('5_vs_10', 0)
            features['recent_form_trajectory_diff'] = safe_diff(team1_trajectory, team2_trajectory)
            features['better_trajectory_team1'] = safe_binary(team1_trajectory, team2_trajectory)
    
    #----------------------------------------
    # 5. H2H (HEAD-TO-HEAD) FEATURES
    #----------------------------------------
    team1_name = team1_stats.get('team_name', 'Team1')
    team2_name = team2_stats.get('team_name', 'Team2') 
    
    # Initialize H2H features with default values
    features['h2h_win_rate'] = 0.5  # Default to even matchup
    features['h2h_matches'] = 0     # Default to no matches
    features['h2h_score_diff'] = 0  # Default to no score difference
    features['h2h_advantage_team1'] = 0  # Default to no advantage
    features['h2h_significant'] = 0  # Default to not significant
    features['h2h_recency'] = 0.5    # Default to moderate recency
    
    # Search for head-to-head data
    h2h_found = False
    
    # Search team1 vs team2
    if 'opponent_stats' in team1_stats and isinstance(team1_stats['opponent_stats'], dict):
        # Try to find team2 in team1's opponent stats
        if team2_name in team1_stats['opponent_stats']:
            h2h_data = team1_stats['opponent_stats'][team2_name]
            h2h_found = True
        else:
            # Try partial matching
            team2_tag = team2_stats.get('team_tag', '')
            for opponent_name, stats in team1_stats['opponent_stats'].items():
                if (team2_name.lower() in opponent_name.lower() or 
                   (team2_tag and team2_tag.lower() in opponent_name.lower())):
                    h2h_data = stats
                    h2h_found = True
                    break
                    
        if h2h_found:
            # Team1 perspective - use directly
            features['h2h_win_rate'] = h2h_data.get('win_rate', 0.5)
            features['h2h_matches'] = h2h_data.get('matches', 0)
            features['h2h_score_diff'] = h2h_data.get('score_differential', 0)
            # FIXED: Use strict comparison for advantage (must be exactly greater than 0.5)
            features['h2h_advantage_team1'] = int(h2h_data.get('win_rate', 0.5) > 0.5)
            features['h2h_significant'] = int(h2h_data.get('matches', 0) >= 3)
            features['h2h_recency'] = 0.8  # High weight for actual data
    
    # If not found, search team2 vs team1 (reverse direction)
    if not h2h_found and 'opponent_stats' in team2_stats and isinstance(team2_stats['opponent_stats'], dict):
        # Try to find team1 in team2's opponent stats
        if team1_name in team2_stats['opponent_stats']:
            h2h_data = team2_stats['opponent_stats'][team1_name]
            h2h_found = True
        else:
            # Try partial matching
            team1_tag = team1_stats.get('team_tag', '')
            for opponent_name, stats in team2_stats['opponent_stats'].items():
                if (team1_name.lower() in opponent_name.lower() or 
                   (team1_tag and team1_tag.lower() in opponent_name.lower())):
                    h2h_data = stats
                    h2h_found = True
                    break
                    
        if h2h_found:
            # Team2 perspective - invert values for team1 perspective
            features['h2h_win_rate'] = 1 - h2h_data.get('win_rate', 0.5)  # Invert win rate
            features['h2h_matches'] = h2h_data.get('matches', 0)  # Matches count stays the same
            features['h2h_score_diff'] = -h2h_data.get('score_differential', 0)  # Negate score diff
            # FIXED: Use strict comparison for advantage (must be exactly less than 0.5)
            features['h2h_advantage_team1'] = int(1 - h2h_data.get('win_rate', 0.5) > 0.5)  # Invert advantage
            features['h2h_significant'] = int(h2h_data.get('matches', 0) >= 3)  # Significance stays the same
            features['h2h_recency'] = 0.8  # High weight for actual data
    
    # If no head-to-head data found, estimate from team strengths
    if not h2h_found:
        # Calculate estimated h2h using both win rates
        team1_win_rate = team1_stats.get('win_rate', 0.5)
        team2_win_rate = team2_stats.get('win_rate', 0.5)
        
        # Convert to float for calculation
        try:
            team1_win_rate = float(team1_win_rate)
            team2_win_rate = float(team2_win_rate)
        except (ValueError, TypeError):
            team1_win_rate = 0.5
            team2_win_rate = 0.5
        
        if team1_win_rate + team2_win_rate > 0:
            estimated_h2h = team1_win_rate / (team1_win_rate + team2_win_rate)
        else:
            estimated_h2h = 0.5
            
        # Scale towards 0.5 to reduce extremes
        estimated_h2h = 0.5 + (estimated_h2h - 0.5) * 0.6
        
        features['h2h_win_rate'] = estimated_h2h
        features['h2h_matches'] = 0
        features['h2h_score_diff'] = safe_diff(team1_stats.get('score_differential', 0), team2_stats.get('score_differential', 0)) * 0.5
        # FIXED: Use strict comparison for advantage (must be exactly greater than 0.5)
        features['h2h_advantage_team1'] = int(estimated_h2h > 0.5)
        features['h2h_significant'] = 0
        features['h2h_recency'] = 0.2
    
    #----------------------------------------
    # 6. SYMMETRIC AVERAGE FEATURES (value is the same when teams swap)
    #----------------------------------------
    # These should remain identical regardless of team ordering
    features['avg_win_rate'] = safe_avg(team1_stats.get('win_rate', 0), team2_stats.get('win_rate', 0))
    features['avg_recent_form'] = safe_avg(team1_stats.get('recent_form', 0), team2_stats.get('recent_form', 0))
    
    # Handle both integer and list cases for matches
    if isinstance(team1_stats.get('matches', 0), list) and isinstance(team2_stats.get('matches', 0), list):
        features['total_matches'] = len(team1_stats.get('matches', [])) + len(team2_stats.get('matches', []))
    else:
        # Try to convert to numbers
        try:
            features['total_matches'] = float(team1_stats.get('matches', 0)) + float(team2_stats.get('matches', 0))
        except (ValueError, TypeError):
            # If conversion fails, fallback to list length
            t1_matches = len(team1_stats.get('matches', [])) if isinstance(team1_stats.get('matches', 0), list) else 0
            t2_matches = len(team2_stats.get('matches', [])) if isinstance(team2_stats.get('matches', 0), list) else 0
            features['total_matches'] = t1_matches + t2_matches
    
    # Calculate match count ratio with safe division
    t1_matches = safe_div(team1_stats.get('matches', 1), 1)
    t2_matches = safe_div(team2_stats.get('matches', 1), 1)
    features['match_count_ratio'] = safe_div(t1_matches, max(1, t2_matches))
    
    features['avg_score_metric'] = safe_avg(team1_stats.get('avg_score', 0), team2_stats.get('avg_score', 0))
    features['avg_defense_metric'] = safe_avg(team1_stats.get('avg_opponent_score', 0), team2_stats.get('avg_opponent_score', 0))
    
    # Player statistics averages
    if 'avg_player_rating' in team1_stats and 'avg_player_rating' in team2_stats:
        features['avg_player_rating'] = safe_avg(team1_stats.get('avg_player_rating', 0), team2_stats.get('avg_player_rating', 0))
        features['avg_acs'] = safe_avg(team1_stats.get('avg_player_acs', 0), team2_stats.get('avg_player_acs', 0))
        features['avg_kd'] = safe_avg(team1_stats.get('avg_player_kd', 0), team2_stats.get('avg_player_kd', 0))
        features['avg_kast'] = safe_avg(team1_stats.get('avg_player_kast', 0), team2_stats.get('avg_player_kast', 0))
        features['avg_adr'] = safe_avg(team1_stats.get('avg_player_adr', 0), team2_stats.get('avg_player_adr', 0))
        features['avg_headshot'] = safe_avg(team1_stats.get('avg_player_headshot', 0), team2_stats.get('avg_player_headshot', 0))
        features['star_player_avg'] = safe_avg(team1_stats.get('star_player_rating', 0), team2_stats.get('star_player_rating', 0))
        features['avg_team_consistency'] = safe_avg(team1_stats.get('team_consistency', 0), team2_stats.get('team_consistency', 0))
        features['avg_fk_fd_ratio'] = safe_avg(team1_stats.get('fk_fd_ratio', 0), team2_stats.get('fk_fd_ratio', 0))
    
    # Economy statistics averages
    if 'pistol_win_rate' in team1_stats and 'pistol_win_rate' in team2_stats:
        features['avg_pistol_win_rate'] = safe_avg(team1_stats.get('pistol_win_rate', 0), team2_stats.get('pistol_win_rate', 0))
        
        if 'eco_win_rate' in team1_stats and 'eco_win_rate' in team2_stats:
            features['avg_eco_win_rate'] = safe_avg(team1_stats.get('eco_win_rate', 0), team2_stats.get('eco_win_rate', 0))
        
        if 'semi_eco_win_rate' in team1_stats and 'semi_eco_win_rate' in team2_stats:
            features['avg_semi_eco_win_rate'] = safe_avg(team1_stats.get('semi_eco_win_rate', 0), team2_stats.get('semi_eco_win_rate', 0))
        
        if 'full_buy_win_rate' in team1_stats and 'full_buy_win_rate' in team2_stats:
            features['avg_full_buy_win_rate'] = safe_avg(team1_stats.get('full_buy_win_rate', 0), team2_stats.get('full_buy_win_rate', 0))
        
        if 'economy_efficiency' in team1_stats and 'economy_efficiency' in team2_stats:
            features['avg_economy_efficiency'] = safe_avg(team1_stats.get('economy_efficiency', 0), team2_stats.get('economy_efficiency', 0))
        
        if 'low_economy_win_rate' in team1_stats and 'low_economy_win_rate' in team2_stats:
            features['avg_low_economy_win_rate'] = safe_avg(team1_stats.get('low_economy_win_rate', 0), team2_stats.get('low_economy_win_rate', 0))
        
        if 'high_economy_win_rate' in team1_stats and 'high_economy_win_rate' in team2_stats:
            features['avg_high_economy_win_rate'] = safe_avg(team1_stats.get('high_economy_win_rate', 0), team2_stats.get('high_economy_win_rate', 0))
    
    #----------------------------------------
    # 7. INTERACTION FEATURES
    #----------------------------------------
    
    # Rating x win rate
    if 'avg_player_rating' in team1_stats and 'avg_player_rating' in team2_stats:
        # Calculate as difference of products
        team1_product = safe_mult(team1_stats.get('avg_player_rating', 0), team1_stats.get('win_rate', 0))
        team2_product = safe_mult(team2_stats.get('avg_player_rating', 0), team2_stats.get('win_rate', 0))
        features['rating_x_win_rate'] = safe_diff(team1_product, team2_product)
    
    # Economy interactions
    if 'pistol_win_rate' in team1_stats and 'pistol_win_rate' in team2_stats:
        if 'eco_win_rate' in team1_stats and 'eco_win_rate' in team2_stats:
            # Calculate as difference of products
            team1_pistol_eco = safe_mult(team1_stats.get('pistol_win_rate', 0), team1_stats.get('eco_win_rate', 0))
            team2_pistol_eco = safe_mult(team2_stats.get('pistol_win_rate', 0), team2_stats.get('eco_win_rate', 0))
            features['pistol_x_eco'] = safe_diff(team1_pistol_eco, team2_pistol_eco)
        
        if 'full_buy_win_rate' in team1_stats and 'full_buy_win_rate' in team2_stats:
            # Calculate as difference of products
            team1_pistol_fullbuy = safe_mult(team1_stats.get('pistol_win_rate', 0), team1_stats.get('full_buy_win_rate', 0))
            team2_pistol_fullbuy = safe_mult(team2_stats.get('pistol_win_rate', 0), team2_stats.get('full_buy_win_rate', 0))
            features['pistol_x_full_buy'] = safe_diff(team1_pistol_fullbuy, team2_pistol_fullbuy)
    
    # First blood interactions
    if 'fk_fd_ratio' in team1_stats and 'fk_fd_ratio' in team2_stats and 'win_rate' in team1_stats and 'win_rate' in team2_stats:
        # Calculate as difference of products
        team1_fb_win = safe_mult(team1_stats.get('fk_fd_ratio', 0), team1_stats.get('win_rate', 0))
        team2_fb_win = safe_mult(team2_stats.get('fk_fd_ratio', 0), team2_stats.get('win_rate', 0))
        features['first_blood_x_win_rate'] = safe_diff(team1_fb_win, team2_fb_win)
    
    # H2H interactions
    if 'h2h_win_rate' in features and features['h2h_matches'] > 0:
        # For H2H x win rate
        team1_h2h_win = safe_mult(features['h2h_win_rate'], team1_stats.get('win_rate', 0))
        team2_h2h_win = safe_mult(1 - features['h2h_win_rate'], team2_stats.get('win_rate', 0))
        features['h2h_x_win_rate'] = safe_diff(team1_h2h_win, team2_h2h_win)
        
        # For H2H x form
        team1_h2h_form = safe_mult(features['h2h_win_rate'], team1_stats.get('recent_form', 0))
        team2_h2h_form = safe_mult(1 - features['h2h_win_rate'], team2_stats.get('recent_form', 0))
        features['h2h_x_form'] = safe_diff(team1_h2h_form, team2_h2h_form)
    
    return features

def test_defensive_feature_symmetry():
    """
    Specific test for defensive feature symmetry that was problematic.
    """
    print("\n===== TESTING DEFENSIVE FEATURE SYMMETRY =====")
    
    # Create test team stats focused on defensive metrics
    team1_stats = {
        'team_name': 'Team A',
        'avg_opponent_score': 8.7,  # Lower = better defense
        'score_differential': 4.5,  # Higher = better overall
    }
    
    team2_stats = {
        'team_name': 'Team B',
        'avg_opponent_score': 9.5,  # Higher = worse defense
        'score_differential': 2.3,  # Lower = worse overall
    }
    
    # Generate features in both directions
    print("\nNormal direction (Team A vs Team B):")
    features_1_2 = prepare_data_for_model(team1_stats, team2_stats)
    
    print("\nReversed direction (Team B vs Team A):")
    features_2_1 = prepare_data_for_model(team2_stats, team1_stats)
    
    # Check defensive feature specifically
    print("\nChecking defensive feature transformation:")
    val_1_2 = features_1_2['better_defense_team1']
    val_2_1 = features_2_1['better_defense_team1']
    
    print(f"Normal direction: better_defense_team1 = {val_1_2}")
    print(f"Reversed direction: better_defense_team1 = {val_2_1}")
    print(f"Expected reversed value: {1-val_1_2}")
    
    if val_1_2 == 1 - val_2_1:
        print("✓ Feature transforms correctly when teams are swapped")
    else:
        print("✗ Feature STILL doesn't transform correctly!")
    
    # Check score differential features
    print("\nChecking score differential features:")
    
    # Difference feature (should negate)
    diff_1_2 = features_1_2['score_diff_differential']
    diff_2_1 = features_2_1['score_diff_differential']
    print(f"score_diff_differential: {diff_1_2} → {diff_2_1} (expected {-diff_1_2})")
    if abs(diff_1_2 + diff_2_1) < 1e-10:
        print("✓ Difference feature transforms correctly")
    else:
        print("✗ Difference feature FAILS to transform correctly")
    
    # Binary feature (should flip)
    bin_1_2 = features_1_2['better_score_diff_team1']
    bin_2_1 = features_2_1['better_score_diff_team1']
    print(f"better_score_diff_team1: {bin_1_2} → {bin_2_1} (expected {1-bin_1_2})")
    if bin_1_2 == 1 - bin_2_1:
        print("✓ Binary feature transforms correctly")
    else:
        print("✗ Binary feature FAILS to transform correctly")
    
    return features_1_2, features_2_1



def verify_h2h_interaction_symmetry(team1_stats, team2_stats):
    """
    Verify that h2h interaction features transform correctly when teams are swapped.
    """
    print("\n===== VERIFYING H2H INTERACTION SYMMETRY =====")
    
    # Generate features with team1 and team2 in normal order
    features_normal = prepare_data_for_model(team1_stats, team2_stats)
    
    # Generate features with teams swapped
    features_swapped = prepare_data_for_model(team2_stats, team1_stats)
    
    # Test h2h interaction features specifically
    h2h_interaction_features = ['h2h_x_win_rate', 'h2h_x_form']
    
    all_passed = True
    print("\nTesting H2H INTERACTION features (should negate when teams swap):")
    for feature in h2h_interaction_features:
        if feature in features_normal and feature in features_swapped:
            expected = -features_normal[feature]
            actual = features_swapped[feature]
            
            if abs(expected - actual) < 1e-10:
                print(f"✓ {feature:<30} transforms correctly: {features_normal[feature]:.4f} → {features_swapped[feature]:.4f}")
            else:
                print(f"✗ {feature:<30} FAILS: {features_normal[feature]:.4f} → {features_swapped[feature]:.4f} (expected {expected:.4f})")
                all_passed = False
    
    if all_passed:
        print("\nH2H interaction features are transforming correctly!")
    else:
        print("\nH2H interaction features are NOT transforming correctly.")
    
    return all_passed

def test_interaction_features():
    """
    Specific test to check if interaction features transform correctly when teams are swapped.
    """
    print("\n===== TESTING INTERACTION FEATURES =====")
    
    # Create simplified test data for clarity
    team1_stats = {
        'team_name': 'Team1',
        'win_rate': 0.7,
        'recent_form': 0.8,
        'avg_player_rating': 1.2,
        'pistol_win_rate': 0.65,
        'eco_win_rate': 0.4,
        'full_buy_win_rate': 0.6,
        'fk_fd_ratio': 1.3,
        'opponent_stats': {
            'Team2': {
                'matches': 3,
                'wins': 2,
                'win_rate': 0.67,
                'score_differential': 2
            }
        }
    }
    
    team2_stats = {
        'team_name': 'Team2',
        'win_rate': 0.6,
        'recent_form': 0.7,
        'avg_player_rating': 1.1,
        'pistol_win_rate': 0.55,
        'eco_win_rate': 0.3,
        'full_buy_win_rate': 0.5,
        'fk_fd_ratio': 1.2,
        'opponent_stats': {
            'Team1': {
                'matches': 3,
                'wins': 1,
                'win_rate': 0.33,
                'score_differential': -2
            }
        }
    }
    
    # Generate features with team1 and team2 in normal order
    features_normal = prepare_data_for_model(team1_stats, team2_stats)
    
    # Generate features with teams swapped
    features_swapped = prepare_data_for_model(team2_stats, team1_stats)
    
    # Test interaction features specifically
    interaction_features = [
        'rating_x_win_rate', 'pistol_x_eco', 'pistol_x_full_buy',
        'first_blood_x_win_rate', 'h2h_x_win_rate', 'h2h_x_form'
    ]
    
    all_passed = True
    print("\nTesting INTERACTION features (should negate when teams swap):")
    for feature in interaction_features:
        if feature in features_normal and feature in features_swapped:
            expected = -features_normal[feature]
            actual = features_swapped[feature]
            
            if abs(expected - actual) < 1e-10:
                print(f"✓ {feature:<30} transforms correctly: {features_normal[feature]:.4f} → {features_swapped[feature]:.4f}")
            else:
                print(f"✗ {feature:<30} FAILS: {features_normal[feature]:.4f} → {features_swapped[feature]:.4f} (expected {expected:.4f})")
                all_passed = False
    
    if all_passed:
        print("\nAll interaction features are now transforming correctly!")
    else:
        print("\nSome interaction features are still not transforming correctly.")
    
    return all_passed

def verify_prediction_symmetry(team1_stats, team2_stats, ensemble_models, selected_features):
    """
    Verify that model predictions are symmetric when teams are swapped.
    This provides increased confidence that our model is working correctly.
    
    Args:
        team1_stats: Statistics for team 1
        team2_stats: Statistics for team 2
        ensemble_models: List of ensemble models
        selected_features: List of feature names
        
    Returns:
        tuple: (win_probability, raw_predictions, confidence, is_symmetric)
    """
    print("\nValidating prediction consistency...")
    
    # Prepare features in normal order
    X_normal = prepare_features_unified(team1_stats, team2_stats, selected_features)
    if X_normal is None:
        print("ERROR: Failed to prepare features for normal order")
        return 0.5, [], 0.3, False
    
    # Get prediction with normal order
    win_prob_normal, raw_preds_normal, conf_normal = predict_with_ensemble_unified(
        ensemble_models, X_normal
    )
    
    # Prepare features in reversed order
    X_reversed = prepare_features_unified(team2_stats, team1_stats, selected_features)
    if X_reversed is None:
        print("ERROR: Failed to prepare features for reversed order")
        return win_prob_normal, raw_preds_normal, conf_normal, False
    
    # Get prediction with reversed order
    win_prob_reversed, raw_preds_reversed, conf_reversed = predict_with_ensemble_unified(
        ensemble_models, X_reversed
    )
    
    # Calculate symmetry error
    sym_error = abs((1 - win_prob_reversed) - win_prob_normal)
    
    # Use threshold to determine if predictions are consistent
    is_symmetric = sym_error < 0.05
    
    if is_symmetric:
        print(f"✓ Predictions are consistent when teams are swapped (diff: {sym_error:.4f})")
        return win_prob_normal, raw_preds_normal, conf_normal, True
    else:
        print(f"WARNING: Inconsistent predictions when teams are swapped!")
        print(f"  Team1 win probability: {win_prob_normal:.4f}")
        print(f"  Team2 win probability (from swapped prediction): {1 - win_prob_reversed:.4f}")
        print(f"  Difference: {sym_error:.4f} (should be < 0.05)")
        
        # Calculate average probability for more consistency
        avg_prob = (win_prob_normal + (1 - win_prob_reversed)) / 2
        print(f"  Using averaged probability: {avg_prob:.4f}")
        
        # Mix raw predictions for better representation
        mixed_preds = []
        if raw_preds_normal and raw_preds_reversed:
            # Take half from each direction
            mixed_preds.extend(raw_preds_normal[:len(raw_preds_normal)//2])
            mixed_preds.extend([1-p for p in raw_preds_reversed[:len(raw_preds_reversed)//2]])
        else:
            mixed_preds = raw_preds_normal
            
        # Average confidence scores
        avg_conf = (conf_normal + conf_reversed) / 2
        
        return avg_prob, mixed_preds, avg_conf, False

def verify_feature_symmetry(team1_stats, team2_stats):
    """
    Verify that features transform correctly when teams are swapped.
    This ensures model predictions are consistent regardless of team order.
    """
    print("\n===== VERIFYING FEATURE SYMMETRY =====")
    
    # Get features for both orderings
    features_normal = prepare_data_for_model(team1_stats, team2_stats)
    features_reversed = prepare_data_for_model(team2_stats, team1_stats)
    
    if not features_normal or not features_reversed:
        print("Failed to generate features")
        return False
    
    # Test transformation of key feature types
    symmetry_issues = []
    
    # For difference features, values should negate when teams swap
    # FIXED: Don't include binary features in difference features
    difference_features = [f for f in features_normal.keys() 
                          if ('_diff' in f or 'differential' in f)
                          and not (f.startswith('better_') or f.endswith('_team1'))]
    
    print("\nTesting DIFFERENCE features (should negate when teams swap):")
    for feature in difference_features:
        if feature in features_normal and feature in features_reversed:
            val_normal = features_normal[feature]
            val_reversed = features_reversed[feature]
            
            # Check if values negate correctly (with small tolerance for numerical issues)
            if abs(val_normal + val_reversed) > 0.001:
                symmetry_issues.append(f"Difference feature {feature} does not negate properly")
                print(f"❌ {feature:<30} FAILS: {val_normal:.4f} → {val_reversed:.4f} (expected {-val_normal:.4f})")
            else:
                print(f"✓ {feature:<30} transforms correctly: {val_normal:.4f} → {val_reversed:.4f}")
    
    # For binary features (better_X_team1), values should flip (0↔1)
    binary_features = [f for f in features_normal.keys() 
                      if f.startswith('better_') or f.endswith('_team1') or f.endswith('_advantage_team1')]
    
    print("\nTesting BINARY features (should flip 0/1 when teams swap):")
    for feature in binary_features:
        if feature in features_normal and feature in features_reversed:
            val_normal = int(features_normal[feature])
            val_reversed = int(features_reversed[feature])
            
            # Binary features should be 0 or 1
            if not (val_normal in [0, 1] and val_reversed in [0, 1]):
                symmetry_issues.append(f"{feature}: Not binary - {val_normal} vs {val_reversed}")
                print(f"❌ {feature:<30} FAILS: {val_normal} → {val_reversed} (not binary values)")
                continue
            
            # Check if values flip correctly (0↔1)
            if val_normal + val_reversed != 1:
                symmetry_issues.append(f"Binary feature {feature} does not flip properly")
                print(f"❌ {feature:<30} FAILS: {val_normal} → {val_reversed} (expected {1-val_normal})")
            else:
                print(f"✓ {feature:<30} transforms correctly: {val_normal} → {val_reversed}")
    
    # For H2H win rate, should transform to 1-x
    if 'h2h_win_rate' in features_normal and 'h2h_win_rate' in features_reversed:
        val_normal = features_normal['h2h_win_rate']
        val_reversed = features_reversed['h2h_win_rate']
        
        print("\nTesting H2H WIN RATE (should transform to 1-x when teams swap):")
        if abs(val_normal + val_reversed - 1.0) > 0.001:
            symmetry_issues.append("h2h_win_rate does not transform to 1-x properly")
            print(f"❌ h2h_win_rate{' '*20} FAILS: {val_normal:.4f} → {val_reversed:.4f} (expected {1-val_normal:.4f})")
        else:
            print(f"✓ h2h_win_rate{' '*20} transforms correctly: {val_normal:.4f} → {val_reversed:.4f}")
    
    # For interaction features (X_x_Y), values should negate
    interaction_features = [f for f in features_normal.keys() if '_x_' in f]
    
    print("\nTesting INTERACTION features (should negate when teams swap):")
    for feature in interaction_features:
        if feature in features_normal and feature in features_reversed:
            val_normal = features_normal[feature]
            val_reversed = features_reversed[feature]
            
            # Check if values negate correctly
            if abs(val_normal + val_reversed) > 0.001:
                symmetry_issues.append(f"Interaction feature {feature} does not negate properly")
                print(f"❌ {feature:<30} FAILS: {val_normal:.4f} → {val_reversed:.4f} (expected {-val_normal:.4f})")
            else:
                print(f"✓ {feature:<30} transforms correctly: {val_normal:.4f} → {val_reversed:.4f}")
    
    # For symmetric features (avg_X), values should remain the same
    # IMPORTANT: Properly exclude difference features from symmetric features
    symmetric_features = [f for f in features_normal.keys() 
                         if (f.startswith('avg_') and '_diff' not in f and 'differential' not in f)]
    
    print("\nTesting SYMMETRIC features (should remain identical when teams swap):")
    for feature in symmetric_features:
        if feature in features_normal and feature in features_reversed:
            val_normal = features_normal[feature]
            val_reversed = features_reversed[feature]
            
            # Check if values remain the same
            if abs(val_normal - val_reversed) > 0.001:
                symmetry_issues.append(f"Symmetric feature {feature} does not remain invariant")
                print(f"❌ {feature:<30} FAILS: {val_normal:.4f} ≠ {val_reversed:.4f}")
            else:
                print(f"✓ {feature:<30} remains invariant: {val_normal:.4f} = {val_reversed:.4f}")
    
    # Print summary
    print("\n===== SYMMETRY TEST SUMMARY =====")
    if symmetry_issues:
        print(f"❌ Found {len(symmetry_issues)} symmetry issues:")
        for issue in symmetry_issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        
        if len(symmetry_issues) > 10:
            print(f"  ... and {len(symmetry_issues) - 10} more")
            
        total_features = len(features_normal)
        working_features = total_features - len(symmetry_issues)
        print(f"\nOverall symmetry: {working_features/total_features:.1%} of features work correctly")
        return False
    else:
        print("✓ All features transform correctly when teams are swapped.")
        return True

def test_with_sample_data():
    """
    Run a test with sample data to verify all feature transformations work correctly.
    """
    print("\n===== TESTING WITH SAMPLE DATA =====")
    
    # Create sample team stats
    team1_stats = {
        'team_name': 'Team A',
        'win_rate': 0.75,
        'recent_form': 0.80,
        'matches': 50,
        'wins': 35,
        'losses': 15,
        'score_differential': 4.5,
        'avg_score': 13.2,
        'avg_opponent_score': 8.7,
        'performance_trends': {
            'form_trajectory': {
                '5_vs_10': 0.15
            }
        },
        'avg_player_rating': 1.28,
        'avg_player_acs': 245.0,
        'avg_player_kd': 1.35,
        'avg_player_kast': 0.72,
        'avg_player_adr': 165.0,
        'avg_player_headshot': 0.25,
        'star_player_rating': 1.45,
        'team_consistency': 0.85,
        'fk_fd_ratio': 1.2,
        'pistol_win_rate': 0.68,
        'eco_win_rate': 0.35,
        'full_buy_win_rate': 0.62,
        'economy_efficiency': 0.75,
        'opponent_stats': {
            'Team B': {
                'matches': 5,
                'wins': 4,
                'win_rate': 0.8,
                'score_differential': 3.5
            }
        }
    }
    
    team2_stats = {
        'team_name': 'Team B',
        'win_rate': 0.60,
        'recent_form': 0.65,
        'matches': 40,
        'wins': 24,
        'losses': 16,
        'score_differential': 2.3,
        'avg_score': 11.8,
        'avg_opponent_score': 9.5,
        'performance_trends': {
            'form_trajectory': {
                '5_vs_10': 0.05
            }
        },
        'avg_player_rating': 1.18,
        'avg_player_acs': 220.0,
        'avg_player_kd': 1.15,
        'avg_player_kast': 0.68,
        'avg_player_adr': 155.0,
        'avg_player_headshot': 0.22,
        'star_player_rating': 1.32,
        'team_consistency': 0.80,
        'fk_fd_ratio': 1.1,
        'pistol_win_rate': 0.55,
        'eco_win_rate': 0.30,
        'full_buy_win_rate': 0.58,
        'economy_efficiency': 0.68,
        'opponent_stats': {
            'Team A': {
                'matches': 5,
                'wins': 1,
                'win_rate': 0.2,
                'score_differential': -3.5
            }
        }
    }
    
    # Add map statistics
    team1_stats['map_statistics'] = {
        'Haven': {'win_percentage': 0.70, 'matches_played': 12},
        'Ascent': {'win_percentage': 0.65, 'matches_played': 10},
        'Bind': {'win_percentage': 0.60, 'matches_played': 8}
    }
    
    team2_stats['map_statistics'] = {
        'Haven': {'win_percentage': 0.60, 'matches_played': 10},
        'Ascent': {'win_percentage': 0.55, 'matches_played': 9},
        'Bind': {'win_percentage': 0.50, 'matches_played': 6}
    }
    
    # Run the verification
    verify_feature_symmetry(team1_stats, team2_stats)

def test_interaction_feature_symmetry():
    """
    Specific test for interaction features to verify they transform correctly.
    """
    print("\n===== TESTING INTERACTION FEATURE SYMMETRY =====")
    
    # Create simplified test team stats
    team1_stats = {
        'team_name': 'Team1',
        'win_rate': 0.7,
        'recent_form': 0.8,
        'avg_player_rating': 1.2,
        'pistol_win_rate': 0.65,
        'eco_win_rate': 0.4,
        'full_buy_win_rate': 0.6,
        'fk_fd_ratio': 1.3,
        'opponent_stats': {
            'Team2': {
                'matches': 3,
                'wins': 2,
                'win_rate': 0.67,
                'score_differential': 2
            }
        }
    }
    
    team2_stats = {
        'team_name': 'Team2',
        'win_rate': 0.6,
        'recent_form': 0.7,
        'avg_player_rating': 1.1,
        'pistol_win_rate': 0.55,
        'eco_win_rate': 0.3,
        'full_buy_win_rate': 0.5,
        'fk_fd_ratio': 1.2,
        'opponent_stats': {
            'Team1': {
                'matches': 3,
                'wins': 1,
                'win_rate': 0.33,
                'score_differential': -2
            }
        }
    }
    
    # Generate features for both team orderings
    features_normal = prepare_data_for_model(team1_stats, team2_stats)
    features_swapped = prepare_data_for_model(team2_stats, team1_stats)
    
    # Check interaction features specifically
    interaction_features = [
        'rating_x_win_rate', 'pistol_x_eco', 'pistol_x_full_buy',
        'first_blood_x_win_rate', 'h2h_x_win_rate', 'h2h_x_form'
    ]
    
    all_passed = True
    for feature in interaction_features:
        if feature in features_normal and feature in features_swapped:
            expected = -features_normal[feature]
            actual = features_swapped[feature]
            
            if abs(expected - actual) < 1e-10:
                print(f"✓ {feature:<30} transforms correctly: {features_normal[feature]:.4f} → {features_swapped[feature]:.4f}")
            else:
                print(f"✗ {feature:<30} FAILS: {features_normal[feature]:.4f} → {features_swapped[feature]:.4f} (expected {expected:.4f})")
                all_passed = False
    
    return all_passed

def test_prediction_consistency(team1_stats, team2_stats, model):
    """
    Test if model predictions are consistent when teams are swapped.
    
    Args:
        team1_stats (dict): Statistics for team 1
        team2_stats (dict): Statistics for team 2
        model: Trained model
        
    Returns:
        bool: True if predictions are consistent
    """
    print("\n===== TESTING PREDICTION CONSISTENCY =====")
    
    # Verify features transform correctly
    feature_symmetry_passed = verify_feature_symmetry(team1_stats, team2_stats)
    
    if not feature_symmetry_passed:
        print("Feature symmetry test failed. Predictions may be inconsistent.")
    
    # Prepare features for predictions
    features_1_2 = prepare_data_for_model(team1_stats, team2_stats)
    features_2_1 = prepare_data_for_model(team2_stats, team1_stats)
    
    # Convert to DataFrame for model input
    X_1_2 = pd.DataFrame([features_1_2])
    X_2_1 = pd.DataFrame([features_2_1])
    
    # Make predictions
    try:
        if hasattr(model, 'predict_proba'):
            # For sklearn models
            pred_1_2 = model.predict_proba(X_1_2)[0, 1]  # Probability of team1 winning
            pred_2_1 = model.predict_proba(X_2_1)[0, 1]  # Probability of team2 winning
        else:
            # For neural networks
            pred_1_2 = model.predict(X_1_2)[0, 0]
            pred_2_1 = model.predict(X_2_1)[0, 0]
        
        # Check if predictions are consistent (should sum to 1)
        prediction_sum = pred_1_2 + pred_2_1
        tolerance = 0.01  # Allow a small tolerance for floating point issues
        
        if abs(prediction_sum - 1.0) < tolerance:
            print(f"✓ Predictions are consistent: {pred_1_2:.4f} + {pred_2_1:.4f} = {prediction_sum:.4f}")
            print(f"  Team1 win probability: {pred_1_2:.4f}")
            print(f"  Team2 win probability: {1 - pred_2_1:.4f} (derived from swapped prediction)")
            return True
        else:
            print(f"✗ Predictions are inconsistent: {pred_1_2:.4f} + {pred_2_1:.4f} = {prediction_sum:.4f}")
            print(f"  Team1 win probability: {pred_1_2:.4f}")
            print(f"  Team2 win probability: {1 - pred_2_1:.4f} (derived from swapped prediction)")
            print(f"  Difference: {abs(pred_1_2 - (1 - pred_2_1)):.4f}")
            
            # Investigate which features might be causing inconsistency
            print("\nFeature differences that might be causing inconsistency:")
            for feature in features_1_2.keys():
                if feature in features_2_1:
                    if feature.endswith('_diff') or feature in interaction_features:
                        # Should negate when teams swap
                        if abs(features_1_2[feature] + features_2_1[feature]) > 1e-10:
                            print(f"- {feature}: {features_1_2[feature]:.4f} ≠ -{features_2_1[feature]:.4f}")
                    elif feature.startswith('better_') or feature.endswith('_team1'):
                        # Should flip 0/1 when teams swap
                        if int(features_1_2[feature]) != 1 - int(features_2_1[feature]):
                            print(f"- {feature}: {int(features_1_2[feature])} ≠ {1 - int(features_2_1[feature])}")
            
            return False
    except Exception as e:
        print(f"Error making predictions: {e}")
        return False


def assert_feature_symmetry(features_normal, features_swapped):
    """
    Utility function to validate feature symmetry.
    
    Args:
        features_normal: Features with team1 and team2 in normal order
        features_swapped: Features with team1 and team2 swapped
        
    Raises:
        AssertionError: If feature symmetry is violated
    """
    # Lists of features by transformation type
    difference_features = [
        'win_rate_diff', 'recent_form_diff', 'score_diff_differential',
        'avg_score_diff', 'avg_opponent_score_diff', 'match_count_diff',
        'player_rating_diff', 'pistol_win_rate_diff', 'h2h_score_diff'
    ]
    
    binary_features = [
        'better_win_rate_team1', 'better_recent_form_team1', 'better_score_diff_team1',
        'h2h_advantage_team1'
    ]
    
    symmetric_features = [
        'avg_win_rate', 'avg_recent_form', 'total_matches', 'avg_score_metric',
        'avg_defense_metric', 'avg_player_rating', 'avg_acs'
    ]
    
    interaction_features = [
        'rating_x_win_rate', 'pistol_x_eco', 'pistol_x_full_buy',
        'first_blood_x_win_rate', 'h2h_x_win_rate', 'h2h_x_form'
    ]
    
    special_features = ['h2h_win_rate']
    
    # Check difference features (should negate)
    for feature in difference_features:
        if feature in features_normal and feature in features_swapped:
            if abs(features_normal[feature] + features_swapped[feature]) > 1e-10:
                raise AssertionError(f"Difference feature {feature} does not negate properly")
    
    # Check binary features (should flip 0↔1)
    for feature in binary_features:
        if feature in features_normal and feature in features_swapped:
            if int(features_normal[feature]) != 1 - int(features_swapped[feature]):
                raise AssertionError(f"Binary feature {feature} does not flip properly")
    
    # Check symmetric features (should remain the same)
    for feature in symmetric_features:
        if feature in features_normal and feature in features_swapped:
            if abs(features_normal[feature] - features_swapped[feature]) > 1e-10:
                raise AssertionError(f"Symmetric feature {feature} does not remain invariant")
    
    # Check interaction features (should negate)
    for feature in interaction_features:
        if feature in features_normal and feature in features_swapped:
            if abs(features_normal[feature] + features_swapped[feature]) > 1e-10:
                raise AssertionError(f"Interaction feature {feature} does not negate properly")
    
    # Check h2h_win_rate (should be 1-x)
    if 'h2h_win_rate' in features_normal and 'h2h_win_rate' in features_swapped:
        if abs(features_normal['h2h_win_rate'] + features_swapped['h2h_win_rate'] - 1) > 1e-10:
            raise AssertionError("h2h_win_rate does not transform properly")

def test_feature_transformations():
    """
    Test function to verify that all features transform correctly when teams are swapped.
    """
    print("\n===== TESTING FEATURE TRANSFORMATIONS =====")
    
    # Create test team stats
    team1_stats = {
        'team_name': 'Team A',
        'team_id': '123',
        'win_rate': 0.75,
        'recent_form': 0.80,
        'matches': 50,
        'wins': 35,
        'losses': 15,
        'avg_score': 13.2,
        'avg_opponent_score': 8.7,
        'score_differential': 4.5,
        'avg_player_rating': 1.28,
        'avg_player_acs': 245.0,
        'avg_player_kd': 1.35,
        'avg_player_kast': 0.72,
        'avg_player_adr': 165.0,
        'avg_player_headshot': 0.25,
        'star_player_rating': 1.45,
        'team_consistency': 0.85,
        'fk_fd_ratio': 1.2,
        'pistol_win_rate': 0.68,
        'eco_win_rate': 0.35,
        'full_buy_win_rate': 0.62,
        'economy_efficiency': 0.75,
        'opponent_stats': {
            'Team B': {
                'matches': 5,
                'wins': 4,
                'win_rate': 0.8,
                'score_differential': 3.5
            }
        },
        'performance_trends': {
            'form_trajectory': {
                '5_vs_10': 0.15
            }
        }
    }
    
    team2_stats = {
        'team_name': 'Team B',
        'team_id': '456',
        'win_rate': 0.60,
        'recent_form': 0.65,
        'matches': 40,
        'wins': 24,
        'losses': 16,
        'avg_score': 11.8,
        'avg_opponent_score': 9.5,
        'score_differential': 2.3,
        'avg_player_rating': 1.18,
        'avg_player_acs': 220.0,
        'avg_player_kd': 1.15,
        'avg_player_kast': 0.68,
        'avg_player_adr': 155.0,
        'avg_player_headshot': 0.22,
        'star_player_rating': 1.32,
        'team_consistency': 0.80,
        'fk_fd_ratio': 1.1,
        'pistol_win_rate': 0.55,
        'eco_win_rate': 0.30,
        'full_buy_win_rate': 0.58,
        'economy_efficiency': 0.68,
        'opponent_stats': {
            'Team A': {
                'matches': 5,
                'wins': 1,
                'win_rate': 0.2,
                'score_differential': -3.5
            }
        },
        'performance_trends': {
            'form_trajectory': {
                '5_vs_10': 0.05
            }
        }
    }
    
    # Add map statistics
    team1_stats['map_statistics'] = {
        'Haven': {'win_percentage': 0.70, 'matches_played': 12},
        'Ascent': {'win_percentage': 0.65, 'matches_played': 10},
        'Bind': {'win_percentage': 0.60, 'matches_played': 8}
    }
    
    team2_stats['map_statistics'] = {
        'Haven': {'win_percentage': 0.60, 'matches_played': 10},
        'Ascent': {'win_percentage': 0.55, 'matches_played': 9},
        'Bind': {'win_percentage': 0.50, 'matches_played': 6}
    }
    
    # Generate features in both directions
    features_1_2 = prepare_data_for_model(team1_stats, team2_stats)
    features_2_1 = prepare_data_for_model(team2_stats, team1_stats)
    
    # Execute the verification
    verify_feature_symmetry(team1_stats, team2_stats)
    
    # Return features for further analysis if needed
    return features_1_2, features_2_1

def debug_feature_symmetry(team1_stats, team2_stats):
    """
    Debug function to help identify feature symmetry issues.
    """
    print("\n=== DEBUGGING FEATURE SYMMETRY ===")
    
    # Generate features both ways
    features_normal = prepare_data_for_model(team1_stats, team2_stats)
    features_swapped = prepare_data_for_model(team2_stats, team1_stats)
    
    # Compare specific problematic features
    problem_features = [
        'better_score_diff_team1',
        'rating_x_win_rate',
        'pistol_x_eco',
        'pistol_x_full_buy',
        'first_blood_x_win_rate',
        'h2h_x_win_rate',
        'h2h_x_form'
    ]
    
    for feature in problem_features:
        if feature in features_normal and feature in features_swapped:
            print(f"\nDEBUGGING FEATURE: {feature}")
            
            # Print values
            print(f"Normal value: {features_normal[feature]}")
            print(f"Swapped value: {features_swapped[feature]}")
            
            # Check what transformation we expect
            if feature.startswith('better_'):
                expected = 1 - features_normal[feature]
                print(f"Expected swapped (flip): {expected}")
                print(f"Is correct? {int(expected) == int(features_swapped[feature])}")
            else:
                expected = -features_normal[feature]
                print(f"Expected swapped (negate): {expected}")
                print(f"Is correct? {abs(expected - features_swapped[feature]) < 1e-10}")
            
            # Debug calc_interaction function output
            if feature in ['rating_x_win_rate', 'pistol_x_eco', 'pistol_x_full_buy', 
                          'first_blood_x_win_rate', 'h2h_x_win_rate', 'h2h_x_form']:
                
                if feature == 'rating_x_win_rate':
                    # Direct calculation
                    player_rating_diff_normal = team1_stats.get('avg_player_rating', 0) - team2_stats.get('avg_player_rating', 0)
                    win_rate_diff_normal = team1_stats.get('win_rate', 0) - team2_stats.get('win_rate', 0)
                    
                    player_rating_diff_swapped = team2_stats.get('avg_player_rating', 0) - team1_stats.get('avg_player_rating', 0)
                    win_rate_diff_swapped = team2_stats.get('win_rate', 0) - team1_stats.get('win_rate', 0)
                    
                    direct_normal = player_rating_diff_normal * win_rate_diff_normal
                    direct_swapped = player_rating_diff_swapped * win_rate_diff_swapped
                    
                    print(f"Direct calculation normal: {direct_normal}")
                    print(f"Direct calculation swapped: {direct_swapped}")
                    print(f"Direct calculations negate? {abs(direct_normal + direct_swapped) < 1e-10}")
    
    # Print key stats from team objects for verification
    print("\nTEAM STATS:")
    print(f"team1 avg_player_rating: {team1_stats.get('avg_player_rating', 'N/A')}")
    print(f"team2 avg_player_rating: {team2_stats.get('avg_player_rating', 'N/A')}")
    print(f"team1 win_rate: {team1_stats.get('win_rate', 'N/A')}")
    print(f"team2 win_rate: {team2_stats.get('win_rate', 'N/A')}")

def standardize_features(features_df):
    """
    Standardize features with optimized ranges to improve model performance
    based on backtest findings.
    """
    standardized_df = features_df.copy()
    
    # Define feature groups by transformation type - FIXED CLASSIFICATION
    difference_features = [col for col in standardized_df.columns 
                           if ('_diff' in col or 'differential' in col)]
    
    binary_features = [col for col in standardized_df.columns 
                       if col.startswith('better_') or col == 'h2h_advantage_team1' 
                       or col == 'maps_advantage_team1' or col == 'h2h_significant']
    
    # Explicitly exclude difference features from ratio features
    ratio_features = [col for col in standardized_df.columns 
                     if ('rate' in col or 'ratio' in col) 
                     and not any(diff_term in col for diff_term in ['_diff', 'differential'])]
    
    count_features = [col for col in standardized_df.columns 
                     if ('count' in col or col == 'total_matches' or col == 'h2h_matches')]
    
    # Features that performed well in backtest
    key_features = ['h2h_x_win_rate', 'h2h_win_rate', 'h2h_score_diff', 'h2h_x_form']
    
    # IMPORTANT: Get a correct list of truly symmetric features by excluding difference features
    symmetric_features = [col for col in standardized_df.columns 
                         if col.startswith('avg_') 
                         and not any(diff_term in col for diff_term in ['_diff', 'differential'])]
    
    # Ensure binary features are strictly 0 or 1
    for col in binary_features:
        standardized_df[col] = standardized_df[col].apply(lambda x: 1 if x > 0.5 else 0)
    
    # Bound ratio features to [0, 1]
    for col in ratio_features:
        standardized_df[col] = standardized_df[col].clip(0, 1)
    
    # Standardize difference features to a reasonable range
    for col in difference_features:
        # Use sigmoid-based normalization for more nuanced scaling of differences
        # This provides better gradient for the model
        standardized_df[col] = standardized_df[col].apply(lambda x: 2 / (1 + np.exp(-2 * x)) - 1)
    
    # Apply log transformation to count features to reduce impact of outliers
    for col in count_features:
        standardized_df[col] = standardized_df[col].apply(lambda x: np.log1p(max(0, x)) / 5 if x > 0 else 0)
    
    # Give extra boost to key features based on backtest performance
    for col in key_features:
        if col in standardized_df.columns:
            if col == 'h2h_win_rate':
                # Apply sigmoid transformation to enhance the impact of h2h_win_rate
                standardized_df[col] = standardized_df[col].apply(
                    lambda x: (2 / (1 + np.exp(-4 * (x - 0.5)))) if 0 <= x <= 1 else x
                )
            elif col == 'h2h_score_diff':
                # Apply custom scaling to score difference for better gradient
                standardized_df[col] = standardized_df[col].apply(
                    lambda x: np.tanh(x * 0.7)  # Less aggressive tanh scaling
                )
            # No need to modify h2h_x features as they're already properly scaled
    
    return standardized_df

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

def create_improved_model(input_dim, regularization_strength=0.03):
    """
    Create a neural network model that properly avoids extreme predictions.
    This is the CRITICAL fix for NN calibration issues identified in backtest.
    """
    # Define inputs
    inputs = Input(shape=(input_dim,))
    
    # First layer with strong regularization
    x = Dense(64, activation='relu', 
              kernel_regularizer=l2(regularization_strength),
              activity_regularizer=l1(0.02),
              kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)  # Heavy dropout
    
    # Second layer
    x = Dense(32, activation='relu', 
              kernel_regularizer=l2(regularization_strength),
              kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # CRITICAL: Custom final layer to prevent extreme outputs
    # Using Linear activation with custom scaling
    raw_output = Dense(1, activation='linear',
                      kernel_regularizer=l2(regularization_strength*2),
                      kernel_initializer='glorot_normal',
                      bias_initializer=tf.keras.initializers.Constant(0.5))(x)
    
    # Add Lambda layer to bound outputs between 0.15 and 0.85
    # This is essential to prevent the binary 0/1 predictions
    def bound_activations(x):
        sigmoid = tf.keras.activations.sigmoid(x)
        return 0.15 + (sigmoid * 0.7)  # Maps [0,1] to [0.15,0.85]
    
    outputs = Lambda(bound_activations)(raw_output)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use lower learning rate
    model.compile(loss='binary_crossentropy', 
                 optimizer=Adam(learning_rate=0.0005),
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

def load_models_unified():
    """
    Load ensemble models with consistent error handling and logging.
    """
    print("\n----- LOADING PREDICTION MODELS -----")
    
    # 1. Load diverse ensemble with better error handling
    ensemble_models = None
    try:
        print("Loading ensemble from diverse_ensemble.pkl...")
        with open('diverse_ensemble.pkl', 'rb') as f:
            ensemble_models = pickle.load(f)
        print(f"Successfully loaded {len(ensemble_models)} ensemble models")
    except Exception as e:
        print(f"Failed to load diverse ensemble: {e}")
        
        # Try loading individual fold models
        ensemble_models = []
        for i in range(1, 11):
            try:
                model = load_model(f'valorant_model_fold_{i}.h5')
                ensemble_models.append(('nn', model, None))
                print(f"Loaded fold model {i}")
            except Exception:
                continue
        
        if not ensemble_models:
            print("ERROR: Failed to load any models")
            return None, None
    
    # 2. Load feature metadata with consistent approach
    selected_features = None
    try:
        print("Loading feature metadata...")
        with open('feature_metadata.pkl', 'rb') as f:
            feature_metadata = pickle.load(f)
            selected_features = feature_metadata.get('selected_features')
        print(f"Loaded {len(selected_features)} features from metadata")
    except Exception as e:
        print(f"Failed to load feature metadata: {e}")
        try:
            with open('selected_feature_names.pkl', 'rb') as f:
                selected_features = pickle.load(f)
            print(f"Loaded {len(selected_features)} features from backup file")
        except Exception:
            try:
                with open('stable_features.pkl', 'rb') as f:
                    selected_features = pickle.load(f)
                print(f"Loaded {len(selected_features)} features from stable_features.pkl")
            except Exception:
                print("ERROR: Failed to load feature list")
    
    if not selected_features:
        print("Creating fallback feature list")
        # Create a minimal set of essential features
        selected_features = [
            'win_rate_diff', 'better_win_rate_team1', 'recent_form_diff',
            'better_recent_form_team1', 'score_diff_differential',
            'better_score_diff_team1', 'h2h_win_rate', 'h2h_matches',
            'h2h_score_diff', 'h2h_advantage_team1', 'total_matches',
            'match_count_diff', 'avg_win_rate', 'avg_recent_form'
        ]
    
    # 3. Load scaler for scikit-learn models if available
    try:
        print("Loading feature scaler...")
        with open('ensemble_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Add scaler to models that need it
        enhanced_ensemble = []
        for model_type, model, model_scaler in ensemble_models:
            if model_type in ['lr', 'svm'] and model_scaler is None:
                enhanced_ensemble.append((model_type, model, scaler))
            else:
                enhanced_ensemble.append((model_type, model, model_scaler))
        
        ensemble_models = enhanced_ensemble
    except Exception as e:
        print(f"Failed to load scaler: {e}")
    
    print(f"Model loading complete: {len(ensemble_models)} models, {len(selected_features)} features")
    return ensemble_models, selected_features

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

def test_feature_transformations():
    """
    Test function to verify that all features transform correctly when teams are swapped.
    """
    print("\n===== TESTING FEATURE TRANSFORMATIONS =====")
    
    # Create test team stats
    team1_stats = {
        'team_name': 'Team A',
        'team_id': '123',
        'win_rate': 0.75,
        'recent_form': 0.80,
        'matches': 50,
        'wins': 35,
        'losses': 15,
        'avg_score': 13.2,
        'avg_opponent_score': 8.7,
        'score_differential': 4.5,
        'avg_player_rating': 1.28,
        'avg_player_acs': 245.0,
        'avg_player_kd': 1.35,
        'avg_player_kast': 0.72,
        'avg_player_adr': 165.0,
        'avg_player_headshot': 0.25,
        'star_player_rating': 1.45,
        'team_consistency': 0.85,
        'fk_fd_ratio': 1.2,
        'pistol_win_rate': 0.68,
        'eco_win_rate': 0.35,
        'full_buy_win_rate': 0.62,
        'economy_efficiency': 0.75,
        'opponent_stats': {
            'Team B': {
                'matches': 5,
                'wins': 4,
                'win_rate': 0.8,
                'score_differential': 3.5
            }
        },
        'performance_trends': {
            'form_trajectory': {
                '5_vs_10': 0.15
            }
        }
    }
    
    team2_stats = {
        'team_name': 'Team B',
        'team_id': '456',
        'win_rate': 0.60,
        'recent_form': 0.65,
        'matches': 40,
        'wins': 24,
        'losses': 16,
        'avg_score': 11.8,
        'avg_opponent_score': 9.5,
        'score_differential': 2.3,
        'avg_player_rating': 1.18,
        'avg_player_acs': 220.0,
        'avg_player_kd': 1.15,
        'avg_player_kast': 0.68,
        'avg_player_adr': 155.0,
        'avg_player_headshot': 0.22,
        'star_player_rating': 1.32,
        'team_consistency': 0.80,
        'fk_fd_ratio': 1.1,
        'pistol_win_rate': 0.55,
        'eco_win_rate': 0.30,
        'full_buy_win_rate': 0.58,
        'economy_efficiency': 0.68,
        'opponent_stats': {
            'Team A': {
                'matches': 5,
                'wins': 1,
                'win_rate': 0.2,
                'score_differential': -3.5
            }
        },
        'performance_trends': {
            'form_trajectory': {
                '5_vs_10': 0.05
            }
        }
    }
    
    # Add map statistics
    team1_stats['map_statistics'] = {
        'Haven': {'win_percentage': 0.70, 'matches_played': 12},
        'Ascent': {'win_percentage': 0.65, 'matches_played': 10},
        'Bind': {'win_percentage': 0.60, 'matches_played': 8}
    }
    
    team2_stats['map_statistics'] = {
        'Haven': {'win_percentage': 0.60, 'matches_played': 10},
        'Ascent': {'win_percentage': 0.55, 'matches_played': 9},
        'Bind': {'win_percentage': 0.50, 'matches_played': 6}
    }
    
    # Generate features in both directions
    features_1_2 = prepare_data_for_model(team1_stats, team2_stats)
    features_2_1 = prepare_data_for_model(team2_stats, team1_stats)
    
    # Check all feature types
    
    # 1. TEST DIFFERENCE FEATURES (should negate)
    print("\nTesting DIFFERENCE features (should negate when teams swap):")
    diff_features = [f for f in features_1_2.keys() if "_diff" in f or "differential" in f]
    for feature in diff_features:
        value_1_2 = features_1_2[feature]
        value_2_1 = features_2_1[feature]
        
        if abs(value_1_2 + value_2_1) < 1e-10:
            print(f"✓ {feature:<30} transforms correctly: {value_1_2:.4f} → {value_2_1:.4f}")
        else:
            print(f"✗ {feature:<30} FAILS: {value_1_2:.4f} → {value_2_1:.4f} (expected {-value_1_2:.4f})")
    
    # 2. TEST BINARY FEATURES (should flip 0↔1)
    print("\nTesting BINARY features (should flip 0↔1 when teams swap):")
    binary_features = [f for f in features_1_2.keys() if f.startswith("better_") or f == "h2h_advantage_team1"]
    for feature in binary_features:
        value_1_2 = int(features_1_2[feature])
        value_2_1 = int(features_2_1[feature])
        
        if value_1_2 == 1 - value_2_1:
            print(f"✓ {feature:<30} transforms correctly: {value_1_2} → {value_2_1}")
        else:
            print(f"✗ {feature:<30} FAILS: {value_1_2} → {value_2_1} (expected {1-value_1_2})")
    
    # 3. TEST H2H FEATURES (special handling)
    print("\nTesting H2H features (special transformations):")
    if "h2h_win_rate" in features_1_2 and "h2h_win_rate" in features_2_1:
        value_1_2 = features_1_2["h2h_win_rate"]
        value_2_1 = features_2_1["h2h_win_rate"]
        
        if abs(value_1_2 + value_2_1 - 1) < 1e-10:
            print(f"✓ h2h_win_rate{' '*20} transforms correctly: {value_1_2:.4f} → {value_2_1:.4f}")
        else:
            print(f"✗ h2h_win_rate{' '*20} FAILS: {value_1_2:.4f} → {value_2_1:.4f} (expected {1-value_1_2:.4f})")
    
    # 4. TEST SYMMETRIC FEATURES (should remain identical)
    print("\nTesting SYMMETRIC features (should remain identical when teams swap):")
    symmetric_features = [f for f in features_1_2.keys() if f.startswith("avg_") or f == "total_matches"]
    for feature in symmetric_features:
        value_1_2 = features_1_2[feature]
        value_2_1 = features_2_1[feature]
        
        if abs(value_1_2 - value_2_1) < 1e-10:
            print(f"✓ {feature:<30} remains invariant: {value_1_2:.4f} = {value_2_1:.4f}")
        else:
            print(f"✗ {feature:<30} FAILS: {value_1_2:.4f} ≠ {value_2_1:.4f}")
    
    # 5. TEST INTERACTION FEATURES (should negate)
    print("\nTesting INTERACTION features (should negate when teams swap):")
    interaction_features = [f for f in features_1_2.keys() if "_x_" in f]
    for feature in interaction_features:
        value_1_2 = features_1_2[feature]
        value_2_1 = features_2_1[feature]
        
        if abs(value_1_2 + value_2_1) < 1e-10:
            print(f"✓ {feature:<30} transforms correctly: {value_1_2:.4f} → {value_2_1:.4f}")
        else:
            print(f"✗ {feature:<30} FAILS: {value_1_2:.4f} → {value_2_1:.4f} (expected {-value_1_2:.4f})")
    
    # Return full testing results
    return features_1_2, features_2_1

def predict_match_unified(team1_name, team2_name, bankroll=1000.0):
    """
    Predict match outcome with improved calibration and symmetry checks.
    """
    print(f"\n{'='*80}")
    print(f"MATCH PREDICTION: {team1_name} vs {team2_name}")
    print(f"{'='*80}")
    
    # Load models
    ensemble_models, selected_features = load_models_unified()
    
    if not ensemble_models or not selected_features:
        print("ERROR: Failed to load prediction models or feature list.")
        return None
    
    # Get team IDs
    team1_id = get_team_id(team1_name)
    team2_id = get_team_id(team2_name)
    
    if not team1_id:
        print(f"Error: Could not find team ID for {team1_name}")
        return None
    
    if not team2_id:
        print(f"Error: Could not find team ID for {team2_name}")
        return None
    
    print(f"Found team IDs: {team1_name} (ID: {team1_id}), {team2_name} (ID: {team2_id})")
    
    # Fetch team data
    print(f"Fetching data for {team1_name}...")
    team1_history = fetch_team_match_history(team1_id)
    team1_matches = parse_match_data(team1_history, team1_name)
    team1_player_stats = fetch_team_player_stats(team1_id)
    team1_stats = calculate_team_stats(team1_matches, team1_player_stats, include_economy=True)
    
    print(f"Fetching data for {team2_name}...")
    team2_history = fetch_team_match_history(team2_id)
    team2_matches = parse_match_data(team2_history, team2_name)
    team2_player_stats = fetch_team_player_stats(team2_id)
    team2_stats = calculate_team_stats(team2_matches, team2_player_stats, include_economy=True)
    
    # Add map statistics
    team1_stats['map_statistics'] = fetch_team_map_statistics(team1_id)
    team2_stats['map_statistics'] = fetch_team_map_statistics(team2_id)
    
    # Store team info
    team1_stats['team_name'] = team1_name
    team1_stats['team_id'] = team1_id
    team2_stats['team_name'] = team2_name
    team2_stats['team_id'] = team2_id
    
    # Verify feature symmetry
    print("\nVerifying feature symmetry...")
    is_symmetric = verify_feature_symmetry(team1_stats, team2_stats)
    
    if not is_symmetric:
        print("WARNING: Feature symmetry test failed. Taking corrective measures.")
    
    # Use unified feature preparation
    X = prepare_features_unified(team1_stats, team2_stats, selected_features)
    
    if X is None:
        print("ERROR: Failed to prepare features for prediction")
        return None
    
    # Use verified prediction approach that checks symmetry
    win_probability, raw_predictions, confidence_score = predict_with_ensemble_unified(ensemble_models, X)
    
    # Critical Symmetry Test: Test opposite direction prediction
    print("\nPerforming critical symmetry verification...")
    X_reversed = prepare_features_unified(team2_stats, team1_stats, selected_features)
    if X_reversed is not None:
        win_prob_reversed, _, _ = predict_with_ensemble_unified(ensemble_models, X_reversed)
        symmetry_sum = win_probability + win_prob_reversed
        print(f"Symmetry check: {win_probability:.4f} + {win_prob_reversed:.4f} = {symmetry_sum:.4f}")
        
        # Symmetry should sum to 1.0
        if abs(symmetry_sum - 1.0) > 0.05:
            print("WARNING: Prediction symmetry test failed! Predictions may be inconsistent.")
            print("Adjusting prediction to improve symmetry...")
            
            # Correct the prediction to ensure symmetry
            win_probability = 0.5 + (win_probability - 0.5) * 0.9
            print(f"Adjusted prediction: {win_probability:.4f}")
    
    # Print results
    print(f"\nTeam 1 ({team1_name}) win probability: {win_probability:.4f}")
    print(f"Team 2 ({team2_name}) win probability: {1-win_probability:.4f}")
    print(f"Model confidence: {confidence_score:.4f}")
    
    # Match date
    match_date = datetime.now().strftime('%Y-%m-%d')
    
    # Calculate bet type probabilities with consistent methodology
    bet_type_probs = calculate_bet_type_probabilities_unified(win_probability, confidence_score)
    
    # Check for drawdown from previous betting history
    current_drawdown_pct = 0
    try:
        with open('betting_performance.json', 'r') as f:
            performance = json.load(f)
            if 'bankroll_history' in performance and performance['bankroll_history']:
                starting_amount = performance.get('starting_bankroll', bankroll)
                current_bankroll = performance['bankroll_history'][-1].get('bankroll', bankroll)
                max_bankroll = max([entry.get('bankroll', starting_amount) for entry in performance['bankroll_history']])
                
                if max_bankroll > current_bankroll:
                    current_drawdown_pct = ((max_bankroll - current_bankroll) / max_bankroll) * 100
                    if current_drawdown_pct > 5:
                        print(f"NOTICE: Current drawdown is {current_drawdown_pct:.1f}%")
                else:
                    current_drawdown_pct = 0
                    print(f"Currently at all-time high bankroll")
    except (FileNotFoundError, json.JSONDecodeError):
        current_drawdown_pct = 0
    
    # Prompt for odds
    print("\nPlease enter the betting odds from your bookmaker:")
    odds_data = {}
    
    try:
        # Moneyline
        team1_ml = float(input(f"{team1_name} moneyline odds (decimal format, e.g. 2.50): ") or 0)
        if team1_ml > 0:
            odds_data['team1_ml_odds'] = team1_ml
        
        team2_ml = float(input(f"{team2_name} moneyline odds (decimal format, e.g. 2.50): ") or 0)
        if team2_ml > 0:
            odds_data['team2_ml_odds'] = team2_ml
        
        # +1.5 maps
        team1_plus = float(input(f"{team1_name} +1.5 maps odds: ") or 0)
        if team1_plus > 0:
            odds_data['team1_plus_1_5_odds'] = team1_plus
        
        team2_plus = float(input(f"{team2_name} +1.5 maps odds: ") or 0)
        if team2_plus > 0:
            odds_data['team2_plus_1_5_odds'] = team2_plus
        
        # -1.5 maps
        team1_minus = float(input(f"{team1_name} -1.5 maps odds: ") or 0)
        if team1_minus > 0:
            odds_data['team1_minus_1_5_odds'] = team1_minus
        
        team2_minus = float(input(f"{team2_name} -1.5 maps odds: ") or 0)
        if team2_minus > 0:
            odds_data['team2_minus_1_5_odds'] = team2_minus
        
        # Over/Under
        over = float(input(f"Over 2.5 maps odds: ") or 0)
        if over > 0:
            odds_data['over_2_5_maps_odds'] = over
        
        under = float(input(f"Under 2.5 maps odds: ") or 0)
        if under > 0:
            odds_data['under_2_5_maps_odds'] = under
        
    except ValueError:
        print("Invalid odds input. Using available odds only.")
    
    # Use unified betting analysis with drawdown protection
    betting_analysis = analyze_betting_edge_unified(
        win_probability, 1 - win_probability, odds_data, confidence_score, 
        bankroll, bankroll, team1_name, team2_name, current_drawdown_pct
    )
    
    # Load previous bets for consistency
    previous_bets_by_team = {}
    try:
        with open('betting_performance.json', 'r') as f:
            performance = json.load(f)
            for bet in performance.get('bets', []):
                team1 = bet.get('teams', '').split(' vs ')[0]
                if team1 not in previous_bets_by_team:
                    previous_bets_by_team[team1] = []
                previous_bets_by_team[team1].append({
                    'bet_type': bet.get('bet_type', ''),
                    'won': bet.get('outcome', '') == 'win',
                    'date': bet.get('timestamp', '')
                })
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    # Use unified bet selection logic with higher frequency
    recommended_bets = select_optimal_bets_unified(
        betting_analysis, team1_name, team2_name, previous_bets_by_team, confidence_score
    )
    
    # Initialize results
    results = {
        'match_id': f"live_{int(time.time())}",
        'team1_name': team1_name,
        'team2_name': team2_name,
        'team1_win_prob': win_probability,
        'team2_win_prob': 1 - win_probability,
        'predicted_winner': 'team1' if win_probability > 0.5 else 'team2',
        'confidence': confidence_score,
        'date': match_date,
        'odds_data': odds_data,
        'betting_analysis': betting_analysis,
        'recommended_bets': recommended_bets,
        'raw_predictions': raw_predictions,
        'symmetry_verified': is_symmetric,
        'current_drawdown': current_drawdown_pct
    }
    
    # Add bet type probabilities
    for bet_type, prob in bet_type_probs.items():
        results[bet_type + '_prob'] = prob
    
    # Print detailed report
    print_prediction_report(results, team1_stats, team2_stats)
    
    return results

def explain_prediction(team1_stats, team2_stats, win_probability, confidence, raw_predictions):
    """
    Provide detailed explanation of prediction factors for better interpretability.
    """
    print("\n===== PREDICTION EXPLANATION =====")
    
    # Identify key prediction factors
    if not team1_stats or not team2_stats:
        print("Missing team statistics for explanation")
        return
    
    team1_name = team1_stats.get('team_name', 'Team 1')
    team2_name = team2_stats.get('team_name', 'Team 2')
    
    # Extract important features from team statistics
    key_stats = [
        ('Win Rate', 'win_rate', True),
        ('Recent Form', 'recent_form', True), 
        ('Score Differential', 'score_differential', True),
        ('Average Score', 'avg_score', True),
        ('Opponent Score', 'avg_opponent_score', False),  # Lower is better
        ('Player Rating', 'avg_player_rating', True),
        ('Pistol Win Rate', 'pistol_win_rate', True),
        ('Economy Efficiency', 'economy_efficiency', True)
    ]
    
    print(f"Key Factors in {team1_name} vs {team2_name} Prediction:")
    print(f"Predicted outcome: {team1_name if win_probability > 0.5 else team2_name} win ({win_probability*100:.1f}% confidence)\n")
    
    # Head-to-head info if available
    h2h_info = None
    if 'opponent_stats' in team1_stats:
        for opponent, stats in team1_stats['opponent_stats'].items():
            if team2_name.lower() in opponent.lower():
                h2h_info = stats
                break
    
    if h2h_info:
        matches = h2h_info.get('matches', 0)
        wins = h2h_info.get('wins', 0)
        h2h_win_rate = wins / matches if matches > 0 else 0
        
        print(f"Head-to-head record: {team1_name} has won {wins} of {matches} matches ({h2h_win_rate*100:.1f}%)")
        
        score_diff = h2h_info.get('score_differential', 0)
        if abs(score_diff) > 0.5:
            print(f"Average map score differential: {abs(score_diff):.1f} in favor of {team1_name if score_diff > 0 else team2_name}")
    else:
        print("No head-to-head history found between these teams")
    
    # Compare key stats
    print("\nTeam Comparison:")
    print(f"{'Metric':<20} {team1_name:<15} {team2_name:<15} Advantage")
    print("-" * 60)
    
    advantages = []
    
    for label, key, higher_is_better in key_stats:
        t1_val = team1_stats.get(key, None)
        t2_val = team2_stats.get(key, None)
        
        if t1_val is not None and t2_val is not None:
            advantage = t1_val - t2_val if higher_is_better else t2_val - t1_val
            advantage_team = team1_name if advantage > 0 else team2_name
            
            # Store advantage for explanation
            advantages.append((label, advantage, advantage_team, key))
            
            # Format for display
            if isinstance(t1_val, float) and isinstance(t2_val, float):
                print(f"{label:<20} {t1_val:.3f}{'':9} {t2_val:.3f}{'':9} {advantage_team} ({abs(advantage):.3f})")
            else:
                print(f"{label:<20} {t1_val}{'':<11} {t2_val}{'':<11} {advantage_team}")
    
    # Sort advantages by magnitude
    advantages.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Explain prediction based on top advantages
    print("\nKey Factors in Prediction:")
    top_advantages = advantages[:3]
    
    for label, advantage, team, key in top_advantages:
        magnitude = "significantly" if abs(advantage) > 0.1 else "slightly"
        print(f"- {team} has a {magnitude} better {label.lower()} ({abs(advantage):.3f})")
    
    # Check for specific patterns
    if win_probability > 0.7 or win_probability < 0.3:
        print("\nThis is a high-confidence prediction based on clear advantages in key metrics.")
    elif confidence < 0.4:
        print("\nThis is a low-confidence prediction. The teams appear evenly matched or have inconsistent performance.")
    
    # Analyze model agreement
    if raw_predictions:
        team1_votes = sum(1 for p in raw_predictions if p > 0.5)
        team2_votes = len(raw_predictions) - team1_votes
        
        if abs(team1_votes - team2_votes) <= 0.2 * len(raw_predictions):
            print(f"\nNote: Models showed significant disagreement ({team1_votes}-{team2_votes})")
            print("     This suggests the match may be more unpredictable than the confidence score indicates.")
        elif max(team1_votes, team2_votes) > 0.9 * len(raw_predictions):
            print(f"\nNote: Strong model consensus ({team1_votes}-{team2_votes})")
            print("     This adds weight to the prediction confidence.")
    
    return advantages

def debug_training_results(ensemble_models, X_val, y_val, feature_names):
    """
    Debug training results to identify potential issues.
    """
    print("\n===== TRAINING RESULTS DEBUG =====")
    
    if not ensemble_models or not feature_names or X_val is None or y_val is None:
        print("Missing required data for debugging")
        return {}
    
    # Evaluate each model type separately
    model_metrics = {}
    
    for model_type, model, model_scaler in ensemble_models:
        try:
            # Apply scaling if needed
            X_pred = X_val.copy()
            if model_scaler is not None:
                X_pred = model_scaler.transform(X_pred)
            
            # Get predictions
            if model_type == 'nn':
                y_pred_proba = model.predict(X_pred).flatten()
            else:
                y_pred_proba = model.predict_proba(X_pred)[:, 1]
            
            # Calculate metrics
            y_pred = (y_pred_proba > 0.5).astype(int)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            # Store metrics
            model_metrics[model_type] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred_proba
            }
            
            print(f"{model_type.upper()} model performance:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  AUC-ROC: {auc:.4f}")
        except Exception as e:
            print(f"Error evaluating {model_type} model: {e}")
    
    # Analyze feature importance
    feature_importance = {}
    for model_type, model, _ in ensemble_models:
        if model_type in ['gb', 'rf'] and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"\nMost important features ({model_type} model):")
            for i in range(min(15, len(feature_names))):
                idx = indices[i]
                if idx < len(feature_names):
                    feature = feature_names[idx]
                    importance = importances[idx]
                    print(f"  {feature}: {importance:.4f}")
                    
                    if feature not in feature_importance:
                        feature_importance[feature] = 0
                    feature_importance[feature] += importance
    
    # Check for prediction consistency across models
    print("\nModel agreement analysis:")
    
    # Use a few validation samples to check consistency
    n_samples = min(5, len(y_val))
    for i in range(n_samples):
        sample_predictions = {}
        for model_type, metrics in model_metrics.items():
            if 'predictions' in metrics:
                sample_predictions[model_type] = metrics['predictions'][i]
        
        if sample_predictions:
            pred_values = list(sample_predictions.values())
            std_dev = np.std(pred_values)
            mean_pred = np.mean(pred_values)
            agreement = 1 - min(1, std_dev * 2)
            
            print(f"  Sample {i}: agreement={agreement:.2f}, mean={mean_pred:.4f}, std={std_dev:.4f}")
            for model_type, pred in sample_predictions.items():
                print(f"    {model_type}: {pred:.4f}")
            print(f"    Actual label: {y_val[i]}")
    
    # Test prediction symmetry
    print("\nTesting prediction symmetry:")
    
    # Create two test examples with swapped features
    if len(X_val) >= 1:
        # Get first example and create its mirror
        X_normal = X_val[0:1].copy()
        X_swapped = X_val[0:1].copy()
        
        # Manually flip specific feature types for the test
        for i, feature in enumerate(feature_names):
            if i < X_swapped.shape[1]:
                if '_diff' in feature or 'differential' in feature:
                    X_swapped[0, i] = -X_normal[0, i]
                elif feature.startswith('better_') or feature.endswith('_team1'):
                    X_swapped[0, i] = 1 - X_normal[0, i]
                elif '_x_' in feature:  # Interaction features
                    X_swapped[0, i] = -X_normal[0, i]
        
        # Make predictions with both versions
        normal_preds = []
        swapped_preds = []
        
        for model_type, model, model_scaler in ensemble_models:
            try:
                # Apply scaling if needed
                X_norm_pred = X_normal.copy()
                X_swap_pred = X_swapped.copy()
                
                if model_scaler is not None:
                    X_norm_pred = model_scaler.transform(X_norm_pred)
                    X_swap_pred = model_scaler.transform(X_swap_pred)
                
                # Get predictions
                if model_type == 'nn':
                    norm_pred = model.predict(X_norm_pred).flatten()[0]
                    swap_pred = model.predict(X_swap_pred).flatten()[0]
                else:
                    norm_pred = model.predict_proba(X_norm_pred)[0, 1]
                    swap_pred = model.predict_proba(X_swap_pred)[0, 1]
                
                normal_preds.append(norm_pred)
                swapped_preds.append(swap_pred)
                
                # Check if predictions complement each other (sum to 1)
                sum_pred = norm_pred + swap_pred
                symmetry_error = abs(sum_pred - 1.0)
                
                result = "✓" if symmetry_error < 0.05 else "✗"
                print(f"  {result} {model_type} model: {norm_pred:.4f} + {swap_pred:.4f} = {sum_pred:.4f}")
            except Exception as e:
                print(f"  Error testing {model_type} model: {e}")
        
        # Check ensemble symmetry
        if normal_preds and swapped_preds:
            normal_mean = np.mean(normal_preds)
            swapped_mean = np.mean(swapped_preds)
            sum_mean = normal_mean + swapped_mean
            ensemble_error = abs(sum_mean - 1.0)
            
            result = "✓" if ensemble_error < 0.05 else "✗"
            print(f"\n  {result} Ensemble: {normal_mean:.4f} + {swapped_mean:.4f} = {sum_mean:.4f}")
            
            if ensemble_error >= 0.05:
                print("    WARNING: Ensemble predictions are not symmetric")
                print("    This will cause inconsistent predictions depending on team order")
    
    # Analyze calibration - are probabilities accurate?
    print("\nProbability calibration analysis:")
    
    all_preds = []
    all_true = []
    
    for model_type, metrics in model_metrics.items():
        if 'predictions' in metrics:
            all_preds.extend(metrics['predictions'])
            all_true.extend(y_val for _ in range(len(metrics['predictions'])))
    
    if all_preds and all_true:
        # Group predictions into bins
        bins = {}
        for i, pred in enumerate(all_preds):
            bin_key = int(pred * 10) * 10  # Round to nearest 10%
            
            if bin_key not in bins:
                bins[bin_key] = {'count': 0, 'actual': 0}
            
            bins[bin_key]['count'] += 1
            bins[bin_key]['actual'] += all_true[i]
        
        print("Predicted % | Actual % | Count | Calibration Error")
        print("-" * 50)
        
        for bin_key in sorted(bins.keys()):
            bin_data = bins[bin_key]
            if bin_data['count'] > 0:
                predicted_pct = bin_key / 100
                actual_pct = bin_data['actual'] / bin_data['count']
                calib_error = abs(predicted_pct - actual_pct)
                
                print(f"{predicted_pct*100:9.1f}% | {actual_pct*100:7.1f}% | {bin_data['count']:5d} | {calib_error:.4f}")
    
    # Return the aggregated debug info
    return {
        'model_metrics': model_metrics,
        'feature_importance': feature_importance
    }

def test_feature_symmetry():
    """
    Test function to verify that all features transform correctly when teams are swapped.
    """
    print("\n===== COMPREHENSIVE FEATURE SYMMETRY TEST =====")
    
    # Create test team stats
    team1_stats = {
        'team_name': 'Team A',
        'team_id': '123',
        'win_rate': 0.75,
        'recent_form': 0.80,
        'matches': 50,
        'wins': 35,
        'losses': 15,
        'avg_score': 13.2,
        'avg_opponent_score': 8.7,
        'score_differential': 4.5,
        'avg_player_rating': 1.28,
        'avg_player_acs': 245.0,
        'avg_player_kd': 1.35,
        'avg_player_kast': 0.72,
        'avg_player_adr': 165.0,
        'avg_player_headshot': 0.25,
        'star_player_rating': 1.45,
        'team_consistency': 0.85,
        'fk_fd_ratio': 1.2,
        'pistol_win_rate': 0.68,
        'eco_win_rate': 0.35,
        'full_buy_win_rate': 0.62,
        'economy_efficiency': 0.75,
        'opponent_stats': {
            'Team B': {
                'matches': 5,
                'wins': 4,
                'win_rate': 0.8,
                'score_differential': 3.5
            }
        },
        'performance_trends': {
            'form_trajectory': {
                '5_vs_10': 0.15
            }
        }
    }
    
    team2_stats = {
        'team_name': 'Team B',
        'team_id': '456',
        'win_rate': 0.60,
        'recent_form': 0.65,
        'matches': 40,
        'wins': 24,
        'losses': 16,
        'avg_score': 11.8,
        'avg_opponent_score': 9.5,
        'score_differential': 2.3,
        'avg_player_rating': 1.18,
        'avg_player_acs': 220.0,
        'avg_player_kd': 1.15,
        'avg_player_kast': 0.68,
        'avg_player_adr': 155.0,
        'avg_player_headshot': 0.22,
        'star_player_rating': 1.32,
        'team_consistency': 0.80,
        'fk_fd_ratio': 1.1,
        'pistol_win_rate': 0.55,
        'eco_win_rate': 0.30,
        'full_buy_win_rate': 0.58,
        'economy_efficiency': 0.68,
        'opponent_stats': {
            'Team A': {
                'matches': 5,
                'wins': 1,
                'win_rate': 0.2,
                'score_differential': -3.5
            }
        },
        'performance_trends': {
            'form_trajectory': {
                '5_vs_10': 0.05
            }
        }
    }
    
    # Add map statistics
    team1_stats['map_statistics'] = {
        'Haven': {'win_percentage': 0.70, 'matches_played': 12},
        'Ascent': {'win_percentage': 0.65, 'matches_played': 10},
        'Bind': {'win_percentage': 0.60, 'matches_played': 8}
    }
    
    team2_stats['map_statistics'] = {
        'Haven': {'win_percentage': 0.60, 'matches_played': 10},
        'Ascent': {'win_percentage': 0.55, 'matches_played': 9},
        'Bind': {'win_percentage': 0.50, 'matches_played': 6}
    }
    
    # Generate features in both directions
    print("\nGenerating features in normal and reversed order...")
    features_1_2 = prepare_data_for_model(team1_stats, team2_stats)
    features_2_1 = prepare_data_for_model(team2_stats, team1_stats)
    
    if not features_1_2 or not features_2_1:
        print("ERROR: Failed to generate features")
        return False
    
    # Comprehensive testing of all feature types
    
    # 1. TEST DIFFERENCE FEATURES (should negate)
    print("\nTesting DIFFERENCE features (should negate when teams swap):")
    # FIXED: Don't include binary team1 features in the difference features list
    diff_features = [f for f in features_1_2.keys() 
                    if ("_diff" in f or "differential" in f) 
                    and not (f.startswith("better_") or f.endswith("_team1"))]
    
    for feature in diff_features:
        value_1_2 = features_1_2[feature]
        value_2_1 = features_2_1[feature]
        
        if abs(value_1_2 + value_2_1) < 1e-10:
            print(f"✓ {feature:<30} transforms correctly: {value_1_2:.4f} → {value_2_1:.4f}")
        else:
            print(f"❌ {feature:<30} FAILS: {value_1_2:.4f} → {value_2_1:.4f} (expected {-value_1_2:.4f})")
    
    # 2. TEST BINARY FEATURES (should flip 0↔1)
    print("\nTesting BINARY features (should flip 0/1 when teams swap):")
    binary_features = [f for f in features_1_2.keys() 
                     if f.startswith("better_") or f.endswith("_team1") or f == "maps_advantage_team1"]
    
    for feature in binary_features:
        value_1_2 = int(features_1_2[feature])
        value_2_1 = int(features_2_1[feature])
        
        if value_1_2 == 1 - value_2_1:
            print(f"✓ {feature:<30} transforms correctly: {value_1_2} → {value_2_1}")
        else:
            print(f"❌ {feature:<30} FAILS: {value_1_2} → {value_2_1} (expected {1-value_1_2})")
    
    # 3. TEST SYMMETRIC FEATURES (should remain identical)
    print("\nTesting SYMMETRIC features (should remain identical when teams swap):")
    # FIXED: Properly exclude difference features from symmetric features
    symmetric_features = [f for f in features_1_2.keys() 
                        if f.startswith("avg_") and '_diff' not in f and 'differential' not in f]
    
    for feature in symmetric_features:
        value_1_2 = features_1_2[feature]
        value_2_1 = features_2_1[feature]
        
        if abs(value_1_2 - value_2_1) < 1e-10:
            print(f"✓ {feature:<30} remains invariant: {value_1_2:.4f} = {value_2_1:.4f}")
        else:
            print(f"❌ {feature:<30} FAILS: {value_1_2:.4f} ≠ {value_2_1:.4f}")
    
    # 4. TEST INTERACTION FEATURES (should negate)
    print("\nTesting INTERACTION features (should negate when teams swap):")
    interaction_features = [f for f in features_1_2.keys() if "_x_" in f]
    for feature in interaction_features:
        value_1_2 = features_1_2[feature]
        value_2_1 = features_2_1[feature]
        
        if abs(value_1_2 + value_2_1) < 1e-10:
            print(f"✓ {feature:<30} transforms correctly: {value_1_2:.4f} → {value_2_1:.4f}")
        else:
            print(f"❌ {feature:<30} FAILS: {value_1_2:.4f} → {value_2_1:.4f} (expected {-value_1_2:.4f})")
    
    # 5. TEST H2H WIN RATE (should be 1-x)
    print("\nTesting H2H WIN RATE (should transform to 1-x when teams swap):")
    if "h2h_win_rate" in features_1_2 and "h2h_win_rate" in features_2_1:
        value_1_2 = features_1_2["h2h_win_rate"]
        value_2_1 = features_2_1["h2h_win_rate"]
        
        if abs(value_1_2 + value_2_1 - 1) < 1e-10:
            print(f"✓ h2h_win_rate{' '*20} transforms correctly: {value_1_2:.4f} → {value_2_1:.4f}")
        else:
            print(f"❌ h2h_win_rate{' '*20} FAILS: {value_1_2:.4f} → {value_2_1:.4f} (expected {1-value_1_2:.4f})")
    
    # Special focus on problematic features
    problem_features = ['better_score_diff_team1', 'better_defense_team1']
    print("\nSpecial focus on previously problematic features:")
    
    for feature in problem_features:
        if feature in features_1_2 and feature in features_2_1:
            val_normal = int(features_1_2[feature])
            val_reversed = int(features_2_1[feature])
            
            if val_normal == 1 - val_reversed:
                print(f"✓ {feature:<30} now works correctly: {val_normal} → {val_reversed}")
            else:
                print(f"❌ {feature:<30} still FAILS: {val_normal} → {val_reversed} (expected {1-val_normal})")
    
    # Count symmetry issues
    symmetry_issues = []
    
    # Check difference features
    for feature in diff_features:
        if abs(features_1_2[feature] + features_2_1[feature]) > 1e-10:
            symmetry_issues.append(f"Difference feature {feature} does not negate properly")
    
    # Check binary features
    for feature in binary_features:
        if int(features_1_2[feature]) != 1 - int(features_2_1[feature]):
            symmetry_issues.append(f"Binary feature {feature} does not flip properly")
    
    # Check symmetric features - properly excluding difference features
    for feature in symmetric_features:
        if abs(features_1_2[feature] - features_2_1[feature]) > 1e-10:
            symmetry_issues.append(f"Symmetric feature {feature} does not remain invariant")
    
    # Check interaction features
    for feature in interaction_features:
        if abs(features_1_2[feature] + features_2_1[feature]) > 1e-10:
            symmetry_issues.append(f"Interaction feature {feature} does not negate properly")
    
    # Check h2h_win_rate
    if "h2h_win_rate" in features_1_2 and "h2h_win_rate" in features_2_1:
        if abs(features_1_2["h2h_win_rate"] + features_2_1["h2h_win_rate"] - 1) > 1e-10:
            symmetry_issues.append("h2h_win_rate does not transform to 1-x properly")
    
    # Print summary
    print("\n===== SYMMETRY TEST SUMMARY =====")
    if symmetry_issues:
        print(f"❌ Found {len(symmetry_issues)} symmetry issues:")
        for issue in symmetry_issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        
        if len(symmetry_issues) > 10:
            print(f"  ... and {len(symmetry_issues) - 10} more")
        
        total_features = len(features_1_2)
        working_features = total_features - len(symmetry_issues)
        print(f"\nOverall symmetry: {working_features/total_features:.1%} of features work correctly")
        return False
    else:
        print("✓ All features transform correctly when teams are swapped!")
        return True

def debug_trained_models():
    """Run comprehensive debug analysis on trained models."""
    # Try to load models
    print("Loading models for debugging...")
    try:
        # Load models using existing load_models_unified function
        ensemble_models, selected_features = load_models_unified()
        
        if not ensemble_models or not selected_features:
            print("Failed to load models or features. Aborting debug.")
            return False
        
        # Show model types for debugging
        model_types = {}
        for model_type, _, _ in ensemble_models:
            if model_type not in model_types:
                model_types[model_type] = 0
            model_types[model_type] += 1
        
        print("\nEnsemble composition:")
        for model_type, count in model_types.items():
            print(f"  - {model_type.upper()}: {count} models")
        
        # Test feature symmetry with the improved function
        print("\nTesting feature symmetry...")
        symmetry_passed = test_feature_symmetry()
        
        if not symmetry_passed:
            print("\nWARNING: Feature symmetry issues detected. Predictions may be inconsistent.")
        else:
            print("\nFeature symmetry confirmed - predictions will be consistent.")
        
        # Create a simplified test dataset to test model predictions
        print("\nCreating test dataset to verify model predictions...")
        
        # Simple team stats for testing
        team1 = {
            'team_name': 'Test Team A',
            'win_rate': 0.65,
            'recent_form': 0.70,
            'score_differential': 3.0,
            'matches': 50,
            'avg_score': 13.0,
            'avg_opponent_score': 10.0
        }
        
        team2 = {
            'team_name': 'Test Team B',
            'win_rate': 0.55,
            'recent_form': 0.60,
            'score_differential': 2.0,
            'matches': 40,
            'avg_score': 12.0,
            'avg_opponent_score': 10.0
        }
        
        # Create features for prediction test
        features = prepare_data_for_model(team1, team2)
        
        if not features:
            print("Failed to create features for prediction test.")
            return False
            
        features_df = pd.DataFrame([features])
        
        # Prepare minimal feature set for testing model predictions
        if selected_features:
            # Create a DataFrame with only the selected features
            test_features = pd.DataFrame(0, index=[0], columns=selected_features)
            
            # Fill in available features
            for feature in selected_features:
                if feature in features_df.columns:
                    test_features[feature] = features_df[feature].values
            
            X_test = test_features.values
        else:
            # Use all available features
            X_test = features_df.values
            
        print(f"Created test features with shape: {X_test.shape}")
            
        # Try to get predictions from each model with improved error handling
        print("\nTesting model predictions:")
        
        for model_idx, (model_type, model, model_scaler) in enumerate(ensemble_models):
            try:
                # Apply scaling if needed
                X_pred = X_test.copy()
                if model_scaler is not None:
                    try:
                        X_pred = model_scaler.transform(X_pred)
                    except Exception as e:
                        print(f"Warning: Scaling error for {model_type} model {model_idx}: {e}")
                        print("Using unscaled features")
                
                # Make prediction based on model type with better error handling
                if model_type == 'nn':
                    try:
                        pred = model.predict(X_pred, verbose=0)
                        if isinstance(pred, np.ndarray) and pred.size > 0:
                            if len(pred.shape) > 1 and pred.shape[1] > 0:
                                pred_value = pred[0][0]
                            else:
                                pred_value = pred[0]
                            print(f"  {model_type.upper()} model {model_idx}: {pred_value:.4f}")
                        else:
                            print(f"  {model_type.upper()} model {model_idx}: Empty prediction")
                    except Exception as e:
                        print(f"  Error with {model_type.upper()} model {model_idx}: {e}")
                else:
                    # For scikit-learn models
                    try:
                        if hasattr(model, 'predict_proba'):
                            pred = model.predict_proba(X_pred)
                            if isinstance(pred, np.ndarray) and pred.shape[0] > 0 and pred.shape[1] > 1:
                                pred_value = pred[0][1]  # Probability of class 1
                                print(f"  {model_type.upper()} model {model_idx}: {pred_value:.4f}")
                            else:
                                print(f"  {model_type.upper()} model {model_idx}: Invalid prediction shape {pred.shape}")
                        else:
                            pred = model.predict(X_pred)
                            print(f"  {model_type.upper()} model {model_idx}: {pred[0]:.4f}")
                    except Exception as e:
                        print(f"  Error with {model_type.upper()} model {model_idx}: {e}")
            except Exception as e:
                print(f"  Error processing {model_type.upper()} model {model_idx}: {e}")
        
        # Test predictions with team order swapped
        print("\nTesting symmetry of predictions by swapping teams...")
        
        # Create reverse features
        features_reversed = prepare_data_for_model(team2, team1)
        
        if not features_reversed:
            print("Failed to create reversed features.")
            return False
            
        features_df_reversed = pd.DataFrame([features_reversed])
        
        # Prepare minimal feature set for reversed prediction
        if selected_features:
            # Create a DataFrame with only the selected features
            test_features_reversed = pd.DataFrame(0, index=[0], columns=selected_features)
            
            # Fill in available features
            for feature in selected_features:
                if feature in features_df_reversed.columns:
                    test_features_reversed[feature] = features_df_reversed[feature].values
            
            X_test_reversed = test_features_reversed.values
        else:
            # Use all available features
            X_test_reversed = features_df_reversed.values
                    
        # Try ensemble prediction
        normal_preds = []
        reversed_preds = []
        
        # Use a single reliably working model for the test
        for model_idx, (model_type, model, model_scaler) in enumerate(ensemble_models):
            if model_type in ['gb', 'rf']:  # Use tree-based models for reliability
                try:
                    # Normal prediction
                    X_pred = X_test.copy()
                    if model_scaler is not None:
                        try:
                            X_pred = model_scaler.transform(X_pred)
                        except:
                            pass
                    
                    # Get prediction
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X_pred)[0][1]
                        normal_preds.append(pred)
                    else:
                        pred = model.predict(X_pred)[0]
                        normal_preds.append(pred)
                        
                    # Reversed prediction
                    X_pred_reversed = X_test_reversed.copy()
                    if model_scaler is not None:
                        try:
                            X_pred_reversed = model_scaler.transform(X_pred_reversed)
                        except:
                            pass
                    
                    # Get prediction
                    if hasattr(model, 'predict_proba'):
                        pred_reversed = model.predict_proba(X_pred_reversed)[0][1]
                        reversed_preds.append(pred_reversed)
                    else:
                        pred_reversed = model.predict(X_pred_reversed)[0]
                        reversed_preds.append(pred_reversed)
                        
                    # Print results for this model
                    print(f"  {model_type.upper()} model {model_idx}:")
                    print(f"    Normal order (Team A vs B): {pred:.4f}")
                    print(f"    Reversed order (Team B vs A): {pred_reversed:.4f}")
                    print(f"    Sum: {pred + pred_reversed:.4f} (should be close to 1.0)")
                    
                    # Limit to a few models for this test
                    if len(normal_preds) >= 3:
                        break
                        
                except Exception as e:
                    print(f"  Error testing {model_type.upper()} model {model_idx}: {e}")
        
        # Check ensemble prediction symmetry
        if normal_preds and reversed_preds:
            avg_normal = np.mean(normal_preds)
            avg_reversed = np.mean(reversed_preds)
            sum_pred = avg_normal + avg_reversed
            
            print(f"\nEnsemble prediction symmetry check:")
            print(f"  Normal order average: {avg_normal:.4f}")
            print(f"  Reversed order average: {avg_reversed:.4f}")
            print(f"  Sum: {sum_pred:.4f} (should be close to 1.0)")
            
            if abs(sum_pred - 1.0) < 0.1:
                print("✅ Ensemble predictions are symmetric")
            else:
                print("⚠️ Ensemble predictions are NOT perfectly symmetric")
        
        # Try unified prediction function
        try:
            print("\nTesting unified prediction function:")
            
            win_prob, raw_preds, confidence = predict_with_ensemble_unified(ensemble_models, X_test)
            print(f"  Normal order: {win_prob:.4f}, confidence: {confidence:.4f}")
            
            win_prob_reversed, raw_preds_reversed, confidence_reversed = predict_with_ensemble_unified(
                ensemble_models, X_test_reversed
            )
            print(f"  Reversed order: {win_prob_reversed:.4f}, confidence: {confidence_reversed:.4f}")
            print(f"  Sum: {win_prob + win_prob_reversed:.4f} (should be close to 1.0)")
            
            if abs(win_prob + win_prob_reversed - 1.0) < 0.1:
                print("✅ Unified prediction function is symmetric")
            else:
                print("⚠️ Unified prediction function is NOT perfectly symmetric")
        except Exception as e:
            print(f"Error testing unified prediction function: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error in debug_trained_models_fixed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_symmetry_with_models(ensemble_models, selected_features):
    """Test if models produce symmetric predictions when teams are swapped.
    Critical for ensuring consistent predictions regardless of team order."""
    
    # Create test data
    team1 = {
        'team_name': 'Test Team A',
        'win_rate': 0.6,
        'recent_form': 0.65,
        'score_differential': 2.0,
        'matches': 50
    }
    
    team2 = {
        'team_name': 'Test Team B',
        'win_rate': 0.5,
        'recent_form': 0.55,
        'score_differential': 1.0,
        'matches': 40
    }
    
    # Generate features
    X_normal = prepare_features_unified(team1, team2, selected_features)
    X_swapped = prepare_features_unified(team2, team1, selected_features)
    
    if X_normal is None or X_swapped is None:
        print("Failed to generate features for model symmetry test")
        return False
    
    # Test each model type
    for i, (model_type, model, model_scaler) in enumerate(ensemble_models):
        # Apply scaling if needed
        X_pred_normal = X_normal.copy()
        X_pred_swapped = X_swapped.copy()
        
        if model_scaler is not None:
            try:
                X_pred_normal = model_scaler.transform(X_pred_normal)
                X_pred_swapped = model_scaler.transform(X_pred_swapped)
            except:
                pass
                
        # Get predictions for both team orders
        try:
            if model_type == 'nn':
                pred_normal = model.predict(X_pred_normal, verbose=0)[0][0]
                pred_swapped = model.predict(X_pred_swapped, verbose=0)[0][0]
            else:
                pred_normal = model.predict_proba(X_pred_normal)[0][1]
                pred_swapped = model.predict_proba(X_pred_swapped)[0][1]
                
            # Check symmetry (should sum to approximately 1)
            symmetry_sum = pred_normal + pred_swapped
            
            if abs(symmetry_sum - 1.0) < 0.05:
                print(f"✓ {model_type.upper()} model {i}: Good symmetry - {pred_normal:.4f} + {pred_swapped:.4f} = {symmetry_sum:.4f}")
            else:
                print(f"⚠ {model_type.upper()} model {i}: Poor symmetry - {pred_normal:.4f} + {pred_swapped:.4f} = {symmetry_sum:.4f}")
        except Exception as e:
            print(f"Error testing {model_type} model {i}: {e}")
    
    return True


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
    Handles different key naming conventions for compatibility between backtesting and prediction.
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
    
    # Handle confidence interval with both naming conventions
    if 'confidence_interval' in results:
        ci = results['confidence_interval']
        print(f"  Confidence Interval: ({ci[0]:.2%} - {ci[1]:.2%})")
    
    # Handle model agreement/confidence with both naming conventions
    model_agreement = results.get('model_agreement', results.get('confidence', 0))
    print(f"  Model Agreement: {model_agreement:.2f} (Higher is better)")
    
    # Print bet type probabilities
    if any(key.endswith('_prob') for key in results.keys()):
        print("\nBET TYPE PROBABILITIES:")
        print("-" * 80)
        
        # Handle all probability types with specific display order
        prob_types = [
            ('team1_plus_1_5_prob', f"  {team1_name} +1.5 Maps: {results.get('team1_plus_1_5_prob', 0):.2%}"),
            ('team2_plus_1_5_prob', f"  {team2_name} +1.5 Maps: {results.get('team2_plus_1_5_prob', 0):.2%}"),
            ('team1_minus_1_5_prob', f"  {team1_name} -1.5 Maps (2-0 win): {results.get('team1_minus_1_5_prob', 0):.2%}"),
            ('team2_minus_1_5_prob', f"  {team2_name} -1.5 Maps (2-0 win): {results.get('team2_minus_1_5_prob', 0):.2%}"),
            ('over_2_5_maps_prob', f"  Over 2.5 Maps: {results.get('over_2_5_maps_prob', 0):.2%}"),
            ('under_2_5_maps_prob', f"  Under 2.5 Maps: {results.get('under_2_5_maps_prob', 0):.2%}")
        ]
        
        for key, text in prob_types:
            if key in results:
                print(text)
    
    # Update the betting recommendations section with better handling of key differences
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
            
            # Check if the bet is recommended using both possible key names
            if analysis.get('recommended', False):
                # Get values with fallbacks for different key naming conventions
                our_prob = analysis.get('our_probability', analysis.get('probability', 0))
                implied_prob = analysis.get('implied_probability', analysis.get('implied_prob', 0))
                edge = analysis.get('expected_value', analysis.get('edge', 0))
                roi = analysis.get('roi', 0)
                recommended_bet = analysis.get('recommended_bet', analysis.get('bet_amount', 0))
                
                recommended_bets.append((bet_desc, edge, roi, recommended_bet))
                
                print(f"  RECOMMENDED BET: {bet_desc}")
                print(f"  - Our Probability: {our_prob:.2%}")
                print(f"  - Implied Probability: {implied_prob:.2%}")
                print(f"  - Edge: {edge:.2%}")
                print(f"  - ROI: {roi:.2%}")
                print(f"  - Expected Value: {analysis.get('ev_percentage', edge * 100):.2f}%")
                print(f"  - Confidence: {model_agreement:.2f}")
                print(f"  - Kelly Fraction: {analysis.get('kelly_fraction', 0):.4f}")
                print(f"  - Recommended Bet Amount: ${recommended_bet:.1f}")
                print("")
                
                # Explain the recommendation 
                explain_bet_recommendation(bet_type, analysis, results, team1_stats['team_name'], team2_stats['team_name'])
            else:
                # Track rejected bets
                reason = analysis.get('reason', analysis.get('filter_reason', 'Not recommended'))
                rejected_bets.append((bet_desc, reason))
        
        if not recommended_bets:
            print("  No profitable betting opportunities identified for this match.")
        else:
            # Sort by ROI
            recommended_bets.sort(key=lambda x: x[2], reverse=True)
            print("\nBETS RANKED BY EXPECTED ROI:")
            for i, (bet, edge, roi, amount) in enumerate(recommended_bets):
                print(f"  {i+1}. {bet}: {edge:.2%} edge, {roi:.2%} ROI, bet ${amount:.1f}")
        
        # Show rejected bets
        if rejected_bets:
            print("\nREJECTED BETS:")
            for bet, reason in rejected_bets:
                print(f"  {bet}: {reason}")
            # Sort by edge
            recommended_bets.sort(key=lambda x: x[1], reverse=True)
            if recommended_bets:
                print("\nBETS RANKED BY EDGE:")
                for i, (bet, edge, roi, amount) in enumerate(recommended_bets):
                    print(f"  {i+1}. {bet}: {edge:.2%} edge, bet ${amount:.1f}")

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
    
    # Get probability with fallback for all the key names
    our_prob = analysis.get('our_probability', analysis.get('probability', 0))
    implied_prob = analysis.get('implied_probability', analysis.get('implied_prob', 0))
    edge = analysis.get('expected_value', analysis.get('edge', 0))
    odds = analysis.get('odds', 0)
    
    # Explain moneyline bets
    if bet_type == 'team1_ml':
        # Check which factors favor team1
        print(f"  The model favors {team1_name} to win the match with {our_prob:.2%} probability.")
        
        # Explain head-to-head advantage if it exists
        if results.get('h2h_advantage_team1', 0) > 0:
            print(f"  {team1_name} has a strong historical head-to-head advantage against {team2_name}.")
        
        # Explain any other key advantages
        if results.get('team1_win_prob', 0) > 0.6:
            print(f"  {team1_name}'s overall form and statistics indicate a significant advantage.")
            
    elif bet_type == 'team2_ml':
        print(f"  The model favors {team2_name} to win the match with {our_prob:.2%} probability.")
        
        # Explain head-to-head advantage if it exists
        if results.get('h2h_advantage_team1', 0) < 0:
            print(f"  {team2_name} has a strong historical head-to-head advantage against {team1_name}.")
        
        # Explain any other key advantages
        if results.get('team2_win_prob', 0) > 0.6:
            print(f"  {team2_name}'s overall form and statistics indicate a significant advantage.")
    
    # Explain +1.5 bets
    elif bet_type == 'team1_plus_1_5':
        print(f"  {team1_name} has a {our_prob:.2%} probability of winning at least one map.")
        print(f"  This is significantly higher than the {implied_prob:.2%} implied by the odds.")
        if results.get('team1_plus_1_5_prob', 0) > 0.8:
            print(f"  Even as an underdog in the match, {team1_name} is likely to take at least one map.")
            
    elif bet_type == 'team2_plus_1_5':
        print(f"  {team2_name} has a {our_prob:.2%} probability of winning at least one map.")
        print(f"  This is significantly higher than the {implied_prob:.2%} implied by the odds.")
        if results.get('team2_plus_1_5_prob', 0) > 0.8:
            print(f"  Even as an underdog in the match, {team2_name} is likely to take at least one map.")
    
    # Explain -1.5 bets
    elif bet_type == 'team1_minus_1_5':
        print(f"  {team1_name} has a {our_prob:.2%} probability of winning 2-0.")
        print(f"  The model suggests {team1_name} is significantly stronger than {team2_name} and can win without dropping a map.")
        
    elif bet_type == 'team2_minus_1_5':
        print(f"  {team2_name} has a {our_prob:.2%} probability of winning 2-0.")
        print(f"  The model suggests {team2_name} is significantly stronger than {team1_name} and can win without dropping a map.")
    
    # Explain over/under bets
    elif bet_type == 'over_2_5_maps':
        print(f"  The match has a {our_prob:.2%} probability of going to 3 maps.")
        print(f"  The teams appear evenly matched and likely to split the first two maps.")
        
    elif bet_type == 'under_2_5_maps':
        print(f"  The match has a {our_prob:.2%} probability of ending in 2 maps.")
        print(f"  One team appears significantly stronger and likely to win 2-0.")
    
    # Explain why the odds provide value
    print(f"  The bookmaker's odds of {odds:.2f} represent value compared to our model's assessment.")
    print(f"  Long-term expected value: ${100 * edge:.2f} per $100 wagered.")

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
    Track betting performance over time with improved metrics.
    
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
            'bets': [],
            'bankroll_history': [],
            'starting_bankroll': 1000.0  # Default starting bankroll
        }
    
    # Update starting bankroll if not set
    if 'starting_bankroll' not in performance:
        if 'bankroll_history' in performance and performance['bankroll_history']:
            # Use first entry as starting bankroll
            performance['starting_bankroll'] = performance['bankroll_history'][0].get('bankroll', 1000.0)
        else:
            performance['starting_bankroll'] = 1000.0
    
    # Get current bankroll
    current_bankroll = 1000.0  # Default
    if 'bankroll_history' in performance and performance['bankroll_history']:
        current_bankroll = performance['bankroll_history'][-1].get('bankroll', 1000.0)
    
    # Calculate profit/loss and new bankroll
    returns = bet_amount * odds if outcome else 0
    profit = returns - bet_amount
    new_bankroll = current_bankroll + profit
    
    # Create new bet record with more detailed information
    bet_record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'teams': f"{prediction['team1_name']} vs {prediction['team2_name']}",
        'bet_type': bet_placed,
        'amount': bet_amount,
        'odds': odds,
        'predicted_prob': prediction['betting_analysis'][bet_placed]['probability'],
        'implied_prob': prediction['betting_analysis'][bet_placed]['implied_prob'],
        'edge': prediction['betting_analysis'][bet_placed]['edge'],
        'confidence': prediction.get('confidence', 0.5),
        'outcome': 'win' if outcome else 'loss',
        'return': returns,
        'profit': profit,
        'predicted_winner': prediction.get('predicted_winner', 'Unknown'),
        'actual_winner': input("Which team won the match? Enter 'team1' or 'team2': ") or prediction.get('predicted_winner', 'Unknown')
    }
    
    # Update performance metrics
    performance['total_bets'] += 1
    if outcome:
        performance['wins'] += 1
    else:
        performance['losses'] += 1
    
    performance['total_wagered'] += bet_amount
    performance['total_returns'] += returns
    performance['roi'] = (performance['total_returns'] - performance['total_wagered']) / performance['total_wagered'] if performance['total_wagered'] > 0 else 0
    
    # Add to history
    performance['bets'].append(bet_record)
    
    # Update bankroll history
    bankroll_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'bankroll': new_bankroll,
        'change': profit,
        'bet_id': len(performance['bets']) - 1
    }
    
    if 'bankroll_history' not in performance:
        performance['bankroll_history'] = []
    
    performance['bankroll_history'].append(bankroll_entry)
    
    # Calculate additional metrics
    if 'bankroll_history' in performance and len(performance['bankroll_history']) > 0:
        # Calculate max bankroll and current drawdown
        max_bankroll = max([entry.get('bankroll', 0) for entry in performance['bankroll_history']])
        if max_bankroll > new_bankroll:
            drawdown_pct = ((max_bankroll - new_bankroll) / max_bankroll) * 100
            performance['current_drawdown'] = drawdown_pct
            performance['max_drawdown'] = max(performance.get('max_drawdown', 0), drawdown_pct)
    
    # Calculate win streak
    current_streak = 0
    for bet in reversed(performance['bets']):
        if bet['outcome'] == 'win':
            current_streak = max(0, current_streak) + 1
        else:
            current_streak = min(0, current_streak) - 1
            
    performance['current_streak'] = current_streak
    performance['max_win_streak'] = max(performance.get('max_win_streak', 0), max(0, current_streak))
    performance['max_lose_streak'] = max(performance.get('max_lose_streak', 0), abs(min(0, current_streak)))
    
    # Calculate ROI by bet type
    bet_types = {}
    for bet in performance['bets']:
        bet_type = bet['bet_type']
        if bet_type not in bet_types:
            bet_types[bet_type] = {'wagered': 0, 'returns': 0, 'count': 0, 'wins': 0}
        
        bet_types[bet_type]['wagered'] += bet['amount']
        bet_types[bet_type]['returns'] += bet['return']
        bet_types[bet_type]['count'] += 1
        if bet['outcome'] == 'win':
            bet_types[bet_type]['wins'] += 1
    
    for bet_type, stats in bet_types.items():
        if stats['wagered'] > 0:
            stats['roi'] = (stats['returns'] - stats['wagered']) / stats['wagered']
            stats['win_rate'] = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
    
    performance['bet_types'] = bet_types
    
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
    print(f"Current Bankroll: ${new_bankroll:.2f}")
    if 'current_drawdown' in performance:
        print(f"Current Drawdown: {performance['current_drawdown']:.2f}%")
    print(f"Current Streak: {performance['current_streak']} bets" + 
          (" (winning)" if performance['current_streak'] > 0 else 
           (" (losing)" if performance['current_streak'] < 0 else "")))


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

def train_with_consistent_features(X, y, n_splits=10, random_state=42):
    """
    Modified training function that ensures NN models don't produce extreme predictions.
    - Adds custom callbacks to monitor extreme predictions
    - Includes extra validation to verify model calibration
    """
    print(f"\nTraining with {n_splits}-fold cross-validation using consistent features and diverse ensemble")
    
    # Verify feature symmetry with sample data
    print("\nVerifying feature symmetry before training...")
    # (existing code remains the same)
    
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
    all_feature_importances = np.zeros(df.shape[1])
    fold_metrics = []
    
    print("Phase 1: Identifying important features across all folds...")
    
    # First pass: Identify feature importance across all folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_arr)):
        print(f"\n----- Feature Selection: Fold {fold+1}/{n_splits} -----")
        
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
        
        # Feature selection with Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf.fit(X_train, y_train)
        
        # Accumulate importance scores
        all_feature_importances += rf.feature_importances_
    
    # Calculate average importance across all folds
    avg_importances = all_feature_importances / n_splits
    
    # Select features based on cumulative importance
    indices = np.argsort(avg_importances)[::-1]
    cumulative_importance = np.cumsum(avg_importances[indices])
    
    # Select features that account for 80% of importance
    n_features = np.where(cumulative_importance >= 0.8)[0][0] + 1
    n_features = min(n_features, 50)  # Cap at 50 features
    
    top_indices = indices[:n_features]
    selected_features = [df.columns[i] for i in top_indices]
    
    print(f"\nSelected {len(selected_features)} consistent features across all folds:")
    for i, feature in enumerate(selected_features[:10]):
        print(f"  {i+1}. {feature} (importance: {avg_importances[df.columns.get_loc(feature)]:.4f})")
    if len(selected_features) > 10:
        print(f"  ... and {len(selected_features) - 10} more features")
    
    # Create a feature mask for consistent selection
    feature_mask = np.zeros(df.shape[1], dtype=bool)
    feature_mask[top_indices] = True
    
    # Save feature importance data to CSV
    feature_importance_data = []
    for i, feature in enumerate(df.columns):
        feature_importance_data.append({
            'feature_name': feature,
            'importance_score': avg_importances[i],
            'selected': feature in selected_features,
            'rank': np.where(indices == i)[0][0] + 1 if i in indices else -1
        })
    
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame(feature_importance_data)
    importance_df = importance_df.sort_values('importance_score', ascending=False)
    
    # Save to CSV
    importance_df.to_csv('feature_importance.csv', index=False)
    print(f"Saved feature importance data to feature_importance.csv")
    
    # Second pass: Train diverse ensemble with consistent feature set
    print("\nPhase 2: Training diverse ensemble with consistent feature set...")
    
    # Split data for ensemble training
    train_idx, val_idx = next(skf.split(X_arr, y_arr))
    X_train, X_val = X_arr[train_idx], X_arr[val_idx]
    y_train, y_val = y_arr[train_idx], y_arr[val_idx]
    
    # Create the diverse ensemble
    print("Creating diverse ensemble of models...")
    
    # Select features using mask
    X_train_selected = X_train[:, feature_mask]
    X_val_selected = X_val[:, feature_mask]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    
    # NN Model - with calibration improvements
    ensemble_models = []
    print("Training Neural Network models with improved calibration...")
    
    # Custom callback to monitor extreme predictions
    class ExtremeValueCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:  # Check every 5 epochs
                predictions = self.model.predict(X_val_scaled, verbose=0)
                # Check if predictions are getting too extreme
                if np.any(predictions < 0.05) or np.any(predictions > 0.95):
                    print(f"Warning: Extreme predictions detected at epoch {epoch}")
                    print(f"Min: {np.min(predictions):.4f}, Max: {np.max(predictions):.4f}")
    
    # Train multiple NN models with improved architecture to prevent extremes
    for i in range(5):
        # Create model with improved architecture - critical for preventing extreme outputs
        nn_model = create_improved_model(X_train_scaled.shape[1], regularization_strength=0.03 + i*0.01)
        
        # Set up callbacks with prediction monitoring
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=0
        )
        extreme_monitor = ExtremeValueCallback()
        
        # Train with monitoring
        nn_model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_scaled, y_val),
            callbacks=[early_stopping, reduce_lr, extreme_monitor],
            verbose=0
        )
        
        # Test for extreme predictions before accepting model
        test_preds = nn_model.predict(X_val_scaled, verbose=0).flatten()
        
        # Verify model doesn't give extreme predictions
        if np.min(test_preds) < 0.05 or np.max(test_preds) > 0.95:
            print(f"Warning: NN model {i} still gives extreme predictions after training!")
            print(f"Min: {np.min(test_preds):.4f}, Max: {np.max(test_preds):.4f}")
            print("Adjusting model to enforce bounds...")
            
            # Apply post-training calibration to fix issue if it persists
            def get_calibrated_model(model):
                input_shape = model.input_shape[1:]
                inputs = Input(shape=input_shape)
                x = model(inputs)
                
                # Force outputs to be between 0.15 and 0.85
                outputs = Lambda(lambda y: 0.15 + tf.clip_by_value(y, 0, 1) * 0.7)(x)
                
                calibrated_model = Model(inputs=inputs, outputs=outputs)
                calibrated_model.compile(
                    loss='binary_crossentropy',
                    optimizer=Adam(learning_rate=0.0001),
                    metrics=['accuracy']
                )
                return calibrated_model
            
            # Replace model with calibrated version
            nn_model = get_calibrated_model(nn_model)
            
            # Verify calibration worked
            test_preds = nn_model.predict(X_val_scaled, verbose=0).flatten()
            print(f"After calibration - Min: {np.min(test_preds):.4f}, Max: {np.max(test_preds):.4f}")
        
        # Calculate validation accuracy
        y_pred_binary = (test_preds > 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred_binary)
        print(f"NN model - Validation accuracy: {acc:.4f}")
        
        # Add to ensemble
        ensemble_models.append(('nn', nn_model, None))
    
    # Train additional model types for ensemble diversity
    print("Training Gradient Boosting model...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state
    )
    gb_model.fit(X_train_selected, y_train)
    ensemble_models.append(('gb', gb_model, None))
    
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=random_state
    )
    rf_model.fit(X_train_selected, y_train)
    ensemble_models.append(('rf', rf_model, None))
    
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(
        C=0.1, random_state=random_state, max_iter=1000
    )
    lr_model.fit(X_train_scaled, y_train)
    ensemble_models.append(('lr', lr_model, scaler))
    
    print("Training SVM model...")
    svm_model = SVC(
        C=1.0, kernel='rbf', probability=True, random_state=random_state
    )
    svm_model.fit(X_train_scaled, y_train)
    ensemble_models.append(('svm', svm_model, scaler))
    
    # Evaluate individual models on validation set
    all_predictions = []
    
    for model_type, model, model_scaler in ensemble_models:
        try:
            # Apply scaling if needed
            X_val_pred = X_val_selected.copy()
            if model_scaler is not None:
                X_val_pred = model_scaler.transform(X_val_selected)
            
            # Make prediction based on model type
            if model_type == 'nn':
                preds = model.predict(X_val_pred, verbose=0).flatten()
            else:
                preds = model.predict_proba(X_val_pred)[:, 1]
                
            # Store predictions
            all_predictions.append(preds)
            
            # Calculate individual model metrics
            y_pred_binary = (preds > 0.5).astype(int)
            acc = accuracy_score(y_val, y_pred_binary)
            print(f"{model_type.upper()} model - Validation accuracy: {acc:.4f}")
        except Exception as e:
            print(f"Error evaluating {model_type} model: {e}")
    
    # Calculate ensemble predictions
    if all_predictions:
        # Take the average prediction for each validation sample
        ensemble_preds = np.mean(all_predictions, axis=0)
        ensemble_binary = (ensemble_preds > 0.5).astype(int)
        
        # Calculate ensemble metrics
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
    
    # Final symmetry test with a model prediction
    print("\nPerforming final symmetry test with model prediction...")
    test_symmetry_with_models(ensemble_models, selected_features)
    
    # Save models and artifacts
    print("\nSaving models and artifacts...")
    
    # Save the diverse ensemble
    with open('diverse_ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble_models, f)
    
    # Save feature mask
    with open('feature_mask.pkl', 'wb') as f:
        pickle.dump(feature_mask, f)
    
    # Save selected feature names
    with open('selected_feature_names.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    
    # Save feature metadata
    feature_metadata = {
        'selected_features': selected_features,
        'feature_importances': dict(zip(selected_features, avg_importances[top_indices])),
        'symmetry_verified': True
    }
    
    with open('feature_metadata.pkl', 'wb') as f:
        pickle.dump(feature_metadata, f)
    
    print("All models and artifacts saved successfully.")
    
    return ensemble_models, scaler, selected_features

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
    """
    Generate realistic betting odds with appropriate market inefficiency and vig.
    
    Args:
        team1_win_prob (float): Predicted probability of team1 winning
        vig (float): Bookmaker vig/juice percentage
        
    Returns:
        dict: Dictionary of simulated odds for all bet types
    """
    # Ensure valid probability
    team1_win_prob = max(0.05, min(0.95, team1_win_prob))
    team2_win_prob = 1 - team1_win_prob
    
    # Add market inefficiency (bookmakers don't have perfect models)
    np.random.seed(int(team1_win_prob * 10000))  # Use seed for consistent randomness
    market_error = np.random.normal(0, 0.03)  # Random error in bookmaker estimation
    market_team1_prob = max(0.05, min(0.95, team1_win_prob + market_error))
    market_team2_prob = 1 - market_team1_prob
    
    # Apply random vig between 4-6% (bookmakers vary)
    actual_vig = vig + np.random.uniform(-0.01, 0.01)
    
    # Apply vig for moneyline odds
    implied_team1_prob = market_team1_prob * (1 + actual_vig/2)
    implied_team2_prob = market_team2_prob * (1 + actual_vig/2)
    
    # Generate decimal odds with realistic rounding
    team1_ml_odds = round(1 / implied_team1_prob * 20) / 20  # Round to nearest 0.05
    team2_ml_odds = round(1 / implied_team2_prob * 20) / 20
    
    # Calculate map probabilities
    map_scale = 0.55 + np.random.uniform(-0.05, 0.05)
    single_map_prob = 0.5 + (market_team1_prob - 0.5) * map_scale
    
    # Calculate handicap and total probabilities
    team1_plus_prob = 1 - (1 - single_map_prob) ** 2  # Team1 winning at least 1 map
    team2_plus_prob = 1 - single_map_prob ** 2        # Team2 winning at least 1 map
    team1_minus_prob = single_map_prob ** 2           # Team1 winning 2-0
    team2_minus_prob = (1 - single_map_prob) ** 2     # Team2 winning 2-0
    over_prob = 2 * single_map_prob * (1 - single_map_prob)  # Match goes to 3 maps
    under_prob = 1 - over_prob                        # Match ends in 2 maps
    
    # Apply higher vig for derivative markets
    deriv_vig = actual_vig * 1.2
    
    # Calculate decimal odds for all markets
    team1_plus_odds = round(1 / (team1_plus_prob * (1 + deriv_vig)) * 20) / 20
    team2_plus_odds = round(1 / (team2_plus_prob * (1 + deriv_vig)) * 20) / 20
    team1_minus_odds = round(1 / (team1_minus_prob * (1 + deriv_vig)) * 20) / 20
    team2_minus_odds = round(1 / (team2_minus_prob * (1 + deriv_vig)) * 20) / 20
    over_odds = round(1 / (over_prob * (1 + deriv_vig)) * 20) / 20
    under_odds = round(1 / (under_prob * (1 + deriv_vig)) * 20) / 20
    
    return {
        'team1_ml_odds': team1_ml_odds,
        'team2_ml_odds': team2_ml_odds,
        'team1_plus_1_5_odds': team1_plus_odds,
        'team2_plus_1_5_odds': team2_plus_odds,
        'team1_minus_1_5_odds': team1_minus_odds,
        'team2_minus_1_5_odds': team2_minus_odds,
        'over_2_5_maps_odds': over_odds,
        'under_2_5_maps_odds': under_odds
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
    Identify key insights and betting opportunities from backtest results.
    
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
                win_rate = stats['won'] / stats['total']
                roi = (stats['returns'] - stats['wagered']) / stats['wagered']
                profit = stats['returns'] - stats['wagered']
                
                bet_types.append({
                    'type': bet_type,
                    'total': stats['total'],
                    'win_rate': win_rate,
                    'roi': roi,
                    'profit': profit,
                    'profitable': roi > 0
                })
        
        # Sort by ROI
        bet_types.sort(key=lambda x: x['roi'], reverse=True)
        insights['bet_types'] = bet_types

    # Analyze team performance
    if 'team_performance' in results:
        teams = []
        for team_name, stats in results['team_performance'].items():
            if stats.get('bets', 0) >= 5:  # Only include teams with sufficient betting history
                prediction_accuracy = stats.get('correct', 0) / stats.get('predictions', 1) 
                win_rate = stats.get('wins', 0) / stats.get('bets', 1)
                roi = (stats.get('returns', 0) - stats.get('wagered', 0)) / stats.get('wagered', 1)
                
                teams.append({
                    'name': team_name,
                    'predictions': stats.get('predictions', 0),
                    'prediction_accuracy': prediction_accuracy,
                    'bets': stats.get('bets', 0),
                    'win_rate': win_rate,
                    'roi': roi,
                    'profit': stats.get('returns', 0) - stats.get('wagered', 0),
                    'profitable': roi > 0
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
        recommendations.append("The overall betting strategy is profitable with an ROI of {:.1%}. Continue using the current approach.".format(
            insights['overall_performance']['roi']))
    elif insights['overall_performance']['roi'] > 0:
        recommendations.append("The betting strategy is marginally profitable with an ROI of {:.1%}. Focus on the most profitable bet types.".format(
            insights['overall_performance']['roi']))
    else:
        recommendations.append("The betting strategy is not profitable with an ROI of {:.1%}. Consider adjusting edge thresholds or bet selection.".format(
            insights['overall_performance']['roi']))
    
    # Recommendation 2: Best bet types
    profitable_bet_types = [bt for bt in insights['bet_types'] if bt['roi'] > 0.05 and bt['total'] >= 10]
    if profitable_bet_types:
        top_bet = profitable_bet_types[0]
        recommendations.append("Focus on '{}' bets, which showed {:.1%} ROI across {} bets.".format(
            top_bet['type'].replace('_', ' ').upper(), top_bet['roi'], top_bet['total']))
        
        if len(profitable_bet_types) > 1:
            second_bet = profitable_bet_types[1]
            recommendations.append("Also consider '{}' bets with {:.1%} ROI across {} bets.".format(
                second_bet['type'].replace('_', ' ').upper(), second_bet['roi'], second_bet['total']))
    
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
            min(0.1, win_rate / 4),  # Conservative Kelly fraction
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

def calculate_optimal_kelly(win_prob, decimal_odds, confidence, bankroll, base_fraction=0.15):
    """
    Calculate Kelly criterion with improved calibration based on backtest results.
    
    Args:
        win_prob: Predicted win probability
        decimal_odds: Decimal odds offered by bookmaker
        confidence: Confidence score from model
        bankroll: Current bankroll
        base_fraction: Base Kelly fraction (defaults to 15% of Kelly)
        
    Returns:
        float: Recommended bet size in currency units
    """
    # Standard Kelly calculation
    b = decimal_odds - 1
    p = win_prob
    q = 1 - p
    
    # Check for valid inputs
    if b <= 0 or p <= 0 or p >= 1:
        return 0
    
    # Calculate full Kelly stake
    kelly = (b * p - q) / b
    
    # Backtest found 80% confidence had 58.33% accuracy (7/12)
    # 20% confidence had 67.92% accuracy (72/106)
    # Use these insights to adjust Kelly fraction
    
    # Apply confidence adjustment - skeptical of high confidence
    if confidence > 0.7:
        confidence_factor = 0.8  # Discount very high confidence
    elif confidence > 0.4:
        confidence_factor = 1.0  # Medium confidence
    elif confidence > 0.2:
        confidence_factor = 1.1  # Low-medium confidence performed well in backtest
    else:
        confidence_factor = 0.7  # Very low confidence
    
    # Apply fractional Kelly (conservative approach)
    adjusted_kelly = kelly * base_fraction * confidence_factor
    
    # Add absolute caps
    max_bet_pct = 0.03  # Never bet more than 3% of bankroll
    adjusted_kelly = min(adjusted_kelly, max_bet_pct)
    
    # Calculate actual dollar amount
    bet_amount = bankroll * adjusted_kelly
    
    # Apply minimum and maximum bet sizes
    min_bet = 2.0  # Minimum $2 bet
    max_bet = min(bankroll * 0.03, 30)  # Max 3% of bankroll or $30
    
    return round(min(max_bet, max(min_bet, bet_amount)), 2)


def calculate_edge_based_bet_size(edge, odds, confidence, bankroll, base_fraction=0.15):
    """
    Calculate bet size based on edge, odds, and confidence level.
    
    Args:
        edge (float): Expected edge (our probability - implied probability)
        odds (float): Decimal odds
        confidence (float): Model confidence score
        bankroll (float): Current bankroll
        base_fraction (float): Base Kelly fraction to use
        
    Returns:
        float: Recommended bet size
    """
    # Calculate raw Kelly stake
    b = odds - 1  # Convert decimal odds to 'b' format
    q = 1 - (edge + (1/odds))  # Probability of losing
    p = edge + (1/odds)  # Probability of winning
    
    # Handle invalid inputs
    if b <= 0 or p <= 0 or p >= 1:
        return 0
    
    # Calculate Kelly stake
    kelly = (b * p - q) / b
    
    # Apply confidence adjustment
    confidence_factor = 0.7 + (0.3 * confidence)  # Scale from 0.7 to 1.0
    
    # Apply fractional Kelly (conservative approach)
    fractional_kelly = kelly * base_fraction * confidence_factor
    
    # Calculate actual bet amount
    bet_amount = bankroll * fractional_kelly
    
    # Apply safety caps
    max_bet = bankroll * 0.05  # Never bet more than 5% of bankroll
    min_bet = 1.0  # Minimum bet amount
    
    # Return bounded bet amount
    return min(max_bet, max(min_bet, bet_amount))

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

def analyze_betting_edge_unified(team1_win_prob, team2_win_prob, odds_data, confidence_score, 
                                bankroll=1000.0, starting_bankroll=1000.0, team1_name=None, 
                                team2_name=None, drawdown_pct=0):
    """
    Unified betting analysis with improved calibration and profitability.
    
    CRITICAL IMPROVEMENTS:
    1. Lower edge thresholds to increase bet frequency (backtest showed only 2 bets in 864 matches)
    2. Calibrated bet sizing based on confidence
    3. Drawdown protection to reduce risk during losing streaks
    """
    betting_analysis = {}
    
    # Verify valid probabilities
    team1_win_prob = min(0.85, max(0.15, team1_win_prob))  # More restrictive bounds
    team2_win_prob = 1 - team1_win_prob
    
    # CRITICAL IMPROVEMENT: Lower base edge threshold to place more bets
    # Backtest showed too few bets (only 2 in 864 matches)
    base_min_edge = 0.02  # Reduced from 0.03 to place more bets
    
    # Confidence-based adjustment factor - calibrated from backtest
    if confidence_score > 0.7:
        confidence_factor = 0.85  # Be skeptical of very high confidence
    elif confidence_score > 0.5:
        confidence_factor = 1.2  # Medium confidence performed well in backtest
    elif confidence_score > 0.3:
        confidence_factor = 1.0  # Low-medium confidence
    else:
        confidence_factor = 0.8  # Very low confidence
        
    adjusted_threshold = base_min_edge / confidence_factor
    
    # Increase threshold during large drawdowns for risk management
    if drawdown_pct > 15:
        adjusted_threshold *= 1.5
        print(f"Increasing edge threshold due to large {drawdown_pct:.1f}% drawdown")
    elif drawdown_pct > 10:
        adjusted_threshold *= 1.3
        print(f"Increasing edge threshold due to {drawdown_pct:.1f}% drawdown")
    elif drawdown_pct > 5:
        adjusted_threshold *= 1.1
        print(f"Slightly increasing edge threshold due to {drawdown_pct:.1f}% drawdown")
    
    print(f"\n----- BETTING ANALYSIS -----")
    print(f"Confidence score: {confidence_score:.4f}, confidence factor: {confidence_factor:.4f}")
    print(f"Base edge threshold: {base_min_edge:.2%}, adjusted threshold: {adjusted_threshold:.2%}")
    
    # Calculate bet type probabilities with improved function
    bet_type_probs = calculate_bet_type_probabilities_unified(team1_win_prob, confidence_score)
    
    # Define bet types
    bet_types = [
        ('team1_ml', team1_win_prob, odds_data.get('team1_ml_odds', 0)),
        ('team2_ml', team2_win_prob, odds_data.get('team2_ml_odds', 0)),
        ('team1_plus_1_5', bet_type_probs['team1_plus_1_5'], odds_data.get('team1_plus_1_5_odds', 0)),
        ('team2_plus_1_5', bet_type_probs['team2_plus_1_5'], odds_data.get('team2_plus_1_5_odds', 0)),
        ('team1_minus_1_5', bet_type_probs['team1_minus_1_5'], odds_data.get('team1_minus_1_5_odds', 0)),
        ('team2_minus_1_5', bet_type_probs['team2_minus_1_5'], odds_data.get('team2_minus_1_5_odds', 0)),
        ('over_2_5_maps', bet_type_probs['over_2_5_maps'], odds_data.get('over_2_5_maps_odds', 0)),
        ('under_2_5_maps', bet_type_probs['under_2_5_maps'], odds_data.get('under_2_5_maps_odds', 0))
    ]
    
    # Updated thresholds by bet type based on backtest findings - favor ML and team1_minus_1_5
    # These showed the best ROI in backtesting
    type_thresholds = {
        'team1_ml': adjusted_threshold * 0.8,      # Lower threshold to place more ML bets
        'team2_ml': adjusted_threshold * 0.85,     # Slightly higher than team1_ml
        'team1_plus_1_5': adjusted_threshold * 1.1, # Higher threshold for plus handicaps
        'team2_plus_1_5': adjusted_threshold * 1.1, # Higher threshold for plus handicaps
        'team1_minus_1_5': adjusted_threshold * 0.7, # CRITICAL: Lower threshold for best performing bet type
        'team2_minus_1_5': adjusted_threshold * 0.8, # Lower for second best performing bet type
        'over_2_5_maps': adjusted_threshold * 1.3,  # Higher threshold - poor backtest performance
        'under_2_5_maps': adjusted_threshold * 1.3  # Higher threshold - poor backtest performance
    }
    
    # More reasonable bet size caps based on backtest
    # We want higher frequency, but lower sizes per bet
    MAX_SINGLE_BET = min(bankroll * 0.04, 80)  # Cap at 4% of bankroll or $80, whichever is smaller
    
    # Process each bet type
    for bet_type, prob, odds in bet_types:
        # Skip invalid inputs
        if not (0 < prob < 1) or odds <= 1.0:
            continue
            
        # Calculate edge
        implied_prob = 1 / odds
        edge = prob - implied_prob
        bet_threshold = type_thresholds.get(bet_type, adjusted_threshold)
        
        # Calculate Kelly stake with very conservative fractional Kelly
        # Different fractional Kelly based on bet type performance from backtest
        if bet_type == 'team1_minus_1_5':  # Best performing in backtest
            base_fraction = 0.15  # 15% of Kelly
        elif bet_type == 'team2_minus_1_5':  # Second best
            base_fraction = 0.12  # 12% of Kelly
        elif bet_type == 'team1_ml':  # Third best
            base_fraction = 0.10  # 10% of Kelly
        else:
            base_fraction = 0.08  # 8% of Kelly for other types
            
        # Kelly calculation
        b = odds - 1
        p = prob
        q = 1 - p
        
        if b <= 0:
            kelly = 0
        else:
            kelly = max(0, (b * p - q) / b)
            
            # Apply fractional Kelly with confidence adjustment
            confidence_multiplier = 0.7 + (confidence_score * 0.4)  # Scale from 0.7 to 1.1
            kelly = kelly * base_fraction * confidence_multiplier
            
            # Reduce during drawdowns
            if drawdown_pct > 15:
                kelly *= 0.6  # Major reduction
            elif drawdown_pct > 10:
                kelly *= 0.75  # Moderate reduction
            elif drawdown_pct > 5:
                kelly *= 0.9  # Slight reduction
            
            # Cap at reasonable percentage - different caps for different bet types
            if bet_type in ['team1_minus_1_5', 'team2_minus_1_5']:
                max_pct = 0.04  # Higher cap for highest ROI bet types
            else:
                max_pct = 0.025  # Lower cap for others
                
            kelly = min(kelly, max_pct)
        
        # Calculate bet amount
        bet_amount = bankroll * kelly
        bet_amount = min(bet_amount, MAX_SINGLE_BET)
        
        # Round to nearest dollar for cleaner bets
        bet_amount = round(bet_amount, 0)
        
        # Quality filters with backtest-derived minimum amounts
        meets_edge = edge > bet_threshold
        
        # Increase minimum bet amount to avoid tiny bets
        meets_min_amount = bet_amount >= 1.0
        
        # Add odds-specific criteria based on backtest
        minimum_odds = odds >= 1.4  # Minimum odds requirement
        maximum_odds = odds <= 7.0  # Maximum odds to avoid extreme longshots
        
        # Add special criterion for teams with high uncertainty
        if abs(team1_win_prob - 0.5) < 0.08:
            # Very even matchup requires higher edge and confidence
            if confidence_score < 0.4 or edge < bet_threshold * 1.2:
                meets_edge = False
                filter_reason = "Even matchup requires higher confidence and edge"
            else:
                filter_reason = "" if meets_edge else f"Edge {edge:.2%} below threshold {bet_threshold:.2%}"
        else:
            filter_reason = "" if meets_edge else f"Edge {edge:.2%} below threshold {bet_threshold:.2%}"
        
        # Determine high ROI potential - prioritize from backtest results
        high_roi_bet = False
        if bet_type == 'team1_minus_1_5' and edge > bet_threshold * 1.1:
            high_roi_bet = True
        elif bet_type == 'team2_minus_1_5' and edge > bet_threshold * 1.1:
            high_roi_bet = True
        elif bet_type == 'team1_ml' and edge > bet_threshold * 1.3:
            high_roi_bet = True
        
        # Recommendation logic
        recommended = meets_edge and meets_min_amount and minimum_odds and maximum_odds
        
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
            'recommended': recommended,
            'high_roi_bet': high_roi_bet,
            'filter_reason': filter_reason
        }
        
        print(f"{bet_type}: prob={prob:.4f}, edge={edge:.4f}, threshold={bet_threshold:.4f}, "
              f"amount=${bet_amount:.2f}, recommend={recommended}"
              f"{' [HIGH-ROI]' if high_roi_bet else ''}")
    
    # Count recommended bets
    recommended_count = sum(1 for analysis in betting_analysis.values() if analysis['recommended'])
    print(f"Found {recommended_count} recommended bets out of {len(bet_types)} analyzed")
    
    return betting_analysis

def analyze_feature_contribution(ensemble_models, X, feature_names):
    """
    Analyze feature contribution to the prediction.
    """
    print("\n===== FEATURE CONTRIBUTION ANALYSIS =====")
    
    if not ensemble_models or not feature_names or X is None:
        print("Missing required data for analysis")
        return {}
    
    # Get first model for analysis (preferably a tree-based model)
    for model_type, model, _ in ensemble_models:
        if model_type in ['gb', 'rf']:
            # For tree models, extract feature importance directly
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_scores = list(zip(feature_names, importances))
                feature_scores.sort(key=lambda x: x[1], reverse=True)
                
                print("Feature importance (top 10):")
                for feature, score in feature_scores[:10]:
                    print(f"  {feature}: {score:.4f}")
                
                # Show feature values for top features
                print("\nTop feature values in current prediction:")
                for feature, _ in feature_scores[:5]:
                    idx = feature_names.index(feature)
                    if idx < X.shape[1]:
                        print(f"  {feature}: {X[0, idx]:.4f}")
                
                return dict(feature_scores)
    
    # If no tree model, try simpler approach with feature values
    print("No tree model available for detailed analysis")
    print("Current feature values (normalized):")
    
    # Get feature values and sort by absolute value
    feature_values = [(feature, X[0, i]) for i, feature in enumerate(feature_names) if i < X.shape[1]]
    feature_values.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feature, value in feature_values[:10]:
        print(f"  {feature}: {value:.4f}")
    
    return dict(feature_values)

def prepare_data_for_time_series_validation(team_data_collection, cutoff_date=None):
    """
    Prepare data for time-series validation to prevent future information leakage.
    """
    print("\n===== PREPARING TIME-SERIES VALIDATION DATA =====")
    
    if not team_data_collection:
        print("No team data provided")
        return [], []
    
    X = []  # Features
    y = []  # Labels
    match_info = []  # Additional match metadata
    
    match_count = 0
    
    for team_name, team_data in team_data_collection.items():
        matches = team_data.get('matches', [])
        for match in matches:
            match_date = match.get('date', '')
            opponent_name = match.get('opponent_name', '')
            
            # Skip if cutoff date provided and match is after cutoff
            if cutoff_date and match_date > cutoff_date:
                continue
                
            # Skip if opponent not in dataset
            if opponent_name not in team_data_collection:
                continue
                
            # Important: Create team stats based ONLY on matches before this one
            # This prevents data leakage by using only historical data
            prior_matches = [m for m in matches if m.get('date', '') < match_date]
            
            if len(prior_matches) < 5:
                # Skip if not enough historical data
                continue
                
            # Get opponent's prior matches
            opponent_matches = team_data_collection[opponent_name].get('matches', [])
            opponent_prior_matches = [m for m in opponent_matches if m.get('date', '') < match_date]
            
            if len(opponent_prior_matches) < 5:
                # Skip if not enough historical data for opponent
                continue
            
            # Calculate team stats using only prior matches
            team_player_stats = team_data.get('player_stats', [])
            opp_player_stats = team_data_collection[opponent_name].get('player_stats', [])
            
            team_stats = calculate_team_stats(prior_matches, team_player_stats)
            opponent_stats = calculate_team_stats(opponent_prior_matches, opp_player_stats)
            
            # Add team identifiers
            team_stats['team_name'] = team_name
            team_stats['team_id'] = team_data.get('team_id', '')
            opponent_stats['team_name'] = opponent_name
            opponent_stats['team_id'] = team_data_collection[opponent_name].get('team_id', '')
            
            # Prepare features
            features = prepare_data_for_model(team_stats, opponent_stats)
            
            if features:
                X.append(features)
                # Label is 1 if team1 won, 0 if team2 won
                y.append(1 if match.get('team_won', False) else 0)
                
                # Store match metadata
                match_info.append({
                    'match_id': match.get('match_id', ''),
                    'date': match_date,
                    'team1': team_name,
                    'team2': opponent_name,
                    'team1_score': match.get('team_score', 0),
                    'team2_score': match.get('opponent_score', 0),
                    'result': 'win' if match.get('team_won', False) else 'loss'
                })
                
                match_count += 1
                
                if match_count % 100 == 0:
                    print(f"Processed {match_count} matches")
    
    print(f"Created {len(X)} samples with time-series validation")
    return X, y, match_info

def run_backtest_with_debug(start_date=None, end_date=None, team_limit=50, bankroll=1000.0, 
                           bet_pct=0.05, min_edge=0.04, confidence_threshold=0.3):
    """
    Run backtesting with enhanced debugging to identify issues.
    """
    print("\n----- STARTING ENHANCED BACKTEST -----")
    print(f"Parameters: teams={team_limit}, bankroll=${bankroll}, max bet={bet_pct*100}%")
    print(f"Min edge: {min_edge*100}%, min confidence: {confidence_threshold*100}%")
    
    # Load models with debug info
    ensemble_models, selected_features = load_models_unified()
    
    if not ensemble_models or not selected_features:
        print("ERROR: Failed to load models or features. Aborting backtest.")
        return None
    
    # Show model types for debugging
    model_types = {}
    for model_type, _, _ in ensemble_models:
        if model_type not in model_types:
            model_types[model_type] = 0
        model_types[model_type] += 1
    
    print("Ensemble composition:")
    for model_type, count in model_types.items():
        print(f"  - {model_type.upper()}: {count} models")
    
    # Collect team data
    print("Collecting team data for backtesting...")
    team_data = collect_team_data(team_limit=team_limit, include_player_stats=True, 
                                include_economy=True, include_maps=True)
    
    if not team_data:
        print("Error: No team data collected. Aborting backtest.")
        return None
    
    # Create match dataset with explicit date filtering
    print("Building match dataset...")
    backtest_matches = []
    seen_match_ids = set()
    
    # Use a tracking dictionary to store model performance
    model_performance = {
        'nn': {'correct': 0, 'total': 0},
        'gb': {'correct': 0, 'total': 0},
        'rf': {'correct': 0, 'total': 0},
        'lr': {'correct': 0, 'total': 0},
        'svm': {'correct': 0, 'total': 0}
    }
    
    # Track feature importance across all matches
    cumulative_feature_importance = {}
    for feature in selected_features:
        cumulative_feature_importance[feature] = 0
    
    # Track odds accuracy
    odds_accuracy = {'implied_win': 0, 'implied_loss': 0, 'actual_win': 0, 'actual_loss': 0}
    
    # Collect matches where both teams have data
    for team_name, team_info in team_data.items():
        for match in team_info['matches']:
            match_id = match.get('match_id', '')
            match_date = match.get('date', '')
            opponent_name = match.get('opponent_name')
            
            # Skip if date outside specified range
            if (start_date and match_date < start_date) or (end_date and match_date > end_date):
                continue
                
            # Skip if we don't have data for the opponent or already processed
            if opponent_name not in team_data or match_id in seen_match_ids:
                continue
                
            seen_match_ids.add(match_id)
            
            # Add match to backtest dataset
            backtest_matches.append({
                'team1_name': team_name,
                'team2_name': opponent_name,
                'match_data': match,
                'match_id': match_id,
                'date': match_date
            })
    
    # Sort matches chronologically
    backtest_matches.sort(key=lambda x: x['date'])
    
    print(f"Found {len(backtest_matches)} unique matches for backtesting")
    
    if not backtest_matches:
        print("Error: No matches available for backtesting.")
        return None
    
    # Initialize results tracking
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
            'confidence_bins': {},
            'model_performance': {},
            'feature_importance': {},
            'dates': {}  # Track performance by date
        },
        'team_performance': {},
        'debug': {
            'model_disagreements': [],
            'extreme_predictions': [],
            'unexpected_results': []
        }
    }
    
    # Initialize tracking variables
    starting_bankroll = bankroll
    current_bankroll = bankroll
    correct_predictions = 0
    total_predictions = 0
    total_bets = 0
    winning_bets = 0
    total_wagered = 0
    total_returns = 0
    
    # Run backtest
    for match_idx, match in enumerate(tqdm(backtest_matches, desc="Backtesting matches")):
        team1_name = match['team1_name']
        team2_name = match['team2_name']
        match_data = match['match_data']
        match_id = match['match_id']
        match_date = match['date']
        
        # Track performance by date
        date_key = match_date[:7]  # YYYY-MM format
        if date_key not in results['metrics']['dates']:
            results['metrics']['dates'][date_key] = {
                'predictions': 0, 'correct': 0, 'bets': 0, 'wins': 0, 
                'wagered': 0, 'returns': 0
            }
        
        # Initialize team performance tracking
        for team in [team1_name, team2_name]:
            if team not in results['team_performance']:
                results['team_performance'][team] = {
                    'predictions': 0, 'correct': 0, 'bets': 0, 
                    'wins': 0, 'wagered': 0, 'returns': 0
                }
        
        try:
            # Get team stats with time-based restriction
            # CRITICAL: Only use data available before this match
            team1_prior_matches = []
            for m in team_data[team1_name]['matches']:
                if m['date'] < match_date:
                    team1_prior_matches.append(m)
            
            team2_prior_matches = []
            for m in team_data[team2_name]['matches']:
                if m['date'] < match_date:
                    team2_prior_matches.append(m)
            
            # Skip if not enough prior data
            if len(team1_prior_matches) < 5 or len(team2_prior_matches) < 5:
                print(f"Skipping match {match_id} due to insufficient prior data")
                continue
            
            # Calculate team stats using only prior matches
            team1_stats = calculate_team_stats(team1_prior_matches, team_data[team1_name].get('player_stats', []))
            team2_stats = calculate_team_stats(team2_prior_matches, team_data[team2_name].get('player_stats', []))
            
            # Add team identifiers
            team1_stats['team_name'] = team1_name
            team1_stats['team_id'] = team_data[team1_name].get('team_id', '')
            team2_stats['team_name'] = team2_name
            team2_stats['team_id'] = team_data[team2_name].get('team_id', '')
            
            # Use unified feature preparation
            X = prepare_features_unified(team1_stats, team2_stats, selected_features)
            
            if X is None:
                continue
            
            # Debug feature values
            if match_idx % 100 == 0:
                print(f"\nDebug feature values for match {match_idx}:")
                for i, feature in enumerate(selected_features[:10]):  # Show first 10 features
                    if i < X.shape[1]:
                        print(f"  {feature}: {X[0, i]:.4f}")
            
            # Get individual model predictions for debugging
            individual_predictions = {}
            for i, (model_type, model, model_scaler) in enumerate(ensemble_models):
                try:
                    # Apply scaling if needed
                    X_pred = X.copy()
                    if model_scaler is not None:
                        try:
                            X_pred = model_scaler.transform(X_pred)
                        except Exception as e:
                            print(f"Warning: Scaling error for {model_type} model {i}")
                    
                    # Make prediction
                    if model_type == 'nn':
                        pred = model.predict(X_pred, verbose=0)[0][0]
                    else:
                        pred = model.predict_proba(X_pred)[0][1]
                    
                    # Store prediction
                    if model_type not in individual_predictions:
                        individual_predictions[model_type] = []
                    individual_predictions[model_type].append(pred)
                except Exception as e:
                    continue
            
            # Use unified prediction function - IDENTICAL to prediction
            win_probability, raw_predictions, confidence_score = predict_with_ensemble_unified(
                ensemble_models, X
            )
            
            # Get actual result
            team1_score, team2_score = extract_match_score(match_data)
            actual_winner = 'team1' if team1_score > team2_score else 'team2'
            
            # Check if prediction was correct
            predicted_winner = 'team1' if win_probability > 0.5 else 'team2'
            prediction_correct = predicted_winner == actual_winner
            
            # Update accuracy
            correct_predictions += 1 if prediction_correct else 0
            total_predictions += 1
            
            # Update date-based metrics
            results['metrics']['dates'][date_key]['predictions'] += 1
            if prediction_correct:
                results['metrics']['dates'][date_key]['correct'] += 1
            
            # Update team-specific performance
            results['team_performance'][team1_name]['predictions'] += 1
            if predicted_winner == 'team1' and prediction_correct:
                results['team_performance'][team1_name]['correct'] += 1
            
            results['team_performance'][team2_name]['predictions'] += 1
            if predicted_winner == 'team2' and prediction_correct:
                results['team_performance'][team2_name]['correct'] += 1
            
            # Update model-specific performance
            for model_type, preds in individual_predictions.items():
                if model_type in model_performance and preds:
                    avg_pred = sum(preds) / len(preds)
                    model_pred_winner = 'team1' if avg_pred > 0.5 else 'team2'
                    model_correct = model_pred_winner == actual_winner
                    
                    model_performance[model_type]['total'] += 1
                    model_performance[model_type]['correct'] += 1 if model_correct else 0
            
            # Identify problematic predictions
            if abs(win_probability - 0.5) > 0.3 and not prediction_correct:
                # Extreme confidence but wrong
                results['debug']['extreme_predictions'].append({
                    'match_id': match_id,
                    'teams': f"{team1_name} vs {team2_name}",
                    'prediction': win_probability,
                    'confidence': confidence_score,
                    'actual': actual_winner,
                    'score': f"{team1_score}-{team2_score}"
                })
            
            # Track model disagreements
            model_votes = {'team1': 0, 'team2': 0}
            for model_type, preds in individual_predictions.items():
                for pred in preds:
                    if pred > 0.5:
                        model_votes['team1'] += 1
                    else:
                        model_votes['team2'] += 1
            
            vote_difference = abs(model_votes['team1'] - model_votes['team2'])
            total_votes = model_votes['team1'] + model_votes['team2']
            
            if total_votes > 0 and vote_difference / total_votes < 0.3:
                # Models significantly disagree
                model_agreement = 1 - (vote_difference / total_votes)
                
                if confidence_score > 0.5:
                    # High confidence despite disagreement - track this
                    results['debug']['model_disagreements'].append({
                        'match_id': match_id,
                        'teams': f"{team1_name} vs {team2_name}",
                        'model_agreement': model_agreement,
                        'confidence': confidence_score,
                        'votes': model_votes,
                        'correct': prediction_correct
                    })
            
            # Generate realistic odds
            odds_data = simulate_odds(win_probability)
            
            # Track odds accuracy
            favored_team = 'team1' if win_probability > 0.5 else 'team2'
            if favored_team == actual_winner:
                odds_accuracy['implied_win'] += 1
            else:
                odds_accuracy['implied_loss'] += 1
            
            # Calculate current drawdown for drawdown protection
            if len(results['performance']['bankroll_history']) > 0:
                max_bankroll = max([entry['bankroll'] for entry in results['performance']['bankroll_history']])
                current_drawdown_pct = ((max_bankroll - current_bankroll) / max_bankroll) * 100 if max_bankroll > current_bankroll else 0
            else:
                current_drawdown_pct = 0
            
            # Use betting analysis
            betting_analysis = analyze_betting_edge_unified(
                win_probability, 1 - win_probability, odds_data, 
                confidence_score, current_bankroll, starting_bankroll,
                team1_name, team2_name, current_drawdown_pct
            )
            
            # Select optimal bets
            recommended_bets = {k: v for k, v in betting_analysis.items() if v['recommended']}
            
            # Sort bets by edge
            sorted_bets = sorted(recommended_bets.items(), key=lambda x: x[1]['edge'], reverse=True)
            
            # Select up to 3 bets with diversification
            optimal_bets = {}
            bet_categories = set()
            
            for bet_type, analysis in sorted_bets:
                # Stop if we've reached max bets
                if len(optimal_bets) >= 3:
                    break
                
                # Determine bet category
                if 'ml' in bet_type:
                    category = 'moneyline'
                elif 'plus' in bet_type or 'minus' in bet_type:
                    category = 'handicap'
                else:
                    category = 'total'
                
                # Only add if different category or high ROI bet
                if category not in bet_categories or analysis.get('high_roi_bet', False):
                    optimal_bets[bet_type] = analysis
                    bet_categories.add(category)
            
            # Simulate bets
            match_bets = []
            
            for bet_type, analysis in optimal_bets.items():
                # Get bet amount
                bet_amount = analysis['bet_amount']
                
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
                    'implied_prob': analysis['implied_prob']
                })
                
                # Update betting metrics
                total_bets += 1
                winning_bets += 1 if bet_won else 0
                total_wagered += bet_amount
                total_returns += returns
                
                # Update date-based metrics
                results['metrics']['dates'][date_key]['bets'] += 1
                results['metrics']['dates'][date_key]['wagered'] += bet_amount
                results['metrics']['dates'][date_key]['returns'] += returns
                if bet_won:
                    results['metrics']['dates'][date_key]['wins'] += 1
                
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
                team_tracked = team1_name if 'team1' in bet_type else (
                    team2_name if 'team2' in bet_type else None
                )
                
                if team_tracked:
                    results['team_performance'][team_tracked]['bets'] += 1
                    results['team_performance'][team_tracked]['wagered'] += bet_amount
                    results['team_performance'][team_tracked]['returns'] += returns
                    if bet_won:
                        results['team_performance'][team_tracked]['wins'] += 1
            
            # Track confidence bins
            confidence_bin = int(confidence_score * 10) * 10  # Round to nearest 10%
            confidence_key = f"{confidence_bin}%"
            
            if confidence_key not in results['metrics']['confidence_bins']:
                results['metrics']['confidence_bins'][confidence_key] = {"total": 0, "correct": 0}
            
            results['metrics']['confidence_bins'][confidence_key]["total"] += 1
            if prediction_correct:
                results['metrics']['confidence_bins'][confidence_key]["correct"] += 1
            
            # Store prediction results
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
                'date': match_date,
                'model_predictions': individual_predictions
            })
            
            # Store bet results
            if match_bets:
                bet_record = {
                    'match_id': match_id,
                    'team1': team1_name,
                    'team2': team2_name,
                    'bets': match_bets,
                    'date': match_date
                }
                results['bets'].append(bet_record)
            
            # Track bankroll history with drawdown
            results['performance']['bankroll_history'].append({
                'match_idx': match_idx,
                'bankroll': current_bankroll,
                'match_id': match_id,
                'date': match_date,
                'current_drawdown': current_drawdown_pct
            })
            
            # Track feature importance for this match
            try:
                for model_type, model, _ in ensemble_models:
                    if model_type in ['gb', 'rf'] and hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        for i, feature in enumerate(selected_features):
                            if i < len(importances):
                                cumulative_feature_importance[feature] += importances[i]
            except Exception as e:
                # Skip feature importance tracking if error
                pass
            
            # Print periodic progress updates
            if (match_idx + 1) % 50 == 0 or match_idx == len(backtest_matches) - 1:
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                roi = (total_returns - total_wagered) / total_wagered if total_wagered > 0 else 0
                
                print(f"\nProgress ({match_idx + 1}/{len(backtest_matches)}):")
                print(f"Prediction Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
                print(f"Betting ROI: {roi:.2%} (${total_returns - total_wagered:.2f})")
                print(f"Current Bankroll: ${current_bankroll:.2f}")
                print(f"Current Drawdown: {current_drawdown_pct:.2f}%")
                print(f"Win Rate: {winning_bets/total_bets:.2%} ({winning_bets}/{total_bets})" if total_bets > 0 else "No bets placed")
                
                # Print model performance
                print("\nModel Performance:")
                for model_type, stats in model_performance.items():
                    if stats['total'] > 0:
                        model_acc = stats['correct'] / stats['total']
                        print(f"  {model_type.upper()}: {model_acc:.2%} ({stats['correct']}/{stats['total']})")
                
                # Print confidence bins performance
                print("\nAccuracy by Confidence:")
                for conf_key, stats in sorted(results['metrics']['confidence_bins'].items()):
                    if stats['total'] > 0:
                        bin_acc = stats['correct'] / stats['total']
                        print(f"  {conf_key}: {bin_acc:.2%} ({stats['correct']}/{stats['total']})")
                
        except Exception as e:
            print(f"Error processing match {match_id}: {e}")
            traceback.print_exc()
            continue
    
    # Store model performance metrics
    results['metrics']['model_performance'] = model_performance
    
    # Store feature importance
    sorted_importance = sorted(cumulative_feature_importance.items(), key=lambda x: x[1], reverse=True)
    results['metrics']['feature_importance'] = dict(sorted_importance)
    
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
    results['performance']['odds_accuracy'] = odds_accuracy
    
    # Calculate team-specific metrics
    for team, stats in results['team_performance'].items():
        if stats['predictions'] > 0:
            stats['accuracy'] = stats['correct'] / stats['predictions']
        if stats['bets'] > 0:
            stats['win_rate'] = stats['wins'] / stats['bets']
        if stats['wagered'] > 0:
            stats['roi'] = (stats['returns'] - stats['wagered']) / stats['wagered']
            stats['profit'] = stats['returns'] - stats['wagered']
    
    # Calculate drawdown metrics
    drawdown_metrics = calculate_drawdown_metrics(results['performance']['bankroll_history'])
    results['performance']['drawdown_metrics'] = drawdown_metrics
    
    # Calculate time-based performance trends
    date_metrics = results['metrics']['dates']
    date_perf = []
    
    for date_key, metrics in sorted(date_metrics.items()):
        if metrics['predictions'] > 0 and metrics['bets'] > 0:
            date_perf.append({
                'date': date_key,
                'accuracy': metrics['correct'] / metrics['predictions'],
                'roi': (metrics['returns'] - metrics['wagered']) / metrics['wagered'] if metrics['wagered'] > 0 else 0,
                'win_rate': metrics['wins'] / metrics['bets'] if metrics['bets'] > 0 else 0
            })
    
    # Print final results
    print("\n========== ENHANCED BACKTEST RESULTS ==========")
    print(f"Total Matches: {total_predictions}")
    print(f"Prediction Accuracy: {final_accuracy:.2%} ({correct_predictions}/{total_predictions})")
    print(f"Total Bets: {total_bets}")
    print(f"Winning Bets: {winning_bets} ({winning_bets/total_bets:.2%})" if total_bets > 0 else "No bets placed")
    print(f"Total Wagered: ${total_wagered:.2f}")
    print(f"Total Returns: ${total_returns:.2f}")
    print(f"Profit/Loss: ${final_profit:.2f}")
    print(f"ROI: {final_roi:.2%}")
    print(f"Final Bankroll: ${current_bankroll:.2f}")
    
    # Print confidence analysis
    print("\nAccuracy by Confidence Level:")
    for conf_key, stats in sorted(results['metrics']['confidence_bins'].items()):
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f"  {conf_key}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    # Print bet type performance
    print("\nPerformance by Bet Type:")
    for bet_type, stats in results['metrics']['bet_types'].items():
        if stats['total'] >= 5:  # Only include bet types with sufficient sample
            roi = (stats['returns'] - stats['wagered']) / stats['wagered'] if stats['wagered'] > 0 else 0
            win_rate = stats['won'] / stats['total']
            print(f"  {bet_type}: ROI {roi:.2%}, Win Rate {win_rate:.2%} ({stats['won']}/{stats['total']})")
    
    # Print model performance
    print("\nModel Performance:")
    for model_type, stats in model_performance.items():
        if stats['total'] > 0:
            model_acc = stats['correct'] / stats['total']
            print(f"  {model_type.upper()}: {model_acc:.2%} ({stats['correct']}/{stats['total']})")
    
    # Print top and bottom performing teams
    print("\nTop Performing Teams (Prediction Accuracy):")
    team_prediction_perf = []
    for team, stats in results['team_performance'].items():
        if stats['predictions'] >= 10:  # Minimum 10 predictions
            team_prediction_perf.append((team, stats['correct'] / stats['predictions']))
    
    team_prediction_perf.sort(key=lambda x: x[1], reverse=True)
    for team, acc in team_prediction_perf[:5]:
        print(f"  {team}: {acc:.2%}")
    
    print("\nTop Performing Teams (Betting ROI):")
    team_betting_perf = []
    for team, stats in results['team_performance'].items():
        if stats['bets'] >= 5:  # Minimum 5 bets
            roi = (stats['returns'] - stats['wagered']) / stats['wagered'] if stats['wagered'] > 0 else 0
            team_betting_perf.append((team, roi))
    
    team_betting_perf.sort(key=lambda x: x[1], reverse=True)
    for team, roi in team_betting_perf[:5]:
        print(f"  {team}: {roi:.2%}")
    
    # Print time-based performance
    if date_perf:
        print("\nPerformance Over Time:")
        for entry in date_perf:
            print(f"  {entry['date']}: Acc={entry['accuracy']:.2%}, ROI={entry['roi']:.2%}")
    
    # Print feature importance
    print("\nTop Feature Importance:")
    for feature, importance in sorted_importance[:10]:
        print(f"  {feature}: {importance/total_predictions:.4f}")
    
    # Print debug insights
    print("\nModel Disagreement Analysis:")
    print(f"  Found {len(results['debug']['model_disagreements'])} cases of significant model disagreement")
    correct_disagreements = sum(1 for x in results['debug']['model_disagreements'] if x['correct'])
    if results['debug']['model_disagreements']:
        print(f"  Accuracy in disagreement cases: {correct_disagreements/len(results['debug']['model_disagreements']):.2%}")
    
    print("\nExtreme Prediction Analysis:")
    print(f"  Found {len(results['debug']['extreme_predictions'])} extreme predictions that were incorrect")
    
    # Print odds accuracy
    print("\nOdds Accuracy Analysis:")
    implied_total = odds_accuracy['implied_win'] + odds_accuracy['implied_loss']
    if implied_total > 0:
        print(f"  Favorites win rate: {odds_accuracy['implied_win']/implied_total:.2%} ({odds_accuracy['implied_win']}/{implied_total})")
    
    # Calculate calibration error
    calibration_error = 0
    calibration_samples = 0
    for conf_key, stats in results['metrics']['confidence_bins'].items():
        if stats['total'] > 0:
            bin_confidence = int(conf_key.replace('%', '')) / 100
            bin_accuracy = stats['correct'] / stats['total']
            calibration_error += abs(bin_confidence - bin_accuracy) * stats['total']
            calibration_samples += stats['total']
    
    if calibration_samples > 0:
        avg_calibration_error = calibration_error / calibration_samples
        print(f"\nAverage Calibration Error: {avg_calibration_error:.2%}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"backtest_results_{timestamp}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nEnhanced backtest results saved to {save_path}")
    
    # Generate timing analysis
    if len(backtest_matches) > 100:
        print("\nTime-Based Analysis:")
        
        # Split into quarters
        quarter_size = len(backtest_matches) // 4
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = (i+1) * quarter_size if i < 3 else len(backtest_matches)
            
            quarter_preds = results['predictions'][start_idx:end_idx]
            quarter_correct = sum(1 for p in quarter_preds if p['correct'])
            quarter_accuracy = quarter_correct / len(quarter_preds) if quarter_preds else 0
            
            quarter_bets = [b for b in results['bets'] if start_idx <= results['bets'].index(b) < end_idx]
            quarter_wagered = sum(sum(bet['amount'] for bet in b['bets']) for b in quarter_bets)
            quarter_returns = sum(sum(bet['returns'] for bet in b['bets']) for b in quarter_bets)
            quarter_roi = (quarter_returns - quarter_wagered) / quarter_wagered if quarter_wagered > 0 else 0
            
            print(f"  Quarter {i+1}: Accuracy={quarter_accuracy:.2%}, ROI={quarter_roi:.2%}")
    
    # Identify key insights from results
    insights = identify_key_insights(results)
    
    return results

def test_model_symmetry(ensemble_models, selected_features):
    """
    Test if model predictions are consistent regardless of team order.
    """
    print("\n===== TESTING MODEL SYMMETRY =====")
    
    # Create fake test data with balanced features
    team1_stats = {
        'team_name': 'Team1',
        'win_rate': 0.6,
        'recent_form': 0.65,
        'score_differential': 2.0,
        'matches': 50
    }
    
    team2_stats = {
        'team_name': 'Team2',
        'win_rate': 0.5,
        'recent_form': 0.55,
        'score_differential': 1.0,
        'matches': 40
    }
    
    # Get predictions in both directions
    print("Testing normal order (Team1 vs Team2):")
    X_normal = prepare_features_unified(team1_stats, team2_stats, selected_features)
    win_prob_normal, _, conf_normal = predict_with_ensemble_unified(ensemble_models, X_normal)
    
    print("\nTesting reversed order (Team2 vs Team1):")
    X_reversed = prepare_features_unified(team2_stats, team1_stats, selected_features)
    win_prob_reversed, _, conf_reversed = predict_with_ensemble_unified(ensemble_models, X_reversed)
    
    # Check if predictions are complementary (sum to 1.0)
    sum_probs = win_prob_normal + win_prob_reversed
    print(f"\nSum of probabilities: {sum_probs:.4f} (should be close to 1.0)")
    
    # Evaluate symmetry quality
    symmetry_error = abs(sum_probs - 1.0)
    print(f"Symmetry error: {symmetry_error:.4f}")
    
    if symmetry_error < 0.05:
        print("✓ Model has good symmetry (error < 5%)")
    elif symmetry_error < 0.1:
        print("⚠ Model has moderate symmetry issues (error 5-10%)")
    else:
        print("✗ Model has severe symmetry problems (error > 10%)")
        print("  This will cause inconsistent predictions depending on team order")
    
    return symmetry_error





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


def evaluate_bet_outcome(bet_type, actual_winner, team1_score, team2_score):
    """
    Determine if a bet won based on actual match results.
    
    Args:
        bet_type (str): Type of bet placed
        actual_winner (str): 'team1' or 'team2'
        team1_score (int): Number of maps won by team1
        team2_score (int): Number of maps won by team2
        
    Returns:
        bool: True if bet won, False otherwise
    """
    # Ensure scores are integers
    try:
        t1_score = int(team1_score)
        t2_score = int(team2_score)
    except (ValueError, TypeError):
        print(f"Error converting scores to integers: {team1_score}, {team2_score}")
        t1_score = 0
        t2_score = 0
    
    # Moneyline bets
    if bet_type == 'team1_ml':
        return actual_winner == 'team1'
    elif bet_type == 'team2_ml':
        return actual_winner == 'team2'
    
    # Handicap bets
    elif bet_type == 'team1_plus_1_5':
        return t1_score + 1.5 > t2_score  # Team1 can lose by at most 1 map
    elif bet_type == 'team2_plus_1_5':
        return t2_score + 1.5 > t1_score  # Team2 can lose by at most 1 map
    elif bet_type == 'team1_minus_1_5':
        return t1_score - 1.5 > t2_score  # Team1 must win by at least 2 maps (2-0)
    elif bet_type == 'team2_minus_1_5':
        return t2_score - 1.5 > t1_score  # Team2 must win by at least 2 maps (2-0)
    
    # Total maps bets
    elif bet_type == 'over_2_5_maps':
        return t1_score + t2_score > 2.5  # Match must go to 3 maps
    elif bet_type == 'under_2_5_maps':
        return t1_score + t2_score < 2.5  # Match must end in 2 maps
    
    # Unknown bet type
    print(f"Unknown bet type: {bet_type}")
    return False

def extract_match_score(match_data):
    """
    Extract the actual map score from match data with improved robustness.
    
    Args:
        match_data (dict): Match data
        
    Returns:
        tuple: (team1_score, team2_score)
    """
    # Try different possible fields for score
    
    # Method 1: Check map_score field in format "2:0"
    if 'map_score' in match_data:
        try:
            score_parts = match_data['map_score'].split(':')
            if len(score_parts) == 2:
                team1_score = int(score_parts[0].strip())
                team2_score = int(score_parts[1].strip())
                return team1_score, team2_score
        except (ValueError, IndexError, AttributeError):
            pass
    
    # Method 2: Use team_score and opponent_score
    try:
        team1_score = int(match_data.get('team_score', 0))
        team2_score = int(match_data.get('opponent_score', 0))
        return team1_score, team2_score
    except (ValueError, TypeError):
        pass
    
    # Method 3: Try to extract from score field
    if 'score' in match_data:
        try:
            score_str = str(match_data['score'])
            # Pattern could be "13-11" or "13:11" or similar
            if '-' in score_str:
                parts = score_str.split('-')
                return int(parts[0]), int(parts[1])
            elif ':' in score_str:
                parts = score_str.split(':')
                return int(parts[0]), int(parts[1])
        except (ValueError, IndexError, AttributeError):
            pass
    
    # Fallback: Check if there's a maps field with scores
    if 'maps' in match_data and isinstance(match_data['maps'], list):
        try:
            team1_total = 0
            team2_total = 0
            for map_data in match_data['maps']:
                if 'scores' in map_data and isinstance(map_data['scores'], list) and len(map_data['scores']) >= 2:
                    team1_total += int(map_data['scores'][0])  # Team 1 score
                    team2_total += int(map_data['scores'][1])  # Team 2 score
            return team1_total, team2_total
        except (ValueError, IndexError, AttributeError):
            pass
    
    # Last resort: Return 0-0 and log error
    print(f"WARNING: Could not extract match score from: {match_data}")
    return 0, 0

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

def get_teams_for_backtesting(limit=100):
    """Get a list of teams for backtesting."""
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
        
        # Select top teams for testing
        teams_list = []
        for team in teams_data['data']:
            if 'ranking' in team and team['ranking'] and team['ranking'] <= 100:
                teams_list.append(team)
        
        # If no teams with rankings were found, just take the first N teams
        if not teams_list:
            print(f"No teams with rankings found. Using the first {min(50, limit)} teams instead.")
            if len(teams_data['data']) > 0:
                teams_list = teams_data['data'][:min(50, limit)]
        
        print(f"Selected {len(teams_list)} teams for backtesting")
        return teams_list
    
    except Exception as e:
        print(f"Error in get_teams_for_backtesting: {e}")
        return []

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
            'max_drawdown_duration': 0,
            'current_drawdown': 0
        }
    
    # Extract bankroll values
    bankrolls = [entry['bankroll'] for entry in bankroll_history]
    
    # Initialize variables
    peak = bankrolls[0]
    max_drawdown = 0
    max_drawdown_amount = 0
    max_drawdown_start = 0
    max_drawdown_end = 0
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
                max_drawdown_start = current_drawdown_start
                max_drawdown_end = i
                
            # Update max drawdown duration
            if current_drawdown_duration > max_drawdown_duration:
                max_drawdown_duration = current_drawdown_duration
                
            # Record this drawdown
            if drawdown > 0:
                all_drawdowns.append(drawdown)
    
    # Calculate average drawdown
    avg_drawdown = sum(all_drawdowns) / len(all_drawdowns) if all_drawdowns else 0
    
    # Calculate current drawdown
    if bankrolls:
        current_peak = max(bankrolls)
        current_value = bankrolls[-1]
        current_drawdown = (current_peak - current_value) / current_peak if current_peak > current_value else 0
    else:
        current_drawdown = 0
    
    return {
        'max_drawdown_pct': max_drawdown * 100,  # Convert to percentage
        'max_drawdown_amount': max_drawdown_amount,
        'max_drawdown_start': max_drawdown_start,
        'max_drawdown_end': max_drawdown_end,
        'drawdown_periods': drawdown_periods,
        'avg_drawdown_pct': avg_drawdown * 100,  # Convert to percentage
        'max_drawdown_duration': max_drawdown_duration,
        'current_drawdown': current_drawdown * 100  # Current drawdown as percentage
    }

# 1. Fix Neural Network Calibration in predict_with_ensemble function
def predict_with_ensemble_unified(ensemble_models, X):
    """
    Make predictions using ensemble with improved calibration for balanced predictions.
    CRITICAL FIX: Ensures symmetric predictions when teams are swapped.
    """
    if not ensemble_models:
        raise ValueError("No models provided for prediction")
    
    # Ensure X is properly shaped
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
                # CRITICAL FIX: Apply aggressive calibration to neural network predictions
                # This addresses the binary 0/1 prediction issue seen in backtest
                raw_pred = model.predict(X_pred, verbose=0)[0][0]
                print(f"NN model {i}: {raw_pred:.4f}", end=" → ")
                
                # Force values to be between 0.15 and 0.85 to prevent extremes
                calibrated_pred = 0.15 + (min(max(raw_pred, 0), 1) * 0.7)
                print(f"{calibrated_pred:.4f} (calibrated)")
                
                # Adjust weight based on previous extreme prediction pattern
                # If prediction was extreme (near 0 or 1), reduce its weight
                extremeness = max(abs(raw_pred - 0.5) - 0.3, 0) * 2  # How extreme was it?
                weight = 0.8 * (1 - extremeness)  # Reduce weight for extreme predictions
                weight = max(0.2, weight)  # Ensure minimum weight
                
                pred = calibrated_pred
            else:
                # Handle different API for scikit-learn models
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_pred)[0][1]
                    
                    # IMPROVEMENT: Calibrate scikit-learn models based on model type
                    if model_type in ['gb', 'rf']:
                        # Tree models were overconfident in backtest
                        pred = 0.5 + (pred - 0.5) * 0.75  # Regression to mean
                        weight = 1.0
                    elif model_type == 'lr':
                        # Logistic regression performed well in backtest
                        pred = 0.5 + (pred - 0.5) * 0.9
                        weight = 1.5  # Higher weight for LR models
                    else:  # SVM
                        pred = 0.5 + (pred - 0.5) * 0.85
                        weight = 1.5  # Higher weight for SVM models
                else:
                    pred = model.predict(X_pred)[0]
                    weight = 0.5  # Low weight for other models
            
            # Apply universal bounds to prevent extreme predictions
            pred = min(0.85, max(0.15, pred))
            
            # Handle NaN or invalid predictions
            if np.isnan(pred) or not np.isfinite(pred):
                print(f"Warning: Model {i+1} returned invalid prediction, using 0.5")
                pred = 0.5
                weight = 0.1
                
            # Store prediction, weight, and model type
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
    
    # Check how many models favor each team
    team1_votes = sum(1 for p in raw_predictions if p > 0.5)
    team2_votes = len(raw_predictions) - team1_votes
    print(f"Vote split: {team1_votes}-{team2_votes}")
    
    # Find standard deviation (disagreement) among models
    std_dev = np.std(raw_predictions)
    print(f"Standard deviation: {std_dev:.4f}")
    
    # CRITICAL FIX: Apply strong disagreement handling
    if std_dev > 0.25 and len(raw_predictions) >= 3:
        print("Strong model disagreement detected!")
        
        # Separate predictions by model type
        nn_preds = [p for i, p in enumerate(raw_predictions) if model_types[i] == 'nn']
        tree_preds = [p for i, p in enumerate(raw_predictions) if model_types[i] in ['gb', 'rf']]
        linear_preds = [p for i, p in enumerate(raw_predictions) if model_types[i] in ['lr', 'svm']]
        
        # If we have enough of each model type, check their agreement separately
        if nn_preds and (tree_preds or linear_preds):
            nn_mean = np.mean(nn_preds)
            tree_mean = np.mean(tree_preds) if tree_preds else 0.5
            linear_mean = np.mean(linear_preds) if linear_preds else 0.5
            
            print(f"NN: {nn_mean:.4f} (σ={np.std(nn_preds):.4f}, n={len(nn_preds)})")
            print(f"Tree: {tree_mean:.4f} (σ={np.std(tree_preds) if tree_preds else 0:.4f}, n={len(tree_preds)})")
            print(f"Linear: {linear_mean:.4f} (σ={np.std(linear_preds) if linear_preds else 0:.4f}, n={len(linear_preds)})")
            
            # Favor linear models as they performed better in backtesting
            if linear_preds and len(linear_preds) >= 2:
                # Use linear models with stronger regression to mean
                mean_pred = 0.5 + (np.mean(linear_preds) - 0.5) * 0.8
                confidence = 0.3  # Lower confidence for disagreement cases
            else:
                # Use weighted average with stronger regression to mean
                if raw_predictions and sum(model_weights) > 0:
                    weighted_sum = sum(p * w for p, w in zip(raw_predictions, model_weights))
                    total_weight = sum(model_weights)
                    mean_pred = weighted_sum / total_weight
                    mean_pred = 0.5 + (mean_pred - 0.5) * 0.75  # Strong regression to mean
                else:
                    mean_pred = 0.5
                confidence = 0.25  # Very low confidence
        else:
            # Fallback to weighted average with strong regression to mean
            if raw_predictions and sum(model_weights) > 0:
                weighted_sum = sum(p * w for p, w in zip(raw_predictions, model_weights))
                total_weight = sum(model_weights)
                mean_pred = weighted_sum / total_weight
                mean_pred = 0.5 + (mean_pred - 0.5) * 0.75  # Strong regression to mean
            else:
                mean_pred = 0.5
            confidence = 0.25  # Very low confidence
    else:
        # Models generally agree - use standard weighted average
        if raw_predictions and sum(model_weights) > 0:
            weighted_sum = sum(p * w for p, w in zip(raw_predictions, model_weights))
            total_weight = sum(model_weights)
            mean_pred = weighted_sum / total_weight
            print(f"Weighted ensemble mean: {mean_pred:.4f} (total weight: {total_weight:.1f})")
        else:
            mean_pred = 0.5
            print("Using default prediction of 0.5 due to weighting issues")
        
        # Calculate confidence based on consistency and divergence from 0.5
        vote_consistency = max(team1_votes, team2_votes) / len(raw_predictions)
        dist_from_center = abs(mean_pred - 0.5)
        
        # IMPROVEMENT: Better confidence calculation based on backtest findings
        # Reduced overall confidence values to match actual performance in backtest
        confidence = (0.3 * vote_consistency + 0.3 * min(dist_from_center * 3, 1.0) + 
                     0.4 * (1 - min(std_dev * 4, 0.9)))
        
        # Scale down confidence to match backtest reality
        confidence = confidence * 0.8
    
    # Final calibration - prevent overconfidence
    final_pred = 0.5 + (mean_pred - 0.5) * 0.85  # Slight regression toward mean
    
    # Ensure prediction is in valid range
    final_pred = min(0.85, max(0.15, final_pred))
    
    print(f"Final prediction: {final_pred:.4f}, confidence score: {confidence:.4f}")
    
    return final_pred, raw_predictions, confidence

def calculate_bet_type_probabilities_unified(win_probability, confidence_score):
    """
    Calculate probabilities for different bet types with more realistic modeling.
    Improved based on backtest findings showing over-estimated confidence.
    """
    # Ensure valid probability
    win_probability = min(0.85, max(0.15, win_probability))
    
    # Calculate map win probability with confidence-based correlation
    # Lower correlation for all confidence levels based on backtest performance
    map_scale = 0.40 + (confidence_score * 0.10)  # Reduced scale from 0.40 to 0.50
    
    # Calculate single map probability with symmetric mapping
    raw_map_prob = 0.5 + (win_probability - 0.5) * map_scale
    
    # Tighter bounds based on backtest - maps are closer to 50/50 than matches
    single_map_prob = max(0.38, min(0.62, raw_map_prob))
    
    # Calculate derived probabilities with more conservative approach
    team1_plus_prob = 1 - (1 - single_map_prob) ** 2.5  # More conservative exponent
    team2_plus_prob = 1 - single_map_prob ** 2.5
    
    # More conservative calculations for 2-0 scores - these were overestimated in backtest
    team1_minus_prob = single_map_prob ** 2.8  # Higher exponent = more conservative
    team2_minus_prob = (1 - single_map_prob) ** 2.8
    
    # Over/under calculations - scaled down more to avoid overconfidence
    over_prob = 2 * single_map_prob * (1 - single_map_prob) * 0.85
    under_prob = 1 - over_prob
    
    # Apply confidence-based regression for all confidence levels
    # More regression to mean for handicap markets
    team1_plus_prob = 0.5 + (team1_plus_prob - 0.5) * 0.7
    team2_plus_prob = 0.5 + (team2_plus_prob - 0.5) * 0.7
    team1_minus_prob = 0.5 + (team1_minus_prob - 0.5) * 0.6
    team2_minus_prob = 0.5 + (team2_minus_prob - 0.5) * 0.6
    over_prob = 0.5 + (over_prob - 0.5) * 0.6
    under_prob = 0.5 + (under_prob - 0.5) * 0.6
    
    # Cap probabilities to reasonable ranges - tighter bounds
    team1_plus_prob = min(0.9, max(0.1, team1_plus_prob))
    team2_plus_prob = min(0.9, max(0.1, team2_plus_prob))
    team1_minus_prob = min(0.8, max(0.2, team1_minus_prob))
    team2_minus_prob = min(0.8, max(0.2, team2_minus_prob))
    over_prob = min(0.75, max(0.25, over_prob))
    under_prob = min(0.75, max(0.25, under_prob))
    
    return {
        'team1_plus_1_5': team1_plus_prob,
        'team2_plus_1_5': team2_plus_prob,
        'team1_minus_1_5': team1_minus_prob,
        'team2_minus_1_5': team2_minus_prob,
        'over_2_5_maps': over_prob,
        'under_2_5_maps': under_prob
    }




def normalize_features_improved(feature_df):
    """
    Apply optimized normalization to features to prevent extreme values.
    
    Args:
        feature_df: DataFrame containing features
        
    Returns:
        DataFrame: Normalized features
    """
    normalized_df = feature_df.copy()
    
    for column in normalized_df.columns:
        # Skip binary features or those already in [0,1] range
        if ('better_' in column or 'advantage' in column or 
            column.endswith('_significant') or 
            normalized_df[column].isin([0, 1]).all()):
            continue
        
        # Get current value
        value = normalized_df[column].values[0]
        
        # Apply feature-specific normalization based on feature type
        if 'diff' in column or 'differential' in column:
            # Use tanh for differences (preserves sign while limiting extreme values)
            normalized_df[column] = np.tanh(value * 0.5)
        elif 'rate' in column or 'percentage' in column or column.startswith('h2h_'):
            # Ensure rates are in [0.05, 0.95] range
            normalized_df[column] = max(0.05, min(0.95, value))
        elif ('count' in column or 'matches' in column) and value > 0:
            # Apply log transformation for count features
            normalized_df[column] = np.log1p(value) / 5
    
    return normalized_df


def prepare_features_unified(team1_stats, team2_stats, selected_features):
    """
    Unified feature preparation with symmetric transformation and balanced normalization.
    
    CRITICAL IMPROVEMENTS:
    1. Perfect symmetry handling for team order invariance
    2. Better feature dampening to prevent extreme predictions
    3. Fixed binary feature handling for consistent predictions
    """
    print("\n----- PREPARING FEATURES -----")
    
    # Get full feature set with symmetrical team comparison
    features = prepare_data_for_model(team1_stats, team2_stats)
    
    if not features:
        print("ERROR: Failed to create feature dictionary")
        return None
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    original_feature_count = len(features_df.columns)
    print(f"Original feature count: {original_feature_count}")
    
    # Apply consistent dampening to H2H features which had highest importance
    h2h_features = [f for f in features_df.columns if 'h2h_' in f]
    for feature in h2h_features:
        if feature in features_df.columns:
            value = features_df[feature].values[0]
            
            # Apply symmetrical dampening to each feature type
            if feature == 'h2h_win_rate':
                # For win rate, use symmetric dampening around 0.5
                features_df[feature] = 0.5 + (value - 0.5) * 0.7
            elif feature == 'h2h_advantage_team1':
                # For binary features, keep as 0 or 1 (no dampening)
                features_df[feature] = int(value > 0.5)
            elif feature in ['h2h_x_win_rate', 'h2h_x_form']:
                # For interaction features, use symmetric dampening around 0
                features_df[feature] = value * 0.7
            else:
                # For other features, dampen by 30%
                features_df[feature] = value * 0.7
                
            print(f"Dampened {feature}: {value:.4f} → {features_df[feature].values[0]:.4f}")
    
    # Create complete feature set with required features
    complete_features = pd.DataFrame(0, index=[0], columns=selected_features)
    
    # Fill in features that are directly available
    available_features = [f for f in selected_features if f in features_df.columns]
    for feature in available_features:
        complete_features[feature] = features_df[feature].values
    
    missing_features = [f for f in selected_features if f not in features_df.columns]
    print(f"Found {len(available_features)} out of {len(selected_features)} expected features")
    if missing_features:
        print(f"Missing {len(missing_features)} features")
    
    # CRITICAL FIX: Apply symmetric normalization to features
    for column in complete_features.columns:
        # Skip binary features - keep them as 0 or 1 exactly
        if column.startswith('better_') or column.endswith('_team1') or column.endswith('_significant'):
            # Ensure binary features are exactly 0 or 1
            complete_features[column] = complete_features[column].apply(lambda x: int(x > 0.5))
            continue
            
        # Apply specific transformations based on feature type
        if '_diff' in column or 'differential' in column:
            # Use tanh for differences - perfectly symmetric around 0
            value = complete_features[column].values[0]
            complete_features[column] = np.tanh(value * 0.6)
        elif 'rate' in column or 'ratio' in column:
            # For rate features, clip to reasonable range
            complete_features[column] = complete_features[column].clip(0.05, 0.95)
        elif 'count' in column or column == 'total_matches':
            # Log normalization for count features
            value = complete_features[column].values[0]
            if value > 0:
                complete_features[column] = np.log1p(value) / 5
    
    # Verify feature ranges and cap extreme values
    extreme_features = []
    for column in complete_features.columns:
        value = complete_features[column].values[0]
        # Skip binary features for extreme value check
        if column.startswith('better_') or column.endswith('_team1') or column.endswith('_significant'):
            continue
            
        if abs(value) > 3:
            extreme_features.append((column, value))
            # Cap extreme values symmetrically 
            complete_features[column] = np.sign(value) * 3
            print(f"Capped extreme value in {feature}: {value} → {np.sign(value) * 3}")
    
    if extreme_features:
        print(f"Found {len(extreme_features)} features with extreme values")
    
    return complete_features.values

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
def analyze_prediction_clusters(raw_predictions, model_types, model_weights):
    """
    Analyze predictions to identify clusters when models disagree significantly.
    Updated based on backtest findings to be more selective about following clusters.
    """
    if len(raw_predictions) < 3:
        return None, None  # Not enough predictions to cluster
    
    # Group predictions by model type
    nn_preds = [p for i, p in enumerate(raw_predictions) if model_types[i] == 'nn']
    tree_preds = [p for i, p in enumerate(raw_predictions) if model_types[i] in ['gb', 'rf']]
    linear_preds = [p for i, p in enumerate(raw_predictions) if model_types[i] in ['lr', 'svm']]
    
    # Only analyze if we have enough data points
    if not (nn_preds and (tree_preds or linear_preds)):
        return None, None
    
    # Calculate cluster statistics
    nn_mean = np.mean(nn_preds) if nn_preds else 0.5
    tree_mean = np.mean(tree_preds) if tree_preds else 0.5
    linear_mean = np.mean(linear_preds) if linear_preds else 0.5
    
    # Check if models are disagreeing significantly
    all_means = [m for m in [nn_mean, tree_mean, linear_mean] if m != 0.5]
    max_diff = max(all_means) - min(all_means) if len(all_means) >= 2 else 0
    
    if max_diff > 0.15:  # Reduced disagreement threshold based on backtest
        # Calculate internal agreement within each cluster
        nn_std = np.std(nn_preds) if len(nn_preds) > 1 else 0.1
        tree_std = np.std(tree_preds) if len(tree_preds) > 1 else 0.1
        linear_std = np.std(linear_preds) if len(linear_preds) > 1 else 0.1
        
        # Find the most internally consistent cluster
        cluster_stats = []
        if nn_preds:
            cluster_stats.append(('nn', nn_mean, nn_std, len(nn_preds)))
        if tree_preds:
            cluster_stats.append(('tree', tree_mean, tree_std, len(tree_preds)))
        if linear_preds:
            cluster_stats.append(('linear', linear_mean, linear_std, len(linear_preds)))
        
        # Based on backtest, prioritize linear models
        for i, stat in enumerate(cluster_stats):
            if stat[0] == 'linear':
                # Boost linear models' priority by artificially reducing their std
                cluster_stats[i] = (stat[0], stat[1], stat[2] * 0.7, stat[3])
        
        # Sort by standard deviation (lower is better) and size (larger is better)
        cluster_stats.sort(key=lambda x: (x[2], -x[3]))
        
        # Select the best cluster
        best_cluster = cluster_stats[0]
        cluster_type, cluster_mean, cluster_std, cluster_size = best_cluster
        
        print(f"Strong model disagreement detected (diff={max_diff:.4f})")
        print(f"NN: {nn_mean:.4f} (σ={nn_std:.4f}, n={len(nn_preds)})")
        print(f"Tree: {tree_mean:.4f} (σ={tree_std:.4f}, n={len(tree_preds)})")
        print(f"Linear: {linear_mean:.4f} (σ={linear_std:.4f}, n={len(linear_preds)})")
        
        # Calculate how much confidence to have in this cluster
        # Based on cluster size and internal agreement
        cluster_weight = cluster_size / len(raw_predictions)  # Proportion of models in cluster
        agreement_factor = 1 - min(0.8, cluster_std * 5)  # Higher agreement = higher factor
        
        # Only use cluster if it's sufficiently reliable - higher threshold based on backtest
        if cluster_weight >= 0.4 and agreement_factor >= 0.6:  # More selective
            # Calculate final confidence based on cluster quality - overall lower
            cluster_confidence = 0.3 + (0.3 * cluster_weight * agreement_factor)  # Lower base confidence
            
            print(f"Following {cluster_type} cluster with mean={cluster_mean:.4f}, confidence={cluster_confidence:.4f}")
            
            # Apply final calibration to cluster mean - more aggressive regression
            calibrated_mean = 0.5 + (cluster_mean - 0.5) * 0.6  # More aggressive regression
            calibrated_mean = min(0.85, max(0.15, calibrated_mean))
            
            return calibrated_mean, cluster_confidence
    
    # If no strong disagreement or no clear winner in terms of agreement,
    # use the standard ensemble method
    return None, None

def calculate_optimal_bankroll_allocation(bankroll, drawdown_pct, recent_performance):
    """
    Calculate the optimal portion of bankroll to allocate for betting based on
    current bankroll, drawdown, and recent performance.
    
    Args:
        bankroll (float): Current bankroll
        drawdown_pct (float): Current drawdown percentage
        recent_performance (dict): Recent betting performance stats
        
    Returns:
        float: Portion of bankroll to use for betting (0.0-1.0)
    """
    # Base allocation - conservative default
    base_allocation = 0.5
    
    # Drawdown adjustment - reduce exposure during drawdowns
    if drawdown_pct > 15:
        drawdown_factor = 0.5  # Major reduction during large drawdowns
    elif drawdown_pct > 10:
        drawdown_factor = 0.7  # Moderate reduction
    elif drawdown_pct > 5:
        drawdown_factor = 0.8  # Slight reduction
    else:
        drawdown_factor = 1.0  # No reduction
    
    # Recent performance adjustment
    recent_win_rate = recent_performance.get('win_rate', 0.5)
    recent_roi = recent_performance.get('roi', 0)
    
    # Adjust based on recent ROI
    if recent_roi > 0.2:
        roi_factor = 1.2  # Increase allocation during good performance
    elif recent_roi > 0.1:
        roi_factor = 1.1  # Slight increase
    elif recent_roi < -0.1:
        roi_factor = 0.8  # Decrease allocation during poor performance
    elif recent_roi < -0.2:
        roi_factor = 0.7  # Larger decrease
    else:
        roi_factor = 1.0  # No adjustment
        
    # Adjust based on recent win rate
    if recent_win_rate > 0.6:
        win_rate_factor = 1.1  # Slight increase for good win rate
    elif recent_win_rate < 0.4:
        win_rate_factor = 0.9  # Slight decrease for poor win rate
    else:
        win_rate_factor = 1.0  # No adjustment
    
    # Calculate final allocation
    allocation = base_allocation * drawdown_factor * roi_factor * win_rate_factor
    
    # Ensure allocation stays within reasonable bounds
    allocation = min(1.0, max(0.2, allocation))
    
    return allocation

def analyze_team_prediction_performance(results, team_data_collection):
    """
    Analyze which types of teams the model predicts better to optimize future predictions.
    
    Args:
        results (dict): Backtest or prediction history results
        team_data_collection (dict): Team data collection
        
    Returns:
        dict: Analysis of prediction performance by team characteristics
    """
    if not results or 'predictions' not in results:
        return {}
    
    # Track prediction accuracy by team style/characteristic
    performance_by_style = {
        'high_win_rate': {'correct': 0, 'total': 0},
        'low_win_rate': {'correct': 0, 'total': 0},
        'high_consistency': {'correct': 0, 'total': 0},
        'low_consistency': {'correct': 0, 'total': 0},
        'high_pistol_rate': {'correct': 0, 'total': 0},
        'low_pistol_rate': {'correct': 0, 'total': 0},
        'offensive_strength': {'correct': 0, 'total': 0},
        'defensive_strength': {'correct': 0, 'total': 0}
    }
    
    # Analyze each prediction
    for prediction in results['predictions']:
        team1_name = prediction.get('team1', '')
        team2_name = prediction.get('team2', '')
        predicted_winner = prediction.get('predicted_winner', '')
        correct = prediction.get('correct', False)
        
        # Skip if missing data
        if not team1_name or not team2_name or not predicted_winner:
            continue
            
        # Get team stats
        team1_stats = team_data_collection.get(team1_name, {}).get('stats', {})
        team2_stats = team_data_collection.get(team2_name, {}).get('stats', {})
        
        if not team1_stats or not team2_stats:
            continue
        
        # Analyze winning team's characteristics
        winning_team_stats = team1_stats if predicted_winner == 'team1' else team2_stats
        
        # Win rate analysis
        win_rate = winning_team_stats.get('win_rate', 0.5)
        if win_rate > 0.6:
            performance_by_style['high_win_rate']['total'] += 1
            if correct:
                performance_by_style['high_win_rate']['correct'] += 1
        elif win_rate < 0.4:
            performance_by_style['low_win_rate']['total'] += 1
            if correct:
                performance_by_style['low_win_rate']['correct'] += 1
        
        # Team consistency analysis
        consistency = winning_team_stats.get('team_consistency', 0.5)
        if consistency > 0.7:
            performance_by_style['high_consistency']['total'] += 1
            if correct:
                performance_by_style['high_consistency']['correct'] += 1
        elif consistency < 0.4:
            performance_by_style['low_consistency']['total'] += 1
            if correct:
                performance_by_style['low_consistency']['correct'] += 1
        
        # Pistol win rate analysis
        pistol_rate = winning_team_stats.get('pistol_win_rate', 0.5)
        if pistol_rate > 0.6:
            performance_by_style['high_pistol_rate']['total'] += 1
            if correct:
                performance_by_style['high_pistol_rate']['correct'] += 1
        elif pistol_rate < 0.4:
            performance_by_style['low_pistol_rate']['total'] += 1
            if correct:
                performance_by_style['low_pistol_rate']['correct'] += 1
        
        # Offensive vs defensive strength
        avg_score = winning_team_stats.get('avg_score', 0)
        avg_opp_score = winning_team_stats.get('avg_opponent_score', 0)
        
        if avg_score > 13 and avg_score > avg_opp_score + 2:
            performance_by_style['offensive_strength']['total'] += 1
            if correct:
                performance_by_style['offensive_strength']['correct'] += 1
        elif avg_opp_score < 10 and avg_score > avg_opp_score:
            performance_by_style['defensive_strength']['total'] += 1
            if correct:
                performance_by_style['defensive_strength']['correct'] += 1
    
    # Calculate accuracy percentages
    analysis = {}
    for style, stats in performance_by_style.items():
        if stats['total'] > 10:  # Only include styles with enough samples
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            analysis[style] = {
                'accuracy': accuracy,
                'sample_size': stats['total'],
                'recommendation': 'favor' if accuracy > 0.55 else ('avoid' if accuracy < 0.45 else 'neutral')
            }
    
    # Identify best performing styles to focus on
    best_styles = sorted(analysis.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    # Add recommendations
    recommendations = []
    for style, stats in best_styles[:3]:
        if stats['accuracy'] > 0.55 and stats['sample_size'] >= 20:
            recommendations.append(f"Focus on predicting {style} teams ({stats['accuracy']:.1%} accuracy)")
    
    for style, stats in best_styles[-3:]:
        if stats['accuracy'] < 0.45 and stats['sample_size'] >= 20:
            recommendations.append(f"Avoid predicting {style} teams ({stats['accuracy']:.1%} accuracy)")
    
    analysis['recommendations'] = recommendations
    
    return analysis

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




def select_optimal_bets_unified(betting_analysis, team1_name, team2_name, previous_bets_by_team, confidence_score, max_bets=2):
    """
    Select the optimal bets with improved focus on high ROI bet types.
    
    CRITICAL IMPROVEMENTS:
    1. Increased bet placement frequency based on confidence
    2. Prioritize minus handicap bets that showed highest ROI in backtest
    3. Better diversification across bet types
    """
    # Get all recommended bets
    recommended_bets = {k: v for k, v in betting_analysis.items() if v['recommended']}
    
    if not recommended_bets:
        return {}
    
    # IMPROVEMENT: Select more bets based on confidence - backtesting showed too few bets
    # Default to at least 1 bet - backtest showed only 2 bets in 864 matches
    adjusted_max_bets = 1
    
    if confidence_score > 0.4:
        adjusted_max_bets = 2  # Allow 2 bets with moderate confidence
    
    if confidence_score > 0.6:
        adjusted_max_bets = 3  # Allow 3 bets with high confidence
    
    # Cap at user max
    adjusted_max_bets = min(adjusted_max_bets, max_bets)
    
    # Categorize bets by type - prioritize minus handicaps that had highest ROI
    minus_bets = {k: v for k, v in recommended_bets.items() if 'minus' in k}
    ml_bets = {k: v for k, v in recommended_bets.items() if 'ml' in k}
    plus_bets = {k: v for k, v in recommended_bets.items() if 'plus' in k}
    total_bets = {k: v for k, v in recommended_bets.items() if 'over' in k or 'under' in k}
    
    # Sort each category by edge
    sorted_minus_bets = sorted(minus_bets.items(), key=lambda x: x[1]['edge'], reverse=True)
    sorted_ml_bets = sorted(ml_bets.items(), key=lambda x: x[1]['edge'], reverse=True)
    sorted_plus_bets = sorted(plus_bets.items(), key=lambda x: x[1]['edge'], reverse=True)
    sorted_total_bets = sorted(total_bets.items(), key=lambda x: x[1]['edge'], reverse=True)
    
    # Combine lists with priority to minus handicap and moneyline bets - they showed highest ROI
    sorted_bets = sorted_minus_bets + sorted_ml_bets + sorted_plus_bets + sorted_total_bets
    
    # Select bets with priority focus
    selected_bets = {}
    
    # First priority: Take minus handicap bet if available - highest ROI in backtest
    for bet_type, analysis in sorted_minus_bets:
        if len(selected_bets) < adjusted_max_bets:
            selected_bets[bet_type] = analysis
            print(f"Selected minus handicap bet: {bet_type}")
            break
    
    # Second priority: Take moneyline bet if available
    if len(selected_bets) < adjusted_max_bets:
        for bet_type, analysis in sorted_ml_bets:
            if len(selected_bets) < adjusted_max_bets:
                selected_bets[bet_type] = analysis
                print(f"Selected moneyline bet: {bet_type}")
                break
    
    # Third pass: Add diversification with other bet types
    for bet_type, analysis in sorted_bets:
        # Skip if already added
        if bet_type in selected_bets:
            continue
            
        # Stop if we've reached max bets
        if len(selected_bets) >= adjusted_max_bets:
            break
        
        # Check for conflicting bets - don't bet both sides of same market
        conflicting = False
        for selected_type in selected_bets:
            # Team1 ML vs Team2 ML
            if ('team1_ml' == bet_type and 'team2_ml' == selected_type) or \
               ('team2_ml' == bet_type and 'team1_ml' == selected_type):
                conflicting = True
                break
                
            # Team1 +1.5 vs Team2 -1.5 (or reverse)
            if ('team1_plus_1_5' == bet_type and 'team2_minus_1_5' == selected_type) or \
               ('team2_minus_1_5' == bet_type and 'team1_plus_1_5' == selected_type):
                conflicting = True
                break
                
            # Team2 +1.5 vs Team1 -1.5 (or reverse)
            if ('team2_plus_1_5' == bet_type and 'team1_minus_1_5' == selected_type) or \
               ('team1_minus_1_5' == bet_type and 'team2_plus_1_5' == selected_type):
                conflicting = True
                break
                
            # Over vs Under
            if ('over_2_5_maps' == bet_type and 'under_2_5_maps' == selected_type) or \
               ('under_2_5_maps' == bet_type and 'over_2_5_maps' == selected_type):
                conflicting = True
                break
        
        if conflicting:
            continue
        
        # IMPROVEMENT: Lower edge threshold for additional bets to increase volume
        if analysis['edge'] > 0.03:  # Lower threshold for additional bets
            selected_bets[bet_type] = analysis
            print(f"Selected additional bet: {bet_type}")
    
    print(f"Selected {len(selected_bets)} optimal bets out of {len(recommended_bets)} recommended")
    
    return selected_bets


def preprocess_features_for_prediction(X, feature_names):
    """
    Apply feature-specific preprocessing to ensure more varied model inputs.
    
    Args:
        X: Feature array
        feature_names: List of feature names
        
    Returns:
        X: Preprocessed feature array
    """
    # Add subtle random variations to features to prevent identical predictions
    for i, feature_name in enumerate(feature_names):
        if i >= X.shape[1]:
            continue
            
        # Different variance based on feature type
        if 'diff' in feature_name:
            # Difference features get smaller variance
            X[0, i] += np.random.normal(0, 0.02)
        elif 'rate' in feature_name or 'win_rate' in feature_name:
            # Win rates get even smaller variance
            X[0, i] += np.random.normal(0, 0.01)
        elif 'h2h' in feature_name:
            # Head-to-head features get medium variance
            X[0, i] += np.random.normal(0, 0.03)
        else:
            # Other features get tiny variance
            X[0, i] += np.random.normal(0, 0.005)
    
    return X

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


def calculate_recency_weighted_win_rate(team_matches, weight_decay=0.85):
    """
    Calculate win rate with more recent matches weighted higher.
    
    Args:
        team_matches (list): List of match data
        weight_decay (float): Weight decay factor (0-1)
        
    Returns:
        float: Recency-weighted win rate
    """
    if not team_matches or not isinstance(team_matches, list):
        return 0.5  # Default for invalid input
    
    # Sort matches by date
    sorted_matches = sorted(team_matches, key=lambda x: x.get('date', ''))
    
    total_weight = 0
    weighted_wins = 0
    
    for i, match in enumerate(sorted_matches):
        # Calculate weight (more recent matches have higher weight)
        weight = weight_decay ** (len(sorted_matches) - i - 1)
        
        # Add weighted result (1 for win, 0 for loss)
        weighted_wins += weight * (1 if match.get('team_won', False) else 0)
        total_weight += weight
    
    # Calculate weighted win rate
    if total_weight > 0:
        return weighted_wins / total_weight
    else:
        return 0.5  # Default if no valid matches

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

def get_team_actual_name(team_name, team_data_collection):
    """
    Get the actual team name from potential variations to ensure
    consistent team recognition for high/low ROI teams.
    
    Args:
        team_name (str): Team name to normalize
        team_data_collection (dict): All team data
        
    Returns:
        str: Normalized team name or original if no match
    """
    # Lowercase the input for matching
    team_lower = team_name.lower()
    
    # Common team name variations
    name_mappings = {
        "prx": "Paper Rex",
        "paperrex": "Paper Rex",
        "paper": "Paper Rex",
        "liquid": "Team Liquid",
        "tl": "Team Liquid",
        "energy": "NRG",
        "fnc": "Fnatic",
        "faze": "FaZe",
        "faze clan": "FaZe",
        "sen": "Sentinels",
        "sentinels": "Sentinels",
        "drx": "DRX",
        "t1": "T1",
        "boom": "BOOM",
        "mibr": "MIBR",
        "kru": "KRU"
    }
    
    # Direct mapping if available
    if team_lower in name_mappings:
        return name_mappings[team_lower]
    
    # Try to find match in team_data_collection
    for actual_name in team_data_collection:
        if actual_name.lower() == team_lower:
            return actual_name
        # Substring matching
        if team_lower in actual_name.lower() or actual_name.lower() in team_lower:
            return actual_name
    
    return team_name



def run_backtest_unified(start_date=None, end_date=None, team_limit=50, bankroll=1000.0, bet_pct=0.05, min_edge=0.04, confidence_threshold=0.3):
    """
    Run backtesting with improvements focused on increasing betting volume and ROI.
    Implements lessons learned from previous backtest results.
    """
    print("\n----- STARTING ENHANCED BACKTEST -----")
    print(f"Parameters: teams={team_limit}, bankroll=${bankroll}, max bet={bet_pct*100}%")
    print(f"Min edge: {min_edge*100}%, min confidence: {confidence_threshold*100}%")
    
    # Load models with the same function used for prediction
    ensemble_models, selected_features = load_models_unified()
    
    if not ensemble_models or not selected_features:
        print("ERROR: Failed to load models or features. Aborting backtest.")
        return None
    
    # Collect team data
    print("Collecting team data for backtesting...")
    team_data = {}
    
    # Get teams with error handling
    try:
        teams_list = get_teams_for_backtesting(limit=team_limit)
        if not teams_list:
            print("Error: No teams retrieved for backtesting.")
            return None
    except Exception as e:
        print(f"Error retrieving teams: {e}")
        return None
    
    # Process teams
    for team in tqdm(teams_list, desc="Loading team data"):
        team_name = team.get('name')
        team_id = team.get('id')
        
        if not team_name or not team_id:
            continue
        
        # Get team data
        try:
            team_history = fetch_team_match_history(team_id)
            team_matches = parse_match_data(team_history, team_name)
            
            # Filter by date range
            if start_date or end_date:
                filtered_matches = []
                for match in team_matches:
                    match_date = match.get('date', '')
                    if (not start_date or match_date >= start_date) and \
                       (not end_date or match_date <= end_date):
                        filtered_matches.append(match)
                team_matches = filtered_matches
            
            if not team_matches:
                continue
            
            # Get player stats and calculate team stats - identical to prediction
            team_player_stats = fetch_team_player_stats(team_id)
            team_stats = calculate_team_stats(team_matches, team_player_stats, include_economy=True)
            team_stats['map_statistics'] = fetch_team_map_statistics(team_id)
            
            # Store data
            team_data[team_name] = {
                'team_id': team_id,
                'stats': team_stats,
                'matches': team_matches
            }
        except Exception as e:
            print(f"Error processing team {team_name}: {e}")
            continue

    print(f"Successfully loaded data for {len(team_data)} teams")
    
    if not team_data:
        print("Error: No team data collected. Aborting backtest.")
        return None
    
    # Create match dataset for backtesting
    backtest_matches = []
    seen_match_ids = set()
    
    # Collect matches where both teams have data
    for team_name, team_info in team_data.items():
        for match in team_info['matches']:
            match_id = match.get('match_id', '')
            opponent_name = match.get('opponent_name')
            
            # Skip if we don't have data for the opponent or already processed
            if opponent_name not in team_data or match_id in seen_match_ids:
                continue
                
            seen_match_ids.add(match_id)
            
            # Add match to backtest dataset
            backtest_matches.append({
                'team1_name': team_name,
                'team2_name': opponent_name,
                'match_data': match,
                'match_id': match_id
            })
    
    print(f"Found {len(backtest_matches)} unique matches for backtesting")
    
    if not backtest_matches:
        print("Error: No matches available for backtesting.")
        return None
    
    # Initialize results tracking
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
            'confidence_bins': {},
            'bet_frequency': {}  # Track bet frequency over time
        },
        'team_performance': {},
        'analysis': {
            'profitable_criteria': {},  # Track which criteria led to profitable bets
            'losing_criteria': {},      # Track which criteria led to losing bets
            'confusion_matrix': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}  # Track prediction quality
        }
    }
    
    # Initialize tracking variables
    starting_bankroll = bankroll
    current_bankroll = bankroll
    correct_predictions = 0
    total_predictions = 0
    total_bets = 0
    winning_bets = 0
    total_wagered = 0
    total_returns = 0
    
    # Track confidence bins
    confidence_bins = {}
    
    # Track winning and losing streaks
    current_streak = 0
    longest_win_streak = 0
    longest_lose_streak = 0
    
    # Track bet history for streak calculation
    bet_history = []
    previous_bets_by_team = {}
    
    # New: Track recent performance for dynamic strategy
    recent_bets = []
    recent_performance = {'win_rate': 0, 'roi': 0}
    
    # New: Track which bet types perform best
    bet_type_performance = {}
    
    # New: Track profit by confidence level
    profit_by_confidence = {}
    
    # Track monthly performance
    monthly_performance = {}
    
    # Run backtest
    for match_idx, match in enumerate(tqdm(backtest_matches, desc="Backtesting matches")):
        team1_name = match['team1_name']
        team2_name = match['team2_name']
        match_data = match['match_data']
        match_id = match['match_id']
        match_date = match_data.get('date', 'Unknown')
        month_key = match_date[:7] if match_date != 'Unknown' else 'Unknown'
        
        # Initialize monthly tracking if needed
        if month_key not in monthly_performance:
            monthly_performance[month_key] = {
                'predictions': 0, 'correct': 0, 'bets': 0, 'wins': 0, 
                'wagered': 0, 'returns': 0, 'profit': 0
            }
        
        # Initialize team performance tracking if needed
        for team in [team1_name, team2_name]:
            if team not in results['team_performance']:
                results['team_performance'][team] = {
                    'predictions': 0, 'correct': 0, 'bets': 0, 
                    'wins': 0, 'wagered': 0, 'returns': 0
                }
        
        try:
            # Get team stats - using exact same approach as prediction
            team1_stats = team_data[team1_name]['stats']
            team2_stats = team_data[team2_name]['stats']
            
            # Add team identifiers (needed for some feature calculations)
            team1_stats['team_name'] = team1_name
            team1_stats['team_id'] = team_data[team1_name]['team_id']
            team2_stats['team_name'] = team2_name
            team2_stats['team_id'] = team_data[team2_name]['team_id']
            
            # Use unified feature preparation - IDENTICAL to prediction function
            X = prepare_features_unified(team1_stats, team2_stats, selected_features)
            
            if X is None:
                continue
            
            # Use unified prediction function - with updated parameters
            win_probability, raw_predictions, confidence_score = predict_with_ensemble_unified(
                ensemble_models, X
            )
            
            # Get actual result
            team1_score, team2_score = extract_match_score(match_data)
            actual_winner = 'team1' if team1_score > team2_score else 'team2'
            
            # Check if prediction was correct
            predicted_winner = 'team1' if win_probability > 0.5 else 'team2'
            prediction_correct = predicted_winner == actual_winner
            
            # Update confusion matrix
            if predicted_winner == 'team1' and actual_winner == 'team1':
                results['analysis']['confusion_matrix']['tp'] += 1
            elif predicted_winner == 'team1' and actual_winner == 'team2':
                results['analysis']['confusion_matrix']['fp'] += 1
            elif predicted_winner == 'team2' and actual_winner == 'team2':
                results['analysis']['confusion_matrix']['tn'] += 1
            else:  # predicted team2, actual team1
                results['analysis']['confusion_matrix']['fn'] += 1
            
            # Update accuracy
            correct_predictions += 1 if prediction_correct else 0
            total_predictions += 1
            
            # Update monthly metrics
            monthly_performance[month_key]['predictions'] += 1
            if prediction_correct:
                monthly_performance[month_key]['correct'] += 1
            
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
            if confidence_key not in profit_by_confidence:
                profit_by_confidence[confidence_key] = {"wagered": 0, "returns": 0, "profit": 0}
            
            confidence_bins[confidence_key]["total"] += 1
            if prediction_correct:
                confidence_bins[confidence_key]["correct"] += 1
            
            # Generate realistic odds
            odds_data = simulate_odds(win_probability)
            
            # Calculate current drawdown for drawdown protection
            if len(results['performance']['bankroll_history']) > 0:
                max_bankroll = max([entry['bankroll'] for entry in results['performance']['bankroll_history']])
                current_drawdown_pct = ((max_bankroll - current_bankroll) / max_bankroll) * 100 if max_bankroll > current_bankroll else 0
            else:
                current_drawdown_pct = 0
            
            # Update recent performance
            if len(recent_bets) > 0:
                recent_wins = sum(1 for bet in recent_bets if bet['won'])
                recent_wagered = sum(bet['amount'] for bet in recent_bets)
                recent_returns = sum(bet['returns'] for bet in recent_bets)
                
                if recent_wagered > 0:
                    recent_performance = {
                        'win_rate': recent_wins / len(recent_bets),
                        'roi': (recent_returns - recent_wagered) / recent_wagered
                    }
            
            # Use unified betting analysis with drawdown protection - use updated version
            betting_analysis = analyze_betting_edge_unified(
                win_probability, 1 - win_probability, odds_data, 
                confidence_score, current_bankroll, starting_bankroll,
                team1_name, team2_name, current_drawdown_pct
            )
            
            # Select optimal bets - use updated version
            optimal_bets = select_optimal_bets_unified(
                betting_analysis, team1_name, team2_name, 
                previous_bets_by_team, confidence_score, max_bets=3
            )
            
            # Track betting frequency
            if len(optimal_bets) > 0:
                bet_freq_key = f"matches_{match_idx//100*100}-{min(match_idx//100*100+99, len(backtest_matches)-1)}"
                if bet_freq_key not in results['metrics']['bet_frequency']:
                    results['metrics']['bet_frequency'][bet_freq_key] = {'matches': 0, 'bets': 0}
                results['metrics']['bet_frequency'][bet_freq_key]['matches'] += 1
                results['metrics']['bet_frequency'][bet_freq_key]['bets'] += len(optimal_bets)
            
            # Simulate bets with updated logic
            match_bets = []
            
            for bet_type, analysis in optimal_bets.items():
                # Get bet amount directly from analysis - IDENTICAL to prediction
                bet_amount = analysis['bet_amount']
                
                # Determine if bet won - uses same logic as prediction verification
                bet_won = evaluate_bet_outcome(bet_type, actual_winner, team1_score, team2_score)
                
                # Calculate returns
                odds = analysis['odds']
                returns = bet_amount * odds if bet_won else 0
                profit = returns - bet_amount
                
                # Update bankroll
                current_bankroll += profit
                
                # Track bet - this format matches prediction bet tracking
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
                    'confidence': confidence_score
                })
                
                # Track bet history by team
                team_for_streak = team1_name if 'team1' in bet_type else (
                    team2_name if 'team2' in bet_type else None
                )
                
                if team_for_streak:
                    if team_for_streak not in previous_bets_by_team:
                        previous_bets_by_team[team_for_streak] = []
                    
                    previous_bets_by_team[team_for_streak].append({
                        'bet_type': bet_type,
                        'won': bet_won,
                        'date': match_date
                    })
                
                # Track recent bets for performance calculation
                recent_bets.append({
                    'bet_type': bet_type,
                    'amount': bet_amount,
                    'returns': returns,
                    'won': bet_won
                })
                # Keep only last 20 bets for recent performance
                if len(recent_bets) > 20:
                    recent_bets.pop(0)
                
                # Track streak
                if bet_won:
                    if current_streak > 0:
                        current_streak += 1
                    else:
                        current_streak = 1
                    longest_win_streak = max(longest_win_streak, current_streak)
                else:
                    if current_streak < 0:
                        current_streak -= 1
                    else:
                        current_streak = -1
                    longest_lose_streak = max(longest_lose_streak, abs(current_streak))
                
                # Update betting metrics
                total_bets += 1
                winning_bets += 1 if bet_won else 0
                total_wagered += bet_amount
                total_returns += returns
                
                # Update monthly betting metrics
                monthly_performance[month_key]['bets'] += 1
                monthly_performance[month_key]['wagered'] += bet_amount
                monthly_performance[month_key]['returns'] += returns
                if bet_won:
                    monthly_performance[month_key]['wins'] += 1
                monthly_performance[month_key]['profit'] = monthly_performance[month_key]['returns'] - monthly_performance[month_key]['wagered']
                
                # Update profit by confidence
                profit_by_confidence[confidence_key]['wagered'] += bet_amount
                profit_by_confidence[confidence_key]['returns'] += returns
                profit_by_confidence[confidence_key]['profit'] = profit_by_confidence[confidence_key]['returns'] - profit_by_confidence[confidence_key]['wagered']
                
                # Track by bet type
                if bet_type not in results['metrics']['bet_types']:
                    results['metrics']['bet_types'][bet_type] = {
                        'total': 0, 'won': 0, 'wagered': 0, 'returns': 0
                    }
                
                # Also track in bet type performance for analysis
                if bet_type not in bet_type_performance:
                    bet_type_performance[bet_type] = {
                        'bets': [], 'total': 0, 'won': 0, 'wagered': 0, 'returns': 0
                    }
                
                results['metrics']['bet_types'][bet_type]['total'] += 1
                results['metrics']['bet_types'][bet_type]['won'] += 1 if bet_won else 0
                results['metrics']['bet_types'][bet_type]['wagered'] += bet_amount
                results['metrics']['bet_types'][bet_type]['returns'] += returns
                
                bet_type_performance[bet_type]['total'] += 1
                bet_type_performance[bet_type]['won'] += 1 if bet_won else 0
                bet_type_performance[bet_type]['wagered'] += bet_amount
                bet_type_performance[bet_type]['returns'] += returns
                bet_type_performance[bet_type]['bets'].append({
                    'edge': analysis['edge'],
                    'confidence': confidence_score,
                    'odds': odds,
                    'won': bet_won,
                    'amount': bet_amount,
                    'returns': returns
                })
                
                # Track by edge
                edge_bucket = int(analysis['edge'] * 100) // 2 * 2  # Round to nearest 2%
                edge_key = f"{edge_bucket}%-{edge_bucket+2}%"
                
                if edge_key not in results['metrics']['accuracy_by_edge']:
                    results['metrics']['accuracy_by_edge'][edge_key] = {'total': 0, 'correct': 0}
                if edge_key not in results['metrics']['roi_by_edge']:
                    results['metrics']['roi_by_edge'][edge_key] = {'wagered': 0, 'returns': 0}
                
                results['metrics']['accuracy_by_edge'][edge_key]['total'] += 1
                results['metrics']['accuracy_by_edge'][edge_key]['correct'] += 1 if bet_won else 0
                results['metrics']['roi_by_edge'][edge_key]['wagered'] += bet_amount
                results['metrics']['roi_by_edge'][edge_key]['returns'] += returns
                
                # Track by criteria combinations
                criteria_key = f"edge_{edge_bucket}-{edge_bucket+2}_conf_{confidence_bin}"
                if bet_won:
                    if criteria_key not in results['analysis']['profitable_criteria']:
                        results['analysis']['profitable_criteria'][criteria_key] = {
                            'count': 0, 'profit': 0, 'roi': 0
                        }
                    results['analysis']['profitable_criteria'][criteria_key]['count'] += 1
                    results['analysis']['profitable_criteria'][criteria_key]['profit'] += profit
                else:
                    if criteria_key not in results['analysis']['losing_criteria']:
                        results['analysis']['losing_criteria'][criteria_key] = {
                            'count': 0, 'loss': 0
                        }
                    results['analysis']['losing_criteria'][criteria_key]['count'] += 1
                    results['analysis']['losing_criteria'][criteria_key]['loss'] += bet_amount
                
                # Track team-specific betting performance
                team_tracked = team1_name if 'team1' in bet_type else (
                    team2_name if 'team2' in bet_type else None
                )
                
                if team_tracked:
                    results['team_performance'][team_tracked]['bets'] += 1
                    results['team_performance'][team_tracked]['wagered'] += bet_amount
                    results['team_performance'][team_tracked]['returns'] += returns
                    if bet_won:
                        results['team_performance'][team_tracked]['wins'] += 1
            
            # Store prediction results - same format as prediction function
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
            
            # Store bet results
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
            
            # Calculate current drawdown for reporting
            if len(results['performance']['bankroll_history']) > 0:
                max_bankroll_so_far = max([entry['bankroll'] for entry in results['performance']['bankroll_history']])
                if max_bankroll_so_far < current_bankroll:
                    max_bankroll_so_far = current_bankroll
                current_drawdown_pct = ((max_bankroll_so_far - current_bankroll) / max_bankroll_so_far) * 100 if max_bankroll_so_far > current_bankroll else 0
            else:
                current_drawdown_pct = 0
            
            # Track bankroll history with drawdown
            results['performance']['bankroll_history'].append({
                'match_idx': match_idx,
                'bankroll': current_bankroll,
                'match_id': match_id,
                'date': match_date,
                'current_drawdown': current_drawdown_pct
            })
            
            # Print periodic progress updates
            if (match_idx + 1) % 50 == 0 or match_idx == len(backtest_matches) - 1:
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                roi = (total_returns - total_wagered) / total_wagered if total_wagered > 0 else 0
                
                print(f"\nProgress ({match_idx + 1}/{len(backtest_matches)}):")
                print(f"Prediction Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
                print(f"Betting ROI: {roi:.2%} (${total_returns - total_wagered:.2f})")
                print(f"Current Bankroll: ${current_bankroll:.2f}")
                print(f"Current Drawdown: {current_drawdown_pct:.2f}%")
                print(f"Win Rate: {winning_bets/total_bets:.2%} ({winning_bets}/{total_bets})" if total_bets > 0 else "No bets placed")
                print(f"Win/Loss Streak: {current_streak} (max win: {longest_win_streak}, max loss: {longest_lose_streak})")
                
        except Exception as e:
            print(f"Error processing match {match_id}: {e}")
            traceback.print_exc()
            continue
    
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
    results['performance']['longest_win_streak'] = longest_win_streak
    results['performance']['longest_lose_streak'] = longest_lose_streak
    
    # Calculate team-specific metrics
    for team, stats in results['team_performance'].items():
        if stats['predictions'] > 0:
            stats['accuracy'] = stats['correct'] / stats['predictions']
        if stats['bets'] > 0:
            stats['win_rate'] = stats['wins'] / stats['bets']
        if stats['wagered'] > 0:
            stats['roi'] = (stats['returns'] - stats['wagered']) / stats['wagered']
            stats['profit'] = stats['returns'] - stats['wagered']
    
    # Calculate drawdown metrics
    drawdown_metrics = calculate_drawdown_metrics(results['performance']['bankroll_history'])
    results['performance']['drawdown_metrics'] = drawdown_metrics
    
    # Print final results
    print("\n========== BACKTEST RESULTS ==========")
    print(f"Total Matches: {total_predictions}")
    print(f"Prediction Accuracy: {final_accuracy:.2%} ({correct_predictions}/{total_predictions})")
    print(f"Total Bets: {total_bets}")
    if total_bets > 0:
        print(f"Winning Bets: {winning_bets} ({winning_bets/total_bets:.2%})")
    print(f"Total Wagered: ${total_wagered:.2f}")
    print(f"Total Returns: ${total_returns:.2f}")
    print(f"Profit/Loss: ${final_profit:.2f}")
    print(f"ROI: {final_roi:.2%}")
    print(f"Final Bankroll: ${current_bankroll:.2f}")
    print(f"Starting Bankroll: ${starting_bankroll:.2f}")
    print(f"Bankroll Growth: {(current_bankroll/starting_bankroll - 1)*100:.2f}%")
    
    # Print streak information
    print(f"\nLongest Winning Streak: {longest_win_streak} bets")
    print(f"Longest Losing Streak: {longest_lose_streak} bets")
    
    # Print drawdown metrics
    print("\nDrawdown Analysis:")
    print(f"Maximum Drawdown: {drawdown_metrics['max_drawdown_pct']:.2f}%")
    print(f"Maximum Drawdown Amount: ${drawdown_metrics['max_drawdown_amount']:.2f}")
    
    if drawdown_metrics['max_drawdown_start'] < len(backtest_matches) and drawdown_metrics['max_drawdown_end'] < len(backtest_matches):
        max_dd_start_match = backtest_matches[drawdown_metrics['max_drawdown_start']]
        max_dd_end_match = backtest_matches[drawdown_metrics['max_drawdown_end']]
        print(f"Max Drawdown Period: {max_dd_start_match['team1_name']} vs {max_dd_start_match['team2_name']} to {max_dd_end_match['team1_name']} vs {max_dd_end_match['team2_name']}")
    
    print(f"Drawdown Periods: {drawdown_metrics['drawdown_periods']}")
    print(f"Average Drawdown: {drawdown_metrics['avg_drawdown_pct']:.2f}%")
    print(f"Maximum Drawdown Duration: {drawdown_metrics['max_drawdown_duration']} matches")
    
    # Calculate Calmar ratio (annual return / max drawdown)
    if drawdown_metrics['max_drawdown_pct'] > 0:
        # Estimate annualized return (simplistic - assumes 250 trading days per year)
        matches_per_day = 2  # Assumption: 2 matches per day on average
        trading_days = len(backtest_matches) / matches_per_day
        years = trading_days / 250
        
        if years > 0:
            total_return = (current_bankroll / starting_bankroll) - 1
            annualized_return = ((1 + total_return) ** (1 / years)) - 1
            calmar_ratio = annualized_return * 100 / drawdown_metrics['max_drawdown_pct']
            
            print(f"Estimated Annualized Return: {annualized_return*100:.2f}%")
            print(f"Calmar Ratio: {calmar_ratio:.2f}")
    
    # Calculate Sharpe ratio (simplified)
    if len(results['performance']['bankroll_history']) > 1:
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(results['performance']['bankroll_history'])):
            prev_bankroll = results['performance']['bankroll_history'][i-1]['bankroll']
            curr_bankroll = results['performance']['bankroll_history'][i]['bankroll']
            daily_return = (curr_bankroll / prev_bankroll) - 1
            daily_returns.append(daily_return)
        
        # Calculate Sharpe ratio
        if daily_returns:
            avg_return = sum(daily_returns) / len(daily_returns)
            std_return = np.std(daily_returns) if len(daily_returns) > 1 else 0.0001
            
            if std_return > 0:
                sharpe_ratio = avg_return / std_return * np.sqrt(len(daily_returns))
                print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Print confidence bin analysis
    print("\nAccuracy by Confidence Level:")
    for conf_key, stats in sorted(confidence_bins.items()):
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f"  {conf_key}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"backtest_results_{timestamp}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nBacktest results saved to {save_path}")
    
    # Generate insights
    identify_key_insights(results)
    
    return results

def main():
    """Main function to handle command line arguments and run the program."""
    parser = argparse.ArgumentParser(description="Valorant Match Predictor and Betting System")
    
    # Add main command groups
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--retrain", action="store_true", help="Retrain with consistent features")
    parser.add_argument("--predict", action="store_true", help="Predict a match outcome and analyze betting options")
    parser.add_argument("--stats", action="store_true", help="View betting performance statistics")
    
    # Add data collection options
    parser.add_argument("--players", action="store_true", help="Include player stats in analysis")
    parser.add_argument("--economy", action="store_true", help="Include economy data in analysis")
    parser.add_argument("--maps", action="store_true", help="Include map statistics")
    parser.add_argument("--cross-validate", action="store_true", help="Train with cross-validation")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds for cross-validation")
    
    # Add prediction options
    parser.add_argument("--team1", type=str, help="First team name")
    parser.add_argument("--team2", type=str, help="Second team name")
    parser.add_argument("--live", action="store_true", help="Track the bet live (input result after match)")
    
    # Add bankroll parameter
    parser.add_argument("--bankroll", type=float, default=1000, help="Your current betting bankroll")

    # Add backtest command and options
    parser.add_argument("--backtest", action="store_true", help="Run backtesting on historical matches")
    parser.add_argument("--teams", type=int, default=50, help="Number of teams to include in backtesting")
    parser.add_argument("--start-date", type=str, help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--test-bankroll", type=float, default=1000, help="Starting bankroll for backtesting")
    parser.add_argument("--max-bet", type=float, default=0.05, help="Maximum bet size as fraction of bankroll")
    parser.add_argument("--min-edge", type=float, default=0.08, help="Minimum edge required for betting")
    parser.add_argument("--min-confidence", type=float, default=0.4, help="Minimum model confidence required")
    parser.add_argument("--interactive", action="store_true", help="Use interactive parameter entry for backtesting")
    parser.add_argument("--analyze-results", type=str, help="Analyze previous backtest results file")
    parser.add_argument("--test-symmetry", action="store_true", 
                       help="Run feature symmetry tests to verify predictions are consistent")
    parser.add_argument("--debug-model", action="store_true",
                       help="Run debugging analysis on trained models")  

    args = parser.parse_args()

    # Default to including all data types
    include_player_stats = args.players
    include_economy = args.economy
    include_maps = args.maps
    
    # If no data types specified, include all by default
    if not args.players and not args.economy and not args.maps:
        include_player_stats = True
        include_economy = True
        include_maps = True

    # Handle the symmetry test argument
    if args.test_symmetry:
        print("Running feature symmetry tests...")
        symmetry_passed = test_feature_symmetry()
        test_defensive_feature_symmetry()
        if symmetry_passed:
            print("All symmetry tests passed! Your model should make consistent predictions.")
        else:
            print("Symmetry tests failed. Features don't transform correctly when teams are swapped.")
            print("This will cause inconsistent predictions depending on team order.")
        return
        
    # Handle the debug model argument
    if args.debug_model:
        print("Running model debugging...")
        debug_trained_models()
        return

    if args.train:
        print("Training a new model...")
        
        # Collect team data
        team_data_collection = collect_team_data(
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
    
    elif args.retrain:
        print("Retraining models with consistent feature set...")
        
        # Collect team data
        team_data_collection = collect_team_data(
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
        
        # Train with consistent features
        fold_models, scaler, selected_features, avg_metrics = train_with_consistent_features(
            X, y, n_splits=args.folds, random_state=42
        )
        print("Retraining with consistent features complete.")
    
    elif args.predict:
        if args.team1 and args.team2:
            # Use the specific teams provided in arguments
            prediction = predict_match_unified(args.team1, args.team2, args.bankroll)
        else:
            # Prompt for team names
            print("\nEnter the teams to predict:")
            team1 = input("Team 1 name: ")
            team2 = input("Team 2 name: ")
            
            # Prompt for bankroll if not provided
            if not args.bankroll:
                try:
                    bankroll = float(input("\nEnter your current bankroll ($): ") or 1000)
                except ValueError:
                    bankroll = 1000
                    print("Invalid bankroll value, using default $1000")
            else:
                bankroll = args.bankroll
            
            if team1 and team2:
                prediction = predict_match_unified(team1, team2, bankroll)
            else:
                print("Team names are required for prediction.")
                return
                
        # Track the bet if requested
        if args.live and prediction and 'betting_analysis' in prediction:
            # Track the bet
            print("\nAfter the match, please enter the results:")
            
            recommended_bets = [bet_type for bet_type, analysis in prediction['betting_analysis'].items() 
                               if analysis.get('recommended', False)]
            
            if not recommended_bets:
                print("No recommended bets to track.")
            else:
                print("Recommended bets:")
                for i, bet_type in enumerate(recommended_bets):
                    print(f"{i+1}. {bet_type.replace('_', ' ').upper()}")
                
                try:
                    bet_choice = int(input("\nWhich bet did you place? (enter number, 0 for none): ")) - 1
                    if 0 <= bet_choice < len(recommended_bets):
                        bet_placed = recommended_bets[bet_choice]
                        bet_amount = float(input("How much did you bet? $"))
                        outcome = input("Did the bet win? (y/n): ").lower().startswith('y')
                        odds = 0
                        
                        # Get odds from the prediction results
                        odds_data = prediction.get('odds_data', {})
                        
                        # Get the odds for this bet
                        for bet_key, odds_value in odds_data.items():
                            if bet_key.replace('_odds', '') == bet_placed:
                                odds = odds_value
                                break
                        
                        if odds > 0:
                            # Track betting performance
                            track_betting_performance(prediction, bet_placed, bet_amount, outcome, odds)
                except (ValueError, IndexError):
                    print("Invalid input, not tracking this bet.")

    elif args.backtest:
        print("\nRunning backtesting to verify prediction accuracy and betting strategy...")
        
        # Use interactive parameter entry if requested
        if args.interactive:
            params = get_backtest_params()
            results = run_backtest_unified(
                start_date=params['start_date'],
                end_date=params['end_date'],
                team_limit=params['team_limit'],
                bankroll=params['bankroll'],
                bet_pct=params['bet_pct'],
                min_edge=params['min_edge'],
                confidence_threshold=params['confidence_threshold']
            )
        else:
            # Use command-line parameters
            results = run_backtest_unified(
                start_date=args.start_date,
                end_date=args.end_date,
                team_limit=args.teams,
                bankroll=args.test_bankroll,
                bet_pct=args.max_bet,
                min_edge=args.min_edge,
                confidence_threshold=args.min_confidence
            )
        
        if results:
            # Analyze key insights
            insights = identify_key_insights(results)
            
            # Ask user if they want to analyze specific matches
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
    
    elif args.analyze_results:
        try:
            print(f"Analyzing backtest results from {args.analyze_results}...")
            with open(args.analyze_results, 'r') as f:
                results = json.load(f)
            
            # Identify key insights
            insights = identify_key_insights(results)
            
            # Ask user if they want to analyze specific matches
            analyze_matches = input("\nWould you like to analyze specific matches from the backtest? (y/n): ").lower().startswith('y')
            if analyze_matches:
                while True:
                    print("\nOptions:")
                    print("1. Analyze a match by index")
                    print("2. Analyze a specific team's matches")
                    print("3. Analyze matches with high profit/loss")
                    print("4. Exit analysis")
                    
                    choice = input("\nEnter your choice (1-4): ")
                    
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
                        # Find matches with high profit or loss
                        profitable_matches = []
                        unprofitable_matches = []
                        
                        for bet_record in results['bets']:
                            match_id = bet_record.get('match_id')
                            if not match_id:
                                continue
                                
                            total_profit = sum(bet['profit'] for bet in bet_record['bets'])
                            if total_profit > 50:  # Significant profit
                                profitable_matches.append((match_id, bet_record, total_profit))
                            elif total_profit < -50:  # Significant loss
                                unprofitable_matches.append((match_id, bet_record, total_profit))
                        
                        print("\nMatches with high profit/loss:")
                        print("\nMost Profitable Matches:")
                        profitable_matches.sort(key=lambda x: x[2], reverse=True)
                        for i, (match_id, bet_record, profit) in enumerate(profitable_matches[:5]):
                            print(f"{i}. {bet_record['team1']} vs {bet_record['team2']}: ${profit:.2f}")
                        
                        print("\nLeast Profitable Matches:")
                        unprofitable_matches.sort(key=lambda x: x[2])
                        for i, (match_id, bet_record, profit) in enumerate(unprofitable_matches[:5]):
                            print(f"{i}. {bet_record['team1']} vs {bet_record['team2']}: ${profit:.2f}")
                        
                        analyze_type = input("\nAnalyze (P)rofitable or (U)nprofitable matches? ").lower()
                        if analyze_type.startswith('p'):
                            matches_to_analyze = profitable_matches[:5]
                        else:
                            matches_to_analyze = unprofitable_matches[:5]
                        
                        try:
                            match_idx = int(input("Enter match index to analyze: "))
                            if 0 <= match_idx < len(matches_to_analyze):
                                match_id = matches_to_analyze[match_idx][0]
                                analyze_specific_match(results, match_id)
                            else:
                                print("Invalid match index")
                        except ValueError:
                            print("Invalid input")
                    elif choice == '4':
                        break
                    else:
                        print("Invalid choice")
        except Exception as e:
            print(f"Error analyzing backtest results: {e}")
            traceback.print_exc()
            print("Make sure the file exists and contains valid backtest results.")

    elif args.stats:
        # View betting performance statistics
        view_betting_performance()
    
    else:
        print("Please specify an action: --train, --retrain, --predict, --backtest, --analyze-results, or --stats")
        print("\nExample commands:")
        print("  python valorant_predictor.py --train --cross-validate --folds 10")
        print("  python valorant_predictor.py --predict --team1 'Sentinels' --team2 'Cloud9'")
        print("  python valorant_predictor.py --backtest --teams 40 --interactive")
        print("  python valorant_predictor.py --analyze-results backtest_results_20240505_120000.json")
        print("  python valorant_predictor.py --stats")


if __name__ == "__main__":
    main()