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

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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
            print(f"No teams with rankings found. Using the first {min(150, team_limit)} teams instead.")
            if len(teams_data['data']) > 0:
                top_teams = teams_data['data'][:min(150, team_limit)]
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

def create_improved_model(input_dim, regularization_strength=0.01, dropout_rate=0.5):
    """
    Create a deep learning model with stronger regularization to prevent extreme predictions.
    
    Args:
        input_dim (int): Input dimension
        regularization_strength (float): L2 regularization parameter
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        Model: Tensorflow/Keras model
    """
    # Define inputs
    inputs = Input(shape=(input_dim,))
    
    # First layer - shared feature processing with higher regularization
    x = Dense(128, activation='relu', 
              kernel_regularizer=l2(regularization_strength),
              kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second layer - deeper processing
    x = Dense(64, activation='relu', 
              kernel_regularizer=l2(regularization_strength),
              kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.8)(x)
    
    # Third layer - deeper processing
    x = Dense(32, activation='relu', 
              kernel_regularizer=l2(regularization_strength),
              kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.6)(x)
    
    # Output layer with sigmoid squeeze to avoid extreme values
    raw_output = Dense(1, activation='sigmoid',
                     kernel_regularizer=l2(regularization_strength))(x)
    
    # Apply a squeezing function to avoid extreme values (0 or 1)
    # This restricts predictions to the range [0.05, 0.95]
    squeezed_output = Lambda(lambda x: 0.05 + 0.9 * x)(raw_output)
    
    # Create model
    model = Model(inputs=inputs, outputs=squeezed_output)
    
    # Compile model with lower learning rate
    model.compile(loss='binary_crossentropy', 
                 optimizer=Adam(learning_rate=0.0001),
                 metrics=['accuracy'])
    
    return model

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

def predict_match(team1_name, team2_name, bankroll=1000):
    """
    Predict match outcome with the diverse ensemble.
    """
    print(f"\n{'='*80}")
    print(f"MATCH PREDICTION: {team1_name} vs {team2_name}")
    print(f"{'='*80}")
    
    # Load models and artifacts
    ensemble_models, _, selected_features = load_prediction_artifacts()
    
    if not ensemble_models:
        print("Failed to load prediction models. Please train models first.")
        return None
    
    if not selected_features:
        print("Failed to load feature list. Please retrain models.")
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
    
    # Add map statistics if available
    team1_stats['map_statistics'] = fetch_team_map_statistics(team1_id)
    team2_stats['map_statistics'] = fetch_team_map_statistics(team2_id)
    
    # Store team info
    team1_stats['team_name'] = team1_name
    team1_stats['team_id'] = team1_id
    team2_stats['team_name'] = team2_name
    team2_stats['team_id'] = team2_id
    
    # Use improved feature preparation
    X = prepare_ensemble_features(team1_stats, team2_stats, selected_features)
    
    if X is None:
        print("ERROR: Failed to prepare features for prediction")
        return None
    
    # Make predictions with diverse ensemble
    print("Running diverse ensemble prediction...")
    raw_predictions, calibrated_prediction, model_agreement = predict_with_diverse_ensemble(ensemble_models, X)
    
    if not raw_predictions:
        print("ERROR: All prediction models failed")
        return None
    
    # Format the results
    print(f"Raw predictions: {[f'{p:.4f}' for p in raw_predictions]}")
    print(f"Calibrated prediction: {calibrated_prediction:.4f}")
    print(f"Model agreement: {model_agreement:.4f}")
    
    # Calculate average prediction
    avg_prediction = np.mean(raw_predictions)
    std_prediction = np.std(raw_predictions)
    
    # Add safeguards for extreme predictions
    if avg_prediction < 0.05:
        print("WARNING: Extremely low win probability prediction. Adjusting to minimum 5%")
        avg_prediction = 0.05
    elif avg_prediction > 0.95:
        print("WARNING: Extremely high win probability prediction. Adjusting to maximum 95%")
        avg_prediction = 0.95
        
    # Use the calibrated prediction instead of raw average
    avg_prediction = calibrated_prediction
    
    print(f"Average prediction: {avg_prediction:.4f} ± {std_prediction:.4f}")
    
    # Calculate confidence interval (95%)
    conf_interval = (
        max(0, avg_prediction - 1.96 * std_prediction / np.sqrt(len(raw_predictions))),
        min(1, avg_prediction + 1.96 * std_prediction / np.sqrt(len(raw_predictions)))
    )
    
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
        over = float(input("Over 2.5 maps odds: ") or 0)
        if over > 0:
            odds_data['over_2_5_maps_odds'] = over
        
        under = float(input("Under 2.5 maps odds: ") or 0)
        if under > 0:
            odds_data['under_2_5_maps_odds'] = under
        
    except ValueError:
        print("Invalid odds input. Using available odds only.")
    
    # Calculate probability for each bet type
    results = {
        'team1_name': team1_name,
        'team2_name': team2_name,
        'team1_win_prob': avg_prediction,
        'team2_win_prob': 1 - avg_prediction,
        'confidence_interval': conf_interval,
        'model_agreement': model_agreement,
        'odds_data': odds_data
    }
    
    # Calculate probabilities for other bet types using the calibrated prediction
    single_map_prob = avg_prediction
    
    # Adjust based on historical map win rates if available
    if team1_stats['map_statistics'] and team2_stats['map_statistics']:
        t1_map_win_rate = 0
        t2_map_win_rate = 0
        count = 0
        
        for map_name in set(team1_stats['map_statistics'].keys()) & set(team2_stats['map_statistics'].keys()):
            t1_map = team1_stats['map_statistics'][map_name]
            t2_map = team2_stats['map_statistics'][map_name]
            
            if 'win_percentage' in t1_map and 'win_percentage' in t2_map:
                t1_map_win_rate += t1_map['win_percentage']
                t2_map_win_rate += t2_map['win_percentage']
                count += 1
        
        if count > 0:
            t1_map_win_rate /= count
            t2_map_win_rate /= count
            
            # Adjust single map probability based on map win rates
            map_ratio = t1_map_win_rate / (t1_map_win_rate + t2_map_win_rate) if (t1_map_win_rate + t2_map_win_rate) > 0 else 0.5
            single_map_prob = (single_map_prob + map_ratio) / 2  # Blend overall and map-specific probabilities
    
    # Probability calculations for different bet types
    results['team1_plus_1_5_prob'] = 1 - (1 - single_map_prob) ** 3
    results['team2_plus_1_5_prob'] = 1 - single_map_prob ** 3
    results['team1_minus_1_5_prob'] = single_map_prob ** 2
    results['team2_minus_1_5_prob'] = (1 - single_map_prob) ** 2
    results['over_2_5_maps_prob'] = 2 * single_map_prob * (1 - single_map_prob)
    results['under_2_5_maps_prob'] = single_map_prob ** 2 + (1 - single_map_prob) ** 2
    
    # Add betting analysis
    if odds_data:
        results['betting_analysis'] = analyze_betting_opportunities(results, odds_data, bankroll)
    
    # Print detailed report
    print_prediction_report(results, team1_stats, team2_stats)
    
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
    print(f"\nTraining with {n_splits}-fold cross-validation using consistent features and diverse ensemble")
    
    # Prepare and clean data
    df = clean_feature_data(X)
    X_arr = df.values
    y_arr = np.array(y)
    
    # Set up cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize arrays to store results for feature selection
    all_feature_importances = np.zeros(df.shape[1])
    fold_metrics = []
    
    print("Phase 1: Identifying important features across all folds...")
    
    # First pass: Identify feature importance across all folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        print(f"\n----- Feature Selection: Fold {fold+1}/{n_splits} -----")
        
        # Split data
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
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
    ensemble_models, scaler = create_diverse_ensemble(X_train, y_train, feature_mask, random_state)
    
    # Evaluate ensemble on validation set
    X_val_selected = X_val[:, feature_mask]
    predictions = []
    
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
            predictions.append(preds)
            
            # Calculate individual model metrics
            y_pred_binary = (preds > 0.5).astype(int)
            acc = accuracy_score(y_val, y_pred_binary)
            print(f"{model_type.upper()} model - Validation accuracy: {acc:.4f}")
        except Exception as e:
            print(f"Error evaluating {model_type} model: {e}")
    
    # Calculate ensemble predictions
    if predictions:
        # Take the average prediction for each validation sample
        ensemble_preds = np.mean(predictions, axis=0)
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
        'feature_importances': dict(zip(selected_features, avg_importances[top_indices]))
    }
    
    with open('feature_metadata.pkl', 'wb') as f:
        pickle.dump(feature_metadata, f)
    
    print("All models and artifacts saved successfully.")
    
    # Return ensemble, feature information, and metrics
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


def analyze_betting_edge_for_backtesting(team1_win_prob, team2_win_prob, odds_data, confidence_score, bankroll=1000.0):
    """
    Analyze betting edges with adjusted thresholds and more realistic Kelly sizing
    based on confidence level.
    """
    betting_analysis = {}
    
    # IMPROVED: Dynamic threshold adjustment based on confidence
    # Base minimum edge (absolute minimum edge for a bet)
    base_min_edge = 0.06  # Reduced from 0.08 to allow more bets
    
    # Confidence adjustment - lower confidence requires higher edge
    confidence_factor = 0.5 + (confidence_score * 0.5)  # Scale from 0.5 to 1.0
    adjusted_threshold = base_min_edge / confidence_factor
    
    print(f"\n----- BETTING ANALYSIS -----")
    print(f"Confidence score: {confidence_score:.4f}, confidence factor: {confidence_factor:.4f}")
    print(f"Base edge threshold: {base_min_edge:.2%}, adjusted threshold: {adjusted_threshold:.2%}")
    
    # Calculate map probabilities from match probabilities - more realistic
    # Use confidence-aware scaling
    map_scale = 0.5 + (confidence_score * 0.2)  # Between 0.5 and 0.7
    single_map_prob = 0.5 + (team1_win_prob - 0.5) * map_scale
    
    # Ensure map probability is reasonable
    single_map_prob = max(0.35, min(0.65, single_map_prob))
    print(f"Match probability: {team1_win_prob:.4f}, map probability: {single_map_prob:.4f} (scale: {map_scale:.2f})")
    
    # Calculate probabilities for different bet types
    team1_plus_prob = 1 - (1 - single_map_prob) ** 2  # Need to win at least 1 map
    team2_plus_prob = 1 - single_map_prob ** 2
    
    team1_minus_prob = single_map_prob ** 2  # Need to win 2 maps straight
    team2_minus_prob = (1 - single_map_prob) ** 2
    
    # Map total probabilities (more realistic for Valorant)
    over_prob = 2 * single_map_prob * (1 - single_map_prob)  # Prob of 1-1 after 2 maps
    under_prob = 1 - over_prob
    
    # Calculate edges for all bet types
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
    
    # IMPROVED: Analyze each bet type with more realistic bet sizing
    for bet_type, prob, odds in bet_types:
        # Calculate implied probability and edge
        implied_prob = 1 / odds
        edge = prob - implied_prob
        
        # IMPROVED: Kelly calculation with confidence-based fractional adjustment
        # Base fractional Kelly (higher confidence = higher fraction)
        base_fraction = 0.1 + (confidence_score * 0.1)  # Between 0.1 and 0.2
        
        # Calculate Kelly stake
        b = odds - 1  # Decimal odds to b format
        p = prob
        q = 1 - p
        
        # Handle extreme cases and zero division
        if b <= 0:
            kelly = 0
        else:
            kelly = (b * p - q) / b
            
            # Apply fractional Kelly based on confidence
            kelly = kelly * base_fraction
            
            # Additional cap based on edge size (higher edge = less reduction)
            if edge > 0:
                # Scale from 0.5 to 1.0 based on edge size
                edge_factor = 0.5 + min(0.5, edge * 5)
                kelly = kelly * edge_factor
            
            # Apply absolute cap for safety
            kelly = min(kelly, 0.05)  # Never bet more than 5% regardless of Kelly formula
        
        # Calculate bet amount
        bet_amount = round(bankroll * kelly, 2)
        
        # IMPROVED: More realistic bet qualification logic
        # Custom thresholds by bet type
        type_thresholds = {
            'team1_ml': adjusted_threshold,
            'team2_ml': adjusted_threshold,
            'team1_plus_1_5': adjusted_threshold * 0.9,  # Slightly lower for +1.5 bets
            'team2_plus_1_5': adjusted_threshold * 0.9,
            'team1_minus_1_5': adjusted_threshold * 1.3,  # Higher for riskier -1.5 bets
            'team2_minus_1_5': adjusted_threshold * 1.3,
            'over_2_5_maps': adjusted_threshold * 1.1,  # Slightly higher for totals
            'under_2_5_maps': adjusted_threshold * 1.1
        }
        
        # Get specific threshold for this bet type
        bet_threshold = type_thresholds.get(bet_type, adjusted_threshold)
        
        # Additional filters based on bet type and confidence
        extra_filter = True
        
        # Higher confidence requirements for riskier bets
        if ('minus_1_5' in bet_type) and confidence_score < 0.5:
            extra_filter = False
            filter_reason = "Low confidence for -1.5 spread"
        
        # For 50/50 probabilities, require higher edge
        elif 0.45 < prob < 0.55:
            # Higher edge required when probability is close to 50/50
            if edge < bet_threshold * 1.2:
                extra_filter = False
                filter_reason = "Insufficient edge for 50/50 probability"
        else:
            filter_reason = "Passed all filters"
        
        # Final edge check against bet-specific threshold
        meets_edge = edge > bet_threshold
        
        # Minimum bet amount check - avoid tiny bets
        meets_min_amount = bet_amount >= 5
        
        # Final recommendation logic
        recommended = meets_edge and meets_min_amount and extra_filter
        
        # Store analysis results
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
            'recommended': recommended
        }
        
        # Print details for logging
        print(f"{bet_type}: prob={prob:.4f}, edge={edge:.4f}, threshold={bet_threshold:.4f}, " + 
              f"amount=${bet_amount:.2f}, recommend={recommended}")
    
    # Count recommended bets
    recommended_count = sum(1 for analysis in betting_analysis.values() if analysis['recommended'])
    print(f"Found {recommended_count} recommended bets out of {len(bet_types)} analyzed")
    
    return betting_analysis

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
            print(f"No teams with rankings found. Using the first {min(100, limit)} teams instead.")
            if len(teams_data['data']) > 0:
                teams_list = teams_data['data'][:min(100, limit)]
        
        print(f"Selected {len(teams_list)} teams for backtesting")
        return teams_list
    
    except Exception as e:
        print(f"Error in get_teams_for_backtesting: {e}")
        return []

def predict_with_ensemble(ensemble_models, X):
    """
    Make predictions using ensemble of models with improved calibration and weighting.
    
    Args:
        ensemble_models (list): List of (model_type, model, scaler) tuples
        X (numpy.ndarray): Preprocessed features for prediction
        
    Returns:
        tuple: (win_probability, raw_predictions, confidence_score)
    """
    if not ensemble_models:
        raise ValueError("No models provided for prediction")
    
    # Ensure X is properly shaped for model input
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
        
    # Get predictions from each model
    raw_predictions = []
    model_weights = []
    
    for i, (model_type, model, model_scaler) in enumerate(ensemble_models):
        try:
            # Apply scaling if needed
            X_pred = X.copy()
            if model_scaler is not None:
                try:
                    X_pred = model_scaler.transform(X_pred)
                except:
                    print(f"Scaling error in model {i}, using unscaled features")
            
            # Make prediction based on model type
            if model_type == 'nn':
                pred = model.predict(X_pred, verbose=0)[0][0]
            else:
                pred = model.predict_proba(X_pred)[0][1]
            
            # Handle NaN or invalid predictions
            if np.isnan(pred) or not np.isfinite(pred):
                print(f"Model {i+1} returned invalid prediction, using 0.5")
                pred = 0.5
            
            # Set weight based on model type to reduce neural network dominance
            if model_type == 'nn':
                weight = 0.5  # Reduce neural network weight
            elif model_type == 'gb' or model_type == 'rf':
                weight = 2.0  # Increase tree-based model weight
            else:
                weight = 1.0
                
            # Store both prediction and weight
            raw_predictions.append(pred)
            model_weights.append(weight)
            
            print(f"{model_type.upper()} model prediction: {pred:.4f}")
        except Exception as e:
            print(f"Error with model {i+1}: {e}")
            continue
    
    if not raw_predictions:
        print("All models failed to produce predictions, using default value")
        return 0.5, [0.5], 0.0
    
    # Calculate weighted average
    if raw_predictions and sum(model_weights) > 0:
        mean_pred = np.average(raw_predictions, weights=model_weights)
    else:
        mean_pred = 0.5
        
    # Calculate standard deviation for confidence calculation
    std_pred = np.std(raw_predictions)
    confidence_score = 1 - min(1, std_pred * 2)  # Higher means more agreement
    
    print(f"Raw ensemble mean: {mean_pred:.4f}, std: {std_pred:.4f}, confidence: {confidence_score:.4f}")
    
    # More aggressive calibration to fix overconfidence
    if std_pred > 0.25:  # Lower threshold for regression
        # Regress extreme predictions more heavily toward 0.5
        calibrated_pred = 0.5 + (mean_pred - 0.5) * 0.3
    else:
        # For normal predictions, still apply moderate calibration
        # Move predictions closer to 0.5 by 40%
        calibrated_pred = 0.5 + (mean_pred - 0.5) * 0.6
    
    # Extra safeguard against extreme predictions
    calibrated_pred = max(0.3, min(0.7, calibrated_pred))
    
    return calibrated_pred, raw_predictions, confidence_score

def prepare_features_for_backtest(team1_stats, team2_stats, selected_features):
    """
    Enhanced feature preparation for backtesting with consistent feature engineering,
    improved feature derivation, and better error handling.
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
    
    # STEP 2: Derive missing features using domain knowledge
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
        
        # Average metrics
        elif feature == 'avg_win_rate':
            val = (team1_stats.get('win_rate', 0.5) + team2_stats.get('win_rate', 0.5)) / 2
            complete_features[feature] = val
            missing_features.remove(feature)
            derived_features.append(feature)
            
        elif feature == 'avg_recent_form':
            val = (team1_stats.get('recent_form', 0.5) + team2_stats.get('recent_form', 0.5)) / 2
            complete_features[feature] = val
            missing_features.remove(feature)
            derived_features.append(feature)
            
        # Head-to-head features
        elif feature == 'h2h_win_rate':
            # Try to find opponent stats in team1's stats
            h2h_win_rate = 0.5  # Default is even match
            team2_name = team2_stats.get('team_name', '')
            
            if 'opponent_stats' in team1_stats and team2_name:
                opponent_found = False
                
                # Check different variations of the name
                for opponent_name, stats in team1_stats['opponent_stats'].items():
                    if (opponent_name.lower() == team2_name.lower() or 
                        team2_name.lower() in opponent_name.lower() or
                        opponent_name.lower() in team2_name.lower()):
                        if isinstance(stats, dict) and 'win_rate' in stats:
                            h2h_win_rate = stats['win_rate']
                            opponent_found = True
                            break
                
                # If no match, create an estimate based on relative strength
                if not opponent_found:
                    team1_win_rate = team1_stats.get('win_rate', 0.5)
                    team2_win_rate = team2_stats.get('win_rate', 0.5)
                    
                    if team1_win_rate + team2_win_rate > 0:
                        h2h_win_rate = team1_win_rate / (team1_win_rate + team2_win_rate)
                    
                    # Scale to reduce extremes
                    h2h_win_rate = 0.5 + (h2h_win_rate - 0.5) * 0.7
            
            complete_features[feature] = h2h_win_rate
            missing_features.remove(feature)
            derived_features.append(feature)
            
        # Player stats features
        elif feature == 'player_rating_diff':
            val = team1_stats.get('avg_player_rating', 1.0) - team2_stats.get('avg_player_rating', 1.0)
            complete_features[feature] = val
            missing_features.remove(feature)
            derived_features.append(feature)
            
        elif feature == 'better_player_rating_team1' and 'player_rating_diff' in complete_features.columns:
            val = 1 if complete_features['player_rating_diff'].values[0] > 0 else 0
            complete_features[feature] = val
            missing_features.remove(feature)
            derived_features.append(feature)
            
        # Economy features
        elif feature == 'pistol_win_rate_diff':
            val = team1_stats.get('pistol_win_rate', 0.5) - team2_stats.get('pistol_win_rate', 0.5)
            complete_features[feature] = val
            missing_features.remove(feature)
            derived_features.append(feature)
            
        # Interaction terms
        elif feature == 'rating_x_win_rate' and 'player_rating_diff' in complete_features.columns and 'win_rate_diff' in complete_features.columns:
            val = complete_features['player_rating_diff'].values[0] * complete_features['win_rate_diff'].values[0]
            complete_features[feature] = val
            missing_features.remove(feature)
            derived_features.append(feature)
            
        # If feature name contains "padding_feature", it's ok to leave it at 0
        elif "padding_feature" in feature:
            missing_features.remove(feature)
    
    # STEP 3: Normalize extreme values to prevent model bias
    for feature in selected_features:
        if feature in complete_features.columns:
            value = complete_features[feature].values[0]
            # Cap extreme values to prevent unreasonable predictions
            if feature.endswith('_diff') or feature.endswith('differential'):
                if value > 0.7:
                    complete_features[feature] = 0.7
                    print(f"Normalized high value in {feature}: {value} → 0.7")
                elif value < -0.7:
                    complete_features[feature] = -0.7
                    print(f"Normalized low value in {feature}: {value} → -0.7")
    
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
    
    return X

def run_backtest(start_date=None, end_date=None, team_limit=50, bankroll=1000.0, bet_pct=0.05, min_edge=0.08, confidence_threshold=0.4):
    """
    Enhanced backtesting system with improved risk management and analysis capabilities.
    
    Args:
        start_date (str): Start date for backtesting (YYYY-MM-DD)
        end_date (str): End date for backtesting (YYYY-MM-DD)
        team_limit (int): Maximum number of teams to analyze
        bankroll (float): Starting bankroll
        bet_pct (float): Maximum percentage of bankroll to bet
        min_edge (float): Minimum edge required for betting
        confidence_threshold (float): Minimum model confidence required
        
    Returns:
        dict: Detailed backtest results
    """
    print("\n========== STARTING ENHANCED BACKTEST ==========")
    print(f"Parameters: teams={team_limit}, bankroll=${bankroll}, max bet={bet_pct*100}%")
    print(f"Min edge: {min_edge*100}%, min confidence: {confidence_threshold*100}%")
    
    # Load models and metadata
    ensemble_models, selected_features = load_backtesting_models()
    if not ensemble_models or not selected_features:
        print("Failed to load required models and features. Aborting backtest.")
        return None
    
    # Collect team data with progress tracking
    print("Collecting team data for backtesting...")
    team_data = {}
    team_ids = {}
    team_count = 0
    
    # Get list of teams with enhanced error handling
    try:
        teams_list = get_teams_for_backtesting(limit=team_limit)
        if not teams_list:
            print("Error: No teams retrieved for backtesting. Check API connection.")
            return None
    except Exception as e:
        print(f"Error retrieving teams: {e}")
        return None
    
    # Process teams with better progress tracking
    for team_idx, team in enumerate(tqdm(teams_list, desc="Loading team data")):
        team_name = team.get('name')
        team_id = team.get('id')
        
        if not team_name or not team_id:
            continue
        
        # Store team ID
        team_ids[team_name] = team_id
        
        # Get team data with error handling
        try:
            team_history = fetch_team_match_history(team_id)
            if not team_history:
                print(f"No match history for {team_name}, skipping.")
                continue
            
            team_matches = parse_match_data(team_history, team_name)
            if not team_matches:
                print(f"No parsed match data for {team_name}, skipping.")
                continue
            
            # Filter by date range if specified
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
            
            # Get player stats with timeout and error handling
            team_player_stats = None
            try:
                team_player_stats = fetch_team_player_stats(team_id)
            except Exception as e:
                print(f"Error fetching player stats for {team_name}: {e}")
            
            # Calculate team stats
            team_stats = calculate_team_stats(team_matches, team_player_stats, include_economy=True)
            
            # Store data
            team_data[team_name] = {
                'team_id': team_id,
                'stats': team_stats,
                'matches': team_matches
            }
            
            team_count += 1
            print(f"Loaded data for {team_name}: {len(team_matches)} matches ({team_idx+1}/{len(teams_list)})")
            
            # Break if we reach the team limit
            if team_count >= team_limit:
                break
                
        except Exception as e:
            print(f"Error processing team {team_name}: {e}")
            continue
    
    print(f"Successfully loaded data for {len(team_data)} teams")
    
    if not team_data:
        print("Error: No team data collected. Aborting backtest.")
        return None
    
    # Create dataset for backtesting with deduplication
    backtest_matches = []
    seen_match_ids = set()
    
    # Collect matches with available data for both teams
    for team_name, team_info in team_data.items():
        for match in team_info['matches']:
            match_id = match.get('match_id', '')
            opponent_name = match.get('opponent_name')
            
            # Skip if we don't have data for the opponent
            if opponent_name not in team_data:
                continue
            
            # Skip if already processed
            if match_id in seen_match_ids:
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
        print("Error: No matches available for backtesting. Check match data collection.")
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
            'confidence_bins': {}
        },
        'team_performance': {}  # Track performance by team
    }
    
    # Initialize tracking variables
    current_bankroll = bankroll
    correct_predictions = 0
    total_predictions = 0
    
    # Track betting metrics
    total_bets = 0
    winning_bets = 0
    total_wagered = 0
    total_returns = 0
    
    # Track performance by confidence level
    confidence_bins = {}
    
    # Run backtest with improved progress tracking
    for match_idx, match in enumerate(tqdm(backtest_matches, desc="Backtesting matches")):
        team1_name = match['team1_name']
        team2_name = match['team2_name']
        match_data = match['match_data']
        match_id = match['match_id']
        
        # Track metrics by team if not already initialized
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
            # Get team stats
            team1_stats = team_data[team1_name]['stats']
            team2_stats = team_data[team2_name]['stats']
            
            # Prepare features for prediction with better error handling
            X = prepare_features_for_backtest(team1_stats, team2_stats, selected_features)
            
            if X is None:
                print(f"Failed to prepare features for {team1_name} vs {team2_name}, skipping")
                continue
            
            # Make prediction with improved error handling
            try:
                win_probability, raw_predictions, confidence_score = predict_with_ensemble(
                    ensemble_models, X
                )
            except Exception as e:
                print(f"Error making prediction for {team1_name} vs {team2_name}: {e}")
                continue
            
            # Get actual result with better parsing
            try:
                team1_score, team2_score = extract_match_score(match_data)
                actual_winner = 'team1' if team1_score > team2_score else 'team2'
            except Exception as e:
                print(f"Error extracting match score for {match_id}: {e}")
                continue
            
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
            
            # Analyze betting edges with enhanced risk management
            betting_analysis = analyze_betting_edge_for_backtesting(
                win_probability, 1 - win_probability, odds_data, 
                confidence_score, current_bankroll
            )
            
            # Apply enhanced filtering criteria
            filtered_analysis = {}
            for bet_type, analysis in betting_analysis.items():
                # Override recommendation based on enhanced criteria
                is_recommended = (
                    analysis['edge'] >= min_edge and
                    analysis['bet_amount'] > 0 and
                    confidence_score >= confidence_threshold
                )
                
                # Copy analysis and update recommendation
                filtered_analysis[bet_type] = analysis.copy()
                filtered_analysis[bet_type]['recommended'] = is_recommended
            
            # Simulate bets with better record keeping
            match_bets = []
            
            for bet_type, analysis in filtered_analysis.items():
                if analysis['recommended']:
                    # Calculate bet size (respecting max bet percentage and current bankroll)
                    max_bet = current_bankroll * bet_pct
                    bet_amount = min(analysis['bet_amount'], max_bet)
                    
                    # Ensure minimum bet size for practicality
                    if bet_amount < 1.0:
                        continue
                    
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
                'date': match_data.get('date', 'Unknown')
            })
            
            # Track bets
            if match_bets:
                results['bets'].append({
                    'match_id': match_id,
                    'team1': team1_name,
                    'team2': team2_name,
                    'bets': match_bets,
                    'date': match_data.get('date', 'Unknown')
                })
            
            # Track bankroll history with timestamp
            results['performance']['bankroll_history'].append({
                'match_idx': match_idx,
                'bankroll': current_bankroll,
                'match_id': match_id,
                'date': match_data.get('date', 'Unknown')
            })
            
            # Print periodic progress updates
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
    
    # Print accuracy by edge
    print("\nAccuracy by Edge:")
    for edge_key, stats in sorted(results['metrics']['accuracy_by_edge'].items()):
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {edge_key}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    # Print ROI by edge
    print("\nROI by Edge:")
    for edge_key, stats in sorted(results['metrics']['roi_by_edge'].items()):
        roi = (stats['returns'] - stats['wagered']) / stats['wagered'] if stats['wagered'] > 0 else 0
        print(f"  {edge_key}: {roi:.2%} (${stats['returns'] - stats['wagered']:.2f})")
    
    # Print performance by bet type
    print("\nPerformance by Bet Type:")
    for bet_type, stats in results['metrics']['bet_types'].items():
        win_rate = stats['won'] / stats['total'] if stats['total'] > 0 else 0
        roi = (stats['returns'] - stats['wagered']) / stats['wagered'] if stats['wagered'] > 0 else 0
        print(f"  {bet_type}: {win_rate:.2%} win rate, {roi:.2%} ROI (${stats['returns'] - stats['wagered']:.2f})")
    
    # Print top and bottom performing teams
    print("\nTop 5 Teams by ROI (min 5 bets):")
    team_roi = [(team, stats['roi'], stats['profit'], stats['bets']) 
                for team, stats in results['team_performance'].items() 
                if stats.get('bets', 0) >= 5 and 'roi' in stats]
    team_roi.sort(key=lambda x: x[1], reverse=True)
    
    for i, (team, roi, profit, bets) in enumerate(team_roi[:5]):
        print(f"  {i+1}. {team}: {roi:.2%} ROI, ${profit:.2f} profit ({bets} bets)")
    
    # Create enhanced visualizations
    create_enhanced_backtest_visualizations(results)
    
    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"backtest_results_{timestamp}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary to CSV for easy analysis
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

def create_enhanced_backtest_visualizations(results):
    """Create enhanced visualizations for backtest results with additional insights."""
    # Create directory for visualizations
    os.makedirs("backtest_plots", exist_ok=True)
    
    # 1. Bankroll history with improved formatting
    plt.figure(figsize=(14, 7))
    
    # Extract bankroll data
    bankroll_history = [entry['bankroll'] for entry in results['performance']['bankroll_history']]
    match_indices = [entry['match_idx'] for entry in results['performance']['bankroll_history']]
    
    # Plot with better formatting
    plt.plot(match_indices, bankroll_history, 'b-', linewidth=2)
    plt.axhline(y=bankroll_history[0], color='r', linestyle='--', alpha=0.5, label='Initial Bankroll')
    
    # Add annotations for significant points
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
    
    plt.title('Bankroll Progression During Backtest', fontsize=16)
    plt.xlabel('Match Number', fontsize=14)
    plt.ylabel('Bankroll ($)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add final bankroll annotation
    if bankroll_history:
        plt.annotate(f'Final: ${bankroll_history[-1]:.2f}',
                    xy=(match_indices[-1], bankroll_history[-1]),
                    xytext=(match_indices[-1]-5, bankroll_history[-1] + 30),
                    ha='right')
    
    plt.tight_layout()
    plt.savefig('backtest_plots/bankroll_history.png', dpi=300)
    plt.close()
    
    # 2. Enhanced calibration curve
    plt.figure(figsize=(12, 8))
    
    # Create bins for prediction probabilities
    prob_bins = np.arange(0, 1.1, 0.1)
    bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
    
    # Initialize arrays for tracking
    bin_counts = np.zeros(len(bin_centers))
    bin_correct = np.zeros(len(bin_centers))
    
    # Sort predictions into bins
    for pred in results['predictions']:
        # Get the probability for the predicted winner
        if pred['predicted_winner'] == 'team1':
            prob = pred['team1_prob']
        else:
            prob = pred['team2_prob']
        
        # Find the appropriate bin
        bin_idx = np.digitize(prob, prob_bins) - 1
        if bin_idx < len(bin_counts):
            bin_counts[bin_idx] += 1
            bin_correct[bin_idx] += 1 if pred['correct'] else 0
    
    # Calculate actual frequencies
    actual_freqs = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        if bin_counts[i] > 0:
            actual_freqs[i] = bin_correct[i] / bin_counts[i]
    
    # Plot calibration curve with enhanced styling
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2, alpha=0.7)
    plt.plot(bin_centers, actual_freqs, 'bo-', label='Model Calibration', linewidth=2, markersize=8)
    
    # Add sample counts as bar chart on secondary axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, bin_counts, alpha=0.2, width=0.08, color='gray', label='Sample Count')
    ax2.set_ylabel('Number of Predictions', fontsize=14)
    
    # Add labels
    for i, (center, count, freq) in enumerate(zip(bin_centers, bin_counts, actual_freqs)):
        if count > 0:
            ax1.annotate(f"{int(count)}", (center, freq + 0.03),
                        ha='center', va='bottom', fontsize=9)
    
    # Add overall metrics
    total_correct = sum(bin_correct)
    total_preds = sum(bin_counts)
    overall_accuracy = total_correct / total_preds if total_preds > 0 else 0
    
    # Calculate calibration metrics
    bins_with_data = [i for i in range(len(bin_counts)) if bin_counts[i] > 0]
    calibration_error = np.mean([abs(actual_freqs[i] - bin_centers[i]) for i in bins_with_data]) if bins_with_data else 0
    
    plt.figtext(0.15, 0.15, f"Overall Accuracy: {overall_accuracy:.2%}\nCalibration Error: {calibration_error:.3f}",
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.title('Prediction Calibration Analysis', fontsize=16)
    plt.xlabel('Predicted Probability', fontsize=14)
    plt.ylabel('Actual Frequency', fontsize=14)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('backtest_plots/calibration_curve.png', dpi=300)
    plt.close()
    
    # 3. Enhanced ROI by edge analysis
    edges = []
    rois = []
    counts = []
    profits = []
    
    for edge_key, stats in results['metrics']['roi_by_edge'].items():
        if stats['wagered'] > 0:
            # Extract edge value from key (e.g., "5%-10%" -> 7.5)
            edge_parts = edge_key.replace('%', '').split('-')
            avg_edge = (float(edge_parts[0]) + float(edge_parts[1])) / 2 / 100
            
            roi = (stats['returns'] - stats['wagered']) / stats['wagered']
            profit = stats['returns'] - stats['wagered']
            edges.append(avg_edge)
            rois.append(roi)
            counts.append(stats['wagered'])
            profits.append(profit)
    
    if edges:
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with size based on amount wagered
        scatter = plt.scatter(edges, rois, s=[c/20 for c in counts], alpha=0.7, 
                             c=profits, cmap='RdYlGn', vmin=min(profits), vmax=max(profits))
        
        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Profit/Loss ($)', fontsize=12)
        
        # Add reference line
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Add regression line
        if len(edges) > 1:
            z = np.polyfit(edges, rois, 1)
            p = np.poly1d(z)
            plt.plot(sorted(edges), p(sorted(edges)), "r--", alpha=0.7, 
                    label=f"Trend: y = {z[0]:.2f}x + {z[1]:.2f}")
            
            # Calculate correlation
            correlation = np.corrcoef(edges, rois)[0, 1]
            plt.annotate(f"Correlation: {correlation:.2f}", 
                        xy=(0.05, 0.05), xycoords='axes fraction',
                        bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title('ROI vs. Predicted Edge', fontsize=16)
        plt.xlabel('Predicted Edge', fontsize=14)
        plt.ylabel('Realized ROI', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis as percentage
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.tight_layout()
        plt.savefig('backtest_plots/roi_by_edge.png', dpi=300)
        plt.close()
    
    # 4. Enhanced bet type performance with win rates and ROI
    bet_types = list(results['metrics']['bet_types'].keys())
    
    if bet_types:
        win_rates = []
        rois = []
        volumes = []
        
        for bet_type in bet_types:
            stats = results['metrics']['bet_types'][bet_type]
            win_rates.append(stats['won'] / stats['total'] if stats['total'] > 0 else 0)
            rois.append((stats['returns'] - stats['wagered']) / stats['wagered'] if stats['wagered'] > 0 else 0)
            volumes.append(stats['wagered'])
        
        # Sort by volume for better visualization
        sorted_indices = np.argsort(volumes)[::-1]
        sorted_bet_types = [bet_types[i] for i in sorted_indices]
        sorted_win_rates = [win_rates[i] for i in sorted_indices]
        sorted_rois = [rois[i] for i in sorted_indices]
        sorted_volumes = [volumes[i] for i in sorted_indices]
        
        plt.figure(figsize=(14, 8))
        
        # Create a more informative grouped bar chart
        x = np.arange(len(sorted_bet_types))
        width = 0.35
        
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot win rates and ROI as bars
        bars1 = ax1.bar(x - width/2, sorted_win_rates, width, label='Win Rate', color='skyblue')
        bars2 = ax1.bar(x + width/2, sorted_rois, width, label='ROI', color='lightgreen')
        
        # Plot volume as a line on secondary axis
        ax2.plot(x, sorted_volumes, 'ro-', label='Volume Wagered ($)', linewidth=2)
        
        # Add value labels to bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.annotate(f'{height:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax1.annotate(f'{height:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        # Add a horizontal line at y=0
        ax1.axhline(y=0, color='r', linestyle='-', alpha=0.2)
        
        # Add labels and legend
        ax1.set_xlabel('Bet Type', fontsize=14)
        ax1.set_ylabel('Rate (%)', fontsize=14)
        ax2.set_ylabel('Volume Wagered ($)', fontsize=14)
        
        ax1.set_title('Performance by Bet Type', fontsize=16)
        ax1.set_xticks(x)
        ax1.set_xticklabels([bt.replace('_', ' ').upper() for bt in sorted_bet_types], rotation=45, ha='right')
        
        # Format y-axis as percentage
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Add combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('backtest_plots/bet_type_performance.png', dpi=300)
        plt.close()
    
    # 5. New visualization: Team performance analysis
    if 'team_performance' in results:
        # Get teams with significant betting activity
        significant_teams = {team: stats for team, stats in results['team_performance'].items() 
                           if stats.get('bets', 0) >= 5}
        
        if significant_teams:
            # Sort teams by ROI
            sorted_teams = sorted(significant_teams.items(), 
                                 key=lambda x: x[1].get('roi', 0) if 'roi' in x[1] else 0, 
                                 reverse=True)
            
            # Limit to top 15 teams for readability
            top_teams = sorted_teams[:15]
            
            team_names = [t[0] for t in top_teams]
            team_rois = [t[1].get('roi', 0) if 'roi' in t[1] else 0 for t in top_teams]
            team_win_rates = [t[1].get('win_rate', 0) if 'win_rate' in t[1] else 0 for t in top_teams]
            team_bets = [t[1].get('bets', 0) for t in top_teams]
            
            plt.figure(figsize=(14, 8))
            
            # Create horizontal bar chart for better readability with long team names
            y_pos = np.arange(len(team_names))
            
            # Create bars
            plt.barh(y_pos, team_rois, align='center', height=0.5, color='lightgreen', 
                    label='ROI')
            
            # Add win rate as text
            for i, (roi, win_rate, bets) in enumerate(zip(team_rois, team_win_rates, team_bets)):
                plt.annotate(f"Win: {win_rate:.1%}, Bets: {bets}", 
                            xy=(roi + 0.02, i), 
                            verticalalignment='center',
                            fontsize=9)
            
            # Add vertical line at x=0
            plt.axvline(x=0, color='r', linestyle='-', alpha=0.2)
            
            # Add labels
            plt.xlabel('Return on Investment (ROI)', fontsize=14)
            plt.title('Team Performance by ROI (Min. 5 Bets)', fontsize=16)
            plt.yticks(y_pos, team_names)
            
            # Format x-axis as percentage
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
            
            plt.tight_layout()
            plt.savefig('backtest_plots/team_performance.png', dpi=300)
            plt.close()
    
    # 6. New visualization: Model confidence vs. accuracy
    if 'metrics' in results and 'confidence_bins' in results['metrics']:
        confidence_data = results['metrics']['confidence_bins']
        
        if confidence_data:
            conf_levels = []
            accuracies = []
            counts = []
            
            for conf_key, stats in confidence_data.items():
                if stats['total'] > 0:
                    # Extract confidence level from key (e.g., "70%" -> 0.7)
                    conf_level = float(conf_key.replace('%', '')) / 100
                    accuracy = stats['correct'] / stats['total']
                    
                    conf_levels.append(conf_level)
                    accuracies.append(accuracy)
                    counts.append(stats['total'])
            
            if conf_levels:
                plt.figure(figsize=(12, 8))
                
                # Create scatter plot with size based on sample count
                scatter = plt.scatter(conf_levels, accuracies, s=[c*2 for c in counts], 
                                     alpha=0.7, c=accuracies, cmap='viridis')
                
                # Add perfect calibration line
                plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                
                # Add labels
                for i, (x, y, count) in enumerate(zip(conf_levels, accuracies, counts)):
                    plt.annotate(f"n={count}", (x, y), 
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=9)
                
                # Calculate correlation and add trend line
                if len(conf_levels) > 1:
                    z = np.polyfit(conf_levels, accuracies, 1)
                    p = np.poly1d(z)
                    plt.plot(sorted(conf_levels), p(sorted(conf_levels)), "r--", alpha=0.7,
                            label=f"Trend: y = {z[0]:.2f}x + {z[1]:.2f}")
                    
                    correlation = np.corrcoef(conf_levels, accuracies)[0, 1]
                    plt.annotate(f"Correlation: {correlation:.2f}", 
                                xy=(0.05, 0.90), xycoords='axes fraction',
                                bbox=dict(facecolor='white', alpha=0.7))
                
                plt.title('Model Confidence vs. Actual Accuracy', fontsize=16)
                plt.xlabel('Model Confidence', fontsize=14)
                plt.ylabel('Actual Accuracy', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Format axes as percentage
                plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
                
                plt.tight_layout()
                plt.savefig('backtest_plots/confidence_vs_accuracy.png', dpi=300)
                plt.close()
    
    # 7. New visualization: Daily profit/loss over time
    if 'bets' in results and results['bets']:
        # Group bets by date
        daily_profits = {}
        daily_bets = {}
        
        for bet_record in results['bets']:
            date = bet_record.get('date', 'Unknown')
            if date == 'Unknown':
                continue
                
            # Extract just the date part if it's a datetime string
            date_parts = date.split()
            if date_parts:
                date = date_parts[0]
                
            if date not in daily_profits:
                daily_profits[date] = 0
                daily_bets[date] = 0
                
            # Sum profits for all bets on this match
            for bet in bet_record['bets']:
                daily_profits[date] += bet.get('profit', 0)
                daily_bets[date] += 1
        
        if daily_profits:
            # Convert to lists for plotting
            dates = sorted(daily_profits.keys())
            profits = [daily_profits[d] for d in dates]
            bets_count = [daily_bets[d] for d in dates]
            
            # Calculate cumulative profit
            cum_profit = np.cumsum(profits)
            
            plt.figure(figsize=(14, 10))
            
            # Create subplots
            gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
            
            # Top plot: Daily and cumulative profit
            ax1 = plt.subplot(gs[0])
            
            # Plot daily profit/loss as bars
            bars = ax1.bar(dates, profits, alpha=0.7, label='Daily P/L')
            
            # Color bars based on profit/loss
            for i, p in enumerate(profits):
                bars[i].set_color('green' if p >= 0 else 'red')
            
            # Plot cumulative profit as line on same axis
            ax1_b = ax1.twinx()
            ax1_b.plot(dates, cum_profit, 'b-', label='Cumulative P/L', linewidth=2)
            
            # Add labels
            ax1.set_title('Daily Profit/Loss Over Time', fontsize=16)
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Daily Profit/Loss ($)', fontsize=12)
            ax1_b.set_ylabel('Cumulative Profit/Loss ($)', fontsize=12)
            
            # Add combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_b.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Format x-axis for readability
            if len(dates) > 10:
                ax1.set_xticks(ax1.get_xticks()[::len(dates)//10])
                
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Bottom plot: Number of bets per day
            ax2 = plt.subplot(gs[1], sharex=ax1)
            ax2.bar(dates, bets_count, alpha=0.7, color='skyblue', label='Number of Bets')
            ax2.set_ylabel('Number of Bets', fontsize=12)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig('backtest_plots/daily_profit_loss.png', dpi=300)
            plt.close()
    
    # Create an index HTML file for easy viewing of all plots
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Results Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1, h2 {
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .plot-container {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 20px;
            }
            .plot-image {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
            }
            .description {
                margin-top: 10px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Valorant Match Prediction Backtest Results</h1>
            <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            
            <div class="plot-container">
                <h2>Bankroll History</h2>
                <img class="plot-image" src="bankroll_history.png" alt="Bankroll History">
                <div class="description">
                    Shows how the bankroll changed over the course of the backtest. Annotations indicate maximum and minimum points.
                </div>
            </div>
            
            <div class="plot-container">
                <h2>Prediction Calibration</h2>
                <img class="plot-image" src="calibration_curve.png" alt="Calibration Curve">
                <div class="description">
                    Compares predicted probabilities vs. actual outcomes. A well-calibrated model should follow the diagonal line.
                </div>
            </div>
            
            <div class="plot-container">
                <h2>ROI vs. Predicted Edge</h2>
                <img class="plot-image" src="roi_by_edge.png" alt="ROI by Edge">
                <div class="description">
                    Shows the relationship between predicted edge and actual ROI. Points are sized by amount wagered.
                </div>
            </div>
            
            <div class="plot-container">
                <h2>Performance by Bet Type</h2>
                <img class="plot-image" src="bet_type_performance.png" alt="Bet Type Performance">
                <div class="description">
                    Compares win rates and ROI across different bet types, with volume information.
                </div>
            </div>
            
            <div class="plot-container">
                <h2>Team Performance</h2>
                <img class="plot-image" src="team_performance.png" alt="Team Performance">
                <div class="description">
                    Shows ROI and win rates for teams with at least 5 bets placed, sorted by ROI.
                </div>
            </div>
            
            <div class="plot-container">
                <h2>Model Confidence vs. Accuracy</h2>
                <img class="plot-image" src="confidence_vs_accuracy.png" alt="Confidence vs Accuracy">
                <div class="description">
                    Analyzes how model confidence correlates with actual prediction accuracy.
                </div>
            </div>
            
            <div class="plot-container">
                <h2>Daily Profit/Loss</h2>
                <img class="plot-image" src="daily_profit_loss.png" alt="Daily Profit/Loss">
                <div class="description">
                    Shows daily profit/loss and cumulative profit over time, along with number of bets placed each day.
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open('backtest_plots/index.html', 'w') as f:
        f.write(html_content)
    
    print(f"Created visualization dashboard at backtest_plots/index.html")

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
            prediction = predict_match(args.team1, args.team2, args.bankroll)
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
                prediction = predict_match(team1, team2, bankroll)
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
                        
                        # FIXED: Get odds_data from the prediction results instead of undefined variable
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
        print("\nRunning enhanced backtesting to verify prediction accuracy and betting strategy...")
        
        # Use interactive parameter entry if requested
        if args.interactive:
            params = get_backtest_params()
            results = run_backtest(
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
            results = run_backtest(
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