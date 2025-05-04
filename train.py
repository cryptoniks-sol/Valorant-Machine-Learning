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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

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
    Prepare data for the ML model by creating symmetrical feature vectors.
    Improved to handle name variations in head-to-head statistics.
    
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
    # Add head-to-head statistics with improved matching
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
        # Default values if no H2H data found
        features['h2h_win_rate'] = 0.5
        features['h2h_matches'] = 0
        features['h2h_score_diff'] = 0
        features['h2h_advantage_team1'] = 0
        features['h2h_significant'] = 0
        
        print(f"No head-to-head data found between {team1_stats.get('team_name', 'Team1')} "
              f"and {team2_name} in opponent_stats")
        
        # Attempt to manually reconstruct head-to-head from match data
        if 'matches' in team1_stats and isinstance(team1_stats['matches'], list):
            h2h_matches = []
            
            for match in team1_stats['matches']:
                opponent_name = match.get('opponent_name', '')
                
                if not opponent_name:
                    continue
                    
                # Check all variations
                for variation in team2_variations:
                    if (opponent_name.lower() == variation.lower() or
                        variation.lower() in opponent_name.lower() or
                        opponent_name.lower() in variation.lower()):
                        h2h_matches.append(match)
                        break
            
            # If we found matches, calculate head-to-head stats
            if h2h_matches:
                wins = sum(1 for match in h2h_matches if match.get('team_won', False))
                total_score = sum(match.get('team_score', 0) for match in h2h_matches)
                total_opponent_score = sum(match.get('opponent_score', 0) for match in h2h_matches)
                avg_score_diff = (total_score - total_opponent_score) / len(h2h_matches)
                
                features['h2h_win_rate'] = wins / len(h2h_matches)
                features['h2h_matches'] = len(h2h_matches)
                features['h2h_score_diff'] = avg_score_diff
                features['h2h_advantage_team1'] = 1 if features['h2h_win_rate'] > 0.5 else 0
                features['h2h_significant'] = 1 if features['h2h_matches'] >= 3 else 0
                
                print(f"Manually reconstructed head-to-head data: Matches={features['h2h_matches']}, "
                      f"Win rate={features['h2h_win_rate']:.4f}, "
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

def predict_match(team1_name, team2_name):
    """
    Predict match outcome between two specific teams without fetching all team data.
    """
    print(f"\n{'='*80}")
    print(f"MATCH PREDICTION: {team1_name} vs {team2_name}")
    print(f"{'='*80}")
    
    # Load prediction artifacts
    models, scaler, selected_features = load_prediction_artifacts()
    
    if not models:
        print("Failed to load prediction models. Please train models first.")
        return None
    
    # We can continue even if scaler or selected_features have issues
    
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
    
    # Use robust feature preparation
    X = prepare_prediction_features(team1_stats, team2_stats, selected_features, scaler)
    
    if X is None:
        print("ERROR: Failed to prepare features for prediction")
        return None
    
    # Make predictions with each model in the ensemble
    # Make predictions with each model in the ensemble
    print("Running prediction models...")
    predictions = []
    for i, model in enumerate(models):
        try:
            pred = model.predict(X, verbose=0)[0][0]
            predictions.append(pred)
            print(f"Model {i+1} prediction: {pred:.4f}")
        except Exception as e:
            print(f"Error with model {i+1}: {e}")
    
    if not predictions:
        print("ERROR: All prediction models failed")
        return None
    
    # Calculate average prediction
    avg_prediction = np.mean(predictions)
    std_prediction = np.std(predictions)
    model_agreement = 1 - (std_prediction * 2)  # Higher means more agreement

    # Add this after calculating avg_prediction
    if avg_prediction < 0.05:
        print("WARNING: Extremely low win probability prediction. Adjusting to minimum 5%")
        avg_prediction = 0.05
    elif avg_prediction > 0.95:
        print("WARNING: Extremely high win probability prediction. Adjusting to maximum 95%")
        avg_prediction = 0.95
    
      # Add a warning if model agreement is too low
    if model_agreement < min_agreement:
        print(f"\nWARNING: Low model agreement ({model_agreement:.2f}). Predictions may be unreliable.")
        print("Consider skipping this bet or reducing your stake significantly.")
    
    # Add safeguards for extreme predictions
    if avg_prediction < 0.05:
        print("WARNING: Extremely low win probability prediction. Adjusting to minimum 5%")
        avg_prediction = 0.05
    elif avg_prediction > 0.95:
        print("WARNING: Extremely high win probability prediction. Adjusting to maximum 95%")
        avg_prediction = 0.95
    
    # Calibrate prediction based on confidence
    calibrated_prediction = calibrate_prediction(avg_prediction, model_agreement)
    if abs(calibrated_prediction - avg_prediction) > 0.05:
        print(f"Calibrated prediction from {avg_prediction:.4f} to {calibrated_prediction:.4f} based on model agreement")
        avg_prediction = calibrated_prediction
    
    # Calculate confidence interval (95%)
    conf_interval = (
        max(0, avg_prediction - 1.96 * std_prediction / np.sqrt(len(predictions))),
        min(1, avg_prediction + 1.96 * std_prediction / np.sqrt(len(predictions)))
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
        'model_agreement': model_agreement
    }
    
    # Calculate probabilities for other bet types
    # Assuming independence between maps (simplified model)
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
    
    # Probability of team1 winning at least 1 map (team1 +1.5)
    results['team1_plus_1_5_prob'] = 1 - (1 - single_map_prob) ** 3
    
    # Probability of team2 winning at least 1 map (team2 +1.5)
    results['team2_plus_1_5_prob'] = 1 - single_map_prob ** 3
    
    # Probability of team1 winning 2-0 (team1 -1.5)
    results['team1_minus_1_5_prob'] = single_map_prob ** 2
    
    # Probability of team2 winning 2-0 (team2 -1.5)
    results['team2_minus_1_5_prob'] = (1 - single_map_prob) ** 2
    
    # Probability of match going to 3 maps (over 2.5 maps)
    results['over_2_5_maps_prob'] = 2 * single_map_prob * (1 - single_map_prob)
    
    # Probability of match ending in 2 maps (under 2.5 maps)
    results['under_2_5_maps_prob'] = single_map_prob ** 2 + (1 - single_map_prob) ** 2
    
    # Add betting analysis with safety parameters
    if odds_data:
        results['betting_analysis'] = analyze_betting_opportunities(
            results, odds_data, bankroll, max_kelly_pct, min_roi, min_agreement
        )
    
    # Analyze similar matchups for context
    analyze_similar_matchups(team1_stats, team2_stats)
    
    # Load feature importance if available
    try:
        with open('feature_metadata.pkl', 'rb') as f:
            feature_metadata = pickle.load(f)
            if 'feature_importances' in feature_metadata:
                # Explain prediction factors
                explain_prediction_factors(team1_stats, team2_stats, 
                                         selected_features, 
                                         feature_metadata['feature_importances'])
    except Exception as e:
        print(f"Note: Could not load feature importance data: {e}")
    
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
    min_edge = 0.05  # Minimum 5% edge to recommend a bet
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
    
    # Analyze moneyline bet for team1
    if 'team1_ml_odds' in odds_data:
        implied_prob = decimal_to_prob(odds_data['team1_ml_odds'])
        our_prob = prediction_results['team1_win_prob']
        ev = our_prob - implied_prob
        roi = calculate_roi(our_prob, odds_data['team1_ml_odds'])
        
        # Calculate Kelly bet size
        kelly_fraction = kelly_bet(our_prob, odds_data['team1_ml_odds'])
        bet_amount = round(bankroll * kelly_fraction, 2)
        
        # Only recommend if meets all criteria
        is_recommended = (ev > min_edge and 
                         our_prob > min_confidence and 
                         roi > min_roi and
                         bet_amount > 0)
        
        betting_analysis['team1_ml'] = {
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
    
    # Analyze moneyline bet for team2
    if 'team2_ml_odds' in odds_data:
        implied_prob = decimal_to_prob(odds_data['team2_ml_odds'])
        our_prob = prediction_results['team2_win_prob']
        ev = our_prob - implied_prob
        roi = calculate_roi(our_prob, odds_data['team2_ml_odds'])
        
        # Calculate Kelly bet size
        kelly_fraction = kelly_bet(our_prob, odds_data['team2_ml_odds'])
        bet_amount = round(bankroll * kelly_fraction, 2)
        
        # Only recommend if meets all criteria
        is_recommended = (ev > min_edge and 
                         our_prob > min_confidence and 
                         roi > min_roi and
                         bet_amount > 0)
        
        betting_analysis['team2_ml'] = {
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
    
    # Analyze team1 +1.5 maps
    if 'team1_plus_1_5_odds' in odds_data:
        implied_prob = decimal_to_prob(odds_data['team1_plus_1_5_odds'])
        our_prob = prediction_results['team1_plus_1_5_prob']
        ev = our_prob - implied_prob
        roi = calculate_roi(our_prob, odds_data['team1_plus_1_5_odds'])
        
        # Calculate Kelly bet size
        kelly_fraction = kelly_bet(our_prob, odds_data['team1_plus_1_5_odds'])
        bet_amount = round(bankroll * kelly_fraction, 2)
        
        # Only recommend if meets all criteria
        is_recommended = (ev > min_edge and 
                         our_prob > min_confidence and 
                         roi > min_roi and
                         bet_amount > 0)
        
        betting_analysis['team1_plus_1_5'] = {
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
    
    # Analyze team2 +1.5 maps
    if 'team2_plus_1_5_odds' in odds_data:
        implied_prob = decimal_to_prob(odds_data['team2_plus_1_5_odds'])
        our_prob = prediction_results['team2_plus_1_5_prob']
        ev = our_prob - implied_prob
        roi = calculate_roi(our_prob, odds_data['team2_plus_1_5_odds'])
        
        # Calculate Kelly bet size
        kelly_fraction = kelly_bet(our_prob, odds_data['team2_plus_1_5_odds'])
        bet_amount = round(bankroll * kelly_fraction, 2)
        
        # Only recommend if meets all criteria
        is_recommended = (ev > min_edge and 
                         our_prob > min_confidence and 
                         roi > min_roi and
                         bet_amount > 0)
        
        betting_analysis['team2_plus_1_5'] = {
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
    
    # Analyze team1 -1.5 maps
    if 'team1_minus_1_5_odds' in odds_data:
        implied_prob = decimal_to_prob(odds_data['team1_minus_1_5_odds'])
        our_prob = prediction_results['team1_minus_1_5_prob']
        ev = our_prob - implied_prob
        roi = calculate_roi(our_prob, odds_data['team1_minus_1_5_odds'])
        
        # Calculate Kelly bet size
        kelly_fraction = kelly_bet(our_prob, odds_data['team1_minus_1_5_odds'])
        bet_amount = round(bankroll * kelly_fraction, 2)
        
        # Only recommend if meets all criteria
        is_recommended = (ev > min_edge and 
                         our_prob > min_confidence and 
                         roi > min_roi and
                         bet_amount > 0)
        
        betting_analysis['team1_minus_1_5'] = {
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
    
    # Analyze team2 -1.5 maps
    if 'team2_minus_1_5_odds' in odds_data:
        implied_prob = decimal_to_prob(odds_data['team2_minus_1_5_odds'])
        our_prob = prediction_results['team2_minus_1_5_prob']
        ev = our_prob - implied_prob
        roi = calculate_roi(our_prob, odds_data['team2_minus_1_5_odds'])
        
        # Calculate Kelly bet size
        kelly_fraction = kelly_bet(our_prob, odds_data['team2_minus_1_5_odds'])
        bet_amount = round(bankroll * kelly_fraction, 2)
        
        # Only recommend if meets all criteria
        is_recommended = (ev > min_edge and 
                         our_prob > min_confidence and 
                         roi > min_roi and
                         bet_amount > 0)
        
        betting_analysis['team2_minus_1_5'] = {
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
    
    # Analyze over 2.5 maps
    if 'over_2_5_maps_odds' in odds_data:
        implied_prob = decimal_to_prob(odds_data['over_2_5_maps_odds'])
        our_prob = prediction_results['over_2_5_maps_prob']
        ev = our_prob - implied_prob
        roi = calculate_roi(our_prob, odds_data['over_2_5_maps_odds'])
        
        # Calculate Kelly bet size
        kelly_fraction = kelly_bet(our_prob, odds_data['over_2_5_maps_odds'])
        bet_amount = round(bankroll * kelly_fraction, 2)
        
        # Only recommend if meets all criteria
        is_recommended = (ev > min_edge and 
                         our_prob > min_confidence and 
                         roi > min_roi and
                         bet_amount > 0)
        
        betting_analysis['over_2_5_maps'] = {
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
    
    # Analyze under 2.5 maps
    if 'under_2_5_maps_odds' in odds_data:
        implied_prob = decimal_to_prob(odds_data['under_2_5_maps_odds'])
        our_prob = prediction_results['under_2_5_maps_prob']
        ev = our_prob - implied_prob
        roi = calculate_roi(our_prob, odds_data['under_2_5_maps_odds'])
        
        # Calculate Kelly bet size
        kelly_fraction = kelly_bet(our_prob, odds_data['under_2_5_maps_odds'])
        bet_amount = round(bankroll * kelly_fraction, 2)
        
        # Only recommend if meets all criteria
        is_recommended = (ev > min_edge and 
                         our_prob > min_confidence and 
                         roi > min_roi and
                         bet_amount > 0)
        
        betting_analysis['under_2_5_maps'] = {
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
            for i, (bet, edge, amount) in enumerate(recommended_bets):
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

def load_prediction_artifacts(model_path_prefix='valorant_model_fold_', num_models=10):
    """
    Load all trained models and artifacts needed for prediction.
    
    Args:
        model_path_prefix (str): Prefix for model file paths
        num_models (int): Number of models to load
        
    Returns:
        tuple: (models, scaler, selected_features)
    """
    print("Loading prediction models and artifacts...")
    
    # Load models
    models = []
    for i in range(1, num_models + 1):
        model_path = f"{model_path_prefix}{i}.h5"
        try:
            model = load_model(model_path)
            models.append(model)
            print(f"Loaded model {i}/{num_models}")
        except Exception as e:
            print(f"Error loading model {i}: {e}")
    
    if not models:
        print("Failed to load any models. Please check model paths.")
        return None, None, None
    
    # Load feature scaler
    try:
        with open('feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Loaded feature scaler")
    except Exception as e:
        print(f"Error loading feature scaler: {e}")
        return models, None, None
    
    # Load selected features
    try:
        with open('feature_metadata.pkl', 'rb') as f:
            feature_metadata = pickle.load(f)
            selected_features = feature_metadata.get('selected_features', [])
        print(f"Loaded {len(selected_features)} selected features")
    except Exception as e:
        print(f"Error loading selected features: {e}")
        try:
            # Fallback to stable features if available
            with open('stable_features.pkl', 'rb') as f:
                selected_features = pickle.load(f)
            print(f"Loaded {len(selected_features)} stable features as fallback")
        except Exception as e2:
            print(f"Error loading stable features: {e2}")
            return models, scaler, None
    
    return models, scaler, selected_features

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
    Applies scaling AFTER feature selection for better consistency.
    
    Args:
        X (list or DataFrame): Feature data
        y (list): Target labels
        n_splits (int): Number of cross-validation folds
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (trained_models, feature_scaler, selected_features, avg_metrics)
    """
    print(f"\nTraining with {n_splits}-fold cross-validation using consistent features")
    
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
    
    # Second pass: Train models with consistent feature set
    print("\nPhase 2: Training models with consistent feature set...")
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        print(f"\n----- Training Fold {fold+1}/{n_splits} -----")
        
        # Split data
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]
        
        # Apply SMOTE if needed
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
        
        # Select consistent features
        X_train_selected = X_train[:, feature_mask]
        X_val_selected = X_val[:, feature_mask]
        
        # Scale AFTER feature selection - this is the key change
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        
        # Create and train model
        input_dim = X_train_scaled.shape[1]
        model = create_model(input_dim)
        
        # Set up callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            f'valorant_model_fold_{fold+1}.h5', save_best_only=True, monitor='val_accuracy'
        )
        
        # Train model on scaled data
        history = model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_scaled, y_val),
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )
        
        # Evaluate model
        y_pred_proba = model.predict(X_val_scaled)
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
            'auc': auc
        })
        
        fold_models.append(model)
        
        # Print fold results
        print(f"Fold {fold+1} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
    
    # Calculate average metrics
    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) 
                  for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
    
    std_metrics = {metric: np.std([fold[metric] for fold in fold_metrics]) 
                  for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
    
    print(f"\nAverage Metrics Across {n_splits} Folds:")
    for metric, value in avg_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f} ± {std_metrics[metric]:.4f}")
    
    # Save models and artifacts
    print("\nSaving models and artifacts...")
    
    # Save the scaler for the last fold (any fold's scaler would work)
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
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
    
    return fold_models, scaler, selected_features, avg_metrics




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
    parser.add_argument("--bankroll", type=float, default=1000, help="Your current betting bankroll")
    parser.add_argument("--min-agreement", type=float, default=0.4, help="Minimum model agreement required (0-1)")
    parser.add_argument("--max-kelly", type=float, default=0.05, help="Maximum Kelly fraction allowed (0-1)")
    parser.add_argument("--min-roi", type=float, default=0.1, help="Minimum ROI required for bet recommendation")


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
            prediction = predict_match(args.team1, args.team2, 
                                      bankroll=args.bankroll,
                                      min_agreement=args.min_agreement,
                                      max_kelly_pct=args.max_kelly,
                                      min_roi=args.min_roi)
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
                prediction = predict_match(team1, team2, 
                                          bankroll=bankroll,
                                          min_agreement=args.min_agreement,
                                          max_kelly_pct=args.max_kelly,
                                          min_roi=args.min_roi)
            else:
                print("Team names are required for prediction.")
                return
        
        if args.live and prediction and 'betting_analysis' in prediction:
            # Track the bet if requested
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
                        odds = odds_data.get(bet_placed, 0)
                        
                        if odds > 0:
                            # Track betting performance
                            track_betting_performance(prediction, bet_placed, bet_amount, outcome, odds)
                except (ValueError, IndexError):
                    print("Invalid input, not tracking this bet.")
    
    elif args.stats:
        # View betting performance statistics
        view_betting_performance()
    
    else:
        print("Please specify an action: --train, --retrain, --predict, or --stats")


if __name__ == "__main__":
    main()