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
# MAIN FUNCTION AND CLI INTERFACE
#-------------------------------------------------------------------------

def main():
    """Main function to handle command line arguments and run the program."""
    parser = argparse.ArgumentParser(description="Valorant Match Predictor and Betting System")
    
    # Add main command groups
    parser.add_argument("--train", action="store_true", help="Train a new model")
    # Add data collection options
    parser.add_argument("--players", action="store_true", help="Include player stats in analysis")
    parser.add_argument("--economy", action="store_true", help="Include economy data in analysis")
    parser.add_argument("--maps", action="store_true", help="Include map statistics")
    parser.add_argument("--cross-validate", action="store_true", help="Train with cross-validation")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for cross-validation")

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
    

    else:
        print("Please specify an action: --train")


if __name__ == "__main__":
    main()         