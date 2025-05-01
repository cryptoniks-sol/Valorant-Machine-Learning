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

def collect_team_data(team_limit=100, include_player_stats=True, include_economy=True, include_maps=True):
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
        print(f"No teams with rankings found. Using the first {min(25, team_limit)} teams instead.")
        top_teams = teams_data['data'][:min(25, team_limit)]
    
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
    Prepare features for a single match for prediction.
    
    Args:
        team1_stats (dict): Statistics for team 1
        team2_stats (dict): Statistics for team 2
        stable_features (list): List of feature names to use
        scaler (object): Trained feature scaler
        
    Returns:
        numpy.ndarray: Scaled feature vector ready for model prediction
    """
    # Use the existing prepare_data_for_model function to get base features
    features = prepare_data_for_model(team1_stats, team2_stats)
    
    if not features:
        print("Failed to prepare match features.")
        return None
    
    # Convert to DataFrame for easier manipulation
    features_df = pd.DataFrame([features])
    
    # Select only stable features that were identified during cross-validation
    available_features = [f for f in stable_features if f in features_df.columns]
    
    if len(available_features) < len(stable_features):
        missing = len(stable_features) - len(available_features)
        print(f"Warning: {missing} stable features are missing from the input data.")
    
    # Select and order features to match the training data
    X = features_df[available_features].values
    
    # Scale features
    if scaler:
        X_scaled = scaler.transform(X)
        return X_scaled
    else:
        print("Warning: No scaler provided. Using unscaled features.")
        return X

#-------------------------------------------------------------------------
# PREDICTION AND BETTING FUNCTIONS
#-------------------------------------------------------------------------

def ensemble_predict(models, X, calibrate=True, confidence=True):
    """
    Get predictions from ensemble of models with confidence scores.
    
    Args:
        models (list): List of trained models
        X (numpy.ndarray): Scaled feature vector
        calibrate (bool): Whether to calibrate probabilities
        confidence (bool): Whether to return confidence intervals
        
    Returns:
        dict: Prediction results with probabilities and confidence intervals
    """
    if not models or X is None:
        return None
    
    # Get predictions from each model
    predictions = []
    for i, model in enumerate(models):
        try:
            pred = model.predict(X)
            predictions.append(pred[0][0])  # Extract raw probability
        except Exception as e:
            print(f"Error getting prediction from model {i+1}: {e}")
    
    # Calculate ensemble statistics
    if predictions:
        mean_prob = np.mean(predictions)
        median_prob = np.median(predictions)
        std_dev = np.std(predictions)
        
        # Apply calibration to correct for overconfidence
        if calibrate:
            # Simple calibration: push extreme probabilities toward the center
            # More sophisticated methods like Platt scaling could be used with more data
            calibrated_prob = 0.5 + (mean_prob - 0.5) * 0.9  # Shrink by 10%
        else:
            calibrated_prob = mean_prob
        
        result = {
            'team1_win_probability': float(calibrated_prob),
            'team2_win_probability': float(1 - calibrated_prob),
            'raw_predictions': predictions,
            'std_dev': float(std_dev),
            'ensemble_size': len(predictions)
        }
        
        # Add confidence intervals if requested
        if confidence:
            # 95% confidence interval
            alpha = 0.05
            lower_bound = max(0, mean_prob - 1.96 * std_dev / np.sqrt(len(predictions)))
            upper_bound = min(1, mean_prob + 1.96 * std_dev / np.sqrt(len(predictions)))
            
            result['confidence_interval'] = {
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'confidence_level': 1 - alpha
            }
        
        return result
    
    return None

def predict_series_outcome(team1_win_prob, format_type="bo3"):
    """
    Calculate probabilities for different match outcomes in a series.
    
    Args:
        team1_win_prob (float): Probability of team1 winning a single map
        format_type (str): Format of the series ("bo1", "bo3", "bo5")
        
    Returns:
        dict: Probabilities for different outcomes
    """
    team2_win_prob = 1 - team1_win_prob
    
    if format_type == "bo1":
        # Best of 1 (single map)
        return {
            'team1_win_series': team1_win_prob,
            'team2_win_series': team2_win_prob,
            'team1_win_2_0': 0,
            'team2_win_2_0': 0,
            'team1_win_2_1': 0,
            'team2_win_2_1': 0,
            'maps_over_1.5': 0,  # Always under in a BO1
            'maps_under_1.5': 1.0
        }
    
    elif format_type == "bo3":
        # Best of 3 (first to 2 maps)
        team1_2_0 = team1_win_prob * team1_win_prob  # Win first two maps
        team2_2_0 = team2_win_prob * team2_win_prob  # Win first two maps
        
        team1_2_1 = 2 * team1_win_prob * team2_win_prob * team1_win_prob  # Win, lose, win or lose, win, win
        team2_2_1 = 2 * team2_win_prob * team1_win_prob * team2_win_prob  # Win, lose, win or lose, win, win
        
        team1_win_series = team1_2_0 + team1_2_1
        team2_win_series = team2_2_0 + team2_2_1
        
        maps_over_2_5 = team1_2_1 + team2_2_1  # 3 maps played
        maps_under_2_5 = team1_2_0 + team2_2_0  # 2 maps played
        
        # Maps_over_1.5 means at least 2 maps played, which is always true in BO3
        
        return {
            'team1_win_series': team1_win_series,
            'team2_win_series': team2_win_series,
            'team1_win_2_0': team1_2_0,
            'team2_win_2_0': team2_2_0,
            'team1_win_2_1': team1_2_1,
            'team2_win_2_1': team2_2_1,
            'maps_over_2.5': maps_over_2_5,
            'maps_under_2.5': maps_under_2_5,
            'maps_over_1.5': 1.0,
            'maps_under_1.5': 0.0
        }
    
    elif format_type == "bo5":
        # Best of 5 (first to 3 maps)
        team1_3_0 = team1_win_prob ** 3
        team2_3_0 = team2_win_prob ** 3
        
        team1_3_1 = 3 * (team1_win_prob ** 3) * team2_win_prob
        team2_3_1 = 3 * (team2_win_prob ** 3) * team1_win_prob
        
        team1_3_2 = 6 * (team1_win_prob ** 3) * (team2_win_prob ** 2)
        team2_3_2 = 6 * (team2_win_prob ** 3) * (team1_win_prob ** 2)
        
        team1_win_series = team1_3_0 + team1_3_1 + team1_3_2
        team2_win_series = team2_3_0 + team2_3_1 + team2_3_2
        
        maps_over_3_5 = team1_3_1 + team2_3_1 + team1_3_2 + team2_3_2
        maps_under_3_5 = team1_3_0 + team2_3_0
        
        maps_over_4_5 = team1_3_2 + team2_3_2
        maps_under_4_5 = team1_3_0 + team2_3_0 + team1_3_1 + team2_3_1
        
        return {
            'team1_win_series': team1_win_series,
            'team2_win_series': team2_win_series,
            'team1_win_3_0': team1_3_0,
            'team2_win_3_0': team2_3_0,
            'team1_win_3_1': team1_3_1,
            'team2_win_3_1': team2_3_1,
            'team1_win_3_2': team1_3_2,
            'team2_win_3_2': team2_3_2,
            'maps_over_3.5': maps_over_3_5,
            'maps_under_3.5': maps_under_3_5,
            'maps_over_4.5': maps_over_4_5,
            'maps_under_4.5': maps_under_4_5
        }
    
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

def calculate_implied_probability(odds):
    """
    Calculate implied probability from decimal odds.
    
    Args:
        odds (float): Decimal odds
        
    Returns:
        float: Implied probability
    """
    return 1 / odds

def calculate_expected_value(probability, odds):
    """
    Calculate expected value of a bet.
    
    Args:
        probability (float): Estimated probability of winning
        odds (float): Decimal odds offered by bookmaker
        
    Returns:
        float: Expected value (positive = profitable)
    """
    return probability * (odds - 1) - (1 - probability)

def kelly_criterion(probability, odds, bankroll, fraction=1.0):
    """
    Calculate optimal bet size using the Kelly Criterion.
    
    Args:
        probability (float): Estimated probability of winning
        odds (float): Decimal odds offered by bookmaker
        bankroll (float): Current bankroll size
        fraction (float): Fraction of full Kelly to use (0.0-1.0), lower = more conservative
        
    Returns:
        float: Recommended bet size
    """
    # Calculate implied probability from odds
    implied_prob = calculate_implied_probability(odds)
    
    # Kelly formula: f* = (bp - q) / b
    # where f* is the fraction of bankroll to bet
    # p is the probability of winning
    # q is the probability of losing (1-p)
    # b is the odds - 1
    
    b = odds - 1  # Net odds received on the bet
    p = probability
    q = 1 - p
    
    if p <= 0 or p >= 1:
        return 0
    
    # Standard Kelly formula
    if p > implied_prob:  # Only bet when we have an edge
        kelly_stake = (b * p - q) / b
    else:
        kelly_stake = 0
    
    # Apply the fraction (fractional Kelly)
    kelly_stake *= fraction
    
    # Limit the stake to a reasonable amount (e.g., max 20% of bankroll)
    kelly_stake = min(kelly_stake, 0.2)
    
    # Calculate the actual bet size
    bet_size = bankroll * kelly_stake
    
    return bet_size

def find_value_bets(prediction_results, available_odds, format_type='bo3', bankroll=1000.0, kelly_fraction=0.25, min_edge=0.05):
    """
    Identify value bets with positive expected value and calculate optimal stakes.
    
    Args:
        prediction_results (dict): Model predictions for the match
        available_odds (dict): Available odds from bookmakers
        format_type (str): Match format (bo1, bo3, bo5)
        bankroll (float): Current bankroll size
        kelly_fraction (float): Conservative fraction of Kelly stake to use
        min_edge (float): Minimum edge required to place a bet
        
    Returns:
        dict: Value bets with optimal stake sizes and expected values
    """
    # Calculate series outcome probabilities
    map_win_prob = prediction_results['team1_win_probability']
    series_probs = predict_series_outcome(map_win_prob, format_type)
    
    # Initialize value bets container
    value_bets = {}
    
    # Check each bet type for value
    for bet_type, odds in available_odds.items():
        if bet_type in series_probs:
            our_prob = series_probs[bet_type]
            implied_prob = calculate_implied_probability(odds)
            edge = our_prob - implied_prob
            
            # Additional edge check: make sure edge meets minimum threshold
            edge_percent = edge / implied_prob * 100.0
            
            if edge > min_edge:
                ev = calculate_expected_value(our_prob, odds)
                stake = kelly_criterion(our_prob, odds, bankroll, kelly_fraction)
                
                if stake > 0:
                    value_bets[bet_type] = {
                        'odds': odds,
                        'our_probability': our_prob,
                        'implied_probability': implied_prob,
                        'edge': edge,
                        'edge_percent': edge_percent,
                        'expected_value': ev,
                        'recommended_stake': stake,
                        'profit_if_win': stake * (odds - 1),
                        'confidence': prediction_results.get('confidence_interval', {})
                    }
    
    return value_bets

def evaluate_models_profitability(models, test_data, odds_data, initial_bankroll=1000.0, fixed_stake=False, plot=True):
    """
    Evaluate model profitability on historical data with real odds.
    
    Args:
        models (list): List of trained models
        test_data (DataFrame): Test dataset with features and outcomes
        odds_data (DataFrame): Historical odds data for the matches
        initial_bankroll (float): Starting bankroll
        fixed_stake (bool/float): Use fixed stake instead of Kelly (False or stake amount)
        plot (bool): Whether to plot results
        
    Returns:
        dict: Profitability metrics including ROI and final bankroll
    """
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bet_history = []
    dates = []
    
    for i, match in test_data.iterrows():
        # Prepare match features
        X = match['features'].reshape(1, -1)
        
        # Get model prediction
        prediction = ensemble_predict(models, X)
        
        # Get actual outcome
        actual = match['outcome']
        
        # Get odds for this match
        match_odds = odds_data.loc[odds_data['match_id'] == match['match_id']].to_dict('records')[0]
        
        # Calculate recommended bets
        value_bets = find_value_bets(
            prediction, 
            match_odds['odds'], 
            format_type=match_odds['format'],
            bankroll=bankroll
        )
        
        # Simulate betting on all value bets
        for bet_type, bet_info in value_bets.items():
            # Determine stake amount
            if fixed_stake:
                stake = fixed_stake if isinstance(fixed_stake, (int, float)) else 50.0
            else:
                stake = bet_info['recommended_stake']
            
            # Ensure we don't bet more than what we have
            stake = min(stake, bankroll * 0.2)  # Max 20% per bet
            
            # Record bet
            bet_record = {
                'match_id': match['match_id'],
                'date': match['date'],
                'bet_type': bet_type,
                'odds': bet_info['odds'],
                'our_probability': bet_info['our_probability'],
                'implied_probability': bet_info['implied_probability'],
                'edge': bet_info['edge'],
                'stake': stake,
                'won': match[f'{bet_type}_result'] == 1,
                'profit': stake * (bet_info['odds'] - 1) if match[f'{bet_type}_result'] == 1 else -stake,
                'bankroll_before': bankroll
            }
            
            # Update bankroll
            bankroll += bet_record['profit']
            bet_record['bankroll_after'] = bankroll
            
            bet_history.append(bet_record)
            bankroll_history.append(bankroll)
            dates.append(match['date'])
    
    # Calculate profitability metrics
    bet_df = pd.DataFrame(bet_history)
    total_bets = len(bet_df)
    
    if total_bets > 0:
        total_stakes = bet_df['stake'].sum()
        total_profit = bet_df['profit'].sum()
        win_rate = (bet_df['won'] == True).mean()
        roi = total_profit / total_stakes if total_stakes > 0 else 0
        
        # Calculate win rate by bet type
        bet_type_analysis = bet_df.groupby('bet_type').agg({
            'stake': 'sum',
            'profit': 'sum',
            'won': ['count', 'mean']
        })
        
        # Plot results if requested
        if plot:
            plt.figure(figsize=(12, 8))
            
            # Bankroll evolution
            plt.subplot(2, 1, 1)
            plt.plot(range(len(bankroll_history)), bankroll_history, 'b-')
            plt.axhline(y=initial_bankroll, color='r', linestyle='--')
            plt.title('Bankroll Evolution')
            plt.xlabel('Bet Number')
            plt.ylabel('Bankroll ($)')
            plt.grid(True)
            
            # ROI by bet type
            plt.subplot(2, 1, 2)
            bet_types = bet_type_analysis.index
            rois = bet_type_analysis[('profit', 'sum')] / bet_type_analysis[('stake', 'sum')]
            
            plt.bar(bet_types, rois)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('ROI by Bet Type')
            plt.xlabel('Bet Type')
            plt.ylabel('ROI')
            plt.xticks(rotation=45)
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('profitability_analysis.png')
            plt.close()
        
        return {
            'initial_bankroll': initial_bankroll,
            'final_bankroll': bankroll,
            'profit': total_profit,
            'roi': roi,
            'win_rate': win_rate,
            'total_bets': total_bets,
            'total_stakes': total_stakes,
            'bankroll_history': bankroll_history,
            'bet_type_analysis': bet_type_analysis.to_dict(),
            'bet_history': bet_df
        }
    
    return {
        'initial_bankroll': initial_bankroll,
        'final_bankroll': bankroll,
        'profit': 0,
        'roi': 0,
        'win_rate': 0,
        'total_bets': 0,
        'error': 'No valid bets found'
    }

#-------------------------------------------------------------------------
# BACKTESTING AND CALIBRATION
#-------------------------------------------------------------------------

def calibrate_model_probabilities(models, calibration_data):
    """
    Calibrate model probabilities to match empirical outcomes.
    
    Args:
        models (list): List of trained models
        calibration_data (DataFrame): Data with features and known outcomes
        
    Returns:
        function: Calibration function to correct raw probabilities
    """
    # Get raw predictions for calibration set
    raw_probs = []
    actual_outcomes = []
    
    for i, match in calibration_data.iterrows():
        X = match['features'].reshape(1, -1)
        prediction = ensemble_predict(models, X, calibrate=False)
        
        if prediction:
            raw_probs.append(prediction['team1_win_probability'])
            actual_outcomes.append(match['outcome'])
    
    # Convert to numpy arrays
    raw_probs = np.array(raw_probs)
    actual_outcomes = np.array(actual_outcomes)
    
    # Break probabilities into bins and check actual win rates
    bins = 10
    bin_edges = np.linspace(0, 1, bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(raw_probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins-1)
    
    # Calculate actual win rates per bin
    bin_win_rates = []
    bin_counts = []
    
    for i in range(bins):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            win_rate = np.mean(actual_outcomes[mask])
            bin_win_rates.append(win_rate)
            bin_counts.append(np.sum(mask))
        else:
            bin_win_rates.append(np.nan)
            bin_counts.append(0)
    
    # Create calibration mapping
    bin_centers_valid = []
    bin_win_rates_valid = []
    
    for i in range(bins):
        if not np.isnan(bin_win_rates[i]) and bin_counts[i] >= 5:
            bin_centers_valid.append(bin_centers[i])
            bin_win_rates_valid.append(bin_win_rates[i])
    
    # If not enough data points, use a simple scaling approach
    if len(bin_centers_valid) < 3:
        print("Not enough data for full calibration, using simple scaling.")
        
        # Fit a simple scaling factor
        if raw_probs.size > 0 and actual_outcomes.size > 0:
            mean_raw_prob = np.mean(raw_probs)
            mean_actual = np.mean(actual_outcomes)
            
            if mean_raw_prob > 0:
                scaling_factor = mean_actual / mean_raw_prob
            else:
                scaling_factor = 1.0
                
            # Calibration function with simple scaling
            def calibrate_prob(raw_prob):
                calibrated = raw_prob * scaling_factor
                return max(0.01, min(0.99, calibrated))
        else:
            # Default no-op calibration
            def calibrate_prob(raw_prob):
                return raw_prob
    else:
        # Fit a logistic regression or isotonic regression for calibration
        from sklearn.isotonic import IsotonicRegression
        
        calibration_model = IsotonicRegression(out_of_bounds='clip')
        calibration_model.fit(np.array(bin_centers_valid), np.array(bin_win_rates_valid))
        
        def calibrate_prob(raw_prob):
            return max(0.01, min(0.99, calibration_model.predict([raw_prob])[0]))
    
    # Plot calibration curve if we have enough data
    if len(bin_centers_valid) >= 3:
        plt.figure(figsize=(10, 6))
        
        # Plot ideal calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        
        # Plot actual calibration points
        plt.scatter(bin_centers, bin_win_rates, s=[max(20, min(400, c*20)) for c in bin_counts], alpha=0.7)
        
        # Add point counts
        for i, (x, y, count) in enumerate(zip(bin_centers, bin_win_rates, bin_counts)):
            if not np.isnan(y) and count > 0:
                plt.annotate(f'{count}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Plot fitted calibration curve
        x_range = np.linspace(0, 1, 100)
        y_calibrated = [calibrate_prob(x) for x in x_range]
        plt.plot(x_range, y_calibrated, 'r-', label='Calibration Curve')
        
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Predicted Probability')
        plt.ylabel('Actual Win Rate')
        plt.title('Probability Calibration Curve')
        plt.legend(loc='best')
        plt.grid(True)
        
        plt.savefig('probability_calibration.png')
        plt.close()
    
    return calibrate_prob

def validate_calibration(models, validation_data, calibration_func=None):
    """
    Validate the calibration of model probabilities.
    
    Args:
        models (list): List of trained models
        validation_data (DataFrame): Data with features and known outcomes
        calibration_func (function): Function to calibrate raw probabilities
        
    Returns:
        dict: Calibration metrics including reliability and resolution
    """
    raw_probs = []
    calibrated_probs = []
    actual_outcomes = []
    
    for i, match in validation_data.iterrows():
        X = match['features'].reshape(1, -1)
        prediction = ensemble_predict(models, X, calibrate=False)
        
        if prediction:
            raw_prob = prediction['team1_win_probability']
            raw_probs.append(raw_prob)
            
            if calibration_func:
                calibrated_prob = calibration_func(raw_prob)
            else:
                calibrated_prob = raw_prob
                
            calibrated_probs.append(calibrated_prob)
            actual_outcomes.append(match['outcome'])
    
    # Convert to numpy arrays
    raw_probs = np.array(raw_probs)
    calibrated_probs = np.array(calibrated_probs)
    actual_outcomes = np.array(actual_outcomes)
    
    # Calculate Brier score (lower is better)
    raw_brier = np.mean((raw_probs - actual_outcomes) ** 2)
    calibrated_brier = np.mean((calibrated_probs - actual_outcomes) ** 2)
    
    # Calculate log loss
    def safe_log(x):
        return np.log(max(0.001, min(0.999, x)))
    
    raw_log_loss = -np.mean(actual_outcomes * safe_log(raw_probs) + 
                           (1 - actual_outcomes) * safe_log(1 - raw_probs))
    
    calibrated_log_loss = -np.mean(actual_outcomes * safe_log(calibrated_probs) + 
                                  (1 - actual_outcomes) * safe_log(1 - calibrated_probs))
    
    # Calculate reliability diagram data
    bins = 10
    bin_edges = np.linspace(0, 1, bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    raw_bin_indices = np.digitize(raw_probs, bin_edges) - 1
    raw_bin_indices = np.clip(raw_bin_indices, 0, bins-1)
    
    calibrated_bin_indices = np.digitize(calibrated_probs, bin_edges) - 1
    calibrated_bin_indices = np.clip(calibrated_bin_indices, 0, bins-1)
    
    # Calculate actual win rates per bin
    raw_bin_win_rates = []
    raw_bin_counts = []
    
    calibrated_bin_win_rates = []
    calibrated_bin_counts = []
    
    for i in range(bins):
        # Raw probabilities
        mask = (raw_bin_indices == i)
        if np.sum(mask) > 0:
            win_rate = np.mean(actual_outcomes[mask])
            raw_bin_win_rates.append(win_rate)
            raw_bin_counts.append(np.sum(mask))
        else:
            raw_bin_win_rates.append(np.nan)
            raw_bin_counts.append(0)
        
        # Calibrated probabilities
        mask = (calibrated_bin_indices == i)
        if np.sum(mask) > 0:
            win_rate = np.mean(actual_outcomes[mask])
            calibrated_bin_win_rates.append(win_rate)
            calibrated_bin_counts.append(np.sum(mask))
        else:
            calibrated_bin_win_rates.append(np.nan)
            calibrated_bin_counts.append(0)
    
    # Plot calibration curves
    plt.figure(figsize=(12, 8))
    
    # Raw probabilities calibration curve
    plt.subplot(2, 1, 1)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    # Plot raw probability calibration points
    plt.scatter(bin_centers, raw_bin_win_rates, 
                s=[max(20, min(400, c*20)) for c in raw_bin_counts], 
                alpha=0.7, 
                label='Raw Probabilities')
    
    # Add counts for raw probabilities
    for i, (x, y, count) in enumerate(zip(bin_centers, raw_bin_win_rates, raw_bin_counts)):
        if not np.isnan(y) and count > 0:
            plt.annotate(f'{count}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Win Rate')
    plt.title('Raw Probability Calibration')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Calibrated probabilities curve
    plt.subplot(2, 1, 2)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    # Plot calibrated probability points
    plt.scatter(bin_centers, calibrated_bin_win_rates, 
                s=[max(20, min(400, c*20)) for c in calibrated_bin_counts], 
                alpha=0.7, 
                label='Calibrated Probabilities')
    
    # Add counts for calibrated probabilities
    for i, (x, y, count) in enumerate(zip(bin_centers, calibrated_bin_win_rates, calibrated_bin_counts)):
        if not np.isnan(y) and count > 0:
            plt.annotate(f'{count}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Win Rate')
    plt.title('Calibrated Probability Calibration')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('calibration_validation.png')
    plt.close()
    
    return {
        'raw_brier_score': float(raw_brier),
        'calibrated_brier_score': float(calibrated_brier),
        'raw_log_loss': float(raw_log_loss),
        'calibrated_log_loss': float(calibrated_log_loss),
        'improvement': {
            'brier_score': float(raw_brier - calibrated_brier),
            'log_loss': float(raw_log_loss - calibrated_log_loss)
        },
        'reliability_diagram': {
            'bin_centers': bin_centers.tolist(),
            'raw_win_rates': [float(x) if not np.isnan(x) else None for x in raw_bin_win_rates],
            'calibrated_win_rates': [float(x) if not np.isnan(x) else None for x in calibrated_bin_win_rates],
            'raw_counts': raw_bin_counts,
            'calibrated_counts': calibrated_bin_counts
        }
    }

def backtest_betting_strategy(models, historical_data, odds_data, scaler=None, stable_features=None,
                             initial_bankroll=1000.0, kelly_fraction=0.25, min_edge=0.05,
                             calibration_data=None, use_calibration=True):
    """
    Backtest a betting strategy on historical match data.
    
    Args:
        models (list): List of trained models
        historical_data (DataFrame): Historical match data
        odds_data (DataFrame): Historical odds data
        scaler (object): Feature scaler
        stable_features (list): List of stable features
        initial_bankroll (float): Starting bankroll
        kelly_fraction (float): Conservative fraction of Kelly criterion
        min_edge (float): Minimum required edge for a bet
        calibration_data (DataFrame): Data to use for calibration
        use_calibration (bool): Whether to use probability calibration
        
    Returns:
        dict: Backtest results including profitability metrics
    """
    print("Starting betting strategy backtest...")
    
    # Split data into calibration, validation, and test sets if needed
    if calibration_data is None and use_calibration:
        # Use 20% of historical data for calibration if not provided
        data_size = len(historical_data)
        calibration_size = int(data_size * 0.2)
        validation_size = int(data_size * 0.2)
        test_size = data_size - calibration_size - validation_size
        
        # Shuffle data
        shuffled_indices = np.random.permutation(data_size)
        
        calibration_indices = shuffled_indices[:calibration_size]
        validation_indices = shuffled_indices[calibration_size:calibration_size+validation_size]
        test_indices = shuffled_indices[calibration_size+validation_size:]
        
        calibration_data = historical_data.iloc[calibration_indices].reset_index(drop=True)
        validation_data = historical_data.iloc[validation_indices].reset_index(drop=True)
        test_data = historical_data.iloc[test_indices].reset_index(drop=True)
    else:
        # Use all data for testing if no calibration needed
        test_data = historical_data
    
    # Calibrate probabilities if requested
    calibration_func = None
    if use_calibration and calibration_data is not None:
        print("Calibrating model probabilities...")
        calibration_func = calibrate_model_probabilities(models, calibration_data)
        
        # Validate calibration if we have validation data
        if 'validation_data' in locals():
            print("Validating probability calibration...")
            calibration_metrics = validate_calibration(models, validation_data, calibration_func)
            print(f"Calibration improved Brier score by {calibration_metrics['improvement']['brier_score']:.6f}")
    
    # Define a custom prediction function that applies calibration
    def predict_with_calibration(X):
        raw_prediction = ensemble_predict(models, X, calibrate=False)
        
        if raw_prediction and calibration_func:
            raw_prob = raw_prediction['team1_win_probability']
            calibrated_prob = calibration_func(raw_prob)
            
            # Update the prediction with calibrated probability
            raw_prediction['team1_win_probability'] = calibrated_prob
            raw_prediction['team2_win_probability'] = 1 - calibrated_prob
        
        return raw_prediction
    
    # Initialize tracking variables
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bet_history = []
    dates = []
    
    print(f"Starting backtest with {len(test_data)} matches...")
    
    # Iterate through test data chronologically
    for i, match in test_data.iterrows():
        if i % 10 == 0:
            print(f"Processing match {i+1}/{len(test_data)}...")
        
        try:
            # Prepare match features
            team1_name = match['team1_name']
            team2_name = match['team2_name']
            match_date = match['date']
            match_id = match['match_id']
            
            # Get model prediction
            X = match['features'].reshape(1, -1)
            prediction = predict_with_calibration(X)
            
            if not prediction:
                print(f"Failed to get prediction for match {match_id}.")
                continue
            
            # Get match odds
            match_odds = odds_data[odds_data['match_id'] == match_id].to_dict('records')
            
            if not match_odds:
                print(f"No odds data found for match {match_id}.")
                continue
                
            match_odds = match_odds[0]
            
            # Find value bets
            value_bets = find_value_bets(
                prediction, 
                match_odds['odds'], 
                format_type=match_odds['format'],
                bankroll=bankroll,
                kelly_fraction=kelly_fraction,
                min_edge=min_edge
            )
            
            # Record match data regardless of whether we bet
            match_record = {
                'match_id': match_id,
                'date': match_date,
                'team1': team1_name,
                'team2': team2_name,
                'predicted_prob': prediction['team1_win_probability'],
                'actual_outcome': match['outcome'],
                'format': match_odds['format'],
                'num_value_bets': len(value_bets)
            }
            
            # Process each value bet
            for bet_type, bet_info in value_bets.items():
                # Extract real outcome for this bet type
                bet_result = match.get(f'{bet_type}_result', 0)
                
                # Calculate stake based on Kelly criterion
                stake = bet_info['recommended_stake']
                
                # Cap stake at 5% of bankroll as a safety measure
                stake = min(stake, bankroll * 0.05)
                
                # Calculate profit/loss
                if bet_result == 1:  # Bet won
                    profit = stake * (bet_info['odds'] - 1)
                else:  # Bet lost
                    profit = -stake
                
                # Update bankroll
                bankroll += profit
                
                # Record bet details
                bet_record = {
                    'match_id': match_id,
                    'date': match_date,
                    'team1': team1_name,
                    'team2': team2_name,
                    'bet_type': bet_type,
                    'odds': bet_info['odds'],
                    'our_probability': bet_info['our_probability'],
                    'implied_probability': bet_info['implied_probability'],
                    'edge': bet_info['edge'],
                    'edge_percent': bet_info['edge_percent'],
                    'stake': stake,
                    'result': bet_result,
                    'profit': profit,
                    'bankroll_after': bankroll
                }
                
                bet_history.append(bet_record)
                bankroll_history.append(bankroll)
                dates.append(match_date)
                
                # Add bet details to match record
                match_record[f'bet_{bet_type}_odds'] = bet_info['odds']
                match_record[f'bet_{bet_type}_stake'] = stake
                match_record[f'bet_{bet_type}_profit'] = profit
        
        except Exception as e:
            print(f"Error processing match {match_id if 'match_id' in locals() else i}: {e}")
            continue
    
    # Calculate overall metrics
    if bet_history:
        bet_df = pd.DataFrame(bet_history)
        
        total_bets = len(bet_df)
        total_matches_bet = len(bet_df['match_id'].unique())
        total_matches = len(test_data)
        
        if total_bets > 0:
            total_stake = bet_df['stake'].sum()
            total_profit = bet_df['profit'].sum()
            
            overall_roi = total_profit / total_stake if total_stake > 0 else 0
            profit_per_bet = total_profit / total_bets
            profit_per_match = total_profit / total_matches_bet
            
            win_rate = (bet_df['result'] == 1).mean()
            
            # Calculate average odds
            avg_odds = bet_df['odds'].mean()
            
            # Calculate metrics by bet type
            bet_type_metrics = bet_df.groupby('bet_type').agg({
                'stake': 'sum',
                'profit': 'sum',
                'result': ['count', 'mean'],
                'odds': 'mean'
            })
            
            # Calculate ROI by bet type
            bet_type_roi = bet_type_metrics[('profit', 'sum')] / bet_type_metrics[('stake', 'sum')]
            bet_type_metrics['roi'] = bet_type_roi
            
            # Generate visualizations
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Bankroll evolution
            plt.subplot(2, 2, 1)
            plt.plot(range(len(bankroll_history)), bankroll_history, 'b-')
            plt.axhline(y=initial_bankroll, color='r', linestyle='--')
            plt.title('Bankroll Evolution')
            plt.xlabel('Bet Number')
            plt.ylabel('Bankroll ($)')
            plt.grid(True)
            
            # Plot 2: ROI by bet type
            plt.subplot(2, 2, 2)
            bet_types = bet_type_roi.index
            
            plt.bar(bet_types, bet_type_roi)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('ROI by Bet Type')
            plt.xlabel('Bet Type')
            plt.ylabel('ROI')
            plt.xticks(rotation=45)
            plt.grid(True)
            
            # Plot 3: Bet count by bet type
            plt.subplot(2, 2, 3)
            bet_counts = bet_type_metrics[('result', 'count')]
            
            plt.bar(bet_types, bet_counts)
            plt.title('Number of Bets by Type')
            plt.xlabel('Bet Type')
            plt.ylabel('Number of Bets')
            plt.xticks(rotation=45)
            plt.grid(True)
            
            # Plot 4: Win rate vs. predicted probability
            plt.subplot(2, 2, 4)
            
            # Create bins for predicted probabilities
            prob_bins = np.linspace(0, 1, 11)
            bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
            
            # Group bets by predicted probability
            bet_df['prob_bin'] = pd.cut(bet_df['our_probability'], prob_bins)
            bin_stats = bet_df.groupby('prob_bin').agg({
                'result': ['count', 'mean']
            })
            
            bin_win_rates = bin_stats[('result', 'mean')]
            bin_counts = bin_stats[('result', 'count')]
            
            # Plot win rates by probability bin
            plt.scatter(bin_centers, bin_win_rates, s=bin_counts*5, alpha=0.7)
            plt.plot([0, 1], [0, 1], 'k--')
            
            plt.title('Win Rate vs. Predicted Probability')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Actual Win Rate')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('backtest_results.png')
            plt.close()
            
            # Return backtest results
            return {
                'summary': {
                    'initial_bankroll': initial_bankroll,
                    'final_bankroll': bankroll,
                    'total_profit': total_profit,
                    'overall_roi': overall_roi,
                    'win_rate': win_rate,
                    'total_bets': total_bets,
                    'total_matches_bet': total_matches_bet,
                    'percentage_of_matches_bet': total_matches_bet / total_matches,
                    'total_stake': total_stake,
                    'avg_odds': avg_odds,
                    'profit_per_bet': profit_per_bet,
                    'profit_per_match': profit_per_match
                },
                'bet_type_metrics': bet_type_metrics.to_dict(),
                'bankroll_history': bankroll_history,
                'bet_history': bet_df.to_dict('records'),
                'final_bankroll_multiple': bankroll / initial_bankroll
            }
        else:
            return {
                'summary': {
                    'initial_bankroll': initial_bankroll,
                    'final_bankroll': initial_bankroll,
                    'total_profit': 0,
                    'overall_roi': 0,
                    'win_rate': 0,
                    'total_bets': 0,
                    'total_matches_bet': 0,
                    'percentage_of_matches_bet': 0,
                    'total_stake': 0
                },
                'error': 'No bets were placed during the backtest'
            }
    else:
        return {
            'summary': {
                'initial_bankroll': initial_bankroll,
                'final_bankroll': initial_bankroll,
                'total_profit': 0,
                'overall_roi': 0,
                'total_bets': 0
            },
            'error': 'No bet history was created during the backtest'
        }

#-------------------------------------------------------------------------
# BETTING STRATEGY APPLICATION
#-------------------------------------------------------------------------

def get_best_format_for_match(team1_stats, team2_stats):
    """
    Determine the best match format for optimal betting based on team characteristics.
    
    Args:
        team1_stats (dict): Team 1 statistics
        team2_stats (dict): Team 2 statistics
        
    Returns:
        str: Recommended format ("bo1", "bo3", "bo5")
    """
    # Calculate relevant metrics
    team1_consistency = team1_stats.get('team_consistency', 0.5)
    team2_consistency = team2_stats.get('team_consistency', 0.5)
    
    skill_gap = abs(team1_stats.get('win_rate', 0.5) - team2_stats.get('win_rate', 0.5))
    
    # Bo1 is better for underdogs since there's more variance
    # Bo3/Bo5 better for favorites since skill wins out over more maps
    
    if skill_gap > 0.2:
        # Large skill difference
        if team1_stats.get('win_rate', 0.5) > team2_stats.get('win_rate', 0.5):
            # Team 1 is significantly better
            if team1_consistency > 0.7:
                return "bo3"  # Team 1 is consistent, bet on them in bo3
            else:
                return "bo1"  # Team 1 is inconsistent, bo1 is riskier but higher odds
        else:
            # Team 2 is significantly better
            if team2_consistency < 0.7:
                return "bo1"  # Team 2 is inconsistent, opportunity for upset in bo1
            else:
                return "bo5"  # Team 2 is consistent, bo5 may offer value on underdog maps
    else:
        # Teams are close in skill
        avg_consistency = (team1_consistency + team2_consistency) / 2
        
        if avg_consistency > 0.8:
            return "bo5"  # Both teams consistent, likely to have competitive maps
        elif avg_consistency > 0.6:
            return "bo3"  # Moderate consistency, bo3 offers balanced variance
        else:
            return "bo1"  # Both teams inconsistent, high variance match

def apply_betting_strategy(team1_name, team2_name, api_options=None, bookmaker_odds=None,
                          format_type="bo3", bankroll=1000.0, kelly_fraction=0.25,
                          min_edge=0.05, confidence_threshold=0.05, max_bet_percentage=0.05,
                          models=None, scaler=None, stable_features=None, calibration_func=None):
    """
    Apply the betting strategy to a specific match.
    
    Args:
        team1_name (str): Name of team 1
        team2_name (str): Name of team 2
        api_options (dict): Options for API calls
        bookmaker_odds (dict): Odds from bookmakers
        format_type (str): Match format (bo1, bo3, bo5)
        bankroll (float): Current bankroll
        kelly_fraction (float): Conservative Kelly stake fraction
        min_edge (float): Minimum edge required for a bet
        confidence_threshold (float): Required confidence margin
        max_bet_percentage (float): Maximum percentage of bankroll to bet
        models (list): Loaded ensemble models
        scaler (object): Feature scaler
        stable_features (list): Stable feature names
        calibration_func (function): Probability calibration function
        
    Returns:
        dict: Recommended bets and analysis
    """
    # Load models if not provided
    if models is None:
        model_paths = [
            'valorant_model_fold_1.h5', 
            'valorant_model_fold_2.h5', 
            'valorant_model_fold_3.h5', 
            'valorant_model_fold_4.h5', 
            'valorant_model_fold_5.h5'
        ]
        scaler_path = 'ensemble_scaler.pkl'
        features_path = 'stable_features.pkl'
        
        models, scaler, stable_features = load_ensemble_models(model_paths, scaler_path, features_path)
    
    print(f"Analyzing match: {team1_name} vs {team2_name} ({format_type})")
    
    try:
        # Get team stats
        team1_id = get_team_id(team1_name)
        team2_id = get_team_id(team2_name)
        
        if not team1_id or not team2_id:
            return {
                'error': 'Failed to find one or both teams',
                'team1_id_found': team1_id is not None,
                'team2_id_found': team2_id is not None
            }
        
        # Fetch team stats
        team1_stats = {}
        team2_stats = {}
        
        # Load team match history
        team1_history = fetch_team_match_history(team1_id)
        team2_history = fetch_team_match_history(team2_id)
        
        if team1_history and team2_history:
            # Parse match data
            team1_matches = parse_match_data(team1_history, team1_name)
            team2_matches = parse_match_data(team2_history, team2_name)
            
            # Get player stats
            team1_player_stats = fetch_team_player_stats(team1_id)
            team2_player_stats = fetch_team_player_stats(team2_id)
            
            # Calculate team stats
            team1_stats = calculate_team_stats(team1_matches, team1_player_stats, include_economy=True)
            team2_stats = calculate_team_stats(team2_matches, team2_player_stats, include_economy=True)
            
            # Add team names
            team1_stats['team_name'] = team1_name
            team2_stats['team_name'] = team2_name
        else:
            return {
                'error': 'Failed to fetch match history for one or both teams',
                'team1_history_found': team1_history is not None,
                'team2_history_found': team2_history is not None
            }
        
        # Prepare features for prediction
        match_features = prepare_match_features(team1_stats, team2_stats, stable_features, scaler)
        
        if match_features is None:
            return {
                'error': 'Failed to prepare match features',
                'team1_stats_available': team1_stats != {},
                'team2_stats_available': team2_stats != {}
            }
        
        # Get prediction from ensemble
        prediction = ensemble_predict(models, match_features, calibrate=True, confidence=True)
        
        if not prediction:
            return {
                'error': 'Failed to get prediction from models'
            }
        
        # Apply calibration if available
        if calibration_func:
            raw_prob = prediction['team1_win_probability']
            calibrated_prob = calibration_func(raw_prob)
            
            prediction['raw_probability'] = raw_prob
            prediction['team1_win_probability'] = calibrated_prob
            prediction['team2_win_probability'] = 1 - calibrated_prob
        
        # Calculate series outcome probabilities
        series_probs = predict_series_outcome(prediction['team1_win_probability'], format_type)
        
        # If no bookmaker odds provided, give best format to bet
        if not bookmaker_odds:
            suggested_format = get_best_format_for_match(team1_stats, team2_stats)
            
            return {
                'match': {
                    'team1': team1_name,
                    'team2': team2_name,
                    'current_format': format_type,
                    'suggested_format': suggested_format
                },
                'prediction': prediction,
                'series_probabilities': series_probs,
                'message': 'No bookmaker odds provided. Please input odds to get betting recommendations.'
            }
        
        # Find value bets
        value_bets = find_value_bets(
            prediction, 
            bookmaker_odds, 
            format_type=format_type,
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            min_edge=min_edge
        )
        
        # Additional filtering based on confidence
        filtered_bets = {}
        for bet_type, bet_info in value_bets.items():
            # Check if confidence interval supports the value assessment
            confidence = bet_info.get('confidence', {})
            lower_bound = confidence.get('lower_bound', 0)
            
            implied_prob = bet_info['implied_probability']
            
            # Only recommend bet if the lower bound of our confidence interval still gives us an edge
            edge_with_confidence = lower_bound - implied_prob
            
            if edge_with_confidence >= confidence_threshold:
                # Cap bet at maximum percentage
                max_bet = bankroll * max_bet_percentage
                recommended_stake = min(bet_info['recommended_stake'], max_bet)
                
                bet_info['recommended_stake'] = recommended_stake
                bet_info['edge_with_confidence'] = edge_with_confidence
                
                filtered_bets[bet_type] = bet_info
        
        # Prepare team comparison for output
        team_comparison = []
        
        # Add win rate comparison
        team_comparison.append({
            'metric': 'Win Rate',
            'team1_value': team1_stats.get('win_rate', 0) * 100,
            'team2_value': team2_stats.get('win_rate', 0) * 100,
            'difference': (team1_stats.get('win_rate', 0) - team2_stats.get('win_rate', 0)) * 100
        })
        
        # Add recent form comparison
        team_comparison.append({
            'metric': 'Recent Form',
            'team1_value': team1_stats.get('recent_form', 0) * 100,
            'team2_value': team2_stats.get('recent_form', 0) * 100,
            'difference': (team1_stats.get('recent_form', 0) - team2_stats.get('recent_form', 0)) * 100
        })
        
        # Add player ratings comparison if available
        if 'avg_player_rating' in team1_stats and 'avg_player_rating' in team2_stats:
            team_comparison.append({
                'metric': 'Player Rating',
                'team1_value': team1_stats.get('avg_player_rating', 0),
                'team2_value': team2_stats.get('avg_player_rating', 0),
                'difference': team1_stats.get('avg_player_rating', 0) - team2_stats.get('avg_player_rating', 0)
            })
        
        # Add economy stats comparison if available
        if 'pistol_win_rate' in team1_stats and 'pistol_win_rate' in team2_stats:
            team_comparison.append({
                'metric': 'Pistol Win Rate',
                'team1_value': team1_stats.get('pistol_win_rate', 0) * 100,
                'team2_value': team2_stats.get('pistol_win_rate', 0) * 100,
                'difference': (team1_stats.get('pistol_win_rate', 0) - team2_stats.get('pistol_win_rate', 0)) * 100
            })
        
        # Return comprehensive analysis and recommendations
        return {
            'match': {
                'team1': team1_name,
                'team2': team2_name,
                'format': format_type,
            },
            'prediction': {
                'team1_win_probability': prediction['team1_win_probability'],
                'team2_win_probability': prediction['team2_win_probability'],
                'confidence_interval': prediction.get('confidence_interval', {}),
                'model_confidence': 1 - prediction.get('std_dev', 0)
            },
            'series_probabilities': series_probs,
            'team_comparison': team_comparison,
            'recommended_bets': filtered_bets,
            'all_value_bets': value_bets,
            'bankroll': bankroll,
            'max_exposure': sum(bet['recommended_stake'] for bet in filtered_bets.values())
        }
    
    except Exception as e:
        print(f"Error in apply_betting_strategy: {e}")
        return {
            'error': f'An error occurred: {str(e)}'
        }

def build_historical_dataset(team_data_collection, odds_data=None):
    """
    Build a dataset for backtesting from team data and historical odds.
    
    Args:
        team_data_collection (dict): Team statistics collection
        odds_data (DataFrame): Historical odds data
        
    Returns:
        DataFrame: Dataset for backtesting
    """
    historical_data = []
    
    for team_name, team_data in team_data_collection.items():
        matches = team_data.get('matches', [])
        
        for match in matches:
            opponent_name = match.get('opponent_name')
            
            # Skip if we don't have data for the opponent
            if opponent_name not in team_data_collection:
                continue
            
            # Get match ID
            match_id = match.get('match_id')
            if not match_id:
                continue
            
            # Get match date
            match_date = match.get('date')
            if not match_date:
                continue
            
            # Get match result (from team1's perspective)
            team1_won = match.get('team_won', False)
            
            # Get team stats
            team1_stats = team_data
            team2_stats = team_data_collection[opponent_name]
            
            # Prepare features for this match
            features = prepare_data_for_model(team1_stats, team2_stats)
            
            if not features:
                continue
                
            # Convert features to a numpy array
            X = np.array(list(features.values())).reshape(1, -1)
            
            # Get map count if available
            map_score = match.get('map_score', '')
            if map_score and ':' in map_score:
                try:
                    team1_maps, team2_maps = map_score.split(':')
                    team1_maps = int(team1_maps)
                    team2_maps = int(team2_maps)
                    
                    # Determine format (bo1, bo3, bo5) from map count
                    if team1_maps + team2_maps == 1:
                        match_format = 'bo1'
                    elif team1_maps + team2_maps <= 3:
                        match_format = 'bo3'
                    else:
                        match_format = 'bo5'
                        
                    # Calculate result types
                    if match_format == 'bo1':
                        team1_win_match = team1_maps > team2_maps
                        team2_win_match = team2_maps > team1_maps
                        
                        maps_over_0_5 = team1_maps + team2_maps > 0.5
                        
                        # Add results for different bet types
                        result_dict = {
                            'team1_win_series': 1 if team1_win_match else 0,
                            'team2_win_series': 1 if team2_win_match else 0,
                            'maps_over_0.5': 1 if maps_over_0_5 else 0,
                            'maps_under_0.5': 1 if not maps_over_0_5 else 0
                        }
                    
                    elif match_format == 'bo3':
                        team1_win_match = team1_maps > team2_maps
                        team2_win_match = team2_maps > team1_maps
                        
                        team1_win_2_0 = team1_maps == 2 and team2_maps == 0
                        team2_win_2_0 = team2_maps == 2 and team1_maps == 0
                        
                        team1_win_2_1 = team1_maps == 2 and team2_maps == 1
                        team2_win_2_1 = team2_maps == 2 and team1_maps == 1
                        
                        maps_over_2_5 = team1_maps + team2_maps > 2.5
                        maps_under_2_5 = team1_maps + team2_maps < 2.5
                        
                        maps_over_1_5 = team1_maps + team2_maps > 1.5
                        maps_under_1_5 = team1_maps + team2_maps < 1.5
                        
                        # Add results for different bet types
                        result_dict = {
                            'team1_win_series': 1 if team1_win_match else 0,
                            'team2_win_series': 1 if team2_win_match else 0,
                            'team1_win_2_0': 1 if team1_win_2_0 else 0,
                            'team2_win_2_0': 1 if team2_win_2_0 else 0,
                            'team1_win_2_1': 1 if team1_win_2_1 else 0,
                            'team2_win_2_1': 1 if team2_win_2_1 else 0,
                            'maps_over_2.5': 1 if maps_over_2_5 else 0,
                            'maps_under_2.5': 1 if maps_under_2_5 else 0,
                            'maps_over_1.5': 1 if maps_over_1_5 else 0,
                            'maps_under_1.5': 1 if maps_under_1_5 else 0
                        }
                    
                    elif match_format == 'bo5':
                        team1_win_match = team1_maps > team2_maps
                        team2_win_match = team2_maps > team1_maps
                        
                        team1_win_3_0 = team1_maps == 3 and team2_maps == 0
                        team2_win_3_0 = team2_maps == 3 and team1_maps == 0
                        
                        team1_win_3_1 = team1_maps == 3 and team2_maps == 1
                        team2_win_3_1 = team2_maps == 3 and team1_maps == 1
                        
                        team1_win_3_2 = team1_maps == 3 and team2_maps == 2
                        team2_win_3_2 = team2_maps == 3 and team1_maps == 2
                        
                        maps_over_3_5 = team1_maps + team2_maps > 3.5
                        maps_under_3_5 = team1_maps + team2_maps < 3.5
                        
                        maps_over_4_5 = team1_maps + team2_maps > 4.5
                        maps_under_4_5 = team1_maps + team2_maps < 4.5
                        
                        # Add results for different bet types
                        result_dict = {
                            'team1_win_series': 1 if team1_win_match else 0,
                            'team2_win_series': 1 if team2_win_match else 0,
                            'team1_win_3_0': 1 if team1_win_3_0 else 0,
                            'team2_win_3_0': 1 if team2_win_3_0 else 0,
                            'team1_win_3_1': 1 if team1_win_3_1 else 0,
                            'team2_win_3_1': 1 if team2_win_3_1 else 0,
                            'team1_win_3_2': 1 if team1_win_3_2 else 0,
                            'team2_win_3_2': 1 if team2_win_3_2 else 0,
                            'maps_over_3.5': 1 if maps_over_3_5 else 0,
                            'maps_under_3.5': 1 if maps_under_3_5 else 0,
                            'maps_over_4.5': 1 if maps_over_4_5 else 0,
                            'maps_under_4.5': 1 if maps_under_4_5 else 0
                        }
                    
                    else:
                        # Unknown format
                        result_dict = {
                            'team1_win_series': 1 if team1_won else 0,
                            'team2_win_series': 1 if not team1_won else 0
                        }
                    
                    # Prepare match entry for the dataset
                    match_entry = {
                        'match_id': match_id,
                        'date': match_date,
                        'team1_name': team_name,
                        'team2_name': opponent_name,
                        'features': X[0],  # Save the features as a numpy array
                        'outcome': 1 if team1_won else 0,  # 1 if team1 won, 0 if team2 won
                        'format': match_format
                    }
                    
                    # Add results for each bet type
                    for bet_type, result in result_dict.items():
                        match_entry[f'{bet_type}_result'] = result
                    
                    # Add odds if available
                    if odds_data is not None:
                        match_odds = odds_data[odds_data['match_id'] == match_id]
                        
                        if not match_odds.empty:
                            match_odds = match_odds.iloc[0]
                            
                            for bet_type in result_dict.keys():
                                if bet_type in match_odds:
                                    match_entry[f'{bet_type}_odds'] = match_odds[bet_type]
                    
                    historical_data.append(match_entry)
                    
                except (ValueError, TypeError) as e:
                    # Skip this match if there's an error parsing the map score
                    print(f"Error parsing map score for match {match_id}: {e}")
                    continue
    
    # Convert to DataFrame
    if historical_data:
        historical_df = pd.DataFrame(historical_data)
        return historical_df
    else:
        return pd.DataFrame()

def prepare_new_matches_for_prediction(upcoming_matches, team_data_collection, models, scaler, stable_features, calibration_func=None):
    """
    Prepare upcoming matches for prediction and betting.
    
    Args:
        upcoming_matches (list): List of upcoming matches
        team_data_collection (dict): Team statistics collection
        models (list): Loaded ensemble models
        scaler (object): Feature scaler
        stable_features (list): Stable feature names
        calibration_func (function): Probability calibration function
        
    Returns:
        list: Prepared matches with predictions
    """
    prepared_matches = []
    
    for match in upcoming_matches:
        try:
            team1_name = match.get('team1', {}).get('name')
            team2_name = match.get('team2', {}).get('name')
            
            match_id = match.get('id')
            match_date = match.get('date')
            
            # Skip if missing team names
            if not team1_name or not team2_name:
                print(f"Skipping match {match_id}: Missing team names")
                continue
            
            # Skip if teams are not in our data collection
            if team1_name not in team_data_collection or team2_name not in team_data_collection:
                print(f"Skipping match {match_id}: One or both teams not in data collection")
                continue
            
            # Get team stats
            team1_stats = team_data_collection[team1_name]
            team2_stats = team_data_collection[team2_name]
            
            # Prepare features
            match_features = prepare_match_features(team1_stats, team2_stats, stable_features, scaler)
            
            if match_features is None:
                print(f"Skipping match {match_id}: Failed to prepare features")
                continue
            
            # Get prediction
            prediction = ensemble_predict(models, match_features, calibrate=True, confidence=True)
            
            if not prediction:
                print(f"Skipping match {match_id}: Failed to get prediction")
                continue
            
            # Apply calibration if available
            if calibration_func:
                raw_prob = prediction['team1_win_probability']
                calibrated_prob = calibration_func(raw_prob)
                
                prediction['raw_probability'] = raw_prob
                prediction['team1_win_probability'] = calibrated_prob
                prediction['team2_win_probability'] = 1 - calibrated_prob
            
            # Determine match format
            # Default to bo3 if not specified
            format_type = match.get('format', 'bo3')
            
            # Calculate series outcome probabilities
            series_probs = predict_series_outcome(prediction['team1_win_probability'], format_type)
            
            # Prepare match entry
            match_entry = {
                'match_id': match_id,
                'date': match_date,
                'team1': team1_name,
                'team2': team2_name,
                'format': format_type,
                'prediction': prediction,
                'series_probabilities': series_probs
            }
            
            prepared_matches.append(match_entry)
            
        except Exception as e:
            print(f"Error preparing match {match.get('id', 'unknown')}: {e}")
            continue
    
    return prepared_matches

def simulate_betting_strategy(prediction_model, initial_bankroll=1000.0, num_bets=1000, 
                             win_rate=0.56, avg_odds=2.0, kelly_fraction=0.25, bankrupt_threshold=10.0,
                             plot=True):
    """
    Simulate a betting strategy over a large number of bets to assess long-term profitability.
    
    Args:
        prediction_model (object): Prediction model or function
        initial_bankroll (float): Starting bankroll
        num_bets (int): Number of bets to simulate
        win_rate (float): Average win rate (model accuracy)
        avg_odds (float): Average odds for bets
        kelly_fraction (float): Fraction of Kelly to use
        bankrupt_threshold (float): Minimum bankroll to continue betting
        plot (bool): Whether to generate plots
        
    Returns:
        dict: Simulation results
    """
    print(f"Simulating betting strategy with {num_bets} bets...")
    print(f"Parameters: Win Rate = {win_rate:.2f}, Avg Odds = {avg_odds:.2f}, Kelly Fraction = {kelly_fraction:.2f}")
    
    # Initialize simulation variables
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    
    bet_results = []
    bet_sizes = []
    
    # Track metrics
    num_wins = 0
    num_losses = 0
    total_stake = 0
    max_bankroll = bankroll
    min_bankroll = bankroll
    max_drawdown = 0
    current_drawdown = 0
    
    # Generate random outcomes based on win rate
    np.random.seed(42)  # For reproducibility
    outcomes = np.random.random(num_bets) < win_rate
    
    # Simulate each bet
    for i in range(num_bets):
        # Skip if bankroll is too low (effectively bankrupt)
        if bankroll < bankrupt_threshold:
            print(f"Simulation stopped at bet {i} due to insufficient bankroll (${bankroll:.2f})")
            break
        
        # Vary the odds slightly for realism
        odds_variation = 0.2  # Vary by +/- 20%
        bet_odds = avg_odds * (1 + (np.random.random() - 0.5) * odds_variation)
        
        # Vary the model's predicted probability
        true_prob = win_rate
        model_error = 0.05  # 5% standard error in probability estimation
        predicted_prob = true_prob + np.random.normal(0, model_error)
        predicted_prob = max(0.01, min(0.99, predicted_prob))  # Keep within valid range
        
        # Calculate Kelly stake
        kelly_stake = kelly_criterion(predicted_prob, bet_odds, bankroll, kelly_fraction)
        
        # Cap stake at 5% of bankroll as safety measure
        stake = min(kelly_stake, bankroll * 0.05)
        bet_sizes.append(stake)
        total_stake += stake
        
        # Determine outcome
        won = outcomes[i]
        
        # Update bankroll
        if won:
            profit = stake * (bet_odds - 1)
            bankroll += profit
            num_wins += 1
            current_drawdown = 0
        else:
            bankroll -= stake
            num_losses += 1
            current_drawdown += stake
        
        # Track bankroll history
        bankroll_history.append(bankroll)
        
        # Update max/min bankroll
        max_bankroll = max(max_bankroll, bankroll)
        min_bankroll = min(min_bankroll, bankroll)
        
        # Update max drawdown
        max_drawdown = max(max_drawdown, current_drawdown)
        
        # Record bet result
        bet_results.append({
            'bet_number': i + 1,
            'predicted_probability': predicted_prob,
            'true_probability': true_prob,
            'odds': bet_odds,
            'stake': stake,
            'won': won,
            'profit': profit if won else -stake,
            'bankroll': bankroll
        })
    
    # Calculate final metrics
    final_bankroll = bankroll
    total_bets = num_wins + num_losses
    
    if total_bets > 0:
        actual_win_rate = num_wins / total_bets
        roi = (final_bankroll - initial_bankroll) / total_stake if total_stake > 0 else 0
        
        # Calculate statistical significance of results
        expected_win_rate = 1 / avg_odds  # Break-even win rate for average odds
        p_value = calculate_betting_significance(actual_win_rate, expected_win_rate, total_bets)
        
        # Plot results if requested
        if plot:
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Bankroll evolution
            plt.subplot(2, 2, 1)
            plt.plot(range(len(bankroll_history)), bankroll_history, 'b-')
            plt.axhline(y=initial_bankroll, color='r', linestyle='--')
            plt.title('Bankroll Evolution')
            plt.xlabel('Bet Number')
            plt.ylabel('Bankroll ($)')
            plt.grid(True)
            
            # Plot 2: Bet size distribution
            plt.subplot(2, 2, 2)
            plt.hist(bet_sizes, bins=20, alpha=0.7)
            plt.title('Bet Size Distribution')
            plt.xlabel('Bet Size ($)')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # Plot 3: Cumulative ROI
            plt.subplot(2, 2, 3)
            
            cumulative_stake = np.cumsum([b['stake'] for b in bet_results])
            cumulative_profit = np.cumsum([b['profit'] for b in bet_results])
            cumulative_roi = cumulative_profit / cumulative_stake
            
            plt.plot(range(1, len(cumulative_roi) + 1), cumulative_roi, 'g-')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Cumulative ROI')
            plt.xlabel('Bet Number')
            plt.ylabel('ROI')
            plt.grid(True)
            
            # Plot 4: Win rate confidence interval
            plt.subplot(2, 2, 4)
            
            # Calculate confidence intervals over time
            window_size = 50
            win_rates = []
            confidence_intervals = []
            x_values = []
            
            for i in range(window_size, len(bet_results), window_size//2):
                window = bet_results[max(0, i-window_size):i]
                wins = sum(1 for b in window if b['won'])
                window_win_rate = wins / len(window)
                
                # 95% confidence interval
                ci = 1.96 * np.sqrt(window_win_rate * (1 - window_win_rate) / len(window))
                
                win_rates.append(window_win_rate)
                confidence_intervals.append(ci)
                x_values.append(i)
            
            plt.errorbar(x_values, win_rates, yerr=confidence_intervals, fmt='o-')
            plt.axhline(y=win_rate, color='r', linestyle='--', label='Target Win Rate')
            plt.axhline(y=expected_win_rate, color='g', linestyle='--', label='Break-even Win Rate')
            
            plt.title('Win Rate with 95% Confidence Interval')
            plt.xlabel('Bet Number')
            plt.ylabel('Win Rate')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('simulation_results.png')
            plt.close()
            
        # Generate summary
        return {
            'summary': {
                'initial_bankroll': initial_bankroll,
                'final_bankroll': final_bankroll,
                'profit': final_bankroll - initial_bankroll,
                'total_bets': total_bets,
                'wins': num_wins,
                'losses': num_losses,
                'actual_win_rate': actual_win_rate,
                'target_win_rate': win_rate,
                'break_even_win_rate': expected_win_rate,
                'total_stake': total_stake,
                'roi': roi,
                'max_bankroll': max_bankroll,
                'min_bankroll': min_bankroll,
                'max_drawdown': max_drawdown,
                'max_drawdown_percentage': max_drawdown / max_bankroll if max_bankroll > 0 else 0,
                'p_value': p_value,
                'statistically_significant': p_value < 0.05
            },
            'bankroll_history': bankroll_history,
            'bet_results': bet_results
        }
    else:
        return {
            'error': 'No bets were placed during simulation',
            'initial_bankroll': initial_bankroll,
            'final_bankroll': final_bankroll
        }

def calculate_betting_significance(observed_win_rate, expected_win_rate, sample_size):
    """
    Calculate the statistical significance (p-value) of betting results.
    
    Args:
        observed_win_rate (float): Actual win rate achieved
        expected_win_rate (float): Expected win rate (e.g., break-even rate)
        sample_size (int): Number of bets
        
    Returns:
        float: p-value (lower = more significant)
    """
    # Calculate the standard error of the observed win rate
    std_error = np.sqrt(expected_win_rate * (1 - expected_win_rate) / sample_size)
    
    # Calculate the z-score
    z_score = (observed_win_rate - expected_win_rate) / std_error
    
    # Calculate the p-value (one-tailed test since we care if we beat the expected win rate)
    p_value = 1 - norm.cdf(z_score)
    
    return p_value

def run_monte_carlo_simulation(team1_stats, team2_stats, format_type="bo3", num_simulations=10000,
                              models=None, scaler=None, stable_features=None, calibration_func=None):
    """
    Run Monte Carlo simulations to estimate probabilities for different match outcomes.
    
    Args:
        team1_stats (dict): Team 1 statistics
        team2_stats (dict): Team 2 statistics
        format_type (str): Match format (bo1, bo3, bo5)
        num_simulations (int): Number of simulations to run
        models (list): Trained models
        scaler (object): Feature scaler
        stable_features (list): Stable feature names
        calibration_func (function): Probability calibration function
        
    Returns:
        dict: Simulation results
    """
    print(f"Running {num_simulations} Monte Carlo simulations for {team1_stats['team_name']} vs {team2_stats['team_name']} ({format_type})")
    
    # Prepare features
    match_features = prepare_match_features(team1_stats, team2_stats, stable_features, scaler)
    
    if match_features is None:
        return {'error': 'Failed to prepare match features'}
    
    # Get base prediction
    base_prediction = ensemble_predict(models, match_features, calibrate=True, confidence=True)
    
    if not base_prediction:
        return {'error': 'Failed to get base prediction'}
    
    # Extract win probability and standard deviation
    base_prob = base_prediction['team1_win_probability']
    std_dev = base_prediction.get('std_dev', 0.1)  # Default to 0.1 if not available
    
    # Apply calibration if available
    if calibration_func:
        base_prob = calibration_func(base_prob)
    
    # Initialize simulation results
    simulation_results = {
        'team1_win_series': 0,
        'team2_win_series': 0
    }
    
    # Add format-specific outcomes
    if format_type == "bo3":
        simulation_results.update({
            'team1_win_2_0': 0,
            'team2_win_2_0': 0,
            'team1_win_2_1': 0,
            'team2_win_2_1': 0,
            'maps_over_2.5': 0,
            'maps_under_2.5': 0
        })
    elif format_type == "bo5":
        simulation_results.update({
            'team1_win_3_0': 0,
            'team2_win_3_0': 0,
            'team1_win_3_1': 0,
            'team2_win_3_1': 0,
            'team1_win_3_2': 0,
            'team2_win_3_2': 0,
            'maps_over_3.5': 0,
            'maps_over_4.5': 0,
            'maps_under_3.5': 0,
            'maps_under_4.5': 0
        })
    
    # Run simulations
    for i in range(num_simulations):
        # Add random variation to the base probability to simulate uncertainty
        simulated_prob = np.random.normal(base_prob, std_dev)
        simulated_prob = max(0.01, min(0.99, simulated_prob))  # Keep within valid range
        
        # Simulate series outcome
        if format_type == "bo1":
            # Simple coin flip for bo1
            team1_wins = np.random.random() < simulated_prob
            
            if team1_wins:
                simulation_results['team1_win_series'] += 1
            else:
                simulation_results['team2_win_series'] += 1
                
        elif format_type == "bo3":
            # Simulate up to 3 maps
            map_results = []
            
            # First map
            map_results.append(np.random.random() < simulated_prob)
            
            # Second map (with slight adjustment based on first map result)
            momentum_factor = 0.05  # Small momentum factor
            map2_prob = simulated_prob
            
            if map_results[0]:  # Team 1 won first map
                map2_prob = min(0.99, simulated_prob + momentum_factor)
            else:  # Team 2 won first map
                map2_prob = max(0.01, simulated_prob - momentum_factor)
                
            map_results.append(np.random.random() < map2_prob)
            
            # Check if we need a third map
            if map_results[0] == map_results[1]:
                # One team won both maps
                team1_wins = map_results[0]
                maps_played = 2
                
                if team1_wins:
                    simulation_results['team1_win_series'] += 1
                    simulation_results['team1_win_2_0'] += 1
                else:
                    simulation_results['team2_win_series'] += 1
                    simulation_results['team2_win_2_0'] += 1
            else:
                # Need a third map
                # Further adjust probability based on map 2 result
                map3_prob = simulated_prob
                
                if map_results[1]:  # Team 1 won second map
                    map3_prob = min(0.99, simulated_prob + momentum_factor)
                else:  # Team 2 won second map
                    map3_prob = max(0.01, simulated_prob - momentum_factor)
                
                map_results.append(np.random.random() < map3_prob)
                maps_played = 3
                
                # Determine winner
                team1_maps = sum(map_results)
                team1_wins = team1_maps > 1
                
                if team1_wins:
                    simulation_results['team1_win_series'] += 1
                    simulation_results['team1_win_2_1'] += 1
                else:
                    simulation_results['team2_win_series'] += 1
                    simulation_results['team2_win_2_1'] += 1
            
            # Update map count stats
            if maps_played > 2.5:
                simulation_results['maps_over_2.5'] += 1
            else:
                simulation_results['maps_under_2.5'] += 1
                
        elif format_type == "bo5":
            # Simulate up to 5 maps
            map_results = []
            team1_maps = 0
            team2_maps = 0
            maps_played = 0
            
            # Simulate until one team gets 3 wins
            while team1_maps < 3 and team2_maps < 3 and maps_played < 5:
                # Adjust probability based on momentum
                momentum_factor = 0.03 * (team1_maps - team2_maps)  # Momentum increases with map difference
                current_prob = max(0.01, min(0.99, simulated_prob + momentum_factor))
                
                # Simulate map
                team1_wins_map = np.random.random() < current_prob
                
                if team1_wins_map:
                    team1_maps += 1
                else:
                    team2_maps += 1
                    
                map_results.append(team1_wins_map)
                maps_played += 1
            
            # Determine series winner
            team1_wins = team1_maps > team2_maps
            
            if team1_wins:
                simulation_results['team1_win_series'] += 1
                
                if team2_maps == 0:
                    simulation_results['team1_win_3_0'] += 1
                elif team2_maps == 1:
                    simulation_results['team1_win_3_1'] += 1
                elif team2_maps == 2:
                    simulation_results['team1_win_3_2'] += 1
            else:
                simulation_results['team2_win_series'] += 1
                
                if team1_maps == 0:
                    simulation_results['team2_win_3_0'] += 1
                elif team1_maps == 1:
                    simulation_results['team2_win_3_1'] += 1
                elif team1_maps == 2:
                    simulation_results['team2_win_3_2'] += 1
            
            # Update map count stats
            if maps_played > 3.5:
                simulation_results['maps_over_3.5'] += 1
            else:
                simulation_results['maps_under_3.5'] += 1
                
            if maps_played > 4.5:
                simulation_results['maps_over_4.5'] += 1
            else:
                simulation_results['maps_under_4.5'] += 1
    
    # Calculate probabilities
    for key in simulation_results:
        simulation_results[key] = simulation_results[key] / num_simulations
    
    # Add confidence intervals
    confidence_intervals = {}
    for key, prob in simulation_results.items():
        # 95% confidence interval using normal approximation
        std_error = np.sqrt(prob * (1 - prob) / num_simulations)
        margin_of_error = 1.96 * std_error
        
        confidence_intervals[key] = {
            'lower': max(0, prob - margin_of_error),
            'upper': min(1, prob + margin_of_error),
            'std_error': std_error
        }
    
    return {
        'base_probability': base_prob,
        'simulation_results': simulation_results,
        'confidence_intervals': confidence_intervals,
        'num_simulations': num_simulations
    }

#-------------------------------------------------------------------------
# BETTING CLI INTERFACE
#-------------------------------------------------------------------------

def load_betting_models(model_paths=None, scaler_path=None, features_path=None):
    """
    Load all models and data needed for betting predictions.
    
    Args:
        model_paths (list): Paths to trained models
        scaler_path (str): Path to scaler
        features_path (str): Path to stable features
        
    Returns:
        dict: Loaded betting models and data
    """
    # Default paths if not provided
    if model_paths is None:
        model_paths = [
            'valorant_model_fold_1.h5',
            'valorant_model_fold_2.h5',
            'valorant_model_fold_3.h5',
            'valorant_model_fold_4.h5',
            'valorant_model_fold_5.h5'
        ]
    
    if scaler_path is None:
        scaler_path = 'ensemble_scaler.pkl'
    
    if features_path is None:
        features_path = 'stable_features.pkl'
    
    # Load models, scaler and features
    models, scaler, stable_features = load_ensemble_models(model_paths, scaler_path, features_path)
    
    # Check if we have all components
    if not models:
        print("Failed to load ensemble models.")
        return None
    
    if scaler is None:
        print("Warning: No feature scaler loaded. Predictions may be inaccurate.")
    
    if stable_features is None or len(stable_features) == 0:
        print("Warning: No stable features loaded. Using all features.")
    
    return {
        'models': models,
        'scaler': scaler,
        'stable_features': stable_features
    }

def parse_odds_input(odds_string):
    """
    Parse odds input string into a dictionary.
    
    Args:
        odds_string (str): String containing odds in format "bet_type:odds,bet_type:odds,..."
        
    Returns:
        dict: Parsed odds dictionary
    """
    # Example input: "team1_win_series:1.75,team2_win_series:2.05,maps_over_2.5:1.90,maps_under_2.5:1.85"
    odds_dict = {}
    
    if not odds_string:
        return odds_dict
    
    # Split by commas to get each bet type
    bet_parts = odds_string.split(',')
    
    for part in bet_parts:
        if ':' in part:
            bet_type, odds_str = part.split(':', 1)
            bet_type = bet_type.strip()
            
            try:
                odds = float(odds_str.strip())
                odds_dict[bet_type] = odds
            except ValueError:
                print(f"Warning: Could not parse odds value '{odds_str}' for bet type '{bet_type}'")
    
    return odds_dict

def betting_cli():
    """
    Command-line interface for the betting strategy.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Valorant Betting Strategy CLI")
    
    # Add command line arguments
    parser.add_argument("--predict", action="store_true", help="Predict a specific match")
    parser.add_argument("--team1", type=str, help="First team name")
    parser.add_argument("--team2", type=str, help="Second team name")
    parser.add_argument("--format", type=str, default="bo3", choices=["bo1", "bo3", "bo5"], 
                      help="Match format (bo1, bo3, bo5)")
    parser.add_argument("--odds", type=str, help="Bookmaker odds in format 'bet_type:odds,bet_type:odds,...'")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Current bankroll")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (0.0-1.0)")
    parser.add_argument("--edge", type=float, default=0.05, help="Minimum edge required for bet")
    
    parser.add_argument("--backtest", action="store_true", help="Run betting strategy backtest")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of historical matches to use")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate probabilities in backtest")
    
    parser.add_argument("--simulation", action="store_true", help="Run Monte Carlo simulation for a match")
    parser.add_argument("--num-sims", type=int, default=10000, help="Number of simulations to run")
    
    parser.add_argument("--analyze-future", action="store_true", help="Analyze all upcoming matches")
    
    args = parser.parse_args()
    
    # Load betting models
    print("Loading betting models...")
    betting_models = load_betting_models()
    
    if betting_models is None:
        print("Failed to load betting models. Exiting.")
        return
    
    models = betting_models['models']
    scaler = betting_models['scaler']
    stable_features = betting_models['stable_features']
    
    # Load team data
    print("Loading team data...")
    team_data_collection = collect_team_data(include_player_stats=True, include_economy=True)
    
    if not team_data_collection:
        print("Failed to collect team data. Exiting.")
        return
    
    print(f"Loaded data for {len(team_data_collection)} teams.")
    
    # Process command
    if args.predict and args.team1 and args.team2:
        print(f"Predicting match: {args.team1} vs {args.team2} ({args.format})")
        
        # Parse odds if provided
        bookmaker_odds = None
        if args.odds:
            bookmaker_odds = parse_odds_input(args.odds)
        
        # Apply betting strategy
        result = apply_betting_strategy(
            args.team1, 
            args.team2, 
            format_type=args.format, 
            bankroll=args.bankroll,
            kelly_fraction=args.kelly,
            min_edge=args.edge,
            bookmaker_odds=bookmaker_odds,
            models=models,
            scaler=scaler,
            stable_features=stable_features
        )
        
        # Print results
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print("\n--- MATCH PREDICTION ---")
            print(f"Teams: {result['match']['team1']} vs {result['match']['team2']}")
            print(f"Format: {result['match']['format']}")
            print(f"Prediction: {result['match']['team1']} win probability: {result['prediction']['team1_win_probability']:.2%}")
            print(f"Confidence interval: {result['prediction']['confidence_interval'].get('lower_bound', 0):.2%} - {result['prediction']['confidence_interval'].get('upper_bound', 1):.2%}")
            
            print("\n--- SERIES PROBABILITIES ---")
            for outcome, prob in result['series_probabilities'].items():
                print(f"{outcome}: {prob:.2%}")
            
            if 'recommended_bets' in result and result['recommended_bets']:
                print("\n--- RECOMMENDED BETS ---")
                for bet_type, bet_info in result['recommended_bets'].items():
                    print(f"{bet_type}:")
                    print(f"  Odds: {bet_info['odds']:.2f}")
                    print(f"  Our probability: {bet_info['our_probability']:.2%}")
                    print(f"  Implied probability: {bet_info['implied_probability']:.2%}")
                    print(f"  Edge: {bet_info['edge']:.2%} ({bet_info['edge_percent']:.2f}%)")
                    print(f"  Recommended stake: ${bet_info['recommended_stake']:.2f}")
                    print(f"  Profit if win: ${bet_info['profit_if_win']:.2f}")
            elif 'all_value_bets' in result and result['all_value_bets']:
                print("\n--- POTENTIAL VALUE BETS (BELOW CONFIDENCE THRESHOLD) ---")
                for bet_type, bet_info in result['all_value_bets'].items():
                    if bet_type not in result.get('recommended_bets', {}):
                        print(f"{bet_type}:")
                        print(f"  Odds: {bet_info['odds']:.2f}")
                        print(f"  Our probability: {bet_info['our_probability']:.2%}")
                        print(f"  Implied probability: {bet_info['implied_probability']:.2%}")
                        print(f"  Edge: {bet_info['edge']:.2%}")
            else:
                print("\nNo value bets found with the current bookmaker odds and confidence threshold.")
                
            if 'team_comparison' in result:
                print("\n--- TEAM COMPARISON ---")
                for comp in result['team_comparison']:
                    print(f"{comp['metric']}: {result['match']['team1']} {comp['team1_value']:.2f} vs {result['match']['team2']} {comp['team2_value']:.2f} (Diff: {comp['difference']:.2f})")
    
    elif args.backtest:
        print("Running betting strategy backtest...")
        
        # Build historical dataset
        historical_data = build_historical_dataset(team_data_collection)
        
        if historical_data.empty:
            print("No historical data available for backtest.")
            return
        
        # Run backtest
        backtest_results = backtest_betting_strategy(
            models,
            historical_data,
            None,  # No odds data available, will use model-derived odds
            scaler=scaler,
            stable_features=stable_features,
            initial_bankroll=args.bankroll,
            kelly_fraction=args.kelly,
            min_edge=args.edge,
            use_calibration=args.calibrate
        )
        
        # Print results
        if 'error' in backtest_results:
            print(f"Backtest error: {backtest_results['error']}")
        else:
            print("\n--- BACKTEST RESULTS ---")
            print(f"Initial bankroll: ${backtest_results['summary']['initial_bankroll']:.2f}")
            print(f"Final bankroll: ${backtest_results['summary']['final_bankroll']:.2f}")
            print(f"Total profit: ${backtest_results['summary']['total_profit']:.2f}")
            print(f"ROI: {backtest_results['summary']['overall_roi']:.2%}")
            print(f"Win rate: {backtest_results['summary']['win_rate']:.2%}")
            print(f"Total bets: {backtest_results['summary']['total_bets']}")
            print(f"Total matches bet: {backtest_results['summary']['total_matches_bet']}")
            print(f"Percentage of matches bet: {backtest_results['summary']['percentage_of_matches_bet']:.2%}")
            
            if 'bet_type_metrics' in backtest_results:
                print("\n--- BET TYPE PERFORMANCE ---")
                for bet_type, metrics in backtest_results['bet_type_metrics'].items():
                    if isinstance(metrics, dict) and ('roi' in metrics):
                        print(f"{bet_type}:")
                        print(f"  ROI: {metrics['roi']:.2%}")
                        print(f"  Bets: {metrics[('result', 'count')]}")
                        print(f"  Win rate: {metrics[('result', 'mean')]:.2%}")
                        print(f"  Avg odds: {metrics[('odds', 'mean')]:.2f}")
            
            print("\nBacktest visualization saved as 'backtest_results.png'")
    
    elif args.simulation and args.team1 and args.team2:
        print(f"Running Monte Carlo simulation for: {args.team1} vs {args.team2} ({args.format})")
        
        # Find teams in data collection
        if args.team1 not in team_data_collection:
            print(f"Team '{args.team1}' not found in data collection.")
            return
            
        if args.team2 not in team_data_collection:
            print(f"Team '{args.team2}' not found in data collection.")
            return
            
        team1_stats = team_data_collection[args.team1]
        team2_stats = team_data_collection[args.team2]
        
        # Run simulation
        simulation_results = run_monte_carlo_simulation(
            team1_stats,
            team2_stats,
            format_type=args.format,
            num_simulations=args.num_sims,
            models=models,
            scaler=scaler,
            stable_features=stable_features
        )
        
        # Print results
        if 'error' in simulation_results:
            print(f"Simulation error: {simulation_results['error']}")
        else:
            print("\n--- MONTE CARLO SIMULATION RESULTS ---")
            print(f"Base win probability for {args.team1}: {simulation_results['base_probability']:.2%}")
            print(f"Number of simulations: {simulation_results['num_simulations']}")
            
            print("\nOutcome probabilities:")
            for outcome, prob in simulation_results['simulation_results'].items():
                ci = simulation_results['confidence_intervals'][outcome]
                print(f"{outcome}: {prob:.2%} (95% CI: {ci['lower']:.2%} - {ci['upper']:.2%})")
    
    elif args.analyze_future:
        print("Analyzing upcoming matches...")
        
        # Fetch upcoming matches
        upcoming_matches = fetch_upcoming_matches()
        
        if not upcoming_matches:
            print("No upcoming matches found.")
            return
            
        # Prepare matches for prediction
        prepared_matches = prepare_new_matches_for_prediction(
            upcoming_matches,
            team_data_collection,
            models,
            scaler,
            stable_features
        )
        
        if not prepared_matches:
            print("Failed to prepare any matches for prediction.")
            return
            
        print(f"\nAnalyzed {len(prepared_matches)} upcoming matches:")
        
        # List matches by date
        sorted_matches = sorted(prepared_matches, key=lambda x: x.get('date', ''))
        
        for match in sorted_matches:
            team1 = match['team1']
            team2 = match['team2']
            format_type = match['format']
            
            team1_prob = match['prediction']['team1_win_probability']
            suggested_format = get_best_format_for_match(
                team_data_collection[team1], team_data_collection[team2]
            )
            
            print(f"\n{match.get('date', 'TBD')} - {team1} vs {team2} ({format_type})")
            print(f"Prediction: {team1} {team1_prob:.2%} vs {team2} {1-team1_prob:.2%}")
            print(f"Suggested betting format: {suggested_format}")
            
            # Show key series probabilities
            series_probs = match['series_probabilities']
            
            if format_type == 'bo3':
                print(f"Maps over 2.5: {series_probs.get('maps_over_2.5', 0):.2%}")
                print(f"{team1} win 2-0: {series_probs.get('team1_win_2_0', 0):.2%}")
                print(f"{team2} win 2-0: {series_probs.get('team2_win_2_0', 0):.2%}")
            elif format_type == 'bo5':
                print(f"Maps over 3.5: {series_probs.get('maps_over_3.5', 0):.2%}")
    
    else:
        print("No valid command specified. Use --predict, --backtest, --simulation, or --analyze-future.")
        print("For help, use --help")

#-------------------------------------------------------------------------
# ENTRY POINT
#-------------------------------------------------------------------------

if __name__ == "__main__":
    betting_cli()

#-------------------------------------------------------------------------
# MAIN FUNCTION AND CLI INTERFACE
#-------------------------------------------------------------------------

def main():
    """Main function to handle command line arguments and run the program."""
    parser = argparse.ArgumentParser(description="Valorant Match Predictor")
    
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
    
    elif args.predict and args.team1 and args.team2:
        # Check if ensemble models exist
        ensemble_exists = any(os.path.exists(f'valorant_model_fold_{i+1}.h5') for i in range(5))
        
        if ensemble_exists:
            print(f"Predicting match between {args.team1} and {args.team2} using ensemble model...")
            prediction = predict_with_ensemble(args.team1, args.team2, analyze_features=args.analyze_features)
        else:
            print(f"Predicting match between {args.team1} and {args.team2} with standard model...")
            prediction = predict_match(args.team1, args.team2, include_maps=include_maps)
        
        if prediction:
            display_prediction_results(prediction)
            visualize_prediction(prediction)
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
            confidence_threshold=args.confidence
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

    # New betting analysis functionality with manual odds input
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
            
            # Add circuit breaker to prevent infinite loops
            # Dictionary to track function calls
            call_count = {}
            
            # Run the analysis with debugging
            print("Starting analysis...")
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
            else:
                print("Analysis failed to produce results.")
        except ValueError as e:
            print(f"Error: Invalid odds input. Please enter valid decimal odds (e.g., 1.85).")
            print(f"Exception: {e}")
            return
    
    
    else:
        print("Please specify an action: --train, --predict, --analyze, --backtest, or --betting")
        print("For predictions and betting analysis, specify --team1 and --team2")
        print("For betting analysis, use --betting --manual-odds, and consider using --bankroll, --min-ev, and --kelly")
        print("For backtesting, specify --cutoff-date YYYY/MM/DD")


if __name__ == "__main__":
    main()         

