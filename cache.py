#!/usr/bin/env python3
print("Starting Deep Learning Valorant Match Predictor...")

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

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
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

def get_team_id(team_name, region=None):
    """
    Search for a team ID by name, optionally filtering by region.
    
    Args:
        team_name (str): The name of the team to search for
        region (str, optional): Region code like 'na', 'eu', 'kr', etc.
    
    Returns:
        str: Team ID if found, None otherwise
    """
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

def fetch_team_details(team_id):
    """Fetch detailed information about a team, including the team tag."""
    if not team_id:
        return None, None
    
    print(f"Fetching details for team ID: {team_id}")
    response = requests.get(f"{API_URL}/teams/{team_id}")
    
    if response.status_code != 200:
        print(f"Error fetching team {team_id}: {response.status_code}")
        return None, None
    
    team_data = response.json()
    
    # Be nice to the API
     
    
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
        print(f"Team tag not found in data structure: {json.dumps(team_data.get('data', {}), indent=2)[:200]}...")
    
    return team_data, team_tag

def fetch_team_player_stats(team_id):
    """
    Fetch detailed player statistics for a team directly from the team endpoint.
    
    Args:
        team_id (str): The ID of the team
    
    Returns:
        list: List of player statistics dictionaries
    """
    if not team_id:
        print(f"Invalid team ID: {team_id}")
        return []
    
    # Get team details with roster information
    print(f"Fetching details for team ID: {team_id}")
    response = requests.get(f"{API_URL}/teams/{team_id}")
    
    if response.status_code != 200:
        print(f"Error fetching team {team_id}: {response.status_code}")
        return []
    
    team_data = response.json()
    
    # Check if we got valid data and it contains player information
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

def extract_map_statistics(team_stats_data):
    """
    Extract detailed map statistics from team stats API response.
    
    Args:
        team_stats_data (dict): Data from team-stats API endpoint
        
    Returns:
        dict: Processed map statistics with various metrics
    """
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

def fetch_team_map_statistics(team_id):
    """
    Fetch detailed map statistics for a team using the team-stats API endpoint.
    
    Args:
        team_id (str): The ID of the team
        
    Returns:
        dict: Processed map statistics with various metrics
    """
    if not team_id:
        print("Invalid team ID")
        return {}
    
    print(f"Fetching map statistics for team ID: {team_id}")
    response = requests.get(f"{API_URL}/team-stats/{team_id}")
    
    if response.status_code != 200:
        print(f"Error fetching team stats for {team_id}: {response.status_code}")
        return {}
    
    team_stats_data = response.json()
    

    
    # Process the data
    map_statistics = extract_map_statistics(team_stats_data)
    
    print(f"Processed statistics for {len(map_statistics)} maps")
    return map_statistics

def calculate_team_player_stats(player_stats_list):
    """
    Calculate team-level statistics from individual player stats.
    
    Args:
        player_stats_list (list): List of player statistics dictionaries
    
    Returns:
        dict: Aggregate team statistics based on player data
    """
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
    
    print(f"Processing {len(player_stats_list)} player stats")
    print(f"Sample player data: {json.dumps(player_stats_list[0], indent=2)}")
    
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

def fetch_team_match_history(team_id):
    """Fetch match history for a specific team."""
    if not team_id:
        return None
    
    print(f"Fetching match history for team ID: {team_id}")
    response = requests.get(f"{API_URL}/match-history/{team_id}")
    
    if response.status_code != 200:
        print(f"Error fetching match history for team {team_id}: {response.status_code}")
        return None
    
    match_history = response.json()
    
    # Be nice to the API
     
    
    return match_history

def fetch_match_details(match_id):
    """Fetch detailed information about a specific match."""
    if not match_id:
        return None
    
    print(f"Fetching details for match ID: {match_id}")
    response = requests.get(f"{API_URL}/match-details/{match_id}")
    
    if response.status_code != 200:
        print(f"Error fetching match details {match_id}: {response.status_code}")
        return None
    
    match_details = response.json()
    
    # Be nice to the API
     
    
    return match_details

def fetch_events():
    """Fetch all events from the API."""
    print("Fetching events...")
    response = requests.get(f"{API_URL}/events?limit=100")
    
    if response.status_code != 200:
        print(f"Error fetching events: {response.status_code}")
        return []
    
    events_data = response.json()
    
    # Be nice to the API
     
    
    return events_data.get('data', [])

def fetch_upcoming_matches():
    """Fetch upcoming matches."""
    print("Fetching upcoming matches...")
    response = requests.get(f"{API_URL}/matches")
    
    if response.status_code != 200:
        print(f"Error fetching upcoming matches: {response.status_code}")
        return []
    
    matches_data = response.json()
    
    # Be nice to the API
     
    
    return matches_data.get('data', [])

def parse_match_data(match_history, team_name):
    """Parse match history data for a team."""
    if not match_history or 'data' not in match_history:
        return []
    
    matches = []
    
    # Add debug output
    print(f"Parsing {len(match_history['data'])} matches for {team_name}")
    
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
                
                # Debug output
                print(f"Match {match['id']}: {team1.get('name', '')} ({team1_score}) vs {team2.get('name', '')} ({team2_score})")
                
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
                match_info['team_won'] = team_won  # Use calculated value
                match_info['team_country'] = our_team.get('country', '')
                
                # Add opponent's info
                match_info['opponent_name'] = opponent_team.get('name', '')
                match_info['opponent_score'] = int(opponent_team.get('score', 0))
                match_info['opponent_won'] = not team_won  # Opponent's result is opposite of our team
                match_info['opponent_country'] = opponent_team.get('country', '')
             # *** ADD THIS LINE ***
                match_info['result'] = 'win' if team_won else 'loss'               


                print(f"  Determined: {match_info['team_name']} won? {match_info['team_won']}")
                

                # Fetch match details for deeper statistics
                match_details = fetch_match_details(match_info['match_id'])
                if match_details:
                    # Add match details to the match_info
                    match_info['details'] = match_details
                
                matches.append(match_info)
            
        except Exception as e:
            print(f"Error parsing match: {e}")
            continue
    
    # Summarize wins/losses
    wins = sum(1 for match in matches if match['team_won'])
    print(f"Processed {len(matches)} matches for {team_name}: {wins} wins, {len(matches) - wins} losses")
    
    return matches

def calculate_team_stats(matches, player_stats=None):
    """Calculate comprehensive statistics for a team from its matches and player data."""
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
    
    # Debug output to check if matches contain expected data
    print(f"Total matches: {total_matches}, Wins: {wins}, Losses: {losses}")
    if total_matches > 0:
        # Print the first match to debug structure
        print(f"Sample match data: {matches[0]}")
    
    # Opponent-specific stats
    opponent_stats = {}
    for match in matches:
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
    for match in matches:
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
    sorted_matches = sorted(matches, key=lambda x: x.get('date', ''))
    recent_matches = sorted_matches[-5:] if len(sorted_matches) >= 5 else sorted_matches
    recent_form = sum(1 for match in recent_matches if match['team_won']) / len(recent_matches) if recent_matches else 0
    
    # Extract match details stats if available
    advanced_stats = extract_match_details_stats(matches)
    
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

    return team_stats

calculate_team_stats_original = calculate_team_stats

def fetch_match_economy_details(match_id):
    """Fetch economic details for a specific match."""
    if not match_id:
        return None
    
    print(f"Fetching economy details for match ID: {match_id}")
    response = requests.get(f"{API_URL}/match-details/{match_id}?tab=economy")
    
    if response.status_code != 200:
        print(f"Error fetching economy details for match {match_id}: {response.status_code}")
        return None
    
    economy_details = response.json()
    
    # Be nice to the API
     
    return economy_details

def calculate_team_stats_with_economy(team_matches, player_stats=None):
    """Calculate comprehensive statistics for a team from its matches, including economy data."""
    # First use the original function to get base stats
    base_stats = calculate_team_stats_original(team_matches, player_stats)
    
    if not team_matches:
        return base_stats
    
    # Initialize economy stats
    economy_stats = {
        'pistol_win_rate': 0,
        'eco_win_rate': 0,
        'semi_eco_win_rate': 0,
        'semi_buy_win_rate': 0,
        'full_buy_win_rate': 0,
        'economy_efficiency': 0,
        'total_economy_matches': 0
    }
    
    # Get team tag or name from the matches if available
    team_tag = None
    team_name = None
    team_id = None
    
    if team_matches:
        team_tag = team_matches[0].get('team_tag')
        team_name = team_matches[0].get('team_name')
        team_id = team_matches[0].get('team_id')
    
    print(f"Processing economy data for team: {team_name} (Tag: {team_tag}, ID: {team_id})")
    
    # Track aggregated economy stats across matches
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
    
    # Process each match to extract economy data
    for match in team_matches:
        match_id = match.get('match_id')
        if not match_id:
            continue
            
        # Get team tag and name for better matching
        match_team_tag = match.get('team_tag', team_tag)
        match_team_name = match.get('team_name', team_name)
        
        # Debug output
        print(f"Processing match ID {match_id} economy data for team: {match_team_name} (Tag: {match_team_tag})")
            
        # Get economy data with additional debug output
        print(f"Fetching economy details for match ID: {match_id}")
        economy_data = fetch_match_economy_details(match_id)
        
        if not economy_data:
            print(f"No economy data returned for match ID: {match_id}")
            continue
            
        if 'data' not in economy_data:
            print(f"No 'data' field in economy response for match ID: {match_id}")
            continue
            
        if 'teams' not in economy_data['data']:
            print(f"No 'teams' field in economy data for match ID: {match_id}")
            continue
            
        # Try to use team_tag first if available, otherwise fall back to team_name
        team_identifier = match_team_tag
        fallback_name = match_team_name
        
        if not team_identifier and not fallback_name:
            print(f"Warning: No team identifier (tag or name) available for match ID: {match_id}")
            continue
            
        # Debug: show teams in economy data
        teams_in_data = [team.get('name', 'Unknown') for team in economy_data['data']['teams']]
        print(f"Teams in economy data: {teams_in_data}")
            
        # Extract team-specific economy data using the identifier with more debug info
        print(f"Looking for team with identifier: {team_identifier} (Fallback: {fallback_name})")
        our_team_metrics = extract_economy_metrics(economy_data, team_identifier, fallback_name)
        
        if not our_team_metrics:
            print(f"Warning: No economy metrics extracted for team: {team_identifier or fallback_name}")
            continue
        
        print(f"Successfully extracted economy metrics for match ID: {match_id}")
            
        # Aggregate stats
        total_pistol_won += our_team_metrics.get('pistol_rounds_won', 0)
        total_pistol_rounds += our_team_metrics.get('total_pistol_rounds', 2)  # Usually 2 pistol rounds per map
        
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
    
    # Debug summary
    print(f"Economy data summary for {team_name}:")
    print(f"- Economy matches processed: {economy_matches_count}")
    print(f"- Total pistol rounds: {total_pistol_rounds}, Won: {total_pistol_won}")
    print(f"- Total eco rounds: {total_eco_rounds}, Won: {total_eco_won}")
    print(f"- Total full buy rounds: {total_full_buy_rounds}, Won: {total_full_buy_won}")
    
    # Calculate aggregate economy stats if we have data
    if economy_matches_count > 0:
        economy_stats['total_economy_matches'] = economy_matches_count
        economy_stats['pistol_win_rate'] = total_pistol_won / total_pistol_rounds if total_pistol_rounds > 0 else 0
        economy_stats['eco_win_rate'] = total_eco_won / total_eco_rounds if total_eco_rounds > 0 else 0
        economy_stats['semi_eco_win_rate'] = total_semi_eco_won / total_semi_eco_rounds if total_semi_eco_rounds > 0 else 0
        economy_stats['semi_buy_win_rate'] = total_semi_buy_won / total_semi_buy_rounds if total_semi_buy_rounds > 0 else 0
        economy_stats['full_buy_win_rate'] = total_full_buy_won / total_full_buy_rounds if total_full_buy_rounds > 0 else 0
        economy_stats['economy_efficiency'] = total_efficiency_sum / economy_matches_count
        
        # Add useful compound metrics
        economy_stats['low_economy_win_rate'] = (total_eco_won + total_semi_eco_won) / (total_eco_rounds + total_semi_eco_rounds) if (total_eco_rounds + total_semi_eco_rounds) > 0 else 0
        economy_stats['high_economy_win_rate'] = (total_semi_buy_won + total_full_buy_won) / (total_semi_buy_rounds + total_full_buy_rounds) if (total_semi_buy_rounds + total_full_buy_rounds) > 0 else 0
        economy_stats['pistol_round_sample_size'] = total_pistol_rounds
        economy_stats['pistol_confidence'] = 1.0 - (1.0 / (1.0 + 0.1 * total_pistol_rounds)) 
        
        print(f"Successfully calculated economy stats for {team_name}")
        print(f"- Pistol win rate: {economy_stats['pistol_win_rate']:.2f}")
        print(f"- Eco win rate: {economy_stats['eco_win_rate']:.2f}")
        print(f"- Full buy win rate: {economy_stats['full_buy_win_rate']:.2f}")
    else:
        print(f"No economy data calculated for {team_name} - no matches with valid economy data")
    
    # Add economy stats to base stats
    base_stats.update(economy_stats)
    
    # Also check for and add map statistics
    if team_id:
        map_stats = fetch_team_map_statistics(team_id)
        if map_stats:
            base_stats['map_statistics'] = map_stats
            print(f"Added map statistics for {team_name} ({len(map_stats)} maps)")
    
    return base_stats

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
            
            # Determine tournament tier (simplified - you could create a more nuanced system)
            event_name = tournament_key.split(':')[0].lower()
            if any(major in event_name for major in ['masters', 'champions', 'last chance']):
                stats['tier'] = 3  # Top tier
            elif any(medium in event_name for medium in ['challenger', 'regional', 'national']):
                stats['tier'] = 2  # Mid tier
            else:
                stats['tier'] = 1  # Lower tier
    
    return tournament_performance

def analyze_feature_importance(model, feature_names):
    """
    Analyze the importance of different features, particularly focusing on economy and player features.
    
    Args:
        model: Trained TensorFlow model
        feature_names: List of feature names in the order they were used for training
    
    Returns:
        dict: Feature importance scores
    """
    print("\n============================================")
    print("ANALYZING FEATURE IMPORTANCE")
    print("============================================")
    
    # This is a simple approach for TensorFlow models - extract weights from the first Dense layer
    try:
        # Get weights from the first layer
        weights = model.layers[1].get_weights()[0]
        
        # Calculate absolute weights as a simple measure of feature importance
        importance = np.abs(weights).mean(axis=1)
        
        # Create a dictionary mapping feature names to importance scores
        feature_importance = {feature_names[i]: float(importance[i]) for i in range(len(feature_names))}
        
        # Sort by importance
        sorted_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
        
        # Group by feature categories with enhanced categorization
        economy_features = {}
        player_features = {}
        map_features = {}
        team_features = {}
        opponent_features = {}
        
        for feature, score in sorted_importance.items():
            if any(term in feature.lower() for term in ['eco', 'pistol', 'buy', 'economy']):
                economy_features[feature] = score
            elif any(term in feature.lower() for term in ['rating', 'acs', 'kd', 'adr', 'headshot', 
                                                       'clutch', 'aces', 'first_blood', 'entry_kills']):
                player_features[feature] = score
            elif 'map_' in feature.lower():
                map_features[feature] = score
            elif 'opponent' in feature.lower() or 'h2h' in feature.lower():
                opponent_features[feature] = score
            else:
                team_features[feature] = score
        
        # Calculate average importance by category
        avg_economy = sum(economy_features.values()) / len(economy_features) if economy_features else 0
        avg_player = sum(player_features.values()) / len(player_features) if player_features else 0
        avg_map = sum(map_features.values()) / len(map_features) if map_features else 0
        avg_team = sum(team_features.values()) / len(team_features) if team_features else 0
        avg_opponent = sum(opponent_features.values()) / len(opponent_features) if opponent_features else 0
        
        # Create report
        importance_report = {
            'top_features': list(sorted_importance.keys())[:15],
            'economy_features': economy_features,
            'player_features': player_features,
            'map_features': map_features,
            'team_features': team_features,
            'opponent_features': opponent_features,
            'category_importance': {
                'economy': float(avg_economy),
                'player': float(avg_player),
                'map': float(avg_map),
                'team': float(avg_team),
                'opponent': float(avg_opponent)
            }
        }
        
        print("\nFeature Importance Analysis:")
        print(f"Economy Features: {len(economy_features)} features, avg importance: {avg_economy:.4f}")
        print(f"Player Features: {len(player_features)} features, avg importance: {avg_player:.4f}")
        print(f"Map Features: {len(map_features)} features, avg importance: {avg_map:.4f}")
        print(f"Team Features: {len(team_features)} features, avg importance: {avg_team:.4f}")
        print(f"Opponent Features: {len(opponent_features)} features, avg importance: {avg_opponent:.4f}")
        
        # Print top 5 features by category
        if economy_features:
            print("\nTop 5 Economy Features:")
            for i, (feature, score) in enumerate(sorted(economy_features.items(), key=lambda x: x[1], reverse=True)[:5]):
                print(f"  {i+1}. {feature}: {score:.4f}")
                
        if player_features:
            print("\nTop 5 Player Features:")
            for i, (feature, score) in enumerate(sorted(player_features.items(), key=lambda x: x[1], reverse=True)[:5]):
                print(f"  {i+1}. {feature}: {score:.4f}")
        
        return importance_report
    
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")
        return None

def get_team_ranking(team_id):
    """
    Get a team's current ranking directly from the team details endpoint.
    
    Args:
        team_id (str): The ID of the team
    
    Returns:
        tuple: (ranking, rating) if found, (None, None) otherwise
    """
    if not team_id:
        return None, None
    
    print(f"Fetching team details for ID: {team_id}")
    response = requests.get(f"{API_URL}/teams/{team_id}")
    
    if response.status_code != 200:
        print(f"Error fetching team details: {response.status_code}")
        return None, None
    
    team_data = response.json()
    
    if 'data' not in team_data:
        print(f"No data field found in response for team ID: {team_id}")
        return None, None
    
    team_info = team_data['data']
    
    # Extract ranking information
    ranking = None
    if 'countryRanking' in team_info and 'rank' in team_info['countryRanking']:
        try:
            ranking = int(team_info['countryRanking']['rank'])
            print(f"Found ranking: {ranking} ({team_info['countryRanking'].get('country', 'Unknown')})")
        except (ValueError, TypeError):
            print(f"Invalid ranking format: {team_info['countryRanking'].get('rank')}")
    
    # Extract rating information
    rating = None
    if 'rating' in team_info:
        # The rating format is complex, typically like "1432 1W 5L 1580 6W 1L"
        # Extract the first number which is the overall rating
        try:
            rating_parts = team_info['rating'].split()
            if rating_parts and rating_parts[0].isdigit():
                rating = float(rating_parts[0])
                print(f"Found rating: {rating}")
        except (ValueError, IndexError, AttributeError):
            print(f"Invalid rating format: {team_info.get('rating')}")
    
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
    team_name = None
    
    if team_matches and len(team_matches) > 0:
        team_name = team_matches[0].get('team_name')
    
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

def prepare_data_for_model(team1_stats, team2_stats):
    """Prepare data for the ML model by creating feature vectors."""
    if not team1_stats or not team2_stats:
        return None
    
    # Create feature dictionary
    features = {}
    
    # Debug to understand the structure
    print("Team1 stats keys:", team1_stats.keys())
    
    # Basic win rates and match counts - handle lists carefully
    
    # For matches, the error suggests this might be a list - let's debug and handle it properly
    if 'matches' in team1_stats:
        # Check if matches is a list (containing actual match data) or a number
        if isinstance(team1_stats['matches'], list):
            # If it's a list of matches, use the length
            features['team1_matches'] = len(team1_stats['matches'])
        else:
            # Otherwise try to convert to int
            features['team1_matches'] = int(team1_stats.get('matches', 0))
    else:
        features['team1_matches'] = 0
        
    if 'matches' in team2_stats:
        if isinstance(team2_stats['matches'], list):
            features['team2_matches'] = len(team2_stats['matches'])
        else:
            features['team2_matches'] = int(team2_stats.get('matches', 0))
    else:
        features['team2_matches'] = 0
    
    # For other stats, use float conversion with safety checks
    features['team1_win_rate'] = float(team1_stats.get('win_rate', 0.5))
    features['team2_win_rate'] = float(team2_stats.get('win_rate', 0.5))
    
    # Score differentials
    features['team1_score_differential'] = float(team1_stats.get('score_differential', 0))
    features['team2_score_differential'] = float(team2_stats.get('score_differential', 0))
    
    # Recent form
    features['team1_recent_form'] = float(team1_stats.get('recent_form', 0.5))
    features['team2_recent_form'] = float(team2_stats.get('recent_form', 0.5))
    
    # Team 1 player stats
    features['team1_avg_player_rating'] = float(team1_stats.get('avg_player_rating', 0))
    features['team1_avg_player_acs'] = float(team1_stats.get('avg_player_acs', 0))
    features['team1_avg_player_kd'] = float(team1_stats.get('avg_player_kd', 0))
    features['team1_avg_player_kast'] = float(team1_stats.get('avg_player_kast', 0))
    features['team1_avg_player_adr'] = float(team1_stats.get('avg_player_adr', 0))
    features['team1_avg_player_headshot'] = float(team1_stats.get('avg_player_headshot', 0))
    features['team1_star_player_rating'] = float(team1_stats.get('star_player_rating', 0))
    features['team1_team_consistency'] = float(team1_stats.get('team_consistency', 0))
    features['team1_fk_fd_ratio'] = float(team1_stats.get('fk_fd_ratio', 0))
    
    # Team 2 player stats
    features['team2_avg_player_rating'] = float(team2_stats.get('avg_player_rating', 0))
    features['team2_avg_player_acs'] = float(team2_stats.get('avg_player_acs', 0))
    features['team2_avg_player_kd'] = float(team2_stats.get('avg_player_kd', 0))
    features['team2_avg_player_kast'] = float(team2_stats.get('avg_player_kast', 0))
    features['team2_avg_player_adr'] = float(team2_stats.get('avg_player_adr', 0))
    features['team2_avg_player_headshot'] = float(team2_stats.get('avg_player_headshot', 0))
    features['team2_star_player_rating'] = float(team2_stats.get('star_player_rating', 0))
    features['team2_team_consistency'] = float(team2_stats.get('team_consistency', 0))
    features['team2_fk_fd_ratio'] = float(team2_stats.get('fk_fd_ratio', 0))
    
    # Calculate player stat differentials
    features['rating_differential'] = features['team1_avg_player_rating'] - features['team2_avg_player_rating']
    features['acs_differential'] = features['team1_avg_player_acs'] - features['team2_avg_player_acs']
    features['kd_differential'] = features['team1_avg_player_kd'] - features['team2_avg_player_kd']
    features['kast_differential'] = features['team1_avg_player_kast'] - features['team2_avg_player_kast']
    features['adr_differential'] = features['team1_avg_player_adr'] - features['team2_avg_player_adr']
    features['headshot_differential'] = features['team1_avg_player_headshot'] - features['team2_avg_player_headshot']
    features['star_player_differential'] = features['team1_star_player_rating'] - features['team2_star_player_rating']
    features['consistency_differential'] = features['team1_team_consistency'] - features['team2_team_consistency']
    features['fk_fd_differential'] = features['team1_fk_fd_ratio'] - features['team2_fk_fd_ratio']

    # Head-to-head stats if available
    team2_name = None
    if 'opponent_stats' in team1_stats:
        for opponent, stats in team1_stats['opponent_stats'].items():
            if opponent.lower() in [k.lower() for k in team2_stats.keys() if isinstance(k, str)]:
                team2_name = opponent
                break
    
    if team2_name and team2_name in team1_stats['opponent_stats']:
        features['h2h_win_rate'] = float(team1_stats['opponent_stats'][team2_name].get('win_rate', 0.5))
        features['h2h_score_diff'] = float(team1_stats['opponent_stats'][team2_name].get('score_differential', 0))
        features['h2h_matches'] = int(team1_stats['opponent_stats'][team2_name].get('matches', 0))
    else:
        features['h2h_win_rate'] = 0.5  # No head-to-head data, use neutral value
        features['h2h_score_diff'] = 0.0
        features['h2h_matches'] = 0
    
    # Map-specific stats for common maps
    common_maps = set()
    if 'map_stats' in team1_stats and 'map_stats' in team2_stats:
        common_maps = set(team1_stats['map_stats'].keys()) & set(team2_stats['map_stats'].keys())
    
    features['common_maps'] = len(common_maps)
    
    for map_name in common_maps:
        features[f'team1_winrate_{map_name}'] = float(team1_stats['map_stats'][map_name].get('win_rate', 0.5))
        features[f'team2_winrate_{map_name}'] = float(team2_stats['map_stats'][map_name].get('win_rate', 0.5))
    
    # Advanced stats if available - with safety checks
    if 'advanced_stats' in team1_stats:
        for key, value in team1_stats['advanced_stats'].items():
            if isinstance(value, (int, float)) and key != 'agent_usage':  # Skip agent_usage as it's a dict
                features[f'team1_{key}'] = float(value)
    
    if 'advanced_stats' in team2_stats:
        for key, value in team2_stats['advanced_stats'].items():
            if isinstance(value, (int, float)) and key != 'agent_usage':  # Skip agent_usage as it's a dict
                features[f'team2_{key}'] = float(value)
    
    # Performance trends - careful with nested structures
    if 'performance_trends' in team1_stats:
        for trend_category, trend_data in team1_stats['performance_trends'].items():
            if isinstance(trend_data, dict): # Check if it's a dictionary
                if trend_category == 'recent_matches':
                    for metric, value in trend_data.items():
                        if isinstance(value, (int, float)):
                            features[f'team1_{metric}'] = float(value)
    
    if 'performance_trends' in team2_stats:
        for trend_category, trend_data in team2_stats['performance_trends'].items():
            if isinstance(trend_data, dict): # Check if it's a dictionary
                if trend_category == 'recent_matches':
                    for metric, value in trend_data.items():
                        if isinstance(value, (int, float)):
                            features[f'team2_{metric}'] = float(value)
    
    # Opponent quality metrics
    if 'opponent_quality' in team1_stats:
        for key, value in team1_stats['opponent_quality'].items():
            if isinstance(value, (int, float)):
                features[f'team1_{key}'] = float(value)
    
    if 'opponent_quality' in team2_stats:
        for key, value in team2_stats['opponent_quality'].items():
            if isinstance(value, (int, float)):
                features[f'team2_{key}'] = float(value)
    
    # Debug: Add a check for non-numeric values
    for key, value in list(features.items()):
        if not isinstance(value, (int, float)):
            print(f"WARNING: Non-numeric feature detected: {key} = {value} (type: {type(value)})")
            # Remove or convert problematic features
            del features[key]
    
    return features

def prepare_fully_symmetrical_data(team1_stats, team2_stats):
    """Create completely symmetrical features for all relevant metrics."""
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
    # 2. PERFORMANCE TRENDS
    #----------------------------------------
    if ('performance_trends' in team1_stats and 'performance_trends' in team2_stats and
        team1_stats.get('performance_trends') and team2_stats.get('performance_trends')):
        
        # Recent matches form (based on last 5, 10, 20 matches)
        for window in [5, 10, 20]:
            t1_key = f'last_{window}_win_rate'
            if (f'recent_matches' in team1_stats['performance_trends'] and 
                f'recent_matches' in team2_stats['performance_trends'] and
                t1_key in team1_stats['performance_trends']['recent_matches'] and
                t1_key in team2_stats['performance_trends']['recent_matches']):
                
                t1_rate = team1_stats['performance_trends']['recent_matches'][t1_key]
                t2_rate = team2_stats['performance_trends']['recent_matches'][t1_key]
                
                features[f'recent_{window}_diff'] = t1_rate - t2_rate
                features[f'better_recent_{window}_team1'] = 1 if t1_rate > t2_rate else 0
                features[f'avg_recent_{window}'] = (t1_rate + t2_rate) / 2
        
        # Form trajectory (momentum)
        if ('form_trajectory' in team1_stats['performance_trends'] and
            'form_trajectory' in team2_stats['performance_trends']):
            
            for key in ['5_vs_10', '10_vs_20']:
                if (key in team1_stats['performance_trends']['form_trajectory'] and
                    key in team2_stats['performance_trends']['form_trajectory']):
                    
                    t1_traj = team1_stats['performance_trends']['form_trajectory'][key]
                    t2_traj = team2_stats['performance_trends']['form_trajectory'][key]
                    
                    features[f'momentum_{key}_diff'] = t1_traj - t2_traj
                    features[f'better_momentum_{key}_team1'] = 1 if t1_traj > t2_traj else 0
                    features[f'avg_momentum_{key}'] = (t1_traj + t2_traj) / 2
        
        # Recency-weighted win rate
        if ('recency_weighted_win_rate' in team1_stats['performance_trends'] and
            'recency_weighted_win_rate' in team2_stats['performance_trends']):
            
            t1_weighted = team1_stats['performance_trends']['recency_weighted_win_rate']
            t2_weighted = team2_stats['performance_trends']['recency_weighted_win_rate']
            
            features['weighted_win_rate_diff'] = t1_weighted - t2_weighted
            features['better_weighted_win_rate_team1'] = 1 if t1_weighted > t2_weighted else 0
            features['avg_weighted_win_rate'] = (t1_weighted + t2_weighted) / 2
    
    #----------------------------------------
    # 3. PLAYER STATS
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
        
        # Player count
        if 'player_stats' in team1_stats and 'player_stats' in team2_stats:
            t1_players = team1_stats['player_stats'].get('player_count', 0)
            t2_players = team2_stats['player_stats'].get('player_count', 0)
            features['player_count_diff'] = t1_players - t2_players
            features['player_count_ratio'] = t1_players / max(t2_players, 1)
            features['avg_player_count'] = (t1_players + t2_players) / 2
    
    #----------------------------------------
    # 4. ECONOMY STATS
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
        
        # Semi-eco win rate
        features['semi_eco_win_rate_diff'] = team1_stats.get('semi_eco_win_rate', 0) - team2_stats.get('semi_eco_win_rate', 0)
        features['better_semi_eco_team1'] = 1 if team1_stats.get('semi_eco_win_rate', 0) > team2_stats.get('semi_eco_win_rate', 0) else 0
        features['avg_semi_eco_win_rate'] = (team1_stats.get('semi_eco_win_rate', 0) + team2_stats.get('semi_eco_win_rate', 0)) / 2
        
        # Semi-buy win rate
        features['semi_buy_win_rate_diff'] = team1_stats.get('semi_buy_win_rate', 0) - team2_stats.get('semi_buy_win_rate', 0)
        features['better_semi_buy_team1'] = 1 if team1_stats.get('semi_buy_win_rate', 0) > team2_stats.get('semi_buy_win_rate', 0) else 0
        features['avg_semi_buy_win_rate'] = (team1_stats.get('semi_buy_win_rate', 0) + team2_stats.get('semi_buy_win_rate', 0)) / 2
        
        # Full-buy win rate
        features['full_buy_win_rate_diff'] = team1_stats.get('full_buy_win_rate', 0) - team2_stats.get('full_buy_win_rate', 0)
        features['better_full_buy_team1'] = 1 if team1_stats.get('full_buy_win_rate', 0) > team2_stats.get('full_buy_win_rate', 0) else 0
        features['avg_full_buy_win_rate'] = (team1_stats.get('full_buy_win_rate', 0) + team2_stats.get('full_buy_win_rate', 0)) / 2
        
        # Low economy win rate
        features['low_economy_win_rate_diff'] = team1_stats.get('low_economy_win_rate', 0) - team2_stats.get('low_economy_win_rate', 0)
        features['better_low_economy_team1'] = 1 if team1_stats.get('low_economy_win_rate', 0) > team2_stats.get('low_economy_win_rate', 0) else 0
        features['avg_low_economy_win_rate'] = (team1_stats.get('low_economy_win_rate', 0) + team2_stats.get('low_economy_win_rate', 0)) / 2
        
        # High economy win rate
        features['high_economy_win_rate_diff'] = team1_stats.get('high_economy_win_rate', 0) - team2_stats.get('high_economy_win_rate', 0)
        features['better_high_economy_team1'] = 1 if team1_stats.get('high_economy_win_rate', 0) > team2_stats.get('high_economy_win_rate', 0) else 0
        features['avg_high_economy_win_rate'] = (team1_stats.get('high_economy_win_rate', 0) + team2_stats.get('high_economy_win_rate', 0)) / 2
        
        # Economy efficiency
        features['economy_efficiency_diff'] = team1_stats.get('economy_efficiency', 0) - team2_stats.get('economy_efficiency', 0)
        features['better_economy_efficiency_team1'] = 1 if team1_stats.get('economy_efficiency', 0) > team2_stats.get('economy_efficiency', 0) else 0
        features['avg_economy_efficiency'] = (team1_stats.get('economy_efficiency', 0) + team2_stats.get('economy_efficiency', 0)) / 2
        
        # Pistol confidence (higher sample size is better)
        features['pistol_confidence_diff'] = team1_stats.get('pistol_confidence', 0) - team2_stats.get('pistol_confidence', 0)
        features['better_pistol_confidence_team1'] = 1 if team1_stats.get('pistol_confidence', 0) > team2_stats.get('pistol_confidence', 0) else 0
        features['avg_pistol_confidence'] = (team1_stats.get('pistol_confidence', 0) + team2_stats.get('pistol_confidence', 0)) / 2
        
        # Pistol sample size
        features['pistol_sample_diff'] = team1_stats.get('pistol_round_sample_size', 0) - team2_stats.get('pistol_round_sample_size', 0)
        features['better_pistol_sample_team1'] = 1 if team1_stats.get('pistol_round_sample_size', 0) > team2_stats.get('pistol_round_sample_size', 0) else 0
        features['avg_pistol_sample'] = (team1_stats.get('pistol_round_sample_size', 0) + team2_stats.get('pistol_round_sample_size', 0)) / 2
    
    #----------------------------------------
    # 5. ADVANCED STATS (if available)
    #----------------------------------------
    if ('advanced_stats' in team1_stats and 'advanced_stats' in team2_stats and
        team1_stats.get('advanced_stats') and team2_stats.get('advanced_stats')):
        
        # First bloods
        t1_fb = team1_stats['advanced_stats'].get('avg_first_bloods', 0)
        t2_fb = team2_stats['advanced_stats'].get('avg_first_bloods', 0)
        features['first_bloods_diff'] = t1_fb - t2_fb
        features['better_first_bloods_team1'] = 1 if t1_fb > t2_fb else 0
        features['avg_first_bloods'] = (t1_fb + t2_fb) / 2
        
        # Clutches
        t1_clutches = team1_stats['advanced_stats'].get('avg_clutches', 0)
        t2_clutches = team2_stats['advanced_stats'].get('avg_clutches', 0)
        features['clutches_diff'] = t1_clutches - t2_clutches
        features['better_clutches_team1'] = 1 if t1_clutches > t2_clutches else 0
        features['avg_clutches'] = (t1_clutches + t2_clutches) / 2
        
        # Aces
        t1_aces = team1_stats['advanced_stats'].get('avg_aces', 0)
        t2_aces = team2_stats['advanced_stats'].get('avg_aces', 0)
        features['aces_diff'] = t1_aces - t2_aces
        features['better_aces_team1'] = 1 if t1_aces > t2_aces else 0
        features['avg_aces'] = (t1_aces + t2_aces) / 2
        
        # Entry kills
        t1_entry = team1_stats['advanced_stats'].get('avg_entry_kills', 0)
        t2_entry = team2_stats['advanced_stats'].get('avg_entry_kills', 0)
        features['entry_kills_diff'] = t1_entry - t2_entry
        features['better_entry_kills_team1'] = 1 if t1_entry > t2_entry else 0
        features['avg_entry_kills'] = (t1_entry + t2_entry) / 2
        
        # First kill differential
        t1_fkdiff = team1_stats['advanced_stats'].get('avg_first_kill_diff', 0)
        t2_fkdiff = team2_stats['advanced_stats'].get('avg_first_kill_diff', 0)
        features['first_kill_diff_differential'] = t1_fkdiff - t2_fkdiff
        features['better_first_kill_diff_team1'] = 1 if t1_fkdiff > t2_fkdiff else 0
        features['avg_first_kill_diff'] = (t1_fkdiff + t2_fkdiff) / 2
        
        # Headshot percentage from advanced stats
        if 'avg_headshot_percentage' in team1_stats['advanced_stats'] and 'avg_headshot_percentage' in team2_stats['advanced_stats']:
            t1_hs_adv = team1_stats['advanced_stats'].get('avg_headshot_percentage', 0)
            t2_hs_adv = team2_stats['advanced_stats'].get('avg_headshot_percentage', 0)
            features['headshot_percentage_diff'] = t1_hs_adv - t2_hs_adv
            features['better_headshot_percentage_team1'] = 1 if t1_hs_adv > t2_hs_adv else 0
            features['avg_headshot_percentage'] = (t1_hs_adv + t2_hs_adv) / 2
        
        # KAST from advanced stats
        if 'avg_kast' in team1_stats['advanced_stats'] and 'avg_kast' in team2_stats['advanced_stats']:
            t1_kast_adv = team1_stats['advanced_stats'].get('avg_kast', 0)
            t2_kast_adv = team2_stats['advanced_stats'].get('avg_kast', 0)
            features['kast_adv_diff'] = t1_kast_adv - t2_kast_adv
            features['better_kast_adv_team1'] = 1 if t1_kast_adv > t2_kast_adv else 0
            features['avg_kast_adv'] = (t1_kast_adv + t2_kast_adv) / 2
        
        # ADR from advanced stats
        if 'avg_adr' in team1_stats['advanced_stats'] and 'avg_adr' in team2_stats['advanced_stats']:
            t1_adr_adv = team1_stats['advanced_stats'].get('avg_adr', 0)
            t2_adr_adv = team2_stats['advanced_stats'].get('avg_adr', 0)
            features['adr_adv_diff'] = t1_adr_adv - t2_adr_adv
            features['better_adr_adv_team1'] = 1 if t1_adr_adv > t2_adr_adv else 0
            features['avg_adr_adv'] = (t1_adr_adv + t2_adr_adv) / 2
        
        # Agent usage comparison (if available)
        if ('agent_usage' in team1_stats['advanced_stats'] and 
            'agent_usage' in team2_stats['advanced_stats']):
            
            t1_agents = team1_stats['advanced_stats']['agent_usage']
            t2_agents = team2_stats['advanced_stats']['agent_usage']
            
            # Find most used agents
            t1_top_agent = next(iter(t1_agents.items()))[0] if t1_agents else None
            t2_top_agent = next(iter(t2_agents.items()))[0] if t2_agents else None
            
            if t1_top_agent and t2_top_agent:
                features['same_top_agent'] = 1 if t1_top_agent == t2_top_agent else 0
            
            # Calculate agent overlap (Jaccard similarity)
            t1_agent_set = set(t1_agents.keys())
            t2_agent_set = set(t2_agents.keys())
            
            intersection = len(t1_agent_set.intersection(t2_agent_set))
            union = len(t1_agent_set.union(t2_agent_set))
            
            features['agent_overlap'] = intersection / union if union > 0 else 0
    
    #----------------------------------------
    # 6. MAP STATS
    #----------------------------------------
    if ('map_performance' in team1_stats and 'map_performance' in team2_stats and
        team1_stats.get('map_performance') and team2_stats.get('map_performance')):
        
        # Find common maps
        common_maps = set(team1_stats['map_performance'].keys()) & set(team2_stats['map_performance'].keys())
        features['common_map_count'] = len(common_maps)
        
        # Calculate overall map win rate difference
        t1_map_win_rates = [team1_stats['map_performance'][m].get('win_rate', 0) for m in team1_stats['map_performance'] if m != 'Unknown']
        t2_map_win_rates = [team2_stats['map_performance'][m].get('win_rate', 0) for m in team2_stats['map_performance'] if m != 'Unknown']
        
        if t1_map_win_rates and t2_map_win_rates:
            t1_avg_map_win_rate = sum(t1_map_win_rates) / len(t1_map_win_rates)
            t2_avg_map_win_rate = sum(t2_map_win_rates) / len(t2_map_win_rates)
            
            features['overall_map_win_rate_diff'] = t1_avg_map_win_rate - t2_avg_map_win_rate
            features['better_map_win_rate_team1'] = 1 if t1_avg_map_win_rate > t2_avg_map_win_rate else 0
        
        # For each common map, create symmetrical features
        for map_name in common_maps:
            if map_name != 'Unknown':
                t1_map = team1_stats['map_performance'][map_name]
                t2_map = team2_stats['map_performance'][map_name]
                
                # Map win rate
                t1_wr = t1_map.get('win_rate', 0)
                t2_wr = t2_map.get('win_rate', 0)
                map_key = map_name.replace(' ', '_').lower()
                
                features[f'{map_key}_win_rate_diff'] = t1_wr - t2_wr
                features[f'better_{map_key}_team1'] = 1 if t1_wr > t2_wr else 0
                features[f'avg_{map_key}_win_rate'] = (t1_wr + t2_wr) / 2
                
                # Attack win rate if available
                if 'attack_win_rate' in t1_map and 'attack_win_rate' in t2_map:
                    t1_attack = t1_map.get('attack_win_rate', 0)
                    t2_attack = t2_map.get('attack_win_rate', 0)
                    
                    features[f'{map_key}_attack_diff'] = t1_attack - t2_attack
                    features[f'better_{map_key}_attack_team1'] = 1 if t1_attack > t2_attack else 0
                    
                # Defense win rate if available
                if 'defense_win_rate' in t1_map and 'defense_win_rate' in t2_map:
                    t1_defense = t1_map.get('defense_win_rate', 0)
                    t2_defense = t2_map.get('defense_win_rate', 0)
                    
                    features[f'{map_key}_defense_diff'] = t1_defense - t2_defense
                    features[f'better_{map_key}_defense_team1'] = 1 if t1_defense > t2_defense else 0
                    features[f'avg_{map_key}_defense'] = (t1_defense + t2_defense) / 2
                
                # Round win rates
                if 'round_win_rate' in t1_map and 'round_win_rate' in t2_map:
                    t1_round = t1_map.get('round_win_rate', 0)
                    t2_round = t2_map.get('round_win_rate', 0)
                    
                    features[f'{map_key}_round_win_rate_diff'] = t1_round - t2_round
                    features[f'better_{map_key}_round_team1'] = 1 if t1_round > t2_round else 0
                    features[f'avg_{map_key}_round'] = (t1_round + t2_round) / 2
    
    #----------------------------------------
    # 7. OPPONENT QUALITY
    #----------------------------------------
    if ('opponent_quality' in team1_stats and 'opponent_quality' in team2_stats and
        team1_stats.get('opponent_quality') and team2_stats.get('opponent_quality')):
        
        # Average opponent ranking
        t1_opp_rank = team1_stats['opponent_quality'].get('avg_opponent_ranking', 0)
        t2_opp_rank = team2_stats['opponent_quality'].get('avg_opponent_ranking', 0)
        # Lower rank is better, so reverse the comparison
        features['opponent_rank_diff'] = t2_opp_rank - t1_opp_rank  
        features['better_opponent_quality_team1'] = 1 if t1_opp_rank < t2_opp_rank else 0
        features['avg_opponent_rank'] = (t1_opp_rank + t2_opp_rank) / 2
        
        # Average opponent rating
        t1_opp_rating = team1_stats['opponent_quality'].get('avg_opponent_rating', 0)
        t2_opp_rating = team2_stats['opponent_quality'].get('avg_opponent_rating', 0)
        features['opponent_rating_diff'] = t1_opp_rating - t2_opp_rating
        features['better_opponent_rating_team1'] = 1 if t1_opp_rating > t2_opp_rating else 0
        features['avg_opponent_rating'] = (t1_opp_rating + t2_opp_rating) / 2
        
        # Top 10 win rate
        t1_top10 = team1_stats['opponent_quality'].get('top_10_win_rate', 0)
        t2_top10 = team2_stats['opponent_quality'].get('top_10_win_rate', 0)
        features['top_10_win_rate_diff'] = t1_top10 - t2_top10
        features['better_top_10_team1'] = 1 if t1_top10 > t2_top10 else 0
        features['avg_top_10_win_rate'] = (t1_top10 + t2_top10) / 2
        
        # Bottom 50 win rate
        t1_bot50 = team1_stats['opponent_quality'].get('bottom_50_win_rate', 0)
        t2_bot50 = team2_stats['opponent_quality'].get('bottom_50_win_rate', 0)
        features['bottom_50_win_rate_diff'] = t1_bot50 - t2_bot50
        features['better_bottom_50_team1'] = 1 if t1_bot50 > t2_bot50 else 0
        features['avg_bottom_50_win_rate'] = (t1_bot50 + t2_bot50) / 2
        
        # Upset factor
        t1_upset = team1_stats['opponent_quality'].get('upset_factor', 0)
        t2_upset = team2_stats['opponent_quality'].get('upset_factor', 0)
        features['upset_factor_diff'] = t1_upset - t2_upset
        features['better_upset_team1'] = 1 if t1_upset > t2_upset else 0
        features['avg_upset_factor'] = (t1_upset + t2_upset) / 2
        
        # Upset vulnerability
        t1_vuln = team1_stats['opponent_quality'].get('upset_vulnerability', 0)
        t2_vuln = team2_stats['opponent_quality'].get('upset_vulnerability', 0)
        # Lower vulnerability is better, so reverse comparison
        features['upset_vulnerability_diff'] = t2_vuln - t1_vuln
        features['less_vulnerable_team1'] = 1 if t1_vuln < t2_vuln else 0
        features['avg_upset_vulnerability'] = (t1_vuln + t2_vuln) / 2
        
        # Team's own ranking and rating
        t1_rank = team1_stats['opponent_quality'].get('team_ranking', 0)
        t2_rank = team2_stats['opponent_quality'].get('team_ranking', 0)
        if t1_rank and t2_rank:
            # Lower rank is better, so reverse comparison
            features['team_rank_diff'] = t2_rank - t1_rank
            features['better_ranked_team1'] = 1 if t1_rank < t2_rank else 0
            features['avg_team_rank'] = (t1_rank + t2_rank) / 2
            
            # Calculate rank gap (absolute)
            features['rank_gap'] = abs(t1_rank - t2_rank)
        
        t1_rating = team1_stats['opponent_quality'].get('team_rating', 0)
        t2_rating = team2_stats['opponent_quality'].get('team_rating', 0)
        if t1_rating and t2_rating:
            features['team_rating_quality_diff'] = t1_rating - t2_rating
            features['better_rated_quality_team1'] = 1 if t1_rating > t2_rating else 0
            features['avg_team_rating_quality'] = (t1_rating + t2_rating) / 2
            
            # Calculate rating gap (absolute)
            features['rating_gap'] = abs(t1_rating - t2_rating)
    
    #----------------------------------------
    # 8. TOURNAMENT PERFORMANCE
    #----------------------------------------
    if ('tournament_performance' in team1_stats and 'tournament_performance' in team2_stats and
        team1_stats.get('tournament_performance') and team2_stats.get('tournament_performance')):
        
        # Calculate average tournament win rates
        t1_tournaments = team1_stats['tournament_performance']
        t2_tournaments = team2_stats['tournament_performance']
        
        # High tier tournament performance (if available)
        t1_high_tourneys = [t for k, t in t1_tournaments.items() if t.get('tier', 0) == 3]
        t2_high_tourneys = [t for k, t in t2_tournaments.items() if t.get('tier', 0) == 3]
        
        if t1_high_tourneys and t2_high_tourneys:
            t1_high_wr = sum(t.get('win_rate', 0) for t in t1_high_tourneys) / len(t1_high_tourneys)
            t2_high_wr = sum(t.get('win_rate', 0) for t in t2_high_tourneys) / len(t2_high_tourneys)
            
            features['high_tier_tourney_diff'] = t1_high_wr - t2_high_wr
            features['better_high_tier_team1'] = 1 if t1_high_wr > t2_high_wr else 0
            features['avg_high_tier_win_rate'] = (t1_high_wr + t2_high_wr) / 2
        
        # Mid tier tournament performance
        t1_mid_tourneys = [t for k, t in t1_tournaments.items() if t.get('tier', 0) == 2]
        t2_mid_tourneys = [t for k, t in t2_tournaments.items() if t.get('tier', 0) == 2]
        
        if t1_mid_tourneys and t2_mid_tourneys:
            t1_mid_wr = sum(t.get('win_rate', 0) for t in t1_mid_tourneys) / len(t1_mid_tourneys)
            t2_mid_wr = sum(t.get('win_rate', 0) for t in t2_mid_tourneys) / len(t2_mid_tourneys)
            
            features['mid_tier_tourney_diff'] = t1_mid_wr - t2_mid_wr
            features['better_mid_tier_team1'] = 1 if t1_mid_wr > t2_mid_wr else 0
            features['avg_mid_tier_win_rate'] = (t1_mid_wr + t2_mid_wr) / 2
        
        # Overall tournament performance
        t1_all_tourneys = list(t1_tournaments.values())
        t2_all_tourneys = list(t2_tournaments.values())
        
        if t1_all_tourneys and t2_all_tourneys:
            t1_overall_wr = sum(t.get('win_rate', 0) for t in t1_all_tourneys) / len(t1_all_tourneys)
            t2_overall_wr = sum(t.get('win_rate', 0) for t in t2_all_tourneys) / len(t2_all_tourneys)
            
            features['overall_tourney_diff'] = t1_overall_wr - t2_overall_wr
            features['better_tourney_team1'] = 1 if t1_overall_wr > t2_overall_wr else 0
            features['avg_tourney_win_rate'] = (t1_overall_wr + t2_overall_wr) / 2
    
    #----------------------------------------
    # 9. H2H STATS
    #----------------------------------------
    # Add head-to-head statistics if available
    if ('opponent_stats' in team1_stats and 'opponent_stats' in team2_stats):
        # Try to find team2 in team1's opponents
        team2_name = None
        for opponent in team1_stats['opponent_stats']:
            # Try case-insensitive match
            if str(opponent).lower() == str(team2_stats.get('team_name', '')).lower():
                team2_name = opponent
                break
        
        # If not found, try a fuzzy match
        if not team2_name:
            for opponent in team1_stats['opponent_stats']:
                # Try partial match
                if (str(opponent).lower() in str(team2_stats.get('team_name', '')).lower() or
                    str(team2_stats.get('team_name', '')).lower() in str(opponent).lower()):
                    team2_name = opponent
                    break
        
        if team2_name and team2_name in team1_stats['opponent_stats']:
            h2h_stats = team1_stats['opponent_stats'][team2_name]
            
            # Add head-to-head metrics
            features['h2h_win_rate'] = h2h_stats.get('win_rate', 0.5)  # From team1's perspective
            features['h2h_matches'] = h2h_stats.get('matches', 0)
            features['h2h_score_diff'] = h2h_stats.get('score_differential', 0)
            
            # Create binary indicators
            features['h2h_advantage_team1'] = 1 if features['h2h_win_rate'] > 0.5 else 0
            features['h2h_significant'] = 1 if features['h2h_matches'] >= 3 else 0
            
            # Create h2h confidence based on sample size
            if features['h2h_matches'] > 0:
                features['h2h_confidence'] = 1.0 - (1.0 / (1.0 + 0.1 * features['h2h_matches']))
            else:
                features['h2h_confidence'] = 0
    
    #----------------------------------------
    # 10. INTERACTION TERMS
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
    
    # Star player interactions
    if 'star_player_diff' in features and 'consistency_diff' in features:
        features['star_x_consistency'] = features['star_player_diff'] * features['consistency_diff']
    
    # H2H and recent form
    if 'h2h_win_rate' in features and 'recent_form_diff' in features:
        features['h2h_x_form'] = (features['h2h_win_rate'] - 0.5) * features['recent_form_diff']
    
    # Headshot and KD interactions
    if 'headshot_diff' in features and 'kd_diff' in features:
        features['headshot_x_kd'] = features['headshot_diff'] * features['kd_diff']
    
    # Win rate and opponent quality
    if 'win_rate_diff' in features and 'opponent_rank_diff' in features:
        features['win_rate_x_opp_quality'] = features['win_rate_diff'] * features['opponent_rank_diff']
    
    # First blood interactions
    if 'first_bloods_diff' in features and 'win_rate_diff' in features:
        features['first_blood_x_win_rate'] = features['first_bloods_diff'] * features['win_rate_diff']
    
    # Clutch and consistency interactions
    if 'clutches_diff' in features and 'consistency_diff' in features:
        features['clutch_x_consistency'] = features['clutches_diff'] * features['consistency_diff']
    
    #----------------------------------------
    # 11. META FEATURES
    #----------------------------------------
    # Create "is better" aggregated count
    better_cols = [col for col in features.keys() if col.startswith('better_') and '_team1' in col]
    if better_cols:
        features['team1_better_count'] = sum(features[col] for col in better_cols)
        features['team1_better_ratio'] = features['team1_better_count'] / len(better_cols)
    
    # Add categorical features for significant differences
    for diff_col in [col for col in features.keys() if col.endswith('_diff')]:
        base_name = diff_col.replace('_diff', '')
        diff_value = features[diff_col]
        
        # Create binary indicators for significant differences
        if abs(diff_value) > 0.1:  # 10% threshold
            features[f'{base_name}_significant_diff'] = 1
            features[f'{base_name}_significant_advantage_team1'] = 1 if diff_value > 0 else 0
        else:
            features[f'{base_name}_significant_diff'] = 0
            features[f'{base_name}_significant_advantage_team1'] = 0
    
    return features    

def prepare_data_for_model_with_economy(team1_stats, team2_stats):
    """Prepare data for the ML model using symmetrical features to reduce overfitting."""
    if not team1_stats or not team2_stats:
        return None
    
    print("\n===== ADDING SYMMETRICAL FEATURES FOR TRAINING =====")
    
    # Create completely symmetrical features
    features = prepare_fully_symmetrical_data(team1_stats, team2_stats)
    
    # Add map-specific features if available for both teams
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
            
            # Win percentage comparison
            features[f'{map_name}_win_rate_diff'] = t1_map['win_percentage'] - t2_map['win_percentage']
            features[f'better_{map_name}_team1'] = 1 if t1_map['win_percentage'] > t2_map['win_percentage'] else 0
            
            # Side preference comparison
            features[f'{map_name}_side_pref_diff'] = (1 if t1_map['side_preference'] == 'Attack' else -1) - (1 if t2_map['side_preference'] == 'Attack' else -1)
            features[f'{map_name}_atk_win_rate_diff'] = t1_map['atk_win_rate'] - t2_map['atk_win_rate']
            features[f'{map_name}_def_win_rate_diff'] = t1_map['def_win_rate'] - t2_map['def_win_rate']
            
            # Overtime performance
            if t1_map['overtime_stats']['matches'] > 0 and t2_map['overtime_stats']['matches'] > 0:
                features[f'{map_name}_ot_win_rate_diff'] = t1_map['overtime_stats']['win_rate'] - t2_map['overtime_stats']['win_rate']
            
            # Recent form on this map
            features[f'{map_name}_recent_form_diff'] = t1_map['recent_form'] - t2_map['recent_form']
            
            # Agent flexibility
            features[f'{map_name}_comp_variety_diff'] = t1_map['composition_variety'] - t2_map['composition_variety']
            
            # Common agents - calculate overlap in most played agents
            t1_agents = set(t1_map['most_played_agents'])
            t2_agents = set(t2_map['most_played_agents'])
            agent_overlap = len(t1_agents.intersection(t2_agents))
            features[f'{map_name}_agent_overlap'] = agent_overlap
        
        # Add strongest/weakest map comparisons
        if 'strongest_maps' in team1_stats and 'strongest_maps' in team2_stats:
            t1_strong = set(team1_stats.get('strongest_maps', []))
            t2_strong = set(team2_stats.get('strongest_maps', []))
            
            # Calculate strongest map overlap
            strong_overlap = len(t1_strong.intersection(t2_strong))
            features['strong_map_overlap'] = strong_overlap
            
            # Calculate if team1's strongest maps are team2's weakest
            t2_weak = set(team2_stats.get('weakest_maps', []))
            advantage_maps = len(t1_strong.intersection(t2_weak))
            features['team1_map_advantage_count'] = advantage_maps
            
            # And vice versa
            t1_weak = set(team1_stats.get('weakest_maps', []))
            disadvantage_maps = len(t2_strong.intersection(t1_weak))
            features['team1_map_disadvantage_count'] = disadvantage_maps
        
        # Add aggregate map-based metrics
        if 'avg_map_win_rate' in team1_stats and 'avg_map_win_rate' in team2_stats:
            features['avg_map_win_rate_diff'] = team1_stats['avg_map_win_rate'] - team2_stats['avg_map_win_rate']
        
        if 'avg_atk_win_rate' in team1_stats and 'avg_atk_win_rate' in team2_stats:
            features['avg_atk_win_rate_diff'] = team1_stats['avg_atk_win_rate'] - team2_stats['avg_atk_win_rate']
            features['avg_def_win_rate_diff'] = team1_stats['avg_def_win_rate'] - team2_stats['avg_def_win_rate']
        
        if 'overtime_win_rate' in team1_stats and 'overtime_win_rate' in team2_stats:
            features['overtime_win_rate_diff'] = team1_stats['overtime_win_rate'] - team2_stats['overtime_win_rate']
            features['better_overtime_team1'] = 1 if team1_stats['overtime_win_rate'] > team2_stats['overtime_win_rate'] else 0

    # Debug: Check for non-numeric values
    for key, value in list(features.items()):
        if not isinstance(value, (int, float)):
            print(f"WARNING: Non-numeric feature detected: {key} = {value} (type: {type(value)})")
            # Remove problematic features
            del features[key]
    
    print(f"\nTotal features for training: {len(features)}")
    print("===== FEATURE PREPARATION COMPLETE =====\n")
    
    return features

def display_prediction_results_with_economy_and_maps(prediction, team1_stats, team2_stats):
    """Display detailed prediction results including economy data and map-specific statistics."""
    # First use the original display function
    display_prediction_results_with_economy(prediction, team1_stats, team2_stats)
    
    # Now add map-specific display
    if not prediction:
        return
    
    team1 = prediction['team1']
    team2 = prediction['team2']
    
    width = 70  # Total width of the display
    
    # Check if map statistics are available
    if ('map_statistics' in team1_stats and 'map_statistics' in team2_stats and
        team1_stats['map_statistics'] and team2_stats['map_statistics']):
        
        print("\n" + "=" * width)
        print(f"{' Map Performance Comparison ':=^{width}}")
        
        # Find common maps
        team1_maps = set(team1_stats['map_statistics'].keys())
        team2_maps = set(team2_stats['map_statistics'].keys())
        common_maps = sorted(team1_maps.intersection(team2_maps))
        
        if common_maps:
            print(f"{'Map':<10} {'Win Rate':<23} {'ATK Win Rate':<23} {'DEF Win Rate'}")
            print("-" * width)
            
            for map_name in common_maps:
                t1_map = team1_stats['map_statistics'][map_name]
                t2_map = team2_stats['map_statistics'][map_name]
                
                t1_win = f"{t1_map['win_percentage']*100:.1f}%"
                t2_win = f"{t2_map['win_percentage']*100:.1f}%"
                
                t1_atk = f"{t1_map['atk_win_rate']*100:.1f}%"
                t2_atk = f"{t2_map['atk_win_rate']*100:.1f}%"
                
                t1_def = f"{t1_map['def_win_rate']*100:.1f}%"
                t2_def = f"{t2_map['def_win_rate']*100:.1f}%"
                
                print(f"{map_name:<10} {team1}: {t1_win:<12} {team2}: {t2_win:<7} "
                      f"{team1}: {t1_atk:<12} {team2}: {t2_atk:<7} "
                      f"{team1}: {t1_def:<12} {team2}: {t2_def:<7}")
            
            # Print side preferences
            print()
            print(f"{' Side Preferences ':—^{width}}")
            for map_name in common_maps:
                t1_map = team1_stats['map_statistics'][map_name]
                t2_map = team2_stats['map_statistics'][map_name]
                
                t1_pref = t1_map['side_preference']
                t2_pref = t2_map['side_preference']
                
                t1_strength = t1_map['side_preference_strength'] * 100
                t2_strength = t2_map['side_preference_strength'] * 100
                
                print(f"{map_name:<10} {team1}: {t1_pref:<8} ({t1_strength:.1f}% advantage)   "
                      f"{team2}: {t2_pref:<8} ({t2_strength:.1f}% advantage)")
            
            # Print overtime performance
            print()
            print(f"{' Overtime Performance ':—^{width}}")
            for map_name in common_maps:
                t1_map = team1_stats['map_statistics'][map_name]
                t2_map = team2_stats['map_statistics'][map_name]
                
                t1_ot_matches = t1_map['overtime_stats']['matches']
                t2_ot_matches = t2_map['overtime_stats']['matches']
                
                if t1_ot_matches > 0 or t2_ot_matches > 0:
                    t1_ot_rate = t1_map['overtime_stats']['win_rate'] * 100 if t1_ot_matches > 0 else 'N/A'
                    t2_ot_rate = t2_map['overtime_stats']['win_rate'] * 100 if t2_ot_matches > 0 else 'N/A'
                    
                    t1_display = f"{t1_ot_rate:.1f}% ({t1_ot_matches} matches)" if t1_ot_matches > 0 else "N/A"
                    t2_display = f"{t2_ot_rate:.1f}% ({t2_ot_matches} matches)" if t2_ot_matches > 0 else "N/A"
                    
                    print(f"{map_name:<10} {team1}: {t1_display:<25} {team2}: {t2_display}")
            
            # Print top agent compositions
            print()
            print(f"{' Top Agent Compositions ':—^{width}}")
            for map_name in common_maps:
                t1_map = team1_stats['map_statistics'][map_name]
                t2_map = team2_stats['map_statistics'][map_name]
                
                if t1_map['most_played_composition'] or t2_map['most_played_composition']:
                    print(f"{map_name}:")
                    
                    if t1_map['most_played_composition']:
                        t1_comp = ', '.join(t1_map['most_played_composition'])
                        print(f"  {team1}: {t1_comp}")
                    
                    if t2_map['most_played_composition']:
                        t2_comp = ', '.join(t2_map['most_played_composition'])
                        print(f"  {team2}: {t2_comp}")
                    
                    print()
        
        # Print strongest/weakest maps
        if ('strongest_maps' in team1_stats and 'strongest_maps' in team2_stats and
            team1_stats['strongest_maps'] and team2_stats['strongest_maps']):
            
            print(f"{' Strongest and Weakest Maps ':—^{width}}")
            
            # Team 1 strong maps
            t1_strong = ', '.join(team1_stats['strongest_maps'])
            print(f"{team1} strongest maps: {t1_strong}")
            
            # Team 2 strong maps
            t2_strong = ', '.join(team2_stats['strongest_maps'])
            print(f"{team2} strongest maps: {t2_strong}")
            
            if 'weakest_maps' in team1_stats and 'weakest_maps' in team2_stats:
                # Team 1 weak maps
                t1_weak = ', '.join(team1_stats['weakest_maps'])
                print(f"{team1} weakest maps: {t1_weak}")
                
                # Team 2 weak maps
                t2_weak = ', '.join(team2_stats['weakest_maps'])
                print(f"{team2} weakest maps: {t2_weak}")
            
            # Calculate potential map advantages
            t1_strong_set = set(team1_stats['strongest_maps'])
            t2_weak_set = set(team2_stats.get('weakest_maps', []))
            t1_advantage_maps = t1_strong_set.intersection(t2_weak_set)
            
            t2_strong_set = set(team2_stats['strongest_maps'])
            t1_weak_set = set(team1_stats.get('weakest_maps', []))
            t2_advantage_maps = t2_strong_set.intersection(t1_weak_set)
            
            if t1_advantage_maps:
                print(f"\nPotential {team1} advantage maps: {', '.join(t1_advantage_maps)}")
            
            if t2_advantage_maps:
                print(f"Potential {team2} advantage maps: {', '.join(t2_advantage_maps)}")
        
        print("=" * width + "\n")

def visualize_prediction_with_economy_and_maps(prediction_result):
    """Visualize the match prediction with player stats, economy data, and map-specific information."""
    if not prediction_result:
        print("No prediction to visualize.")
        return
    
    team1 = prediction_result['team1']
    team2 = prediction_result['team2']
    team1_prob = prediction_result['team1_win_probability']
    team2_prob = prediction_result['team2_win_probability']
    predicted_winner = prediction_result['predicted_winner']
    confidence = prediction_result['confidence']
    
    # Create figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Bar chart for win probabilities in first subplot
    teams = [team1, team2]
    probs = [team1_prob, team2_prob]
    colors = ['#3498db' if team == predicted_winner else '#e74c3c' for team in teams]
    
    bars = ax1.bar(teams, probs, color=colors, alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=12)
    
    # Add title and labels
    ax1.set_title(f'Win Probabilities', fontsize=16)
    ax1.set_ylabel('Win Probability', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Player stats comparison in second subplot
    t1_summary = prediction_result.get('team1_stats_summary', {})
    t2_summary = prediction_result.get('team2_stats_summary', {})
    
    # Extract player stats if available
    metrics = []
    t1_values = []
    t2_values = []
    
    if 'avg_player_rating' in t1_summary and 'avg_player_rating' in t2_summary:
        metrics.append('Player Rating')
        t1_values.append(t1_summary['avg_player_rating'])
        t2_values.append(t2_summary['avg_player_rating'])
        
        if 'star_player_rating' in t1_summary and 'star_player_rating' in t2_summary:
            metrics.append('Star Player Rating')
            t1_values.append(t1_summary['star_player_rating'])
            t2_values.append(t2_summary['star_player_rating'])
    
    # Add more traditional metrics
    metrics.extend(['Win Rate', 'Recent Form'])
    t1_values.extend([t1_summary.get('win_rate', 0), t1_summary.get('recent_form', 0)])
    t2_values.extend([t2_summary.get('win_rate', 0), t2_summary.get('recent_form', 0)])
    
    # Convert to numpy arrays for easier computation
    metrics = np.array(metrics)
    t1_values = np.array(t1_values)
    t2_values = np.array(t2_values)
    
    # Set up bar positions
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create bars
    rects1 = ax2.bar(x - width/2, t1_values, width, label=team1, color='#3498db', alpha=0.7)
    rects2 = ax2.bar(x + width/2, t2_values, width, label=team2, color='#e74c3c', alpha=0.7)
    
    # Add value labels
    for rect in rects1:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
                
    for rect in rects2:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    ax2.set_title(f'Team Stats Comparison', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=0)
    ax2.legend()
    
    # Economy metrics in third subplot
    econ_metrics = []
    t1_econ_values = []
    t2_econ_values = []
    
    # Check if economy metrics are available
    if ('pistol_win_rate' in t1_summary and 'pistol_win_rate' in t2_summary):
        econ_metrics.extend(['Pistol Win Rate', 'Eco Win Rate', 'Full Buy Win Rate'])
        t1_econ_values.extend([
            t1_summary.get('pistol_win_rate', 0),
            t1_summary.get('eco_win_rate', 0),
            t1_summary.get('full_buy_win_rate', 0)
        ])
        t2_econ_values.extend([
            t2_summary.get('pistol_win_rate', 0),
            t2_summary.get('eco_win_rate', 0),
            t2_summary.get('full_buy_win_rate', 0)
        ])
    
    if econ_metrics:
        # Convert to numpy arrays
        econ_metrics = np.array(econ_metrics)
        t1_econ_values = np.array(t1_econ_values)
        t2_econ_values = np.array(t2_econ_values)
        
        # Set up bar positions
        x_econ = np.arange(len(econ_metrics))
        
        # Create bars
        rects1_econ = ax3.bar(x_econ - width/2, t1_econ_values, width, label=team1, color='#3498db', alpha=0.7)
        rects2_econ = ax3.bar(x_econ + width/2, t2_econ_values, width, label=team2, color='#e74c3c', alpha=0.7)
        
        # Add value labels
        for rect in rects1_econ:
            height = rect.get_height()
            ax3.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
                    
        for rect in rects2_econ:
            height = rect.get_height()
            ax3.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        # Add labels and title
        ax3.set_title(f'Economy Performance', fontsize=16)
        ax3.set_xticks(x_econ)
        ax3.set_xticklabels(econ_metrics, rotation=15)
        ax3.set_ylim(0, 1)
        ax3.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax3.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax3.legend()
    else:
        # No economy data available
        ax3.text(0.5, 0.5, 'No Economy Data Available',
                ha='center', va='center', fontsize=14)
    
    # Map performance in fourth subplot
    team1_stats = prediction_result.get('team1_stats', {})
    team2_stats = prediction_result.get('team2_stats', {})
    
    if ('map_statistics' in team1_stats and 'map_statistics' in team2_stats and
        team1_stats['map_statistics'] and team2_stats['map_statistics']):
        
        # Find common maps
        team1_maps = set(team1_stats['map_statistics'].keys())
        team2_maps = set(team2_stats['map_statistics'].keys())
        common_maps = sorted(team1_maps.intersection(team2_maps))
        
        if common_maps:
            # Select up to 5 most played maps for visualization
            map_play_counts = []
            for map_name in common_maps:
                t1_count = team1_stats['map_statistics'][map_name]['matches_played']
                t2_count = team2_stats['map_statistics'][map_name]['matches_played']
                avg_count = (t1_count + t2_count) / 2
                map_play_counts.append((map_name, avg_count))
            
            # Sort by play count and take top 5
            top_maps = [m[0] for m in sorted(map_play_counts, key=lambda x: x[1], reverse=True)[:5]]
            
            # Get win rates for visualization
            map_names = []
            t1_map_win_rates = []
            t2_map_win_rates = []
            
            for map_name in top_maps:
                map_names.append(map_name)
                t1_map_win_rates.append(team1_stats['map_statistics'][map_name]['win_percentage'])
                t2_map_win_rates.append(team2_stats['map_statistics'][map_name]['win_percentage'])
            
            # Set up bar positions
            x_maps = np.arange(len(map_names))
            
            # Create bars
            rects1_maps = ax4.bar(x_maps - width/2, t1_map_win_rates, width, label=team1, color='#3498db', alpha=0.7)
            rects2_maps = ax4.bar(x_maps + width/2, t2_map_win_rates, width, label=team2, color='#e74c3c', alpha=0.7)
            
            # Add value labels
            for rect in rects1_maps:
                height = rect.get_height()
                ax4.text(rect.get_x() + rect.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=10)
                        
            for rect in rects2_maps:
                height = rect.get_height()
                ax4.text(rect.get_x() + rect.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=10)
            
            # Add labels and title
            ax4.set_title(f'Map Win Rates', fontsize=16)
            ax4.set_xticks(x_maps)
            ax4.set_xticklabels(map_names, rotation=0)
            ax4.set_ylim(0, 1)
            ax4.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax4.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax4.legend()
            
            # Highlight predicted best map
            best_diff = -1
            best_map = None
            worst_diff = -1
            worst_map = None
            
            for map_name in map_names:
                t1_win = team1_stats['map_statistics'][map_name]['win_percentage']
                t2_win = team2_stats['map_statistics'][map_name]['win_percentage']
                diff = t1_win - t2_win
                
                if diff > best_diff:
                    best_diff = diff
                    best_map = map_name
                
                if -diff > worst_diff:
                    worst_diff = -diff
                    worst_map = map_name
            
            if best_map:
                best_index = map_names.index(best_map)
                ax4.axvline(x=x_maps[best_index] - width/2, color='green', linestyle='--', alpha=0.7)
                ax4.text(x_maps[best_index] - width/2, 0.95, f"Best for {team1}", 
                        ha='center', va='top', rotation=90, color='green', fontsize=10)
            
            if worst_map:
                worst_index = map_names.index(worst_map)
                ax4.axvline(x=x_maps[worst_index] + width/2, color='green', linestyle='--', alpha=0.7)
                ax4.text(x_maps[worst_index] + width/2, 0.95, f"Best for {team2}", 
                        ha='center', va='top', rotation=90, color='green', fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'No Common Maps Data Available',
                    ha='center', va='center', fontsize=14)
    else:
        ax4.text(0.5, 0.5, 'No Map Stats Available',
                ha='center', va='center', fontsize=14)
    
    # Add prediction summary
    plt.figtext(0.5, 0.01, 
                f"Predicted Winner: {predicted_winner} (Confidence: {confidence:.1%})",
                ha="center", fontsize=14, bbox={"facecolor":"#f9f9f9", "alpha":0.5, "pad":5})
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('match_prediction_with_maps.png')
    plt.show()        

def build_training_dataset_with_economy(team_data_collection):
    """Build a training dataset with economy and player features from team data collection."""
    X = []
    y = []

        # Add debugging
    print(f"Building dataset from {len(team_data_collection)} teams")
    
    # Track team matches for debugging
    team_match_counts = {}
    for team_name, team_data in team_data_collection.items():
        match_count = len(team_data.get('matches', []))
        team_match_counts[team_name] = match_count
        print(f"Team: {team_name}, Matches: {match_count}")
    
    # Track match pairing attempts
    pairing_attempts = 0
    successful_pairings = 0

    # For each team, create positive samples (wins) and negative samples (losses)
    for team_name, team_data in list(team_data_collection.items())[:1]:  # Just check first team
        if 'matches' in team_data and team_data['matches']:
            print("\nDEBUG: Sample match structure:")
            sample_match = team_data['matches'][0]
            print(json.dumps(sample_match, indent=2))
            print("\nAvailable keys in match:", list(sample_match.keys()))
        
        # For each match, create a sample
        for match in team_data['matches']:
            # Skip matches without opponent name or result
            if not 'opponent_name' in match or not 'result' in match:
                print(f"Match for {team_name} missing opponent or result, skipping")
                continue
            
            # Get opponent data
            opponent_name = match['opponent_name']
            pairing_attempts += 1
            
            if opponent_name not in team_data_collection:
                print(f"Opponent {opponent_name} not in our dataset, skipping")
                continue
    
    economy_sample_count = 0
    player_stats_sample_count = 0
    both_sample_count = 0
    teams_with_economy = set()
    teams_with_player_stats = set()
    
    print("\n" + "="*50)
    print("TRAINING DATASET STATISTICS")
    print("="*50)
    
    # Track different types of features for debugging
    economy_features = set()
    player_features = set()
    map_features = set()
    
    # For each team, create positive samples (wins) and negative samples (losses)
    for team_name, team_data in team_data_collection.items():
        # Skip teams with no match data
        if not 'matches' in team_data or not team_data['matches']:
            continue
        
        # For each match, create a sample
        for match in team_data['matches']:
            # Skip matches without opponent name or result
            if not 'opponent_name' in match or not 'result' in match:
                continue
            
            # Get opponent data
            opponent_name = match['opponent_name']
            
            if opponent_name not in team_data_collection:
                continue
                
            opponent_data = team_data_collection[opponent_name]
            
            # Prepare sample features
            sample = {}
            
            # Add win rate diff
            if 'win_rate' in team_data and 'win_rate' in opponent_data:
                sample['win_rate_diff'] = team_data['win_rate'] - opponent_data['win_rate']
                sample['better_win_rate_team1'] = 1 if team_data['win_rate'] > opponent_data['win_rate'] else 0
                sample['avg_win_rate'] = (team_data['win_rate'] + opponent_data['win_rate']) / 2
            
            # Add recent form diff
            if 'recent_form' in team_data and 'recent_form' in opponent_data:
                sample['recent_form_diff'] = team_data['recent_form'] - opponent_data['recent_form']
                sample['better_recent_form_team1'] = 1 if team_data['recent_form'] > opponent_data['recent_form'] else 0
                sample['avg_recent_form'] = (team_data['recent_form'] + opponent_data['recent_form']) / 2

            # Add score differential
            if 'avg_score_diff' in team_data and 'avg_score_diff' in opponent_data:
                sample['score_diff_differential'] = team_data['avg_score_diff'] - opponent_data['avg_score_diff']
                sample['better_score_diff_team1'] = 1 if team_data['avg_score_diff'] > opponent_data['avg_score_diff'] else 0
            
            # Add match count
            if 'match_count' in team_data and 'match_count' in opponent_data:
                sample['total_matches'] = team_data['match_count'] + opponent_data['match_count']
                sample['match_count_diff'] = team_data['match_count'] - opponent_data['match_count']
            
            # Add wins and losses
            if 'wins' in team_data and 'wins' in opponent_data:
                sample['wins_diff'] = team_data['wins'] - opponent_data['wins']
            
            if 'losses' in team_data and 'losses' in opponent_data:
                sample['losses_diff'] = team_data['losses'] - opponent_data['losses']
            
            # Add win/loss ratio
            if 'win_loss_ratio' in team_data and 'win_loss_ratio' in opponent_data:
                sample['win_loss_ratio_diff'] = team_data['win_loss_ratio'] - opponent_data['win_loss_ratio']
            
            # Add avg score diff
            if 'avg_score_diff' in team_data and 'avg_score_diff' in opponent_data:
                sample['avg_score_diff'] = team_data['avg_score_diff'] - opponent_data['avg_score_diff']
                sample['better_avg_score_team1'] = 1 if team_data['avg_score_diff'] > opponent_data['avg_score_diff'] else 0
                sample['avg_score_metric'] = (team_data['avg_score_diff'] + opponent_data['avg_score_diff']) / 2
            
            # Add avg opponent score diff
            if 'avg_opponent_score_diff' in team_data and 'avg_opponent_score_diff' in opponent_data:
                sample['avg_opponent_score_diff'] = team_data['avg_opponent_score_diff'] - opponent_data['avg_opponent_score_diff']
            
            # Add defense/attacking metrics
            if 'defense_rating' in team_data and 'defense_rating' in opponent_data:
                sample['better_defense_team1'] = 1 if team_data['defense_rating'] > opponent_data['defense_rating'] else 0
                sample['avg_defense_metric'] = (team_data['defense_rating'] + opponent_data['defense_rating']) / 2
            
            # Add recent performance metrics
            for period in [5, 10, 20]:
                period_key = f'recent_{period}'
                if period_key in team_data and period_key in opponent_data:
                    sample[f'{period_key}_diff'] = team_data[period_key] - opponent_data[period_key]
                    sample[f'better_{period_key}_team1'] = 1 if team_data[period_key] > opponent_data[period_key] else 0
                    sample[f'avg_{period_key}'] = (team_data[period_key] + opponent_data[period_key]) / 2
            
            # Add momentum metrics
            for momentum_pair in [(5, 10), (10, 20)]:
                recent_a, recent_b = momentum_pair
                momentum_key = f'momentum_{recent_a}_vs_{recent_b}'
                if momentum_key in team_data and momentum_key in opponent_data:
                    sample[f'{momentum_key}_diff'] = team_data[momentum_key] - opponent_data[momentum_key]
                    sample[f'better_{momentum_key}_team1'] = 1 if team_data[momentum_key] > opponent_data[momentum_key] else 0
                    sample[f'avg_{momentum_key}'] = (team_data[momentum_key] + opponent_data[momentum_key]) / 2
            
            # Add weighted win rate
            if 'weighted_win_rate' in team_data and 'weighted_win_rate' in opponent_data:
                sample['weighted_win_rate_diff'] = team_data['weighted_win_rate'] - opponent_data['weighted_win_rate']
                sample['better_weighted_win_rate_team1'] = 1 if team_data['weighted_win_rate'] > opponent_data['weighted_win_rate'] else 0
                sample['avg_weighted_win_rate'] = (team_data['weighted_win_rate'] + opponent_data['weighted_win_rate']) / 2
            
            # Add player stats metrics if available for both teams
            has_player_stats = False
            
            if ('avg_player_rating' in team_data and team_data['avg_player_rating'] > 0 and
                'avg_player_rating' in opponent_data and opponent_data['avg_player_rating'] > 0):
                
                has_player_stats = True
                teams_with_player_stats.add(team_name)
                teams_with_player_stats.add(opponent_name)
                
                # Player rating
                sample['player_rating_diff'] = team_data['avg_player_rating'] - opponent_data['avg_player_rating']
                sample['better_player_rating_team1'] = 1 if team_data['avg_player_rating'] > opponent_data['avg_player_rating'] else 0
                sample['avg_player_rating'] = (team_data['avg_player_rating'] + opponent_data['avg_player_rating']) / 2
                
                player_features.add('player_rating_diff')
                player_features.add('better_player_rating_team1')
                player_features.add('avg_player_rating')
                
                # ACS
                if 'avg_player_acs' in team_data and 'avg_player_acs' in opponent_data:
                    sample['acs_diff'] = team_data['avg_player_acs'] - opponent_data['avg_player_acs']
                    sample['better_acs_team1'] = 1 if team_data['avg_player_acs'] > opponent_data['avg_player_acs'] else 0
                    sample['avg_acs'] = (team_data['avg_player_acs'] + opponent_data['avg_player_acs']) / 2
                    
                    player_features.add('acs_diff')
                    player_features.add('better_acs_team1')
                    player_features.add('avg_acs')
                
                # K/D
                if 'avg_player_kd' in team_data and 'avg_player_kd' in opponent_data:
                    sample['kd_diff'] = team_data['avg_player_kd'] - opponent_data['avg_player_kd']
                    sample['better_kd_team1'] = 1 if team_data['avg_player_kd'] > opponent_data['avg_player_kd'] else 0
                    sample['avg_kd'] = (team_data['avg_player_kd'] + opponent_data['avg_player_kd']) / 2
                    
                    player_features.add('kd_diff')
                    player_features.add('better_kd_team1')
                    player_features.add('avg_kd')
                
                # KAST
                if 'avg_player_kast' in team_data and 'avg_player_kast' in opponent_data:
                    sample['kast_diff'] = team_data['avg_player_kast'] - opponent_data['avg_player_kast']
                    sample['better_kast_team1'] = 1 if team_data['avg_player_kast'] > opponent_data['avg_player_kast'] else 0
                    sample['avg_kast'] = (team_data['avg_player_kast'] + opponent_data['avg_player_kast']) / 2
                    
                    player_features.add('kast_diff')
                    player_features.add('better_kast_team1')
                    player_features.add('avg_kast')
                
                # ADR
                if 'avg_player_adr' in team_data and 'avg_player_adr' in opponent_data:
                    sample['adr_diff'] = team_data['avg_player_adr'] - opponent_data['avg_player_adr']
                    sample['better_adr_team1'] = 1 if team_data['avg_player_adr'] > opponent_data['avg_player_adr'] else 0
                    sample['avg_adr'] = (team_data['avg_player_adr'] + opponent_data['avg_player_adr']) / 2
                    
                    player_features.add('adr_diff')
                    player_features.add('better_adr_team1')
                    player_features.add('avg_adr')
                
                # Headshot %
                if 'avg_player_headshot' in team_data and 'avg_player_headshot' in opponent_data:
                    sample['headshot_diff'] = team_data['avg_player_headshot'] - opponent_data['avg_player_headshot']
                    sample['better_headshot_team1'] = 1 if team_data['avg_player_headshot'] > opponent_data['avg_player_headshot'] else 0
                    sample['avg_headshot'] = (team_data['avg_player_headshot'] + opponent_data['avg_player_headshot']) / 2
                    
                    player_features.add('headshot_diff')
                    player_features.add('better_headshot_team1')
                    player_features.add('avg_headshot')
                
                # Star player comparison
                if ('star_player_rating' in team_data and 'star_player_rating' in opponent_data and
                    team_data['star_player_rating'] > 0 and opponent_data['star_player_rating'] > 0):
                    sample['star_player_diff'] = team_data['star_player_rating'] - opponent_data['star_player_rating']
                    sample['better_star_player_team1'] = 1 if team_data['star_player_rating'] > opponent_data['star_player_rating'] else 0
                    sample['avg_star_player'] = (team_data['star_player_rating'] + opponent_data['star_player_rating']) / 2
                    
                    player_features.add('star_player_diff')
                    player_features.add('better_star_player_team1')
                    player_features.add('avg_star_player')
                
                # Player consistency
                if 'player_consistency' in team_data and 'player_consistency' in opponent_data:
                    sample['consistency_diff'] = team_data['player_consistency'] - opponent_data['player_consistency']
                    sample['better_consistency_team1'] = 1 if team_data['player_consistency'] > opponent_data['player_consistency'] else 0
                    sample['avg_consistency'] = (team_data['player_consistency'] + opponent_data['player_consistency']) / 2
                    
                    player_features.add('consistency_diff')
                    player_features.add('better_consistency_team1')
                    player_features.add('avg_consistency')
                
                # First kills vs first deaths ratio
                if 'fk_fd_ratio' in team_data and 'fk_fd_ratio' in opponent_data:
                    sample['fk_fd_diff'] = team_data['fk_fd_ratio'] - opponent_data['fk_fd_ratio']
                    sample['better_fk_fd_team1'] = 1 if team_data['fk_fd_ratio'] > opponent_data['fk_fd_ratio'] else 0
                    sample['avg_fk_fd'] = (team_data['fk_fd_ratio'] + opponent_data['fk_fd_ratio']) / 2
                    
                    player_features.add('fk_fd_diff')
                    player_features.add('better_fk_fd_team1')
                    player_features.add('avg_fk_fd')
                
                # Player count difference
                if 'player_count' in team_data and 'player_count' in opponent_data:
                    sample['player_count_diff'] = team_data['player_count'] - opponent_data['player_count']
                    sample['player_count_ratio'] = team_data['player_count'] / max(opponent_data['player_count'], 1)
                    sample['avg_player_count'] = (team_data['player_count'] + opponent_data['player_count']) / 2
            
            # Add economy metrics if available for both teams
            has_economy_stats = False
            
            if ('pistol_win_rate' in team_data and 'pistol_win_rate' in opponent_data and
                'eco_win_rate' in team_data and 'eco_win_rate' in opponent_data and
                'full_buy_win_rate' in team_data and 'full_buy_win_rate' in opponent_data):
                
                has_economy_stats = True
                teams_with_economy.add(team_name)
                teams_with_economy.add(opponent_name)
                
                # Pistol round performance
                sample['pistol_win_rate_diff'] = team_data['pistol_win_rate'] - opponent_data['pistol_win_rate']
                sample['better_pistol_team1'] = 1 if team_data['pistol_win_rate'] > opponent_data['pistol_win_rate'] else 0
                sample['avg_pistol_win_rate'] = (team_data['pistol_win_rate'] + opponent_data['pistol_win_rate']) / 2
                
                economy_features.add('pistol_win_rate_diff')
                economy_features.add('better_pistol_team1')
                economy_features.add('avg_pistol_win_rate')
                
                # Eco round performance
                sample['eco_win_rate_diff'] = team_data['eco_win_rate'] - opponent_data['eco_win_rate']
                sample['better_eco_team1'] = 1 if team_data['eco_win_rate'] > opponent_data['eco_win_rate'] else 0
                sample['avg_eco_win_rate'] = (team_data['eco_win_rate'] + opponent_data['eco_win_rate']) / 2
                
                economy_features.add('eco_win_rate_diff')
                economy_features.add('better_eco_team1')
                economy_features.add('avg_eco_win_rate')
                
                # Semi-eco round performance
                if 'semi_eco_win_rate' in team_data and 'semi_eco_win_rate' in opponent_data:
                    sample['semi_eco_win_rate_diff'] = team_data['semi_eco_win_rate'] - opponent_data['semi_eco_win_rate']
                    sample['better_semi_eco_team1'] = 1 if team_data['semi_eco_win_rate'] > opponent_data['semi_eco_win_rate'] else 0
                    sample['avg_semi_eco_win_rate'] = (team_data['semi_eco_win_rate'] + opponent_data['semi_eco_win_rate']) / 2
                    
                    economy_features.add('semi_eco_win_rate_diff')
                    economy_features.add('better_semi_eco_team1')
                    economy_features.add('avg_semi_eco_win_rate')
                
                # Semi-buy round performance
                if 'semi_buy_win_rate' in team_data and 'semi_buy_win_rate' in opponent_data:
                    sample['semi_buy_win_rate_diff'] = team_data['semi_buy_win_rate'] - opponent_data['semi_buy_win_rate']
                    sample['better_semi_buy_team1'] = 1 if team_data['semi_buy_win_rate'] > opponent_data['semi_buy_win_rate'] else 0
                    sample['avg_semi_buy_win_rate'] = (team_data['semi_buy_win_rate'] + opponent_data['semi_buy_win_rate']) / 2
                    
                    economy_features.add('semi_buy_win_rate_diff')
                    economy_features.add('better_semi_buy_team1')
                    economy_features.add('avg_semi_buy_win_rate')
                
                # Full buy round performance
                sample['full_buy_win_rate_diff'] = team_data['full_buy_win_rate'] - opponent_data['full_buy_win_rate']
                sample['better_full_buy_team1'] = 1 if team_data['full_buy_win_rate'] > opponent_data['full_buy_win_rate'] else 0
                sample['avg_full_buy_win_rate'] = (team_data['full_buy_win_rate'] + opponent_data['full_buy_win_rate']) / 2
                
                economy_features.add('full_buy_win_rate_diff')
                economy_features.add('better_full_buy_team1')
                economy_features.add('avg_full_buy_win_rate')
                
                # Low economy performance (eco + semi-eco)
                if 'low_economy_win_rate' in team_data and 'low_economy_win_rate' in opponent_data:
                    sample['low_economy_win_rate_diff'] = team_data['low_economy_win_rate'] - opponent_data['low_economy_win_rate']
                    sample['better_low_economy_team1'] = 1 if team_data['low_economy_win_rate'] > opponent_data['low_economy_win_rate'] else 0
                    sample['avg_low_economy_win_rate'] = (team_data['low_economy_win_rate'] + opponent_data['low_economy_win_rate']) / 2
                    
                    economy_features.add('low_economy_win_rate_diff')
                    economy_features.add('better_low_economy_team1')
                    economy_features.add('avg_low_economy_win_rate')
                
                # High economy performance (semi-buy + full-buy)
                if 'high_economy_win_rate' in team_data and 'high_economy_win_rate' in opponent_data:
                    sample['high_economy_win_rate_diff'] = team_data['high_economy_win_rate'] - opponent_data['high_economy_win_rate']
                    sample['better_high_economy_team1'] = 1 if team_data['high_economy_win_rate'] > opponent_data['high_economy_win_rate'] else 0
                    sample['avg_high_economy_win_rate'] = (team_data['high_economy_win_rate'] + opponent_data['high_economy_win_rate']) / 2
                    
                    economy_features.add('high_economy_win_rate_diff')
                    economy_features.add('better_high_economy_team1')
                    economy_features.add('avg_high_economy_win_rate')
                
                # Economy efficiency metric
                if 'economy_efficiency' in team_data and 'economy_efficiency' in opponent_data:
                    sample['economy_efficiency_diff'] = team_data['economy_efficiency'] - opponent_data['economy_efficiency']
                    sample['better_economy_efficiency_team1'] = 1 if team_data['economy_efficiency'] > opponent_data['economy_efficiency'] else 0
                    sample['avg_economy_efficiency'] = (team_data['economy_efficiency'] + opponent_data['economy_efficiency']) / 2
                    
                    economy_features.add('economy_efficiency_diff')
                    economy_features.add('better_economy_efficiency_team1')
                    economy_features.add('avg_economy_efficiency')
                
                # Pistol round confidence
                if 'pistol_confidence' in team_data and 'pistol_confidence' in opponent_data:
                    sample['pistol_confidence_diff'] = team_data['pistol_confidence'] - opponent_data['pistol_confidence']
                    sample['better_pistol_confidence_team1'] = 1 if team_data['pistol_confidence'] > opponent_data['pistol_confidence'] else 0
                    sample['avg_pistol_confidence'] = (team_data['pistol_confidence'] + opponent_data['pistol_confidence']) / 2
                    
                    economy_features.add('pistol_confidence_diff')
                    economy_features.add('better_pistol_confidence_team1')
                    economy_features.add('avg_pistol_confidence')
                
                # Pistol round sample size
                if 'pistol_sample' in team_data and 'pistol_sample' in opponent_data:
                    sample['pistol_sample_diff'] = team_data['pistol_sample'] - opponent_data['pistol_sample']
                    sample['better_pistol_sample_team1'] = 1 if team_data['pistol_sample'] > opponent_data['pistol_sample'] else 0
                    sample['avg_pistol_sample'] = (team_data['pistol_sample'] + opponent_data['pistol_sample']) / 2
            
            # Add additional player metrics if available
            if ('first_bloods' in team_data and 'first_bloods' in opponent_data and
                team_data['first_bloods'] > 0 and opponent_data['first_bloods'] > 0):
                sample['first_bloods_diff'] = team_data['first_bloods'] - opponent_data['first_bloods']
                sample['better_first_bloods_team1'] = 1 if team_data['first_bloods'] > opponent_data['first_bloods'] else 0
                sample['avg_first_bloods'] = (team_data['first_bloods'] + opponent_data['first_bloods']) / 2
                
                player_features.add('first_bloods_diff')
                player_features.add('better_first_bloods_team1')
                player_features.add('avg_first_bloods')
            
            if ('clutches' in team_data and 'clutches' in opponent_data and
                team_data['clutches'] > 0 and opponent_data['clutches'] > 0):
                sample['clutches_diff'] = team_data['clutches'] - opponent_data['clutches']
                sample['better_clutches_team1'] = 1 if team_data['clutches'] > opponent_data['clutches'] else 0
                sample['avg_clutches'] = (team_data['clutches'] + opponent_data['clutches']) / 2
                
                player_features.add('clutches_diff')
                player_features.add('better_clutches_team1')
                player_features.add('avg_clutches')
            
            if ('aces' in team_data and 'aces' in opponent_data and
                team_data['aces'] > 0 and opponent_data['aces'] > 0):
                sample['aces_diff'] = team_data['aces'] - opponent_data['aces']
                sample['better_aces_team1'] = 1 if team_data['aces'] > opponent_data['aces'] else 0
                sample['avg_aces'] = (team_data['aces'] + opponent_data['aces']) / 2
                
                player_features.add('aces_diff')
                player_features.add('better_aces_team1')
                player_features.add('avg_aces')
            
            if ('entry_kills' in team_data and 'entry_kills' in opponent_data and
                team_data['entry_kills'] > 0 and opponent_data['entry_kills'] > 0):
                sample['entry_kills_diff'] = team_data['entry_kills'] - opponent_data['entry_kills']
                sample['better_entry_kills_team1'] = 1 if team_data['entry_kills'] > opponent_data['entry_kills'] else 0
                sample['avg_entry_kills'] = (team_data['entry_kills'] + opponent_data['entry_kills']) / 2
                
                player_features.add('entry_kills_diff')
                player_features.add('better_entry_kills_team1')
                player_features.add('avg_entry_kills')
                
            # Add first kill differential
            if ('first_kill_diff' in team_data and 'first_kill_diff' in opponent_data):
                sample['first_kill_diff_differential'] = team_data['first_kill_diff'] - opponent_data['first_kill_diff']
                sample['better_first_kill_diff_team1'] = 1 if team_data['first_kill_diff'] > opponent_data['first_kill_diff'] else 0
                sample['avg_first_kill_diff'] = (team_data['first_kill_diff'] + opponent_data['first_kill_diff']) / 2
                
                player_features.add('first_kill_diff_differential')
                player_features.add('better_first_kill_diff_team1')
                player_features.add('avg_first_kill_diff')
            
            # Add headshot percentage
            if ('headshot_percentage' in team_data and 'headshot_percentage' in opponent_data):
                sample['headshot_percentage_diff'] = team_data['headshot_percentage'] - opponent_data['headshot_percentage']
                sample['better_headshot_percentage_team1'] = 1 if team_data['headshot_percentage'] > opponent_data['headshot_percentage'] else 0
                sample['avg_headshot_percentage'] = (team_data['headshot_percentage'] + opponent_data['headshot_percentage']) / 2
                
                player_features.add('headshot_percentage_diff')
                player_features.add('better_headshot_percentage_team1')
                player_features.add('avg_headshot_percentage')
            
            # Add KAST advantage
            if ('kast_advantage' in team_data and 'kast_advantage' in opponent_data):
                sample['kast_adv_diff'] = team_data['kast_advantage'] - opponent_data['kast_advantage']
                sample['better_kast_adv_team1'] = 1 if team_data['kast_advantage'] > opponent_data['kast_advantage'] else 0
                sample['avg_kast_adv'] = (team_data['kast_advantage'] + opponent_data['kast_advantage']) / 2
                
                player_features.add('kast_adv_diff')
                player_features.add('better_kast_adv_team1')
                player_features.add('avg_kast_adv')
            
            # Add ADR advantage
            if ('adr_advantage' in team_data and 'adr_advantage' in opponent_data):
                sample['adr_adv_diff'] = team_data['adr_advantage'] - opponent_data['adr_advantage']
                sample['better_adr_adv_team1'] = 1 if team_data['adr_advantage'] > opponent_data['adr_advantage'] else 0
                sample['avg_adr_adv'] = (team_data['adr_advantage'] + opponent_data['adr_advantage']) / 2
                
                player_features.add('adr_adv_diff')
                player_features.add('better_adr_adv_team1')
                player_features.add('avg_adr_adv')

            # Add map-specific metrics
            if ('map_performance' in team_data and 'map_performance' in opponent_data and
                match.get('map_name') and match['map_name'] in team_data['map_performance'] and
                match['map_name'] in opponent_data['map_performance']):
                # Get map performance for this specific map
                team_map_perf = team_data['map_performance'][match['map_name']]
                opp_map_perf = opponent_data['map_performance'][match['map_name']]
                
                # Only include if we have win rates for both teams
                if ('win_rate' in team_map_perf and 'win_rate' in opp_map_perf):
                    # Convert dict metrics to scalars
                    map_name = match['map_name']
                    sample[f'map_{map_name}'] = 1
                    map_features.add(f'map_{map_name}')
            
            # Calculate common metrics - agent overlap
            if ('agent_composition' in team_data and 'agent_composition' in opponent_data and
                match.get('map_name') and match['map_name'] in team_data.get('agent_composition', {}) and
                match['map_name'] in opponent_data.get('agent_composition', {})):
                
                team_agents = team_data['agent_composition'][match['map_name']]
                opp_agents = opponent_data['agent_composition'][match['map_name']]
                
                if isinstance(team_agents, list) and isinstance(opp_agents, list):
                    overlap = len(set(team_agents) & set(opp_agents))
                    sample['agent_overlap'] = overlap
            
            # Calculate map overlap
            if 'map_performance' in team_data and 'map_performance' in opponent_data:
                team_maps = set(team_data['map_performance'].keys())
                opp_maps = set(opponent_data['map_performance'].keys())
                common_maps = len(team_maps & opp_maps)
                sample['common_map_count'] = common_maps
            
            # Add opponent quality metrics
            if ('opponent_quality' in team_data and 'opponent_quality' in opponent_data and
                'avg_opponent_rank' in team_data['opponent_quality'] and 'avg_opponent_rank' in opponent_data['opponent_quality']):
                
                sample['opponent_rank_diff'] = team_data['opponent_quality']['avg_opponent_rank'] - opponent_data['opponent_quality']['avg_opponent_rank']
                sample['better_opponent_quality_team1'] = 1 if team_data['opponent_quality']['avg_opponent_rank'] < opponent_data['opponent_quality']['avg_opponent_rank'] else 0
                sample['avg_opponent_rank'] = (team_data['opponent_quality']['avg_opponent_rank'] + opponent_data['opponent_quality']['avg_opponent_rank']) / 2
            
            if ('opponent_quality' in team_data and 'opponent_quality' in opponent_data and
                'avg_opponent_rating' in team_data['opponent_quality'] and 'avg_opponent_rating' in opponent_data['opponent_quality']):
                
                sample['opponent_rating_diff'] = team_data['opponent_quality']['avg_opponent_rating'] - opponent_data['opponent_quality']['avg_opponent_rating']
                sample['better_opponent_rating_team1'] = 1 if team_data['opponent_quality']['avg_opponent_rating'] > opponent_data['opponent_quality']['avg_opponent_rating'] else 0
                sample['avg_opponent_rating'] = (team_data['opponent_quality']['avg_opponent_rating'] + opponent_data['opponent_quality']['avg_opponent_rating']) / 2
            
            # Add performance against top teams
            if ('opponent_quality' in team_data and 'opponent_quality' in opponent_data and
                'top_10_win_rate' in team_data['opponent_quality'] and 'top_10_win_rate' in opponent_data['opponent_quality']):
                
                sample['top_10_win_rate_diff'] = team_data['opponent_quality']['top_10_win_rate'] - opponent_data['opponent_quality']['top_10_win_rate']
                sample['better_top_10_team1'] = 1 if team_data['opponent_quality']['top_10_win_rate'] > opponent_data['opponent_quality']['top_10_win_rate'] else 0
                sample['avg_top_10_win_rate'] = (team_data['opponent_quality']['top_10_win_rate'] + opponent_data['opponent_quality']['top_10_win_rate']) / 2
            
            # Add performance against bottom teams
            if ('opponent_quality' in team_data and 'opponent_quality' in opponent_data and
                'bottom_50_win_rate' in team_data['opponent_quality'] and 'bottom_50_win_rate' in opponent_data['opponent_quality']):
                
                sample['bottom_50_win_rate_diff'] = team_data['opponent_quality']['bottom_50_win_rate'] - opponent_data['opponent_quality']['bottom_50_win_rate']
                sample['better_bottom_50_team1'] = 1 if team_data['opponent_quality']['bottom_50_win_rate'] > opponent_data['opponent_quality']['bottom_50_win_rate'] else 0
                sample['avg_bottom_50_win_rate'] = (team_data['opponent_quality']['bottom_50_win_rate'] + opponent_data['opponent_quality']['bottom_50_win_rate']) / 2
            
            # Add upset metrics
            if ('opponent_quality' in team_data and 'opponent_quality' in opponent_data and
                'upset_factor' in team_data['opponent_quality'] and 'upset_factor' in opponent_data['opponent_quality']):
                
                sample['upset_factor_diff'] = team_data['opponent_quality']['upset_factor'] - opponent_data['opponent_quality']['upset_factor']
                sample['better_upset_team1'] = 1 if team_data['opponent_quality']['upset_factor'] > opponent_data['opponent_quality']['upset_factor'] else 0
                sample['avg_upset_factor'] = (team_data['opponent_quality']['upset_factor'] + opponent_data['opponent_quality']['upset_factor']) / 2
            
            # Add upset vulnerability metrics
            if ('opponent_quality' in team_data and 'opponent_quality' in opponent_data and
                'upset_vulnerability' in team_data['opponent_quality'] and 'upset_vulnerability' in opponent_data['opponent_quality']):
                
                sample['upset_vulnerability_diff'] = team_data['opponent_quality']['upset_vulnerability'] - opponent_data['opponent_quality']['upset_vulnerability']
                sample['less_vulnerable_team1'] = 1 if team_data['opponent_quality']['upset_vulnerability'] < opponent_data['opponent_quality']['upset_vulnerability'] else 0
                sample['avg_upset_vulnerability'] = (team_data['opponent_quality']['upset_vulnerability'] + opponent_data['opponent_quality']['upset_vulnerability']) / 2
            
            # Add team ranking metrics
            if ('ranking' in team_data and 'ranking' in opponent_data):
                sample['team_rank_diff'] = team_data['ranking'] - opponent_data['ranking']
                sample['better_ranked_team1'] = 1 if team_data['ranking'] < opponent_data['ranking'] else 0
                sample['avg_team_rank'] = (team_data['ranking'] + opponent_data['ranking']) / 2
                sample['rank_gap'] = abs(team_data['ranking'] - opponent_data['ranking'])
            
            # Add team rating metrics
            if ('rating' in team_data and 'rating' in opponent_data):
                sample['team_rating_quality_diff'] = team_data['rating'] - opponent_data['rating']
                sample['better_rated_quality_team1'] = 1 if team_data['rating'] > opponent_data['rating'] else 0
                sample['avg_team_rating_quality'] = (team_data['rating'] + opponent_data['rating']) / 2
                sample['rating_gap'] = abs(team_data['rating'] - opponent_data['rating'])
            
            # Add tournament performance metrics
            if ('tournament_performance' in team_data and 'tournament_performance' in opponent_data):
                # Mid-tier tournaments
                if ('mid_tier_win_rate' in team_data['tournament_performance'] and 
                    'mid_tier_win_rate' in opponent_data['tournament_performance']):
                    
                    t1_mid = team_data['tournament_performance']['mid_tier_win_rate']
                    t2_mid = opponent_data['tournament_performance']['mid_tier_win_rate']
                    
                    # Convert from dict to float if needed
                    if isinstance(t1_mid, dict) and 'win_rate' in t1_mid:
                        t1_mid = t1_mid['win_rate']
                    if isinstance(t2_mid, dict) and 'win_rate' in t2_mid:
                        t2_mid = t2_mid['win_rate']
                    
                    # Only use if both are numeric
                    if isinstance(t1_mid, (int, float)) and isinstance(t2_mid, (int, float)):
                        sample['mid_tier_tourney_diff'] = t1_mid - t2_mid
                        sample['better_mid_tier_team1'] = 1 if t1_mid > t2_mid else 0
                        sample['avg_mid_tier_win_rate'] = (t1_mid + t2_mid) / 2
                
                # Overall tournament performance
                if ('overall_win_rate' in team_data['tournament_performance'] and 
                    'overall_win_rate' in opponent_data['tournament_performance']):
                    
                    t1_overall = team_data['tournament_performance']['overall_win_rate']
                    t2_overall = opponent_data['tournament_performance']['overall_win_rate']
                    
                    # Convert from dict to float if needed
                    if isinstance(t1_overall, dict) and 'win_rate' in t1_overall:
                        t1_overall = t1_overall['win_rate']
                    if isinstance(t2_overall, dict) and 'win_rate' in t2_overall:
                        t2_overall = t2_overall['win_rate']
                    
                    # Only use if both are numeric
                    if isinstance(t1_overall, (int, float)) and isinstance(t2_overall, (int, float)):
                        sample['overall_tourney_diff'] = t1_overall - t2_overall
                        sample['better_tourney_team1'] = 1 if t1_overall > t2_overall else 0
                        sample['avg_tourney_win_rate'] = (t1_overall + t2_overall) / 2
            
            # Add head-to-head metrics
            # (We're not accessing nested dictionaries here to avoid errors)
            h2h_win_rate = 0.5  # Default to 50% if no H2H data
            h2h_matches = 0
            h2h_score_diff = 0
            h2h_advantage = 0
            h2h_significant = 0
            
            # Get H2H data from team_data
            if ('head_to_head' in team_data and opponent_name in team_data['head_to_head']):
                h2h_data = team_data['head_to_head'][opponent_name]
                if isinstance(h2h_data, dict):
                    if 'win_rate' in h2h_data and isinstance(h2h_data['win_rate'], (int, float)):
                        h2h_win_rate = h2h_data['win_rate']
                    if 'matches' in h2h_data and isinstance(h2h_data['matches'], (int, float)):
                        h2h_matches = h2h_data['matches']
                    if 'score_diff' in h2h_data and isinstance(h2h_data['score_diff'], (int, float)):
                        h2h_score_diff = h2h_data['score_diff']
            
            # Compute simple H2H metrics
            h2h_advantage = 1 if h2h_win_rate > 0.5 else 0
            h2h_significant = 1 if h2h_matches >= 3 else 0
            h2h_confidence = min(1.0, h2h_matches / 10.0)  # Scale up to 10 matches
            
            # Add H2H metrics to sample
            sample['h2h_win_rate'] = h2h_win_rate
            sample['h2h_matches'] = h2h_matches
            sample['h2h_score_diff'] = h2h_score_diff
            sample['h2h_advantage_team1'] = h2h_advantage
            sample['h2h_significant'] = h2h_significant
            sample['h2h_confidence'] = h2h_confidence
            
            # Feature interactions (to capture non-linear relationships)
            # Only include if both components are valid
            if 'avg_player_rating' in sample and 'win_rate_diff' in sample:
                sample['rating_x_win_rate'] = sample['avg_player_rating'] * sample['win_rate_diff']
            
            if 'avg_pistol_win_rate' in sample and 'avg_eco_win_rate' in sample:
                sample['pistol_x_eco'] = sample['avg_pistol_win_rate'] * sample['avg_eco_win_rate']
            
            if 'avg_pistol_win_rate' in sample and 'avg_full_buy_win_rate' in sample:
                sample['pistol_x_full_buy'] = sample['avg_pistol_win_rate'] * sample['avg_full_buy_win_rate']
            
            if 'star_player_diff' in sample and 'consistency_diff' in sample:
                sample['star_x_consistency'] = sample['star_player_diff'] * sample['consistency_diff']
            
            if 'h2h_win_rate' in sample and 'recent_form_diff' in sample:
                sample['h2h_x_form'] = sample['h2h_win_rate'] * sample['recent_form_diff']
            
            if 'headshot_diff' in sample and 'kd_diff' in sample:
                sample['headshot_x_kd'] = sample['headshot_diff'] * sample['kd_diff']
            
            if 'win_rate_diff' in sample and 'opponent_rank_diff' in sample:
                sample['win_rate_x_opp_quality'] = sample['win_rate_diff'] * (1 / (sample['opponent_rank_diff'] + 1))
            
            if 'first_bloods_diff' in sample and 'win_rate_diff' in sample:
                sample['first_blood_x_win_rate'] = sample['first_bloods_diff'] * sample['win_rate_diff']
            
            if 'clutches_diff' in sample and 'consistency_diff' in sample:
                sample['clutch_x_consistency'] = sample['clutches_diff'] * sample['consistency_diff']
            
            # Count how many metrics favor team1
            team1_better_count = sum(1 for key, value in sample.items() if key.startswith('better_') and value == 1)
            total_better_metrics = sum(1 for key in sample.keys() if key.startswith('better_'))
            sample['team1_better_count'] = team1_better_count
            sample['team1_better_ratio'] = team1_better_count / max(total_better_metrics, 1)
            
            # Add "significant difference" features
            for key in list(sample.keys()):
                if key.endswith('_diff') and not key.endswith('_significant_diff'):
                    # Get the threshold for this metric (can be customized per metric)
                    threshold = 0.1  # Default threshold
                    
                    # Special thresholds for certain metrics
                    if 'win_rate' in key:
                        threshold = 0.1  # 10% difference in win rate is significant
                    elif 'form' in key:
                        threshold = 0.15  # 15% difference in form is significant
                    elif 'player_rating' in key:
                        threshold = 0.2  # 0.2 difference in player rating is significant
                    elif 'score' in key:
                        threshold = 3  # 3 rounds difference is significant
                    
                    # Is the difference significant?
                    value = sample[key]
                    significant = 1 if abs(value) >= threshold else 0
                    
                    # Who has the advantage? (1 for team1, 0 for team2)
                    advantage = 1 if value > 0 else 0
                    
                    # Add new features
                    significant_key = key.replace('_diff', '_significant_diff')
                    advantage_key = key.replace('_diff', '_significant_advantage_team1')
                    
                    sample[significant_key] = significant
                    sample[advantage_key] = advantage if significant else 0
            
            # At this point, we might have nested dictionaries in the sample
            # Let's remove them to avoid issues with the model
            clean_sample = {}
            for key, value in sample.items():
                # Skip dictionaries, lists, or any other non-scalar values
                if isinstance(value, (int, float, bool, str)):
                    clean_sample[key] = value
                elif value is None:
                    clean_sample[key] = 0  # Convert None to 0
            
            # Add to training data
            X.append(clean_sample)
            y.append(1 if match['result'] == 'win' else 0)  # 1 for win, 0 for loss
            
            # Count samples with different data types
            has_both = has_player_stats and has_economy_stats
            
            if has_player_stats:
                player_stats_sample_count += 1
            
            if has_economy_stats:
                economy_sample_count += 1
            
            if has_both:
                both_sample_count += 1
    
    print(f"Created {len(X)} training samples with:")
    print(f"  - Samples with economy data: {economy_sample_count}")
    print(f"  - Samples with player stats: {player_stats_sample_count}")
    print(f"  - Samples with both: {both_sample_count}")
    print(f"  - Teams with economy data: {len(teams_with_economy)}/{len(team_data_collection)}")
    print(f"  - Teams with player stats: {len(teams_with_player_stats)}/{len(team_data_collection)}")
    
    print("\nFeature breakdown:")
    print(f"  - Economy features: {len(economy_features)}")
    print(f"  - Player features: {len(player_features)}")
    print(f"  - Map features: {len(map_features)}")
    print(f"  - Total features: {len(set().union(economy_features, player_features, map_features))}")
    
    print("\nSample economy features:")
    for feature in list(economy_features)[:5]:
        print(f"  - {feature}")
    
    print("\nSample player features:")
    for feature in list(player_features)[:5]:
        print(f"  - {feature}")

    successful_pairings += 1    
    
    return X, y

def debug_and_fix_match_data(team_data_collection):
    """Analyze and fix the match data structure for training."""
    total_teams = len(team_data_collection)
    teams_with_matches = 0
    total_matches = 0
    fixed_matches = 0
    
    print("\n=== MATCH DATA STRUCTURE ANALYSIS ===")
    
    for team_name, team_data in team_data_collection.items():
        if 'matches' not in team_data or not team_data['matches']:
            continue
            
        teams_with_matches += 1
        team_matches = team_data['matches']
        total_matches += len(team_matches)
        
        # For the first team, print detailed match structure
        if teams_with_matches == 1:
            print(f"\nSample match structure for {team_name}:")
            if team_matches:
                sample_match = team_matches[0]
                print(f"Keys in match: {list(sample_match.keys())}")
                
                # Check specific fields
                print(f"  Has 'opponent_name'? {'opponent_name' in sample_match}")
                print(f"  Has 'team_won'? {'team_won' in sample_match}")
                print(f"  Has 'result'? {'result' in sample_match}")
                
                # Print values for key fields
                if 'opponent_name' in sample_match:
                    print(f"  opponent_name: {sample_match['opponent_name']}")
                if 'team_won' in sample_match:
                    print(f"  team_won: {sample_match['team_won']}")
        
        # Fix all matches for this team
        for match in team_matches:
            # Make sure opponent_name exists
            if 'opponent_name' not in match and 'opponent' in match and isinstance(match['opponent'], dict):
                match['opponent_name'] = match['opponent'].get('name', 'Unknown')
            
            # Make sure result exists based on team_won
            if 'result' not in match and 'team_won' in match:
                match['result'] = 'win' if match['team_won'] else 'loss'
                fixed_matches += 1
            
            # If neither exists, try to determine from scores
            if 'result' not in match and 'team_won' not in match:
                if 'team_score' in match and 'opponent_score' in match:
                    match['team_won'] = match['team_score'] > match['opponent_score']
                    match['result'] = 'win' if match['team_won'] else 'loss'
                    fixed_matches += 1
    
    print(f"\n=== MATCH DATA ANALYSIS SUMMARY ===")
    print(f"Total teams: {total_teams}")
    print(f"Teams with matches: {teams_with_matches}")
    print(f"Total matches: {total_matches}")
    print(f"Fixed matches: {fixed_matches}")
    
    return team_data_collection

def analyze_selected_features(selected_features):
    """
    Analyze the types of features selected by the model.
    
    Args:
        selected_features (list): List of selected feature names
        
    Returns:
        dict: Categorized features
    """
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    
    # Categorize features
    economy_features = [f for f in selected_features if any(term in str(f).lower() for term in 
                                                ['eco', 'pistol', 'buy', 'economy'])]
    
    player_features = [f for f in selected_features if any(term in str(f).lower() for term in 
                                               ['rating', 'acs', 'kd', 'adr', 'headshot',
                                                'clutch', 'aces', 'first_blood', 'player'])]
    
    map_features = [f for f in selected_features if 'map_' in str(f).lower()]
    
    consistency_features = [f for f in selected_features if 'consistency' in str(f).lower()]
    
    differential_features = [f for f in selected_features if 'diff' in str(f).lower()]
    
    form_features = [f for f in selected_features if 'form' in str(f).lower() or 
                                               'recent' in str(f).lower()]
    
    win_rate_features = [f for f in selected_features if 'win_rate' in str(f).lower()]
    
    # Calculate category counts
    categories = {
        'Economy': len(economy_features),
        'Player Statistics': len(player_features),
        'Map-specific': len(map_features),
        'Consistency': len(consistency_features),
        'Differential': len(differential_features),
        'Form/Recent Performance': len(form_features),
        'Win Rate': len(win_rate_features),
        'Other': len(selected_features) - len(economy_features) - len(player_features) - 
                len(map_features) - len(consistency_features) - len(differential_features) - 
                len(form_features) - len(win_rate_features)
    }
    
    # Print categorization
    print("\nFeature Category Breakdown:")
    for category, count in categories.items():
        percentage = count / len(selected_features) * 100
        print(f"  {category}: {count} features ({percentage:.1f}%)")
    
    # Print sample features from each non-empty category
    print("\nSample Features by Category:")
    if economy_features:
        print("\nEconomy Features:")
        for f in economy_features[:5]:
            print(f"  - {f}")
        if len(economy_features) > 5:
            print(f"  ... and {len(economy_features) - 5} more")
    
    if player_features:
        print("\nPlayer Statistics Features:")
        for f in player_features[:5]:
            print(f"  - {f}")
        if len(player_features) > 5:
            print(f"  ... and {len(player_features) - 5} more")
    
    if map_features:
        print("\nMap-specific Features:")
        for f in map_features[:5]:
            print(f"  - {f}")
        if len(map_features) > 5:
            print(f"  ... and {len(map_features) - 5} more")
    
    result = {
        'economy_features': economy_features,
        'player_features': player_features,
        'map_features': map_features,
        'consistency_features': consistency_features,
        'differential_features': differential_features,
        'form_features': form_features,
        'win_rate_features': win_rate_features,
        'all_features': selected_features,
        'counts': categories
    }
    
    return result

def check_prediction_consistency(prediction, team1_stats, team2_stats):
    """Verify that model predictions align with basic statistical advantages."""
    team1_advantages = 0
    team2_advantages = 0
    
    key_metrics = [
        ('win_rate', 'Win rate'),
        ('recent_form', 'Recent form'),
        ('avg_player_rating', 'Player rating'),
        ('score_differential', 'Score differential')
    ]
    
    print("\nStatistical advantages check:")
    for key, label in key_metrics:
        t1_value = team1_stats.get(key, 0)
        t2_value = team2_stats.get(key, 0)
        advantage = "Team 1" if t1_value > t2_value else "Team 2" if t2_value > t1_value else "Even"
        
        if advantage == "Team 1":
            team1_advantages += 1
        elif advantage == "Team 2":
            team2_advantages += 1
            
        print(f"  {label}: {advantage} ({t1_value:.2f} vs {t2_value:.2f})")
    
    # Check head-to-head if available
    h2h_advantage = "Unknown"
    if 'opponent_stats' in team1_stats and team2_stats.get('name') in team1_stats['opponent_stats']:
        h2h_win_rate = team1_stats['opponent_stats'][team2_stats.get('name')].get('win_rate', 0.5)
        
        if h2h_win_rate > 0.55:
            h2h_advantage = "Team 1"
            team1_advantages += 1
        elif h2h_win_rate < 0.45:
            h2h_advantage = "Team 2"
            team2_advantages += 1
        else:
            h2h_advantage = "Even"
            
        print(f"  Head-to-head: {h2h_advantage} ({h2h_win_rate:.2f} win rate for Team 1)")
    
    predicted_winner = "Team 1" if prediction > 0.5 else "Team 2"
    statistical_favorite = "Team 1" if team1_advantages > team2_advantages else "Team 2" if team2_advantages > team1_advantages else "Even"
    
    print(f"\nStatistical favorite: {statistical_favorite} ({team1_advantages}-{team2_advantages})")
    print(f"Model prediction: {predicted_winner} ({prediction:.2f})")
    
    if statistical_favorite != "Even" and statistical_favorite != predicted_winner:
        print("WARNING: Model prediction contradicts statistical advantages!")
        
    return {
        'team1_advantages': team1_advantages,
        'team2_advantages': team2_advantages,
        'statistical_favorite': statistical_favorite,
        'prediction_consistent': statistical_favorite == "Even" or statistical_favorite == predicted_winner
    }

def create_learning_rate_scheduler():
    """Create a cosine annealing learning rate scheduler for better convergence."""
    def lr_scheduler(epoch, lr):
        # Warm-up phase
        if epoch < 5:
            return 0.001 * (1 + epoch/5)
        # Cosine annealing
        else:
            return 0.001 * (1 + np.cos((epoch-5) * np.pi / 45))
    
    return tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

def build_training_dataset_from_matches(matches, team_data_collection):
    """Build a dataset for model training from a specific set of matches."""
    X = []  # Feature vectors
    y = []  # Labels (1 if team1 won, 0 if team2 won)
    
    print(f"Building training dataset from {len(matches)} matches...")
    
    # Process each match
    for match in matches:
        team1_name = match.get('team_name')
        team2_name = match.get('opponent_name')
        
        # Skip if we don't have data for either team
        if team1_name not in team_data_collection or team2_name not in team_data_collection:
            continue
        
        # Get stats for both teams
        team1_stats = team_data_collection[team1_name]
        team2_stats = team_data_collection[team2_name]
        
        # Prepare feature vector
        features = prepare_data_for_model(team1_stats, team2_stats)
        
        if features:
            # Ensure all values are numeric
            is_valid = True
            for key, value in features.items():
                if not isinstance(value, (int, float)):
                    is_valid = False
                    break
            
            if is_valid:
                X.append(features)
                y.append(1 if match.get('team_won', False) else 0)
    
    print(f"Created {len(X)} training samples from match data")
    return X, y

def build_training_dataset(team_data_collection):
    """Build a dataset for model training from historical match data."""
    X = []  # Feature vectors
    y = []  # Labels (1 if team1 won, 0 if team2 won)
    
    print(f"Building training dataset from {len(team_data_collection)} teams...")
    
    # For each team, loop through their matches
    for team_name, team_data in team_data_collection.items():
        matches = team_data.get('matches', [])
        
        print(f"Processing {len(matches)} matches for {team_name}")
        
        for match in matches:
            # Get opponent name
            opponent_name = match.get('opponent_name')
            
            # Skip if we don't have data for the opponent
            if opponent_name not in team_data_collection:
                print(f"Skipping match against {opponent_name} - no data available")
                continue
            
            # Get stats for both teams at the time of the match
            # Note: For accurate historical analysis, we'd need stats from BEFORE the match
            # This is simplified and uses current stats
            team1_stats = team_data
            team2_stats = team_data_collection[opponent_name]
            
            # Add weighted pistol win rate feature that accounts for sample size
            t1_pistol_confidence = team1_stats.get('pistol_confidence', 0.5)
            t2_pistol_confidence = team2_stats.get('pistol_confidence', 0.5)
            
            # Raw pistol win rates
            t1_pistol_raw = team1_stats.get('pistol_win_rate', 0.5)
            t2_pistol_raw = team2_stats.get('pistol_win_rate', 0.5)
            
            # Weighted pistol win rates (more games = more reliable)
            features['team1_pistol_win_rate_weighted'] = t1_pistol_raw * t1_pistol_confidence
            features['team2_pistol_win_rate_weighted'] = t2_pistol_raw * t2_pistol_confidence
            
            # The rest of the economy features remain the same
            features['team1_pistol_win_rate'] = float(team1_stats.get('pistol_win_rate', 0.5))
            features['team2_pistol_win_rate'] = float(team2_stats.get('pistol_win_rate', 0.5))
            
            # Prepare feature vector
            features = prepare_data_for_model(team1_stats, team2_stats)
            
            if features:
                # Ensure all values are numeric before adding to X
                is_valid = True
                for key, value in features.items():
                    if not isinstance(value, (int, float)):
                        print(f"Invalid feature detected in match {team_name} vs {opponent_name}: {key} = {value}")
                        is_valid = False
                        break
                
                if is_valid:
                    X.append(features)
                    y.append(1 if match.get('team_won', False) else 0)
    
    print(f"Created {len(X)} training samples from match data")
    return X, y

def fetch_match_economy_details(match_id):
    """Fetch economic details for a specific match."""
    if not match_id:
        return None
    
    print(f"Fetching economy details for match ID: {match_id}")
    response = requests.get(f"{API_URL}/match-details/{match_id}?tab=economy")
    
    if response.status_code != 200:
        print(f"Error fetching economy details for match {match_id}: {response.status_code}")
        return None
    
    economy_details = response.json()

    
    # Be nice to the API
     
    
    return economy_details

def extract_economy_metrics(match_economy_data, team_identifier=None, fallback_name=None):
    """Extract relevant economic performance metrics from match economy data for a specific team.
    
    Args:
        match_economy_data (dict): The economy data from the match
        team_identifier (str): The team tag to identify the team
        fallback_name (str): The team name to use as fallback if tag matching fails
        
    Returns:
        dict: Economy metrics for the team, or empty dict if not found
    """
    if not match_economy_data or 'data' not in match_economy_data or 'teams' not in match_economy_data['data']:
        print("\nNo valid economy data found in match_economy_data")
        return {}
    
    teams_data = match_economy_data['data']['teams']
    if len(teams_data) < 2:
        print("\nNot enough teams found in economy data")
        return {}
    
    print(f"\nLooking for team with identifier: {team_identifier}")
    
    # Find the team with matching tag or name
    target_team_data = None
    
    # First try to match by tag if available
    if team_identifier:
        for team in teams_data:
            # Try to match by tag (lowercase for case-insensitive comparison)
            team_tag = team.get('tag', '').lower()
            if team_tag and team_tag == team_identifier.lower():
                target_team_data = team
                print(f"Found team by tag: {team.get('tag', '')}")
                break
    
    # If no match by tag and fallback_name is provided, try matching by name
    if not target_team_data and fallback_name:
        print(f"No match found by tag, trying fallback name: {fallback_name}")
        for team in teams_data:
            # Try to match by name (lowercase for case-insensitive comparison)
            team_name = team.get('name', '').lower()
            if team_name and (team_name == fallback_name.lower() or 
                              fallback_name.lower() in team_name or 
                              team_name in fallback_name.lower()):
                target_team_data = team
                print(f"Found team by name: {team.get('name', '')}")
                break
    
    if not target_team_data:
        print(f"\nNo team found with identifier: {team_identifier} or fallback name: {fallback_name}")
        return {}
    
    print(f"\nFound team data for {team_identifier or fallback_name}")
    
    # Extract metrics for the target team only
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
    
    print(f"\nExtracted metrics for {team_identifier or fallback_name}:")
    print(json.dumps(metrics, indent=2))
    
    return metrics

def fetch_player_stats(player_name):
    """
    Fetch detailed player statistics using the new endpoint.
    
    Args:
        player_name (str): The name of the player to fetch stats for
    
    Returns:
        dict: Player statistics if found, None otherwise
    """
    if not player_name:
        return None
    
    print(f"Fetching stats for player: {player_name}")
    response = requests.get(f"{API_URL}/player-stats/{player_name}")
    
    if response.status_code != 200:
        print(f"Error fetching stats for player {player_name}: {response.status_code}")
        return None
    
    player_data = response.json()
    
    # Be nice to the API
     
    
    # Return player stats if successful
    if player_data.get('status') == 'OK' and 'data' in player_data:
        return player_data['data']

    player_data = response.json()
    print(f"Player data for {player_name}: {json.dumps(player_data, indent=2)}")
    
    return None    

# Update function to create a better deep learning model with player stats
def create_deep_learning_model_with_economy_and_regularization(input_dim, regularization_strength=0.001, dropout_rate=0.4):
    """Create an enhanced deep learning model with adjustable regularization parameters."""
    # Define inputs
    inputs = Input(shape=(input_dim,))
    
    # First layer - shared feature processing
    x = Dense(256, activation='relu', 
              kernel_regularizer=l2(regularization_strength),
              kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second layer - deeper processing
    x = Dense(128, activation='relu', 
              kernel_regularizer=l2(regularization_strength/2),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.75)(x)
    
    # Player stats pathway with additional neurons
    x = Dense(96, activation='relu', 
              kernel_regularizer=l2(regularization_strength/4),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.6)(x)
    
    # Economy-specific pathway with expanded capacity
    x = Dense(64, activation='relu', 
              kernel_regularizer=l2(regularization_strength/8),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.5)(x)
    
    # Combined pathway
    x = Dense(32, activation='relu', 
              kernel_regularizer=l2(regularization_strength/16),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.4)(x)
    
    # Output
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(loss='binary_crossentropy', 
                 optimizer=Adam(learning_rate=0.0005),
                 metrics=['accuracy'])
    
    # Print model summary to see the expanded architecture
    print("\nModel Architecture with Regularization:")
    model.summary()
    
    return model

# 2. Add an advanced learning rate scheduler
def create_deep_learning_model_with_advanced_lr(input_dim, regularization_strength=0.001, dropout_rate=0.4):
    """Create a deep learning model with advanced learning rate scheduling."""
    # Define inputs
    inputs = Input(shape=(input_dim,))
    
    # First layer - shared feature processing
    x = Dense(256, activation='relu', 
              kernel_regularizer=l2(regularization_strength),
              kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second layer - deeper processing
    x = Dense(128, activation='relu', 
              kernel_regularizer=l2(regularization_strength/2),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.75)(x)
    
    # Player stats pathway with additional neurons
    x = Dense(96, activation='relu', 
              kernel_regularizer=l2(regularization_strength/4),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.6)(x)
    
    # Economy-specific pathway with expanded capacity
    x = Dense(64, activation='relu', 
              kernel_regularizer=l2(regularization_strength/8),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.5)(x)
    
    # Combined pathway
    x = Dense(32, activation='relu', 
              kernel_regularizer=l2(regularization_strength/16),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.4)(x)
    
    # Output
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    # Using a very small initial learning rate that will be controlled by the scheduler
    model.compile(loss='binary_crossentropy', 
                 optimizer=Adam(learning_rate=0.0005),
                 metrics=['accuracy'])
    
    # Print model summary
    print("\nModel Architecture with Advanced LR:")
    model.summary()
    
    return model


def find_optimal_regularization(X, y, test_size=0.2, random_state=42):
    """Find optimal regularization parameters to combat overfitting."""
    print("\n" + "="*60)
    print("FINDING OPTIMAL REGULARIZATION PARAMETERS")
    print("="*60)
    
    # Setup data same as in train_model_with_learning_curves
    df = pd.DataFrame(X).fillna(0)
    
    # Handle non-numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(float)
            except (ValueError, TypeError):
                df = df.drop(columns=[col])
    
    if df.empty:
        print("Error: Empty feature dataframe after cleaning")
        return None
    
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
    if np.min(class_counts) / np.sum(class_counts) < 0.4:
        try:
            if np.min(class_counts) >= 5:
                min_samples = np.min(class_counts)
                k_neighbors = min(5, min_samples-1)
                smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                X_train, y_train = smote.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"Error applying SMOTE: {e}")
    
    # Define parameter grid
    regularization_strengths = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    dropout_rates = [0.2, 0.3, 0.4, 0.5, 0.6]
    
    # Store results
    results = []
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
    )
    
    # Try different combinations
    input_dim = X_train.shape[1]
    total_combos = len(regularization_strengths) * len(dropout_rates)
    combo_count = 0
    
    for reg_strength in regularization_strengths:
        for dropout_rate in dropout_rates:
            combo_count += 1
            print(f"\nTesting combination {combo_count}/{total_combos}: L2={reg_strength}, Dropout={dropout_rate}")
            
            # Create and train model
            model = create_deep_learning_model_with_economy_and_regularization(
                input_dim, regularization_strength=reg_strength, dropout_rate=dropout_rate
            )
            
            history = model.fit(
                X_train, y_train,
                epochs=30,  # Reduced epochs for faster testing
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate results
            train_acc = history.history['accuracy'][-1]
            val_acc = history.history['val_accuracy'][-1]
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            
            acc_gap = train_acc - val_acc
            epochs_trained = len(history.history['accuracy'])
            
            results.append({
                'reg_strength': reg_strength,
                'dropout_rate': dropout_rate,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'acc_gap': acc_gap,
                'epochs_trained': epochs_trained
            })
            
            print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Gap: {acc_gap:.4f}, Epochs: {epochs_trained}")
    
    # Find best parameters
    # Sort by validation accuracy, with a penalty for large gaps
    df_results = pd.DataFrame(results)
    df_results['score'] = df_results['val_acc'] - 0.5 * df_results['acc_gap']
    df_results = df_results.sort_values('score', ascending=False)
    
    best_params = df_results.iloc[0]
    print("\nBest regularization parameters found:")
    print(f"L2 Regularization Strength: {best_params['reg_strength']}")
    print(f"Dropout Rate: {best_params['dropout_rate']}")
    print(f"Validation Accuracy: {best_params['val_acc']:.4f}")
    print(f"Train-Val Accuracy Gap: {best_params['acc_gap']:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot the effect of regularization strength
    plt.subplot(2, 2, 1)
    reg_effect = df_results.groupby('reg_strength')[['val_acc', 'acc_gap']].mean()
    reg_effect.plot(kind='line', marker='o', ax=plt.gca())
    plt.title('Effect of L2 Regularization', fontsize=14)
    plt.xlabel('L2 Strength', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot the effect of dropout rate
    plt.subplot(2, 2, 2)
    dropout_effect = df_results.groupby('dropout_rate')[['val_acc', 'acc_gap']].mean()
    dropout_effect.plot(kind='line', marker='o', ax=plt.gca())
    plt.title('Effect of Dropout Rate', fontsize=14)
    plt.xlabel('Dropout Rate', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot validation accuracy heatmap
    plt.subplot(2, 2, 3)
    pivot_val_acc = pd.pivot_table(
        df_results, values='val_acc', index='dropout_rate', columns='reg_strength'
    )
    sns.heatmap(pivot_val_acc, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Validation Accuracy by Regularization Parameters', fontsize=14)
    plt.xlabel('L2 Strength', fontsize=12)
    plt.ylabel('Dropout Rate', fontsize=12)
    
    # Plot accuracy gap heatmap
    plt.subplot(2, 2, 4)
    pivot_acc_gap = pd.pivot_table(
        df_results, values='acc_gap', index='dropout_rate', columns='reg_strength'
    )
    sns.heatmap(pivot_acc_gap, annot=True, cmap='coolwarm_r', fmt='.3f')
    plt.title('Accuracy Gap by Regularization Parameters', fontsize=14)
    plt.xlabel('L2 Strength', fontsize=12)
    plt.ylabel('Dropout Rate', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('regularization_parameter_search.png', dpi=300)
    plt.show()
    
    # Return best parameters
    return best_params['reg_strength'], best_params['dropout_rate']

def train_model_with_overfitting_detection(X, y, test_size=0.2, random_state=42):
    """Complete pipeline to train model with overfitting detection and mitigation."""
    # Step 1: Train with learning curves to diagnose overfitting
    print("\nStep 1: Training model with learning curves to diagnose overfitting...")
    _, _, _, learning_curve_data = train_model_with_learning_curves(X, y, test_size, random_state)
    
    # Check if severe overfitting was detected
    if learning_curve_data['diagnosis'] in ["SEVERE OVERFITTING", "MODERATE OVERFITTING"]:
        print(f"\nOverfitting detected: {learning_curve_data['diagnosis']}")
        
        # Step 2: Find optimal regularization parameters
        print("\nStep 2: Finding optimal regularization parameters...")
        best_reg_strength, best_dropout_rate = find_optimal_regularization(X, y, test_size, random_state)
        
        # Step 3: Retrain with optimal regularization
        print("\nStep 3: Retraining model with optimal regularization...")
        
        # Convert and prepare data
        df = pd.DataFrame(X).fillna(0)
        
        # Handle non-numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype(float)
                except (ValueError, TypeError):
                    df = df.drop(columns=[col])
        
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
        if np.min(class_counts) / np.sum(class_counts) < 0.4:
            try:
                if np.min(class_counts) >= 5:
                    min_samples = np.min(class_counts)
                    k_neighbors = min(5, min_samples-1)
                    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"Error applying SMOTE: {e}")
        
        # Train final model with optimal regularization
        input_dim = X_train.shape[1]
        model = create_deep_learning_model_with_economy_and_regularization(
            input_dim, regularization_strength=best_reg_strength, dropout_rate=best_dropout_rate
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_valorant_model_final.h5', 
            save_best_only=True, 
            monitor='val_accuracy'
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )
        
        # Save model and artifacts
        model.save('valorant_model_regularized.h5')
        
        with open('feature_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(list(df.columns), f)
        
        # Evaluate final model
        y_pred_proba = model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        print("\n" + "="*60)
        print("FINAL MODEL EVALUATION (WITH REGULARIZATION)")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        # Compare with initial model
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        final_acc_gap = train_acc - val_acc
        
        print(f"\nFinal Train-Val Accuracy Gap: {final_acc_gap:.4f}")
        print(f"Initial Accuracy Gap: {learning_curve_data['accuracy_gap'][-1]:.4f}")
        
        reduction = 100 * (1 - final_acc_gap / learning_curve_data['accuracy_gap'][-1])
        print(f"Overfitting Reduction: {reduction:.2f}%")
        
        # Plot final learning curves comparison
        plt.figure(figsize=(15, 10))
        
        # Compare original vs regularized validation accuracy
        plt.subplot(2, 2, 1)
        plt.plot(learning_curve_data['epochs'], learning_curve_data['validation_accuracy'], 
                 label='Original Model', marker='o')
        plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], 
                 label='Regularized Model', marker='o')
        plt.title('Validation Accuracy Comparison', fontsize=14)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        
        # Compare original vs regularized accuracy gaps
        plt.subplot(2, 2, 2)
        original_gaps = learning_curve_data['accuracy_gap']
        regularized_gaps = [train_acc - val_acc for train_acc, val_acc in 
                           zip(history.history['accuracy'], history.history['val_accuracy'])]
        
        plt.plot(learning_curve_data['epochs'], original_gaps, 
                 label='Original Model', marker='o')
        plt.plot(range(1, len(regularized_gaps) + 1), regularized_gaps, 
                 label='Regularized Model', marker='o')
        plt.axhline(y=0.05, color='green', linestyle='--', label='Acceptable Gap Threshold')
        plt.title('Overfitting Comparison', fontsize=14)
        plt.ylabel('Train-Val Accuracy Gap', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        
        # Plot validation loss comparison
        plt.subplot(2, 2, 3)
        plt.plot(learning_curve_data['epochs'], learning_curve_data['validation_loss'], 
                 label='Original Model', marker='o')
        plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], 
                 label='Regularized Model', marker='o')
        plt.title('Validation Loss Comparison', fontsize=14)
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        
        # Plot final metrics comparison
        plt.subplot(2, 2, 4)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        original_metrics = [
            accuracy_score(y_val, (model.predict(X_val) > 0.5).astype(int).flatten()),
            precision_score(y_val, (model.predict(X_val) > 0.5).astype(int).flatten()),
            recall_score(y_val, (model.predict(X_val) > 0.5).astype(int).flatten()),
            f1_score(y_val, (model.predict(X_val) > 0.5).astype(int).flatten()),
            roc_auc_score(y_val, model.predict(X_val))
        ]
        
        regularized_metrics = [accuracy, precision, recall, f1, auc]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, original_metrics, width, label='Original Model')
        plt.bar(x + width/2, regularized_metrics, width, label='Regularized Model')
        
        plt.title('Performance Metrics Comparison', fontsize=14)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(x, metrics)
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig('regularization_comparison.png', dpi=300)
        plt.show()
        
        return model, scaler, list(df.columns)
    else:
        print(f"\nNo severe overfitting detected. Diagnosis: {learning_curve_data['diagnosis']}")
        print("Using original model without additional regularization.")
        
        # Load the original model
        model = load_model('valorant_model.h5')
        
        with open('feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        return model, scaler, feature_names

def train_with_fixed_feature_count(X, y, feature_count=71, test_size=0.2, random_state=42):
    """
    Train model using a fixed number of top features.
    
    Args:
        X (list/DataFrame): Feature data
        y (list/array): Target labels
        feature_count (int): Number of top features to use (default: 71)
        test_size (float): Validation split ratio
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (model, scaler, selected_features)
    """
    print(f"\n{'='*60}")
    print(f"TRAINING WITH FIXED TOP {feature_count} FEATURES")
    print(f"{'='*60}")
    
    # Convert data to DataFrame
    if isinstance(X, list):
        df = pd.DataFrame(X)
    else:
        df = X.copy()
    
    # Fill missing values and handle non-numeric columns
    df = df.fillna(0)
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(float)
            except (ValueError, TypeError):
                print(f"Dropping column {col} due to non-numeric values")
                df = df.drop(columns=[col])
    
    # Convert to numpy array for scaling
    X_arr = df.values
    y_arr = np.array(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )
    
    # Train a Random Forest to determine feature importance
    print("Training Random Forest for feature selection...")
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print top features
    print(f"\nTop {min(10, feature_count)} most important features:")
    feature_names = list(df.columns)
    for i in range(min(10, feature_count)):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Select top features
    top_indices = indices[:feature_count]
    top_features = [feature_names[i] for i in top_indices]
    
    print(f"\nSelected {feature_count} top features")
    
    # Extract selected features
    X_train_selected = X_train[:, top_indices]
    X_val_selected = X_val[:, top_indices]
    
    # Create model
    print("\nTraining neural network model with selected features...")
    input_dim = X_train_selected.shape[1]
    model = create_deep_learning_model_with_economy(input_dim)
    
    # Train model
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        f'valorant_model_top{feature_count}.h5', 
        save_best_only=True, 
        monitor='val_accuracy'
    )
    
    history = model.fit(
        X_train_selected, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_selected, y_val),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Evaluate model
    val_metrics = model.evaluate(X_val_selected, y_val)
    y_pred = (model.predict(X_val_selected) > 0.5).astype(int).flatten()
    
    print("\n" + "="*60)
    print(f"MODEL EVALUATION (TOP {feature_count} FEATURES)")
    print("="*60)
    print(f"Validation Loss: {val_metrics[0]:.4f}")
    print(f"Validation Accuracy: {val_metrics[1]:.4f}")
    
    # Calculate additional metrics
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, model.predict(X_val_selected))
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Calculate overfitting metrics
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    acc_gap = train_acc - val_acc
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Accuracy Gap (Overfitting): {acc_gap:.4f}")
    
    # Save model and feature information
    model.save(f'valorant_model_top{feature_count}.h5')
    
    with open(f'feature_scaler_top{feature_count}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    with open(f'selected_features_top{feature_count}.pkl', 'wb') as f:
        pickle.dump(top_features, f)
        
    # Save feature importance information
    importance_dict = {feature_names[i]: float(importances[i]) for i in top_indices}
    with open(f'feature_importance_top{feature_count}.json', 'w') as f:
        json.dump(importance_dict, f, indent=2)
    
    print(f"\nModel and feature information saved:")
    print(f"  - Model: valorant_model_top{feature_count}.h5")
    print(f"  - Scaler: feature_scaler_top{feature_count}.pkl")
    print(f"  - Features: selected_features_top{feature_count}.pkl")
    print(f"  - Importance: feature_importance_top{feature_count}.json")
    
    return model, scaler, top_features

# 1. Add k-fold cross-validation functionality
def train_with_cross_validation(X, y, n_splits=5, random_state=42):
    """Train model using k-fold cross-validation for more robust performance estimation."""
    from sklearn.model_selection import StratifiedKFold
    
    print("\n" + "="*60)
    print(f"TRAINING WITH {n_splits}-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # Convert data to the right format
    if isinstance(X, list):
        # Check for dictionary values in the list
        for i, sample in enumerate(X):
            for key, value in list(sample.items()):
                if isinstance(value, dict):
                    print(f"Warning: Found dictionary value for feature '{key}' in sample {i}")
                    # Either flatten the dict or remove it
                    del sample[key]
                elif not isinstance(value, (int, float, bool, str)):
                    print(f"Warning: Non-numeric value for feature '{key}' in sample {i}: {type(value)}")
                    del sample[key]
        
        # Convert to DataFrame for easier handling
        X = pd.DataFrame(X)
    
    # Ensure all values are numeric
    if isinstance(X, pd.DataFrame):
        # Replace any remaining dictionaries or non-numeric values
        for col in X.columns:
            if X[col].apply(lambda x: isinstance(x, dict)).any():
                print(f"Removing column '{col}' because it contains dictionary values")
                X = X.drop(columns=[col])
            elif not pd.api.types.is_numeric_dtype(X[col]):
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    print(f"Converted column '{col}' to numeric")
                except:
                    print(f"Removing column '{col}' because it contains non-numeric values")
                    X = X.drop(columns=[col])
    
    # Convert to numpy array
    X = np.array(X)
    if isinstance(y, list):
        y = np.array(y)
    
    
    # Set up cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize arrays to store results
    fold_metrics = []
    fold_models = []
    
    # For storing feature importances across folds
    all_feature_importances = {}
    
    # Scale features outside the loop to ensure consistent scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        print(f"\n----- Training Fold {fold+1}/{n_splits} -----")
        
        # Split data
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Check for class imbalance and apply SMOTE if needed
        class_counts = np.bincount(y_train)
        if np.min(class_counts) / np.sum(class_counts) < 0.4:
            try:
                min_samples = np.min(class_counts)
                k_neighbors = min(5, min_samples-1)
                smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"Applied SMOTE resampling: {np.bincount(y_train)}")
            except Exception as e:
                print(f"Error applying SMOTE: {e}")
        
        # Feature selection for this fold
        X_train_selected, selected_features, _ = select_optimal_features(
            pd.DataFrame(X_train), y_train, test_size=0.2, random_state=random_state)
        
        # Apply same feature selection to validation set
        if isinstance(selected_features[0], int):
            # If features are indices
            X_val_selected = X_val[:, selected_features]
        else:
            # If features are column names
            feature_indices = [list(df.columns).index(f) for f in selected_features]
            X_val_selected = X_val[:, feature_indices]
        
        # Store feature importances
        for feature in selected_features:
            if feature in all_feature_importances:
                all_feature_importances[feature] += 1
            else:
                all_feature_importances[feature] = 1
        
        # Train model
        input_dim = X_train_selected.shape[1]
        model = create_deep_learning_model_with_advanced_lr(input_dim)
        
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        )
        
        # Use CosineAnnealingLR instead of ReduceLROnPlateau
        cosine_lr = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * (1 + np.cos(epoch * np.pi / 50)), verbose=0
        )
        
        # Train the model
        history = model.fit(
            X_train_selected, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_selected, y_val),
            callbacks=[early_stopping, cosine_lr],
            verbose=1
        )
        
        # Evaluate model
        y_pred_proba = model.predict(X_val_selected)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics with NaN handling
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        # Safely calculate AUC - handle NaN values
        try:
            # Check if there are NaN values in the predictions
            if np.isnan(y_pred_proba).any():
                print(f"  Warning: NaN values detected in predictions. Using fallback AUC value.")
                auc = 0.5  # Default to random guessing
            else:
                auc = roc_auc_score(y_val, y_pred_proba)
        except Exception as e:
            print(f"  Error calculating AUC: {e}")
            auc = 0.5  # Default to random guessing



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
    
    # Analyze cross-validation results
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*60)
    
    # Calculate average metrics
    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) 
                   for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
    
    std_metrics = {metric: np.std([fold[metric] for fold in fold_metrics]) 
                  for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
    
    print(f"Average Metrics Across {n_splits} Folds:")
    for metric, value in avg_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f} ± {std_metrics[metric]:.4f}")
    
    # Analyze feature importance consistency
    sorted_features = sorted(all_feature_importances.items(), 
                           key=lambda x: x[1], reverse=True)
    
    print("\nTop Features By Selection Frequency:")
    for feature, count in sorted_features[:20]:
        print(f"  {feature}: Selected in {count}/{n_splits} folds")
    
    # Create a stable feature set
    stable_features = [feature for feature, count in sorted_features 
                      if count >= n_splits * 0.8]  # Features selected in at least 80% of folds
    
    print(f"\nStable Feature Set: {len(stable_features)} features")
    
    return fold_models, stable_features, avg_metrics, fold_metrics, scaler

def select_optimal_features(X, y, test_size=0.2, random_state=42):
    """Perform feature selection to reduce overfitting and improve model performance."""
    print("\n" + "="*60)
    print("PERFORMING FEATURE SELECTION")
    print("="*60)
    
    # Convert data to DataFrame
    df = pd.DataFrame(X)
    
    # Fill missing values and handle non-numeric columns
    df = df.fillna(0)
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(float)
            except (ValueError, TypeError):
                df = df.drop(columns=[col])
    
    X_arr = df.values
    y_arr = np.array(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )
    
    # Create baseline model
    input_dim = X_train.shape[1]
    baseline_model = create_deep_learning_model_with_economy(input_dim)
    
    # Evaluate importance of each feature using permutation importance
    from sklearn.inspection import permutation_importance
    from sklearn.ensemble import RandomForestClassifier
    
    print("Training a Random Forest model for feature importance analysis...")
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    
    # Compute feature importances
    importances = rf.feature_importances_
    feature_importances = list(zip(df.columns, importances))
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 20 important features
    print("\nTop 20 most important features:")
    for feature, importance in feature_importances[:20]:
        print(f"{feature}: {importance:.4f}")
    
    # Try different feature count thresholds
    feature_counts = [int(input_dim * ratio) for ratio in [0.25, 0.5, 0.75, 1.0]]
    feature_counts = sorted(list(set([min(count, input_dim) for count in feature_counts] + [20, 50, 100])))
    feature_counts = [count for count in feature_counts if count <= input_dim]
    
    print(f"\nTesting models with different feature counts: {feature_counts}")
    
    results = []
    for n_features in feature_counts:
        print(f"\nTesting with top {n_features} features...")
        
        # Select top features
        top_features = [feature for feature, _ in feature_importances[:n_features]]
        X_train_selected = X_train[:, [list(df.columns).index(feature) for feature in top_features]]
        X_val_selected = X_val[:, [list(df.columns).index(feature) for feature in top_features]]
        
        # Train model with selected features
        model = create_deep_learning_model_with_economy(X_train_selected.shape[1])
        
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
        )
        
        history = model.fit(
            X_train_selected, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val_selected, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model
        y_pred_proba = model.predict(X_val_selected)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        # Calculate overfitting
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        acc_gap = train_acc - val_acc
        
        print(f"  Features: {n_features}, Val Acc: {val_acc:.4f}, Gap: {acc_gap:.4f}, AUC: {auc:.4f}")
        
        results.append({
            'n_features': n_features,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'acc_gap': acc_gap,
            'auc': auc,
            'features': top_features
        })
    
    # Find optimal feature count
    df_results = pd.DataFrame(results)
    
    # Score based on validation accuracy with penalty for overfitting
    df_results['score'] = df_results['val_acc'] - 0.5 * df_results['acc_gap'] + 0.2 * df_results['auc']
    
    best_result = df_results.loc[df_results['score'].idxmax()]
    optimal_feature_count = best_result['n_features']
    optimal_features = best_result['features']
    
    print("\n" + "="*60)
    print(f"OPTIMAL FEATURE SET: {optimal_feature_count} features")
    print("="*60)
    print(f"Validation Accuracy: {best_result['val_acc']:.4f}")
    print(f"Train-Val Accuracy Gap: {best_result['acc_gap']:.4f}")
    print(f"AUC: {best_result['auc']:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot feature count vs metrics
    plt.subplot(2, 2, 1)
    plt.plot(df_results['n_features'], df_results['val_acc'], marker='o', label='Validation Accuracy')
    plt.plot(df_results['n_features'], df_results['train_acc'], marker='o', label='Training Accuracy')
    plt.axvline(x=optimal_feature_count, color='red', linestyle='--', label=f'Optimal Count: {optimal_feature_count}')
    plt.title('Accuracy vs Feature Count', fontsize=14)
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Plot feature count vs overfitting
    plt.subplot(2, 2, 2)
    plt.plot(df_results['n_features'], df_results['acc_gap'], marker='o', color='red')
    plt.axvline(x=optimal_feature_count, color='green', linestyle='--', label=f'Optimal Count: {optimal_feature_count}')
    plt.title('Overfitting vs Feature Count', fontsize=14)
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Train-Val Accuracy Gap', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Plot feature count vs AUC
    plt.subplot(2, 2, 3)
    plt.plot(df_results['n_features'], df_results['auc'], marker='o', color='purple')
    plt.axvline(x=optimal_feature_count, color='green', linestyle='--', label=f'Optimal Count: {optimal_feature_count}')
    plt.title('AUC vs Feature Count', fontsize=14)
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Plot feature count vs score
    plt.subplot(2, 2, 4)
    plt.plot(df_results['n_features'], df_results['score'], marker='o', color='blue')
    plt.axvline(x=optimal_feature_count, color='green', linestyle='--', label=f'Optimal Count: {optimal_feature_count}')
    plt.title('Score vs Feature Count', fontsize=14)
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Combined Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('feature_selection_results.png', dpi=300)
    plt.show()
    
    # Categorize and analyze the selected features
    if optimal_features:
        print("\nAnalyzing selected features by category:")
        
        # Check if optimal_features contains integers or indices instead of feature names
        if optimal_features and isinstance(optimal_features[0], int):
            # Convert indices to feature names
            optimal_feature_names = [list(df.columns)[i] for i in optimal_features]
        else:
            optimal_feature_names = optimal_features
        
        # Economy features - use str() for safety
        economy_features = [f for f in optimal_feature_names if any(term in str(f).lower() for term in 
                                                        ['eco', 'pistol', 'buy', 'economy'])]
        print(f"\nEconomy features ({len(economy_features)}/{len(economy_features)}): ")
        for feature in economy_features[:10]:  # Show first 10
            print(f"  - {feature}")
        
        # Player features - use str() for safety
        player_features = [f for f in optimal_feature_names if any(term in str(f).lower() for term in 
                                                      ['rating', 'acs', 'kd', 'adr', 'headshot',
                                                       'clutch', 'aces', 'first_blood'])]
        print(f"\nPlayer features ({len(player_features)}/{len(player_features)}): ")
        for feature in player_features[:10]:  # Show first 10
            print(f"  - {feature}")
        
        # Map features - use str() for safety
        map_features = [f for f in optimal_feature_names if 'map_' in str(f).lower()]
        print(f"\nMap features ({len(map_features)}/{len(map_features)}): ")
        for feature in map_features[:10]:  # Show first 10
            print(f"  - {feature}")
        
        # Calculate retention percentage by category
        all_economy_features = [f for f in df.columns if any(term in str(f).lower() for term in 
                                                  ['eco', 'pistol', 'buy', 'economy'])]
        all_player_features = [f for f in df.columns if any(term in str(f).lower() for term in 
                                                 ['rating', 'acs', 'kd', 'adr', 'headshot',
                                                  'clutch', 'aces', 'first_blood'])]
        all_map_features = [f for f in df.columns if 'map_' in str(f).lower()]
        
        economy_retention = len(economy_features) / max(len(all_economy_features), 1) * 100
        player_retention = len(player_features) / max(len(all_player_features), 1) * 100
        map_retention = len(map_features) / max(len(all_map_features), 1) * 100
        
        print("\nFeature retention by category:")
        print(f"  Economy features: {economy_retention:.1f}%")
        print(f"  Player features: {player_retention:.1f}%")
        print(f"  Map features: {map_retention:.1f}%")
        
        # Prepare selected feature dataset - use original optimal_features
        if isinstance(optimal_features[0], int):
            # If optimal_features contains indices
            selected_features_indices = optimal_features
        else:
            # If optimal_features contains feature names
            selected_features_indices = [list(df.columns).index(feature) for feature in optimal_features]
            
        X_selected = X_arr[:, selected_features_indices]
        
        # Scale selected features
        scaler_selected = StandardScaler()
        X_selected_scaled = scaler_selected.fit_transform(X_selected)
        
        # Return original optimal_features (names or indices) to maintain consistency
        return X_selected_scaled, optimal_features, scaler_selected
    
    return X_scaled, list(df.columns), scaler 

def optimize_model_pipeline(X, y, test_size=0.2, random_state=42):
    """Complete pipeline to optimize model with feature selection and regularization."""
    # Step 1: Perform feature selection
    print("\nStep 1: Performing feature selection to reduce overfitting...")
    X_selected, selected_features, scaler_selected = select_optimal_features(X, y, test_size, random_state)
    
    # Step 2: Diagnose overfitting with learning curves
    print("\nStep 2: Training model with selected features and diagnosing overfitting...")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train model with learning curves
    input_dim = X_train.shape[1]
    model = create_deep_learning_model_with_economy(input_dim)
    
    # Setup learning curve tracking
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    epochs_completed = 0
    
    class LearningCurvesCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            nonlocal epochs_completed
            epochs_completed = epoch + 1
            
            training_losses.append(logs.get('loss'))
            validation_losses.append(logs.get('val_loss'))
            training_accuracies.append(logs.get('accuracy'))
            validation_accuracies.append(logs.get('val_accuracy'))
    
    learning_curves_callback = LearningCurvesCallback()
    
    # Define other callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_valorant_model_selected_features.h5', 
        save_best_only=True, 
        monitor='val_accuracy'
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, model_checkpoint, learning_curves_callback],
        verbose=1
    )
    
    # Check for overfitting
    train_acc = training_accuracies[-1]
    val_acc = validation_accuracies[-1]
    acc_gap = train_acc - val_acc
    
    if acc_gap > 0.08:  # Moderate to severe overfitting
        print(f"\nOverfitting detected after feature selection. Gap: {acc_gap:.4f}")
        print("\nStep 3: Finding optimal regularization parameters...")
        
        # Find optimal regularization parameters
        regularization_strengths = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        dropout_rates = [0.3, 0.4, 0.5]
        
        # Store results
        reg_results = []
        
        # Try different combinations
        total_combos = len(regularization_strengths) * len(dropout_rates)
        combo_count = 0
        
        for reg_strength in regularization_strengths:
            for dropout_rate in dropout_rates:
                combo_count += 1
                print(f"\nTesting combination {combo_count}/{total_combos}: L2={reg_strength}, Dropout={dropout_rate}")
                
                # Create and train model
                model = create_deep_learning_model_with_economy_and_regularization(
                    input_dim, regularization_strength=reg_strength, dropout_rate=dropout_rate
                )
                
                # Train with early stopping
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
                
                history = model.fit(
                    X_train, y_train,
                    epochs=30,  # Reduced epochs for faster testing
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop],
                    verbose=0
                )
                
                # Evaluate results
                train_acc = history.history['accuracy'][-1]
                val_acc = history.history['val_accuracy'][-1]
                train_loss = history.history['loss'][-1]
                val_loss = history.history['val_loss'][-1]
                
                acc_gap = train_acc - val_acc
                
                reg_results.append({
                    'reg_strength': reg_strength,
                    'dropout_rate': dropout_rate,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'acc_gap': acc_gap
                })
                
                print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Gap: {acc_gap:.4f}")
        
        # Find best parameters
        df_results = pd.DataFrame(reg_results)
        df_results['score'] = df_results['val_acc'] - 0.5 * df_results['acc_gap']
        best_params = df_results.loc[df_results['score'].idxmax()]
        
        best_reg_strength = best_params['reg_strength']
        best_dropout_rate = best_params['dropout_rate']
        
        print("\nBest regularization parameters found:")
        print(f"L2 Regularization Strength: {best_reg_strength}")
        print(f"Dropout Rate: {best_dropout_rate}")
        
        # Train final model with best parameters
        print("\nStep 4: Training final model with optimal feature set and regularization...")
        final_model = create_deep_learning_model_with_economy_and_regularization(
            input_dim, regularization_strength=best_reg_strength, dropout_rate=best_dropout_rate
        )
        
        final_history = final_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )
        
        # Save final model
        final_model.save('valorant_model_optimized.h5')
        
        # Save feature list and scaler
        with open('selected_features.pkl', 'wb') as f:
            pickle.dump(selected_features, f)
        
        with open('feature_scaler_optimized.pkl', 'wb') as f:
            pickle.dump(scaler_selected, f)
        
        # Evaluate final model
        y_pred_proba = final_model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        print("\n" + "="*60)
        print("FINAL OPTIMIZED MODEL EVALUATION")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        final_train_acc = final_history.history['accuracy'][-1]
        final_val_acc = final_history.history['val_accuracy'][-1]
        final_acc_gap = final_train_acc - final_val_acc
        
        print(f"\nFinal Train-Val Accuracy Gap: {final_acc_gap:.4f}")
        print(f"Initial Gap: {acc_gap:.4f}")
        
        gap_reduction = 100 * (1 - final_acc_gap / acc_gap)
        print(f"Overfitting Reduction: {gap_reduction:.2f}%")
        
        return final_model, scaler_selected, selected_features
    else:
        print(f"\nNo significant overfitting detected after feature selection. Gap: {acc_gap:.4f}")
        print("Using model with selected features without additional regularization.")
        
        # Load the best model from checkpoint
        final_model = load_model('best_valorant_model_selected_features.h5')
        
        # Save feature list and scaler for future use
        with open('selected_features.pkl', 'wb') as f:
            pickle.dump(selected_features, f)
        
        with open('feature_scaler_optimized.pkl', 'wb') as f:
            pickle.dump(scaler_selected, f)
        
        # Evaluate final model
        y_pred_proba = final_model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        print("\n" + "="*60)
        print("FEATURE-SELECTED MODEL EVALUATION")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        # Compare with original full feature set
        print("\nFeature Selection Summary:")
        print(f"Original feature count: {len(X[0]) if isinstance(X, list) else X.shape[1]}")
        print(f"Selected feature count: {len(selected_features)}")
        reduction_pct = (1 - len(selected_features) / (len(X[0]) if isinstance(X, list) else X.shape[1])) * 100
        print(f"Feature reduction: {reduction_pct:.1f}%")
        
        return final_model, scaler_selected, selected_features

def train_model_with_learning_curves(X, y, test_size=0.2, random_state=42):
    """Train the deep learning model with detailed learning curves for early overfitting detection."""
    # Check if we have data
    if not X or len(X) == 0:
        print("Error: No training data available")
        return None, None, None, None
        
    # Convert feature dictionary to DataFrame and then to numpy array
    df = pd.DataFrame(X)
    
    # Fill NA values with 0
    df = df.fillna(0)
    
    # Print column info for debugging
    print("\nFeature columns and their types:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
        # Print a few examples if the type is object (non-numeric)
        if df[col].dtype == 'object':
            print(f"  Examples: {df[col].head(3).tolist()}")
            # Convert objects to numeric if possible, otherwise drop
            try:
                df[col] = df[col].astype(float)
                print(f"  Converted {col} to float")
            except (ValueError, TypeError):
                print(f"  Dropping column {col} due to non-numeric values")
                df = df.drop(columns=[col])
    
    # Check if DataFrame is empty after cleaning
    if df.empty:
        print("Error: Empty feature dataframe after cleaning")
        return None, None, None, None
    
    # Convert to numpy array
    X_arr = df.values
    y_arr = np.array(y)
    
    print(f"\nFinal feature matrix shape: {X_arr.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )
    
    # Further split training data to create a smaller training subset for learning curves
    X_train_subset, _, y_train_subset, _ = train_test_split(
        X_train, y_train, train_size=0.5, random_state=random_state, stratify=y_train
    )
    
    # Check for class imbalance
    class_counts = np.bincount(y_train)
    print(f"Class distribution: {class_counts}")
    
    # Handle class imbalance if necessary
    if np.min(class_counts) / np.sum(class_counts) < 0.4:  # If imbalanced
        print("Detected class imbalance, applying SMOTE...")
        try:
            if np.min(class_counts) < 5:
                print("Not enough samples in minority class for SMOTE. Using original data.")
            else:
                min_samples = np.min(class_counts)
                k_neighbors = min(5, min_samples-1)
                print(f"Using k_neighbors={k_neighbors} for SMOTE (min_samples={min_samples})")
                
                smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                X_train_subset, y_train_subset = smote.fit_resample(X_train_subset, y_train_subset)
                print(f"After SMOTE: X_train shape: {X_train.shape}, X_train_subset shape: {X_train_subset.shape}")
        except Exception as e:
            print(f"Error applying SMOTE: {e}")
            print("Continuing with original data.")
    
    # Create and train model
    input_dim = X_train.shape[1]
    model = create_deep_learning_model_with_economy(input_dim)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_valorant_model.h5', 
        save_best_only=True, 
        monitor='val_accuracy'
    )
    
    # Lists to store metrics for learning curves
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    subset_training_losses = []
    subset_training_accuracies = []
    epochs_completed = 0
    
    class LearningCurvesCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            nonlocal epochs_completed
            epochs_completed = epoch + 1
            
            # Store main metrics
            training_losses.append(logs.get('loss'))
            validation_losses.append(logs.get('val_loss'))
            training_accuracies.append(logs.get('accuracy'))
            validation_accuracies.append(logs.get('val_accuracy'))
            
            # Evaluate on training subset (to diagnose bias/variance)
            subset_metrics = self.model.evaluate(X_train_subset, y_train_subset, verbose=0)
            subset_training_losses.append(subset_metrics[0])
            subset_training_accuracies.append(subset_metrics[1])
            
            # Print overfitting diagnostic
            train_val_loss_gap = logs.get('loss') - logs.get('val_loss')
            train_val_acc_gap = logs.get('accuracy') - logs.get('val_accuracy')
            
            overfitting_status = "SEVERE OVERFITTING" if train_val_acc_gap > 0.15 else \
                               "Moderate Overfitting" if train_val_acc_gap > 0.08 else \
                               "Slight Overfitting" if train_val_acc_gap > 0.03 else \
                               "Good Fit"
            
            print(f"\nEpoch {epoch+1} Overfitting Status: {overfitting_status}")
            print(f"Train-Val Accuracy Gap: {train_val_acc_gap:.4f}, Loss Gap: {train_val_loss_gap:.4f}")
    
    learning_curves_callback = LearningCurvesCallback()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, model_checkpoint, learning_curves_callback],
        verbose=1
    )
    
    # Plot detailed learning curves
    plt.figure(figsize=(20, 10))
    
    # Loss curves - Main plot
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs_completed + 1), training_losses, label='Training Loss', marker='o')
    plt.plot(range(1, epochs_completed + 1), validation_losses, label='Validation Loss', marker='o')
    plt.title('Model Loss', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Accuracy curves - Main plot
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs_completed + 1), training_accuracies, label='Training Accuracy', marker='o')
    plt.plot(range(1, epochs_completed + 1), validation_accuracies, label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Training Loss comparison (full vs subset)
    plt.subplot(2, 2, 3)
    plt.plot(range(1, epochs_completed + 1), training_losses, label='Full Training Set Loss', marker='o')
    plt.plot(range(1, epochs_completed + 1), subset_training_losses, label='Training Subset Loss', marker='o')
    plt.title('Training Loss: Full Set vs Subset', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Plot accuracy gaps (to visualize overfitting)
    plt.subplot(2, 2, 4)
    acc_gaps = [train - val for train, val in zip(training_accuracies, validation_accuracies)]
    plt.plot(range(1, epochs_completed + 1), acc_gaps, label='Train-Validation Accuracy Gap', marker='o', color='red')
    plt.axhline(y=0.05, color='green', linestyle='--', label='Acceptable Gap Threshold')
    plt.axhline(y=0.10, color='orange', linestyle='--', label='Moderate Overfitting Threshold')
    plt.axhline(y=0.15, color='red', linestyle='--', label='Severe Overfitting Threshold')
    plt.title('Overfitting Analysis', fontsize=14)
    plt.ylabel('Accuracy Gap', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('detailed_learning_curves.png', dpi=300)
    plt.show()
    
    # Create overfitting diagnosis report
    final_train_acc = training_accuracies[-1]
    final_val_acc = validation_accuracies[-1]
    final_acc_gap = final_train_acc - final_val_acc
    
    max_val_acc = max(validation_accuracies)
    max_val_acc_epoch = validation_accuracies.index(max_val_acc) + 1
    
    optimal_stopping_epoch = early_stopping.best_epoch + 1 if hasattr(early_stopping, 'best_epoch') else max_val_acc_epoch
    
    print("\n" + "="*60)
    print("OVERFITTING DIAGNOSIS REPORT")
    print("="*60)
    
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Accuracy Gap: {final_acc_gap:.4f}")
    
    print(f"\nHighest Validation Accuracy: {max_val_acc:.4f} (Epoch {max_val_acc_epoch})")
    print(f"Optimal Stopping Epoch: {optimal_stopping_epoch}")
    
    if final_acc_gap > 0.15:
        overfitting_diagnosis = "SEVERE OVERFITTING"
        recommendations = [
            "1. Increase regularization (L1, L2, or both)",
            "2. Increase dropout rate",
            "3. Reduce model complexity (fewer layers/neurons)",
            "4. Apply early stopping at epoch " + str(optimal_stopping_epoch),
            "5. Collect more training data if possible",
            "6. Try feature selection to reduce dimensionality"
        ]
    elif final_acc_gap > 0.08:
        overfitting_diagnosis = "MODERATE OVERFITTING"
        recommendations = [
            "1. Slightly increase regularization",
            "2. Slightly increase dropout rate",
            "3. Apply early stopping at epoch " + str(optimal_stopping_epoch),
            "4. Consider adding more diverse data augmentation"
        ]
    elif final_acc_gap > 0.03:
        overfitting_diagnosis = "SLIGHT OVERFITTING"
        recommendations = [
            "1. Apply early stopping at epoch " + str(optimal_stopping_epoch),
            "2. Consider adding small amounts of regularization",
            "3. Feature selection might help"
        ]
    else:
        overfitting_diagnosis = "GOOD FIT"
        recommendations = [
            "1. Model is well-balanced",
            "2. Consider training longer if validation accuracy is still improving",
            "3. Try increasing model capacity slightly to improve performance"
        ]
    
    print(f"\nDiagnosis: {overfitting_diagnosis}")
    print("\nRecommendations:")
    for rec in recommendations:
        print(rec)
    
    print("\nAdvanced Analysis:")
    if max(acc_gaps) > 0.15 and max(validation_accuracies) < 0.65:
        print("- Model shows signs of SEVERE overfitting but low overall performance")
        print("- This suggests the model is memorizing noise in a difficult dataset")
        print("- Consider better feature engineering before increasing regularization")
    elif max_val_acc < 0.6:
        print("- Model isn't achieving high validation accuracy")
        print("- This suggests underfitting or poor feature quality")
        print("- Consider better feature engineering or a more complex model")
    
    # Analyze learning rate effects
    if hasattr(history.history, 'lr'):
        lr_changes = [i for i, (lr1, lr2) in enumerate(zip(history.history['lr'][:-1], history.history['lr'][1:])) if lr1 != lr2]
        if lr_changes:
            print("\nLearning Rate Effect Analysis:")
            for epoch in lr_changes:
                print(f"- Learning rate changed at epoch {epoch+1}")
                print(f"  Before change: Val Acc = {validation_accuracies[epoch]:.4f}, Val Loss = {validation_losses[epoch]:.4f}")
                print(f"  After change: Val Acc = {validation_accuracies[epoch+1]:.4f}, Val Loss = {validation_losses[epoch+1]:.4f}")
    
    # Save learning curve data for future analysis
    learning_curve_data = {
        'epochs': list(range(1, epochs_completed + 1)),
        'training_loss': training_losses,
        'validation_loss': validation_losses,
        'training_accuracy': training_accuracies,
        'validation_accuracy': validation_accuracies,
        'subset_training_loss': subset_training_losses,
        'subset_training_accuracy': subset_training_accuracies,
        'accuracy_gap': acc_gaps,
        'diagnosis': overfitting_diagnosis,
        'recommendations': recommendations
    }
    
    with open('learning_curve_data.pkl', 'wb') as f:
        pickle.dump(learning_curve_data, f)
    
    # Evaluate on test set
    y_pred_proba = model.predict(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\nModel Evaluation on Validation Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("="*60)
    
    # Save model artifacts
    model.save('valorant_model.h5')
    
    # Save scaler for future use
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(list(df.columns), f)
    
    return model, scaler, list(df.columns), learning_curve_data

# 4. Update the main function to use cross-validation and ensembling
def train_and_evaluate_model(X, y, n_splits=5, random_state=42):
    """Complete pipeline for training with cross-validation and ensembling."""
    # Train with cross-validation
    ensemble_models, stable_features, avg_metrics, fold_metrics, scaler = train_with_cross_validation(
        X, y, n_splits=n_splits, random_state=random_state
    )
    
    # Save the ensemble models
    for i, model in enumerate(ensemble_models):
        model.save(f'valorant_model_fold_{i+1}.h5')
    
    # Save the stable features
    with open('stable_features.pkl', 'wb') as f:
        pickle.dump(stable_features, f)
    
    # Save the scaler
    with open('ensemble_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save the ensemble metadata
    ensemble_metadata = {
        'n_models': len(ensemble_models),
        'stable_features': stable_features,
        'avg_metrics': avg_metrics,
        'fold_metrics': fold_metrics
    }
    
    with open('ensemble_metadata.pkl', 'wb') as f:
        pickle.dump(ensemble_metadata, f)
    
    print("\n" + "="*60)
    print("ENSEMBLE MODEL SAVED")
    print("="*60)
    print(f"Number of models in ensemble: {len(ensemble_models)}")
    print(f"Number of stable features: {len(stable_features)}")
    print(f"Average accuracy: {avg_metrics['accuracy']:.4f}")
    
    return ensemble_models, stable_features, scaler, ensemble_metadata

def train_with_optimal_features(X, y, optimal_count=71):
    """Train model using only the optimal number of features."""
    # Convert data to DataFrame
    df = pd.DataFrame(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train a simple model to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_importances = list(zip(df.columns, importances))
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    
    # Select top features
    top_features = [feature for feature, _ in feature_importances[:optimal_count]]
    X_train_selected = X_train[:, [list(df.columns).index(feature) for feature in top_features]]
    X_val_selected = X_val[:, [list(df.columns).index(feature) for feature in top_features]]
    
    # Train neural network with selected features
    input_dim = X_train_selected.shape[1]
    model = create_deep_learning_model_with_economy(input_dim)
    
    # Add learning rate scheduler
    lr_scheduler = create_learning_rate_scheduler()
    
    # Train model
    model.fit(
        X_train_selected, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_selected, y_val),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            lr_scheduler,
            ModelCheckpoint('valorant_model_optimal_features.h5', save_best_only=True, monitor='val_accuracy')
        ],
        verbose=1
    )
    
    # Save selected features
    with open('optimal_features.pkl', 'wb') as f:
        pickle.dump(top_features, f)
    
    # Save scaler
    with open('feature_scaler_optimal.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler, top_features

def predict_match_with_ensemble(team1_name, team2_name):
    """Predict match outcome using the ensemble model for improved stability."""
    print(f"Predicting match between {team1_name} and {team2_name} using ensemble model...")
    
    # Get team IDs
    team1_id = get_team_id(team1_name)
    team2_id = get_team_id(team2_name)
    
    if not team1_id or not team2_id:
        print("Could not find one or both teams. Please check team names.")
        return None

    # Fetch team details to get team tags
    team1_details, team1_tag = fetch_team_details(team1_id)
    team2_details, team2_tag = fetch_team_details(team2_id)    
    print(f"Team tags: {team1_name} = {team1_tag}, {team2_name} = {team2_tag}")
    
    # Fetch match histories
    team1_history = fetch_team_match_history(team1_id)
    team2_history = fetch_team_match_history(team2_id)
    
    if not team1_history or not team2_history:
        print("Could not fetch match history for one or both teams.")
        return None
    
    # Parse match data
    team1_matches = parse_match_data(team1_history, team1_name)
    team2_matches = parse_match_data(team2_history, team2_name)

    # Store team tags for use in economy data matching
    for match in team1_matches:
        match['team_tag'] = team1_tag
    
    for match in team2_matches:
        match['team_tag'] = team2_tag
    
    # Fetch player stats for both teams
    team1_player_stats = fetch_team_player_stats(team1_id)
    team2_player_stats = fetch_team_player_stats(team2_id)

    # Calculate team stats with economy data
    team1_stats = calculate_team_stats_with_economy(team1_matches, team1_player_stats)
    team2_stats = calculate_team_stats_with_economy(team2_matches, team2_player_stats)
    
    # Store team tags in the stats
    team1_stats['team_tag'] = team1_tag
    team2_stats['team_tag'] = team2_tag
    
    # Extract additional metrics
    team1_map_performance = extract_map_performance(team1_matches)
    team2_map_performance = extract_map_performance(team2_matches)
    
    team1_tournament_performance = extract_tournament_performance(team1_matches)
    team2_tournament_performance = extract_tournament_performance(team2_matches)
    
    team1_performance_trends = analyze_performance_trends(team1_matches)
    team2_performance_trends = analyze_performance_trends(team2_matches)
    
    team1_opponent_quality = analyze_opponent_quality(team1_matches, team1_id)
    team2_opponent_quality = analyze_opponent_quality(team2_matches, team2_id)
    
    # Add derived metrics to team stats
    team1_stats['map_performance'] = team1_map_performance
    team2_stats['map_performance'] = team2_map_performance
    
    team1_stats['tournament_performance'] = team1_tournament_performance
    team2_stats['tournament_performance'] = team2_tournament_performance
    
    team1_stats['performance_trends'] = team1_performance_trends
    team2_stats['performance_trends'] = team2_performance_trends
    
    team1_stats['opponent_quality'] = team1_opponent_quality
    team2_stats['opponent_quality'] = team2_opponent_quality
    
    # Prepare data for model
    all_features = prepare_data_for_model_with_economy(team1_stats, team2_stats)
    
    if not all_features:
        print("Could not prepare features for prediction.")
        return None
    
      # Load ensemble models
    ensemble_models = []
    for i in range(5):  # Assuming 5-fold CV by default
        try:
            model = load_model(f'valorant_model_fold_{i+1}.h5')
            ensemble_models.append(model)
        except:
            print(f"Could not load model for fold {i+1}")
    
    if not ensemble_models:
        print("No ensemble models found. Please train an ensemble first.")
        return None
    
    # Load stable features
    try:
        with open('stable_features.pkl', 'rb') as f:
            stable_features = pickle.load(f)
    except:
        print("Could not load stable features. Using all features.")
        stable_features = None
    
    # Load scaler
    try:
        with open('ensemble_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except:
        print("Could not load scaler. Please train an ensemble first.")
        return None
    
    # Make prediction using ensemble
    ensemble_result = predict_with_ensemble(
        ensemble_models, all_features, stable_features, scaler
    )
    
    # Prepare result
    prediction = ensemble_result['prediction']
    confidence = 1.0 - ensemble_result['confidence_interval']  
    
    result = {
        'team1': team1_name,
        'team2': team2_name,
        'team1_win_probability': float(prediction),
        'team2_win_probability': float(1 - prediction),
        'predicted_winner': team1_name if prediction > 0.5 else team2_name,
        'confidence': float(confidence),
        'team1_stats_summary': {
            'matches_played': team1_stats['matches'] if isinstance(team1_stats['matches'], int) else len(team1_stats['matches']),
            'win_rate': team1_stats['win_rate'],
            'recent_form': team1_stats['recent_form'],
            'avg_player_rating': team1_stats.get('avg_player_rating', 0),
            'star_player': team1_stats.get('player_stats', {}).get('star_player_name', ''),
            'star_player_rating': team1_stats.get('star_player_rating', 0),
            'pistol_win_rate': team1_stats.get('pistol_win_rate', 0),
            'eco_win_rate': team1_stats.get('eco_win_rate', 0),
            'full_buy_win_rate': team1_stats.get('full_buy_win_rate', 0),
            'economy_efficiency': team1_stats.get('economy_efficiency', 0)
        },
        'team2_stats_summary': {
            'matches_played': team2_stats['matches'] if isinstance(team2_stats['matches'], int) else len(team2_stats['matches']),
            'win_rate': team2_stats['win_rate'],
            'recent_form': team2_stats['recent_form'],
            'avg_player_rating': team2_stats.get('avg_player_rating', 0),
            'star_player': team2_stats.get('player_stats', {}).get('star_player_name', ''),
            'star_player_rating': team2_stats.get('star_player_rating', 0),
            'pistol_win_rate': team2_stats.get('pistol_win_rate', 0),
            'eco_win_rate': team2_stats.get('eco_win_rate', 0),
            'full_buy_win_rate': team2_stats.get('full_buy_win_rate', 0),
            'economy_efficiency': team2_stats.get('economy_efficiency', 0)
        },
        'model_info': {
            'features_used': len(feature_names) if feature_names else 'all',
            'feature_names': feature_names if feature_names else 'all',
            'model_type': 'optimized' if os.path.exists('valorant_model_optimized.h5') else 'regular'
        }
    }
    
    # Export prediction data if requested
    if export_data:
        export_prediction_data_with_economy(result, team1_stats, team2_stats)
    
    # Display detailed results if requested
    if display_details:
        display_prediction_results_with_economy(result, team1_stats, team2_stats)
        
        # Additionally display model information
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        print(f"Model type: {result['model_info']['model_type']}")
        print(f"Features used: {result['model_info']['features_used']}")
        
        if feature_names and len(feature_names) <= 20:
            print("\nSelected features:")
            for feature in feature_names:
                print(f"  - {feature}")
        elif feature_names:
            print(f"\nUsing {len(feature_names)} selected features")
            
            # Count feature types
            economy_features = sum(1 for f in feature_names if any(term in f.lower() for term in 
                                                     ['eco', 'pistol', 'buy', 'economy']))
            player_features = sum(1 for f in feature_names if any(term in f.lower() for term in 
                                                    ['rating', 'acs', 'kd', 'adr', 'headshot',
                                                     'clutch', 'aces', 'first_blood']))
            map_features = sum(1 for f in feature_names if 'map_' in f.lower())
            
            print(f"  Economy features: {economy_features}")
            print(f"  Player features: {player_features}")
            print(f"  Map features: {map_features}")
    
    return result

def predict_match_with_optimized_model(team1_name, team2_name, model=None, scaler=None, feature_names=None, export_data=True, display_details=True):
    """Predict match outcome using the optimized model with enhanced map-specific features."""
    print(f"Predicting match between {team1_name} and {team2_name} using optimized model with map data...")
    
    # Get team IDs
    team1_id = get_team_id(team1_name)
    team2_id = get_team_id(team2_name)
    
    if not team1_id or not team2_id:
        print("Could not find one or both teams. Please check team names.")
        return None

    # Fetch team details to get team tags
    team1_details, team1_tag = fetch_team_details(team1_id)
    team2_details, team2_tag = fetch_team_details(team2_id)    
    print(f"Team tags: {team1_name} = {team1_tag or 'None (will use name as fallback)'}, {team2_name} = {team2_tag or 'None (will use name as fallback)'}")
    
    # Fetch match histories
    team1_history = fetch_team_match_history(team1_id)
    team2_history = fetch_team_match_history(team2_id)
    
    if not team1_history or not team2_history:
        print("Could not fetch match history for one or both teams.")
        return None
    
    # Parse match data
    team1_matches = parse_match_data(team1_history, team1_name)
    team2_matches = parse_match_data(team2_history, team2_name)

    # Store team tags and IDs for use in data extraction
    for match in team1_matches:
        match['team_tag'] = team1_tag
        match['team_id'] = team1_id
        match['team_name'] = team1_name  # Ensure team_name is explicitly set for fallback
    
    for match in team2_matches:
        match['team_tag'] = team2_tag
        match['team_id'] = team2_id
        match['team_name'] = team2_name  # Ensure team_name is explicitly set for fallback
    
    # Fetch player stats for both teams
    team1_player_stats = fetch_team_player_stats(team1_id)
    team2_player_stats = fetch_team_player_stats(team2_id)

    # Calculate team stats with economy data
    team1_stats = calculate_team_stats_with_economy(team1_matches, team1_player_stats)
    team2_stats = calculate_team_stats_with_economy(team2_matches, team2_player_stats)
    
    # Store team tags and names in the stats
    team1_stats['team_tag'] = team1_tag
    team1_stats['team_name'] = team1_name
    team1_stats['team_id'] = team1_id
    
    team2_stats['team_tag'] = team2_tag
    team2_stats['team_name'] = team2_name
    team2_stats['team_id'] = team2_id
    
    # Fetch and add map-specific statistics
    team1_map_stats = fetch_team_map_statistics(team1_id)
    team2_map_stats = fetch_team_map_statistics(team2_id)
    
    if team1_map_stats:
        team1_stats['map_statistics'] = team1_map_stats
        print(f"Added map statistics for {team1_name} ({len(team1_map_stats)} maps)")
    
    if team2_map_stats:
        team2_stats['map_statistics'] = team2_map_stats
        print(f"Added map statistics for {team2_name} ({len(team2_map_stats)} maps)")
    
    # Prepare data for model
    all_features = prepare_data_for_model_with_economy(team1_stats, team2_stats)
    
    if not all_features:
        print("Could not prepare features for prediction.")
        return None
    
    # Load model if not provided
    if model is None:
        try:
            # Try to load optimized model first
            if os.path.exists('valorant_model_optimized.h5'):
                model = load_model('valorant_model_optimized.h5')
                
                with open('feature_scaler_optimized.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                
                with open('selected_features.pkl', 'rb') as f:
                    feature_names = pickle.load(f)
                    
                print("Loaded optimized model with selected features.")
            else:
                # Fall back to regular model
                model = load_model('valorant_model.h5')
                
                with open('feature_scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                
                with open('feature_names.pkl', 'rb') as f:
                    feature_names = pickle.load(f)
                
                print("Loaded regular model (optimized model not found).")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train a model first or provide a trained model.")
            return None
    
    # Convert features to DataFrame
    features_df = pd.DataFrame([all_features])
    
    # Select only the features used in the model
    if feature_names:
        # Add missing features with default values
        for feature in feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Keep only the selected features
        features_df = features_df[feature_names]
    
    # Scale features
    X = scaler.transform(features_df.values)
    
    # Make prediction
    prediction = model.predict(X)[0][0]
    
    # Rest of the function remains unchanged...
    
    # Determine if prediction should be adjusted based on statistical norms
    team1_advantage_count = 0
    team2_advantage_count = 0
    
    # Check win rate
    if team1_stats.get('win_rate', 0) > team2_stats.get('win_rate', 0):
        team1_advantage_count += 1
    else:
        team2_advantage_count += 1
    
    # Check recent form
    if team1_stats.get('recent_form', 0) > team2_stats.get('recent_form', 0):
        team1_advantage_count += 1
    else:
        team2_advantage_count += 1
    
    # Check player ratings
    if team1_stats.get('avg_player_rating', 0) > team2_stats.get('avg_player_rating', 0):
        team1_advantage_count += 1
    else:
        team2_advantage_count += 1
    
    # Check map advantages
    if ('map_statistics' in team1_stats and 'map_statistics' in team2_stats):
        team1_maps = set(team1_stats['map_statistics'].keys())
        team2_maps = set(team2_stats['map_statistics'].keys())
        common_maps = team1_maps.intersection(team2_maps)
        
        team1_map_advantages = 0
        team2_map_advantages = 0
        
        for map_name in common_maps:
            t1_win_rate = team1_stats['map_statistics'][map_name]['win_percentage']
            t2_win_rate = team2_stats['map_statistics'][map_name]['win_percentage']
            
            if t1_win_rate > t2_win_rate + 0.1:  # 10% advantage threshold
                team1_map_advantages += 1
            elif t2_win_rate > t1_win_rate + 0.1:
                team2_map_advantages += 1
        
        if team1_map_advantages > team2_map_advantages:
            team1_advantage_count += 1
        elif team2_map_advantages > team1_map_advantages:
            team2_advantage_count += 1
    
    # Check head-to-head
    h2h_win_rate = 0.5  # Default to even
    if 'opponent_stats' in team1_stats and team2_name in team1_stats['opponent_stats']:
        h2h_win_rate = team1_stats['opponent_stats'][team2_name].get('win_rate', 0.5)
    
    if h2h_win_rate > 0.5:
        team1_advantage_count += 2  # Give extra weight to head-to-head
    elif h2h_win_rate < 0.5:
        team2_advantage_count += 2
    
    # Determine if prediction seems flipped
    team1_should_be_favored = team1_advantage_count > team2_advantage_count
    prediction_favors_team1 = prediction > 0.5
    
    print(f"Team1 advantages: {team1_advantage_count}, Team2 advantages: {team2_advantage_count}")
    print(f"Team1 should be favored based on stats: {team1_should_be_favored}")
    print(f"Prediction favors Team1: {prediction_favors_team1}")
    
    # Fix for "flipped" predictions
    adjusted_prediction = prediction
    if team1_should_be_favored != prediction_favors_team1:
        print("WARNING: Model prediction appears to be inverted based on team statistics!")
        print(f"Adjusting prediction from {prediction:.4f} to {1-prediction:.4f}")
        adjusted_prediction = 1 - prediction
    
    # Calculate confidence
    confidence = max(adjusted_prediction, 1 - adjusted_prediction)
    
    # Prepare result object with all statistics
    result = {
        'team1': team1_name,
        'team2': team2_name,
        'team1_win_probability': float(adjusted_prediction),
        'team2_win_probability': float(1 - adjusted_prediction),
        'predicted_winner': team1_name if adjusted_prediction > 0.5 else team2_name,
        'confidence': float(confidence),
        'team1_stats_summary': {
            'matches_played': team1_stats['matches'] if isinstance(team1_stats['matches'], int) else len(team1_stats['matches']),
            'win_rate': team1_stats['win_rate'],
            'recent_form': team1_stats['recent_form'],
            'avg_player_rating': team1_stats.get('avg_player_rating', 0),
            'star_player': team1_stats.get('player_stats', {}).get('star_player_name', ''),
            'star_player_rating': team1_stats.get('star_player_rating', 0),
            'pistol_win_rate': team1_stats.get('pistol_win_rate', 0),
            'eco_win_rate': team1_stats.get('eco_win_rate', 0),
            'full_buy_win_rate': team1_stats.get('full_buy_win_rate', 0),
            'economy_efficiency': team1_stats.get('economy_efficiency', 0)
        },
        'team2_stats_summary': {
            'matches_played': team2_stats['matches'] if isinstance(team2_stats['matches'], int) else len(team2_stats['matches']),
            'win_rate': team2_stats['win_rate'],
            'recent_form': team2_stats['recent_form'],
            'avg_player_rating': team2_stats.get('avg_player_rating', 0),
            'star_player': team2_stats.get('player_stats', {}).get('star_player_name', ''),
            'star_player_rating': team2_stats.get('star_player_rating', 0),
            'pistol_win_rate': team2_stats.get('pistol_win_rate', 0),
            'eco_win_rate': team2_stats.get('eco_win_rate', 0),
            'full_buy_win_rate': team2_stats.get('full_buy_win_rate', 0),
            'economy_efficiency': team2_stats.get('economy_efficiency', 0)
        },
        'model_info': {
            'features_used': len(feature_names) if feature_names else 'all',
            'feature_names': feature_names if feature_names else 'all',
            'model_type': 'optimized' if os.path.exists('valorant_model_optimized.h5') else 'regular',
            'prediction_adjusted': team1_should_be_favored != prediction_favors_team1
        },
        'team1_stats': team1_stats,
        'team2_stats': team2_stats
    }
    
    # Export prediction data if requested
    if export_data:
        export_prediction_data_with_economy(result, team1_stats, team2_stats)
    
    # Display detailed results if requested
    if display_details:
        display_prediction_results_with_economy_and_maps(result, team1_stats, team2_stats)
    
    # Visualize the prediction with maps
    visualize_prediction_with_economy_and_maps(result)
    
    return result

# 3. Implement an ensemble prediction method
def predict_with_ensemble(ensemble_models, features, stable_features, scaler):
    """Make predictions using an ensemble of models for improved stability."""
    # Prepare features
    features_df = pd.DataFrame([features])
    
    # Select only stable features
    if stable_features:
        # Add missing features with default values
        for feature in stable_features:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Keep only the stable features
        features_df = features_df[stable_features]
    
    # Scale features
    X = scaler.transform(features_df.values)
    
    # Make predictions with each model in the ensemble
    all_predictions = []
    for model in ensemble_models:
        pred = model.predict(X)[0][0]
        all_predictions.append(pred)
    
    # Calculate ensemble prediction (mean of all models)
    ensemble_prediction = np.mean(all_predictions)
    
    # Calculate prediction variance (for confidence assessment)
    prediction_variance = np.var(all_predictions)
    
    # Calculate confidence interval (95%)
    confidence_interval = 1.96 * np.sqrt(prediction_variance / len(ensemble_models))
    
    result = {
        'prediction': float(ensemble_prediction),
        'variance': float(prediction_variance),
        'confidence_interval': float(confidence_interval),
        'individual_predictions': all_predictions
    }
    
    return result

def visualize_feature_importance(importance_report):
    """
    Visualize feature importance with focus on economy and player features.
    
    Args:
        importance_report: Report from analyze_feature_importance function
    """
    if not importance_report:
        print("No feature importance data to visualize.")
        return
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot top 10 overall features
    top_features = importance_report['top_features'][:10]
    
    # Determine the category for each feature and its value
    top_values = []
    colors = []
    
    for f in top_features:
        if f in importance_report['economy_features']:
            top_values.append(importance_report['economy_features'][f])
            colors.append('#3498db')  # Blue for economy
        elif f in importance_report['player_features']:
            top_values.append(importance_report['player_features'][f])
            colors.append('#2ecc71')  # Green for player
        elif f in importance_report['map_features']: 
            top_values.append(importance_report['map_features'][f])
            colors.append('#e74c3c')  # Red for map
        elif f in importance_report['opponent_features']:
            top_values.append(importance_report['opponent_features'][f])
            colors.append('#f39c12')  # Orange for opponent
        elif f in importance_report['team_features']:
            top_values.append(importance_report['team_features'][f])
            colors.append('#9b59b6')  # Purple for team
        else:
            # Default case if not found in any category
            top_values.append(0)
            colors.append('#95a5a6')  # Gray for unknown
    
    # Plot top features
    bars = ax1.barh(top_features[::-1], top_values[::-1], color=colors[::-1], alpha=0.7)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}',
                ha='left', va='center', fontsize=10)
    
    # Add labels and title
    ax1.set_title('Top 10 Most Important Features', fontsize=16)
    ax1.set_xlabel('Importance Score', fontsize=12)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.7, label='Economy Features'),
        Patch(facecolor='#2ecc71', alpha=0.7, label='Player Features'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Map Features'),
        Patch(facecolor='#f39c12', alpha=0.7, label='Opponent Features'),
        Patch(facecolor='#9b59b6', alpha=0.7, label='Team Features')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Plot category importance in second subplot
    categories = list(importance_report['category_importance'].keys())
    category_values = list(importance_report['category_importance'].values())
    
    # Category colors
    cat_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    # Plot category importance
    bars2 = ax2.bar(categories, category_values, color=cat_colors, alpha=0.7)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12)
    
    # Add labels and title
    ax2.set_title('Feature Category Importance', fontsize=16)
    ax2.set_ylabel('Average Importance Score', fontsize=12)
    ax2.set_ylim(0, max(category_values) * 1.2)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    # Print summary
    print("\nFeature Importance Analysis:")
    print(f"Economy Features Avg Importance: {importance_report['category_importance']['economy']:.3f}")
    print(f"Player Features Avg Importance: {importance_report['category_importance']['player']:.3f}")
    print(f"Map Features Avg Importance: {importance_report['category_importance']['map']:.3f}")
    print(f"Team Features Avg Importance: {importance_report['category_importance']['team']:.3f}")
    print(f"Opponent Features Avg Importance: {importance_report['category_importance']['opponent']:.3f}")
    
    # Print top economy features
    if importance_report['economy_features']:
        print("\nTop Economy Features:")
        for i, (feature, importance) in enumerate(sorted(importance_report['economy_features'].items(), 
                                                         key=lambda x: x[1], reverse=True)[:10]):
            print(f"{i+1}. {feature}: {importance:.3f}")
    
    # Print top player features
    if importance_report['player_features']:
        print("\nTop Player Features:")
        for i, (feature, importance) in enumerate(sorted(importance_report['player_features'].items(), 
                                                         key=lambda x: x[1], reverse=True)[:10]):
            print(f"{i+1}. {feature}: {importance:.3f}")

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

# Update visualize_prediction to include player stats
def visualize_prediction_with_economy(prediction_result):
    """Visualize the match prediction with player stats and economy data."""
    if not prediction_result:
        print("No prediction to visualize.")
        return
    
    team1 = prediction_result['team1']
    team2 = prediction_result['team2']
    team1_prob = prediction_result['team1_win_probability']
    team2_prob = prediction_result['team2_win_probability']
    predicted_winner = prediction_result['predicted_winner']
    confidence = prediction_result['confidence']
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    # Bar chart for win probabilities in first subplot
    teams = [team1, team2]
    probs = [team1_prob, team2_prob]
    colors = ['#3498db' if team == predicted_winner else '#e74c3c' for team in teams]
    
    bars = ax1.bar(teams, probs, color=colors, alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=12)
    
    # Add title and labels
    ax1.set_title(f'Win Probabilities', fontsize=16)
    ax1.set_ylabel('Win Probability', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Player stats comparison in second subplot
    t1_summary = prediction_result.get('team1_stats_summary', {})
    t2_summary = prediction_result.get('team2_stats_summary', {})
    
    # Extract player stats if available
    metrics = []
    t1_values = []
    t2_values = []
    
    if 'avg_player_rating' in t1_summary and 'avg_player_rating' in t2_summary:
        metrics.append('Player Rating')
        t1_values.append(t1_summary['avg_player_rating'])
        t2_values.append(t2_summary['avg_player_rating'])
        
        if 'star_player_rating' in t1_summary and 'star_player_rating' in t2_summary:
            metrics.append('Star Player Rating')
            t1_values.append(t1_summary['star_player_rating'])
            t2_values.append(t2_summary['star_player_rating'])
    
    # Add more traditional metrics
    metrics.extend(['Win Rate', 'Recent Form'])
    t1_values.extend([t1_summary.get('win_rate', 0), t1_summary.get('recent_form', 0)])
    t2_values.extend([t2_summary.get('win_rate', 0), t2_summary.get('recent_form', 0)])
    
    # Convert to numpy arrays for easier computation
    metrics = np.array(metrics)
    t1_values = np.array(t1_values)
    t2_values = np.array(t2_values)
    
    # Set up bar positions
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create bars
    rects1 = ax2.bar(x - width/2, t1_values, width, label=team1, color='#3498db', alpha=0.7)
    rects2 = ax2.bar(x + width/2, t2_values, width, label=team2, color='#e74c3c', alpha=0.7)
    
    # Add value labels
    for rect in rects1:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
                
    for rect in rects2:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    ax2.set_title(f'Team Stats Comparison', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=0)
    ax2.legend()
    
    # Economy metrics in third subplot
    econ_metrics = []
    t1_econ_values = []
    t2_econ_values = []
    
    # Check if economy metrics are available
    if ('pistol_win_rate' in t1_summary and 'pistol_win_rate' in t2_summary):
        econ_metrics.extend(['Pistol Win Rate', 'Eco Win Rate', 'Full Buy Win Rate'])
        t1_econ_values.extend([
            t1_summary.get('pistol_win_rate', 0),
            t1_summary.get('eco_win_rate', 0),
            t1_summary.get('full_buy_win_rate', 0)
        ])
        t2_econ_values.extend([
            t2_summary.get('pistol_win_rate', 0),
            t2_summary.get('eco_win_rate', 0),
            t2_summary.get('full_buy_win_rate', 0)
        ])
    
    if econ_metrics:
        # Convert to numpy arrays
        econ_metrics = np.array(econ_metrics)
        t1_econ_values = np.array(t1_econ_values)
        t2_econ_values = np.array(t2_econ_values)
        
        # Set up bar positions
        x_econ = np.arange(len(econ_metrics))
        
        # Create bars
        rects1_econ = ax3.bar(x_econ - width/2, t1_econ_values, width, label=team1, color='#3498db', alpha=0.7)
        rects2_econ = ax3.bar(x_econ + width/2, t2_econ_values, width, label=team2, color='#e74c3c', alpha=0.7)
        
        # Add value labels
        for rect in rects1_econ:
            height = rect.get_height()
            ax3.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
                    
        for rect in rects2_econ:
            height = rect.get_height()
            ax3.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        # Add labels and title
        ax3.set_title(f'Economy Performance', fontsize=16)
        ax3.set_xticks(x_econ)
        ax3.set_xticklabels(econ_metrics, rotation=15)
        ax3.set_ylim(0, 1)
        ax3.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax3.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax3.legend()
    else:
        # No economy data available
        ax3.text(0.5, 0.5, 'No Economy Data Available',
                ha='center', va='center', fontsize=14)
    
    # Add prediction summary
    plt.figtext(0.5, 0.01, 
                f"Predicted Winner: {predicted_winner} (Confidence: {confidence:.1%})",
                ha="center", fontsize=14, bbox={"facecolor":"#f9f9f9", "alpha":0.5, "pad":5})
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('match_prediction_with_economy.png')
    plt.show()
    
    # Print summary
    print(f"\nMatch Prediction: {team1} vs {team2}")
    print(f"Predicted Winner: {predicted_winner}")
    print(f"Win Probabilities: {team1}: {team1_prob:.1%}, {team2}: {team2_prob:.1%}")
    print(f"Prediction Confidence: {confidence:.1%}")

def analyze_upcoming_matches():
    """Fetch and analyze all upcoming matches."""
    # Fetch upcoming matches
    upcoming = fetch_upcoming_matches()
    
    if not upcoming:
        print("No upcoming matches found.")
        return
    
    print(f"Found {len(upcoming)} upcoming matches.")
    
    # Try to load existing model
    try:
        model = load_model('valorant_model.h5')
        with open('feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        print("Loaded existing model for predictions.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train a model first.")
        return
    
    # Make predictions for each match
    predictions = []
    for match in tqdm(upcoming, desc="Predicting matches"):
        if 'teams' in match and len(match['teams']) >= 2:
            team1_name = match['teams'][0].get('name', '')
            team2_name = match['teams'][1].get('name', '')
            
            if team1_name and team2_name:
                prediction = predict_match_with_optimized_model(team1_name, team2_name, model, scaler, feature_names)
                
                if prediction:
                    # Add match details to prediction
                    prediction['match_id'] = match.get('id', '')
                    
                    # Handle both string and dictionary event formats
                    if isinstance(match.get('event'), dict):
                        prediction['event'] = match.get('event', {}).get('name', '')
                    else:
                        prediction['event'] = str(match.get('event', ''))
                        
                    prediction['date'] = match.get('date', '')
                    
                    predictions.append(prediction)
    
    # Save predictions to file
    if predictions:
        df = pd.DataFrame(predictions)
        df.to_csv('upcoming_match_predictions.csv', index=False)
        
        print(f"Made predictions for {len(predictions)} matches.")
        print("Results saved to 'upcoming_match_predictions.csv'")
        
        # Visualize a few predictions
        for i, pred in enumerate(predictions[:3]):
            visualize_prediction_with_economy(pred)
            if i < len(predictions) - 1:
                plt.figure()  # Create a new figure for the next prediction


def collect_all_team_data(include_player_stats=True, include_economy=True, include_maps=False, verbose=False):
    """Collect data for all teams to use in backtesting, with improved economy, player stats, and map data."""
    print("\n========================================================")
    print("COLLECTING TEAM DATA WITH ECONOMY AND PLAYER STATISTICS")
    print("========================================================")
    print(f"Including player stats: {include_player_stats}")
    print(f"Including economy data: {include_economy}")
    print(f"Including map data: {include_maps}")
    print(f"Using team name fallbacks for missing team tags: enabled")
    
    # Fetch all teams
    teams_response = requests.get(f"{API_URL}/teams?limit=500")
    if teams_response.status_code != 200:
        print(f"Error fetching teams: {teams_response.status_code}")
        return {}
    
    teams_data = teams_response.json()
    
    if 'data' not in teams_data:
        print("No teams data found.")
        return {}
    
    # Select teams based on ranking or use a limited sample for faster processing
    top_teams = []
    for team in teams_data['data']:
        if 'ranking' in team and team['ranking'] and team['ranking'] <= 50:
            top_teams.append(team)
    
    # If no teams with rankings were found, just take the first 20 teams
    if not top_teams:
        print("No teams with rankings found. Using the first 20 teams instead.")
        top_teams = teams_data['data'][:5]
    
    print(f"Selected {len(top_teams)} teams for data collection.")
    
    # Collect match data for each team
    team_data_collection = {}
    
    # Track counts for data availability
    economy_data_count = 0
    player_stats_count = 0
    map_stats_count = 0
    teams_with_tag = 0
    teams_using_fallback = 0
    
    for team in tqdm(top_teams, desc="Collecting team data"):
        team_id = team['id']
        team_name = team['name']
        
        print(f"\n{'='*50}")
        print(f"Processing team: {team_name} (ID: {team_id})")
        print(f"{'='*50}")
        
        # Fetch team details to get team tag for economy matching
        team_details, team_tag = fetch_team_details(team_id)
        if team_tag:
            teams_with_tag += 1
            if verbose:
                print(f"Team tag found: {team_tag}")
        else:
            teams_using_fallback += 1
            print(f"No team tag found for {team_name}. Will use team name as fallback.")
        
        team_history = fetch_team_match_history(team_id)
        if not team_history:
            print(f"No match history found for {team_name}")
            continue
            
        team_matches = parse_match_data(team_history, team_name)
        
        # Skip teams with no match data
        if not team_matches:
            print(f"No parsed match data found for {team_name}")
            continue
            
        # Add team tag and team name to all matches for economy data extraction
        for match in team_matches:
            match['team_tag'] = team_tag
            match['team_id'] = team_id
            match['team_name'] = team_name  # Always set team_name for fallback purposes
        
        # Fetch player stats if requested
        team_player_stats = None
        if include_player_stats:
            if verbose:
                print(f"\n{'*'*30}")
                print(f"Fetching player stats for team: {team_name}")
                print(f"{'*'*30}")
            
            team_player_stats = fetch_team_player_stats(team_id)
            
            if team_player_stats:
                player_stats_count += 1
                print(f"Found {len(team_player_stats)} players for {team_name}")
                for i, player in enumerate(team_player_stats[:3]):  # Show first 3 players
                    print(f"  {i+1}. {player.get('player', 'Unknown')} - Rating: {player.get('stats', {}).get('rating', 'N/A')}")
                if len(team_player_stats) > 3:
                    print(f"  ... and {len(team_player_stats) - 3} more players")
            else:
                print(f"No player stats found for {team_name}")
                
        # Calculate team stats with the right method based on flags
        if include_economy:
            print(f"\nUsing enhanced economy stats calculation for {team_name}")
            print(f"Team identifier: {team_tag or team_name} (using {'tag' if team_tag else 'name as fallback'})")
            team_stats = calculate_team_stats_with_economy(team_matches, team_player_stats)
            
            # Check if we actually got economy data
            if 'pistol_win_rate' in team_stats and team_stats['pistol_win_rate'] > 0:
                economy_data_count += 1
                print(f"Successfully collected economy data for {team_name}")
                print(f"  - Pistol win rate: {team_stats['pistol_win_rate']:.2f}")
                print(f"  - Eco win rate: {team_stats['eco_win_rate']:.2f}")
                print(f"  - Full buy win rate: {team_stats['full_buy_win_rate']:.2f}")
                
                # Log whether we used tag or fallback name
                if not team_tag:
                    print(f"  - Successfully used team name fallback for economy data")
            else:
                print(f"No valid economy data found for {team_name}")
        else:
            print(f"\nUsing standard team stats calculation for {team_name}")
            team_stats = calculate_team_stats(team_matches, team_player_stats)
        
        # Store team tag and name in the stats
        team_stats['team_tag'] = team_tag
        team_stats['team_name'] = team_name
        team_stats['team_id'] = team_id
        team_stats['used_name_fallback'] = not bool(team_tag)  # Track if we used fallback

        # Fetch and process map statistics if requested
        if include_maps:
            print(f"\nFetching map statistics for {team_name}")
            map_stats = fetch_team_map_statistics(team_id)
            
            if map_stats:
                map_stats_count += 1
                team_stats['map_statistics'] = map_stats
                print(f"Successfully collected map statistics for {team_name} ({len(map_stats)} maps)")
                
                # Print a sample of the map stats if verbose
                if verbose:
                    print("\nSample map statistics:")
                    maps_to_show = list(map_stats.keys())[:3]  # Show stats for first 3 maps
                    for map_name in maps_to_show:
                        print(f"  {map_name}:")
                        print(f"    Win Rate: {map_stats[map_name]['win_percentage']*100:.2f}%")
                        print(f"    ATK Win Rate: {map_stats[map_name]['atk_win_rate']*100:.2f}%")
                        print(f"    DEF Win Rate: {map_stats[map_name]['def_win_rate']*100:.2f}%")
                    
                    if len(map_stats) > 3:
                        print(f"  ... and {len(map_stats) - 3} more maps")
            else:
                print(f"No map statistics found for {team_name}")

        # Extract additional analyses
        team_map_performance = extract_map_performance(team_matches)
        team_tournament_performance = extract_tournament_performance(team_matches)
        team_performance_trends = analyze_performance_trends(team_matches)
        team_opponent_quality = analyze_opponent_quality(team_matches, team_id)
        
        # Add these metrics to the team stats
        team_stats['map_performance'] = team_map_performance
        team_stats['tournament_performance'] = team_tournament_performance
        team_stats['performance_trends'] = team_performance_trends
        team_stats['opponent_quality'] = team_opponent_quality
        
        # Add team matches to stats object
        team_stats['matches'] = team_matches
        
        # Add to collection
        team_data_collection[team_name] = team_stats
    
    print(f"\n===================================================")
    print(f"TEAM DATA COLLECTION SUMMARY")
    print(f"===================================================")
    print(f"Collected data for {len(team_data_collection)} teams:")
    print(f"  - Teams with economy data: {economy_data_count}")
    print(f"  - Teams with player stats: {player_stats_count}")
    print(f"  - Teams with map stats: {map_stats_count}")
    print(f"  - Teams with both economy and player stats: {min(economy_data_count, player_stats_count)}")
    print(f"\nTeam identification summary:")
    print(f"  - Teams with tags: {teams_with_tag}")
    print(f"  - Teams using name fallback: {teams_using_fallback}")
    
    # Calculate how many teams with no tag still got economy data
    fallback_success_count = sum(1 for name, data in team_data_collection.items() 
                                if data.get('used_name_fallback', False) and 
                                   'pistol_win_rate' in data and data['pistol_win_rate'] > 0)
    
    if teams_using_fallback > 0:
        fallback_success_rate = fallback_success_count / teams_using_fallback * 100
        print(f"  - Teams successfully using name fallback: {fallback_success_count}/{teams_using_fallback} ({fallback_success_rate:.1f}%)")
    
    return team_data_collection




def backtest_model(cutoff_date, bet_amount=100, confidence_threshold=0.6):
    """Backtest the model using historical data split by date with economy and player stats."""
    # Get all teams and matches with economy data
    print("Collecting team data for backtesting with economy and player statistics...")
    team_data_collection = collect_all_team_data(include_player_stats=True, include_economy=True, include_maps=True)
    
    if not team_data_collection:
        print("Failed to collect team data. Aborting backtesting.")
        return None
    
    # Track what percentage of teams have economy data
    teams_with_economy = sum(1 for team_data in team_data_collection.values() 
                           if 'pistol_win_rate' in team_data and team_data['pistol_win_rate'] > 0)
    
    print(f"Collected data for {len(team_data_collection)} teams, {teams_with_economy} with economy data.")
    
    # Split matches into before and after cutoff date
    train_matches = []
    test_matches = []
    
    cutoff = datetime.strptime(cutoff_date, '%Y/%m/%d')
    print(f"Using cutoff date: {cutoff.strftime('%Y/%m/%d')}")
    
    for team_name, team_data in team_data_collection.items():
        for match in team_data.get('matches', []):
            # Parse date safely
            match_date_str = match.get('date', '')
            try:
                # Handle various date formats
                if '/' in match_date_str:
                    match_date = datetime.strptime(match_date_str, '%Y/%m/%d')
                elif '-' in match_date_str:
                    match_date = datetime.strptime(match_date_str, '%Y-%m-%d')
                else:
                    # If unparseable, skip this match
                    continue
                
                # Add team tag to the match if available
                if 'team_tag' not in match and 'team_tag' in team_data:
                    match['team_tag'] = team_data['team_tag']
                
                if match_date < cutoff:
                    train_matches.append(match)
                else:
                    test_matches.append(match)
            except (ValueError, TypeError):
                # Skip matches with invalid dates
                continue
    
    print(f"Split matches: {len(train_matches)} for training, {len(test_matches)} for testing")
    
    if len(train_matches) < 10 or len(test_matches) < 10:
        print("Not enough matches for reliable backtesting.")
        return None
    
    # Train model on older matches with economy and player features
    print("Building training dataset with economy and player features...")
    X_train, y_train = build_training_dataset_with_economy(team_data_collection)
    
    if len(X_train) < 10:
        print("Not enough training samples. Try using an earlier cutoff date.")
        return None
    
    print(f"Training model with {len(X_train)} samples...")
    model, scaler, feature_names = train_model(X_train, y_train)
    
    if not model:
        print("Failed to train model. Aborting backtesting.")
        return None
    
    # Print economy and player features used
    if feature_names:
        economy_features = [f for f in feature_names if any(term in f.lower() for term in 
                                                     ['eco', 'pistol', 'buy', 'economy'])]
        player_features = [f for f in feature_names if any(term in f.lower() for term in 
                                                    ['rating', 'acs', 'kd', 'adr', 'headshot',
                                                     'clutch', 'aces', 'first_blood'])]
        
        print(f"Model using {len(economy_features)} economy features and {len(player_features)} player features")
    
    # Test on newer matches
    print(f"Testing model on {len(test_matches)} matches...")
    correct_predictions = 0
    total_profit = 0
    total_bets = 0
    correct_high_conf = 0
    total_high_conf = 0
    
    results_data = []
    
    for match in tqdm(test_matches, desc="Evaluating matches"):
        team1_name = match.get('team_name')
        team2_name = match.get('opponent_name')
        
        # Skip matches where we don't have both teams
        if team1_name not in team_data_collection or team2_name not in team_data_collection:
            continue
        
        # Get prediction with full features
        prediction = predict_match_with_optimized_model(team1_name, team2_name, model, scaler, feature_names)
        
        if prediction:
            team1_win_prob = prediction['team1_win_probability']
            model_confidence = prediction['confidence']
            predicted_winner = prediction['predicted_winner']
            actual_winner = team1_name if match.get('team_won', True) else team2_name
            
            # Record all predictions
            match_result = {
                'date': match.get('date', ''),
                'team1': team1_name,
                'team2': team2_name,
                'predicted_winner': predicted_winner,
                'actual_winner': actual_winner,
                'confidence': model_confidence,
                'correct': predicted_winner == actual_winner,
                'economy_features_used': len(economy_features) if 'economy_features' in locals() else 'Unknown',
                'player_features_used': len(player_features) if 'player_features' in locals() else 'Unknown'
            }
            results_data.append(match_result)
            
            # Only bet if confidence is high enough
            if model_confidence >= confidence_threshold:
                total_bets += 1
                total_high_conf += 1
                
                if predicted_winner == actual_winner:
                    correct_predictions += 1
                    correct_high_conf += 1
                    total_profit += bet_amount
                else:
                    total_profit -= bet_amount
            
    # Calculate metrics
    overall_accuracy = sum(1 for r in results_data if r['correct']) / len(results_data) if results_data else 0
    high_conf_accuracy = correct_high_conf / total_high_conf if total_high_conf > 0 else 0
    betting_accuracy = correct_predictions / total_bets if total_bets > 0 else 0
    roi = total_profit / (total_bets * bet_amount) if total_bets > 0 else 0
    
    # Save detailed results to CSV with added information about economy and player features
    if results_data:
        df = pd.DataFrame(results_data)
        df.to_csv('backtesting_results_with_economy.csv', index=False)
        print(f"Detailed results saved to 'backtesting_results_with_economy.csv'")
    
    results = {
        'overall_accuracy': overall_accuracy,
        'high_conf_accuracy': high_conf_accuracy,
        'betting_accuracy': betting_accuracy,
        'roi': roi,
        'total_profit': total_profit,
        'total_predictions': len(results_data),
        'total_bets': total_bets,
        'bet_amount': bet_amount,
        'confidence_threshold': confidence_threshold,
        'cutoff_date': cutoff_date,
        'economy_features_used': len(economy_features) if 'economy_features' in locals() else 'Unknown',
        'player_features_used': len(player_features) if 'player_features' in locals() else 'Unknown'
    }
    
    return results



# Update export_prediction_data function to include player stats
def export_prediction_data_with_economy(prediction, team1_stats, team2_stats, filename=None):
    """
    Export the prediction data including map statistics to a JSON file.
    
    Args:
        prediction (dict): The prediction result
        team1_stats (dict): Statistics for team1
        team2_stats (dict): Statistics for team2
        filename (str, optional): Custom filename, defaults to automatic name based on teams and date
    
    Returns:
        str: Path to the saved JSON file
    """
    if not prediction:
        print("No prediction data to export.")
        return None
    
    # Create directory if it doesn't exist
    os.makedirs("betting_predictions", exist_ok=True)
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        team1_name = prediction['team1'].replace(" ", "_")
        team2_name = prediction['team2'].replace(" ", "_")
        filename = f"betting_predictions/{team1_name}_vs_{team2_name}_{timestamp}.json"
    
    # Clean up stats to make them JSON serializable
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() if k != 'matches'}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    # Create comprehensive export data
    export_data = {
        "prediction": {
            "match": f"{prediction['team1']} vs {prediction['team2']}",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_winner": prediction['predicted_winner'],
            "team1_win_probability": prediction['team1_win_probability'],
            "team2_win_probability": prediction['team2_win_probability'],
            "confidence": prediction['confidence']
        },
        "team_stats": {
            prediction['team1']: clean_for_json(team1_stats),
            prediction['team2']: clean_for_json(team2_stats)
        },
        "analysis": {
            "key_factors": [],  # Will be populated below
            "economy_factors": [],  # Economy insights
            "map_factors": []   # Map insights
        }
    }
    
    # Add map-specific analysis
    if ('map_statistics' in team1_stats and 'map_statistics' in team2_stats and
        team1_stats['map_statistics'] and team2_stats['map_statistics']):
        
        # Find common maps
        team1_maps = set(team1_stats['map_statistics'].keys())
        team2_maps = set(team2_stats['map_statistics'].keys())
        common_maps = team1_maps.intersection(team2_maps)
        
        # Calculate significant map advantages
        for map_name in common_maps:
            t1_map = team1_stats['map_statistics'][map_name]
            t2_map = team2_stats['map_statistics'][map_name]
            
            # Win rate comparison
            win_diff = t1_map['win_percentage'] - t2_map['win_percentage']
            if abs(win_diff) > 0.15:  # 15% threshold for significant advantage
                better_team = prediction['team1'] if win_diff > 0 else prediction['team2']
                worse_team = prediction['team2'] if win_diff > 0 else prediction['team1']
                win_pct_better = max(t1_map['win_percentage'], t2_map['win_percentage']) * 100
                win_pct_worse = min(t1_map['win_percentage'], t2_map['win_percentage']) * 100
                
                export_data['analysis']['map_factors'].append(
                    f"{better_team} has a significant win rate advantage on {map_name} ({win_pct_better:.1f}% vs {win_pct_worse:.1f}%)"
                )
            
            # Side performance comparison
            if 'side_preference' in t1_map and 'side_preference' in t2_map:
                t1_side = t1_map['side_preference']
                t2_side = t2_map['side_preference']
                t1_strength = t1_map.get('side_preference_strength', 0) * 100
                t2_strength = t2_map.get('side_preference_strength', 0) * 100
                
                # If teams have different side preferences with significant strength
                if t1_side != t2_side and (t1_strength > 10 or t2_strength > 10):
                    export_data['analysis']['map_factors'].append(
                        f"On {map_name}, {prediction['team1']} prefers {t1_side} ({t1_strength:.1f}% better) while {prediction['team2']} prefers {t2_side} ({t2_strength:.1f}% better)"
                    )
            
            # Overtime performance
            if ('overtime_stats' in t1_map and 'overtime_stats' in t2_map and
                t1_map['overtime_stats']['matches'] > 1 and t2_map['overtime_stats']['matches'] > 1):
                
                t1_ot_rate = t1_map['overtime_stats']['win_rate']
                t2_ot_rate = t2_map['overtime_stats']['win_rate']
                ot_diff = t1_ot_rate - t2_ot_rate
                
                if abs(ot_diff) > 0.25:  # 25% threshold for significant OT advantage
                    better_ot_team = prediction['team1'] if ot_diff > 0 else prediction['team2']
                    export_data['analysis']['map_factors'].append(
                        f"{better_ot_team} performs significantly better in overtime on {map_name}"
                    )
            
            # Agent composition comparison
            if ('agent_compositions' in t1_map and 'agent_compositions' in t2_map and
                t1_map['agent_compositions'] and t2_map['agent_compositions']):
                
                t1_agents = set(t1_map.get('most_played_agents', []))
                t2_agents = set(t2_map.get('most_played_agents', []))
                
                # Calculate agent overlap percentage
                common_agents = t1_agents.intersection(t2_agents)
                total_agents = t1_agents.union(t2_agents)
                
                if total_agents:
                    overlap_pct = (len(common_agents) / len(total_agents)) * 100
                    
                    if overlap_pct < 40:  # Low agent overlap
                        export_data['analysis']['map_factors'].append(
                            f"Teams use very different agent compositions on {map_name} (only {overlap_pct:.1f}% overlap)"
                        )
        
        # Add strongest/weakest map comparison
        if ('strongest_maps' in team1_stats and 'strongest_maps' in team2_stats and
            'weakest_maps' in team1_stats and 'weakest_maps' in team2_stats):
            
            t1_strong = set(team1_stats['strongest_maps'])
            t2_weak = set(team2_stats['weakest_maps'])
            
            # Find maps where team1's strength overlaps with team2's weakness
            advantage_maps = t1_strong.intersection(t2_weak)
            
            if advantage_maps:
                export_data['analysis']['map_factors'].append(
                    f"{prediction['team1']}'s strongest maps ({', '.join(advantage_maps)}) are {prediction['team2']}'s weakest maps"
                )
            
            # And vice versa
            t2_strong = set(team2_stats['strongest_maps'])
            t1_weak = set(team1_stats['weakest_maps'])
            
            disadvantage_maps = t2_strong.intersection(t1_weak)
            
            if disadvantage_maps:
                export_data['analysis']['map_factors'].append(
                    f"{prediction['team2']}'s strongest maps ({', '.join(disadvantage_maps)}) are {prediction['team1']}'s weakest maps"
                )
    
    # Add economy analysis - reusing logic from original function
    if ('pistol_win_rate' in team1_stats and 'pistol_win_rate' in team2_stats):
        # Compare pistol win rates
        t1_pistol = team1_stats.get('pistol_win_rate', 0)
        t2_pistol = team2_stats.get('pistol_win_rate', 0)
        if abs(t1_pistol - t2_pistol) > 0.1:  # If significant difference
            better_team = prediction['team1'] if t1_pistol > t2_pistol else prediction['team2']
            export_data['analysis']['economy_factors'].append(
                f"{better_team} has a significantly better pistol round win rate ({max(t1_pistol, t2_pistol):.1%} vs {min(t1_pistol, t2_pistol):.1%})"
            )
        
        # Compare eco round performance
        t1_eco = team1_stats.get('eco_win_rate', 0)
        t2_eco = team2_stats.get('eco_win_rate', 0)
        if abs(t1_eco - t2_eco) > 0.15:  # If significant difference
            better_eco = prediction['team1'] if t1_eco > t2_eco else prediction['team2']
            export_data['analysis']['economy_factors'].append(
                f"{better_eco} performs better in eco rounds ({max(t1_eco, t2_eco):.1%} vs {min(t1_eco, t2_eco):.1%})"
            )
            
        # Compare full buy performance
        t1_full = team1_stats.get('full_buy_win_rate', 0)
        t2_full = team2_stats.get('full_buy_win_rate', 0)
        if abs(t1_full - t2_full) > 0.1:  # If significant difference
            better_full = prediction['team1'] if t1_full > t2_full else prediction['team2']
            export_data['analysis']['economy_factors'].append(
                f"{better_full} has stronger performance in full buy rounds ({max(t1_full, t2_full):.1%} vs {min(t1_full, t2_full):.1%})"
            )
            
        # Economy efficiency comparison
        t1_eff = team1_stats.get('economy_efficiency', 0)
        t2_eff = team2_stats.get('economy_efficiency', 0)
        if abs(t1_eff - t2_eff) > 0.1:  # If significant difference
            better_eff = prediction['team1'] if t1_eff > t2_eff else prediction['team2']
            export_data['analysis']['economy_factors'].append(
                f"{better_eff} shows better overall economy management efficiency ({max(t1_eff, t2_eff):.2f} vs {min(t1_eff, t2_eff):.2f})"
            )
    
    # Add key matchup factors
    # Win rate difference
    win_rate_diff = abs(team1_stats.get('win_rate', 0) - team2_stats.get('win_rate', 0))
    if win_rate_diff > 0.15:
        better_team = prediction['team1'] if team1_stats.get('win_rate', 0) > team2_stats.get('win_rate', 0) else prediction['team2']
        export_data['analysis']['key_factors'].append(
            f"{better_team} has a significantly better overall win rate ({win_rate_diff:.1%} difference)"
        )
    
    # Recent form
    form_diff = abs(team1_stats.get('recent_form', 0) - team2_stats.get('recent_form', 0))
    if form_diff > 0.2:
        better_form = prediction['team1'] if team1_stats.get('recent_form', 0) > team2_stats.get('recent_form', 0) else prediction['team2']
        export_data['analysis']['key_factors'].append(
            f"{better_form} has much better recent form ({form_diff:.1%} difference)"
        )
    
    # Player rating comparison
    if ('avg_player_rating' in team1_stats and 'avg_player_rating' in team2_stats):
        rating_diff = abs(team1_stats.get('avg_player_rating', 0) - team2_stats.get('avg_player_rating', 0))
        if rating_diff > 0.2:
            better_rated = prediction['team1'] if team1_stats.get('avg_player_rating', 0) > team2_stats.get('avg_player_rating', 0) else prediction['team2']
            export_data['analysis']['key_factors'].append(
                f"{better_rated} has higher-rated players ({rating_diff:.2f} rating difference)"
            )
    
    # Head-to-head analysis
    h2h_win_rate = 0.5  # Default to even
    if 'opponent_stats' in team1_stats and prediction['team2'] in team1_stats['opponent_stats']:
        h2h_win_rate = team1_stats['opponent_stats'][prediction['team2']].get('win_rate', 0.5)
        h2h_matches = team1_stats['opponent_stats'][prediction['team2']].get('matches', 0)
        
        if h2h_matches >= 3 and abs(h2h_win_rate - 0.5) > 0.15:
            h2h_better = prediction['team1'] if h2h_win_rate > 0.5 else prediction['team2']
            export_data['analysis']['key_factors'].append(
                f"{h2h_better} has a significant head-to-head advantage ({max(h2h_win_rate, 1-h2h_win_rate):.1%} win rate in {h2h_matches} matches)"
            )
    
    # Save to JSON file
    try:
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Prediction data exported to {filename}")
        return filename
    except Exception as e:
        print(f"Error exporting prediction data: {e}")
        return None

def create_deep_learning_model_with_economy(input_dim):
    """Create an enhanced deep learning model for match prediction with player stats and economy data."""
    # Define inputs
    inputs = Input(shape=(input_dim,))
    
    # First layer - shared feature processing
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Second layer - deeper processing
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Player stats pathway with additional neurons
    x = Dense(96, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    # Economy-specific pathway with expanded capacity
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    # Combined pathway
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(loss='binary_crossentropy', 
                 optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),  # Add clipnorm
                 metrics=['accuracy'])
    
    
    # Print model summary to see the expanded architecture
    print("\nModel Architecture:")
    model.summary()
    
    return model

def display_prediction_results(prediction, team1_stats, team2_stats):
    """
    Display detailed prediction results in a formatted console output.
    
    Args:
        prediction (dict): The prediction result
        team1_stats (dict): Statistics for team1
        team2_stats (dict): Statistics for team2
    """
    if not prediction:
        print("No prediction to display.")
        return
    
    team1 = prediction['team1']
    team2 = prediction['team2']
    winner = prediction['predicted_winner']
    team1_prob = prediction['team1_win_probability'] * 100
    team2_prob = prediction['team2_win_probability'] * 100
    
    # Extract matches count safely
    team1_matches = team1_stats['matches'] if isinstance(team1_stats['matches'], int) else len(team1_stats['matches'])
    team2_matches = team2_stats['matches'] if isinstance(team2_stats['matches'], int) else len(team2_stats['matches'])
    
    # Get head-to-head stats if available
    h2h_matches = 0
    team1_h2h_wins = 0
    team2_h2h_wins = 0
    team1_h2h_rate = 0
    team2_h2h_rate = 0
    
    if 'opponent_stats' in team1_stats and team2 in team1_stats['opponent_stats']:
        h2h = team1_stats['opponent_stats'][team2]
        h2h_matches = h2h.get('matches', 0)
        team1_h2h_rate = h2h.get('win_rate', 0)
        team1_h2h_wins = int(h2h_matches * team1_h2h_rate)
        team2_h2h_wins = h2h_matches - team1_h2h_wins
        team2_h2h_rate = 1 - team1_h2h_rate
    
    # Calculate additional metrics
    team1_score = team1_stats.get('avg_score', 0)
    team2_score = team2_stats.get('avg_score', 0)
    team1_score_diff = team1_stats.get('score_differential', 0)
    team2_score_diff = team2_stats.get('score_differential', 0)
    
    # Extract player stats if available
    team1_player_stats = team1_stats.get('player_stats', {})
    team2_player_stats = team2_stats.get('player_stats', {})
    

    # Print formatted output
    width = 70  # Total width of the display
    
    print("\n" + "=" * width)
    print(f"{' MATCH PREDICTION ':=^{width}}")
    print(f"Team 1: {team1}")
    print(f"Team 2: {team2}")
    print()
    print(f"Predicted Winner: {winner}")
    print(f"Win Probability: {team1_prob:.2f}% vs {team2_prob:.2f}%")
    print()
    
    # Team Stats Comparison
    print(f"{' Team Stats Comparison ':—^{width}}")
    print(f"{'Statistic':<20} {team1:<25} {team2}")
    print("-" * width)
    print(f"{'Matches':<20} {team1_matches:<25} {team2_matches}")
    print(f"{'Wins':<20} {team1_stats.get('wins', 0):<25} {team2_stats.get('wins', 0)}")
    print(f"{'Win Rate':<20} {team1_stats.get('win_rate', 0)*100:.2f}%{'':<20} {team2_stats.get('win_rate', 0)*100:.2f}%")
    print(f"{'Avg Score':<20} {team1_score:<25.2f} {team2_score:.2f}")
    print(f"{'Score Diff':<20} {team1_score_diff:<25.2f} {team2_score_diff:.2f}")
    print(f"{'Recent Form':<20} {team1_stats.get('recent_form', 0)*100:.2f}%{'':<20} {team2_stats.get('recent_form', 0)*100:.2f}%")
    
    # Display player stats
    print()
    print(f"{' Player Stats Comparison ':—^{width}}")
    print(f"{'Statistic':<20} {team1:<25} {team2}")
    print("-" * width)
    print(f"{'Avg Rating':<20} {team1_stats.get('avg_player_rating', 0):<25.2f} {team2_stats.get('avg_player_rating', 0):.2f}")
    print(f"{'Avg ACS':<20} {team1_stats.get('avg_player_acs', 0):<25.2f} {team2_stats.get('avg_player_acs', 0):.2f}")
    print(f"{'Avg K/D':<20} {team1_stats.get('avg_player_kd', 0):<25.2f} {team2_stats.get('avg_player_kd', 0):.2f}")
    print(f"{'Avg KAST':<20} {team1_stats.get('avg_player_kast', 0)*100:<25.2f}% {team2_stats.get('avg_player_kast', 0)*100:.2f}%")
    print(f"{'Avg ADR':<20} {team1_stats.get('avg_player_adr', 0):<25.2f} {team2_stats.get('avg_player_adr', 0):.2f}")
    print(f"{'FK/FD Ratio':<20} {team1_stats.get('fk_fd_ratio', 0):<25.2f} {team2_stats.get('fk_fd_ratio', 0):.2f}")
    
    # Display star players
    if team1_player_stats and team2_player_stats:
        print()
        print(f"{' Star Players ':—^{width}}")
        print(f"{team1}: {team1_player_stats.get('star_player_name', 'Unknown')} (Rating: {team1_player_stats.get('star_player_rating', 0):.2f})")
        print(f"{team2}: {team2_player_stats.get('star_player_name', 'Unknown')} (Rating: {team2_player_stats.get('star_player_rating', 0):.2f})")
    

    # Head-to-Head Stats
    print()
    print(f"{' Head-to-Head Stats ':—^{width}}")
    print(f"Total H2H Matches: {h2h_matches}")
    print(f"{team1} H2H Wins: {team1_h2h_wins}")
    print(f"{team2} H2H Wins: {team2_h2h_wins}")
    print(f"{team1} H2H Win Rate: {team1_h2h_rate*100:.2f}%")
    print(f"{team2} H2H Win Rate: {team2_h2h_rate*100:.2f}%")
    
    # Map Stats if available
    if 'map_stats' in team1_stats and 'map_stats' in team2_stats:
        common_maps = set(team1_stats['map_stats'].keys()) & set(team2_stats['map_stats'].keys())
        if common_maps:
            print()
            print(f"{' Map Win Rates ':—^{width}}")
            print(f"{'Map':<20} {team1:<25} {team2}")
            print("-" * width)
            
            for map_name in common_maps:
                t1_map_wr = team1_stats['map_stats'][map_name].get('win_rate', 0) * 100
                t2_map_wr = team2_stats['map_stats'][map_name].get('win_rate', 0) * 100
                print(f"{map_name:<20} {t1_map_wr:.2f}%{'':<20} {t2_map_wr:.2f}%")
    
    # Key Advantages
    advantages = []
    
    # Win rate advantage
    wr_diff = team1_stats.get('win_rate', 0) - team2_stats.get('win_rate', 0)
    if abs(wr_diff) > 0.1:
        better_team = team1 if wr_diff > 0 else team2
        advantages.append(f"{better_team} has a better overall win rate ({abs(wr_diff)*100:.1f}% difference)")
    
    # Recent form advantage
    form_diff = team1_stats.get('recent_form', 0) - team2_stats.get('recent_form', 0)
    if abs(form_diff) > 0.15:
        better_form = team1 if form_diff > 0 else team2
        advantages.append(f"{better_form} has better recent form ({abs(form_diff)*100:.1f}% difference)")
    
    # Player rating advantage
    rating_diff = team1_stats.get('avg_player_rating', 0) - team2_stats.get('avg_player_rating', 0)
    if abs(rating_diff) > 0.1:
        better_rated = team1 if rating_diff > 0 else team2
        advantages.append(f"{better_rated} has higher-rated players ({abs(rating_diff):.2f} rating difference)")
    
    # Star player advantage
    star_diff = team1_stats.get('star_player_rating', 0) - team2_stats.get('star_player_rating', 0)
    if abs(star_diff) > 0.2:
        better_star = team1 if star_diff > 0 else team2
        advantages.append(f"{better_star} has a stronger star player ({abs(star_diff):.2f} rating difference)")
    
    # First kill ratio advantage
    fk_fd_diff = team1_stats.get('fk_fd_ratio', 0) - team2_stats.get('fk_fd_ratio', 0) 
    if abs(fk_fd_diff) > 0.5:
        better_fk = team1 if fk_fd_diff > 0 else team2
        advantages.append(f"{better_fk} has better first kill performance ({abs(fk_fd_diff):.2f} FK/FD difference)")

    # H2H advantage
    if h2h_matches >= 3 and abs(team1_h2h_rate - 0.5) > 0.1:
        h2h_better = team1 if team1_h2h_rate > 0.5 else team2
        advantages.append(f"{h2h_better} has a head-to-head advantage ({max(team1_h2h_rate, team2_h2h_rate)*100:.1f}% win rate)")
    
    if advantages:
        print()
        print(f"{' Key Advantages ':—^{width}}")
        for adv in advantages:
            print(f"• {adv}")
    
    print("=" * width + "\n")

def display_prediction_results_with_economy(prediction, team1_stats, team2_stats):
    """Display detailed prediction results in a formatted console output, including economy data."""
    # First use the original display function
    display_prediction_results(prediction, team1_stats, team2_stats)
    
    # Now add economy-specific display
    if not prediction:
        return
    
    team1 = prediction['team1']
    team2 = prediction['team2']
    
    width = 70  # Total width of the display
    
    # Check if economy data is available
    if ('pistol_win_rate' in team1_stats or 'eco_win_rate' in team1_stats or 
        'pistol_win_rate' in team2_stats or 'eco_win_rate' in team2_stats):
        
        print("\n" + "=" * width)
        print(f"{' Economy Stats Comparison ':=^{width}}")
        print(f"{'Statistic':<20} {team1:<25} {team2}")
        print("-" * width)
        
        # Pistol rounds
        t1_pistol = team1_stats.get('pistol_win_rate', 0) * 100
        t2_pistol = team2_stats.get('pistol_win_rate', 0) * 100
        print(f"{'Pistol Win Rate':<20} {t1_pistol:<25.2f}% {t2_pistol:.2f}%")
        
        # Eco rounds
        t1_eco = team1_stats.get('eco_win_rate', 0) * 100
        t2_eco = team2_stats.get('eco_win_rate', 0) * 100
        print(f"{'Eco Win Rate':<20} {t1_eco:<25.2f}% {t2_eco:.2f}%")
        
        # Semi-eco rounds
        t1_semi_eco = team1_stats.get('semi_eco_win_rate', 0) * 100
        t2_semi_eco = team2_stats.get('semi_eco_win_rate', 0) * 100
        print(f"{'Semi-Eco Win Rate':<20} {t1_semi_eco:<25.2f}% {t2_semi_eco:.2f}%")
        
        # Semi-buy rounds
        t1_semi_buy = team1_stats.get('semi_buy_win_rate', 0) * 100
        t2_semi_buy = team2_stats.get('semi_buy_win_rate', 0) * 100
        print(f"{'Semi-Buy Win Rate':<20} {t1_semi_buy:<25.2f}% {t2_semi_buy:.2f}%")
        
        # Full-buy rounds
        t1_full_buy = team1_stats.get('full_buy_win_rate', 0) * 100
        t2_full_buy = team2_stats.get('full_buy_win_rate', 0) * 100
        print(f"{'Full-Buy Win Rate':<20} {t1_full_buy:<25.2f}% {t2_full_buy:.2f}%")
        
        # Economy efficiency
        t1_efficiency = team1_stats.get('economy_efficiency', 0)
        t2_efficiency = team2_stats.get('economy_efficiency', 0)
        print(f"{'Economy Efficiency':<20} {t1_efficiency:<25.3f} {t2_efficiency:.3f}")
        
        # Low economy win rate
        t1_low_econ = team1_stats.get('low_economy_win_rate', 0) * 100
        t2_low_econ = team2_stats.get('low_economy_win_rate', 0) * 100
        print(f"{'Low Econ Win Rate':<20} {t1_low_econ:<25.2f}% {t2_low_econ:.2f}%")
        
        # High economy win rate
        t1_high_econ = team1_stats.get('high_economy_win_rate', 0) * 100
        t2_high_econ = team2_stats.get('high_economy_win_rate', 0) * 100
        print(f"{'High Econ Win Rate':<20} {t1_high_econ:<25.2f}% {t2_high_econ:.2f}%")
        
        # Economy Advantages analysis
        print()
        print(f"{' Economy Advantages ':—^{width}}")
        
        advantages = []
        
        # Check pistol round advantage
        pistol_diff = abs(t1_pistol - t2_pistol)
        if pistol_diff > 10:  # 10% difference threshold
            better_pistol = team1 if t1_pistol > t2_pistol else team2
            advantages.append(f"{better_pistol} has significantly better pistol round performance ({max(t1_pistol, t2_pistol):.1f}% vs {min(t1_pistol, t2_pistol):.1f}%)")
        
        # Check eco round advantage
        eco_diff = abs(t1_eco - t2_eco)
        if eco_diff > 10:
            better_eco = team1 if t1_eco > t2_eco else team2
            advantages.append(f"{better_eco} performs better in eco rounds ({max(t1_eco, t2_eco):.1f}% vs {min(t1_eco, t2_eco):.1f}%)")
        
        # Check full-buy advantage
        full_buy_diff = abs(t1_full_buy - t2_full_buy)
        if full_buy_diff > 10:
            better_full_buy = team1 if t1_full_buy > t2_full_buy else team2
            advantages.append(f"{better_full_buy} has stronger full-buy performance ({max(t1_full_buy, t2_full_buy):.1f}% vs {min(t1_full_buy, t2_full_buy):.1f}%)")
        
        # Check economy efficiency
        efficiency_diff = abs(t1_efficiency - t2_efficiency)
        if efficiency_diff > 0.1:
            better_efficiency = team1 if t1_efficiency > t2_efficiency else team2
            advantages.append(f"{better_efficiency} has better overall economy efficiency ({max(t1_efficiency, t2_efficiency):.3f} vs {min(t1_efficiency, t2_efficiency):.3f})")
        
        # Display the advantages
        if advantages:
            for adv in advantages:
                print(f"• {adv}")
        else:
            print("• No significant economy advantages detected between the teams")
        
        print("=" * width + "\n")

def visualize_optimization_results(original_metrics, optimized_metrics):
    """Visualize the improvement from model optimization."""
    plt.figure(figsize=(15, 10))
    
    # Metrics comparison
    plt.subplot(2, 2, 1)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, original_metrics, width, label='Original Model')
    plt.bar(x + width/2, optimized_metrics, width, label='Optimized Model')
    
    plt.title('Performance Metrics Comparison', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x, metrics)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.legend(fontsize=12)
    
    # Calculate improvement percentages
    improvements = [(opt - orig) / orig * 100 for orig, opt in zip(original_metrics, optimized_metrics)]
    
    # Improvement visualization
    plt.subplot(2, 2, 2)
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    plt.bar(metrics, improvements, color=colors)
    plt.axhline(y=0, color='gray', linestyle='-')
    plt.title('Percentage Improvement in Metrics', fontsize=14)
    plt.ylabel('Improvement (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # Add improvement values
    for i, imp in enumerate(improvements):
        plt.text(i, imp + (1 if imp > 0 else -2), f"{imp:.1f}%", 
                 ha='center', va='bottom' if imp > 0 else 'top', fontsize=10)
    
    # Confusion matrix comparison
    plt.subplot(2, 2, 3)
    # [code to display confusion matrices - would need to be implemented]
    plt.title('Original Model Confusion Matrix', fontsize=14)
    
    plt.subplot(2, 2, 4)
    # [code to display confusion matrices - would need to be implemented]
    plt.title('Optimized Model Confusion Matrix', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=300)
    plt.show()

# Update main function to include player stats
def main():
    """Main function to handle command line arguments and run the program with enhanced map statistics."""
    parser = argparse.ArgumentParser(description="Valorant Match Predictor with Map Statistics")
    
    # Add command line arguments
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--optimize", action="store_true", help="Run complete model optimization pipeline")
    parser.add_argument("--predict", action="store_true", help="Predict a specific match")
    parser.add_argument("--team1", type=str, help="First team name")
    parser.add_argument("--team2", type=str, help="Second team name")
    parser.add_argument("--analyze", action="store_true", help="Analyze all upcoming matches")
    parser.add_argument("--test-teams", nargs='+', help="List of teams to use for testing")
    parser.add_argument("--backtest", action="store_true", help="Perform backtesting")
    parser.add_argument("--cutoff-date", type=str, help="Cutoff date for backtesting (YYYY/MM/DD)")
    parser.add_argument("--bet-amount", type=float, default=100, help="Bet amount for backtesting")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold for backtesting")
    parser.add_argument("--players", action="store_true", help="Include player stats in analysis")
    parser.add_argument("--economy", action="store_true", help="Include economy data in analysis")
    parser.add_argument("--maps", action="store_true", help="Include enhanced map statistics")
    parser.add_argument("--learning-curves", action="store_true", help="Generate detailed learning curves")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    parser.add_argument("--cross-validate", action="store_true", 
                      help="Train with cross-validation and create ensemble model")
    parser.add_argument("--folds", type=int, default=5, 
                      help="Number of folds for cross-validation")   
    parser.add_argument("--feature-count", type=int, default=71, 
                      help="Number of top features to use (default: 71)")

    args = parser.parse_args()
    
    # Set default behavior - include map stats if --maps is specified
    include_maps = args.maps
    
    if args.train:
        print("Training a new model with player statistics and economy data...")
        
        # Fetch all teams
        teams_response = requests.get(f"{API_URL}/teams?limit=500")
        if teams_response.status_code != 200:
            print(f"Error fetching teams: {teams_response.status_code}")
            return
        
        teams_data = teams_response.json()
        
        if 'data' not in teams_data:
            print("No teams data found.")
            return
        
        # Select teams for testing or use ranked teams
        top_teams = []
        
        if args.train and args.feature_count:
            print(f"Training model with top {args.feature_count} features...")

        # Use specific teams if provided
        if args.test_teams and len(args.test_teams) > 0:
            print(f"Using {len(args.test_teams)} specified test teams.")
            for test_team_name in args.test_teams:
                found = False
                # Try exact match first
                for team in teams_data['data']:
                    if team['name'].lower() == test_team_name.lower():
                        top_teams.append(team)
                        print(f"Found test team: {team['name']} (ID: {team['id']})")
                        found = True
                        break
                
                # If no exact match, try partial match
                if not found:
                    for team in teams_data['data']:
                        if test_team_name.lower() in team['name'].lower() or team['name'].lower() in test_team_name.lower():
                            top_teams.append(team)
                            print(f"Found partial match: {team['name']} (ID: {team['id']})")
                            found = True
                            break
                
                if not found:
                    print(f"Could not find team: {test_team_name}")
        else:
            # Select teams based on ranking
            for team in teams_data['data']:
                if 'ranking' in team and team['ranking'] and team['ranking'] <= 100:
                    top_teams.append(team)
            
            # If no teams with rankings were found, just take the first 50 teams
            if not top_teams:
                print("No teams with rankings found. Using the first 20 teams instead.")
                top_teams = teams_data['data'][:50]
        
        print(f"Selected {len(top_teams)} teams for training data.")
        
        team_data_collection = {}
        

        for team in tqdm(top_teams, desc="Collecting team data"):
            team_id = team['id']
            team_name = team['name']

            if args.verbose:
                print(f"\n{'='*50}")
                print(f"Processing team: {team_name} (ID: {team_id})")
                print(f"{'='*50}")
            
            # Get team tag for economy data matching
            team_details, team_tag = fetch_team_details(team_id)
            if args.verbose and team_tag:
                print(f"Team tag found: {team_tag}")
            
            team_history = fetch_team_match_history(team_id)
            if not team_history:
                if args.verbose:
                    print(f"Could not fetch match history for {team_name}")
                continue
                
            team_matches = parse_match_data(team_history, team_name)
            
            # Skip teams with no match data
            if not team_matches:
                if args.verbose:
                    print(f"No match data found for {team_name}")
                continue
            
            # Add team tag and ID to all matches for data extraction
            for match in team_matches:
                match['team_tag'] = team_tag
                match['team_id'] = team_id
            
            # Fetch player stats if requested
            team_player_stats = None
            if args.players:
                if args.verbose:
                    print(f"\n{'*'*30}")
                    print(f"Fetching player stats for {team_name}")
                    print(f"{'*'*30}")
                team_player_stats = fetch_team_player_stats(team_id)
                if args.verbose:
                    if team_player_stats:
                        print(f"Found {len(team_player_stats)} players for {team_name}")
                        for i, player in enumerate(team_player_stats):
                            print(f"  {i+1}. {player.get('player', 'Unknown')} - Rating: {player.get('stats', {}).get('rating', 'N/A')}")
                    else:
                        print(f"No player stats found for {team_name}")
            
            # Calculate team stats with appropriate features
            if args.economy:
                if args.verbose:
                    print(f"\nUsing enhanced economy stats calculation for {team_name}")
                team_stats = calculate_team_stats_with_economy(team_matches, team_player_stats)
            else:
                if args.verbose:
                    print(f"\nUsing standard team stats calculation for {team_name}")
                team_stats = calculate_team_stats(team_matches, team_player_stats)
            
            # Store team tag and ID in the stats
            team_stats['team_tag'] = team_tag
            team_stats['team_name'] = team_name
            team_stats['team_id'] = team_id
            
            # Fetch and add map statistics if requested
            if include_maps:
                map_stats = fetch_team_map_statistics(team_id)
                if map_stats:
                    team_stats['map_statistics'] = map_stats
                    if args.verbose:
                        print(f"Added map statistics for {team_name} ({len(map_stats)} maps)")
            
            # Extract additional analyses
            team_map_performance = extract_map_performance(team_matches)
            team_tournament_performance = extract_tournament_performance(team_matches)
            team_performance_trends = analyze_performance_trends(team_matches)
            team_opponent_quality = analyze_opponent_quality(team_matches, team_id)
            
            # Add these metrics to the team stats
            team_stats['map_performance'] = team_map_performance
            team_stats['tournament_performance'] = team_tournament_performance
            team_stats['performance_trends'] = team_performance_trends
            team_stats['opponent_quality'] = team_opponent_quality
            
            # Add team matches to stats object
            team_stats['matches'] = team_matches
            
            # Add to collection
            team_data_collection[team_name] = team_stats

            if args.verbose:
                print(f"\nStats summary for {team_name}:")
                print(f"  Win rate: {team_stats.get('win_rate', 0)*100:.2f}%")
                
                # Print player stats if available
                if team_player_stats:
                    print(f"\nPlayer stats summary for {team_name}:")
                    print(f"  Average Rating: {team_stats.get('avg_player_rating', 0):.2f}")
                    print(f"  Average ACS: {team_stats.get('avg_player_acs', 0):.2f}")
                    print(f"  Average K/D: {team_stats.get('avg_player_kd', 0):.2f}")
                    print(f"  Average KAST: {team_stats.get('avg_player_kast', 0)*100:.2f}%")
                    if 'player_stats' in team_stats and 'star_player_name' in team_stats.get('player_stats', {}):
                        print(f"  Star Player: {team_stats['player_stats']['star_player_name']} (Rating: {team_stats.get('star_player_rating', 0):.2f})")
                
                # Print economy stats if available
                if args.economy and 'pistol_win_rate' in team_stats:
                    print(f"\nEconomy stats summary for {team_name}:")
                    print(f"  Pistol Win Rate: {team_stats.get('pistol_win_rate', 0)*100:.2f}%")
                    print(f"  Eco Win Rate: {team_stats.get('eco_win_rate', 0)*100:.2f}%")
                    print(f"  Full Buy Win Rate: {team_stats.get('full_buy_win_rate', 0)*100:.2f}%")
                    print(f"  Economy Efficiency: {team_stats.get('economy_efficiency', 0):.3f}")
                
                # Print map stats if available
                if include_maps and 'map_statistics' in team_stats:
                    print(f"\nMap stats summary for {team_name}:")
                    for map_name, map_data in team_stats['map_statistics'].items():
                        print(f"  {map_name}: {map_data.get('win_percentage', 0)*100:.1f}% win rate, "
                              f"ATK: {map_data.get('atk_win_rate', 0)*100:.1f}%, "
                              f"DEF: {map_data.get('def_win_rate', 0)*100:.1f}%")
        
        print(f"Collected data for {len(team_data_collection)} teams.")
        
        # Check if we have player stats in the data
        teams_with_player_stats = sum(1 for team_data in team_data_collection.values() 
                                     if 'avg_player_rating' in team_data and team_data['avg_player_rating'] > 0)
        
        teams_with_economy_stats = sum(1 for team_data in team_data_collection.values() 
                                      if 'pistol_win_rate' in team_data and team_data['pistol_win_rate'] > 0)
        
        teams_with_map_stats = sum(1 for team_data in team_data_collection.values()
                                  if 'map_statistics' in team_data and team_data['map_statistics'])
        
        print(f"\nTeams with player statistics: {teams_with_player_stats}/{len(team_data_collection)}")
        print(f"Teams with economy statistics: {teams_with_economy_stats}/{len(team_data_collection)}")
        
        if include_maps:
            print(f"Teams with map statistics: {teams_with_map_stats}/{len(team_data_collection)}")
        
        if teams_with_player_stats == 0 and args.players:
            print("\nWARNING: No teams have player statistics even though --players flag was used.")
            print("Check your fetch_team_player_stats function for errors.")
            user_continue = input("Continue with training without player stats? (y/n): ")
            if user_continue.lower() != 'y':
                print("Training aborted.")
                return
                
        if teams_with_economy_stats == 0 and args.economy:
            print("\nWARNING: No teams have economy statistics even though --economy flag was used.")
            print("Check your calculate_team_stats_with_economy function for errors.")
            user_continue = input("Continue with training without economy stats? (y/n): ")
            if user_continue.lower() != 'y':
                print("Training aborted.")
                return
                
        if teams_with_map_stats == 0 and include_maps:
            print("\nWARNING: No teams have map statistics even though --maps flag was used.")
            print("Check your fetch_team_map_statistics function for errors.")
            user_continue = input("Continue with training without map stats? (y/n): ")
            if user_continue.lower() != 'y':
                print("Training aborted.")
                return

        # Build training dataset with all features
        print("\nBuilding training dataset with all specified features...")
        X, y = build_training_dataset_with_economy(team_data_collection)
        
        print(f"Built training dataset with {len(X)} samples.")
        
        # Check if we have enough data to train
        if len(X) < 10:
            print("Not enough training data. Please collect more match data.")
            return
        
        # Train with appropriate method based on args
        if args.cross_validate:
            print(f"Training with {args.folds}-fold cross-validation and ensemble modeling...")
            
            team_data_collection = debug_and_fix_match_data(team_data_collection)   
            
            # Check if we have enough data
            if len(X) < args.folds * 2:  # Need at least 2 samples per fold
                print(f"Not enough training data for {args.folds}-fold cross-validation.")
                print(f"Need at least {args.folds * 2} samples, but only have {len(X)}.")
                return
                
            # Train with cross-validation
            ensemble_models, stable_features, scaler, ensemble_metadata = train_and_evaluate_model(
                X, y, n_splits=args.folds, random_state=42
            )
            
            print("Ensemble model training complete.")
        elif args.learning_curves:
            # Train model with detailed learning curves for overfitting detection
            print("\nTraining model with detailed learning curves for overfitting detection...")
            model, scaler, feature_names, learning_curve_data = train_model_with_learning_curves(X, y)
            
            # Display overfitting diagnosis
            print(f"\nLearning Curve Diagnosis: {learning_curve_data['diagnosis']}")
            print("\nRecommendations:")
            for rec in learning_curve_data['recommendations']:
                print(f"  {rec}")
        elif args.optimize:
            # Run complete model optimization pipeline
            print("\nRunning complete model optimization pipeline...")
            model, scaler, feature_names = optimize_model_pipeline(X, y)
        elif args.feature_count:
            # Train with fixed feature count
            model, scaler, selected_features = train_with_fixed_feature_count(
                X, y, feature_count=args.feature_count
            )    
            
            # Analyze selected features
            feature_analysis = analyze_selected_features(selected_features)
        else:
            # Train regular model
            print("\nTraining standard model...")
            model, scaler, feature_names = train_model(X, y)
        
        print("Model training complete.")
    
    elif args.predict and args.team1 and args.team2:
        # Check if ensemble models exist
        ensemble_exists = any(os.path.exists(f'valorant_model_fold_{i+1}.h5') for i in range(5))
        
        if ensemble_exists:
            print(f"Predicting match between {args.team1} and {args.team2} using ensemble model...")
            prediction = predict_match_with_ensemble(args.team1, args.team2)
        elif include_maps:
            print(f"Predicting match between {args.team1} and {args.team2} with enhanced map statistics...")
            prediction = predict_match_with_optimized_model(args.team1, args.team2)
        else:
            print(f"Predicting match between {args.team1} and {args.team2} with standard model...")
            prediction = predict_match_with_optimized_model(args.team1, args.team2)
        
        if prediction:
            if include_maps:
                visualize_prediction_with_economy_and_maps(prediction)
            else:
                visualize_prediction_with_economy(prediction)
        else:
            print(f"Could not generate prediction for {args.team1} vs {args.team2}")
    
    elif args.optimize and not args.train:
        # Run optimization on existing model data
        print("Running optimization on existing model data...")
        try:
            # Load existing training data
            with open('feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
                
            # Collect new data for optimization
            team_data_collection = collect_all_team_data(
                include_player_stats=True, 
                include_economy=True, 
                include_maps=include_maps,
                verbose=args.verbose
            )
            
            if team_data_collection:
                X, y = build_training_dataset_with_economy(team_data_collection)
                
                if len(X) > 10:
                    # Run optimization pipeline
                    model, scaler, optimized_features = optimize_model_pipeline(X, y)
                    print("Model optimization complete.")
                else:
                    print("Not enough training data for optimization.")
            else:
                print("Failed to collect team data for optimization.")
        except Exception as e:
            print(f"Error during optimization: {e}")
            print("Please train a model first before optimization.")
            
    elif args.learning_curves and not args.train:
        # Generate learning curves for existing model data
        print("Generating learning curves for existing model data...")
        try:
            # Load existing training data
            with open('feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
                
            # Collect new data for analysis
            team_data_collection = collect_all_team_data(
                include_player_stats=True, 
                include_economy=True, 
                include_maps=include_maps,
                verbose=args.verbose
            )
            
            if team_data_collection:
                X, y = build_training_dataset_with_economy(team_data_collection)
                
                if len(X) > 10:
                    # Run learning curve analysis
                    model, scaler, feature_names, learning_curve_data = train_model_with_learning_curves(X, y)
                    
                    # Display overfitting diagnosis
                    print(f"\nLearning Curve Diagnosis: {learning_curve_data['diagnosis']}")
                    print("\nRecommendations:")
                    for rec in learning_curve_data['recommendations']:
                        print(f"  {rec}")
                else:
                    print("Not enough training data for learning curve analysis.")
            else:
                print("Failed to collect team data for learning curve analysis.")
        except Exception as e:
            print(f"Error during learning curve analysis: {e}")
            print("Please train a model first before generating learning curves.")
    
    elif args.analyze:
        # Analyze upcoming matches using the best available model
        if include_maps:
            print("Analyzing upcoming matches with enhanced map statistics...")
            analyze_upcoming_matches_with_maps()
        else:
            print("Analyzing upcoming matches with standard model...")
            analyze_upcoming_matches()
            
    elif args.backtest:
        if not args.cutoff_date:
            print("Please specify a cutoff date with --cutoff-date YYYY/MM/DD")
            return
        
        print(f"Performing backtesting with cutoff date: {args.cutoff_date}")
        print(f"Bet amount: ${args.bet_amount}, Confidence threshold: {args.confidence}")
        
        # Determine which model to use for backtesting
        if include_maps:
            print("Using enhanced map statistics for backtesting...")
            results = backtest_model_with_maps(
                args.cutoff_date, 
                bet_amount=args.bet_amount, 
                confidence_threshold=args.confidence
            )
        else:
            print("Using standard model for backtesting...")
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
            
            if include_maps and 'map_accuracy' in results:
                print("\nMap-Specific Accuracy:")
                for map_name, acc in results['map_accuracy'].items():
                    print(f"  {map_name}: {acc:.4f}")
        else:
            print("Backtesting failed or returned no results.")
    
    else:
        print("Please specify an action: --train, --optimize, --learning-curves, --predict, --analyze, or --backtest")
        print("For predictions, specify --team1 and --team2")
        print("For backtesting, specify --cutoff-date YYYY/MM/DD")
        print("To include player statistics, add --players")
        print("To include economy data, add --economy")
        print("To include map statistics, add --maps")
        print("To test with specific teams, use --train --test-teams \"Team Name 1\" \"Team Name 2\" ...")

if __name__ == "__main__":
    main()