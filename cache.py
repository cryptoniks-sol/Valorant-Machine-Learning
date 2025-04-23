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
    """Fetch detailed information about a team."""
    if not team_id:
        return None
    
    print(f"Fetching details for team ID: {team_id}")
    response = requests.get(f"{API_URL}/teams/{team_id}")
    
    if response.status_code != 200:
        print(f"Error fetching team {team_id}: {response.status_code}")
        return None
    
    team_data = response.json()
    
    # Be nice to the API
    time.sleep(0.5)
    
    return team_data

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
    time.sleep(0.5)
    
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
    time.sleep(0.5)
    
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
    time.sleep(0.5)
    
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
    time.sleep(0.5)
    
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
    time.sleep(0.5)
    
    # Return player stats if successful
    if player_data.get('status') == 'OK' and 'data' in player_data:
        return player_data['data']

    player_data = response.json()
    print(f"Player data for {player_name}: {json.dumps(player_data, indent=2)}")
    
    return None    

# Update function to create a better deep learning model with player stats
def create_deep_learning_model(input_dim):
    """Create an enhanced deep learning model for match prediction with player stats."""
    # Define inputs
    inputs = Input(shape=(input_dim,))
    
    # Feature processing pathway
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Team comparison pathway
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Player stats pathway
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
                 optimizer=Adam(learning_rate=0.0005),
                 metrics=['accuracy'])
    
    return model

def train_model(X, y, test_size=0.2, random_state=42):
    """Train the deep learning model and evaluate its performance."""
    # Check if we have data
    if not X or len(X) == 0:
        print("Error: No training data available")
        return None, None, None
        
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
        return None, None, None
    
    # Convert to numpy array
    X_arr = df.values
    y_arr = np.array(y)
    
    print(f"\nFinal feature matrix shape: {X_arr.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_arr, test_size=test_size, random_state=random_state
    )
    
    # Check for class imbalance
    class_counts = np.bincount(y_train)
    print(f"Class distribution: {class_counts}")
    
    # Try to apply SMOTE if there's enough data in each class
    if np.min(class_counts) < 5:
        print("Not enough samples in minority class for SMOTE. Using original data.")
    else:
        if np.min(class_counts) / np.sum(class_counts) < 0.4:  # If imbalanced
            print("Applying SMOTE to handle class imbalance...")
            try:
                # Use k_neighbors=min(5, min_samples-1) to ensure we don't ask for too many neighbors
                min_samples = np.min(class_counts)
                k_neighbors = min(5, min_samples-1)
                print(f"Using k_neighbors={k_neighbors} for SMOTE (min_samples={min_samples})")
                
                smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE resampling: X_train shape: {X_train.shape}")
            except Exception as e:
                print(f"Error applying SMOTE: {e}")
                print("Continuing with original data.")
    
    # Create and train model
    model = create_deep_learning_model(X_train.shape[1])
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_valorant_model.h5', 
        save_best_only=True, 
        monitor='val_accuracy'
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Evaluate on test set
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Save model artifacts
    model.save('valorant_model.h5')
    
    # Save scaler for future use
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(list(df.columns), f)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return model, scaler, list(df.columns)

def predict_match(team1_name, team2_name, model=None, scaler=None, feature_names=None, export_data=True, display_details=True):
    """Predict the outcome of a match between two teams."""
    print(f"Predicting match between {team1_name} and {team2_name}...")
    
    # Get team IDs
    team1_id = get_team_id(team1_name)
    team2_id = get_team_id(team2_name)
    
    if not team1_id or not team2_id:
        print("Could not find one or both teams. Please check team names.")
        return None
    
    # Fetch match histories
    team1_history = fetch_team_match_history(team1_id)
    team2_history = fetch_team_match_history(team2_id)
    
    if not team1_history or not team2_history:
        print("Could not fetch match history for one or both teams.")
        return None
    
    # Parse match data
    team1_matches = parse_match_data(team1_history, team1_name)
    team2_matches = parse_match_data(team2_history, team2_name)
    
    # Fetch player stats for both teams
    team1_player_stats = fetch_team_player_stats(team1_id)
    team2_player_stats = fetch_team_player_stats(team2_id)

    team1_stats = calculate_team_stats(team1_matches, team1_player_stats)
    team2_stats = calculate_team_stats(team2_matches, team2_player_stats)
    
    # Extract additional metrics
    team1_map_performance = extract_map_performance(team1_matches)
    team2_map_performance = extract_map_performance(team2_matches)
    
    team1_tournament_performance = extract_tournament_performance(team1_matches)
    team2_tournament_performance = extract_tournament_performance(team2_matches)
    
    team1_performance_trends = analyze_performance_trends(team1_matches)
    team2_performance_trends = analyze_performance_trends(team2_matches)
    
    # Fetch all teams for opponent quality analysis
    all_teams = []
    try:
        response = requests.get(f"{API_URL}/teams?limit=500")
        if response.status_code == 200:
            all_teams = response.json().get('data', [])
    except Exception as e:
        print(f"Error fetching teams: {e}")
    
    team1_opponent_quality = analyze_opponent_quality(team1_matches, all_teams)
    team2_opponent_quality = analyze_opponent_quality(team2_matches, all_teams)
    
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
    features = prepare_data_for_model(team1_stats, team2_stats)
    
    if not features:
        print("Could not prepare features for prediction.")
        return None
    
    # Load model if not provided
    if model is None:
        try:
            model = load_model('valorant_model.h5')
            with open('feature_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train a model first or provide a trained model.")
            return None
    
    # Convert features to DataFrame and align with expected feature names
    features_df = pd.DataFrame([features])
    
    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in features_df.columns:
            features_df[feature] = 0  # Default value for missing features
    
    # Keep only the features used during training
    features_df = features_df[feature_names]
    
    # Scale features
    X = scaler.transform(features_df.values)
    
    # Make prediction
    prediction = model.predict(X)[0][0]
    
    # Calculate confidence
    confidence = max(prediction, 1 - prediction)
    
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
            'star_player_rating': team1_stats.get('star_player_rating', 0)
        },
        'team2_stats_summary': {
            'matches_played': team2_stats['matches'] if isinstance(team2_stats['matches'], int) else len(team2_stats['matches']),
            'win_rate': team2_stats['win_rate'],
            'recent_form': team2_stats['recent_form'],
            'avg_player_rating': team2_stats.get('avg_player_rating', 0),
            'star_player': team2_stats.get('player_stats', {}).get('star_player_name', ''),
            'star_player_rating': team2_stats.get('star_player_rating', 0)
        }
    }
    
    # Export prediction data if requested
    if export_data:
        export_prediction_data(result, team1_stats, team2_stats)
    
    # Display detailed results if requested
    if display_details:
        display_prediction_results(result, team1_stats, team2_stats)
    
    return result

# Update visualize_prediction to include player stats
def visualize_prediction(prediction_result):
    """Visualize the match prediction with player stats."""
    if not prediction_result:
        print("No prediction to visualize.")
        return
    
    team1 = prediction_result['team1']
    team2 = prediction_result['team2']
    team1_prob = prediction_result['team1_win_probability']
    team2_prob = prediction_result['team2_win_probability']
    predicted_winner = prediction_result['predicted_winner']
    confidence = prediction_result['confidence']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
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
    
    # Add prediction summary
    plt.figtext(0.5, 0.01, 
                f"Predicted Winner: {predicted_winner} (Confidence: {confidence:.1%})",
                ha="center", fontsize=14, bbox={"facecolor":"#f9f9f9", "alpha":0.5, "pad":5})
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('match_prediction.png')
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
                prediction = predict_match(team1_name, team2_name, model, scaler, feature_names)
                
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
            visualize_prediction(pred)
            if i < len(predictions) - 1:
                plt.figure()  # Create a new figure for the next prediction


def collect_all_team_data(include_player_stats=True, verbose=False):
    """Collect data for all teams to use in backtesting."""
    print("Collecting data for all teams...")
    
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
        top_teams = teams_data['data'][:20]
    
    print(f"Selected {len(top_teams)} teams for data collection.")
    
    # Collect match data for each team
    team_data_collection = {}
    
    for team in tqdm(top_teams, desc="Collecting team data"):
        team_id = team['id']
        team_name = team['name']
        
        if verbose:
            print(f"\nProcessing team: {team_name} (ID: {team_id})")
        
        team_history = fetch_team_match_history(team_id)
        if not team_history:
            continue
            
        team_matches = parse_match_data(team_history, team_name)
        
        # Skip teams with no match data
        if not team_matches:
            continue
            
        # Fetch player stats if requested
        team_player_stats = None
        if include_player_stats:
            if verbose:
                print(f"Fetching player stats for team: {team_name}")
            team_player_stats = fetch_team_player_stats(team_id)
            if verbose and team_player_stats:
                print(f"Found {len(team_player_stats)} players for {team_name}")
                
        team_stats = calculate_team_stats(team_matches, team_player_stats)
        
        # Add team matches to stats object
        team_stats['matches'] = team_matches
        
        # Add to collection
        team_data_collection[team_name] = team_stats
    
    print(f"Collected data for {len(team_data_collection)} teams.")
    return team_data_collection


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


def backtest_model(cutoff_date, bet_amount=100, confidence_threshold=0.6):
    """Backtest the model using historical data split by date."""
    # Get all teams and matches
    team_data_collection = collect_all_team_data()
    
    if not team_data_collection:
        print("Failed to collect team data. Aborting backtesting.")
        return None
    
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
    
    # Train model on older matches
    X_train, y_train = build_training_dataset_from_matches(train_matches, team_data_collection)
    
    if len(X_train) < 10:
        print("Not enough training samples. Try using an earlier cutoff date.")
        return None
    
    print(f"Training model with {len(X_train)} samples...")
    model, scaler, feature_names = train_model(X_train, y_train)
    
    if not model:
        print("Failed to train model. Aborting backtesting.")
        return None
    
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
        
        # Get prediction
        prediction = predict_match(team1_name, team2_name, model, scaler, feature_names)
        
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
                'correct': predicted_winner == actual_winner
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
    
    # Save detailed results to CSV
    if results_data:
        df = pd.DataFrame(results_data)
        df.to_csv('backtesting_results.csv', index=False)
        print(f"Detailed results saved to 'backtesting_results.csv'")
    
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
        'cutoff_date': cutoff_date
    }
    
    return results



# Update export_prediction_data function to include player stats
def export_prediction_data(prediction, team1_stats, team2_stats, filename=None):
    """
    Export the prediction data and team statistics to a JSON file with player stats.
    
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
    
    # Clean up team stats to make them JSON serializable
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
            "key_factors": []  # Will be populated below
        }
    }
    
    # Add key matchup factors
    if 'team1_stats_summary' in prediction and 'team2_stats_summary' in prediction:
        # Compare win rates
        t1_wr = prediction['team1_stats_summary'].get('win_rate', 0)
        t2_wr = prediction['team2_stats_summary'].get('win_rate', 0)
        if abs(t1_wr - t2_wr) > 0.1:  # If significant difference
            better_team = prediction['team1'] if t1_wr > t2_wr else prediction['team2']
            export_data['analysis']['key_factors'].append(
                f"{better_team} has a significantly better win rate ({max(t1_wr, t2_wr):.1%} vs {min(t1_wr, t2_wr):.1%})"
            )
        
        # Compare recent form
        t1_form = prediction['team1_stats_summary'].get('recent_form', 0)
        t2_form = prediction['team2_stats_summary'].get('recent_form', 0)
        if abs(t1_form - t2_form) > 0.15:  # If significant difference
            better_form = prediction['team1'] if t1_form > t2_form else prediction['team2']
            export_data['analysis']['key_factors'].append(
                f"{better_form} has better recent form ({max(t1_form, t2_form):.1%} vs {min(t1_form, t2_form):.1%})"
            )
            
        # Compare player ratings
        t1_rating = prediction['team1_stats_summary'].get('avg_player_rating', 0)
        t2_rating = prediction['team2_stats_summary'].get('avg_player_rating', 0)
        if abs(t1_rating - t2_rating) > 0.1:  # If significant difference
            better_rated = prediction['team1'] if t1_rating > t2_rating else prediction['team2']
            export_data['analysis']['key_factors'].append(
                f"{better_rated} has higher-rated players ({max(t1_rating, t2_rating):.2f} vs {min(t1_rating, t2_rating):.2f})"
            )
            
        # Compare star players
        t1_star = prediction['team1_stats_summary'].get('star_player_rating', 0)
        t2_star = prediction['team2_stats_summary'].get('star_player_rating', 0)
        if abs(t1_star - t2_star) > 0.2:  # If significant difference
            better_star_team = prediction['team1'] if t1_star > t2_star else prediction['team2']
            star_name = prediction['team1_stats_summary'].get('star_player', '') if t1_star > t2_star else prediction['team2_stats_summary'].get('star_player', '')
            export_data['analysis']['key_factors'].append(
                f"{better_star_team} has a stronger star player ({star_name}, {max(t1_star, t2_star):.2f} rating)"
            )
    
    # Add head-to-head history if available
    h2h_matches = 0
    if 'opponent_stats' in team1_stats and prediction['team2'] in team1_stats['opponent_stats']:
        h2h = team1_stats['opponent_stats'][prediction['team2']]
        h2h_matches = h2h.get('matches', 0)
        h2h_winrate = h2h.get('win_rate', 0)
        export_data['analysis']['head_to_head'] = {
            'matches_played': h2h_matches,
            'team1_win_rate': h2h_winrate,
            'team2_win_rate': 1 - h2h_winrate
        }
        
        # Add as a key factor if significant history
        if h2h_matches >= 3:
            dominant_team = prediction['team1'] if h2h_winrate > 0.55 else prediction['team2'] if h2h_winrate < 0.45 else None
            if dominant_team:
                export_data['analysis']['key_factors'].append(
                    f"{dominant_team} has historically dominated this matchup ({max(h2h_winrate, 1-h2h_winrate):.1%} win rate in {h2h_matches} matches)"
                )
    
    # Add map analysis if available
    common_maps = set()
    if 'map_stats' in team1_stats and 'map_stats' in team2_stats:
        common_maps = set(team1_stats['map_stats'].keys()) & set(team2_stats['map_stats'].keys())
        map_analysis = {}
        
        for map_name in common_maps:
            map_analysis[map_name] = {
                f"{prediction['team1']}_win_rate": team1_stats['map_stats'][map_name].get('win_rate', 0),
                f"{prediction['team2']}_win_rate": team2_stats['map_stats'][map_name].get('win_rate', 0)
            }
            
            # Determine map advantage
            t1_map_wr = team1_stats['map_stats'][map_name].get('win_rate', 0)
            t2_map_wr = team2_stats['map_stats'][map_name].get('win_rate', 0)
            if abs(t1_map_wr - t2_map_wr) > 0.15:  # Significant advantage
                map_advantage = prediction['team1'] if t1_map_wr > t2_map_wr else prediction['team2']
                map_analysis[map_name]['advantage'] = map_advantage
                
                # Add as key factor if it's a strong map for one team
                if max(t1_map_wr, t2_map_wr) > 0.65:
                    export_data['analysis']['key_factors'].append(
                        f"{map_advantage} has a strong advantage on {map_name} ({max(t1_map_wr, t2_map_wr):.1%} vs {min(t1_map_wr, t2_map_wr):.1%})"
                    )
            else:
                map_analysis[map_name]['advantage'] = "neutral"
        
        export_data['analysis']['map_analysis'] = map_analysis
        
    # Add player stats analysis
    if 'player_stats' in team1_stats and 'player_stats' in team2_stats:
        export_data['analysis']['player_comparison'] = {
            'avg_rating': {
                prediction['team1']: team1_stats.get('avg_player_rating', 0),
                prediction['team2']: team2_stats.get('avg_player_rating', 0),
                'difference': team1_stats.get('avg_player_rating', 0) - team2_stats.get('avg_player_rating', 0)
            },
            'avg_acs': {
                prediction['team1']: team1_stats.get('avg_player_acs', 0),
                prediction['team2']: team2_stats.get('avg_player_acs', 0),
                'difference': team1_stats.get('avg_player_acs', 0) - team2_stats.get('avg_player_acs', 0)
            },
            'avg_kd': {
                prediction['team1']: team1_stats.get('avg_player_kd', 0),
                prediction['team2']: team2_stats.get('avg_player_kd', 0),
                'difference': team1_stats.get('avg_player_kd', 0) - team2_stats.get('avg_player_kd', 0)
            },
            'fk_fd_ratio': {
                prediction['team1']: team1_stats.get('fk_fd_ratio', 0),
                prediction['team2']: team2_stats.get('fk_fd_ratio', 0),
                'difference': team1_stats.get('fk_fd_ratio', 0) - team2_stats.get('fk_fd_ratio', 0)
            },
            'star_players': {
                prediction['team1']: {
                    'name': team1_stats.get('player_stats', {}).get('star_player_name', ''),
                    'rating': team1_stats.get('star_player_rating', 0)
                },
                prediction['team2']: {
                    'name': team2_stats.get('player_stats', {}).get('star_player_name', ''),
                    'rating': team2_stats.get('star_player_rating', 0)
                }
            }
        }
    
    # Save to JSON file
    try:
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Prediction data exported to {filename}")
        return filename
    except Exception as e:
        print(f"Error exporting prediction data: {e}")
        return None



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

# Update main function to include player stats
def main():
    """Main function to handle command line arguments and run the program."""
    parser = argparse.ArgumentParser(description="Valorant Match Predictor with Player Stats")
    
    # Add command line arguments
    parser.add_argument("--train", action="store_true", help="Train a new model")
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
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")

    args = parser.parse_args()
    
    if args.train:
        print("Training a new model with player statistics...")
        
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
                top_teams = teams_data['data'][:100]
        
        print(f"Selected {len(top_teams)} teams for training data.")
        
        # Collect match data for each team
        team_data_collection = {}
        
        for team in tqdm(top_teams, desc="Collecting team data"):
            team_id = team['id']
            team_name = team['name']

            if args.verbose:
                print(f"\n{'='*50}")
                print(f"Processing team: {team_name} (ID: {team_id})")
                print(f"{'='*50}")
            
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
                
            # Calculate team stats with player data
            if args.verbose:
                print(f"\nCalculating team stats for {team_name}...")
            team_stats = calculate_team_stats(team_matches, team_player_stats)
            
            # Add team matches to stats object
            team_stats['matches'] = team_matches
            
            # Add to collection
            team_data_collection[team_name] = team_stats

            if args.verbose and team_player_stats:
                print(f"\nPlayer stats summary for {team_name}:")
                print(f"  Average Rating: {team_stats.get('avg_player_rating', 0):.2f}")
                print(f"  Average ACS: {team_stats.get('avg_player_acs', 0):.2f}")
                print(f"  Average K/D: {team_stats.get('avg_player_kd', 0):.2f}")
                print(f"  Average KAST: {team_stats.get('avg_player_kast', 0)*100:.2f}%")
                print(f"  Star Player: {team_stats.get('player_stats', {}).get('star_player_name', 'Unknown')}")
        
        print(f"Collected data for {len(team_data_collection)} teams.")
        
        # Check if we have player stats in the data
        teams_with_player_stats = sum(1 for team_data in team_data_collection.values() 
                                     if 'avg_player_rating' in team_data and team_data['avg_player_rating'] > 0)
        
        print(f"\nTeams with player statistics: {teams_with_player_stats}/{len(team_data_collection)}")
        if teams_with_player_stats == 0 and args.players:
            print("\nWARNING: No teams have player statistics even though --players flag was used.")
            print("Check your fetch_team_player_stats function for errors.")
            user_continue = input("Continue with training without player stats? (y/n): ")
            if user_continue.lower() != 'y':
                print("Training aborted.")
                return

        # Build training dataset
        print("\nBuilding training dataset...")
        X, y = build_training_dataset(team_data_collection)
        
        print(f"Built training dataset with {len(X)} samples.")
        
        # Print some sample features to verify player stats are included
        if X and len(X) > 0 and args.verbose:
            print("\nSample features from first training example:")
            sample_features = X[0]
            player_features = [k for k in sample_features.keys() if any(p in k for p in 
                              ['rating', 'acs', 'kd', 'kast', 'adr', 'headshot', 'star_player'])]
            
            if player_features:
                print("\nPlayer statistics features:")
                for feature in player_features:
                    print(f"  {feature}: {sample_features[feature]}")
            else:
                print("\nWARNING: No player statistics features found in training data.")
        
        # Check if we have enough data to train
        if len(X) < 10:
            print("Not enough training data. Please collect more match data.")
            return
        
        # Train model
        print("\nTraining model...")
        model, scaler, feature_names = train_model(X, y)

        # Check if player features were used in the model
        if feature_names and args.verbose:
            player_features_used = [f for f in feature_names if any(p in f for p in 
                                   ['rating', 'acs', 'kd', 'kast', 'adr', 'headshot', 'star_player'])]
            
            print(f"\nPlayer statistics features used in model: {len(player_features_used)}/{len(feature_names)}")
            if player_features_used:
                print("Examples of player features used:")
                for feature in player_features_used[:10]:  # Show first 5 examples
                    print(f"  {feature}")
        
        print("Model training complete with player statistics included.")
    
    elif args.predict and args.team1 and args.team2:
        # Check if model files exist
        model_exists = os.path.exists('valorant_model.h5')
        scaler_exists = os.path.exists('feature_scaler.pkl')
        features_exists = os.path.exists('feature_names.pkl')
        
        if not (model_exists and scaler_exists and features_exists):
            print("Model files not found. Please train a model first using --train")
            return
            
        # Predict a specific match
        prediction = predict_match(args.team1, args.team2)
        
        if prediction:
            visualize_prediction(prediction)
        else:
            print(f"Could not generate prediction for {args.team1} vs {args.team2}")
    
    elif args.analyze:
        # Check if model files exist
        model_exists = os.path.exists('valorant_model.h5')
        scaler_exists = os.path.exists('feature_scaler.pkl')
        features_exists = os.path.exists('feature_names.pkl')
        
        if not (model_exists and scaler_exists and features_exists):
            print("Model files not found. Please train a model first using --train")
            return
            
        # Analyze all upcoming matches
        analyze_upcoming_matches()

    # Add a new condition for backtesting
    elif args.backtest:
        if not args.cutoff_date:
            print("Please specify a cutoff date with --cutoff-date YYYY/MM/DD")
            return
        
        print(f"Performing backtesting with cutoff date: {args.cutoff_date}")
        print(f"Bet amount: ${args.bet_amount}, Confidence threshold: {args.confidence}")
        print(f"Including player stats: {'Yes' if args.players else 'No'}")
        
        results = backtest_model(args.cutoff_date, 
                                bet_amount=args.bet_amount, 
                                confidence_threshold=args.confidence)
        
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
    
    else:
        print("Please specify an action: --train, --predict, --analyze, or --backtest")
        print("For predictions, specify --team1 and --team2")
        print("For backtesting, specify --cutoff-date YYYY/MM/DD")
        print("To include player statistics, add --players")
        print("To test with specific teams, use --train --test-teams \"Team Name 1\" \"Team Name 2\" ...")

if __name__ == "__main__":
    main()