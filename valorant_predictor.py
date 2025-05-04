#!/usr/bin/env python3
"""
Valorant Match Prediction System

This system utilizes trained ensemble models to predict the outcome of Valorant matches.
It includes advanced features for confidence weighting, head-to-head analysis,
feature processing, ensemble prediction, and detailed reporting.

The system addresses specific challenges such as:
- Sample size imbalance between teams
- Proper weighting of head-to-head records
- Confidence-based ensemble predictions
- Validation and contradiction detection
- Consistent feature processing
- Comprehensive output reporting
"""

import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import json
import argparse
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import time
from datetime import datetime

# API URL - Change this to match your local or remote API
API_URL = "http://localhost:5000/api/v1"

#-------------------------------------------------------------------------
# DATA COLLECTION AND PROCESSING
#-------------------------------------------------------------------------

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

def fetch_team_match_history(team_id):
    """Fetch match history for a specific team."""
    if not team_id:
        return None
    
    print(f"Fetching match history for team ID: {team_id}")
    return fetch_api_data(f"match-history/{team_id}")

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
                
                matches.append(match_info)
            
        except Exception as e:
            print(f"Error parsing match: {e}")
            continue
    
    print(f"Skipped {filtered_count} matches that did not involve {team_name}")   
    # Summarize wins/losses
    wins = sum(1 for match in matches if match['team_won'])
    print(f"Processed {len(matches)} matches for {team_name}: {wins} wins, {len(matches) - wins} losses")
    
    return matches

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

        map_statistics[map_name] = {
            'win_percentage': win_percentage,
            'wins': wins,
            'losses': losses,
            'matches_played': wins + losses,
            'atk_first': atk_first,
            'def_first': def_first,
            'atk_win_rate': atk_win_rate,
            'def_win_rate': def_win_rate,
            'side_preference': 'Attack' if atk_win_rate > def_win_rate else 'Defense',
            'side_preference_strength': abs(atk_win_rate - def_win_rate),
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
        # Add sample size reliability metrics
        'reliability_factor': 1.0 - (1.0 / (1.0 + 0.1 * total_matches)),  # Approaches 1 as matches increase
        'min_reliable_matches': 10,  # Threshold for minimum reliable sample
        'is_reliable_sample': total_matches >= 10,
        'data_confidence': min(1.0, total_matches / 20.0)  # Caps at 1.0 after 20 matches
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
                'pistol_confidence': 1.0 - (1.0 / (1.0 + 0.1 * total_pistol_rounds)),
                'economy_confidence': 1.0 - (1.0 / (1.0 + 0.05 * economy_matches_count))
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

def prepare_data_for_model(team1_stats, team2_stats, verbose=False):
    """
    Prepare data for the ML model by creating feature vectors with enhanced
    head-to-head analysis and sample size normalization.
    
    Args:
        team1_stats (dict): Statistics for team 1
        team2_stats (dict): Statistics for team 2
        verbose (bool): Whether to print detailed feature information
    
    Returns:
        dict: Features prepared for model prediction
    """
    if not team1_stats or not team2_stats:
        print("Missing team statistics data")
        return None
    
    features = {}
    feature_explanations = {}
    
    #----------------------------------------
    # 1. BASIC TEAM STATS WITH RELIABILITY WEIGHTING
    #----------------------------------------
    # Win rates and match-level statistics with reliability adjustments
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
    
    # Calculate reliability factors
    t1_reliability = team1_stats.get('reliability_factor', 1.0 - (1.0 / (1.0 + 0.1 * team1_match_count)))
    t2_reliability = team2_stats.get('reliability_factor', 1.0 - (1.0 / (1.0 + 0.1 * team2_match_count)))
    
    # Apply reliability weighting to win rates
    t1_win_rate = team1_stats.get('win_rate', 0)
    t2_win_rate = team2_stats.get('win_rate', 0)
    
    # Weighted win rates - adjust toward 0.5 for teams with fewer matches
    weighted_t1_win_rate = t1_win_rate * t1_reliability + 0.5 * (1 - t1_reliability)
    weighted_t2_win_rate = t2_win_rate * t2_reliability + 0.5 * (1 - t2_reliability)
    
    # Calculate weighted win rate difference
    features['win_rate_diff'] = weighted_t1_win_rate - weighted_t2_win_rate
    features['raw_win_rate_diff'] = t1_win_rate - t2_win_rate  # Keep raw difference for comparison
    features['better_win_rate_team1'] = 1 if weighted_t1_win_rate > weighted_t2_win_rate else 0
    
    feature_explanations['win_rate_diff'] = {
        'description': 'Difference in reliability-weighted win rates',
        'team1_value': f"{weighted_t1_win_rate:.3f} (raw: {t1_win_rate:.3f}, reliability: {t1_reliability:.2f})",
        'team2_value': f"{weighted_t2_win_rate:.3f} (raw: {t2_win_rate:.3f}, reliability: {t2_reliability:.2f})",
        'importance': 'High'
    }
    
    # Recent form with reliability
    t1_recent_form = team1_stats.get('recent_form', 0)
    t2_recent_form = team2_stats.get('recent_form', 0)
    
    # Recent form is already more reliable (based on last few matches)
    # But still apply some reliability factor for teams with very few matches
    t1_recent_reliability = min(1.0, team1_match_count / 10.0)  # Caps at 1.0 after 10 matches
    t2_recent_reliability = min(1.0, team2_match_count / 10.0)
    
    weighted_t1_recent_form = t1_recent_form * t1_recent_reliability + 0.5 * (1 - t1_recent_reliability)
    weighted_t2_recent_form = t2_recent_form * t2_recent_reliability + 0.5 * (1 - t2_recent_reliability)
    
    features['recent_form_diff'] = weighted_t1_recent_form - weighted_t2_recent_form
    features['raw_recent_form_diff'] = t1_recent_form - t2_recent_form
    features['better_recent_form_team1'] = 1 if weighted_t1_recent_form > weighted_t2_recent_form else 0
    
    feature_explanations['recent_form_diff'] = {
        'description': 'Difference in reliability-weighted recent form',
        'team1_value': f"{weighted_t1_recent_form:.3f} (raw: {t1_recent_form:.3f}, reliability: {t1_recent_reliability:.2f})",
        'team2_value': f"{weighted_t2_recent_form:.3f} (raw: {t2_recent_form:.3f}, reliability: {t2_recent_reliability:.2f})",
        'importance': 'High'
    }
    
    # Score differential with reliability
    t1_score_diff = team1_stats.get('score_differential', 0)
    t2_score_diff = team2_stats.get('score_differential', 0)
    
    weighted_t1_score_diff = t1_score_diff * t1_reliability
    weighted_t2_score_diff = t2_score_diff * t2_reliability
    
    features['score_diff_differential'] = weighted_t1_score_diff - weighted_t2_score_diff
    features['raw_score_diff_differential'] = t1_score_diff - t2_score_diff
    features['better_score_diff_team1'] = 1 if weighted_t1_score_diff > weighted_t2_score_diff else 0
    
    # Sample size information
    features['total_matches'] = team1_match_count + team2_match_count
    features['match_count_diff'] = team1_match_count - team2_match_count
    features['match_count_ratio'] = team1_match_count / max(1, team2_match_count)
    features['sample_size_disparity'] = abs(team1_match_count - team2_match_count) / max(1, team1_match_count + team2_match_count)
    
    # Add average metrics rather than separate team metrics
    features['avg_win_rate'] = (weighted_t1_win_rate + weighted_t2_win_rate) / 2
    features['avg_recent_form'] = (weighted_t1_recent_form + weighted_t2_recent_form) / 2
    
    # Add win/loss counts
    features['wins_diff'] = team1_stats.get('wins', 0) - team2_stats.get('wins', 0)
    features['losses_diff'] = team1_stats.get('losses', 0) - team2_stats.get('losses', 0)
    
    # Win-loss ratio (adjusted for small sample sizes)
    t1_win_loss_ratio = team1_stats.get('wins', 0) / max(team1_stats.get('losses', 1), 1) 
    t2_win_loss_ratio = team2_stats.get('wins', 0) / max(team2_stats.get('losses', 1), 1)
    
    # Apply reliability factors
    weighted_t1_wl_ratio = t1_win_loss_ratio * t1_reliability + 1.0 * (1 - t1_reliability)
    weighted_t2_wl_ratio = t2_win_loss_ratio * t2_reliability + 1.0 * (1 - t2_reliability)
    
    features['win_loss_ratio_diff'] = weighted_t1_wl_ratio - weighted_t2_wl_ratio
    features['raw_win_loss_ratio_diff'] = t1_win_loss_ratio - t2_win_loss_ratio
    
    # Average scores
    t1_avg_score = team1_stats.get('avg_score', 0)
    t2_avg_score = team2_stats.get('avg_score', 0)
    
    weighted_t1_avg_score = t1_avg_score * t1_reliability + 10.0 * (1 - t1_reliability)  # 10 is a neutral score
    weighted_t2_avg_score = t2_avg_score * t2_reliability + 10.0 * (1 - t2_reliability)
    
    features['avg_score_diff'] = weighted_t1_avg_score - weighted_t2_avg_score
    features['raw_avg_score_diff'] = t1_avg_score - t2_avg_score
    features['better_avg_score_team1'] = 1 if weighted_t1_avg_score > weighted_t2_avg_score else 0
    features['avg_score_metric'] = (weighted_t1_avg_score + weighted_t2_avg_score) / 2
    
    # Opponent scores
    t1_avg_opp_score = team1_stats.get('avg_opponent_score', 0)
    t2_avg_opp_score = team2_stats.get('avg_opponent_score', 0)
    
    weighted_t1_avg_opp_score = t1_avg_opp_score * t1_reliability + 10.0 * (1 - t1_reliability)
    weighted_t2_avg_opp_score = t2_avg_opp_score * t2_reliability + 10.0 * (1 - t2_reliability)
    
    features['avg_opponent_score_diff'] = weighted_t1_avg_opp_score - weighted_t2_avg_opp_score
    features['raw_avg_opponent_score_diff'] = t1_avg_opp_score - t2_avg_opp_score
    features['better_defense_team1'] = 1 if weighted_t1_avg_opp_score < weighted_t2_avg_opp_score else 0
    features['avg_defense_metric'] = (weighted_t1_avg_opp_score + weighted_t2_avg_opp_score) / 2
    
    #----------------------------------------
    # 2. PLAYER STATS WITH RELIABILITY ADJUSTMENTS
    #----------------------------------------
    # Only add if both teams have the data
    if ('avg_player_rating' in team1_stats and 'avg_player_rating' in team2_stats and
        team1_stats.get('avg_player_rating', 0) > 0 and team2_stats.get('avg_player_rating', 0) > 0):
        
        # Player rating confidence factor (distinct from overall team reliability)
        t1_player_confidence = min(1.0, team1_match_count / 15.0)  # Caps at 1.0 after 15 matches
        t2_player_confidence = min(1.0, team2_match_count / 15.0)
        
        # Basic player rating stats with confidence weighting
        t1_avg_rating = team1_stats.get('avg_player_rating', 0)
        t2_avg_rating = team2_stats.get('avg_player_rating', 0)
        
        weighted_t1_rating = t1_avg_rating * t1_player_confidence + 1.0 * (1 - t1_player_confidence)  # 1.0 is neutral rating
        weighted_t2_rating = t2_avg_rating * t2_player_confidence + 1.0 * (1 - t2_player_confidence)
        
        features['player_rating_diff'] = weighted_t1_rating - weighted_t2_rating
        features['raw_player_rating_diff'] = t1_avg_rating - t2_avg_rating
        features['better_player_rating_team1'] = 1 if weighted_t1_rating > weighted_t2_rating else 0
        features['avg_player_rating'] = (weighted_t1_rating + weighted_t2_rating) / 2
        
        feature_explanations['player_rating_diff'] = {
            'description': 'Difference in confidence-weighted player ratings',
            'team1_value': f"{weighted_t1_rating:.3f} (raw: {t1_avg_rating:.3f}, confidence: {t1_player_confidence:.2f})",
            'team2_value': f"{weighted_t2_rating:.3f} (raw: {t2_avg_rating:.3f}, confidence: {t2_player_confidence:.2f})",
            'importance': 'Medium'
        }
        
        # Team rating (might be different from avg player rating)
        if 'team_rating' in team1_stats and 'team_rating' in team2_stats:
            t1_team_rating = team1_stats.get('team_rating', 0)
            t2_team_rating = team2_stats.get('team_rating', 0)
            
            # Team ratings are often more robust than individual player ratings
            weighted_t1_team_rating = t1_team_rating * (0.7 + 0.3 * t1_reliability)
            weighted_t2_team_rating = t2_team_rating * (0.7 + 0.3 * t2_reliability)
            
            features['team_rating_diff'] = weighted_t1_team_rating - weighted_t2_team_rating
            features['raw_team_rating_diff'] = t1_team_rating - t2_team_rating
            features['better_team_rating_team1'] = 1 if weighted_t1_team_rating > weighted_t2_team_rating else 0
            features['avg_team_rating'] = (weighted_t1_team_rating + weighted_t2_team_rating) / 2
        
        # ACS (Average Combat Score)
        t1_acs = team1_stats.get('avg_player_acs', 0)
        t2_acs = team2_stats.get('avg_player_acs', 0)
        
        weighted_t1_acs = t1_acs * t1_player_confidence + 200.0 * (1 - t1_player_confidence)  # 200 is typical average
        weighted_t2_acs = t2_acs * t2_player_confidence + 200.0 * (1 - t2_player_confidence)
        
        features['acs_diff'] = weighted_t1_acs - weighted_t2_acs
        features['raw_acs_diff'] = t1_acs - t2_acs
        features['better_acs_team1'] = 1 if weighted_t1_acs > weighted_t2_acs else 0
        features['avg_acs'] = (weighted_t1_acs + weighted_t2_acs) / 2
        
        # K/D Ratio
        t1_kd = team1_stats.get('avg_player_kd', 0)
        t2_kd = team2_stats.get('avg_player_kd', 0)
        
        weighted_t1_kd = t1_kd * t1_player_confidence + 1.0 * (1 - t1_player_confidence)  # 1.0 is neutral K/D
        weighted_t2_kd = t2_kd * t2_player_confidence + 1.0 * (1 - t2_player_confidence)
        
        features['kd_diff'] = weighted_t1_kd - weighted_t2_kd
        features['raw_kd_diff'] = t1_kd - t2_kd
        features['better_kd_team1'] = 1 if weighted_t1_kd > weighted_t2_kd else 0
        features['avg_kd'] = (weighted_t1_kd + weighted_t2_kd) / 2
        
        # KAST (Kill, Assist, Survive, Trade)
        t1_kast = team1_stats.get('avg_player_kast', 0)
        t2_kast = team2_stats.get('avg_player_kast', 0)
        
        weighted_t1_kast = t1_kast * t1_player_confidence + 0.65 * (1 - t1_player_confidence)  # 65% is typical
        weighted_t2_kast = t2_kast * t2_player_confidence + 0.65 * (1 - t2_player_confidence)
        
        features['kast_diff'] = weighted_t1_kast - weighted_t2_kast
        features['raw_kast_diff'] = t1_kast - t2_kast
        features['better_kast_team1'] = 1 if weighted_t1_kast > weighted_t2_kast else 0
        features['avg_kast'] = (weighted_t1_kast + weighted_t2_kast) / 2
        
        # ADR (Average Damage per Round)
        t1_adr = team1_stats.get('avg_player_adr', 0)
        t2_adr = team2_stats.get('avg_player_adr', 0)
        
        weighted_t1_adr = t1_adr * t1_player_confidence + 140.0 * (1 - t1_player_confidence)  # 140 is typical
        weighted_t2_adr = t2_adr * t2_player_confidence + 140.0 * (1 - t2_player_confidence)
        
        features['adr_diff'] = weighted_t1_adr - weighted_t2_adr
        features['raw_adr_diff'] = t1_adr - t2_adr
        features['better_adr_team1'] = 1 if weighted_t1_adr > weighted_t2_adr else 0
        features['avg_adr'] = (weighted_t1_adr + weighted_t2_adr) / 2
        
        # Headshot percentage
        t1_hs = team1_stats.get('avg_player_headshot', 0)
        t2_hs = team2_stats.get('avg_player_headshot', 0)
        
        weighted_t1_hs = t1_hs * t1_player_confidence + 0.25 * (1 - t1_player_confidence)  # 25% is typical
        weighted_t2_hs = t2_hs * t2_player_confidence + 0.25 * (1 - t2_player_confidence)
        
        features['headshot_diff'] = weighted_t1_hs - weighted_t2_hs
        features['raw_headshot_diff'] = t1_hs - t2_hs
        features['better_headshot_team1'] = 1 if weighted_t1_hs > weighted_t2_hs else 0
        features['avg_headshot'] = (weighted_t1_hs + weighted_t2_hs) / 2
        
        # Star player rating
        t1_star = team1_stats.get('star_player_rating', 0)
        t2_star = team2_stats.get('star_player_rating', 0)
        
        weighted_t1_star = t1_star * t1_player_confidence + 1.2 * (1 - t1_player_confidence)  # 1.2 is typical star player
        weighted_t2_star = t2_star * t2_player_confidence + 1.2 * (1 - t2_player_confidence)
        
        features['star_player_diff'] = weighted_t1_star - weighted_t2_star
        features['raw_star_player_diff'] = t1_star - t2_star
        features['better_star_player_team1'] = 1 if weighted_t1_star > weighted_t2_star else 0
        features['avg_star_player'] = (weighted_t1_star + weighted_t2_star) / 2
        
        # Team consistency (star player vs. worst player)
        t1_consistency = team1_stats.get('team_consistency', 0)
        t2_consistency = team2_stats.get('team_consistency', 0)
        
        weighted_t1_consistency = t1_consistency * t1_player_confidence + 0.7 * (1 - t1_player_confidence)  # 0.7 is typical
        weighted_t2_consistency = t2_consistency * t2_player_confidence + 0.7 * (1 - t2_player_confidence)
        
        features['consistency_diff'] = weighted_t1_consistency - weighted_t2_consistency
        features['raw_consistency_diff'] = t1_consistency - t2_consistency
        features['better_consistency_team1'] = 1 if weighted_t1_consistency > weighted_t2_consistency else 0
        features['avg_consistency'] = (weighted_t1_consistency + weighted_t2_consistency) / 2
        
        # First Kill / First Death ratio
        t1_fk_fd = team1_stats.get('fk_fd_ratio', 0)
        t2_fk_fd = team2_stats.get('fk_fd_ratio', 0)
        
        weighted_t1_fk_fd = t1_fk_fd * t1_player_confidence + 1.0 * (1 - t1_player_confidence)  # 1.0 is neutral
        weighted_t2_fk_fd = t2_fk_fd * t2_player_confidence + 1.0 * (1 - t2_player_confidence)
        
        features['fk_fd_diff'] = weighted_t1_fk_fd - weighted_t2_fk_fd
        features['raw_fk_fd_diff'] = t1_fk_fd - t2_fk_fd
        features['better_fk_fd_team1'] = 1 if weighted_t1_fk_fd > weighted_t2_fk_fd else 0
        features['avg_fk_fd'] = (weighted_t1_fk_fd + weighted_t2_fk_fd) / 2
    
    #----------------------------------------
    # 3. ECONOMY STATS WITH RELIABILITY ADJUSTMENTS
    #----------------------------------------
    # Only add if both teams have the data
    if ('pistol_win_rate' in team1_stats and 'pistol_win_rate' in team2_stats and
        team1_stats.get('pistol_win_rate', 0) > 0 and team2_stats.get('pistol_win_rate', 0) > 0):
        
        # Economy confidence factors
        t1_economy_confidence = team1_stats.get('economy_confidence', 0.5)
        t2_economy_confidence = team2_stats.get('economy_confidence', 0.5)
        
        # Pistol win rate
        t1_pistol = team1_stats.get('pistol_win_rate', 0)
        t2_pistol = team2_stats.get('pistol_win_rate', 0)
        
        weighted_t1_pistol = t1_pistol * t1_economy_confidence + 0.5 * (1 - t1_economy_confidence)  # 0.5 is neutral
        weighted_t2_pistol = t2_pistol * t2_economy_confidence + 0.5 * (1 - t2_economy_confidence)
        
        features['pistol_win_rate_diff'] = weighted_t1_pistol - weighted_t2_pistol
        features['raw_pistol_win_rate_diff'] = t1_pistol - t2_pistol
        features['better_pistol_team1'] = 1 if weighted_t1_pistol > weighted_t2_pistol else 0
        features['avg_pistol_win_rate'] = (weighted_t1_pistol + weighted_t2_pistol) / 2
        
        feature_explanations['pistol_win_rate_diff'] = {
            'description': 'Difference in confidence-weighted pistol round win rates',
            'team1_value': f"{weighted_t1_pistol:.3f} (raw: {t1_pistol:.3f}, confidence: {t1_economy_confidence:.2f})",
            'team2_value': f"{weighted_t2_pistol:.3f} (raw: {t2_pistol:.3f}, confidence: {t2_economy_confidence:.2f})",
            'importance': 'Medium'
        }
        
        # Eco win rate
        t1_eco = team1_stats.get('eco_win_rate', 0)
        t2_eco = team2_stats.get('eco_win_rate', 0)
        
        weighted_t1_eco = t1_eco * t1_economy_confidence + 0.2 * (1 - t1_economy_confidence)  # 0.2 is typical
        weighted_t2_eco = t2_eco * t2_economy_confidence + 0.2 * (1 - t2_economy_confidence)
        
        features['eco_win_rate_diff'] = weighted_t1_eco - weighted_t2_eco
        features['raw_eco_win_rate_diff'] = t1_eco - t2_eco
        features['better_eco_team1'] = 1 if weighted_t1_eco > weighted_t2_eco else 0
        features['avg_eco_win_rate'] = (weighted_t1_eco + weighted_t2_eco) / 2
        
        # Full-buy win rate
        t1_full_buy = team1_stats.get('full_buy_win_rate', 0)
        t2_full_buy = team2_stats.get('full_buy_win_rate', 0)
        
        weighted_t1_full_buy = t1_full_buy * t1_economy_confidence + 0.5 * (1 - t1_economy_confidence)
        weighted_t2_full_buy = t2_full_buy * t2_economy_confidence + 0.5 * (1 - t2_economy_confidence)
        
        features['full_buy_win_rate_diff'] = weighted_t1_full_buy - weighted_t2_full_buy
        features['raw_full_buy_win_rate_diff'] = t1_full_buy - t2_full_buy
        features['better_full_buy_team1'] = 1 if weighted_t1_full_buy > weighted_t2_full_buy else 0
        features['avg_full_buy_win_rate'] = (weighted_t1_full_buy + weighted_t2_full_buy) / 2
        
        # Economy efficiency
        t1_efficiency = team1_stats.get('economy_efficiency', 0)
        t2_efficiency = team2_stats.get('economy_efficiency', 0)
        
        weighted_t1_efficiency = t1_efficiency * t1_economy_confidence + 0.5 * (1 - t1_economy_confidence)
        weighted_t2_efficiency = t2_efficiency * t2_economy_confidence + 0.5 * (1 - t2_economy_confidence)
        
        features['economy_efficiency_diff'] = weighted_t1_efficiency - weighted_t2_efficiency
        features['raw_economy_efficiency_diff'] = t1_efficiency - t2_efficiency
        features['better_economy_efficiency_team1'] = 1 if weighted_t1_efficiency > weighted_t2_efficiency else 0
        features['avg_economy_efficiency'] = (weighted_t1_efficiency + weighted_t2_efficiency) / 2
    
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
        
        # Calculate map pool advantage
        if len(team1_maps) > 0 and len(team2_maps) > 0:
            t1_unique_maps = len(team1_maps - team2_maps)
            t2_unique_maps = len(team2_maps - team1_maps)
            
            features['map_pool_advantage'] = t1_unique_maps - t2_unique_maps
            features['map_pool_overlap_percent'] = len(common_maps) / len(team1_maps.union(team2_maps))
        
        # Prepare map-specific comparisons
        map_win_rate_diffs = []
        
        for map_name in common_maps:
            t1_map = team1_stats['map_statistics'][map_name]
            t2_map = team2_stats['map_statistics'][map_name]
            
            # Clean map name for feature naming
            map_key = map_name.replace(' ', '_').lower()
            
            # Calculate match counts for reliability
            t1_map_matches = t1_map.get('matches_played', 0)
            t2_map_matches = t2_map.get('matches_played', 0)
            
            # Map-specific reliability factors
            t1_map_reliability = 1.0 - (1.0 / (1.0 + 0.2 * t1_map_matches))
            t2_map_reliability = 1.0 - (1.0 / (1.0 + 0.2 * t2_map_matches))
            
            # Win percentage comparison with reliability weighting
            t1_win_pct = t1_map.get('win_percentage', 0)
            t2_win_pct = t2_map.get('win_percentage', 0)
            
            weighted_t1_win_pct = t1_win_pct * t1_map_reliability + 0.5 * (1 - t1_map_reliability)
            weighted_t2_win_pct = t2_win_pct * t2_map_reliability + 0.5 * (1 - t2_map_reliability)
            
            win_rate_diff = weighted_t1_win_pct - weighted_t2_win_pct
            map_win_rate_diffs.append(win_rate_diff)
            
            features[f'{map_key}_win_rate_diff'] = win_rate_diff
            features[f'raw_{map_key}_win_rate_diff'] = t1_win_pct - t2_win_pct
            features[f'better_{map_key}_team1'] = 1 if weighted_t1_win_pct > weighted_t2_win_pct else 0
            
            # Side performance comparison
            if 'side_preference' in t1_map and 'side_preference' in t2_map:
                features[f'{map_key}_side_pref_diff'] = (1 if t1_map['side_preference'] == 'Attack' else -1) - (1 if t2_map['side_preference'] == 'Attack' else -1)
                
                # Attack win rates
                t1_atk = t1_map.get('atk_win_rate', 0)
                t2_atk = t2_map.get('atk_win_rate', 0)
                
                weighted_t1_atk = t1_atk * t1_map_reliability + 0.45 * (1 - t1_map_reliability)  # 45% is typical
                weighted_t2_atk = t2_atk * t2_map_reliability + 0.45 * (1 - t2_map_reliability)
                
                features[f'{map_key}_atk_win_rate_diff'] = weighted_t1_atk - weighted_t2_atk
                
                # Defense win rates
                t1_def = t1_map.get('def_win_rate', 0)
                t2_def = t2_map.get('def_win_rate', 0)
                
                weighted_t1_def = t1_def * t1_map_reliability + 0.55 * (1 - t1_map_reliability)  # 55% is typical
                weighted_t2_def = t2_def * t2_map_reliability + 0.55 * (1 - t2_map_reliability)
                
                features[f'{map_key}_def_win_rate_diff'] = weighted_t1_def - weighted_t2_def
        
        # Calculate overall map advantage
        if map_win_rate_diffs:
            features['avg_map_win_rate_diff'] = sum(map_win_rate_diffs) / len(map_win_rate_diffs)
            features['max_map_win_rate_diff'] = max(map_win_rate_diffs)
            features['min_map_win_rate_diff'] = min(map_win_rate_diffs)
            features['map_advantage_team1'] = 1 if features['avg_map_win_rate_diff'] > 0 else 0
    
    #----------------------------------------
    # 5. HEAD-TO-HEAD STATS WITH ENHANCED WEIGHTING
    #----------------------------------------
    # Add head-to-head statistics with improved matching
    team1_name = team1_stats.get('team_name', '')
    team2_name = team2_stats.get('team_name', '')
    h2h_found = False
    h2h_stats = None
    
    # Create variations of team names for matching
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
                if verbose:
                    print(f"Found exact match head-to-head data: {opponent_name}")
                break
                
            # Then check variations
            for variation in team2_variations:
                if (opponent_name.lower() == variation.lower() or
                    variation.lower() in opponent_name.lower() or
                    opponent_name.lower() in variation.lower()):
                    h2h_stats = stats
                    h2h_found = True
                    if verbose:
                        print(f"Found variation match head-to-head data: {opponent_name} ↔ {variation}")
                    break
            
            if h2h_found:
                break

    # Initialize default head-to-head statistics
    features['h2h_win_rate'] = 0.5  # Neutral value
    features['h2h_matches'] = 0
    features['h2h_score_diff'] = 0
    features['h2h_advantage_team1'] = 0
    features['h2h_significant'] = 0
    
    # Add head-to-head features if found
    if h2h_found and h2h_stats:
        h2h_win_rate = h2h_stats.get('win_rate', 0.5)
        h2h_matches = h2h_stats.get('matches', 0)
        h2h_score_diff = h2h_stats.get('score_differential', 0)
        
        # Calculate head-to-head significance factor
        # More matches = more significant
        # Extreme win rates (close to 0 or 1) = more significant
        h2h_significance = min(1.0, h2h_matches / 5.0) * (1.0 + 2.0 * abs(h2h_win_rate - 0.5))
        
        # Extreme head-to-head records get extra weight
        # E.g., 3-0, 4-0, 5-0 records are very significant
        if h2h_matches >= 3 and (h2h_win_rate >= 0.8 or h2h_win_rate <= 0.2):
            h2h_significance *= 1.5
        
        features['h2h_win_rate'] = h2h_win_rate
        features['h2h_matches'] = h2h_matches
        features['h2h_score_diff'] = h2h_score_diff
        features['h2h_advantage_team1'] = 1 if h2h_win_rate > 0.5 else 0
        features['h2h_significant'] = 1 if h2h_significance >= 0.6 else 0
        features['h2h_significance_factor'] = h2h_significance
        
        feature_explanations['h2h_stats'] = {
            'description': 'Head-to-head record between the teams',
            'matches': h2h_matches,
            'win_rate': f"{h2h_win_rate:.3f} ({int(h2h_win_rate * h2h_matches)}-{int(h2h_matches - h2h_win_rate * h2h_matches)})",
            'score_diff': f"{h2h_score_diff:.2f}",
            'significance': f"{h2h_significance:.2f}",
            'importance': 'Very High' if h2h_significance >= 0.8 else 'High' if h2h_significance >= 0.5 else 'Medium'
        }
        
        if verbose:
            print(f"Using head-to-head data: Matches={features['h2h_matches']}, "
                  f"Win rate={features['h2h_win_rate']:.4f}, "
                  f"Score diff={features['h2h_score_diff']:.4f}, "
                  f"Significance={h2h_significance:.2f}, "
                  f"Advantage={'Team1' if features['h2h_advantage_team1'] else 'Team2'}")
    else:
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
                h2h_win_rate = wins / len(h2h_matches)
                total_score = sum(match.get('team_score', 0) for match in h2h_matches)
                total_opponent_score = sum(match.get('opponent_score', 0) for match in h2h_matches)
                avg_score_diff = (total_score - total_opponent_score) / len(h2h_matches)
                
                # Calculate significance as above
                h2h_significance = min(1.0, len(h2h_matches) / 5.0) * (1.0 + 2.0 * abs(h2h_win_rate - 0.5))
                if len(h2h_matches) >= 3 and (h2h_win_rate >= 0.8 or h2h_win_rate <= 0.2):
                    h2h_significance *= 1.5
                
                features['h2h_win_rate'] = h2h_win_rate
                features['h2h_matches'] = len(h2h_matches)
                features['h2h_score_diff'] = avg_score_diff
                features['h2h_advantage_team1'] = 1 if h2h_win_rate > 0.5 else 0
                features['h2h_significant'] = 1 if h2h_significance >= 0.6 else 0
                features['h2h_significance_factor'] = h2h_significance
                
                feature_explanations['h2h_stats'] = {
                    'description': 'Reconstructed head-to-head record between the teams',
                    'matches': len(h2h_matches),
                    'win_rate': f"{h2h_win_rate:.3f} ({wins}-{len(h2h_matches) - wins})",
                    'score_diff': f"{avg_score_diff:.2f}",
                    'significance': f"{h2h_significance:.2f}",
                    'importance': 'Very High' if h2h_significance >= 0.8 else 'High' if h2h_significance >= 0.5 else 'Medium'
                }
                
                if verbose:
                    print(f"Manually reconstructed head-to-head data: Matches={features['h2h_matches']}, "
                          f"Win rate={features['h2h_win_rate']:.4f}, "
                          f"Score diff={features['h2h_score_diff']:.4f}, "
                          f"Significance={h2h_significance:.2f}, "
                          f"Advantage={'Team1' if features['h2h_advantage_team1'] else 'Team2'}")
    
    #----------------------------------------
    # 6. INTERACTION TERMS AND FEATURE COMBINATIONS
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
    
    # H2H interactions - these are particularly important
    if features['h2h_matches'] > 0:
        # Amplify win rate difference if team has a significant head-to-head advantage
        h2h_amplifier = 1.0 + (features['h2h_significance_factor'] * abs(features['h2h_win_rate'] - 0.5))
        
        if 'win_rate_diff' in features:
            # Scale based on head-to-head advantage direction
            h2h_win_rate_mod = (features['h2h_win_rate'] - 0.5) * 2  # Range from -1 to 1
            features['h2h_x_win_rate'] = features['win_rate_diff'] * h2h_amplifier * (1 + h2h_win_rate_mod)
        
        if 'recent_form_diff' in features:
            features['h2h_x_form'] = features['recent_form_diff'] * h2h_amplifier * (1 + h2h_win_rate_mod)
        
        # Create combined head-to-head feature that captures the overall significance
        if features['h2h_advantage_team1']:
            features['significant_h2h_advantage_team1'] = features['h2h_significance_factor'] 
        else:
            features['significant_h2h_advantage_team1'] = -features['h2h_significance_factor']
    
    #----------------------------------------
    # 7. SAMPLE SIZE RELIABILITY METRICS
    #----------------------------------------
    # Add features specifically about sample size reliability
    features['team1_data_confidence'] = team1_stats.get('data_confidence', min(1.0, team1_match_count / 20.0))
    features['team2_data_confidence'] = team2_stats.get('data_confidence', min(1.0, team2_match_count / 20.0))
    features['overall_data_confidence'] = (features['team1_data_confidence'] + features['team2_data_confidence']) / 2
    
    # Add flags for highly uncertain predictions due to limited data
    features['prediction_confidence_flag'] = 1 if features['overall_data_confidence'] >= 0.7 else 0
    features['severe_data_limitation'] = 1 if features['overall_data_confidence'] < 0.4 else 0
    
    # Calculate whether a team has a significant advantage in match experience
    if team1_match_count > 2 * team2_match_count:
        features['match_experience_advantage'] = 1  # Team1 has much more experience
    elif team2_match_count > 2 * team1_match_count:
        features['match_experience_advantage'] = -1  # Team2 has much more experience
    else:
        features['match_experience_advantage'] = 0  # Relatively balanced
    
    # Clean up non-numeric values
    for key, value in list(features.items()):
        if not isinstance(value, (int, float)):
            if verbose:
                print(f"Non-numeric feature value: {key}={value}, type={type(value)}")
            # Try to convert to numeric if possible
            try:
                features[key] = float(value)
            except (ValueError, TypeError):
                if verbose:
                    print(f"Removing non-numeric feature: {key}")
                del features[key]
    
    # Log summary of all critical features if requested
    if verbose:
        print("\nCritical Feature Summary:")
        for key in ['win_rate_diff', 'recent_form_diff', 'score_diff_differential', 'h2h_win_rate', 
                   'h2h_matches', 'h2h_significance_factor', 'overall_data_confidence']:
            if key in features:
                print(f"  {key}: {features[key]:.4f}")
    
    return features, feature_explanations

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

def load_prediction_artifacts(artifact_paths=None):
    """
    Load all artifacts required for prediction.
    
    Args:
        artifact_paths (dict, optional): Dictionary with artifact file paths
    
    Returns:
        tuple: (ensemble_models, scaler, feature_mask, feature_metadata, stable_features)
    """
    # Define default paths if not provided
    if not artifact_paths:
        artifact_paths = {
            'models': ['valorant_model_fold_{}.h5'.format(i+1) for i in range(10)],
            'scaler': 'ensemble_scaler.pkl',
            'feature_mask': 'feature_mask.pkl',
            'feature_metadata': 'feature_metadata.pkl',
            'stable_features': 'stable_features.pkl'
        }
    
    print("Loading prediction artifacts...")
    
    # Load models
    ensemble_models = []
    for model_path in artifact_paths['models']:
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                ensemble_models.append(model)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
    
    # Load scaler
    scaler = None
    if os.path.exists(artifact_paths['scaler']):
        try:
            with open(artifact_paths['scaler'], 'rb') as f:
                scaler = pickle.load(f)
            print(f"Loaded feature scaler from {artifact_paths['scaler']}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
    
    # Load feature mask
    feature_mask = None
    if os.path.exists(artifact_paths['feature_mask']):
        try:
            with open(artifact_paths['feature_mask'], 'rb') as f:
                feature_mask = pickle.load(f)
            print(f"Loaded feature mask from {artifact_paths['feature_mask']}")
        except Exception as e:
            print(f"Error loading feature mask: {e}")
    
    # Load feature metadata
    feature_metadata = None
    if os.path.exists(artifact_paths['feature_metadata']):
        try:
            with open(artifact_paths['feature_metadata'], 'rb') as f:
                feature_metadata = pickle.load(f)
            print(f"Loaded feature metadata from {artifact_paths['feature_metadata']}")
        except Exception as e:
            print(f"Error loading feature metadata: {e}")
    
    # Load stable features
    stable_features = None
    if os.path.exists(artifact_paths['stable_features']):
        try:
            with open(artifact_paths['stable_features'], 'rb') as f:
                stable_features = pickle.load(f)
            print(f"Loaded {len(stable_features)} stable features")
        except Exception as e:
            print(f"Error loading stable features: {e}")
    
    return ensemble_models, scaler, feature_mask, feature_metadata, stable_features

def prepare_match_features(team1_stats, team2_stats, required_features, scaler, feature_mask=None, verbose=False):
    """
    Prepare features for a single match for prediction using only the features
    that were selected during training to avoid overfitting.
    
    Args:
        team1_stats (dict): Statistics for team 1
        team2_stats (dict): Statistics for team 2
        required_features (list): List of required feature names
        scaler (object): Feature scaler
        feature_mask (array, optional): Boolean mask for feature selection
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (X_scaled, features_dict, feature_explanations)
    """
    # Get full feature set
    features, feature_explanations = prepare_data_for_model(team1_stats, team2_stats, verbose=verbose)
    
    if not features:
        print("Failed to prepare match features.")
        return None, None, None
    
    # Convert to DataFrame for easier feature selection
    features_df = pd.DataFrame([features])
    if verbose:
        print(f"Original feature count: {len(features_df.columns)}")
    
    # If required_features exists, use only those features
    if required_features and len(required_features) > 0:
        # Ensure all required features are present
        features_df = ensure_consistent_features(features_df, required_features)
    else:
        print("WARNING: No required features provided. Using all available features.")
    
    # Apply feature mask if provided
    if feature_mask is not None and len(feature_mask) == len(features_df.columns):
        features_df = features_df.iloc[:, feature_mask]
        if verbose:
            print(f"Applied feature mask. Reduced to {len(features_df.columns)} features")
    
    # Convert to numpy array for scaling
    X = features_df.values
    if verbose:
        print(f"Feature array shape after selection: {X.shape}")
    
    # Scale features if scaler is available
    if scaler:
        try:
            X_scaled = scaler.transform(X)
            return X_scaled, features, feature_explanations
        except Exception as e:
            print(f"Error scaling features: {e}")
            return X, features, feature_explanations  # Return unscaled features as fallback
    else:
        print("WARNING: No scaler provided. Using unscaled features.")
        return X, features, feature_explanations

def analyze_contradictions(features, feature_explanations):
    """
    Analyze the feature vector for potential contradictions or anomalies
    that might affect prediction quality.
    
    Args:
        features (dict): Dictionary of features used for prediction
        feature_explanations (dict): Dictionary explaining feature meanings
    
    Returns:
        dict: Dictionary of detected contradictions or anomalies
    """
    contradictions = {
        'detected': False,
        'issues': [],
        'confidence_penalty': 0.0
    }
    
    # Check for sample size issues
    if features.get('severe_data_limitation', 0) == 1:
        contradictions['detected'] = True
        contradictions['issues'].append({
            'type': 'small_sample',
            'severity': 'high',
            'description': 'Limited data available for one or both teams',
            'details': f"Data confidence: {features.get('overall_data_confidence', 0):.2f}"
        })
        contradictions['confidence_penalty'] += 0.3
    
    # Check for head-to-head contradictions
    # If head-to-head strongly favors one team (high significance) but 
    # other stats point in the opposite direction
    if features.get('h2h_significance_factor', 0) > 0.7:
        h2h_favors_team1 = features.get('h2h_advantage_team1', 0) == 1
        
        # Combine key performance indicators
        performance_indicators = [
            features.get('win_rate_diff', 0),
            features.get('recent_form_diff', 0),
            features.get('score_diff_differential', 0)
        ]
        
        # Determine if performance indicators favor team1
        perf_favors_team1 = sum(1 for ind in performance_indicators if ind > 0) > len(performance_indicators) / 2
        
        # Check for contradiction
        if h2h_favors_team1 != perf_favors_team1:
            contradictions['detected'] = True
            contradictions['issues'].append({
                'type': 'h2h_contradiction',
                'severity': 'medium',
                'description': f"Head-to-head record favors {'Team 1' if h2h_favors_team1 else 'Team 2'} but overall performance favors {'Team 2' if h2h_favors_team1 else 'Team 1'}",
                'details': f"H2H significance: {features.get('h2h_significance_factor', 0):.2f}, "
                           f"Win rate diff: {features.get('win_rate_diff', 0):.3f}, "
                           f"Recent form diff: {features.get('recent_form_diff', 0):.3f}"
            })
            contradictions['confidence_penalty'] += 0.2
    
    # Check for extreme statistical advantage with significant h2h disadvantage
    if abs(features.get('win_rate_diff', 0)) > 0.3 and features.get('h2h_significance_factor', 0) > 0.6:
        win_rate_favors_team1 = features.get('win_rate_diff', 0) > 0
        h2h_favors_team1 = features.get('h2h_advantage_team1', 0) == 1
        
        if win_rate_favors_team1 != h2h_favors_team1:
            contradictions['detected'] = True
            contradictions['issues'].append({
                'type': 'statistical_h2h_contradiction',
                'severity': 'high',
                'description': f"Strong win rate advantage for {'Team 1' if win_rate_favors_team1 else 'Team 2'} contradicts significant head-to-head advantage for {'Team 1' if h2h_favors_team1 else 'Team 2'}",
                'details': f"Win rate diff: {features.get('win_rate_diff', 0):.3f}, "
                          f"H2H record: {feature_explanations.get('h2h_stats', {}).get('win_rate', 'N/A')}"
            })
            contradictions['confidence_penalty'] += 0.25
    
    # Check for sample size imbalance
    if abs(features.get('match_count_diff', 0)) > 15:
        more_experienced = "Team 1" if features.get('match_count_diff', 0) > 0 else "Team 2"
        contradictions['detected'] = True
        contradictions['issues'].append({
            'type': 'sample_size_imbalance',
            'severity': 'medium',
            'description': f"{more_experienced} has significantly more match data available",
            'details': f"Match count difference: {abs(features.get('match_count_diff', 0))}"
        })
        contradictions['confidence_penalty'] += 0.15
    
    # Check for recent form vs overall performance contradiction
    if (features.get('recent_form_diff', 0) * features.get('win_rate_diff', 0) < 0 and
        abs(features.get('recent_form_diff', 0)) > 0.2 and
        abs(features.get('win_rate_diff', 0)) > 0.1):
        
        recent_form_favors = "Team 1" if features.get('recent_form_diff', 0) > 0 else "Team 2"
        overall_favors = "Team 1" if features.get('win_rate_diff', 0) > 0 else "Team 2"
        
        contradictions['detected'] = True
        contradictions['issues'].append({
            'type': 'form_contradiction',
            'severity': 'low',
            'description': f"Recent form favors {recent_form_favors} but overall record favors {overall_favors}",
            'details': f"Recent form diff: {features.get('recent_form_diff', 0):.3f}, "
                      f"Win rate diff: {features.get('win_rate_diff', 0):.3f}"
        })
        contradictions['confidence_penalty'] += 0.1
    
    # Cap total confidence penalty
    contradictions['confidence_penalty'] = min(contradictions['confidence_penalty'], 0.7)
    
    return contradictions

def predict_match_with_ensemble(X, ensemble_models, team1_name, team2_name, 
                             features_dict, feature_explanations, 
                             override=None, verbose=False):
    """
    Make a prediction for a single match using an ensemble of models.
    
    Args:
        X (array): Scaled feature array for prediction
        ensemble_models (list): List of trained models
        team1_name (str): Name of team 1
        team2_name (str): Name of team 2
        features_dict (dict): Dictionary of features used for prediction
        feature_explanations (dict): Dictionary explaining feature meanings
        override (dict, optional): Dictionary specifying prediction override
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Prediction results including probabilities and analysis
    """
    if not ensemble_models:
        print("No models available for prediction.")
        return None
    
    # Check for contradictions
    contradictions = analyze_contradictions(features_dict, feature_explanations)
    
    # Get ensemble predictions
    probabilities = []
    individual_preds = []
    
    for i, model in enumerate(ensemble_models):
        try:
            prob = model.predict(X)[0][0]
            probabilities.append(prob)
            individual_preds.append({
                'model': f"Model {i+1}",
                'team1_prob': prob,
                'team2_prob': 1 - prob,
                'predicted_winner': team1_name if prob > 0.5 else team2_name
            })
            if verbose:
                print(f"Model {i+1} prediction: {prob:.4f} ({team1_name if prob > 0.5 else team2_name})")
        except Exception as e:
            print(f"Error with model {i+1}: {e}")
    
    if not probabilities:
        print("All models failed to predict.")
        return None
    
    # Calculate ensemble prediction
    # Use weighted averaging with confidence scores
    avg_prob = sum(probabilities) / len(probabilities)
    
    # Calculate confidence based on:
    # 1. Agreement among models (lower standard deviation = higher confidence)
    # 2. Extremity of prediction (closer to 0 or 1 = higher confidence)
    # 3. Any detected contradictions
    model_std = np.std(probabilities)
    prediction_extremity = abs(avg_prob - 0.5) * 2  # 0 to 1 scale
    
    # Base confidence score
    base_confidence = (1.0 - model_std) * (0.5 + prediction_extremity)
    
    # Apply penalties from contradictions
    confidence = max(0.01, base_confidence - contradictions['confidence_penalty'])
    
    # Determine prediction
    team1_prob = avg_prob
    team2_prob = 1 - avg_prob
    
    if override and 'winner' in override:
        predicted_winner = override['winner']
        forced_override = True
        override_reason = override.get('reason', 'Manual override')
    else:
        predicted_winner = team1_name if avg_prob > 0.5 else team2_name
        forced_override = False
        override_reason = None
    
    # Check for statistical dominance
    win_rate_diff = features_dict.get('raw_win_rate_diff', 0)
    recent_form_diff = features_dict.get('raw_recent_form_diff', 0)
    player_rating_diff = features_dict.get('raw_player_rating_diff', 0) if 'raw_player_rating_diff' in features_dict else 0
    h2h_win_rate = features_dict.get('h2h_win_rate', 0.5)
    
    # Team1 statistical advantages
    team1_advantages = []
    if win_rate_diff > 0.1:
        team1_advantages.append(f"Better win rate: {(win_rate_diff * 100):.1f}% higher")
    if recent_form_diff > 0.1:
        team1_advantages.append(f"Better recent form: {(recent_form_diff * 100):.1f}% higher")
    if player_rating_diff > 0.1:
        team1_advantages.append(f"Higher player rating: {player_rating_diff:.2f} better")
    if h2h_win_rate > 0.6 and features_dict.get('h2h_matches', 0) >= 3:
        team1_advantages.append(f"Head-to-head advantage: {feature_explanations.get('h2h_stats', {}).get('win_rate', 'N/A')}")
    
    # Team2 statistical advantages
    team2_advantages = []
    if win_rate_diff < -0.1:
        team2_advantages.append(f"Better win rate: {(-win_rate_diff * 100):.1f}% higher")
    if recent_form_diff < -0.1:
        team2_advantages.append(f"Better recent form: {(-recent_form_diff * 100):.1f}% higher")
    if player_rating_diff < -0.1:
        team2_advantages.append(f"Higher player rating: {-player_rating_diff:.2f} better")
    if h2h_win_rate < 0.4 and features_dict.get('h2h_matches', 0) >= 3:
        team2_advantages.append(f"Head-to-head advantage: {feature_explanations.get('h2h_stats', {}).get('win_rate', 'N/A')}")
    
    # Find key factors influencing the prediction
    key_factors = []
    
    # Head-to-head record (if significant)
    if features_dict.get('h2h_matches', 0) >= 3 and features_dict.get('h2h_significance_factor', 0) > 0.5:
        h2h_info = feature_explanations.get('h2h_stats', {})
        key_factors.append({
            'factor': 'Head-to-head record',
            'influence': 'Very High' if features_dict.get('h2h_significance_factor', 0) > 0.7 else 'High',
            'description': f"{h2h_info.get('win_rate', 'N/A')} in {h2h_info.get('matches', 0)} matches",
            'favors': team1_name if features_dict.get('h2h_advantage_team1', 0) == 1 else team2_name
        })
    
    # Win rate difference
    if abs(features_dict.get('win_rate_diff', 0)) > 0.05:
        key_factors.append({
            'factor': 'Win rate',
            'influence': 'High',
            'description': f"{abs(features_dict.get('win_rate_diff', 0) * 100):.1f}% difference",
            'favors': team1_name if features_dict.get('win_rate_diff', 0) > 0 else team2_name
        })
    
    # Recent form
    if abs(features_dict.get('recent_form_diff', 0)) > 0.1:
        key_factors.append({
            'factor': 'Recent form',
            'influence': 'Medium',
            'description': f"{abs(features_dict.get('recent_form_diff', 0) * 100):.1f}% difference in recent matches",
            'favors': team1_name if features_dict.get('recent_form_diff', 0) > 0 else team2_name
        })
    
    # Player ratings (if available)
    if 'player_rating_diff' in features_dict and abs(features_dict.get('player_rating_diff', 0)) > 0.1:
        key_factors.append({
            'factor': 'Player skill rating',
            'influence': 'Medium',
            'description': f"{abs(features_dict.get('player_rating_diff', 0)):.2f} rating difference",
            'favors': team1_name if features_dict.get('player_rating_diff', 0) > 0 else team2_name
        })
    
    # Sample size imbalance
    if abs(features_dict.get('match_count_diff', 0)) > 10:
        key_factors.append({
            'factor': 'Match experience',
            'influence': 'Low',
            'description': f"{abs(features_dict.get('match_count_diff', 0))} more matches played",
            'favors': team1_name if features_dict.get('match_count_diff', 0) > 0 else team2_name
        })
    
    # Create prediction result
    prediction_result = {
        'team1': team1_name,
        'team2': team2_name,
        'team1_probability': team1_prob,
        'team2_probability': team2_prob,
        'predicted_winner': predicted_winner,
        'confidence': confidence,
        'model_agreement': 1.0 - model_std,
        'prediction_extremity': prediction_extremity,
        'individual_predictions': individual_preds,
        'contradictions': contradictions,
        'team1_advantages': team1_advantages,
        'team2_advantages': team2_advantages,
        'key_factors': key_factors,
        'override_applied': forced_override,
        'override_reason': override_reason
    }
    
    return prediction_result

def visualize_prediction(prediction_result, save_to_file=None):
    """
    Visualize the prediction result.
    
    Args:
        prediction_result (dict): Prediction result from predict_match_with_ensemble
        save_to_file (str, optional): Path to save the visualization
    """
    if not prediction_result:
        print("No prediction result to visualize.")
        return
    
    # Set up the plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Match Prediction: {prediction_result['team1']} vs {prediction_result['team2']}", fontsize=16)
    
    # Define colors
    team1_color = "#1f77b4"  # Blue
    team2_color = "#ff7f0e"  # Orange
    
    # Plot 1: Prediction probability
    axs[0, 0].bar([prediction_result['team1'], prediction_result['team2']], 
                [prediction_result['team1_probability'], prediction_result['team2_probability']],
                color=[team1_color, team2_color])
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_title("Win Probability")
    axs[0, 0].set_ylabel("Probability")
    
    # Highlight the predicted winner
    winner_idx = 0 if prediction_result['predicted_winner'] == prediction_result['team1'] else 1
    winner_prob = prediction_result['team1_probability'] if winner_idx == 0 else prediction_result['team2_probability']
    axs[0, 0].text(winner_idx, winner_prob + 0.05, "Predicted Winner", 
                ha='center', va='center', fontweight='bold')
    
    # Plot 2: Model agreement
    individual_probs = [pred['team1_prob'] for pred in prediction_result['individual_predictions']]
    axs[0, 1].violinplot(individual_probs, showmeans=True)
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].set_title("Model Distribution")
    axs[0, 1].set_ylabel("Team 1 Win Probability")
    axs[0, 1].set_xticks([1])
    axs[0, 1].set_xticklabels(["Ensemble Models"])
    axs[0, 1].axhline(0.5, color='r', linestyle='--', alpha=0.5)
    
    # Add agreement score and confidence
    agreement_score = round(prediction_result['model_agreement'] * 100)
    confidence_score = round(prediction_result['confidence'] * 100)
    axs[0, 1].text(1.3, 0.9, f"Model Agreement: {agreement_score}%\nConfidence: {confidence_score}%", 
                fontsize=10, va='center')
    
    # Plot 3: Key factors
    key_factors = prediction_result['key_factors']
    if key_factors:
        # Sort factors by influence
        influence_map = {"Very High": 4, "High": 3, "Medium": 2, "Low": 1}
        key_factors.sort(key=lambda x: influence_map.get(x['influence'], 0), reverse=True)
        
        factor_names = [f"{factor['factor']}" for factor in key_factors]
        factor_values = []
        factor_colors = []
        
        for factor in key_factors:
            if factor['favors'] == prediction_result['team1']:
                # Positive value = favors team1
                value = influence_map.get(factor['influence'], 0) / 4.0
                color = team1_color
            else:
                # Negative value = favors team2
                value = -influence_map.get(factor['influence'], 0) / 4.0
                color = team2_color
            factor_values.append(value)
            factor_colors.append(color)
        
        # Create horizontal bar chart
        bars = axs[1, 0].barh(factor_names, factor_values, color=factor_colors)
        axs[1, 0].set_xlim(-1.1, 1.1)
        axs[1, 0].set_title("Key Factors")
        
        # Add team labels
        axs[1, 0].text(-1.0, len(factor_names), f"Favors {prediction_result['team2']}", 
                    ha='center', fontsize=9)
        axs[1, 0].text(1.0, len(factor_names), f"Favors {prediction_result['team1']}", 
                    ha='center', fontsize=9)
        
        # Add description labels
        for i, bar in enumerate(bars):
            x_pos = 0.1 if factor_values[i] < 0 else -0.1
            alignment = 'left' if factor_values[i] < 0 else 'right'
            axs[1, 0].text(x_pos, i, key_factors[i]['description'],
                       va='center', ha=alignment, fontsize=8)
    else:
        axs[1, 0].text(0.5, 0.5, "No key factors identified", ha='center', va='center')
        axs[1, 0].set_title("Key Factors")
    
    # Plot 4: Contradictions and warnings
    contradictions = prediction_result['contradictions']
    if contradictions['detected']:
        axs[1, 1].axis('off')
        axs[1, 1].set_title("Warnings & Contradictions")
        
        warning_text = ""
        for i, issue in enumerate(contradictions['issues']):
            severity_marker = "⚠️" if issue['severity'] == 'high' else "!" if issue['severity'] == 'medium' else "-"
            warning_text += f"{severity_marker} {issue['description']}\n  {issue['details']}\n\n"
        
        if prediction_result['override_applied']:
            warning_text += f"⚠️ MANUAL OVERRIDE APPLIED: {prediction_result['override_reason']}\n"
        
        axs[1, 1].text(0.05, 0.95, warning_text, ha='left', va='top', fontsize=9,
                   wrap=True, transform=axs[1, 1].transAxes)
    else:
        axs[1, 1].axis('off')
        axs[1, 1].set_title("Statistical Advantages")
        
        # Team advantages
        advantage_text = f"{prediction_result['team1']} Advantages:\n"
        for adv in prediction_result['team1_advantages']:
            advantage_text += f"• {adv}\n"
        
        advantage_text += f"\n{prediction_result['team2']} Advantages:\n"
        for adv in prediction_result['team2_advantages']:
            advantage_text += f"• {adv}\n"
        
        axs[1, 1].text(0.05, 0.95, advantage_text, ha='left', va='top', fontsize=9,
                   transform=axs[1, 1].transAxes)
    
    plt.tight_layout()
    
    if save_to_file:
        plt.savefig(save_to_file)
        print(f"Visualization saved to {save_to_file}")
    
    plt.show()

def print_prediction_report(prediction_result):
    """
    Print a detailed prediction report to the console.
    
    Args:
        prediction_result (dict): Prediction result from predict_match_with_ensemble
    """
    if not prediction_result:
        print("No prediction result to report.")
        return
    
    # Create horizontal line
    hr = "-" * 70
    
    # Format team names
    team1 = prediction_result['team1']
    team2 = prediction_result['team2']
    
    # Format probabilities as percentages
    team1_prob = prediction_result['team1_probability'] * 100
    team2_prob = prediction_result['team2_probability'] * 100
    
    # Format confidence
    confidence = prediction_result['confidence'] * 100
    
    # Print header
    print(hr)
    print(f"VALORANT MATCH PREDICTION: {team1} vs {team2}")
    print(hr)
    
    # Print prediction
    print(f"PREDICTED WINNER: {prediction_result['predicted_winner']}")
    print(f"WIN PROBABILITY: {team1}: {team1_prob:.1f}% | {team2}: {team2_prob:.1f}%")
    print(f"PREDICTION CONFIDENCE: {confidence:.1f}%")
    print(hr)
    
    # Print key factors
    print("KEY FACTORS:")
    for factor in prediction_result['key_factors']:
        influence_marker = ""
        if factor['influence'] == "Very High":
            influence_marker = "★★★★"
        elif factor['influence'] == "High":
            influence_marker = "★★★"
        elif factor['influence'] == "Medium":
            influence_marker = "★★"
        elif factor['influence'] == "Low":
            influence_marker = "★"
        
        print(f"• {factor['factor']} {influence_marker}")
        print(f"  {factor['description']} (Favors {factor['favors']})")
    print(hr)
    
    # Print team advantages
    print(f"{team1} ADVANTAGES:")
    if prediction_result['team1_advantages']:
        for adv in prediction_result['team1_advantages']:
            print(f"• {adv}")
    else:
        print("• None identified")
    
    print(f"\n{team2} ADVANTAGES:")
    if prediction_result['team2_advantages']:
        for adv in prediction_result['team2_advantages']:
            print(f"• {adv}")
    else:
        print("• None identified")
    print(hr)
    
    # Print model agreement
    print("MODEL AGREEMENT:")
    team1_votes = sum(1 for pred in prediction_result['individual_predictions'] 
                    if pred['predicted_winner'] == team1)
    team2_votes = len(prediction_result['individual_predictions']) - team1_votes
    print(f"• {team1_votes} models predict {team1} to win")
    print(f"• {team2_votes} models predict {team2} to win")
    print(f"• Agreement score: {prediction_result['model_agreement'] * 100:.1f}%")
    print(hr)
    
    # Print contradictions and warnings
    contradictions = prediction_result['contradictions']
    if contradictions['detected']:
        print("WARNINGS & CONTRADICTIONS:")
        for issue in contradictions['issues']:
            severity = issue['severity'].upper()
            print(f"• [{severity}] {issue['description']}")
            print(f"  {issue['details']}")
        print(hr)
    
    # Print override if applied
    if prediction_result['override_applied']:
        print("MANUAL OVERRIDE APPLIED:")
        print(f"• {prediction_result['override_reason']}")
        print(hr)

#-------------------------------------------------------------------------
# VALORANT MATCH PREDICTOR MAIN CLASS
#-------------------------------------------------------------------------

class ValorantPredictor:
    """
    Main class for the Valorant match prediction system.
    Encapsulates model loading, data preparation, and prediction functionality.
    """
    
    def __init__(self, artifacts_path=None, load_models=True):
        """
        Initialize the predictor with required artifacts.
        
        Args:
            artifacts_path (str, optional): Path to the artifacts directory
            load_models (bool): Whether to load models on initialization
        """
        self.api_url = API_URL
        self.artifacts_path = artifacts_path or "."
        self.models = []
        self.scaler = None
        self.feature_mask = None
        self.feature_metadata = None
        self.stable_features = None
        self.required_features = None
        
        # Load artifacts if requested
        if load_models:
            self.load_artifacts()
    
    def load_artifacts(self):
        """Load all necessary prediction artifacts."""
        artifact_paths = {
            'models': [os.path.join(self.artifacts_path, f'valorant_model_fold_{i+1}.h5') for i in range(10)],
            'scaler': os.path.join(self.artifacts_path, 'ensemble_scaler.pkl'),
            'feature_mask': os.path.join(self.artifacts_path, 'feature_mask.pkl'),
            'feature_metadata': os.path.join(self.artifacts_path, 'feature_metadata.pkl'),
            'stable_features': os.path.join(self.artifacts_path, 'stable_features.pkl')
        }
        
        self.models, self.scaler, self.feature_mask, self.feature_metadata, self.stable_features = load_prediction_artifacts(artifact_paths)
        
        # Determine required features
        if self.feature_metadata and 'selected_features' in self.feature_metadata:
            self.required_features = self.feature_metadata['selected_features']
        elif self.stable_features:
            self.required_features = self.stable_features
        else:
            self.required_features = None
            print("WARNING: No feature list available. Predictions may be unreliable.")
    
    def fetch_team_data(self, team_name, region=None, include_player_stats=True, include_economy=True, include_maps=True):
        """
        Fetch all necessary data for a team.
        
        Args:
            team_name (str): Name of the team
            region (str, optional): Region code to narrow search
            include_player_stats (bool): Whether to include player statistics
            include_economy (bool): Whether to include economy data
            include_maps (bool): Whether to include map statistics
            
        Returns:
            dict: Team statistics data
        """
        print(f"\nFetching data for team: {team_name}")
        
        # Get team ID
        team_id = get_team_id(team_name, region)
        if not team_id:
            print(f"Failed to find team ID for {team_name}")
            return None
        
        # Get team details
        team_details, team_tag = fetch_team_details(team_id)
        if not team_details:
            print(f"Failed to fetch team details for {team_name}")
            return None
        
        # Get match history
        team_history = fetch_team_match_history(team_id)
        if not team_history:
            print(f"Failed to fetch match history for {team_name}")
            return None
        
        # Parse match data
        team_matches = parse_match_data(team_history, team_name)
        if not team_matches:
            print(f"No matches found for {team_name}")
            return None
        
        # Fetch player stats if requested
        team_player_stats = None
        if include_player_stats:
            team_player_stats = fetch_team_player_stats(team_id)
        
        # Calculate team stats
        team_stats = calculate_team_stats(
            team_matches, 
            team_player_stats, 
            include_economy=include_economy
        )
        
        # Add team tag and ID to stats
        team_stats['team_tag'] = team_tag
        team_stats['team_name'] = team_name
        team_stats['team_id'] = team_id
        
        # Fetch and add map statistics if requested
        if include_maps:
            map_stats = fetch_team_map_statistics(team_id)
            if map_stats:
                team_stats['map_statistics'] = map_stats
        
        # Add matches to the stats object
        team_stats['matches'] = team_matches
        
        print(f"Successfully collected data for {team_name} with {len(team_matches)} matches")
        return team_stats
    
    def predict_match(self, team1_name, team2_name, region1=None, region2=None, 
                  include_player_stats=True, include_economy=True, include_maps=True,
                  verbose=False, visualize=False, save_visualization=None, override=None):
        """
        Predict the outcome of a match between two teams.
        
        Args:
            team1_name (str): Name of the first team
            team2_name (str): Name of the second team
            region1 (str, optional): Region code for the first team
            region2 (str, optional): Region code for the second team
            include_player_stats (bool): Whether to include player statistics
            include_economy (bool): Whether to include economy data
            include_maps (bool): Whether to include map statistics
            verbose (bool): Whether to print detailed information
            visualize (bool): Whether to visualize the prediction
            save_visualization (str, optional): Path to save the visualization
            override (dict, optional): Dictionary specifying prediction override
            
        Returns:
            dict: Prediction results including probabilities and analysis
        """
        # Check if models are loaded
        if not self.models:
            print("No models loaded. Loading artifacts...")
            self.load_artifacts()
            if not self.models:
                print("Failed to load models. Aborting prediction.")
                return None
        
        # Step 1: Fetch team data
        team1_stats = self.fetch_team_data(
            team1_name, region1, 
            include_player_stats, include_economy, include_maps
        )
        
        team2_stats = self.fetch_team_data(
            team2_name, region2, 
            include_player_stats, include_economy, include_maps
        )
        
        if not team1_stats or not team2_stats:
            print("Failed to fetch team data. Aborting prediction.")
            return None
        
        # Step 2: Prepare features for the prediction
        X_scaled, features_dict, feature_explanations = prepare_match_features(
            team1_stats, team2_stats, 
            self.required_features, 
            self.scaler,
            self.feature_mask,
            verbose
        )
        
        if X_scaled is None:
            print("Failed to prepare features. Aborting prediction.")
            return None
        
        # Step 3: Make the prediction
        prediction_result = predict_match_with_ensemble(
            X_scaled, 
            self.models, 
            team1_name, 
            team2_name, 
            features_dict, 
            feature_explanations,
            override,
            verbose
        )
        
        if not prediction_result:
            print("Failed to make prediction.")
            return None
        
        # Step 4: Print detailed report
        print_prediction_report(prediction_result)
        
        # Step 5: Visualize if requested
        if visualize:
            visualize_prediction(prediction_result, save_visualization)
        
        return prediction_result

    def predict_match_with_stats(self, team1_stats, team2_stats, verbose=False, visualize=False, save_visualization=None, override=None):
        """
        Predict the outcome of a match between two teams using pre-loaded stats.
        
        Args:
            team1_stats (dict): Statistics for team 1
            team2_stats (dict): Statistics for team 2
            verbose (bool): Whether to print detailed information
            visualize (bool): Whether to visualize the prediction
            save_visualization (str, optional): Path to save the visualization
            override (dict, optional): Dictionary specifying prediction override
            
        Returns:
            dict: Prediction results including probabilities and analysis
        """
        # Check if models are loaded
        if not self.models:
            print("No models loaded. Loading artifacts...")
            self.load_artifacts()
            if not self.models:
                print("Failed to load models. Aborting prediction.")
                return None
        
        # Get team names
        team1_name = team1_stats.get('team_name', 'Team 1')
        team2_name = team2_stats.get('team_name', 'Team 2')
        
        # Prepare features for the prediction
        X_scaled, features_dict, feature_explanations = prepare_match_features(
            team1_stats, team2_stats, 
            self.required_features, 
            self.scaler,
            self.feature_mask,
            verbose
        )
        
        if X_scaled is None:
            print("Failed to prepare features. Aborting prediction.")
            return None
        
        # Make the prediction
        prediction_result = predict_match_with_ensemble(
            X_scaled, 
            self.models, 
            team1_name, 
            team2_name, 
            features_dict, 
            feature_explanations,
            override,
            verbose
        )
        
        if not prediction_result:
            print("Failed to make prediction.")
            return None
        
        # Visualize if requested
        if visualize:
            visualize_prediction(prediction_result, save_visualization)
        
        return prediction_result        

#-------------------------------------------------------------------------
# COMMAND-LINE INTERFACE
#-------------------------------------------------------------------------

def main():
    """
    Main function for command-line operation.
    """
    parser = argparse.ArgumentParser(description="Valorant Match Prediction System")
    
    # Main commands
    parser.add_argument("--team1", type=str, help="First team name")
    parser.add_argument("--team2", type=str, help="Second team name")
    parser.add_argument("--region1", type=str, help="Region code for first team")
    parser.add_argument("--region2", type=str, help="Region code for second team")
    
    # Data options
    parser.add_argument("--no-players", action="store_true", help="Exclude player stats")
    parser.add_argument("--no-economy", action="store_true", help="Exclude economy data")
    parser.add_argument("--no-maps", action="store_true", help="Exclude map statistics")
    
    # Output options
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--visualize", action="store_true", help="Visualize the prediction")
    parser.add_argument("--save-viz", type=str, help="Save visualization to file")
    
    # Override options
    parser.add_argument("--override-winner", type=str, help="Override predicted winner")
    parser.add_argument("--override-reason", type=str, help="Reason for override")
    
    # Path options
    parser.add_argument("--artifacts-path", type=str, default=".", help="Path to artifacts directory")
    
    args = parser.parse_args()
    
    # Check required arguments
    if not args.team1 or not args.team2:
        print("Error: Both --team1 and --team2 are required.")
        parser.print_help()
        return
    
    # Initialize predictor
    predictor = ValorantPredictor(artifacts_path=args.artifacts_path)
    
    # Prepare override if specified
    override = None
    if args.override_winner:
        override = {
            'winner': args.override_winner,
            'reason': args.override_reason or "Manual override"
        }
    
    # Make prediction
    predictor.predict_match(
        team1_name=args.team1,
        team2_name=args.team2,
        region1=args.region1,
        region2=args.region2,
        include_player_stats=not args.no_players,
        include_economy=not args.no_economy,
        include_maps=not args.no_maps,
        verbose=args.verbose,
        visualize=args.visualize,
        save_visualization=args.save_viz,
        override=override
    )


if __name__ == "__main__":
    main()