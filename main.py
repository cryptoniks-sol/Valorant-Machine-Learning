import requests
import json
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV, train_test_split, learning_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
import pickle
import time
import re
from datetime import datetime
import argparse
import matplotlib.pyplot as plt

# API URL
API_URL = "http://localhost:5000/api/v1"

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
    """Fetch detailed player statistics for a team."""
    team_details = fetch_team_details(team_id)
    
    if not team_details or 'players' not in team_details:
        return None
    
    player_stats = []
    
    for player in team_details['players']:
        player_id = player.get('id')
        if not player_id:
            continue
            
        print(f"Fetching stats for player: {player.get('name', 'Unknown')} (ID: {player_id})")
        
        # Fetch detailed player stats
        response = requests.get(f"{API_URL}/players/{player_id}")
        
        if response.status_code != 200:
            print(f"Error fetching player {player_id}: {response.status_code}")
            continue
            
        player_data = response.json()
        player_stats.append(player_data)
        
        # Be nice to the API
        time.sleep(0.5)
    
    return player_stats

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
                
                matches.append(match_info)
            
        except Exception as e:
            print(f"Error parsing match: {e}")
            continue
    
    # Summarize wins/losses
    wins = sum(1 for match in matches if match['team_won'])
    print(f"Processed {len(matches)} matches for {team_name}: {wins} wins, {len(matches) - wins} losses")
    
    return matches

def calculate_team_stats(matches):
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
        'map_stats': map_stats
    }
    
    return team_stats

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
        
        # Try to extract round information
        try:
            # Add map-specific round counting logic here based on the API response structure
            # This is a placeholder - you'll need to adapt to actual data structure
            if 'rounds' in match:
                map_performance[map_name]['rounds_played'] += len(match['rounds'])
                map_performance[map_name]['rounds_won'] += sum(1 for round in match['rounds'] if round.get('winner') == match['team_name'])
                
                # Count attack/defense rounds
                # This logic depends on your specific data structure
                for round in match['rounds']:
                    if round.get('side') == 'attack':
                        map_performance[map_name]['attack_rounds'] += 1
                        if round.get('winner') == match['team_name']:
                            map_performance[map_name]['attack_rounds_won'] += 1
                    elif round.get('side') == 'defense':
                        map_performance[map_name]['defense_rounds'] += 1
                        if round.get('winner') == match['team_name']:
                            map_performance[map_name]['defense_rounds_won'] += 1
                    elif round.get('side') == 'overtime':
                        map_performance[map_name]['overtime_rounds'] += 1
                        if round.get('winner') == match['team_name']:
                            map_performance[map_name]['overtime_rounds_won'] += 1
        except Exception as e:
            print(f"Error extracting round data: {e}")
    
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

def analyze_opponent_quality(team_matches, all_teams_data):
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
        for team in all_teams_data:
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
    
    for match in team_matches:
        opponent_name = match.get('opponent_name')
        opponent_ranking = None
        opponent_rating = None
        
        # Find opponent in all_teams_data
        for team in all_teams_data:
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
    
    return opponent_quality

def analyze_event_performance(team_matches):
    """Analyze performance differences between online and LAN events."""
    event_performance = {
        'lan': {
            'matches': 0,
            'wins': 0,
            'win_rate': 0,
            'avg_score': 0,
            'avg_opponent_score': 0,
            'score_differential': 0
        },
        'online': {
            'matches': 0,
            'wins': 0,
            'win_rate': 0,
            'avg_score': 0,
            'avg_opponent_score': 0,
            'score_differential': 0
        }
    }
    
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
    
    # Calculate averages
    for event_type in ['lan', 'online']:
        stats = event_performance[event_type]
        if stats['matches'] > 0:
            stats['win_rate'] = stats['wins'] / stats['matches']
            stats['avg_score'] = stats['avg_score'] / stats['matches']
            stats['avg_opponent_score'] = stats['avg_opponent_score'] / stats['matches']
            stats['score_differential'] = stats['avg_score'] - stats['avg_opponent_score']
    
    # Calculate LAN vs. online differential
    if event_performance['lan']['matches'] > 0 and event_performance['online']['matches'] > 0:
        event_performance['lan_vs_online_win_rate_diff'] = event_performance['lan']['win_rate'] - event_performance['online']['win_rate']
    else:
        event_performance['lan_vs_online_win_rate_diff'] = 0
    
    return event_performance

def aggregate_player_metrics(player_stats):
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

def calculate_head_to_head(team1_matches, team2_name, team2_matches, team1_name):
    """Calculate head-to-head statistics between two teams without double-counting."""
    # Validate input
    if not team1_name or not team2_name:
        print("Error: Empty team name provided to head-to-head calculation")
        return {
            'team1_h2h_matches': 0,
            'team1_h2h_wins': 0,
            'team1_h2h_win_rate': 0,
            'team2_h2h_matches': 0,
            'team2_h2h_wins': 0,
            'team2_h2h_win_rate': 0,
            'total_h2h_matches': 0
        }
    
    # Add debug output
    print(f"Calculating H2H for {team1_name} vs {team2_name}")
    
    # Find matches between team1 and team2 - EXACT MATCH ONLY
    team1_vs_team2 = [
        match for match in team1_matches 
        if match['opponent_name'].lower() == team2_name.lower()
    ]
    
    team2_vs_team1 = [
        match for match in team2_matches 
        if match['opponent_name'].lower() == team1_name.lower()
    ]
    
    # Add debug output for matches found
    if len(team1_vs_team2) > 0 or len(team2_vs_team1) > 0:
        print(f"H2H Matches found between {team1_name} and {team2_name}:")
        for match in team1_vs_team2:
            print(f"  {team1_name} vs {match['opponent_name']}: ID {match['match_id']}")
        for match in team2_vs_team1:
            print(f"  {team2_name} vs {match['opponent_name']}: ID {match['match_id']}")
    else:
        print(f"No H2H matches found between {team1_name} and {team2_name}")
    
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
    
    # Combine all h2h stats
    h2h_stats = {
        'team1_h2h_matches': len(team1_vs_team2),
        'team1_h2h_wins': team1_h2h_wins,
        'team1_h2h_win_rate': team1_h2h_win_rate,
        'team2_h2h_matches': len(team2_vs_team1),
        'team2_h2h_wins': team2_h2h_wins,
        'team2_h2h_win_rate': team2_h2h_win_rate,
        'total_h2h_matches': total_h2h_matches
    }
    
    return h2h_stats

def create_prediction_features(team1_stats, team2_stats, h2h_stats, team1_details, team2_details):
    """Create a feature vector for prediction."""
    features = {
        # Basic team1 stats
        'team1_matches': team1_stats.get('matches', 0),
        'team1_wins': team1_stats.get('wins', 0),
        'team1_win_rate': team1_stats.get('win_rate', 0.5),
        'team1_avg_score': team1_stats.get('avg_score', 0),
        'team1_score_diff': team1_stats.get('score_differential', 0),
        'team1_recent_form': team1_stats.get('recent_form', 0.5),
        
        # Basic team2 stats
        'team2_matches': team2_stats.get('matches', 0),
        'team2_wins': team2_stats.get('wins', 0),
        'team2_win_rate': team2_stats.get('win_rate', 0.5),
        'team2_avg_score': team2_stats.get('avg_score', 0),
        'team2_score_diff': team2_stats.get('score_differential', 0),
        'team2_recent_form': team2_stats.get('recent_form', 0.5),
        
        # Team ranking and rating
        'team1_ranking': team1_details.get('ranking', 9999) if team1_details else 9999,
        'team2_ranking': team2_details.get('ranking', 9999) if team2_details else 9999,
        'team1_rating': team1_details.get('rating', 1500) if team1_details else 1500,
        'team2_rating': team2_details.get('rating', 1500) if team2_details else 1500,
        
        # Head-to-head stats
        'h2h_matches': h2h_stats.get('total_h2h_matches', 0),
        'team1_h2h_wins': h2h_stats.get('team1_h2h_wins', 0),
        'team1_h2h_win_rate': h2h_stats.get('team1_h2h_win_rate', 0.5),
        
        # Relative strength indicators
        'ranking_diff': (team1_details.get('ranking', 9999) if team1_details else 9999) - 
                        (team2_details.get('ranking', 9999) if team2_details else 9999),
        'rating_diff': (team1_details.get('rating', 1500) if team1_details else 1500) - 
                       (team2_details.get('rating', 1500) if team2_details else 1500),
        'win_rate_diff': team1_stats.get('win_rate', 0.5) - team2_stats.get('win_rate', 0.5),
        'avg_score_diff': team1_stats.get('avg_score', 0) - team2_stats.get('avg_score', 0),
        'recent_form_diff': team1_stats.get('recent_form', 0.5) - team2_stats.get('recent_form', 0.5)
    }
    
    return features

def create_comprehensive_features(team1_id, team2_id):
    """Create a comprehensive feature set for match prediction."""
    # Fetch all necessary data
    team1_details = fetch_team_details(team1_id)
    team2_details = fetch_team_details(team2_id)
    
    team1_history = fetch_team_match_history(team1_id)
    team2_history = fetch_team_match_history(team2_id)
    
    team1_matches = parse_match_data(team1_history, team1_details.get('name', ''))
    team2_matches = parse_match_data(team2_history, team2_details.get('name', ''))
    
    # Fetch player statistics
    team1_player_stats = fetch_team_player_stats(team1_id)
    team2_player_stats = fetch_team_player_stats(team2_id)
    
    # Get all teams for ranking comparison
    all_teams_response = requests.get(f"{API_URL}/teams?limit=300")
    all_teams = all_teams_response.json().get('data', []) if all_teams_response.status_code == 200 else []
    
    # Calculate all advanced statistics
    team1_basic_stats = calculate_team_stats(team1_matches)
    team2_basic_stats = calculate_team_stats(team2_matches)
    
    team1_map_performance = extract_map_performance(team1_matches)
    team2_map_performance = extract_map_performance(team2_matches)
    
    team1_tournament_performance = extract_tournament_performance(team1_matches)
    team2_tournament_performance = extract_tournament_performance(team2_matches)
    
    team1_trends = analyze_performance_trends(team1_matches)
    team2_trends = analyze_performance_trends(team2_matches)
    
    team1_opponent_quality = analyze_opponent_quality(team1_matches, all_teams)
    team2_opponent_quality = analyze_opponent_quality(team2_matches, all_teams)
    
    team1_event_performance = analyze_event_performance(team1_matches)
    team2_event_performance = analyze_event_performance(team2_matches)
    
    team1_player_metrics = aggregate_player_metrics(team1_player_stats)
    team2_player_metrics = aggregate_player_metrics(team2_player_stats)
    
    h2h_stats = calculate_head_to_head(team1_matches, team2_details.get('name', ''), 
                                      team2_matches, team1_details.get('name', ''))
    
    # Create comprehensive feature dictionary
    features = {
        # Basic team stats
        'team1_matches': team1_basic_stats.get('matches', 0),
        'team1_wins': team1_basic_stats.get('wins', 0),
        'team1_win_rate': team1_basic_stats.get('win_rate', 0.5),
        'team1_avg_score': team1_basic_stats.get('avg_score', 0),
        'team1_score_diff': team1_basic_stats.get('score_differential', 0),
        
        'team2_matches': team2_basic_stats.get('matches', 0),
        'team2_wins': team2_basic_stats.get('wins', 0),
        'team2_win_rate': team2_basic_stats.get('win_rate', 0.5),
        'team2_avg_score': team2_basic_stats.get('avg_score', 0),
        'team2_score_diff': team2_basic_stats.get('score_differential', 0),
        
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
        
        'team2_recent_form_5': team2_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5),
        'team2_recent_form_10': team2_trends.get('recent_matches', {}).get('last_10_win_rate', 0.5),
        'team2_form_trajectory': team2_trends.get('form_trajectory', {}).get('5_vs_10', 0),
        'team2_recency_weighted_win_rate': team2_trends.get('recency_weighted_win_rate', 0.5),
        
        # Event type performance
        'team1_lan_win_rate': team1_event_performance.get('lan', {}).get('win_rate', 0.5),
        'team1_online_win_rate': team1_event_performance.get('online', {}).get('win_rate', 0.5),
        'team1_lan_vs_online_diff': team1_event_performance.get('lan_vs_online_win_rate_diff', 0),
        
        'team2_lan_win_rate': team2_event_performance.get('lan', {}).get('win_rate', 0.5),
        'team2_online_win_rate': team2_event_performance.get('online', {}).get('win_rate', 0.5),
        'team2_lan_vs_online_diff': team2_event_performance.get('lan_vs_online_win_rate_diff', 0),
        
        # Opponent quality metrics
        'team1_avg_opponent_ranking': team1_opponent_quality.get('avg_opponent_ranking', 50),
        'team1_top_10_win_rate': team1_opponent_quality.get('top_10_win_rate', 0),
        'team1_upset_factor': team1_opponent_quality.get('upset_factor', 0),
        
        'team2_avg_opponent_ranking': team2_opponent_quality.get('avg_opponent_ranking', 50),
        'team2_top_10_win_rate': team2_opponent_quality.get('top_10_win_rate', 0),
        'team2_upset_factor': team2_opponent_quality.get('upset_factor', 0),
        
        # Player metrics
        'team1_avg_player_rating': team1_player_metrics.get('avg_team_rating', 1),
        'team1_star_player_rating': team1_player_metrics.get('star_player_rating', 1),
        'team1_role_balance': team1_player_metrics.get('role_balance', 1),
        
        'team2_avg_player_rating': team2_player_metrics.get('avg_team_rating', 1),
        'team2_star_player_rating': team2_player_metrics.get('star_player_rating', 1),
        'team2_role_balance': team2_player_metrics.get('role_balance', 1),
        
        # Head-to-head stats
        'h2h_matches': h2h_stats.get('total_h2h_matches', 0),
        'team1_h2h_wins': h2h_stats.get('team1_h2h_wins', 0),
        'team1_h2h_win_rate': h2h_stats.get('team1_h2h_win_rate', 0.5),
        
        # Relative strength indicators
        'ranking_diff': team1_details.get('ranking', 9999) - team2_details.get('ranking', 9999),
        'rating_diff': team1_details.get('rating', 1500) - team2_details.get('rating', 1500),
        'win_rate_diff': team1_basic_stats.get('win_rate', 0.5) - team2_basic_stats.get('win_rate', 0.5),
        'avg_score_diff': team1_basic_stats.get('avg_score', 0) - team2_basic_stats.get('avg_score', 0),
        'recent_form_diff': team1_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5) - 
                            team2_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5),
        'player_rating_diff': team1_player_metrics.get('avg_team_rating', 1) - 
                             team2_player_metrics.get('avg_team_rating', 1),
        'star_player_diff': team1_player_metrics.get('star_player_rating', 1) - 
                           team2_player_metrics.get('star_player_rating', 1)
    }
    
    return features

def create_training_data(team1_matches, team2_matches, team1_stats, team2_stats, team1_details, team2_details):
    """Create training data from match histories."""
    # Combine all opponents faced by both teams
    all_opponents = set()
    for match in team1_matches:
        all_opponents.add(match['opponent_name'])
    for match in team2_matches:
        all_opponents.add(match['opponent_name'])
    
    # Create a dataset for training
    training_data = []
    
    # Process team1's matches
    for match in team1_matches:
        opponent_name = match['opponent_name']
        opponent_stats = team1_stats['opponent_stats'].get(opponent_name, {})
        
        # Skip if we don't have enough data on this opponent
        if not opponent_stats or opponent_stats.get('matches', 0) < 1:
            continue
        
        # Create a feature dictionary
        feature_row = {
            # Team1 stats
            'team1_matches': team1_stats.get('matches', 0),
            'team1_wins': team1_stats.get('wins', 0),
            'team1_win_rate': team1_stats.get('win_rate', 0.5),
            'team1_avg_score': team1_stats.get('avg_score', 0),
            'team1_score_diff': team1_stats.get('score_differential', 0),
            'team1_recent_form': team1_stats.get('recent_form', 0.5),
            
            # Opponent stats
            'team2_matches': opponent_stats.get('matches', 0),
            'team2_wins': opponent_stats.get('wins', 0),
            'team2_win_rate': opponent_stats.get('win_rate', 0.5),
            'team2_avg_score': opponent_stats.get('avg_score', 0),
            'team2_score_diff': opponent_stats.get('score_differential', 0),
            'team2_recent_form': 0.5,  # We don't have this for the opponent
            
            # Team ranking and rating
            'team1_ranking': team1_details.get('ranking', 9999) if team1_details else 9999,
            'team2_ranking': 9999,  # We don't have this for all opponents
            'team1_rating': team1_details.get('rating', 1500) if team1_details else 1500,
            'team2_rating': 1500,  # We don't have this for all opponents
            
            # Head-to-head stats (specific to this match)
            'h2h_matches': opponent_stats.get('matches', 0),
            'team1_h2h_wins': opponent_stats.get('wins', 0),
            'team1_h2h_win_rate': opponent_stats.get('win_rate', 0.5),
            
            # Relative strength indicators
            'ranking_diff': 0,  # We don't have ranking for all opponents
            'rating_diff': 0,  # We don't have rating for all opponents
            'win_rate_diff': team1_stats.get('win_rate', 0.5) - opponent_stats.get('win_rate', 0.5),
            'avg_score_diff': team1_stats.get('avg_score', 0) - opponent_stats.get('avg_score', 0),
            'recent_form_diff': team1_stats.get('recent_form', 0.5) - 0.5,  # Assuming neutral recent form for opponent
            
            # Target variable
            'team1_won': 1 if match['team_won'] else 0
        }
        
        training_data.append(feature_row)
    
    # Process team2's matches
    for match in team2_matches:
        opponent_name = match['opponent_name']
        opponent_stats = team2_stats['opponent_stats'].get(opponent_name, {})
        
        # Skip if we don't have enough data on this opponent
        if not opponent_stats or opponent_stats.get('matches', 0) < 1:
            continue
        
        # Create a feature dictionary (note: team2 is now 'team1' in the features)
        feature_row = {
            # Team2 stats (as team1 in the features)
            'team1_matches': team2_stats.get('matches', 0),
            'team1_wins': team2_stats.get('wins', 0),
            'team1_win_rate': team2_stats.get('win_rate', 0.5),
            'team1_avg_score': team2_stats.get('avg_score', 0),
            'team1_score_diff': team2_stats.get('score_differential', 0),
            'team1_recent_form': team2_stats.get('recent_form', 0.5),
            
            # Opponent stats (as team2 in the features)
            'team2_matches': opponent_stats.get('matches', 0),
            'team2_wins': opponent_stats.get('wins', 0),
            'team2_win_rate': opponent_stats.get('win_rate', 0.5),
            'team2_avg_score': opponent_stats.get('avg_score', 0),
            'team2_score_diff': opponent_stats.get('score_differential', 0),
            'team2_recent_form': 0.5,  # We don't have this for the opponent
            
            # Team ranking and rating
            'team1_ranking': team2_details.get('ranking', 9999) if team2_details else 9999,
            'team2_ranking': 9999,  # We don't have this for all opponents
            'team1_rating': team2_details.get('rating', 1500) if team2_details else 1500,
            'team2_rating': 1500,  # We don't have this for all opponents
            
            # Head-to-head stats (specific to this match)
            'h2h_matches': opponent_stats.get('matches', 0),
            'team1_h2h_wins': opponent_stats.get('wins', 0),
            'team1_h2h_win_rate': opponent_stats.get('win_rate', 0.5),
            
            # Relative strength indicators
            'ranking_diff': 0,  # We don't have ranking for all opponents
            'rating_diff': 0,  # We don't have rating for all opponents
            'win_rate_diff': team2_stats.get('win_rate', 0.5) - opponent_stats.get('win_rate', 0.5),
            'avg_score_diff': team2_stats.get('avg_score', 0) - opponent_stats.get('avg_score', 0),
            'recent_form_diff': team2_stats.get('recent_form', 0.5) - 0.5,  # Assuming neutral recent form for opponent
            
            # Target variable
            'team1_won': 1 if match['team_won'] else 0
        }
        
        training_data.append(feature_row)
    
    # Convert to DataFrame
    training_df = pd.DataFrame(training_data)
    
    # Handle missing values
    training_df = training_df.fillna(0)
    
    return training_df

def create_enhanced_training_data(team1_matches, team2_matches, team1_id, team2_id, all_teams):
    """Create enhanced training data from match histories with advanced features."""
    # Get team details
    team1_details = fetch_team_details(team1_id)
    team2_details = fetch_team_details(team2_id)
    
    # Where the function is called, ensure team names are properly defined
    team1_name = team1_details.get('name', '') if team1_details else ''
    team2_name = team2_details.get('name', '') if team2_details else ''
    
    # Create a dataset for training
    training_data = []
    
    # Get all team matches for better features
    all_team_matches = {}
    
    # Process team1's matches
    for match in team1_matches:
        opponent_name = match['opponent_name']
        opponent_id = None
        
        # Try to find opponent ID
        for team in all_teams:
            if team.get('name') == opponent_name:
                opponent_id = team.get('id')
                break
        
        if not opponent_id:
            continue
            
        # Get opponent's match history if we don't have it already
        if opponent_id not in all_team_matches:
            opponent_history = fetch_team_match_history(opponent_id)
            if opponent_history:
                opponent_matches = parse_match_data(opponent_history, opponent_name)
                all_team_matches[opponent_id] = opponent_matches
        
        # Skip if we couldn't get opponent match history
        if opponent_id not in all_team_matches:
            continue
            
        opponent_matches = all_team_matches[opponent_id]
        
        # Calculate all advanced statistics for both teams
        team1_basic_stats = calculate_team_stats(team1_matches)
        opponent_basic_stats = calculate_team_stats(opponent_matches)
        
        team1_trends = analyze_performance_trends(team1_matches)
        opponent_trends = analyze_performance_trends(opponent_matches)
        
        team1_opponent_quality = analyze_opponent_quality(team1_matches, all_teams)
        opponent_opponent_quality = analyze_opponent_quality(opponent_matches, all_teams)
        
        team1_event_performance = analyze_event_performance(team1_matches)
        opponent_event_performance = analyze_event_performance(opponent_matches)
        
            # Add validation to ensure non-empty team names
    if not team1_name or not team2_name:
        print(f"Warning: Invalid team names for H2H calculation: '{team1_name}' vs '{team2_name}'")
        # Use a default empty h2h_stats
        h2h_stats = {
            'team1_h2h_matches': 0,
            'team1_h2h_wins': 0,
            'team1_h2h_win_rate': 0,
            'team2_h2h_matches': 0,
            'team2_h2h_wins': 0,
            'team2_h2h_win_rate': 0,
            'total_h2h_matches': 0
        }
    else:
        h2h_stats = calculate_head_to_head(team1_matches, team2_name, team2_matches, team1_name)
        
        # Create feature dict for this match
        feature_row = {
            # Basic team stats
            'team1_matches': team1_basic_stats.get('matches', 0),
            'team1_wins': team1_basic_stats.get('wins', 0),
            'team1_win_rate': team1_basic_stats.get('win_rate', 0.5),
            'team1_avg_score': team1_basic_stats.get('avg_score', 0),
            'team1_score_diff': team1_basic_stats.get('score_differential', 0),
            
            'team2_matches': opponent_basic_stats.get('matches', 0),
            'team2_wins': opponent_basic_stats.get('wins', 0),
            'team2_win_rate': opponent_basic_stats.get('win_rate', 0.5),
            'team2_avg_score': opponent_basic_stats.get('avg_score', 0),
            'team2_score_diff': opponent_basic_stats.get('score_differential', 0),
            
            # Team ranking and rating
            'team1_ranking': team1_details.get('ranking', 9999),
            'team2_ranking': 9999,  # Default if we don't have opponent ranking
            'team1_rating': team1_details.get('rating', 1500),
            'team2_rating': 1500,  # Default if we don't have opponent rating
            
            # Recent form and trend features
            'team1_recent_form_5': team1_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5),
            'team1_recent_form_10': team1_trends.get('recent_matches', {}).get('last_10_win_rate', 0.5),
            'team1_form_trajectory': team1_trends.get('form_trajectory', {}).get('5_vs_10', 0),
            'team1_recency_weighted_win_rate': team1_trends.get('recency_weighted_win_rate', 0.5),
            
            'team2_recent_form_5': opponent_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5),
            'team2_recent_form_10': opponent_trends.get('recent_matches', {}).get('last_10_win_rate', 0.5),
            'team2_form_trajectory': opponent_trends.get('form_trajectory', {}).get('5_vs_10', 0),
            'team2_recency_weighted_win_rate': opponent_trends.get('recency_weighted_win_rate', 0.5),
            
            # Event type performance
            'team1_lan_win_rate': team1_event_performance.get('lan', {}).get('win_rate', 0.5),
            'team1_online_win_rate': team1_event_performance.get('online', {}).get('win_rate', 0.5),
            'team1_lan_vs_online_diff': team1_event_performance.get('lan_vs_online_win_rate_diff', 0),
            
            'team2_lan_win_rate': opponent_event_performance.get('lan', {}).get('win_rate', 0.5),
            'team2_online_win_rate': opponent_event_performance.get('online', {}).get('win_rate', 0.5),
            'team2_lan_vs_online_diff': opponent_event_performance.get('lan_vs_online_win_rate_diff', 0),
            
            # Opponent quality metrics
            'team1_avg_opponent_ranking': team1_opponent_quality.get('avg_opponent_ranking', 50),
            'team1_top_10_win_rate': team1_opponent_quality.get('top_10_win_rate', 0),
            'team1_upset_factor': team1_opponent_quality.get('upset_factor', 0),
            
            'team2_avg_opponent_ranking': opponent_opponent_quality.get('avg_opponent_ranking', 50),
            'team2_top_10_win_rate': opponent_opponent_quality.get('top_10_win_rate', 0),
            'team2_upset_factor': opponent_opponent_quality.get('upset_factor', 0),
            
            # Head-to-head stats
            'h2h_matches': h2h_stats.get('total_h2h_matches', 0),
            'team1_h2h_wins': h2h_stats.get('team1_h2h_wins', 0),
            'team1_h2h_win_rate': h2h_stats.get('team1_h2h_win_rate', 0.5),
            
            # Relative strength indicators
            'ranking_diff': 0,  # Default if we don't have rankings
            'rating_diff': 0,  # Default if we don't have ratings
            'win_rate_diff': team1_basic_stats.get('win_rate', 0.5) - opponent_basic_stats.get('win_rate', 0.5),
            'avg_score_diff': team1_basic_stats.get('avg_score', 0) - opponent_basic_stats.get('avg_score', 0),
            'recent_form_diff': team1_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5) - 
                                opponent_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5),
            
            # Target variable
            'team1_won': 1 if match['team_won'] else 0
        }
        
# Look up opponent's ranking and rating if available
        for team in all_teams:
            if team.get('name') == opponent_name:
                feature_row['team2_ranking'] = team.get('ranking', 9999)
                feature_row['team2_rating'] = team.get('rating', 1500)
                # Calculate ranking and rating differentials
                feature_row['ranking_diff'] = team1_details.get('ranking', 9999) - team.get('ranking', 9999)
                feature_row['rating_diff'] = team1_details.get('rating', 1500) - team.get('rating', 1500)
                break
        
        training_data.append(feature_row)
    
    # Process team2's matches in the same way
    for match in team2_matches:
        opponent_name = match['opponent_name']
        opponent_id = None
        
        # Find opponent ID
        for team in all_teams:
            if team.get('name') == opponent_name:
                opponent_id = team.get('id')
                break
        
        if not opponent_id:
            continue
        
        # Get opponent's match history if we don't have it already
        if opponent_id not in all_team_matches:
            opponent_history = fetch_team_match_history(opponent_id)
            if opponent_history:
                opponent_matches = parse_match_data(opponent_history, opponent_name)
                all_team_matches[opponent_id] = opponent_matches
        
        # Skip if we couldn't get opponent match history
        if opponent_id not in all_team_matches:
            continue
        
        opponent_matches = all_team_matches[opponent_id]
        
        # Calculate all advanced statistics for both teams
        team2_basic_stats = calculate_team_stats(team2_matches)
        opponent_basic_stats = calculate_team_stats(opponent_matches)
        
        team2_trends = analyze_performance_trends(team2_matches)
        opponent_trends = analyze_performance_trends(opponent_matches)
        
        team2_opponent_quality = analyze_opponent_quality(team2_matches, all_teams)
        opponent_opponent_quality = analyze_opponent_quality(opponent_matches, all_teams)
        
        team2_event_performance = analyze_event_performance(team2_matches)
        opponent_event_performance = analyze_event_performance(opponent_matches)
        
        # Get h2h stats
        h2h_stats = calculate_head_to_head(team2_matches, opponent_name, opponent_matches, team2_name)
        
        # Create feature dict for this match (note: team2 is now 'team1' in the features)
        feature_row = {
            # Basic team stats
            'team1_matches': team2_basic_stats.get('matches', 0),
            'team1_wins': team2_basic_stats.get('wins', 0),
            'team1_win_rate': team2_basic_stats.get('win_rate', 0.5),
            'team1_avg_score': team2_basic_stats.get('avg_score', 0),
            'team1_score_diff': team2_basic_stats.get('score_differential', 0),
            
            'team2_matches': opponent_basic_stats.get('matches', 0),
            'team2_wins': opponent_basic_stats.get('wins', 0),
            'team2_win_rate': opponent_basic_stats.get('win_rate', 0.5),
            'team2_avg_score': opponent_basic_stats.get('avg_score', 0),
            'team2_score_diff': opponent_basic_stats.get('score_differential', 0),
            
            # Team ranking and rating
            'team1_ranking': team2_details.get('ranking', 9999),
            'team2_ranking': 9999,  # Default if we don't have opponent ranking
            'team1_rating': team2_details.get('rating', 1500),
            'team2_rating': 1500,  # Default if we don't have opponent rating
            
            # Recent form and trend features
            'team1_recent_form_5': team2_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5),
            'team1_recent_form_10': team2_trends.get('recent_matches', {}).get('last_10_win_rate', 0.5),
            'team1_form_trajectory': team2_trends.get('form_trajectory', {}).get('5_vs_10', 0),
            'team1_recency_weighted_win_rate': team2_trends.get('recency_weighted_win_rate', 0.5),
            
            'team2_recent_form_5': opponent_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5),
            'team2_recent_form_10': opponent_trends.get('recent_matches', {}).get('last_10_win_rate', 0.5),
            'team2_form_trajectory': opponent_trends.get('form_trajectory', {}).get('5_vs_10', 0),
            'team2_recency_weighted_win_rate': opponent_trends.get('recency_weighted_win_rate', 0.5),
            
            # Event type performance
            'team1_lan_win_rate': team2_event_performance.get('lan', {}).get('win_rate', 0.5),
            'team1_online_win_rate': team2_event_performance.get('online', {}).get('win_rate', 0.5),
            'team1_lan_vs_online_diff': team2_event_performance.get('lan_vs_online_win_rate_diff', 0),
            
            'team2_lan_win_rate': opponent_event_performance.get('lan', {}).get('win_rate', 0.5),
            'team2_online_win_rate': opponent_event_performance.get('online', {}).get('win_rate', 0.5),
            'team2_lan_vs_online_diff': opponent_event_performance.get('lan_vs_online_win_rate_diff', 0),
            
            # Opponent quality metrics
            'team1_avg_opponent_ranking': team2_opponent_quality.get('avg_opponent_ranking', 50),
            'team1_top_10_win_rate': team2_opponent_quality.get('top_10_win_rate', 0),
            'team1_upset_factor': team2_opponent_quality.get('upset_factor', 0),
            
            'team2_avg_opponent_ranking': opponent_opponent_quality.get('avg_opponent_ranking', 50),
            'team2_top_10_win_rate': opponent_opponent_quality.get('top_10_win_rate', 0),
            'team2_upset_factor': opponent_opponent_quality.get('upset_factor', 0),
            
            # Head-to-head stats
            'h2h_matches': h2h_stats.get('total_h2h_matches', 0),
            'team1_h2h_wins': h2h_stats.get('team1_h2h_wins', 0),
            'team1_h2h_win_rate': h2h_stats.get('team1_h2h_win_rate', 0.5),
            
            # Relative strength indicators
            'ranking_diff': 0,  # Default if we don't have rankings
            'rating_diff': 0,  # Default if we don't have ratings
            'win_rate_diff': team2_basic_stats.get('win_rate', 0.5) - opponent_basic_stats.get('win_rate', 0.5),
            'avg_score_diff': team2_basic_stats.get('avg_score', 0) - opponent_basic_stats.get('avg_score', 0),
            'recent_form_diff': team2_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5) - 
                                opponent_trends.get('recent_matches', {}).get('last_5_win_rate', 0.5),
            
            # Target variable
            'team1_won': 1 if match['team_won'] else 0
        }
        
        # Look up opponent's ranking and rating if available
        for team in all_teams:
            if team.get('name') == opponent_name:
                feature_row['team2_ranking'] = team.get('ranking', 9999)
                feature_row['team2_rating'] = team.get('rating', 1500)
                # Calculate ranking and rating differentials
                feature_row['ranking_diff'] = team2_details.get('ranking', 9999) - team.get('ranking', 9999)
                feature_row['rating_diff'] = team2_details.get('rating', 1500) - team.get('rating', 1500)
                break
        
        training_data.append(feature_row)
    
    # Convert to DataFrame
    training_df = pd.DataFrame(training_data)
    
    # Handle missing values
    training_df = training_df.fillna(0)
    
    return training_df

def train_predictive_model(training_data):
    """Train a Random Forest model on the specific team data."""
    if len(training_data) < 10:
        print(f"Warning: Only {len(training_data)} matches available for training. Model may be unreliable.")
    
    # Separate features and target
    X = training_data.drop('team1_won', axis=1)
    y = training_data['team1_won']
    
    # Check for balanced classes
    class_counts = np.bincount(y)
    if len(class_counts) == 1:
        print(f"Warning: All examples belong to class {class_counts[0]}. Prediction will always be this class.")
        only_class = 0 if class_counts[0] == 0 else 1
        print(f"Class distribution - Only Class {only_class}: {class_counts[0]}")
    else:
        print(f"Class distribution - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model with more trees and different parameters
    if len(class_counts) == 1:
        # If only one class is present, use a dummy classifier that always predicts that class
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy='constant', constant=int(y.iloc[0]))
        model.fit(X_scaled, y)
        
        # Create dummy feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.zeros(len(X.columns))
        })
    else:
        # Train a proper Random Forest model when we have examples from both classes
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
    
    print("Top 10 most important features:")
    print(feature_importance.head(10))
    
    return model, scaler, X.columns.tolist()

def train_advanced_predictive_model(training_data):
    """Train an advanced Random Forest model with cross-validation, hyperparameter tuning and feature selection."""
    if len(training_data) < 10:
        print(f"Warning: Only {len(training_data)} matches available for training. Model may be unreliable.")
    
    # Separate features and target
    X = training_data.drop('team1_won', axis=1)
    y = training_data['team1_won']
    
    # Check for balanced classes
    class_counts = np.bincount(y)
    if len(class_counts) == 1:
        print(f"Warning: All examples belong to class {class_counts[0]}. Prediction will always be this class.")
        only_class = 0 if class_counts[0] == 0 else 1
        print(f"Class distribution - Only Class {only_class}: {class_counts[0]}")
    else:
        print(f"Class distribution - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # If classes are very imbalanced, use SMOTE
    if len(class_counts) > 1 and min(class_counts) / max(class_counts) < 0.3:
        print("Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=42)
        X_scaled, y = smote.fit_resample(X_scaled, y)
        print(f"After SMOTE - Class 0: {sum(y==0)}, Class 1: {sum(y==1)}")
    
    # Feature selection
    print("Performing feature selection...")
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), threshold='median')
    X_selected = selector.fit_transform(X_scaled, y)
    selected_indices = selector.get_support()
    selected_features = [X.columns[i] for i, selected in enumerate(selected_indices) if selected]
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning
    print("Tuning hyperparameters...")
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
    print(f"Best parameters: {best_params}")
    
    # Train final model with best parameters
    print("Training final model with best parameters...")
    model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    if len(set(y_test)) > 1:  # Only calculate AUC if there are both classes in test set
        print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Get feature importance for selected features
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Create a more advanced ensemble model
    print("\nTraining ensemble model with multiple algorithms...")
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
    
    print("\nEnsemble Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, ens_y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, ens_y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, ens_y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, ens_y_pred):.4f}")
    if len(set(y_test)) > 1:  # Only calculate AUC if there are both classes in test set
        print(f"ROC AUC: {roc_auc_score(y_test, ens_y_prob):.4f}")
    
    # Generate learning curves
    print("\nGenerating learning curves...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_scaled, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.title("Learning Curves")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()
    
    # Save learning curves
    os.makedirs("models", exist_ok=True)
    plt.savefig(f"models/learning_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # Calibrate probabilities for better prediction
    print("\nCalibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(ensemble, cv=5)
    
    # Train the calibrated model on all data
    calibrated_model.fit(X_selected, y)
    
    # Cross-validation to evaluate model
    cv_scores = cross_val_score(ensemble, X_selected, y, cv=cv, scoring='f1')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f}")
    
    # Save all important model components
    model_info = {
        'feature_names': selected_features,
        'feature_importance': feature_importance.to_dict(),
        'model_params': best_params,
        'scaling_mean': scaler.mean_.tolist(),
        'scaling_scale': scaler.scale_.tolist()
    }
    
    # Save models
    with open(f"models/ensemble_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl", 'wb') as f:
        pickle.dump(calibrated_model, f)
    
    with open(f"models/selector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl", 'wb') as f:
        pickle.dump(selector, f)
    
    with open(f"models/scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(f"models/model_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(model_info, f, indent=4)
    
    return calibrated_model, scaler, selector, selected_features

def predict_match_outcome(model, scaler, feature_names_or_selector, features):
    """Predict the outcome of a match between two teams."""
    # Create DataFrame from features
    features_df = pd.DataFrame([features])
    
    # Check if we're using selector (advanced mode) or feature_names (basic mode)
    if hasattr(feature_names_or_selector, 'transform'):
        # We're in advanced mode with a selector
        selector = feature_names_or_selector
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Apply feature selection
        features_selected = selector.transform(features_scaled)
        
        # Make prediction
        prediction = model.predict(features_selected)[0]
        prediction_proba = model.predict_proba(features_selected)[0]
    else:
        # We're in basic mode with feature_names
        feature_names = feature_names_or_selector
        
        # Ensure features match what the model expects
        for feature in feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        features_df = features_df[feature_names]
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
    
    return prediction, prediction_proba

def visualize_results(team1_name, team2_name, prediction, prediction_proba, team1_stats, team2_stats, h2h_stats, feature_importance):
    """Generate visualizations for the prediction results."""
    os.makedirs("predictions", exist_ok=True)
    
    # Probability chart
    plt.figure(figsize=(10, 6))
    plt.bar([team1_name, team2_name], [prediction_proba[1], prediction_proba[0]], color=['blue', 'orange'])
    plt.title('Win Probability')
    plt.ylabel('Probability')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    for i, prob in enumerate([prediction_proba[1], prediction_proba[0]]):
        plt.text(i, prob + 0.02, f'{prob:.2%}', ha='center', va='bottom')
    plt.ylim(0, 1.1)
    plt.savefig(f"predictions/{team1_name}_vs_{team2_name}_probability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # Feature importance chart
    if len(feature_importance) > 0:
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(10)
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.title('Top 10 Most Important Features')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"predictions/{team1_name}_vs_{team2_name}_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
    
    # Team stats comparison
    stats_to_compare = ['win_rate', 'avg_score', 'score_differential', 'recent_form']
    stat_labels = ['Win Rate', 'Avg Score', 'Score Differential', 'Recent Form']
    
    team1_values = [team1_stats.get(stat, 0) for stat in stats_to_compare]
    team2_values = [team2_stats.get(stat, 0) for stat in stats_to_compare]
    
    # Convert win_rate and recent_form to percentages
    team1_values[0] *= 100
    team2_values[0] *= 100
    team1_values[3] *= 100
    team2_values[3] *= 100
    
    x = np.arange(len(stat_labels))
    width = 0.35
    
    plt.figure(figsize=(12, 8))
    plt.bar(x - width/2, team1_values, width, label=team1_name)
    plt.bar(x + width/2, team2_values, width, label=team2_name)
    
    plt.title('Team Stats Comparison')
    plt.xticks(x, stat_labels)
    plt.legend()
    
    # Add value labels on the bars
    for i, v in enumerate(team1_values):
        if stats_to_compare[i] in ['win_rate', 'recent_form']:
            plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
        else:
            plt.text(i - width/2, v + 0.1, f'{v:.1f}', ha='center')
            
    for i, v in enumerate(team2_values):
        if stats_to_compare[i] in ['win_rate', 'recent_form']:
            plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
        else:
            plt.text(i + width/2, v + 0.1, f'{v:.1f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"predictions/{team1_name}_vs_{team2_name}_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # Only generate H2H chart if we have H2H matches
    if h2h_stats.get('total_h2h_matches', 0) > 0:
        # Head-to-head stats
        plt.figure(figsize=(10, 6))
        h2h_values = [h2h_stats.get('team1_h2h_wins', 0), h2h_stats.get('team2_h2h_wins', 0)]
        plt.pie(h2h_values, labels=[team1_name, team2_name], autopct='%1.1f%%',
                colors=['blue', 'orange'], startangle=90)
        plt.axis('equal')
        plt.title(f'Head-to-Head Wins ({h2h_stats.get("total_h2h_matches", 0)} matches)')
        plt.savefig(f"predictions/{team1_name}_vs_{team2_name}_h2h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

def create_betting_optimized_json(team1_name, team2_name, prediction, prediction_proba, team1_stats, team2_stats, 
                                 h2h_stats, feature_importance, team1_details, team2_details, 
                                 team1_trends=None, team2_trends=None, team1_map_performance=None, 
                                 team2_map_performance=None, team1_event_performance=None, team2_event_performance=None):
    """
    Create a comprehensive JSON object with all data needed for betting predictions.
    This JSON will be consumed by the ValorantBettingAdvisor.
    """
    
    # Determine winner based on prediction
    if prediction == 1:
        predicted_winner = team1_name
        win_probability = float(prediction_proba[1])
    else:
        predicted_winner = team2_name
        win_probability = float(prediction_proba[0])
    
    # Create the JSON structure
    betting_json = {
        "match": f"{team1_name} vs {team2_name}",
        "prediction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_winner": predicted_winner,
        "win_probability": win_probability,
        
        # Team 1 detailed stats
        "team1_stats": {
            "name": team1_name,
            "matches": team1_stats.get('matches', 0),
            "wins": team1_stats.get('wins', 0),
            "losses": team1_stats.get('losses', 0),
            "win_rate": float(team1_stats.get('win_rate', 0)),
            "avg_score": float(team1_stats.get('avg_score', 0)),
            "avg_opponent_score": float(team1_stats.get('avg_opponent_score', 0)),
            "score_differential": float(team1_stats.get('score_differential', 0)),
            "recent_form": float(team1_stats.get('recent_form', 0)),
            "ranking": team1_details.get('ranking', 9999) if team1_details else 9999,
            "rating": team1_details.get('rating', 1500) if team1_details else 1500,
        },
        
        # Team 2 detailed stats
        "team2_stats": {
            "name": team2_name,
            "matches": team2_stats.get('matches', 0),
            "wins": team2_stats.get('wins', 0),
            "losses": team2_stats.get('losses', 0),
            "win_rate": float(team2_stats.get('win_rate', 0)),
            "avg_score": float(team2_stats.get('avg_score', 0)),
            "avg_opponent_score": float(team2_stats.get('avg_opponent_score', 0)),
            "score_differential": float(team2_stats.get('score_differential', 0)),
            "recent_form": float(team2_stats.get('recent_form', 0)),
            "ranking": team2_details.get('ranking', 9999) if team2_details else 9999,
            "rating": team2_details.get('rating', 1500) if team2_details else 1500,
        },
        
        # Head-to-head stats
        "h2h_stats": {
            "total_h2h_matches": h2h_stats.get('total_h2h_matches', 0),
            "team1_h2h_wins": h2h_stats.get('team1_h2h_wins', 0),
            "team2_h2h_wins": h2h_stats.get('team2_h2h_wins', 0),
            "team1_h2h_win_rate": float(h2h_stats.get('team1_h2h_win_rate', 0)),
            "team2_h2h_win_rate": float(h2h_stats.get('team2_h2h_win_rate', 0))
        },
        
        # Feature importance for the prediction
        "feature_importance": {}
    }
    
    # Add top features that influenced the prediction
    top_features = {}
    for idx, row in feature_importance.head(10).iterrows():
        top_features[row['Feature']] = float(row['Importance'])
    betting_json["feature_importance"] = top_features
    
    # Add statistical differentials (useful for betting)
    betting_json["stat_differentials"] = {
        "win_rate": float(team1_stats.get('win_rate', 0) - team2_stats.get('win_rate', 0)),
        "avg_score": float(team1_stats.get('avg_score', 0) - team2_stats.get('avg_score', 0)),
        "avg_opponent_score": float(team1_stats.get('avg_opponent_score', 0) - team2_stats.get('avg_opponent_score', 0)),
        "score_differential": float(team1_stats.get('score_differential', 0) - team2_stats.get('score_differential', 0)),
        "recent_form": float(team1_stats.get('recent_form', 0) - team2_stats.get('recent_form', 0)),
        "ranking": (team1_details.get('ranking', 9999) if team1_details else 9999) - 
                  (team2_details.get('ranking', 9999) if team2_details else 9999),
        "rating": (team1_details.get('rating', 1500) if team1_details else 1500) - 
                 (team2_details.get('rating', 1500) if team2_details else 1500)
    }
    
    # Add performance trends if available
    if team1_trends or team2_trends:
        betting_json["performance_trends"] = {
            "team1": {},
            "team2": {}
        }
        
        if team1_trends:
            if 'recent_matches' in team1_trends:
                for key, value in team1_trends['recent_matches'].items():
                    betting_json["performance_trends"]["team1"][key] = float(value)
            
            if 'recency_weighted_win_rate' in team1_trends:
                betting_json["performance_trends"]["team1"]["recency_weighted_win_rate"] = float(team1_trends['recency_weighted_win_rate'])
        
        if team2_trends:
            if 'recent_matches' in team2_trends:
                for key, value in team2_trends['recent_matches'].items():
                    betting_json["performance_trends"]["team2"][key] = float(value)
            
            if 'recency_weighted_win_rate' in team2_trends:
                betting_json["performance_trends"]["team2"]["recency_weighted_win_rate"] = float(team2_trends['recency_weighted_win_rate'])
    
    # Add map performance data if available
    if team1_map_performance or team2_map_performance:
        betting_json["map_performance"] = {
            "team1": {},
            "team2": {}
        }
        
        if team1_map_performance:
            for map_name, stats in team1_map_performance.items():
                betting_json["map_performance"]["team1"][map_name] = {
                    "played": stats.get('played', 0),
                    "wins": stats.get('wins', 0),
                    "win_rate": float(stats.get('win_rate', 0))
                }
        
        if team2_map_performance:
            for map_name, stats in team2_map_performance.items():
                betting_json["map_performance"]["team2"][map_name] = {
                    "played": stats.get('played', 0),
                    "wins": stats.get('wins', 0),
                    "win_rate": float(stats.get('win_rate', 0))
                }
        
        # Add map differentials for common maps
        betting_json["map_performance"]["differentials"] = {}
        if team1_map_performance and team2_map_performance:
            common_maps = set(team1_map_performance.keys()).intersection(set(team2_map_performance.keys()))
            for map_name in common_maps:
                team1_win_rate = team1_map_performance[map_name].get('win_rate', 0)
                team2_win_rate = team2_map_performance[map_name].get('win_rate', 0)
                betting_json["map_performance"]["differentials"][map_name] = float(team1_win_rate - team2_win_rate)
    
    # Add event performance data (LAN vs Online) if available
    if team1_event_performance or team2_event_performance:
        betting_json["event_performance"] = {
            "team1": {},
            "team2": {}
        }
        
        if team1_event_performance:
            for event_type in ['lan', 'online']:
                if event_type in team1_event_performance:
                    betting_json["event_performance"]["team1"][event_type] = {
                        "matches": team1_event_performance[event_type].get('matches', 0),
                        "wins": team1_event_performance[event_type].get('wins', 0),
                        "win_rate": float(team1_event_performance[event_type].get('win_rate', 0))
                    }
        
        if team2_event_performance:
            for event_type in ['lan', 'online']:
                if event_type in team2_event_performance:
                    betting_json["event_performance"]["team2"][event_type] = {
                        "matches": team2_event_performance[event_type].get('matches', 0),
                        "wins": team2_event_performance[event_type].get('wins', 0),
                        "win_rate": float(team2_event_performance[event_type].get('win_rate', 0))
                    }
    
    # Add specialized betting indicators
    betting_json["betting_indicators"] = {
        # Estimate expected map count
        "expected_map_count": _estimate_expected_map_count(team1_stats, team2_stats, prediction_proba),
        
        # Probability of a 2-0 win for each team
        "team1_2_0_probability": _calculate_2_0_probability(team1_name, prediction, prediction_proba),
        "team2_2_0_probability": _calculate_2_0_probability(team2_name, prediction, prediction_proba),
        
        # Probability of a team winning at least one map
        "team1_win_map_probability": _calculate_win_map_probability(team1_name, prediction, prediction_proba),
        "team2_win_map_probability": _calculate_win_map_probability(team2_name, prediction, prediction_proba),
        
        # Total maps over/under probabilities
        "over_2_5_maps_probability": _calculate_over_under_probability(True, prediction_proba),
        "under_2_5_maps_probability": _calculate_over_under_probability(False, prediction_proba),
        
        # Confidence level in the prediction
        "confidence_level": _determine_confidence_level(win_probability)
    }
    
    return betting_json

# Helper functions for betting indicators
def _estimate_expected_map_count(team1_stats, team2_stats, prediction_proba):
    """Estimate the expected number of maps based on team stats and prediction"""
    # Calculate how close the match is expected to be
    closeness = 1 - abs(prediction_proba[1] - prediction_proba[0])
    
    # Closer matches tend to go to more maps (max 3 in a Bo3)
    # Base expected map count between 2 and 3
    base_map_count = 2 + (closeness * 0.8)
    
    # Adjust based on teams' historical score differentials
    avg_score_diff = abs(team1_stats.get('score_differential', 0) - team2_stats.get('score_differential', 0))
    adjustment = -min(avg_score_diff * 0.1, 0.3)  # Larger score differential means fewer maps
    
    return float(min(max(base_map_count + adjustment, 2.0), 2.99))  # Clamp between 2.0 and 2.99

def _calculate_2_0_probability(team_name, prediction, prediction_proba):
    """Calculate probability of a team winning 2-0"""
    team1_win_prob = prediction_proba[1]
    team2_win_prob = prediction_proba[0]
    
    # Factor in match closeness
    closeness = 1 - abs(team1_win_prob - team2_win_prob)
    
    # Calculate probabilities of each team winning 2-0
    # A higher win probability increases the chance of a 2-0, but closeness reduces it
    if team_name == "team1" or prediction == 1:
        sweep_factor = max(0.3, 0.8 - (closeness * 0.6))  # Between 0.3 and 0.8
        return float(team1_win_prob * sweep_factor)
    else:
        sweep_factor = max(0.3, 0.8 - (closeness * 0.6))
        return float(team2_win_prob * sweep_factor)

def _calculate_win_map_probability(team_name, prediction, prediction_proba):
    """Calculate probability of a team winning at least 1 map"""
    team1_win_prob = prediction_proba[1]
    team2_win_prob = prediction_proba[0]
    
    # Factor in match closeness
    closeness = 1 - abs(team1_win_prob - team2_win_prob)
    
    # Calculate 2-0 probabilities
    team1_2_0_prob = team1_win_prob * max(0.3, 0.8 - (closeness * 0.6))
    team2_2_0_prob = team2_win_prob * max(0.3, 0.8 - (closeness * 0.6))
    
    # Probability of winning at least 1 map = 1 - Probability of getting swept
    if team_name == "team1" or prediction == 1:
        return float(1 - team2_2_0_prob)  # Team 1 wins at least 1 map
    else:
        return float(1 - team1_2_0_prob)  # Team 2 wins at least 1 map

def _calculate_over_under_probability(is_over, prediction_proba):
    """Calculate probability for over/under 2.5 maps"""
    team1_win_prob = prediction_proba[1]
    team2_win_prob = prediction_proba[0]
    
    # Factor in match closeness
    closeness = 1 - abs(team1_win_prob - team2_win_prob)
    
    # Calculate 2-0 probabilities
    team1_2_0_prob = team1_win_prob * max(0.3, 0.8 - (closeness * 0.6))
    team2_2_0_prob = team2_win_prob * max(0.3, 0.8 - (closeness * 0.6))
    
    # Over 2.5 maps means it goes to map 3 (not a 2-0 for either team)
    if is_over:
        return float(1 - team1_2_0_prob - team2_2_0_prob)
    else:
        return float(team1_2_0_prob + team2_2_0_prob)

def _determine_confidence_level(win_probability):
    """Determine the confidence level in the prediction"""
    if win_probability >= 0.8:
        return "Very High"
    elif win_probability >= 0.65:
        return "High"
    elif win_probability >= 0.55:
        return "Moderate"
    elif win_probability >= 0.45:
        return "Balanced"
    elif win_probability >= 0.35:
        return "Low"
    else:
        return "Very Low"


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict the outcome of a match between two specific Valorant teams')
    parser.add_argument('--team1', type=str, help='Name of the first team')
    parser.add_argument('--team2', type=str, help='Name of the second team')
    parser.add_argument('--team1_region', type=str, help='Region of the first team (na, eu, etc.)', default=None)
    parser.add_argument('--team2_region', type=str, help='Region of the second team (na, eu, etc.)', default=None)
    parser.add_argument('--advanced', action='store_true', help='Use advanced prediction model with more features')
    
    args = parser.parse_args()
    
    # Get team names from command line or prompt the user
    team1_name = args.team1 if args.team1 else input("Enter the name of the first team: ")
    team2_name = args.team2 if args.team2 else input("Enter the name of the second team: ")
    team1_region = args.team1_region
    team2_region = args.team2_region
    use_advanced = args.advanced
    
    # If regions weren't provided through args, ask for them
    if not team1_region:
        team1_region = input(f"Enter region for {team1_name} (e.g., na, eu, kr) or leave blank: ") or None
    if not team2_region:
        team2_region = input(f"Enter region for {team2_name} (e.g., na, eu, kr) or leave blank: ") or None
    
    print(f"\nAnalyzing match: {team1_name} vs {team2_name}")
    print("============================================")
    
    # Step 1: Get team IDs
    team1_id = get_team_id(team1_name, team1_region)
    team2_id = get_team_id(team2_name, team2_region)
    
    if not team1_id or not team2_id:
        print("Error: Could not find team IDs. Please check team names and try again.")
        return
    
    # Step 2: Fetch team details
    team1_details = fetch_team_details(team1_id)
    team2_details = fetch_team_details(team2_id)
    
    # Step 3: Fetch match histories
    team1_history = fetch_team_match_history(team1_id)
    team2_history = fetch_team_match_history(team2_id)
    
    if not team1_history or not team2_history:
        print("Error: Could not fetch match histories. Please try again.")
        return
    
    # Step 4: Parse match data
    team1_matches = parse_match_data(team1_history, team1_name)
    team2_matches = parse_match_data(team2_history, team2_name)
    
    print(f"Found {len(team1_matches)} matches for {team1_name}")
    print(f"Found {len(team2_matches)} matches for {team2_name}")
    
    if len(team1_matches) == 0 or len(team2_matches) == 0:
        print("Error: Not enough match data for prediction.")
        return
    
    # Get all teams for additional metrics
    all_teams_response = requests.get(f"{API_URL}/teams?limit=100")
    all_teams = all_teams_response.json().get('data', []) if all_teams_response.status_code == 200 else []
    
    # Step 5: Calculate team statistics
    team1_stats = calculate_team_stats(team1_matches)
    team2_stats = calculate_team_stats(team2_matches)
    
    # Step 6: Calculate head-to-head statistics
    h2h_stats = calculate_head_to_head(team1_matches, team2_name, team2_matches, team1_name)
    
    # Decide which prediction approach to use
    if use_advanced:
        print("\nUsing advanced prediction model with additional features...")
        
        # Calculate all advanced metrics
        print("Calculating advanced metrics and features...")
        team1_player_stats = fetch_team_player_stats(team1_id)
        team2_player_stats = fetch_team_player_stats(team2_id)
        
        team1_map_performance = extract_map_performance(team1_matches)
        team2_map_performance = extract_map_performance(team2_matches)
        
        team1_tournament_performance = extract_tournament_performance(team1_matches)
        team2_tournament_performance = extract_tournament_performance(team2_matches)
        
        team1_trends = analyze_performance_trends(team1_matches)
        team2_trends = analyze_performance_trends(team2_matches)
        
        team1_opponent_quality = analyze_opponent_quality(team1_matches, all_teams)
        team2_opponent_quality = analyze_opponent_quality(team2_matches, all_teams)
        
        team1_event_performance = analyze_event_performance(team1_matches)
        team2_event_performance = analyze_event_performance(team2_matches)
        
        team1_player_metrics = aggregate_player_metrics(team1_player_stats)
        team2_player_metrics = aggregate_player_metrics(team2_player_stats)
        
        # Generate comprehensive features
        prediction_features = create_comprehensive_features(team1_id, team2_id)
        
        # Step 7: Create enhanced training data
        training_data = create_enhanced_training_data(team1_matches, team2_matches, team1_id, team2_id, all_teams)
        
        if len(training_data) < 10:
            print("Warning: Not enough training data for advanced model. Falling back to basic model.")
            training_data = create_training_data(team1_matches, team2_matches, team1_stats, team2_stats, team1_details, team2_details)
            # Train basic model
            model, scaler, feature_names = train_predictive_model(training_data)
            # Create prediction features
            prediction_features = create_prediction_features(team1_stats, team2_stats, h2h_stats, team1_details, team2_details)
            # Make prediction
            prediction, prediction_proba = predict_match_outcome(model, scaler, feature_names, prediction_features)
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
        else:
            print(f"Created enhanced training dataset with {len(training_data)} matches")
            
            # Step 8: Train advanced predictive model
            model, scaler, selector, selected_features = train_advanced_predictive_model(training_data)
            
            # Filter prediction features to match selected features
            all_training_features = training_data.columns.drop('team1_won').tolist()
            filtered_prediction_features = {}
            
            # First create filtered features dictionary with all training features
            for feature in all_training_features:
                if feature in prediction_features:
                    filtered_prediction_features[feature] = prediction_features[feature]
                else:
                    filtered_prediction_features[feature] = 0
                    
            # Step 9: Make prediction using filtered features
            prediction, prediction_proba = predict_match_outcome(model, scaler, selector, filtered_prediction_features)
            
            # Get feature importance directly from the model's underlying estimator
            if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                # For CalibratedClassifierCV, get the base estimator's feature importances
                if hasattr(model.estimators_[0], 'feature_importances_'):
                    importances = model.estimators_[0].feature_importances_
                    feature_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                else:
                    feature_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': [0] * len(selected_features)
                    })
            else:
                feature_importance = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': [0] * len(selected_features)
                })
    else:
        print("\nUsing basic prediction model...")
        # Step 7: Create training data
        training_data = create_training_data(team1_matches, team2_matches, team1_stats, team2_stats, team1_details, team2_details)
        
        if len(training_data) == 0:
            print("Error: Could not create training data. Not enough matches with common opponents.")
            return
        
        print(f"Created training dataset with {len(training_data)} matches")
        
        # Step 8: Train predictive model
        model, scaler, feature_names = train_predictive_model(training_data)
        
        # Step 9: Create prediction features
        prediction_features = create_prediction_features(team1_stats, team2_stats, h2h_stats, team1_details, team2_details)
        
        # Step 10: Make prediction
        prediction, prediction_proba = predict_match_outcome(model, scaler, feature_names, prediction_features)
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    
    # Step 11: Display results
    print("\n============ MATCH PREDICTION ============")
    print(f"Team 1: {team1_name}")
    print(f"Team 2: {team2_name}")
    
    if prediction == 1:
        winner = team1_name
        win_probability = prediction_proba[1]
    else:
        winner = team2_name
        win_probability = prediction_proba[0]
    
    print(f"\nPredicted Winner: {winner}")
    print(f"Win Probability: {win_probability:.2%}")
    
    # Display team stats comparison
    print("\n--- Team Stats Comparison ---")
    print(f"{'Statistic':<20} {team1_name:<20} {team2_name:<20}")
    print("-" * 60)
    print(f"{'Matches':<20} {team1_stats.get('matches', 0):<20} {team2_stats.get('matches', 0):<20}")
    print(f"{'Wins':<20} {team1_stats.get('wins', 0):<20} {team2_stats.get('wins', 0):<20}")
    
    # Format percentages
    team1_win_rate = team1_stats.get('win_rate', 0) * 100
    team2_win_rate = team2_stats.get('win_rate', 0) * 100
    print(f"{'Win Rate':<20} {team1_win_rate:.2f}%{'':<15} {team2_win_rate:.2f}%")
    
    print(f"{'Avg Score':<20} {team1_stats.get('avg_score', 0):.2f}{'':<18} {team2_stats.get('avg_score', 0):.2f}")
    print(f"{'Score Diff':<20} {team1_stats.get('score_differential', 0):.2f}{'':<18} {team2_stats.get('score_differential', 0):.2f}")
    
    # Format percentages
    team1_recent_form = team1_stats.get('recent_form', 0) * 100
    team2_recent_form = team2_stats.get('recent_form', 0) * 100
    print(f"{'Recent Form':<20} {team1_recent_form:.2f}%{'':<15} {team2_recent_form:.2f}%")
    
    if h2h_stats.get('total_h2h_matches', 0) > 0:
        print("\n--- Head-to-Head Stats ---")
        print(f"Total H2H Matches: {h2h_stats.get('total_h2h_matches', 0)}")
        print(f"{team1_name} H2H Wins: {h2h_stats.get('team1_h2h_wins', 0)}")
        print(f"{team2_name} H2H Wins: {h2h_stats.get('team2_h2h_wins', 0)}")
        
        # Format percentages
        team1_h2h_win_rate = h2h_stats.get('team1_h2h_win_rate', 0) * 100
        team2_h2h_win_rate = h2h_stats.get('team2_h2h_win_rate', 0) * 100
        print(f"{team1_name} H2H Win Rate: {team1_h2h_win_rate:.2f}%")
        print(f"{team2_name} H2H Win Rate: {team2_h2h_win_rate:.2f}%")
    
    # Display top features
    print("\n--- Most Important Prediction Factors ---")
    print(feature_importance.head(5))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_results(team1_name, team2_name, prediction, prediction_proba, team1_stats, team2_stats, h2h_stats, feature_importance)
    
    # Save prediction results
    prediction_result = {
        'match': f"{team1_name} vs {team2_name}",
        'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'predicted_winner': winner,
        'win_probability': float(win_probability),
        'team1_stats': team1_stats,
        'team2_stats': team2_stats,
        'h2h_stats': h2h_stats,
        'feature_importance': feature_importance.head(10).to_dict()
    }
    
    os.makedirs("predictions", exist_ok=True)
    with open(f"predictions/{team1_name}_vs_{team2_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(prediction_result, f, indent=4, default=str)
    
    print("\nPrediction saved to predictions directory.")
    print("Analysis complete!")

    betting_json = create_betting_optimized_json(
        team1_name, team2_name, prediction, prediction_proba, 
        team1_stats, team2_stats, h2h_stats, feature_importance,
        team1_details, team2_details,
        team1_trends=team1_trends if 'team1_trends' in locals() else None,
        team2_trends=team2_trends if 'team2_trends' in locals() else None,
        team1_map_performance=team1_map_performance if 'team1_map_performance' in locals() else None,
        team2_map_performance=team2_map_performance if 'team2_map_performance' in locals() else None,
        team1_event_performance=team1_event_performance if 'team1_event_performance' in locals() else None,
        team2_event_performance=team2_event_performance if 'team2_event_performance' in locals() else None
    )
    
    # Replace spaces with underscores in team names
    team1_name_clean = team1_name.replace(" ", "_")
    team2_name_clean = team2_name.replace(" ", "_")


    # Save the betting JSON to a separate file
    os.makedirs("betting_predictions", exist_ok=True)
    betting_json_path = f"betting_predictions/{team1_name_clean}_vs_{team2_name_clean}.json"
    with open(betting_json_path, 'w') as f:
        json.dump(betting_json, f, indent=4, default=str)
    
    print(f"\nBetting-optimized prediction saved to {betting_json_path}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()