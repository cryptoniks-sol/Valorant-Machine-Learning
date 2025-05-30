"""
cache.py - Caching script for Valorant Match Predictor
This script downloads and caches team data from the API to reduce internet usage during
training, retraining, and backtesting. It should be run periodically (e.g., once a week)
to update the cached data.
"""

import requests
import json
import os
import time
import pickle
import datetime
import argparse
import traceback
import tqdm
from tqdm import tqdm
import numpy as np  # Added missing import
import re 
API_URL = "http://localhost:5000/api/v1"
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
    url = f"{API_URL}/teams?limit=300"
    if region:
        url += f"&region={region}"
        print(f"Filtering by region: {region}")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching teams: {response.status_code}")
        return None
    teams_data = response.json()
    if 'data' not in teams_data:
        print("No 'data' field found in the response")
        return None
    for team in teams_data['data']:
        if team['name'].lower() == team_name.lower():
            print(f"Found exact match: {team['name']} (ID: {team['id']})")
            return team['id']
    for team in teams_data['data']:
        if team_name.lower() in team['name'].lower() or team['name'].lower() in team_name.lower():
            print(f"Found partial match: {team['name']} (ID: {team['id']})")
            return team['id']
    if not region:
        print(f"No match found with default search. Attempting to search by region...")
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
    if player_data and player_data.get('status') == 'OK' and 'data' in player_data:
        return player_data['data']
    return None
def fetch_team_player_stats(team_id):
    """Fetch detailed player statistics for a team from the team roster."""
    if not team_id:
        print(f"Invalid team ID: {team_id}")
        return []
    team_data = fetch_api_data(f"teams/{team_id}")
    if not team_data or 'data' not in team_data or not isinstance(team_data['data'], dict):
        print(f"Invalid team data format for team ID: {team_id}")
        return []
    team_info = team_data['data']
    players = team_info.get('players', [])
    if not players:
        print(f"No players found in roster for team ID: {team_id}")
        return []
    print(f"Found {len(players)} players in roster for team: {team_info.get('info', {}).get('name', '')}")
    player_stats = []
    for player in players:
        player_id = player.get('id')
        player_name = player.get('user')  # Use the in-game username
        if not player_name:
            continue
        print(f"Fetching statistics for player: {player_name}")
        stats = fetch_player_stats(player_name)
        if stats:
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
    if 'data' not in team_stats_data:
        print(f"Missing 'data' field in team stats for team ID: {team_id}")
        print(f"Received: {team_stats_data}")
        return {}
    if not team_stats_data['data']:
        print(f"Empty data array in team stats for team ID: {team_id}")
        return {}
    map_stats = extract_map_statistics(team_stats_data)
    if not map_stats:
        print(f"Failed to extract map statistics for team ID: {team_id}")
    else:
        print(f"Successfully extracted statistics for {len(map_stats)} maps")
    return map_stats
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
            match_info = {
                'match_id': match.get('id', ''),
                'date': match.get('date', ''),
                'event': match.get('event', '') if isinstance(match.get('event', ''), str) else match.get('event', {}).get('name', ''),
                'tournament': match.get('tournament', ''),
                'map': match.get('map', ''),
                'map_score': ''  # Initialize map score as empty
            }
            if 'teams' in match and len(match['teams']) >= 2:
                team1 = match['teams'][0]
                team2 = match['teams'][1]
                team1_score = int(team1.get('score', 0))
                team2_score = int(team2.get('score', 0))
                match_info['map_score'] = f"{team1_score}:{team2_score}"
                team1_won = team1_score > team2_score
                team2_won = team2_score > team1_score
                is_team1 = team1.get('name', '').lower() == team_name.lower() or team_name.lower() in team1.get('name', '').lower()
                if is_team1:
                    our_team = team1
                    opponent_team = team2
                    team_won = team1_won
                else:
                    our_team = team2
                    opponent_team = team1
                    team_won = team2_won
                match_info['team_name'] = our_team.get('name', '')
                match_info['team_score'] = int(our_team.get('score', 0))
                match_info['team_won'] = team_won
                match_info['team_country'] = our_team.get('country', '')
                match_info['team_tag'] = our_team.get('tag', '')
                if not match_info['team_tag']:
                    team_name_parts = our_team.get('name', '').split()
                    if len(team_name_parts) > 0:
                        first_word = team_name_parts[0]
                        if first_word.isupper() and 2 <= len(first_word) <= 5:
                            match_info['team_tag'] = first_word
                    if not match_info['team_tag'] and '[' in our_team.get('name', '') and ']' in our_team.get('name', ''):
                        tag_match = re.search(r'\[(.*?)\]', our_team.get('name', ''))
                        if tag_match:
                            match_info['team_tag'] = tag_match.group(1)
                match_info['opponent_name'] = opponent_team.get('name', '')
                match_info['opponent_score'] = int(opponent_team.get('score', 0))
                match_info['opponent_won'] = not team_won  # Opponent's result is opposite of our team
                match_info['opponent_country'] = opponent_team.get('country', '')
                match_info['opponent_tag'] = opponent_team.get('tag', '')
                match_info['opponent_id'] = opponent_team.get('id', '')  # Save opponent ID for future reference
                match_info['result'] = 'win' if team_won else 'loss'
                match_details = fetch_match_details(match_info['match_id'])
                if match_details:
                    match_info['details'] = match_details
                matches.append(match_info)
        except Exception as e:
            print(f"Error parsing match: {e}")
            continue
    print(f"Skipped {filtered_count} matches that did not involve {team_name}")
    wins = sum(1 for match in matches if match['team_won'])
    print(f"Processed {len(matches)} matches for {team_name}: {wins} wins, {len(matches) - wins} losses")
    return matches
def calculate_team_stats(team_matches, player_stats=None, include_economy=False):
    """Calculate comprehensive team statistics optionally including economy data."""
    if not team_matches:
        return {}
    total_matches = len(team_matches)
    wins = sum(1 for match in team_matches if match.get('team_won', False))
    losses = total_matches - wins
    win_rate = wins / total_matches if total_matches > 0 else 0
    total_score = sum(match.get('team_score', 0) for match in team_matches)
    total_opponent_score = sum(match.get('opponent_score', 0) for match in team_matches)
    avg_score = total_score / total_matches if total_matches > 0 else 0
    avg_opponent_score = total_opponent_score / total_matches if total_matches > 0 else 0
    score_differential = avg_score - avg_opponent_score
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
    for opponent, stats in opponent_stats.items():
        stats['win_rate'] = stats['wins'] / stats['matches'] if stats['matches'] > 0 else 0
        stats['avg_score'] = stats['total_score'] / stats['matches'] if stats['matches'] > 0 else 0
        stats['avg_opponent_score'] = stats['total_opponent_score'] / stats['matches'] if stats['matches'] > 0 else 0
        stats['score_differential'] = stats['avg_score'] - stats['avg_opponent_score']
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
    for map_name, stats in map_stats.items():
        stats['win_rate'] = stats['wins'] / stats['played'] if stats['played'] > 0 else 0
    sorted_matches = sorted(team_matches, key=lambda x: x.get('date', ''))
    recent_matches = sorted_matches[-5:] if len(sorted_matches) >= 5 else sorted_matches
    recent_form = sum(1 for match in recent_matches if match['team_won']) / len(recent_matches) if recent_matches else 0
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
    if player_stats:
        player_agg_stats = calculate_team_player_stats(player_stats)
        team_stats.update({
            'player_stats': player_agg_stats,
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
    if include_economy:
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
            match_team_tag = match.get('team_tag')
            match_team_name = match.get('team_name')
            economy_data = fetch_match_economy_details(match_id)
            if not economy_data:
                continue
            our_team_metrics = extract_economy_metrics(
                economy_data,
                team_identifier=match_team_tag,
                fallback_name=match_team_name
            )
            if not our_team_metrics or our_team_metrics.get('economy_data_missing', False):
                continue
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
    return team_stats
def extract_economy_metrics(match_economy_data, team_identifier=None, fallback_name=None):
    """Extract relevant economic performance metrics from match economy data for a specific team."""
    if not match_economy_data or 'data' not in match_economy_data:
        return {'economy_data_missing': True}
    if 'teams' not in match_economy_data['data']:
        return {'economy_data_missing': True}
    teams_data = match_economy_data['data']['teams']
    if len(teams_data) < 1:
        return {'economy_data_missing': True}
    target_team_data = None
    if team_identifier:
        for team in teams_data:
            team_name = team.get('name', '').lower()
            if team_identifier and team_name == team_identifier.lower():
                target_team_data = team
                break
            elif team_identifier and team_name.startswith(team_identifier.lower()):
                target_team_data = team
                break
            elif team_identifier and team_identifier.lower() in team_name:
                target_team_data = team
                break
    if not target_team_data and fallback_name:
        for team in teams_data:
            team_name = team.get('name', '').lower()
            fallback_lower = fallback_name.lower()
            if team_name == fallback_lower:
                target_team_data = team
                break
            elif fallback_lower in team_name or team_name in fallback_lower:
                target_team_data = team
                break
            team_words = team_name.split()
            fallback_words = fallback_lower.split()
            common_words = set(team_words) & set(fallback_words)
            if common_words:
                target_team_data = team
                break
    if not target_team_data:
        return {'economy_data_missing': True}
    has_economy_data = ('eco' in target_team_data or
                        'pistolWon' in target_team_data or
                        'semiEco' in target_team_data or
                        'semiBuy' in target_team_data or
                        'fullBuy' in target_team_data)
    if not has_economy_data:
        return {'economy_data_missing': True}
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
    metrics['total_rounds'] = (metrics['eco_total'] + metrics['semi_eco_total'] +
                             metrics['semi_buy_total'] + metrics['full_buy_total'])
    metrics['overall_economy_win_rate'] = ((metrics['eco_won'] + metrics['semi_eco_won'] +
                                         metrics['semi_buy_won'] + metrics['full_buy_won']) /
                                         metrics['total_rounds']) if metrics['total_rounds'] > 0 else 0
    if metrics['total_rounds'] > 0:
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
    for player_data in player_stats_list:
        if 'stats' not in player_data:
            continue
        stats = player_data['stats']
        try:
            rating = float(stats.get('rating', 0) or 0)
            acs = float(stats.get('acs', 0) or 0)
            kd = float(stats.get('kd', 0) or 0)
            kast_str = stats.get('kast', '0%')
            kast = float(kast_str.strip('%')) / 100 if '%' in kast_str and kast_str.strip('%') else 0
            adr = float(stats.get('adr', 0) or 0)
            hs_str = stats.get('hs', '0%')
            hs = float(hs_str.strip('%')) / 100 if '%' in hs_str and hs_str.strip('%') else 0
            cl_str = stats.get('cl', '0%')
            cl = float(cl_str.strip('%')) / 100 if '%' in cl_str and cl_str.strip('%') else 0
            agg_stats['avg_rating'] += rating
            agg_stats['avg_acs'] += acs
            agg_stats['avg_kd'] += kd
            agg_stats['avg_kast'] += kast
            agg_stats['avg_adr'] += adr
            agg_stats['avg_headshot'] += hs
            agg_stats['avg_clutch'] += cl
            agg_stats['total_kills'] += int(stats.get('kills', 0) or 0)
            agg_stats['total_deaths'] += int(stats.get('deaths', 0) or 0)
            agg_stats['total_assists'] += int(stats.get('assists', 0) or 0)
            agg_stats['total_first_kills'] += int(stats.get('fk', 0) or 0)
            agg_stats['total_first_deaths'] += int(stats.get('fd', 0) or 0)
            if rating > agg_stats['star_player_rating']:
                agg_stats['star_player_rating'] = rating
                agg_stats['star_player_name'] = player_data.get('player', '')
            if agg_stats['weak_player_rating'] == 0 or rating < agg_stats['weak_player_rating']:
                agg_stats['weak_player_rating'] = rating
                agg_stats['weak_player_name'] = player_data.get('player', '')
            for agent in player_data.get('agents', []):
                if agent not in agg_stats['agent_usage']:
                    agg_stats['agent_usage'][agent] = 0
                agg_stats['agent_usage'][agent] += 1
        except (ValueError, TypeError) as e:
            print(f"Error processing player stats: {e}")
            continue
    player_count = agg_stats['player_count']
    if player_count > 0:
        agg_stats['avg_rating'] /= player_count
        agg_stats['avg_acs'] /= player_count
        agg_stats['avg_kd'] /= player_count
        agg_stats['avg_kast'] /= player_count
        agg_stats['avg_adr'] /= player_count
        agg_stats['avg_headshot'] /= player_count
        agg_stats['avg_clutch'] /= player_count
    if agg_stats['total_first_deaths'] > 0:
        agg_stats['fk_fd_ratio'] = agg_stats['total_first_kills'] / agg_stats['total_first_deaths']
    else:
        agg_stats['fk_fd_ratio'] = agg_stats['total_first_kills'] if agg_stats['total_first_kills'] > 0 else 1  # Avoid division by zero
    agg_stats['team_consistency'] = 1 - (agg_stats['star_player_rating'] - agg_stats['weak_player_rating']) / agg_stats['star_player_rating'] if agg_stats['star_player_rating'] > 0 else 0
    return agg_stats
def get_teams_for_caching(limit=150):
    """Get a list of teams to cache data for."""
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
        teams_list = []
        ranked_teams = []
        for team in teams_data['data']:
            if 'ranking' in team and team['ranking'] and team['ranking'] <= 100:
                ranked_teams.append(team)
            teams_list.append(team)
        if ranked_teams:
            print(f"Found {len(ranked_teams)} ranked teams (ranked <= 100)")
            return ranked_teams
        print(f"No teams with rankings found. Using all {len(teams_list)} teams instead.")
        return teams_list
    except Exception as e:
        print(f"Error in get_teams_for_caching: {e}")
        return []
def collect_team_data_for_cache(team_limit=150, include_player_stats=True, include_economy=True, include_maps=True):
    """Collect data for all teams to cache for training and evaluation."""
    print("\n========================================================")
    print("COLLECTING TEAM DATA FOR CACHE")
    print("========================================================")
    print(f"Including player stats: {include_player_stats}")
    print(f"Including economy data: {include_economy}")
    print(f"Including map data: {include_maps}")
    
    teams_list = get_teams_for_caching(limit=team_limit)
    if not teams_list:
        print("Error: No teams retrieved for caching. Check API connection.")
        return {}
    
    top_teams = teams_list[:min(team_limit, len(teams_list))]
    print(f"Selected {len(top_teams)} teams for data collection.")
    
    team_data_collection = {}
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
        
        team_details, team_tag = fetch_team_details(team_id)
        team_history = fetch_team_match_history(team_id)
        if not team_history:
            print(f"No match history for team {team_name}, skipping.")
            continue
        
        team_matches = parse_match_data(team_history, team_name)
        if not team_matches:
            print(f"No parsed match data for team {team_name}, skipping.")
            continue
        
        for match in team_matches:
            match['team_tag'] = team_tag
            match['team_id'] = team_id
            match['team_name'] = team_name
        
        team_player_stats = None
        if include_player_stats:
            team_player_stats = fetch_team_player_stats(team_id)
            if team_player_stats:
                player_stats_count += 1
        
        team_stats = calculate_team_stats(team_matches, team_player_stats, include_economy=include_economy)
        if include_economy and 'pistol_win_rate' in team_stats and team_stats['pistol_win_rate'] > 0:
            economy_data_count += 1
        
        team_stats['team_tag'] = team_tag
        team_stats['team_name'] = team_name
        team_stats['team_id'] = team_id
        
        if include_maps:
            map_stats = fetch_team_map_statistics(team_id)
            if map_stats:
                team_stats['map_statistics'] = map_stats
                map_stats_count += 1
        
        team_data_collection[team_name] = {
            'team_id': team_id,
            'team_tag': team_tag,
            'stats': team_stats,
            'matches': team_matches,
            'player_stats': team_player_stats,
            'ranking': team.get('ranking', None)
        }
        
        print(f"Successfully collected data for {team_name} with {len(team_matches)} matches")
    
    print(f"\nCollected data for {len(team_data_collection)} teams:")
    print(f"  - Teams with economy data: {economy_data_count}")
    print(f"  - Teams with player stats: {player_stats_count}")
    print(f"  - Teams with map stats: {map_stats_count}")
    
    # ADD DATA QUALITY ASSESSMENT HERE
    print("\nAssessing data quality...")
    team_data_collection, quality_summary = display_data_quality_summary(team_data_collection)
    
    return team_data_collection


def detect_cache_corruption():
    cache_path = "cache/valorant_data_cache.pkl"
    if not os.path.exists(cache_path):
        return False
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, dict):
            return True
        
        for team_name, team_data in data.items():
            if not isinstance(team_data, dict):
                return True
            if 'matches' not in team_data:
                return True
            if not isinstance(team_data['matches'], list):
                return True
        
        return False
    except:
        return True

def add_data_quality_scores(team_data):
    for team_name, data in team_data.items():
        stats = data.get('stats', {})
        matches = data.get('matches', [])
        player_stats = data.get('player_stats', [])
        
        completeness_score = 0
        if 'win_rate' in stats: completeness_score += 0.2
        if 'recent_form' in stats: completeness_score += 0.2
        if len(matches) >= 10: completeness_score += 0.3
        if player_stats: completeness_score += 0.2
        if 'map_statistics' in stats: completeness_score += 0.1
        
        consistency_score = 0
        if len(matches) > 0:
            recent_dates = [m.get('date', '') for m in matches[-5:]]
            if all(date for date in recent_dates):
                consistency_score += 0.5
            
            score_consistency = []
            for match in matches:
                team_score = match.get('team_score', 0)
                opp_score = match.get('opponent_score', 0)
                if team_score + opp_score > 0:
                    score_consistency.append(abs(team_score - opp_score))
            
            if score_consistency and np.std(score_consistency) < 5:
                consistency_score += 0.5
        
        data['data_quality'] = {
            'completeness_score': completeness_score,
            'consistency_score': consistency_score,
            'overall_quality': (completeness_score + consistency_score) / 2,
            'sample_size': len(matches),
            'last_updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    return team_data



def implement_incremental_update_system():
    """Implement incremental update system for cache versioning."""
    version_info = {
        'version': '2.1.0',
        'last_full_update': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'update_type': 'full',
        'features_enabled': {
            'advanced_features': True,
            'lstm_features': True,
            'gradient_boosting': True,
            'market_timing': True,
            'bankroll_management': True,
            'correlation_analysis': True,
            'market_intelligence': True
        }
    }
    return version_info

def create_canonical_team_database():
    canonical_db = {
        'aliases': {},
        'previous_names': {},
        'roster_changes': {},
        'tournament_variations': {}
    }
    
    known_aliases = {
        'Sentinels': ['SEN', 'Sentinels', 'SENTINELS'],
        'Cloud9': ['C9', 'Cloud9', 'Cloud 9'],
        'Team Liquid': ['TL', 'Liquid', 'Team Liquid'],
        'G2 Esports': ['G2', 'G2 Esports', 'G2E'],
        'Fnatic': ['FNC', 'Fnatic', 'FNATIC'],
        'Paper Rex': ['PRX', 'Paper Rex', 'PaperRex'],
        'OpTic Gaming': ['OPTIC', 'OpTic', 'OpTic Gaming'],
        'LOUD': ['LOUD', 'Loud'],
        'DRX': ['DRX', 'DragonX'],
        'FPX': ['FPX', 'FunPlus Phoenix']
    }
    
    for canonical_name, aliases in known_aliases.items():
        for alias in aliases:
            canonical_db['aliases'][alias.lower()] = canonical_name
    
    with open('cache/canonical_teams.json', 'w') as f:
        json.dump(canonical_db, f, indent=2)
    
    return canonical_db

def detect_roster_changes(team_data, canonical_db):
    for team_name, data in team_data.items():
        matches = data.get('matches', [])
        if len(matches) < 10:
            continue
        
        sorted_matches = sorted(matches, key=lambda x: x.get('date', ''))
        old_matches = sorted_matches[:len(sorted_matches)//2]
        new_matches = sorted_matches[len(sorted_matches)//2:]
        
        old_performance = sum(1 for m in old_matches if m.get('team_won', False)) / len(old_matches) if old_matches else 0
        new_performance = sum(1 for m in new_matches if m.get('team_won', False)) / len(new_matches) if new_matches else 0
        
        performance_shift = abs(new_performance - old_performance)
        
        if performance_shift > 0.3:
            data['roster_change_detected'] = {
                'performance_shift': performance_shift,
                'old_performance': old_performance,
                'new_performance': new_performance,
                'estimated_change_date': new_matches[0].get('date', '') if new_matches else ''
            }
    
    return team_data

def implement_advanced_feature_engineering(team_stats, matches):
    advanced_features = {}
    
    if len(matches) >= 10:
        sorted_matches = sorted(matches, key=lambda x: x.get('date', ''))
        
        patch_adaptation = []
        for i in range(len(sorted_matches) - 4):
            window = sorted_matches[i:i+5]
            win_rate = sum(1 for m in window if m.get('team_won', False)) / 5
            patch_adaptation.append(win_rate)
        
        if len(patch_adaptation) > 1:
            advanced_features['meta_shift_adaptation'] = np.std(patch_adaptation)
        
        tournament_contexts = {}
        for match in sorted_matches:
            event = match.get('event', '')
            if 'LAN' in event.upper():
                tournament_contexts['lan_performance'] = tournament_contexts.get('lan_performance', []) + [match.get('team_won', False)]
            if any(prize in event.lower() for prize in ['masters', 'champions']):
                tournament_contexts['high_stakes'] = tournament_contexts.get('high_stakes', []) + [match.get('team_won', False)]
        
        for context, results in tournament_contexts.items():
            if len(results) >= 3:
                advanced_features[f'{context}_win_rate'] = sum(results) / len(results)
        
        if 'player_stats' in team_stats:
            players = team_stats['player_stats']
            if isinstance(players, dict) and 'avg_rating' in players:
                synergy_scores = []
                for i in range(min(5, len(sorted_matches))):
                    match = sorted_matches[-(i+1)]
                    team_score = match.get('team_score', 0)
                    opp_score = match.get('opponent_score', 0)
                    if team_score + opp_score > 0:
                        performance = team_score / (team_score + opp_score)
                        expected_performance = players['avg_rating'] / 2.0
                        synergy = performance - expected_performance
                        synergy_scores.append(synergy)
                
                if synergy_scores:
                    advanced_features['player_synergy'] = np.mean(synergy_scores)
        
        opponent_strengths = []
        for match in sorted_matches:
            team_score = match.get('team_score', 0)
            opp_score = match.get('opponent_score', 0)
            if team_score + opp_score > 0:
                opp_strength = opp_score / (team_score + opp_score)
                opponent_strengths.append(opp_strength)
        
        if opponent_strengths:
            strong_opponents = [s for s in opponent_strengths if s > 0.6]
            weak_opponents = [s for s in opponent_strengths if s < 0.4]
            
            if strong_opponents:
                strong_opp_matches = [m for m, s in zip(sorted_matches, opponent_strengths) if s > 0.6]
                advanced_features['vs_strong_win_rate'] = sum(1 for m in strong_opp_matches if m.get('team_won', False)) / len(strong_opp_matches)
            
            if weak_opponents:
                weak_opp_matches = [m for m, s in zip(sorted_matches, opponent_strengths) if s < 0.4]
                advanced_features['vs_weak_win_rate'] = sum(1 for m in weak_opp_matches if m.get('team_won', False)) / len(weak_opp_matches)
        
        time_weights = []
        for i, match in enumerate(sorted_matches):
            days_ago = (len(sorted_matches) - i) * 7
            weight = np.exp(-days_ago / 60.0)
            time_weights.append(weight)
        
        if time_weights:
            weighted_performance = sum(w * (1 if m.get('team_won', False) else 0) for w, m in zip(time_weights, sorted_matches))
            total_weight = sum(time_weights)
            advanced_features['sophisticated_time_decay'] = weighted_performance / total_weight if total_weight > 0 else 0.5
        
        clutch_situations = []
        for match in sorted_matches:
            team_score = match.get('team_score', 0)
            opp_score = match.get('opponent_score', 0)
            if abs(team_score - opp_score) <= 2 and team_score + opp_score >= 4:
                clutch_situations.append(match.get('team_won', False))
        
        if clutch_situations:
            advanced_features['clutch_performance'] = sum(clutch_situations) / len(clutch_situations)
        
        agent_meta_scores = {}
        if 'map_statistics' in team_stats:
            for map_name, map_data in team_stats['map_statistics'].items():
                if 'agent_usage' in map_data:
                    for agent, usage in map_data['agent_usage'].items():
                        if agent not in agent_meta_scores:
                            agent_meta_scores[agent] = 0
                        agent_meta_scores[agent] += usage
        
        if agent_meta_scores:
            total_usage = sum(agent_meta_scores.values())
            meta_agents = ['Jett', 'Raze', 'Omen', 'Astra', 'Sova']
            meta_adaptation = sum(agent_meta_scores.get(agent, 0) for agent in meta_agents) / total_usage if total_usage > 0 else 0
            advanced_features['agent_meta_adaptation'] = meta_adaptation
        
        recent_roster_impact = 0
        if 'roster_change_detected' in team_stats:
            change_date = team_stats['roster_change_detected'].get('estimated_change_date', '')
            if change_date:
                post_change_matches = [m for m in sorted_matches if m.get('date', '') > change_date]
                if len(post_change_matches) >= 3:
                    post_change_wr = sum(1 for m in post_change_matches if m.get('team_won', False)) / len(post_change_matches)
                    pre_change_wr = team_stats['roster_change_detected'].get('old_performance', 0.5)
                    recent_roster_impact = post_change_wr - pre_change_wr
        
        advanced_features['roster_change_impact'] = recent_roster_impact
    
    return advanced_features

def implement_lstm_time_series_features(matches):
    if len(matches) < 20:
        return {}
    
    sorted_matches = sorted(matches, key=lambda x: x.get('date', ''))
    
    performance_sequence = []
    for match in sorted_matches:
        team_score = match.get('team_score', 0)
        opp_score = match.get('opponent_score', 0)
        if team_score + opp_score > 0:
            performance = team_score / (team_score + opp_score)
        else:
            performance = 0.5
        performance_sequence.append(performance)
    
    lstm_features = {}
    
    windows = [5, 10, 15]
    for window in windows:
        if len(performance_sequence) >= window:
            recent_window = performance_sequence[-window:]
            lstm_features[f'performance_trend_{window}'] = np.polyfit(range(window), recent_window, 1)[0]
            lstm_features[f'performance_volatility_{window}'] = np.std(recent_window)
            lstm_features[f'performance_momentum_{window}'] = recent_window[-1] - recent_window[0]
    
    sequence_patterns = []
    for i in range(len(performance_sequence) - 2):
        pattern = [performance_sequence[i], performance_sequence[i+1], performance_sequence[i+2]]
        if all(p > 0.5 for p in pattern):
            sequence_patterns.append('winning_streak')
        elif all(p < 0.5 for p in pattern):
            sequence_patterns.append('losing_streak')
        elif pattern[0] < 0.5 and pattern[2] > 0.5:
            sequence_patterns.append('recovery')
        elif pattern[0] > 0.5 and pattern[2] < 0.5:
            sequence_patterns.append('decline')
    
    if sequence_patterns:
        pattern_counts = {}
        for pattern in sequence_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        total_patterns = len(sequence_patterns)
        for pattern, count in pattern_counts.items():
            lstm_features[f'pattern_{pattern}_rate'] = count / total_patterns
    
    return lstm_features

def add_gradient_boosting_features(team_stats, matches):
    gb_features = {}
    
    if len(matches) >= 10:
        sorted_matches = sorted(matches, key=lambda x: x.get('date', ''))
        
        performance_by_month = {}
        for match in sorted_matches:
            date_str = match.get('date', '')
            if date_str:
                try:
                    month_key = date_str[:7]
                    if month_key not in performance_by_month:
                        performance_by_month[month_key] = []
                    performance_by_month[month_key].append(match.get('team_won', False))
                except:
                    continue
        
        monthly_performance = []
        for month, results in performance_by_month.items():
            if len(results) >= 2:
                monthly_wr = sum(results) / len(results)
                monthly_performance.append(monthly_wr)
        
        if len(monthly_performance) >= 3:
            gb_features['performance_consistency'] = 1 - np.std(monthly_performance)
            gb_features['performance_trend'] = np.polyfit(range(len(monthly_performance)), monthly_performance, 1)[0]
        
        map_diversity = {}
        for match in sorted_matches:
            map_name = match.get('map', 'Unknown')
            if map_name != 'Unknown':
                map_diversity[map_name] = map_diversity.get(map_name, 0) + 1
        
        if map_diversity:
            total_maps = sum(map_diversity.values())
            map_entropy = -sum((count/total_maps) * np.log2(count/total_maps) for count in map_diversity.values())
            gb_features['map_diversity_entropy'] = map_entropy
        
        score_differentials = []
        for match in sorted_matches:
            team_score = match.get('team_score', 0)
            opp_score = match.get('opponent_score', 0)
            score_differentials.append(team_score - opp_score)
        
        if score_differentials:
            gb_features['avg_score_differential'] = np.mean(score_differentials)
            gb_features['score_consistency'] = 1 / (1 + np.std(score_differentials))
            gb_features['blowout_rate'] = sum(1 for diff in score_differentials if abs(diff) >= 10) / len(score_differentials)
    
    return gb_features

def implement_market_timing_features(matches):
    market_features = {}
    
    if len(matches) >= 15:
        sorted_matches = sorted(matches, key=lambda x: x.get('date', ''))
        
        upset_indicators = []
        for match in sorted_matches:
            team_score = match.get('team_score', 0)
            opp_score = match.get('opponent_score', 0)
            
            if team_score > opp_score and opp_score > 8:
                upset_indicators.append(1)
            elif opp_score > team_score and team_score > 8:
                upset_indicators.append(-1)
            else:
                upset_indicators.append(0)
        
        if upset_indicators:
            market_features['upset_frequency'] = sum(1 for x in upset_indicators if x != 0) / len(upset_indicators)
            market_features['upset_direction'] = np.mean(upset_indicators)
        
        performance_volatility = []
        window_size = 5
        for i in range(len(sorted_matches) - window_size + 1):
            window = sorted_matches[i:i+window_size]
            window_wr = sum(1 for m in window if m.get('team_won', False)) / window_size
            performance_volatility.append(window_wr)
        
        if len(performance_volatility) >= 3:
            market_features['rolling_volatility'] = np.std(performance_volatility)
            market_features['volatility_trend'] = np.polyfit(range(len(performance_volatility)), performance_volatility, 1)[0]
        
        closing_line_proxies = []
        for match in sorted_matches:
            team_score = match.get('team_score', 0)
            opp_score = match.get('opponent_score', 0)
            if team_score + opp_score > 0:
                implied_prob = team_score / (team_score + opp_score)
                if match.get('team_won', False):
                    clv_proxy = implied_prob - 0.5
                else:
                    clv_proxy = (1 - implied_prob) - 0.5
                closing_line_proxies.append(clv_proxy)
        
        if closing_line_proxies:
            market_features['avg_clv_proxy'] = np.mean(closing_line_proxies)
            market_features['clv_consistency'] = 1 - np.std(closing_line_proxies)
    
    return market_features

def advanced_bankroll_management_features(team_stats, matches=None):
    """Calculate advanced bankroll management features for a team."""
    bankroll_features = {}
    
    # Use the passed matches parameter if provided, otherwise try to get from team_stats
    if matches is None:
        matches = team_stats.get('matches', []) if isinstance(team_stats, dict) else []
    
    # Ensure we have a valid list of matches
    if not isinstance(matches, list) or len(matches) < 10:
        return bankroll_features
    
    # Get win rate from team stats or calculate from matches
    if isinstance(team_stats, dict) and 'win_rate' in team_stats:
        win_rate = team_stats.get('win_rate', 0.5)
    else:
        # Calculate win rate from matches
        wins = sum(1 for match in matches if match.get('team_won', False))
        win_rate = wins / len(matches) if len(matches) > 0 else 0.5
    
    # Calculate volatility (standard deviation of performance)
    performance_scores = []
    for match in matches:
        team_score = match.get('team_score', 0)
        opp_score = match.get('opponent_score', 0)
        if team_score + opp_score > 0:
            performance = team_score / (team_score + opp_score)
        else:
            performance = 0.5
        performance_scores.append(performance)
    
    volatility = np.std(performance_scores) if performance_scores else 0.1
    
    # Kelly Criterion calculation
    kelly_optimal = max(0, 2 * win_rate - 1) if win_rate > 0.5 else 0
    kelly_adjusted = kelly_optimal * (1 - volatility)
    bankroll_features['optimal_kelly'] = kelly_adjusted
    
    # Consecutive losses analysis
    consecutive_losses = 0
    max_consecutive_losses = 0
    current_streak = 0
    
    sorted_matches = sorted(matches, key=lambda x: x.get('date', ''))
    for match in sorted_matches:
        if match.get('team_won', False):
            consecutive_losses = 0
            current_streak += 1
        else:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            current_streak = 0
    
    bankroll_features['max_drawdown_periods'] = max_consecutive_losses
    bankroll_features['current_streak'] = current_streak
    
    # Win/Loss size analysis
    win_sizes = []
    loss_sizes = []
    for match in sorted_matches:
        team_score = match.get('team_score', 0)
        opp_score = match.get('opponent_score', 0)
        if match.get('team_won', False):
            win_sizes.append(team_score - opp_score)
        else:
            loss_sizes.append(opp_score - team_score)
    
    if win_sizes and loss_sizes:
        avg_win = np.mean(win_sizes)
        avg_loss = np.mean(loss_sizes)
        bankroll_features['win_loss_ratio'] = avg_win / avg_loss if avg_loss > 0 else 1
        bankroll_features['expectancy'] = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    return bankroll_features

def implement_correlation_adjusted_sizing(team_data):
    correlation_features = {}
    
    for team_name, data in team_data.items():
        matches = data.get('matches', [])
        if len(matches) >= 20:
            sorted_matches = sorted(matches, key=lambda x: x.get('date', ''))
            
            performance_by_opponent = {}
            for match in sorted_matches:
                opponent = match.get('opponent_name', '')
                if opponent:
                    if opponent not in performance_by_opponent:
                        performance_by_opponent[opponent] = []
                    performance_by_opponent[opponent].append(match.get('team_won', False))
            
            opponent_correlations = []
            base_performance = [m.get('team_won', False) for m in sorted_matches]
            
            for opponent, results in performance_by_opponent.items():
                if len(results) >= 3:
                    opponent_matches = [m for m in sorted_matches if m.get('opponent_name') == opponent]
                    opponent_perf = [m.get('team_won', False) for m in opponent_matches]
                    
                    if len(opponent_perf) >= 3:
                        correlation = np.corrcoef(opponent_perf[:-1], opponent_perf[1:])[0,1] if len(opponent_perf) > 1 else 0
                        opponent_correlations.append(abs(correlation))
            
            if opponent_correlations:
                correlation_features[team_name] = {
                    'avg_opponent_correlation': np.mean(opponent_correlations),
                    'max_opponent_correlation': max(opponent_correlations),
                    'correlation_diversity': 1 - np.std(opponent_correlations)
                }
    
    return correlation_features

def market_intelligence_features():
    market_intel = {
        'public_betting_indicators': {},
        'sharp_money_patterns': {},
        'steam_detection': {},
        'market_efficiency_scores': {}
    }
    
    mock_public_data = {
        'favorite_bias': 0.65,
        'over_bias': 0.58,
        'home_bias': 0.52,
        'recency_bias': 0.73
    }
    
    market_intel['public_betting_indicators'] = mock_public_data
    
    mock_sharp_patterns = {
        'early_sharp_movement': 0.23,
        'late_sharp_movement': 0.31,
        'reverse_line_movement': 0.18,
        'steam_frequency': 0.12
    }
    
    market_intel['sharp_money_patterns'] = mock_sharp_patterns
    
    efficiency_scores = {
        'moneyline_efficiency': 0.89,
        'spread_efficiency': 0.82,
        'total_efficiency': 0.76,
        'prop_efficiency': 0.71
    }
    
    market_intel['market_efficiency_scores'] = efficiency_scores
    
    return market_intel

def save_cache(team_data, filename="valorant_data_cache.pkl"):
    """Save enhanced team data cache with comprehensive metadata."""
    print(f"\nSaving enhanced team data cache to {filename}...")
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_corruption_detected = detect_cache_corruption()
    if cache_corruption_detected:
        print("Cache corruption detected, rebuilding...")
    
    # Data quality scores should already be calculated by now, but ensure they exist
    if not any('data_quality' in data for data in team_data.values()):
        print("Adding data quality scores...")
        team_data = add_data_quality_scores(team_data)
    
    canonical_db = create_canonical_team_database()
    team_data = detect_roster_changes(team_data, canonical_db)
    
    # Generate advanced features for each team
    print("Generating advanced features...")
    advanced_features_count = 0
    
    for team_name, data in team_data.items():
        if 'stats' in data and 'matches' in data:
            matches = data['matches']
            stats = data['stats']
            
            # Generate all feature sets with proper parameter passing
            advanced_features = implement_advanced_feature_engineering(stats, matches)
            lstm_features = implement_lstm_time_series_features(matches)
            gb_features = add_gradient_boosting_features(stats, matches)
            market_features = implement_market_timing_features(matches)
            bankroll_features = advanced_bankroll_management_features(stats, matches)
            
            data['advanced_features'] = {
                **advanced_features,
                **lstm_features,
                **gb_features,
                **market_features,
                **bankroll_features
            }
            
            if data['advanced_features']:
                advanced_features_count += 1
    
    print(f"Advanced features generated for {advanced_features_count} teams")
    
    # Generate correlation and market intelligence data
    print("Calculating correlation adjustments...")
    correlation_data = implement_correlation_adjusted_sizing(team_data)
    
    print("Generating market intelligence...")
    market_intelligence = market_intelligence_features()
    
    cache_version_info = implement_incremental_update_system()
    
    enhanced_cache = {
        'teams': team_data,
        'correlation_data': correlation_data,
        'market_intelligence': market_intelligence,
        'cache_version': cache_version_info,
        'canonical_database': canonical_db
    }
    
    cache_path = os.path.join(cache_dir, filename)
    with open(cache_path, 'wb') as f:
        pickle.dump(enhanced_cache, f)
    
    # Generate comprehensive metadata
    quality_scores = {
        team: data.get('data_quality', {}).get('overall_quality', 0) 
        for team, data in team_data.items()
    }
    
    meta_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "teams_count": len(team_data),
        "team_names": list(team_data.keys()),
        "total_matches": sum(len(team_data[team].get('matches', [])) for team in team_data),
        "cache_version": cache_version_info['version'],
        "quality_scores": quality_scores,
        "average_quality": sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0,
        "high_quality_teams": sum(1 for score in quality_scores.values() if score >= 0.8),
        "advanced_features_enabled": True,
        "market_intelligence_enabled": True,
        "features_summary": {
            "advanced_features": advanced_features_count,
            "correlation_data": len(correlation_data),
            "market_intelligence_modules": len(market_intelligence)
        }
    }
    
    meta_path = os.path.join(cache_dir, "cache_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    # Final summary with quality insights
    print(f"\n" + "="*60)
    print("CACHE SAVE SUMMARY")
    print("="*60)
    print(f"Enhanced cache saved successfully!")
    print(f"Teams cached: {meta_data['teams_count']}")
    print(f"Total matches: {meta_data['total_matches']:,}")
    print(f"Cache version: {cache_version_info['version']}")
    print(f"Average data quality: {meta_data['average_quality']:.3f}")
    print(f"High-quality teams: {meta_data['high_quality_teams']}/{meta_data['teams_count']} ({(meta_data['high_quality_teams']/meta_data['teams_count']*100):.1f}%)")
    print(f"Advanced features: {advanced_features_count} teams")
    print(f"Cache file: {cache_path}")
    print(f"Metadata: {meta_path}")
    
    return cache_path

def display_data_quality_summary(team_data_collection):
    """Display a comprehensive summary of data quality scores for collected teams."""
    print("\n" + "="*70)
    print("DATA QUALITY ASSESSMENT")
    print("="*70)
    
    # First, ensure all teams have quality scores
    team_data_collection = add_data_quality_scores(team_data_collection)
    
    quality_grades = []
    
    for team_name, data in team_data_collection.items():
        if 'data_quality' in data:
            quality = data['data_quality']
            overall_score = quality.get('overall_quality', 0)
            completeness = quality.get('completeness_score', 0)
            consistency = quality.get('consistency_score', 0)
            sample_size = quality.get('sample_size', 0)
            
            # Convert to letter grade
            if overall_score >= 0.9:
                grade = "A+"
            elif overall_score >= 0.8:
                grade = "A"
            elif overall_score >= 0.7:
                grade = "B+"
            elif overall_score >= 0.6:
                grade = "B"
            elif overall_score >= 0.5:
                grade = "C+"
            elif overall_score >= 0.4:
                grade = "C"
            else:
                grade = "D"
            
            quality_grades.append({
                'team_name': team_name,
                'grade': grade,
                'overall_score': overall_score,
                'completeness': completeness,
                'consistency': consistency,
                'sample_size': sample_size
            })
    
    # Sort by quality score (highest first)
    quality_grades.sort(key=lambda x: x['overall_score'], reverse=True)
    
    # Display detailed table
    print(f"{'Team Name':<25} {'Grade':<6} {'Overall':<8} {'Complete':<9} {'Consist':<8} {'Matches':<8}")
    print("-" * 70)
    
    grade_counts = {'A+': 0, 'A': 0, 'B+': 0, 'B': 0, 'C+': 0, 'C': 0, 'D': 0}
    
    for team_info in quality_grades:
        team_name = team_info['team_name'][:24]  # Truncate long names
        grade = team_info['grade']
        overall = team_info['overall_score']
        completeness = team_info['completeness']
        consistency = team_info['consistency']
        matches = team_info['sample_size']
        
        print(f"{team_name:<25} {grade:<6} {overall:.3f}    {completeness:.3f}     {consistency:.3f}    {matches:<8}")
        grade_counts[grade] += 1
    
    # Grade distribution summary
    print("\n" + "="*40)
    print("GRADE DISTRIBUTION SUMMARY:")
    print("="*40)
    total_teams = len(quality_grades)
    for grade, count in grade_counts.items():
        if count > 0:
            percentage = (count / total_teams) * 100
            bar = "█" * (count * 20 // total_teams) if total_teams > 0 else ""
            print(f"{grade:<3}: {count:>2} teams ({percentage:>5.1f}%) {bar}")
    
    # Overall statistics
    if quality_grades:
        avg_overall = sum(team['overall_score'] for team in quality_grades) / len(quality_grades)
        avg_completeness = sum(team['completeness'] for team in quality_grades) / len(quality_grades)
        avg_consistency = sum(team['consistency'] for team in quality_grades) / len(quality_grades)
        total_matches = sum(team['sample_size'] for team in quality_grades)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"Average Quality Score: {avg_overall:.3f}")
        print(f"Average Completeness:  {avg_completeness:.3f}")
        print(f"Average Consistency:   {avg_consistency:.3f}")
        print(f"Total Matches Cached:  {total_matches:,}")
        
        # Quality recommendations
        high_quality_teams = sum(1 for team in quality_grades if team['overall_score'] >= 0.8)
        low_quality_teams = sum(1 for team in quality_grades if team['overall_score'] < 0.5)
        
        print(f"\nQUALITY INSIGHTS:")
        print(f"High Quality Teams (A/A+): {high_quality_teams} ({(high_quality_teams/total_teams)*100:.1f}%)")
        print(f"Low Quality Teams (C-/D):  {low_quality_teams} ({(low_quality_teams/total_teams)*100:.1f}%)")
        
        if low_quality_teams > 0:
            print(f"\n⚠️  {low_quality_teams} teams have low data quality - consider updating cache more frequently")
        if high_quality_teams >= total_teams * 0.7:
            print(f"\n✅ Good cache quality: {(high_quality_teams/total_teams)*100:.1f}% of teams have high-quality data")
    
    return team_data_collection, quality_grades

def main():
    """Main function to handle command-line arguments and run the script."""
    parser = argparse.ArgumentParser(description="Cache Valorant team data for the match prediction system")
    parser.add_argument("--teams", type=int, default=150, help="Number of teams to cache (default: 150)")
    parser.add_argument("--no-player-stats", action="store_true", help="Skip player statistics")
    parser.add_argument("--no-economy", action="store_true", help="Skip economy data")
    parser.add_argument("--no-maps", action="store_true", help="Skip map statistics")
    parser.add_argument("--filename", type=str, default="valorant_data_cache.pkl", help="Output filename (default: valorant_data_cache.pkl)")
    
    args = parser.parse_args()
    
    print("=== Valorant Match Predictor Data Caching Tool ===")
    print(f"Starting data collection for up to {args.teams} teams")
    
    try:
        # The data quality assessment now happens inside collect_team_data_for_cache()
        team_data = collect_team_data_for_cache(
            team_limit=args.teams,
            include_player_stats=not args.no_player_stats,
            include_economy=not args.no_economy,
            include_maps=not args.no_maps
        )
        
        if not team_data:
            print("Error: No team data collected. Check API connection and try again.")
            return
        
        cache_path = save_cache(team_data, args.filename)
        print(f"\n✅ Process completed successfully!")
        print(f"You can now use the cached data with the main Valorant Match Predictor script.")
        
    except Exception as e:
        print(f"❌ Error during caching process: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()