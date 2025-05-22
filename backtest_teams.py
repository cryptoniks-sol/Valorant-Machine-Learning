import requests
import pickle
import os
import random
import time
from tqdm import tqdm
import json
from datetime import datetime

API_URL = "http://localhost:5000/api/v1"  # Adjust if your API URL is different


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


def fetch_all_teams(limit=300):
    """Fetch a larger pool of teams from the API."""
    print(f"Fetching up to {limit} teams from API...")
    teams_data = fetch_api_data("teams", {"limit": limit})
    
    if not teams_data or 'data' not in teams_data:
        print("Failed to fetch teams data")
        return []
        
    return teams_data['data']


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


def fetch_team_match_history(team_id):
    """Fetch match history for a specific team."""
    if not team_id:
        return None
    print(f"Fetching match history for team ID: {team_id}")
    return fetch_api_data(f"match-history/{team_id}")


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


def fetch_player_stats(player_name):
    """Fetch detailed player statistics using the API endpoint."""
    if not player_name:
        return None
    print(f"Fetching stats for player: {player_name}")
    player_data = fetch_api_data(f"player-stats/{player_name}")
    if player_data and player_data.get('status') == 'OK' and 'data' in player_data:
        return player_data['data']
    return None


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
        
        # Create a simplified map statistics structure
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
                match_info['opponent_name'] = opponent_team.get('name', '')
                match_info['opponent_score'] = int(opponent_team.get('score', 0))
                match_info['opponent_won'] = not team_won  # Opponent's result is opposite of our team
                match_info['opponent_country'] = opponent_team.get('country', '')
                match_info['opponent_tag'] = opponent_team.get('tag', '')
                match_info['opponent_id'] = opponent_team.get('id', '')  # Save opponent ID for future reference
                match_info['result'] = 'win' if team_won else 'loss'
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
    
    return team_stats


def create_backtest_cache(all_teams_limit=300, random_sample_size=100, 
                         include_player_stats=True, include_economy=False, 
                         include_maps=True, output_path="cache/backtest_data_cache.pkl"):
    """Create a cache file with random teams for backtesting."""
    if not os.path.exists("cache"):
        os.makedirs("cache")
        
    # Fetch a larger pool of teams
    all_teams = fetch_all_teams(limit=all_teams_limit)
    print(f"Fetched {len(all_teams)} teams")
    
    if len(all_teams) == 0:
        print("No teams found. Check API connection.")
        return
    
    # Make sure we have enough teams with sufficient data
    viable_teams = []
    for team in all_teams:
        team_id = team.get('id')
        if team_id:
            # Do a quick check to see if the team has matches
            match_history = fetch_team_match_history(team_id)
            if match_history and 'data' in match_history and len(match_history['data']) >= 10:
                viable_teams.append(team)
                print(f"Team {team.get('name')} has {len(match_history['data'])} matches - adding to viable list")
        
        # Break early if we have enough viable teams
        if len(viable_teams) >= random_sample_size * 1.5:  # 50% extra for margin
            break
    
    print(f"Found {len(viable_teams)} viable teams with sufficient match data")
    
    # Select random teams
    if len(viable_teams) > random_sample_size:
        selected_teams = random.sample(viable_teams, random_sample_size)
        print(f"Randomly selected {random_sample_size} teams from pool of {len(viable_teams)} viable teams")
    else:
        selected_teams = viable_teams
        print(f"Using all {len(viable_teams)} viable teams (requested {random_sample_size} but not enough available)")
    
    # Collect data for selected teams
    team_data_collection = {}
    for team in tqdm(selected_teams, desc="Collecting team data"):
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
            
        team_stats = calculate_team_stats(team_matches, team_player_stats, include_economy=include_economy)
        team_stats['team_tag'] = team_tag
        team_stats['team_name'] = team_name
        team_stats['team_id'] = team_id
        
        if include_maps:
            map_stats = fetch_team_map_statistics(team_id)
            if map_stats:
                team_stats['map_statistics'] = map_stats
                
        team_data_collection[team_name] = {
            'team_id': team_id,
            'team_tag': team_tag,
            'stats': team_stats,
            'matches': team_matches,
            'player_stats': team_player_stats,
            'ranking': team.get('ranking', None)
        }
        
        print(f"Successfully collected data for {team_name} with {len(team_matches)} matches")
    
    # Save the data
    print(f"\nSaving {len(team_data_collection)} teams to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(team_data_collection, f)
        
    # Also save metadata
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "teams_count": len(team_data_collection),
        "total_matches": sum(len(team_data['matches']) for team_data in team_data_collection.values()),
        "parameters": {
            "all_teams_limit": all_teams_limit,
            "random_sample_size": random_sample_size,
            "include_player_stats": include_player_stats,
            "include_economy": include_economy,
            "include_maps": include_maps
        }
    }
    
    with open("cache/backtest_cache_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Cache created successfully!")
    print(f"Teams: {metadata['teams_count']}")
    print(f"Total matches: {metadata['total_matches']}")
    return team_data_collection


def test_backtest_cache(cache_path="cache/backtest_data_cache.pkl"):
    """Test loading the backtest cache and print basic statistics."""
    if not os.path.exists(cache_path):
        print(f"Cache file not found at {cache_path}")
        return
        
    print(f"Loading cache from {cache_path}...")
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
        
    print(f"Cache loaded! Type: {type(cache_data)}")
    print(f"Teams: {len(cache_data)}")
    
    # Print structure of first team
    if cache_data:
        first_team = next(iter(cache_data.keys()))
        print(f"\nFirst team: {first_team}")
        team_data = cache_data[first_team]
        print(f"Keys: {team_data.keys()}")
        
        if 'matches' in team_data:
            matches = team_data['matches']
            print(f"Matches: {len(matches)}")
            if matches:
                print(f"First match keys: {matches[0].keys()}")
                
        if 'stats' in team_data:
            print(f"Stats keys: {team_data['stats'].keys()}")
            
    # Count total matches
    total_matches = sum(len(team_data.get('matches', [])) for team_data in cache_data.values())
    print(f"\nTotal matches in cache: {total_matches}")
    
    # Show team distribution
    print("\nTeams with most matches:")
    team_match_counts = [(team_name, len(team_data.get('matches', []))) 
                        for team_name, team_data in cache_data.items()]
    team_match_counts.sort(key=lambda x: x[1], reverse=True)
    for team_name, match_count in team_match_counts[:10]:
        win_count = sum(1 for m in cache_data[team_name].get('matches', []) if m.get('team_won', False))
        win_rate = win_count / match_count if match_count > 0 else 0
        print(f"  {team_name}: {match_count} matches, {win_rate:.2%} win rate")


if __name__ == "__main__":
    # Command line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description="Create a backtesting cache with random teams")
    parser.add_argument("--teams", type=int, default=300,
                      help="Number of teams to fetch from the API")
    parser.add_argument("--random", type=int, default=100,
                      help="Number of random teams to select for backtesting")
    parser.add_argument("--test", action="store_true",
                      help="Test an existing cache instead of creating a new one")
    parser.add_argument("--cache-path", type=str, default="cache/backtest_data_cache.pkl",
                      help="Path to the cache file")
    parser.add_argument("--no-players", action="store_true",
                      help="Skip player statistics")
    parser.add_argument("--include-economy", action="store_true",
                      help="Include economy data (slower)")
    parser.add_argument("--no-maps", action="store_true",
                      help="Skip map statistics")
    
    args = parser.parse_args()
    
    if args.test:
        test_backtest_cache(args.cache_path)
    else:
        create_backtest_cache(
            all_teams_limit=args.teams,
            random_sample_size=args.random,
            include_player_stats=not args.no_players,
            include_economy=args.include_economy,
            include_maps=not args.no_maps,
            output_path=args.cache_path
        )