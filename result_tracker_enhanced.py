import os
import json
import pandas as pd
import requests
import time
from datetime import datetime
import argparse

# API URL (same as in your main script)
API_URL = "http://localhost:5000/api/v1"

class ResultTracker:
    """
    Tracks the results of match predictions and updates historical data.
    Also fetches actual match data from the API for better analysis.
    """
    def __init__(self, results_dir='prediction_results', history_file='prediction_history.csv'):
        """
        Initialize the result tracker.
        
        Args:
            results_dir: Directory where prediction results are stored
            history_file: CSV file to store prediction history
        """
        self.results_dir = results_dir
        self.history_file = os.path.join(results_dir, history_file)
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize history dataframe if file doesn't exist
        if not os.path.exists(self.history_file):
            self._create_history_file()
    
    def _create_history_file(self):
        """Create a new empty history file with headers."""
        columns = [
            'match_id', 'match_date', 'prediction_date', 'team1', 'team2', 
            'predicted_winner', 'actual_winner', 'win_probability', 
            'prediction_correct', 'model_version', 'notes',
            # New match performance metrics
            'team1_score', 'team2_score', 'map', 'total_rounds',
            'team1_attack_rounds_won', 'team1_defense_rounds_won',
            'team2_attack_rounds_won', 'team2_defense_rounds_won'
        ]
        
        history_df = pd.DataFrame(columns=columns)
        history_df.to_csv(self.history_file, index=False)
        print(f"Created new prediction history file: {self.history_file}")
    
    def load_prediction(self, prediction_file):
        """
        Load a prediction from a JSON file.
        
        Args:
            prediction_file: Path to the prediction JSON file
            
        Returns:
            dict: Prediction data
        """
        with open(prediction_file, 'r') as f:
            prediction_data = json.load(f)
        return prediction_data
    
    def get_team_id(self, team_name, region=None):
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
        try:
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
                    region_id = self.get_team_id(team_name, r)
                    if region_id:
                        return region_id
        
        except Exception as e:
            print(f"Error searching for team ID: {e}")
        
        print(f"No team ID found for '{team_name}'")
        return None
    
    def fetch_team_match_history(self, team_id):
        """Fetch match history for a specific team."""
        if not team_id:
            return None
        
        print(f"Fetching match history for team ID: {team_id}")
        try:
            response = requests.get(f"{API_URL}/match-history/{team_id}")
            
            if response.status_code != 200:
                print(f"Error fetching match history for team {team_id}: {response.status_code}")
                return None
            
            match_history = response.json()
            
            # Be nice to the API
            time.sleep(0.5)
            
            return match_history
        except Exception as e:
            print(f"Error fetching match history: {e}")
            return None
    
    def find_recent_match(self, team1_id, team2_id, team1_name, team2_name):
        """
        Find the most recent match between two teams.
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            team1_name: Name of the first team
            team2_name: Name of the second team
            
        Returns:
            dict: Match data if found, None otherwise
        """
        if not team1_id or not team2_id:
            return None
        
        # Get match history for team1
        team1_history = self.fetch_team_match_history(team1_id)
        if not team1_history or 'data' not in team1_history:
            return None
        
        # Look for matches against team2
        recent_match = None
        
        for match in team1_history['data']:
            # Skip if no teams data
            if 'teams' not in match or len(match['teams']) < 2:
                continue
                
            team1 = match['teams'][0]
            team2 = match['teams'][1]
            
            # Check if one team is team1 and the other is team2
            if (team1.get('name', '').lower() == team1_name.lower() and 
                team2.get('name', '').lower() == team2_name.lower()) or \
               (team1.get('name', '').lower() == team2_name.lower() and 
                team2.get('name', '').lower() == team1_name.lower()):
                
                # Found a match, check if it's more recent than current one
                match_date = match.get('date', '')
                
                if not recent_match or match_date > recent_match.get('date', ''):
                    recent_match = match
        
        return recent_match
    
    def parse_match_data(self, match, team1_name, team2_name):
        """
        Parse detailed match data.
        
        Args:
            match: Raw match data from API
            team1_name: Name of the first team
            team2_name: Name of the second team
            
        Returns:
            dict: Parsed match data
        """
        if not match or 'teams' not in match or len(match['teams']) < 2:
            return {}
        
        # Extract basic match info
        match_info = {
            'match_id': match.get('id', ''),
            'date': match.get('date', ''),
            'event': match.get('event', '') if isinstance(match.get('event', ''), str) else match.get('event', {}).get('name', ''),
            'tournament': match.get('tournament', ''),
            'map': match.get('map', '')
        }
        
        # Extract teams and determine which is team1 and team2
        team_data = match['teams']
        team1_idx = -1
        team2_idx = -1
        
        for i, team in enumerate(team_data):
            team_name = team.get('name', '').lower()
            if team_name == team1_name.lower() or team1_name.lower() in team_name:
                team1_idx = i
            elif team_name == team2_name.lower() or team2_name.lower() in team_name:
                team2_idx = i
        
        # If we couldn't identify the teams, return empty data
        if team1_idx == -1 or team2_idx == -1:
            return {}
        
        # Extract team scores
        team1_score = int(team_data[team1_idx].get('score', 0))
        team2_score = int(team_data[team2_idx].get('score', 0))
        
        match_info['team1_score'] = team1_score
        match_info['team2_score'] = team2_score
        match_info['total_rounds'] = team1_score + team2_score
        
        # Extract round data if available
        match_info['team1_attack_rounds_won'] = 0
        match_info['team1_defense_rounds_won'] = 0
        match_info['team2_attack_rounds_won'] = 0
        match_info['team2_defense_rounds_won'] = 0
        
        if 'rounds' in match:
            for round_data in match['rounds']:
                round_winner = round_data.get('winner', '')
                round_side = round_data.get('side', '')
                
                # Check who won this round
                if round_winner.lower() == team1_name.lower():
                    if round_side.lower() == 'attack':
                        match_info['team1_attack_rounds_won'] += 1
                    elif round_side.lower() == 'defense':
                        match_info['team1_defense_rounds_won'] += 1
                elif round_winner.lower() == team2_name.lower():
                    if round_side.lower() == 'attack':
                        match_info['team2_attack_rounds_won'] += 1
                    elif round_side.lower() == 'defense':
                        match_info['team2_defense_rounds_won'] += 1
        
        # Extract player statistics if available
        if 'players' in match:
            team1_players = []
            team2_players = []
            
            for player in match['players']:
                player_team = player.get('team', '')
                
                if player_team.lower() == team1_name.lower():
                    team1_players.append(player)
                elif player_team.lower() == team2_name.lower():
                    team2_players.append(player)
            
            # Calculate team averages for player stats
            if team1_players:
                match_info['team1_avg_acs'] = sum(p.get('acs', 0) for p in team1_players) / len(team1_players)
                match_info['team1_avg_kills'] = sum(p.get('kills', 0) for p in team1_players) / len(team1_players)
                match_info['team1_avg_deaths'] = sum(p.get('deaths', 0) for p in team1_players) / len(team1_players)
                match_info['team1_avg_assists'] = sum(p.get('assists', 0) for p in team1_players) / len(team1_players)
            
            if team2_players:
                match_info['team2_avg_acs'] = sum(p.get('acs', 0) for p in team2_players) / len(team2_players)
                match_info['team2_avg_kills'] = sum(p.get('kills', 0) for p in team2_players) / len(team2_players)
                match_info['team2_avg_deaths'] = sum(p.get('deaths', 0) for p in team2_players) / len(team2_players)
                match_info['team2_avg_assists'] = sum(p.get('assists', 0) for p in team2_players) / len(team2_players)
        
        return match_info
    
    def record_result(self, prediction_file, actual_winner, match_date=None, notes=None, model_version="1.0", fetch_match_data=True):
        """
        Record the result of a match prediction.
        
        Args:
            prediction_file: Path to the prediction JSON file
            actual_winner: Name of the team that actually won
            match_date: Date of the match (defaults to today)
            notes: Any additional notes about the match
            model_version: Version of the model used for prediction
            fetch_match_data: Whether to fetch actual match data from the API
            
        Returns:
            bool: True if recorded successfully, False otherwise
        """
        # Load prediction data
        try:
            prediction_data = self.load_prediction(prediction_file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading prediction file: {e}")
            return False
        
        # Extract match details
        match = prediction_data.get('match', '')
        if not match or ' vs ' not in match:
            print("Invalid match format in prediction file")
            return False
        
        team1, team2 = match.split(' vs ')
        predicted_winner = prediction_data.get('predicted_winner', '')
        win_probability = prediction_data.get('win_probability', 0.0)
        prediction_date = prediction_data.get('prediction_time', '')
        
        # Generate match ID
        timestamp = datetime.now().strftime("%Y%m%d")
        match_id = f"{team1.replace(' ', '')}_vs_{team2.replace(' ', '')}_{timestamp}"
        
        # Set match date
        if not match_date:
            match_date = datetime.now().strftime("%Y-%m-%d")
        
        # Determine if prediction was correct
        prediction_correct = predicted_winner.lower() == actual_winner.lower()
        
        # Initialize match data
        match_data = {
            'team1_score': None,
            'team2_score': None,
            'map': None,
            'total_rounds': None,
            'team1_attack_rounds_won': None,
            'team1_defense_rounds_won': None,
            'team2_attack_rounds_won': None,
            'team2_defense_rounds_won': None,
            'team1_avg_acs': None,
            'team2_avg_acs': None,
            'team1_avg_kills': None,
            'team2_avg_kills': None,
            'team1_avg_deaths': None,
            'team2_avg_deaths': None,
            'team1_avg_assists': None,
            'team2_avg_assists': None,
        }
        
        # Fetch actual match data from API if requested
        if fetch_match_data:
            print(f"Fetching actual match data for {team1} vs {team2}...")
            
            # Get team IDs
            team1_id = self.get_team_id(team1)
            team2_id = self.get_team_id(team2)
            
            if team1_id and team2_id:
                # Find most recent match between the teams
                recent_match = self.find_recent_match(team1_id, team2_id, team1, team2)
                
                if recent_match:
                    # Parse match data
                    match_data.update(self.parse_match_data(recent_match, team1, team2))
                    print(f"Found match data: {team1} {match_data.get('team1_score', 'N/A')} - {match_data.get('team2_score', 'N/A')} {team2}")
                else:
                    print(f"No recent matches found between {team1} and {team2}")
            else:
                print(f"Could not find team IDs for {team1} and/or {team2}")
        
        # Load existing history
        history_df = pd.read_csv(self.history_file)
        
        # Create new record
        new_record = {
            'match_id': match_id,
            'match_date': match_date,
            'prediction_date': prediction_date,
            'team1': team1,
            'team2': team2,
            'predicted_winner': predicted_winner,
            'actual_winner': actual_winner,
            'win_probability': win_probability,
            'prediction_correct': prediction_correct,
            'model_version': model_version,
            'notes': notes or ''
        }
        
        # Add match data to record
        new_record.update(match_data)
        
        # Append new record
        history_df = pd.concat([history_df, pd.DataFrame([new_record])], ignore_index=True)
        
        # Save updated history
        history_df.to_csv(self.history_file, index=False)
        
        # Also save detailed result with actual outcome
        result_file = os.path.join(self.results_dir, f"{match_id}_result.json")
        
        # Update prediction data with result
        prediction_data['actual_winner'] = actual_winner
        prediction_data['prediction_correct'] = prediction_correct
        prediction_data['match_date'] = match_date
        prediction_data['notes'] = notes or ''
        prediction_data['model_version'] = model_version
        
        # Add match performance data
        prediction_data['match_performance'] = match_data
        
        with open(result_file, 'w') as f:
            json.dump(prediction_data, f, indent=4, default=str)
        
        print(f"Recorded result for {match}. Prediction was {'correct' if prediction_correct else 'incorrect'}.")
        print(f"Detailed result saved to {result_file}")
        
        return True
    
    def get_recent_results(self, n=10):
        """
        Get the most recent prediction results.
        
        Args:
            n: Number of recent results to return
            
        Returns:
            pd.DataFrame: DataFrame with recent results
        """
        history_df = pd.read_csv(self.history_file)
        return history_df.sort_values('match_date', ascending=False).head(n)
    
    def get_results_summary(self):
        """
        Get a summary of all prediction results.
        
        Returns:
            dict: Summary statistics
        """
        history_df = pd.read_csv(self.history_file)
        
        if len(history_df) == 0:
            return {"error": "No prediction results found"}
        
        total_predictions = len(history_df)
        correct_predictions = history_df['prediction_correct'].sum()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Group results by month
        history_df['month'] = pd.to_datetime(history_df['match_date']).dt.strftime('%Y-%m')
        monthly_results = history_df.groupby('month').agg(
            total=('match_id', 'count'),
            correct=('prediction_correct', 'sum')
        )
        monthly_results['accuracy'] = monthly_results['correct'] / monthly_results['total']
        
        # Calculate accuracy by confidence level
        history_df['confidence_level'] = pd.cut(
            history_df['win_probability'], 
            bins=[0, 0.55, 0.65, 0.75, 0.85, 1.0],
            labels=['Very Low (≤55%)', 'Low (56-65%)', 'Medium (66-75%)', 'High (76-85%)', 'Very High (>85%)']
        )
        confidence_results = history_df.groupby('confidence_level').agg(
            total=('match_id', 'count'),
            correct=('prediction_correct', 'sum')
        )
        confidence_results['accuracy'] = confidence_results['correct'] / confidence_results['total']
        
        # Calculate performance by map
        map_performance = None
        if 'map' in history_df.columns and history_df['map'].notna().any():
            map_performance = history_df.groupby('map').agg(
                total=('match_id', 'count'),
                correct=('prediction_correct', 'sum')
            )
            map_performance['accuracy'] = map_performance['correct'] / map_performance['total']
            map_performance = map_performance.to_dict()
        
        # Analyze score differentials
        score_analysis = None
        if 'team1_score' in history_df.columns and 'team2_score' in history_df.columns:
            # Calculate actual score differential
            history_df['score_diff'] = None
            
            # For each row, calculate score diff between winner and loser
            for idx, row in history_df.iterrows():
                if row['actual_winner'] == row['team1']:
                    history_df.at[idx, 'score_diff'] = row['team1_score'] - row['team2_score']
                else:
                    history_df.at[idx, 'score_diff'] = row['team2_score'] - row['team1_score']
            
            # Group by prediction correctness
            score_by_correctness = history_df.groupby('prediction_correct').agg(
                avg_score_diff=('score_diff', 'mean'),
                min_score_diff=('score_diff', 'min'),
                max_score_diff=('score_diff', 'max')
            )
            score_analysis = score_by_correctness.to_dict()
        
        summary = {
            'total_predictions': total_predictions,
            'correct_predictions': int(correct_predictions),
            'overall_accuracy': accuracy,
            'monthly_accuracy': monthly_results.to_dict(),
            'confidence_level_accuracy': confidence_results.to_dict(),
            'map_performance': map_performance,
            'score_analysis': score_analysis
        }
        
        return summary

def main():
    """Command line interface for the result tracker."""
    parser = argparse.ArgumentParser(description='Track and record match prediction results')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Record result command
    record_parser = subparsers.add_parser('record', help='Record a match result')
    record_parser.add_argument('prediction_file', help='Path to the prediction JSON file')
    record_parser.add_argument('actual_winner', help='Name of the team that actually won')
    record_parser.add_argument('--match_date', help='Date of the match (YYYY-MM-DD)')
    record_parser.add_argument('--notes', help='Additional notes about the match')
    record_parser.add_argument('--model_version', default="1.0", help='Version of the model used')
    record_parser.add_argument('--no_fetch', action='store_true', help='Skip fetching match data from API')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show prediction results summary')
    
    # Recent results command
    recent_parser = subparsers.add_parser('recent', help='Show recent prediction results')
    recent_parser.add_argument('--count', type=int, default=10, help='Number of recent results to show')
    
    args = parser.parse_args()
    
    tracker = ResultTracker()
    
    if args.command == 'record':
        tracker.record_result(
            args.prediction_file,
            args.actual_winner,
            args.match_date,
            args.notes,
            args.model_version,
            not args.no_fetch
        )
    
    elif args.command == 'summary':
        summary = tracker.get_results_summary()
        print("\n===== Prediction Results Summary =====")
        print(f"Total Predictions: {summary['total_predictions']}")
        print(f"Correct Predictions: {summary['correct_predictions']}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.2%}")
        
        print("\nMonthly Accuracy:")
        for month, data in summary['monthly_accuracy']['accuracy'].items():
            total = summary['monthly_accuracy']['total'][month]
            correct = summary['monthly_accuracy']['correct'][month]
            print(f"  {month}: {data:.2%} ({correct}/{total})")
        
        print("\nAccuracy by Confidence Level:")
        for level, data in summary['confidence_level_accuracy']['accuracy'].items():
            if pd.isna(level):
                continue
            total = summary['confidence_level_accuracy']['total'][level]
            correct = summary['confidence_level_accuracy']['correct'][level]
            print(f"  {level}: {data:.2%} ({correct}/{total})")
        
        if summary.get('map_performance'):
            print("\nAccuracy by Map:")
            for map_name, data in summary['map_performance']['accuracy'].items():
                total = summary['map_performance']['total'][map_name]
                correct = summary['map_performance']['correct'][map_name]
                print(f"  {map_name}: {data:.2%} ({correct}/{total})")
        
        if summary.get('score_analysis'):
            print("\nScore Analysis:")
            print("  Correct Predictions:")
            correct_stats = summary['score_analysis'].get('avg_score_diff', {}).get(True)
            if correct_stats:
                print(f"    Average winner score differential: {correct_stats:.2f}")
            
            print("  Incorrect Predictions:")
            incorrect_stats = summary['score_analysis'].get('avg_score_diff', {}).get(False)
            if incorrect_stats:
                print(f"    Average winner score differential: {incorrect_stats:.2f}")
    
    elif args.command == 'recent':
        recent_results = tracker.get_recent_results(args.count)
        print(f"\n===== {args.count} Most Recent Prediction Results =====")
        display_cols = ['match_date', 'team1', 'team2', 'predicted_winner', 
                        'actual_winner', 'win_probability', 'prediction_correct']
        
        # Add score columns if they exist
        if 'team1_score' in recent_results.columns and 'team2_score' in recent_results.columns:
            display_cols.extend(['team1_score', 'team2_score'])
        
        print(recent_results[display_cols].to_string(index=False))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()