import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import glob
import sys

class ValorantBettingAnalyzer:
    def __init__(self, bankroll=1000):
        self.predictions = []
        self.odds_data = []
        self.value_bets = []
        self.confidence_threshold = 0.60  # Can be adjusted
        self.value_threshold = 0.10  # 10% edge minimum
        self.bankroll = bankroll  # Starting bankroll in dollars
        self.kelly_fraction = 0.25  # Conservative Kelly criterion

    def load_prediction(self, prediction_file):
        """Load a prediction from a JSON file"""
        with open(prediction_file, 'r') as f:
            prediction_data = json.load(f)
            self.predictions.append(prediction_data)
        return prediction_data
    
    def load_predictions_from_directory(self, directory):
        """Load all prediction files from a directory"""
        prediction_files = glob.glob(os.path.join(directory, "*.json"))
        if not prediction_files:
            print(f"No prediction files found in {directory}")
            return []
        
        print(f"Found {len(prediction_files)} prediction files")
        for file_path in prediction_files:
            try:
                self.load_prediction(file_path)
                print(f"Loaded prediction from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return self.predictions
    
    def add_odds(self, match_id, team1, team2, ml_team1, ml_team2, 
                handicap_team1_plus, handicap_team1_minus, 
                handicap_team2_plus, handicap_team2_minus, 
                map_over, map_under):
        """Add betting odds for a match using decimal odds format"""
        odds_entry = {
            "match_id": match_id,
            "team1": team1,
            "team2": team2,
            "moneyline_team1": ml_team1,
            "moneyline_team2": ml_team2,
            "handicap_team1_plus_1.5": handicap_team1_plus,
            "handicap_team1_minus_1.5": handicap_team1_minus,
            "handicap_team2_plus_1.5": handicap_team2_plus,
            "handicap_team2_minus_1.5": handicap_team2_minus,
            "total_maps_over_2.5": map_over,
            "total_maps_under_2.5": map_under,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.odds_data.append(odds_entry)
        return odds_entry
    
    def collect_odds_interactively(self):
        """Collect odds for each match prediction interactively"""
        if not self.predictions:
            print("No predictions loaded. Please load predictions first.")
            return False
        
        print("\nCollecting odds for each match prediction...\n")
        print("Enter odds in decimal format (e.g., 2.5 means you get 2.5 times your stake if you win)")
        
        for idx, prediction in enumerate(self.predictions):
            match = prediction["prediction"]["match"]
            team1 = match.split(" vs ")[0]
            team2 = match.split(" vs ")[1]
            
            print(f"\nMatch {idx+1}/{len(self.predictions)}: {match}")
            
            try:
                # Moneyline odds
                ml_team1 = float(input(f"Enter moneyline odds for {team1} (decimal format, e.g. 2.40): "))
                ml_team2 = float(input(f"Enter moneyline odds for {team2} (decimal format, e.g. 1.60): "))
                
                # Handicap odds for Team 1
                handicap_team1_plus = float(input(f"Enter +1.5 map handicap odds for {team1}: "))
                handicap_team1_minus = float(input(f"Enter -1.5 map handicap odds for {team1}: "))
                
                # Handicap odds for Team 2
                handicap_team2_plus = float(input(f"Enter +1.5 map handicap odds for {team2}: "))
                handicap_team2_minus = float(input(f"Enter -1.5 map handicap odds for {team2}: "))
                
                # Total maps odds
                map_over = float(input("Enter over 2.5 maps odds: "))
                map_under = float(input("Enter under 2.5 maps odds: "))
                
                # Add odds
                self.add_odds(
                    match_id=f"match_{idx}",
                    team1=team1,
                    team2=team2,
                    ml_team1=ml_team1,
                    ml_team2=ml_team2,
                    handicap_team1_plus=handicap_team1_plus,
                    handicap_team1_minus=handicap_team1_minus,
                    handicap_team2_plus=handicap_team2_plus,
                    handicap_team2_minus=handicap_team2_minus,
                    map_over=map_over,
                    map_under=map_under
                )
                print(f"Added odds for {match}")
                
            except ValueError as e:
                print(f"Error: Invalid input. Please enter odds as decimals (e.g. 2.40).")
                retry = input("Would you like to retry this match? (y/n): ")
                if retry.lower() == 'y':
                    idx -= 1  # Retry this match
                else:
                    print(f"Skipping {match}")
            
            except KeyboardInterrupt:
                print("\nOdds collection interrupted. Processing available data...")
                break
        
        return True
    
    def _decimal_to_implied_probability(self, decimal_odds):
        """Convert decimal odds to implied probability"""
        return 1 / decimal_odds
    
    def analyze_value_bets(self):
        """Find value betting opportunities"""
        self.value_bets = []
        
        for prediction in self.predictions:
            # Find corresponding odds
            match_odds = None
            for odds in self.odds_data:
                if (odds["team1"] == prediction["prediction"]["match"].split(" vs ")[0] and 
                    odds["team2"] == prediction["prediction"]["match"].split(" vs ")[1]):
                    match_odds = odds
                    break
            
            if not match_odds:
                continue
                
            # Get model predictions
            team1 = prediction["prediction"]["match"].split(" vs ")[0]
            team2 = prediction["prediction"]["match"].split(" vs ")[1]
            team1_win_prob = prediction["prediction"]["team1_win_probability"]
            team2_win_prob = prediction["prediction"]["team2_win_probability"]
            confidence = prediction["prediction"]["confidence"]
            
            # Calculate map distribution probabilities based on win probability
            # These coefficients can be adjusted based on historical data
            # team1_win_prob is the probability of team1 winning the match
            if team1_win_prob >= 0.5:
                main_winner = team1
                main_win_prob = team1_win_prob
                underdog = team2
                underdog_win_prob = team2_win_prob
            else:
                main_winner = team2
                main_win_prob = team2_win_prob
                underdog = team1
                underdog_win_prob = team1_win_prob
                
            confidence = max(team1_win_prob, team2_win_prob)
            
            # Calculate score distribution probabilities
            p_main_winner_2_0 = 0.5 * (confidence ** 1.5)
            p_main_winner_2_1 = confidence - p_main_winner_2_0
            p_underdog_2_1 = underdog_win_prob * 0.7
            p_underdog_2_0 = underdog_win_prob - p_underdog_2_1
            
            # Map these generalized probabilities back to team1 and team2
            if main_winner == team1:
                p_team1_2_0 = p_main_winner_2_0
                p_team1_2_1 = p_main_winner_2_1
                p_team2_2_1 = p_underdog_2_1
                p_team2_2_0 = p_underdog_2_0
            else:
                p_team2_2_0 = p_main_winner_2_0
                p_team2_2_1 = p_main_winner_2_1
                p_team1_2_1 = p_underdog_2_1
                p_team1_2_0 = p_underdog_2_0
            
            # Calculate odds for total maps
            p_over_2_5 = p_team1_2_1 + p_team2_2_1  # Matches that go to 3 maps
            p_under_2_5 = p_team1_2_0 + p_team2_2_0  # Matches that end in 2 maps
            
            # Calculate odds for handicaps
            p_team1_plus_1_5 = p_team1_2_0 + p_team1_2_1 + p_team2_2_1  # Team 1 wins at least 1 map
            p_team1_minus_1_5 = p_team1_2_0  # Team 1 wins 2-0
            p_team2_plus_1_5 = p_team2_2_0 + p_team2_2_1 + p_team1_2_1  # Team 2 wins at least 1 map
            p_team2_minus_1_5 = p_team2_2_0  # Team 2 wins 2-0
            
            # Convert bookmaker odds to probabilities (already in decimal format)
            ml_team1_decimal = match_odds["moneyline_team1"]
            ml_team2_decimal = match_odds["moneyline_team2"]
            handicap_team1_plus_decimal = match_odds["handicap_team1_plus_1.5"]
            handicap_team1_minus_decimal = match_odds["handicap_team1_minus_1.5"]
            handicap_team2_plus_decimal = match_odds["handicap_team2_plus_1.5"]
            handicap_team2_minus_decimal = match_odds["handicap_team2_minus_1.5"]
            map_over_decimal = match_odds["total_maps_over_2.5"]
            map_under_decimal = match_odds["total_maps_under_2.5"]
            
            prob_ml_team1 = self._decimal_to_implied_probability(ml_team1_decimal)
            prob_ml_team2 = self._decimal_to_implied_probability(ml_team2_decimal)
            prob_handicap_team1_plus = self._decimal_to_implied_probability(handicap_team1_plus_decimal)
            prob_handicap_team1_minus = self._decimal_to_implied_probability(handicap_team1_minus_decimal)
            prob_handicap_team2_plus = self._decimal_to_implied_probability(handicap_team2_plus_decimal)
            prob_handicap_team2_minus = self._decimal_to_implied_probability(handicap_team2_minus_decimal)
            prob_map_over = self._decimal_to_implied_probability(map_over_decimal)
            prob_map_under = self._decimal_to_implied_probability(map_under_decimal) 
            
            # Calculate value edges
            edge_ml_team1 = team1_win_prob - prob_ml_team1
            edge_ml_team2 = team2_win_prob - prob_ml_team2
            edge_handicap_team1_plus = p_team1_plus_1_5 - prob_handicap_team1_plus
            edge_handicap_team1_minus = p_team1_minus_1_5 - prob_handicap_team1_minus
            edge_handicap_team2_plus = p_team2_plus_1_5 - prob_handicap_team2_plus
            edge_handicap_team2_minus = p_team2_minus_1_5 - prob_handicap_team2_minus
            edge_map_over = p_over_2_5 - prob_map_over
            edge_map_under = p_under_2_5 - prob_map_under
            
            # Get team stats for analysis
            team1_stats = prediction["team_stats"][team1]
            team2_stats = prediction["team_stats"][team2]
            
            # Check head-to-head history
            h2h_history = {}
            if "head_to_head" in prediction["analysis"]:
                h2h_history = prediction["analysis"]["head_to_head"]
            
            # Calculate kelly criterion bet size
            def kelly_bet(edge, decimal_odds):
                if edge <= 0:
                    return 0
                p_win = edge + self._decimal_to_implied_probability(decimal_odds)
                q = 1 - p_win
                b = decimal_odds - 1  # Decimal odds minus 1
                kelly = (p_win * b - q) / b
                # Apply a fraction for safety
                return max(0, kelly * self.kelly_fraction)
            
            # Find value bets with confidence above threshold
            value_bets_for_match = []
            
            # Only consider bets where our model has sufficient confidence
            if confidence >= self.confidence_threshold:
                # Moneyline bets
                if edge_ml_team1 > self.value_threshold:
                    bet_size = kelly_bet(edge_ml_team1, ml_team1_decimal)
                    value_bets_for_match.append({
                        "match": prediction["prediction"]["match"],
                        "bet_type": "Moneyline",
                        "pick": team1,
                        "model_probability": team1_win_prob,
                        "implied_probability": prob_ml_team1,
                        "edge": edge_ml_team1,
                        "odds": match_odds["moneyline_team1"],
                        "confidence": confidence,
                        "kelly_bet_size": bet_size,
                        "recommended_bet": bet_size * self.bankroll,
                        "key_factors": prediction["analysis"]["key_factors"] if "key_factors" in prediction["analysis"] else []
                    })
                
                if edge_ml_team2 > self.value_threshold:
                    bet_size = kelly_bet(edge_ml_team2, ml_team2_decimal)
                    value_bets_for_match.append({
                        "match": prediction["prediction"]["match"],
                        "bet_type": "Moneyline",
                        "pick": team2,
                        "model_probability": team2_win_prob,
                        "implied_probability": prob_ml_team2,
                        "edge": edge_ml_team2,
                        "odds": match_odds["moneyline_team2"],
                        "confidence": confidence,
                        "kelly_bet_size": bet_size,
                        "recommended_bet": bet_size * self.bankroll,
                        "key_factors": prediction["analysis"]["key_factors"] if "key_factors" in prediction["analysis"] else []
                    })
                
                # Handicap bets for Team 1
                if edge_handicap_team1_plus > self.value_threshold:
                    bet_size = kelly_bet(edge_handicap_team1_plus, handicap_team1_plus_decimal)
                    value_bets_for_match.append({
                        "match": prediction["prediction"]["match"],
                        "bet_type": "Handicap +1.5",
                        "pick": team1,
                        "model_probability": p_team1_plus_1_5,
                        "implied_probability": prob_handicap_team1_plus,
                        "edge": edge_handicap_team1_plus,
                        "odds": match_odds["handicap_team1_plus_1.5"],
                        "confidence": confidence,
                        "kelly_bet_size": bet_size,
                        "recommended_bet": bet_size * self.bankroll,
                        "key_factors": prediction["analysis"]["key_factors"] if "key_factors" in prediction["analysis"] else []
                    })
                
                if edge_handicap_team1_minus > self.value_threshold:
                    bet_size = kelly_bet(edge_handicap_team1_minus, handicap_team1_minus_decimal)
                    value_bets_for_match.append({
                        "match": prediction["prediction"]["match"],
                        "bet_type": "Handicap -1.5",
                        "pick": team1,
                        "model_probability": p_team1_minus_1_5,
                        "implied_probability": prob_handicap_team1_minus,
                        "edge": edge_handicap_team1_minus,
                        "odds": match_odds["handicap_team1_minus_1.5"],
                        "confidence": confidence,
                        "kelly_bet_size": bet_size,
                        "recommended_bet": bet_size * self.bankroll,
                        "key_factors": prediction["analysis"]["key_factors"] if "key_factors" in prediction["analysis"] else []
                    })
                
                # Handicap bets for Team 2
                if edge_handicap_team2_plus > self.value_threshold:
                    bet_size = kelly_bet(edge_handicap_team2_plus, handicap_team2_plus_decimal)
                    value_bets_for_match.append({
                        "match": prediction["prediction"]["match"],
                        "bet_type": "Handicap +1.5",
                        "pick": team2,
                        "model_probability": p_team2_plus_1_5,
                        "implied_probability": prob_handicap_team2_plus,
                        "edge": edge_handicap_team2_plus,
                        "odds": match_odds["handicap_team2_plus_1.5"],
                        "confidence": confidence,
                        "kelly_bet_size": bet_size,
                        "recommended_bet": bet_size * self.bankroll,
                        "key_factors": prediction["analysis"]["key_factors"] if "key_factors" in prediction["analysis"] else []
                    })
                
                if edge_handicap_team2_minus > self.value_threshold:
                    bet_size = kelly_bet(edge_handicap_team2_minus, handicap_team2_minus_decimal)
                    value_bets_for_match.append({
                        "match": prediction["prediction"]["match"],
                        "bet_type": "Handicap -1.5",
                        "pick": team2,
                        "model_probability": p_team2_minus_1_5,
                        "implied_probability": prob_handicap_team2_minus,
                        "edge": edge_handicap_team2_minus,
                        "odds": match_odds["handicap_team2_minus_1.5"],
                        "confidence": confidence,
                        "kelly_bet_size": bet_size,
                        "recommended_bet": bet_size * self.bankroll,
                        "key_factors": prediction["analysis"]["key_factors"] if "key_factors" in prediction["analysis"] else []
                    })
                
                # Total maps bets
                if edge_map_over > self.value_threshold:
                    bet_size = kelly_bet(edge_map_over, map_over_decimal)
                    value_bets_for_match.append({
                        "match": prediction["prediction"]["match"],
                        "bet_type": "Total Maps Over 2.5",
                        "pick": "Over 2.5",
                        "model_probability": p_over_2_5,
                        "implied_probability": prob_map_over,
                        "edge": edge_map_over,
                        "odds": match_odds["total_maps_over_2.5"],
                        "confidence": confidence,
                        "kelly_bet_size": bet_size,
                        "recommended_bet": bet_size * self.bankroll,
                        "key_factors": prediction["analysis"]["key_factors"] if "key_factors" in prediction["analysis"] else []
                    })
                
                if edge_map_under > self.value_threshold:
                    bet_size = kelly_bet(edge_map_under, map_under_decimal)
                    value_bets_for_match.append({
                        "match": prediction["prediction"]["match"],
                        "bet_type": "Total Maps Under 2.5",
                        "pick": "Under 2.5",
                        "model_probability": p_under_2_5,
                        "implied_probability": prob_map_under,
                        "edge": edge_map_under,
                        "odds": match_odds["total_maps_under_2.5"],
                        "confidence": confidence,
                        "kelly_bet_size": bet_size,
                        "recommended_bet": bet_size * self.bankroll,
                        "key_factors": prediction["analysis"]["key_factors"] if "key_factors" in prediction["analysis"] else []
                    })
            
            # Add team performance stats for analysis
            for bet in value_bets_for_match:
                if bet["pick"] == team1:
                    bet["team_stats"] = {
                        "win_rate": team1_stats["win_rate"],
                        "recent_form": team1_stats["recent_form"],
                        "avg_player_rating": team1_stats.get("avg_player_rating", 0),
                        "star_player": team1_stats.get("player_stats", {}).get("star_player_name", "")
                    }
                elif bet["pick"] == team2:
                    bet["team_stats"] = {
                        "win_rate": team2_stats["win_rate"],
                        "recent_form": team2_stats["recent_form"],
                        "avg_player_rating": team2_stats.get("avg_player_rating", 0),
                        "star_player": team2_stats.get("player_stats", {}).get("star_player_name", "")
                    }
                
                # Add H2H data
                bet["h2h_data"] = h2h_history
            
            # Add to master list
            self.value_bets.extend(value_bets_for_match)
        
        # Sort by edge value (highest first)
        self.value_bets.sort(key=lambda x: x["edge"], reverse=True)
        return self.value_bets
    
    def display_value_bets(self):
        """Display value bets in a nice format"""
        if not self.value_bets:
            print("No value bets found.")
            return
        
        print(f"\n{'=' * 80}")
        print(f"{'VALORANT BETTING VALUE OPPORTUNITIES':^80}")
        print(f"{'=' * 80}")
        
        for i, bet in enumerate(self.value_bets):
            print(f"\nBET #{i+1}: {bet['match']} - {bet['bet_type']}")
            print(f"Pick: {bet['pick']} @ {bet['odds']:.2f} (Decimal)")
            print(f"Model probability: {bet['model_probability']:.2%}")
            print(f"Implied odds probability: {bet['implied_probability']:.2%}")
            print(f"Edge: {bet['edge']:.2%}")
            print(f"Confidence: {bet['confidence']:.2%}")
            print(f"Recommended bet: ${bet['recommended_bet']:.2f} ({bet['kelly_bet_size']:.2%} of bankroll)")
            
            # Print team stats
            if "team_stats" in bet:
                print(f"Team stats: Win rate {bet['team_stats']['win_rate']:.2%}, Recent form {bet['team_stats']['recent_form']:.2%}")
                if bet['team_stats'].get("avg_player_rating"):
                    print(f"Avg player rating: {bet['team_stats']['avg_player_rating']:.2f}, Star player: {bet['team_stats']['star_player']}")
            
            # Print H2H data if available
            if "h2h_data" in bet and bet["h2h_data"]:
                h2h = bet["h2h_data"]
                team1, team2 = bet['match'].split(" vs ")
                if team1 == bet['pick']:
                    print(f"H2H: {team1} vs {team2}: {h2h.get('team1_win_rate', 0):.2%} win rate over {h2h.get('matches_played', 0)} matches")
                else:
                    print(f"H2H: {team2} vs {team1}: {h2h.get('team2_win_rate', 0):.2%} win rate over {h2h.get('matches_played', 0)} matches")
            
            # Print key factors
            if "key_factors" in bet and bet["key_factors"]:
                print("Key factors:")
                for factor in bet["key_factors"]:
                    print(f"• {factor}")
            
            print(f"{'-' * 80}")
    
    def export_value_bets(self, filename="valorant_value_bets.json"):
        """Export value bets to a JSON file"""
        if not self.value_bets:
            print("No value bets to export.")
            return
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "bankroll": self.bankroll,
                "value_bets": self.value_bets
            }, f, indent=2)
        
        print(f"Value bets exported to {filename}")
    
    def visualize_value_bets(self):
        """Create visualizations of value betting opportunities"""
        if not self.value_bets:
            print("No value bets to visualize.")
            return
        
        # Create a DataFrame from value bets
        df = pd.DataFrame(self.value_bets)
        
        # Plot 1: Edge vs Bet Type
        plt.figure(figsize=(12, 8))
        sns.barplot(x='bet_type', y='edge', data=df)
        plt.title('Edge by Bet Type')
        plt.xlabel('Bet Type')
        plt.ylabel('Edge (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('edge_by_bet_type.png')
        
        # Plot 2: Recommended Bet Size by Edge
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='edge', y='recommended_bet', hue='bet_type', size='confidence', data=df)
        plt.title('Recommended Bet Size by Edge')
        plt.xlabel('Edge (%)')
        plt.ylabel('Recommended Bet ($)')
        plt.tight_layout()
        plt.savefig('bet_size_by_edge.png')
        
        # Plot 3: Model Probability vs Implied Probability
        plt.figure(figsize=(12, 8))
        plt.scatter(df['implied_probability'], df['model_probability'], c=df['edge'], cmap='viridis')
        plt.colorbar(label='Edge')
        plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line where model = implied
        plt.title('Model Probability vs Implied Probability')
        plt.xlabel('Implied Probability from Odds')
        plt.ylabel('Model Probability')
        plt.tight_layout()
        plt.savefig('model_vs_implied.png')
        
        print("Visualizations saved as PNG files.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Valorant Betting Analyzer')
    parser.add_argument('predictions_path', help='Path to prediction file or directory')
    parser.add_argument('--bankroll', type=float, default=1000, help='Starting bankroll amount (default: 1000)')
    parser.add_argument('--confidence', type=float, default=0.60, help='Confidence threshold (default: 0.60)')
    parser.add_argument('--edge', type=float, default=0.10, help='Minimum edge threshold (default: 0.10)')
    parser.add_argument('--kelly', type=float, default=0.25, help='Kelly criterion fraction (default: 0.25)')
    parser.add_argument('--output', type=str, default='valorant_value_bets.json', help='Output file name')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Create analyzer instance
    analyzer = ValorantBettingAnalyzer(bankroll=args.bankroll)
    analyzer.confidence_threshold = args.confidence
    analyzer.value_threshold = args.edge
    analyzer.kelly_fraction = args.kelly
    
    # Load predictions
    if os.path.isdir(args.predictions_path):
        analyzer.load_predictions_from_directory(args.predictions_path)
    elif os.path.isfile(args.predictions_path):
        analyzer.load_prediction(args.predictions_path)
    else:
        print(f"Error: {args.predictions_path} is not a valid file or directory")
        sys.exit(1)
    
    if not analyzer.predictions:
        print("No predictions loaded. Exiting.")
        sys.exit(1)
    
    # Collect odds interactively
    if not analyzer.collect_odds_interactively():
        print("Failed to collect odds. Exiting.")
        sys.exit(1)
    
    # Analyze value bets
    value_bets = analyzer.analyze_value_bets()
    
    # Display results
    analyzer.display_value_bets()
    
    # Export to JSON
    analyzer.export_value_bets(args.output)
    
    # Create visualizations (unless disabled)
    if not args.no_visualize:
        analyzer.visualize_value_bets()

if __name__ == "__main__":
    main()