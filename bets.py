import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

class ValorantBettingAdvisor:
    def __init__(self, prediction_file_path):
        """Initialize the betting advisor with prediction data"""
        with open(prediction_file_path, 'r') as file:
            self.prediction_data = json.load(file)
        
        self.team1_name = self.prediction_data['team1_stats']['name'] if 'name' in self.prediction_data['team1_stats'] else self.prediction_data['match'].split(' vs ')[0]
        self.team2_name = self.prediction_data['team2_stats']['name'] if 'name' in self.prediction_data['team2_stats'] else self.prediction_data['match'].split(' vs ')[1]
        
        # Extract key features from the prediction data
        self.features = self._extract_features()
        
        # Initialize models
        self.moneyline_model = None
        self.handicap_model = None
        self.total_maps_model = None
        self.win_map_model = None
        
        # Training data would be loaded or collected over time
        # For demonstration, we'll simulate model predictions
        self._initialize_models()
    
    def _extract_features(self):
        """Extract key statistical features from the prediction data"""
        team1_stats = self.prediction_data['team1_stats']
        team2_stats = self.prediction_data['team2_stats']
        h2h_stats = self.prediction_data['h2h_stats']
        
        # First, correctly identify the predicted winner
        predicted_winner = self.prediction_data['predicted_winner']
        win_probability = self.prediction_data['win_probability']
        
        # Determine which team is predicted to win
        if predicted_winner == self.team1_name:
            team1_win_prob = win_probability
            team2_win_prob = 1 - win_probability
        else:
            team2_win_prob = win_probability
            team1_win_prob = 1 - win_probability
        
        features = {
            # Team 1 features
            'team1_win_rate': team1_stats['win_rate'],
            'team1_avg_score': team1_stats['avg_score'],
            'team1_avg_opponent_score': team1_stats['avg_opponent_score'],
            'team1_score_differential': team1_stats['score_differential'],
            'team1_recent_form': team1_stats['recent_form'],
            
            # Team 2 features
            'team2_win_rate': team2_stats['win_rate'],
            'team2_avg_score': team2_stats['avg_score'],
            'team2_avg_opponent_score': team2_stats['avg_opponent_score'],
            'team2_score_differential': team2_stats['score_differential'],
            'team2_recent_form': team2_stats['recent_form'],
            
            # Head-to-head features
            'team1_h2h_win_rate': h2h_stats['team1_h2h_win_rate'],
            'team2_h2h_win_rate': h2h_stats['team2_h2h_win_rate'],
            'total_h2h_matches': h2h_stats['total_h2h_matches'],
            
            # Comparative features
            'win_rate_diff': team1_stats['win_rate'] - team2_stats['win_rate'],
            'avg_score_diff': team1_stats['avg_score'] - team2_stats['avg_score'],
            'form_diff': team1_stats['recent_form'] - team2_stats['recent_form'],
            
            # Predicted win probability for each team - correctly assigned now
            'team1_win_probability': team1_win_prob,
            'team2_win_probability': team2_win_prob,
            
            # Original prediction data
            'predicted_winner': predicted_winner,
            'win_probability': win_probability
        }
        
        return features

    def _initialize_models(self):
        """
        Initialize and train predictive models
        In a real implementation, these would be trained on historical data
        """
        # For demonstration, we're using basic models
        # In practice, you would train these on historical match data and betting outcomes
        
        self.moneyline_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.handicap_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.total_maps_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.win_map_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # In a real implementation, you would:
        # 1. Load historical data
        # 2. Create feature sets
        # 3. Train the models using historical outcomes
        
        # For demonstration, we'll simulate trained models

    def _calculate_ev(self, win_probability, odds):
        """
        Calculate Expected Value of a bet
        EV = (probability_of_winning * profit) - (probability_of_losing * stake)
        Where profit = stake * (odds - 1)
        """
        if odds <= 1:
            return -1  # Invalid odds
        
        stake = 1  # Normalized to 1 unit
        profit = stake * (odds - 1)
        loss = stake
        
        ev = (win_probability * profit) - ((1 - win_probability) * loss)
        return ev

    def _calculate_kelly_bet(self, win_probability, odds, bankroll, kelly_fraction=0.25):
        """
        Calculate bet size using the Kelly Criterion
        Kelly = (bp - q) / b
        where:
        - b = odds - 1 (decimal odds minus 1)
        - p = probability of winning
        - q = probability of losing (1 - p)
        
        We apply a fraction to be more conservative
        """
        if odds <= 1 or win_probability <= 0:
            return 0
        
        b = odds - 1
        p = win_probability
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Apply Kelly fraction and bankroll
        if kelly <= 0:
            return 0
        else:
            return kelly * kelly_fraction * bankroll

    def _predict_win_probability(self, model, feature_importance=None):
        """
        Predict win probability using a trained model
        In a real implementation, this would use the actual model prediction
        """
        # For demonstration, we're using the prediction data's win probability
        # and adjusting based on the model and feature importance
        
        base_probability = self.prediction_data['win_probability']
        
        # In reality, you would:
        # features_vector = [list of feature values]
        # probability = model.predict_proba([features_vector])[0][1]
        
        # For demonstration, we'll use reasonable values based on the data
        return base_probability

    def get_betting_advice(self, odds_dict, bankroll):
        """
        Generate betting advice based on odds and prediction data
        
        Args:
            odds_dict: Dictionary containing odds for various bet types
            bankroll: Total bankroll for calculating bet sizes
            
        Returns:
            List of bet recommendations sorted by EV
        """
        recommendations = []
        
        # Get the correct win probabilities for each team
        predicted_winner = self.prediction_data['predicted_winner']
        win_probability = self.prediction_data['win_probability']
        
        # Set probabilities based on predicted winner
        if predicted_winner == self.team1_name:
            team1_ml_prob = win_probability
            team2_ml_prob = 1 - win_probability
        else:
            team2_ml_prob = win_probability
            team1_ml_prob = 1 - win_probability
        
        # Now proceed with the original code using the correct probabilities
        # Team 1 moneyline
        if 'team1_ml' in odds_dict:
            team1_ml_ev = self._calculate_ev(team1_ml_prob, odds_dict['team1_ml'])
            team1_ml_bet = self._calculate_kelly_bet(team1_ml_prob, odds_dict['team1_ml'], bankroll)
            
            if team1_ml_ev > 0:
                recommendations.append({
                    'bet_type': f"Moneyline: {self.team1_name} to win",
                    'odds': odds_dict['team1_ml'],
                    'win_probability': team1_ml_prob,
                    'expected_value': team1_ml_ev,
                    'bet_amount': team1_ml_bet,
                    'reasoning': self._generate_ml_reasoning(self.team1_name, team1_ml_prob)
                })
        
        # Team 2 moneyline
        if 'team2_ml' in odds_dict:
            team2_ml_ev = self._calculate_ev(team2_ml_prob, odds_dict['team2_ml'])
            team2_ml_bet = self._calculate_kelly_bet(team2_ml_prob, odds_dict['team2_ml'], bankroll)
            
            if team2_ml_ev > 0:
                recommendations.append({
                    'bet_type': f"Moneyline: {self.team2_name} to win",
                    'odds': odds_dict['team2_ml'],
                    'win_probability': team2_ml_prob,
                    'expected_value': team2_ml_ev,
                    'bet_amount': team2_ml_bet,
                    'reasoning': self._generate_ml_reasoning(self.team2_name, team2_ml_prob)
                })
        
        # Analyze Map Handicap Bets
        # Team 1 +1.5 maps
        if 'team1_plus1_5' in odds_dict:
            # Probability team1 wins at least 1 map (doesn't get 2-0'd)
            team1_plus1_5_prob = self._calculate_handicap_probability(self.team1_name, 1.5)
            team1_plus1_5_ev = self._calculate_ev(team1_plus1_5_prob, odds_dict['team1_plus1_5'])
            team1_plus1_5_bet = self._calculate_kelly_bet(team1_plus1_5_prob, odds_dict['team1_plus1_5'], bankroll)
            
            if team1_plus1_5_ev > 0:
                recommendations.append({
                    'bet_type': f"Map Handicap: {self.team1_name} +1.5",
                    'odds': odds_dict['team1_plus1_5'],
                    'win_probability': team1_plus1_5_prob,
                    'expected_value': team1_plus1_5_ev,
                    'bet_amount': team1_plus1_5_bet,
                    'reasoning': self._generate_handicap_reasoning(self.team1_name, "+1.5", team1_plus1_5_prob)
                })
        
        # Team 2 +1.5 maps
        if 'team2_plus1_5' in odds_dict:
            # Probability team2 wins at least 1 map (doesn't get 2-0'd)
            team2_plus1_5_prob = self._calculate_handicap_probability(self.team2_name, 1.5)
            team2_plus1_5_ev = self._calculate_ev(team2_plus1_5_prob, odds_dict['team2_plus1_5'])
            team2_plus1_5_bet = self._calculate_kelly_bet(team2_plus1_5_prob, odds_dict['team2_plus1_5'], bankroll)
            
            if team2_plus1_5_ev > 0:
                recommendations.append({
                    'bet_type': f"Map Handicap: {self.team2_name} +1.5",
                    'odds': odds_dict['team2_plus1_5'],
                    'win_probability': team2_plus1_5_prob,
                    'expected_value': team2_plus1_5_ev,
                    'bet_amount': team2_plus1_5_bet,
                    'reasoning': self._generate_handicap_reasoning(self.team2_name, "+1.5", team2_plus1_5_prob)
                })
        
        # Team 1 -1.5 maps (win 2-0)
        if 'team1_minus1_5' in odds_dict:
            team1_minus1_5_prob = self._calculate_handicap_probability(self.team1_name, -1.5)
            team1_minus1_5_ev = self._calculate_ev(team1_minus1_5_prob, odds_dict['team1_minus1_5'])
            team1_minus1_5_bet = self._calculate_kelly_bet(team1_minus1_5_prob, odds_dict['team1_minus1_5'], bankroll)
            
            if team1_minus1_5_ev > 0:
                recommendations.append({
                    'bet_type': f"Map Handicap: {self.team1_name} -1.5",
                    'odds': odds_dict['team1_minus1_5'],
                    'win_probability': team1_minus1_5_prob,
                    'expected_value': team1_minus1_5_ev,
                    'bet_amount': team1_minus1_5_bet,
                    'reasoning': self._generate_handicap_reasoning(self.team1_name, "-1.5", team1_minus1_5_prob)
                })
        
        # Team 2 -1.5 maps (win 2-0)
        if 'team2_minus1_5' in odds_dict:
            team2_minus1_5_prob = self._calculate_handicap_probability(self.team2_name, -1.5)
            team2_minus1_5_ev = self._calculate_ev(team2_minus1_5_prob, odds_dict['team2_minus1_5'])
            team2_minus1_5_bet = self._calculate_kelly_bet(team2_minus1_5_prob, odds_dict['team2_minus1_5'], bankroll)
            
            if team2_minus1_5_ev > 0:
                recommendations.append({
                    'bet_type': f"Map Handicap: {self.team2_name} -1.5",
                    'odds': odds_dict['team2_minus1_5'],
                    'win_probability': team2_minus1_5_prob,
                    'expected_value': team2_minus1_5_ev,
                    'bet_amount': team2_minus1_5_bet,
                    'reasoning': self._generate_handicap_reasoning(self.team2_name, "-1.5", team2_minus1_5_prob)
                })
        
        # Analyze Over/Under Total Maps
        if 'over_2_5_maps' in odds_dict:
            over_2_5_prob = self._calculate_over_under_probability(True)
            over_2_5_ev = self._calculate_ev(over_2_5_prob, odds_dict['over_2_5_maps'])
            over_2_5_bet = self._calculate_kelly_bet(over_2_5_prob, odds_dict['over_2_5_maps'], bankroll)
            
            if over_2_5_ev > 0:
                recommendations.append({
                    'bet_type': "Total Maps: Over 2.5",
                    'odds': odds_dict['over_2_5_maps'],
                    'win_probability': over_2_5_prob,
                    'expected_value': over_2_5_ev,
                    'bet_amount': over_2_5_bet,
                    'reasoning': self._generate_over_under_reasoning(True, over_2_5_prob)
                })
        
        if 'under_2_5_maps' in odds_dict:
            under_2_5_prob = self._calculate_over_under_probability(False)
            under_2_5_ev = self._calculate_ev(under_2_5_prob, odds_dict['under_2_5_maps'])
            under_2_5_bet = self._calculate_kelly_bet(under_2_5_prob, odds_dict['under_2_5_maps'], bankroll)
            
            if under_2_5_ev > 0:
                recommendations.append({
                    'bet_type': "Total Maps: Under 2.5",
                    'odds': odds_dict['under_2_5_maps'],
                    'win_probability': under_2_5_prob,
                    'expected_value': under_2_5_ev,
                    'bet_amount': under_2_5_bet,
                    'reasoning': self._generate_over_under_reasoning(False, under_2_5_prob)
                })
        
        # Analyze "Win at least 1 map" bets
        if 'team1_win_map' in odds_dict:
            team1_win_map_prob = self._calculate_win_map_probability(self.team1_name)
            team1_win_map_ev = self._calculate_ev(team1_win_map_prob, odds_dict['team1_win_map'])
            team1_win_map_bet = self._calculate_kelly_bet(team1_win_map_prob, odds_dict['team1_win_map'], bankroll)
            
            if team1_win_map_ev > 0:
                recommendations.append({
                    'bet_type': f"{self.team1_name} to win at least 1 map",
                    'odds': odds_dict['team1_win_map'],
                    'win_probability': team1_win_map_prob,
                    'expected_value': team1_win_map_ev,
                    'bet_amount': team1_win_map_bet,
                    'reasoning': self._generate_win_map_reasoning(self.team1_name, team1_win_map_prob)
                })
        
        if 'team2_win_map' in odds_dict:
            team2_win_map_prob = self._calculate_win_map_probability(self.team2_name)
            team2_win_map_ev = self._calculate_ev(team2_win_map_prob, odds_dict['team2_win_map'])
            team2_win_map_bet = self._calculate_kelly_bet(team2_win_map_prob, odds_dict['team2_win_map'], bankroll)
            
            if team2_win_map_ev > 0:
                recommendations.append({
                    'bet_type': f"{self.team2_name} to win at least 1 map",
                    'odds': odds_dict['team2_win_map'],
                    'win_probability': team2_win_map_prob,
                    'expected_value': team2_win_map_ev,
                    'bet_amount': team2_win_map_bet,
                    'reasoning': self._generate_win_map_reasoning(self.team2_name, team2_win_map_prob)
                })
        
        # Sort recommendations by expected value (highest to lowest)
        recommendations.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return recommendations

    def _calculate_handicap_probability(self, team_name, handicap):
        """
        Calculate probability for handicap bets
        """
        # Get the correct win probabilities for each team
        predicted_winner = self.prediction_data['predicted_winner']
        win_probability = self.prediction_data['win_probability']
        
        # Set probabilities based on predicted winner
        if predicted_winner == self.team1_name:
            team1_win_prob = win_probability
            team2_win_prob = 1 - win_probability
        else:
            team2_win_prob = win_probability
            team1_win_prob = 1 - win_probability
        
        # Base probabilities for different map outcomes
        # Based on model prediction and historical patterns
        # These would ideally come from trained models in a real implementation
        
        # With strong probability split, tighter match is more likely
        close_match_factor = 1 - abs(team1_win_prob - team2_win_prob) * 2  # Higher when teams are close
        
        # Probability of 2-0 vs 2-1 outcomes
        prob_2_0_if_team1_wins = 0.55 - (close_match_factor * 0.2)  # Less likely 2-0 when teams close
        prob_2_1_if_team1_wins = 1 - prob_2_0_if_team1_wins
        
        prob_0_2_if_team2_wins = 0.55 - (close_match_factor * 0.2)
        prob_1_2_if_team2_wins = 1 - prob_0_2_if_team2_wins
        
        # Calculate specific outcome probabilities
        prob_2_0 = team1_win_prob * prob_2_0_if_team1_wins
        prob_2_1 = team1_win_prob * prob_2_1_if_team1_wins
        prob_0_2 = team2_win_prob * prob_0_2_if_team2_wins
        prob_1_2 = team2_win_prob * prob_1_2_if_team2_wins
        
        # Determine probability based on team and handicap
        if team_name == self.team1_name:
            if handicap == 1.5:  # Team 1 +1.5 (needs to win at least 1 map)
                return prob_2_0 + prob_2_1 + prob_1_2  # All except 0-2
            elif handicap == -1.5:  # Team 1 -1.5 (needs to win 2-0)
                return prob_2_0
        else:  # Team 2
            if handicap == 1.5:  # Team 2 +1.5 (needs to win at least 1 map)
                return prob_0_2 + prob_1_2 + prob_2_1  # All except 2-0
            elif handicap == -1.5:  # Team 2 -1.5 (needs to win 2-0)
                return prob_0_2
                
        return 0.5  # Default fallback

    def _calculate_over_under_probability(self, is_over):
        """Calculate probability for over/under total maps bets"""
        # Get the correct win probabilities for each team
        predicted_winner = self.prediction_data['predicted_winner']
        win_probability = self.prediction_data['win_probability']
        
        # Set probabilities based on predicted winner
        if predicted_winner == self.team1_name:
            team1_win_prob = win_probability
            team2_win_prob = 1 - win_probability
        else:
            team2_win_prob = win_probability
            team1_win_prob = 1 - win_probability
        
        # Same logic as handicap probability calculation
        close_match_factor = 1 - abs(team1_win_prob - team2_win_prob) * 2
        
        prob_2_0_if_team1_wins = 0.55 - (close_match_factor * 0.2)
        prob_2_1_if_team1_wins = 1 - prob_2_0_if_team1_wins
        
        prob_0_2_if_team2_wins = 0.55 - (close_match_factor * 0.2)
        prob_1_2_if_team2_wins = 1 - prob_0_2_if_team2_wins
        
        # Calculate specific outcome probabilities
        prob_2_0 = team1_win_prob * prob_2_0_if_team1_wins
        prob_2_1 = team1_win_prob * prob_2_1_if_team1_wins
        prob_0_2 = team2_win_prob * prob_0_2_if_team2_wins
        prob_1_2 = team2_win_prob * prob_1_2_if_team2_wins
        
        # For over 2.5, we need map 3 to be played (2-1 or 1-2)
        if is_over:
            return prob_2_1 + prob_1_2
        else:  # Under 2.5 means 2-0 or 0-2
            return prob_2_0 + prob_0_2

    def _calculate_win_map_probability(self, team_name):
        """Calculate probability for 'win at least 1 map' bets"""
        # Get the correct win probabilities for each team
        predicted_winner = self.prediction_data['predicted_winner']
        win_probability = self.prediction_data['win_probability']
        
        # Set probabilities based on predicted winner
        if predicted_winner == self.team1_name:
            team1_win_prob = win_probability
            team2_win_prob = 1 - win_probability
        else:
            team2_win_prob = win_probability
            team1_win_prob = 1 - win_probability
        
        # This is essentially the same as the +1.5 handicap calculation
        close_match_factor = 1 - abs(team1_win_prob - team2_win_prob) * 2
        
        prob_2_0_if_team1_wins = 0.55 - (close_match_factor * 0.2)
        prob_2_1_if_team1_wins = 1 - prob_2_0_if_team1_wins
        
        prob_0_2_if_team2_wins = 0.55 - (close_match_factor * 0.2)
        prob_1_2_if_team2_wins = 1 - prob_0_2_if_team2_wins
        
        prob_2_0 = team1_win_prob * prob_2_0_if_team1_wins
        prob_2_1 = team1_win_prob * prob_2_1_if_team1_wins
        prob_0_2 = team2_win_prob * prob_0_2_if_team2_wins
        prob_1_2 = team2_win_prob * prob_1_2_if_team2_wins
        
        if team_name == self.team1_name:
            return prob_2_0 + prob_2_1 + prob_1_2  # All except 0-2
        else:
            return prob_0_2 + prob_1_2 + prob_2_1  # All except 2-0

    def _generate_ml_reasoning(self, team_name, win_probability):
        """Generate reasoning for moneyline bets"""
        if team_name == self.team1_name:
            team_stats = self.prediction_data['team1_stats']
            opponent_stats = self.prediction_data['team2_stats']
            h2h_win_rate = self.prediction_data['h2h_stats']['team1_h2h_win_rate']
        else:
            team_stats = self.prediction_data['team2_stats']
            opponent_stats = self.prediction_data['team1_stats']
            h2h_win_rate = self.prediction_data['h2h_stats']['team2_h2h_win_rate']
        
        reasoning = []
        
        # Win rate comparison
        if team_stats['win_rate'] > opponent_stats['win_rate']:
            reasoning.append(f"{team_name} has a {team_stats['win_rate']:.1%} win rate, superior to opponent's {opponent_stats['win_rate']:.1%}")
        
        # Recent form
        if team_stats['recent_form'] > 0.5:
            reasoning.append(f"Strong recent form ({team_stats['recent_form']:.1%} win rate in last 5 matches)")
        
        # Score differential
        if team_stats['score_differential'] > 0:
            reasoning.append(f"Positive score differential of {team_stats['score_differential']:.2f} per match")
        
        # Head-to-head record
        if h2h_win_rate > 0.5 and self.prediction_data['h2h_stats']['total_h2h_matches'] > 0:
            reasoning.append(f"Strong head-to-head record ({h2h_win_rate:.1%} win rate)")
        
        # Model prediction
        model_confidence = "high" if abs(win_probability - 0.5) > 0.15 else "moderate"
        reasoning.append(f"Model gives {model_confidence} confidence with {win_probability:.1%} win probability")
        
        return "; ".join(reasoning)

    def _generate_handicap_reasoning(self, team_name, handicap, win_probability):
        """Generate reasoning for handicap bets"""
        if team_name == self.team1_name:
            team_stats = self.prediction_data['team1_stats']
            opponent_stats = self.prediction_data['team2_stats']
        else:
            team_stats = self.prediction_data['team2_stats']
            opponent_stats = self.prediction_data['team1_stats']
        
        reasoning = []
        
        if handicap == "+1.5":
            # For +1.5 handicaps (team needs to win at least 1 map)
            reasoning.append(f"{team_name} has {win_probability:.1%} chance to win at least one map")
            
            if team_stats['win_rate'] > 0.4:
                reasoning.append(f"Reasonable {team_stats['win_rate']:.1%} overall win rate")
            
            # Getting 2-0'd analysis
            sweep_against_rate = 1 - win_probability
            reasoning.append(f"Only {sweep_against_rate:.1%} chance of getting swept 0-2")
            
        elif handicap == "-1.5":
            # For -1.5 handicaps (team needs to win 2-0)
            reasoning.append(f"{team_name} has {win_probability:.1%} chance to win 2-0")
            
            if team_stats['win_rate'] > 0.6:
                reasoning.append(f"Strong {team_stats['win_rate']:.1%} overall win rate")
                
            if team_stats['score_differential'] > 0.3:
                reasoning.append(f"Dominant score differential of {team_stats['score_differential']:.2f}")
        
        return "; ".join(reasoning)

    def _generate_over_under_reasoning(self, is_over, win_probability):
        """Generate reasoning for over/under bets"""
        # Get the correct win probabilities for each team
        predicted_winner = self.prediction_data['predicted_winner']
        pred_win_probability = self.prediction_data['win_probability']
        
        # Set probabilities based on predicted winner
        if predicted_winner == self.team1_name:
            team1_win_prob = pred_win_probability
            team2_win_prob = 1 - pred_win_probability
        else:
            team2_win_prob = pred_win_probability
            team1_win_prob = 1 - pred_win_probability
        
        team1_name = self.team1_name
        team2_name = self.team2_name
        
        reasoning = []
        
        if is_over:
            # For over 2.5 maps
            reasoning.append(f"{win_probability:.1%} chance that match goes to 3 maps")
            
            # Competitive match analysis
            if abs(team1_win_prob - team2_win_prob) < 0.1:
                reasoning.append(f"Evenly matched teams ({team1_win_prob:.1%} vs {team2_win_prob:.1%} win probability)")
                
            # Competitiveness in previous matches
            if self.prediction_data['h2h_stats']['total_h2h_matches'] > 0:
                reasoning.append(f"{self.prediction_data['h2h_stats']['total_h2h_matches']} previous matches between these teams")
        else:
            # For under 2.5 maps
            reasoning.append(f"{win_probability:.1%} chance that match ends in 2-0 sweep")
            
            
            # Lopsided match analysis
            if abs(team1_win_prob - 0.5) > 0.15:
                stronger_team = team1_name if team1_win_prob > team2_win_prob else team2_name
                reasoning.append(f"Clear favorite in {stronger_team}")
        
        return "; ".join(reasoning)

    def _generate_win_map_reasoning(self, team_name, win_probability):
        """Generate reasoning for 'win at least 1 map' bets"""
        # Similar to +1.5 handicap reasoning
        if team_name == self.team1_name:
            team_stats = self.prediction_data['team1_stats']
        else:
            team_stats = self.prediction_data['team2_stats']
        
        reasoning = []
        reasoning.append(f"{team_name} has {win_probability:.1%} chance to win at least one map")
        
        if team_stats['win_rate'] > 0.4:
            reasoning.append(f"Reasonable {team_stats['win_rate']:.1%} overall win rate")
        
        if team_stats['recent_form'] > 0.5:
            reasoning.append(f"In good form recently ({team_stats['recent_form']:.1%} in last 5 matches)")
        
        # Analysis of getting swept
        swept_prob = 1 - win_probability
        reasoning.append(f"Only {swept_prob:.1%} chance of getting swept 0-2")
        
        return "; ".join(reasoning)

    def print_recommendations(self, recommendations):
            """Print betting recommendations in a formatted way"""
            if not recommendations:
                print("\nNo positive EV bets found for this match.")
                return
            
            print("\n===== BETTING RECOMMENDATIONS =====")
            print(f"Match: {self.team1_name} vs {self.team2_name}")
            print(f"Prediction: {self.prediction_data['predicted_winner']} ({self.prediction_data['win_probability']:.2%} win probability)")
            print("==================================")
            
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['bet_type']}")
                print(f"   Odds: {rec['odds']:.2f}")
                print(f"   Win Probability: {rec['win_probability']:.2%}")
                print(f"   Expected Value: {rec['expected_value']:.4f} units per unit wagered")
                print(f"   Recommended Bet: ${rec['bet_amount']:.2f}")
                print(f"   Reasoning: {rec['reasoning']}")
            
            print("\n==================================")
            print(f"Total recommended investment: ${sum(rec['bet_amount'] for rec in recommendations):.2f}")

def main():
    """Main function to run the betting advisor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Valorant Betting Advisor')
    parser.add_argument('prediction_file', type=str, help='Path to the prediction JSON file')
    parser.add_argument('--bankroll', type=float, default=1000, help='Total bankroll for betting (default: $1000)')
    args = parser.parse_args()
    
    # Initialize the betting advisor
    advisor = ValorantBettingAdvisor(args.prediction_file)
    
    # Get user input for odds
    print("\n===== ENTER BETTING ODDS =====")
    print("(Enter decimal odds, e.g. 2.5 means a $1 bet returns $2.50)")
    print("(Leave blank to skip a bet type)\n")
    
    odds_dict = {}
    
    # Moneyline odds
    team1_ml = input(f"{advisor.team1_name} to win (moneyline): ")
    if team1_ml:
        odds_dict['team1_ml'] = float(team1_ml)
    
    team2_ml = input(f"{advisor.team2_name} to win (moneyline): ")
    if team2_ml:
        odds_dict['team2_ml'] = float(team2_ml)
    
    # Handicap odds
    team1_plus1_5 = input(f"{advisor.team1_name} +1.5 maps: ")
    if team1_plus1_5:
        odds_dict['team1_plus1_5'] = float(team1_plus1_5)
    
    team2_plus1_5 = input(f"{advisor.team2_name} +1.5 maps: ")
    if team2_plus1_5:
        odds_dict['team2_plus1_5'] = float(team2_plus1_5)
    
    team1_minus1_5 = input(f"{advisor.team1_name} -1.5 maps: ")
    if team1_minus1_5:
        odds_dict['team1_minus1_5'] = float(team1_minus1_5)
    
    team2_minus1_5 = input(f"{advisor.team2_name} -1.5 maps: ")
    if team2_minus1_5:
        odds_dict['team2_minus1_5'] = float(team2_minus1_5)
    
    # Over/Under odds
    over_2_5_maps = input("Over 2.5 total maps: ")
    if over_2_5_maps:
        odds_dict['over_2_5_maps'] = float(over_2_5_maps)
    
    under_2_5_maps = input("Under 2.5 total maps: ")
    if under_2_5_maps:
        odds_dict['under_2_5_maps'] = float(under_2_5_maps)
    
    # Win at least 1 map odds
    team1_win_map = input(f"{advisor.team1_name} to win at least 1 map: ")
    if team1_win_map:
        odds_dict['team1_win_map'] = float(team1_win_map)
    
    team2_win_map = input(f"{advisor.team2_name} to win at least 1 map: ")
    if team2_win_map:
        odds_dict['team2_win_map'] = float(team2_win_map)
    
    # Get recommendations
    recommendations = advisor.get_betting_advice(odds_dict, args.bankroll)
    
    # Print recommendations
    advisor.print_recommendations(recommendations)

if __name__ == "__main__":
    main()