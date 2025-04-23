import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import glob
import sys
from typing import Dict, List, Optional, Tuple, Union, Any
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import joblib
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
from bs4 import BeautifulSoup
from dataclasses import dataclass, field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("valorant_betting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class BetRecord:
    """Class to track bet history"""
    match_id: str
    match: str
    bet_type: str
    pick: str
    odds: float
    stake: float
    model_probability: float
    implied_probability: float
    edge: float
    confidence: float
    timestamp: str
    result: Optional[str] = None
    profit_loss: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class BettingHistory:
    """Class to manage betting history"""
    bets: List[BetRecord] = field(default_factory=list)
    
    def add_bet(self, bet: BetRecord) -> None:
        self.bets.append(bet)
    
    def update_result(self, match_id: str, bet_type: str, result: str, profit_loss: float) -> None:
        for bet in self.bets:
            if bet.match_id == match_id and bet.bet_type == bet_type:
                bet.result = result
                bet.profit_loss = profit_loss
                return
        logger.warning(f"Bet not found for match_id {match_id} and bet_type {bet_type}")
    
    def get_results(self) -> Dict[str, Any]:
        """Calculate and return betting performance metrics"""
        if not self.bets:
            return {"status": "No bets recorded"}
        
        total_bets = len([b for b in self.bets if b.result is not None])
        if total_bets == 0:
            return {"status": "No settled bets"}
        
        wins = len([b for b in self.bets if b.result == "win"])
        roi = sum([b.profit_loss for b in self.bets if b.profit_loss is not None]) / sum([b.stake for b in self.bets if b.profit_loss is not None])
        
        return {
            "total_bets": total_bets,
            "wins": wins,
            "win_rate": wins / total_bets if total_bets > 0 else 0,
            "roi": roi,
            "net_profit": sum([b.profit_loss for b in self.bets if b.profit_loss is not None]),
            "bets_by_type": self._group_by_bet_type()
        }
    
    def _group_by_bet_type(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for bet_type in set([b.bet_type for b in self.bets]):
            type_bets = [b for b in self.bets if b.bet_type == bet_type and b.result is not None]
            if not type_bets:
                continue
            
            wins = len([b for b in type_bets if b.result == "win"])
            
            result[bet_type] = {
                "count": len(type_bets),
                "win_rate": wins / len(type_bets) if len(type_bets) > 0 else 0,
                "roi": sum([b.profit_loss for b in type_bets if b.profit_loss is not None]) / sum([b.stake for b in type_bets if b.profit_loss is not None])
            }
        
        return result
    
    def save(self, filename: str = "betting_history.json") -> None:
        with open(filename, 'w') as f:
            json.dump([bet.__dict__ for bet in self.bets], f, indent=2)
    
    @classmethod
    def load(cls, filename: str = "betting_history.json") -> 'BettingHistory':
        if not os.path.exists(filename):
            return cls()
        
        with open(filename, 'r') as f:
            bet_dicts = json.load(f)
        
        history = cls()
        for bet_dict in bet_dicts:
            history.add_bet(BetRecord(**bet_dict))
        
        return history
    
    def calculate_calibration(self) -> Dict[str, float]:
        """Calculate probability calibration metrics"""
        if not self.bets or not any(b.result is not None for b in self.bets):
            return {"error": "No settled bets to analyze"}
        
        probabilities = []
        outcomes = []
        
        for bet in self.bets:
            if bet.result is not None:
                probabilities.append(bet.model_probability)
                outcomes.append(1 if bet.result == "win" else 0)
        
        # Group predictions into bins
        bin_edges = np.linspace(0, 1, 11)  # 10 bins
        bin_indices = np.digitize(probabilities, bin_edges)
        
        bin_sums = np.zeros(10)
        bin_totals = np.zeros(10)
        bin_actual = np.zeros(10)
        
        for i, bin_idx in enumerate(bin_indices):
            if bin_idx > 0 and bin_idx <= 10:  # Valid bin
                bin_idx -= 1  # Adjust index to 0-9
                bin_sums[bin_idx] += probabilities[i]
                bin_totals[bin_idx] += 1
                bin_actual[bin_idx] += outcomes[i]
        
        # Calculate average predicted probability and actual frequency per bin
        bin_avg_pred = np.zeros(10)
        bin_freq = np.zeros(10)
        
        for i in range(10):
            if bin_totals[i] > 0:
                bin_avg_pred[i] = bin_sums[i] / bin_totals[i]
                bin_freq[i] = bin_actual[i] / bin_totals[i]
        
        # Calculate calibration metrics
        valid_bins = bin_totals > 0
        if not any(valid_bins):
            return {"error": "No valid calibration bins"}
        
        calibration_error = np.mean(np.abs(bin_avg_pred[valid_bins] - bin_freq[valid_bins]))
        
        return {
            "calibration_error": calibration_error,
            "bins": {
                "edges": bin_edges.tolist(),
                "predicted": bin_avg_pred.tolist(),
                "actual": bin_freq.tolist(),
                "counts": bin_totals.tolist()
            }
        }


class OddsScraperBase:
    """Base class for odds scrapers from different sites"""
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def scrape_odds(self, team1: str, team2: str) -> Dict[str, float]:
        """Scrape odds for a match - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")


class DummyOddsScraper(OddsScraperBase):
    """Dummy scraper that returns fixed odds for testing"""
    def scrape_odds(self, team1: str, team2: str) -> Dict[str, float]:
        # Return dummy odds for testing
        return {
            "moneyline_team1": 2.10,
            "moneyline_team2": 1.70,
            "handicap_team1_plus_1.5": 1.20,
            "handicap_team1_minus_1.5": 3.50,
            "handicap_team2_plus_1.5": 1.25,
            "handicap_team2_minus_1.5": 3.20,
            "total_maps_over_2.5": 1.90,
            "total_maps_under_2.5": 1.90
        }


class OddsScraperFactory:
    """Factory for creating odds scrapers"""
    @staticmethod
    def create_scraper(scraper_type: str = "dummy") -> OddsScraperBase:
        if scraper_type.lower() == "dummy":
            return DummyOddsScraper()
        else:
            logger.warning(f"Unknown scraper type: {scraper_type}, using dummy scraper")
            return DummyOddsScraper()


class ValorantBettingAnalyzer:
    def __init__(self, bankroll: float = 1000.0, odds_scraper: Optional[OddsScraperBase] = None):
        self.predictions: List[Dict] = []
        self.odds_data: List[Dict] = []
        self.value_bets: List[Dict] = []
        self.confidence_threshold: float = 0.60  # Can be adjusted
        self.value_threshold: float = 0.10  # 10% edge minimum
        self.bankroll: float = bankroll  # Starting bankroll in dollars
        self.kelly_fraction: float = 0.25  # Conservative Kelly criterion
        self.betting_history: BettingHistory = BettingHistory.load()
        self.odds_scraper: OddsScraperBase = odds_scraper or OddsScraperFactory.create_scraper()
        self.calibration_model = None
        self.tournament_weights: Dict[str, float] = self._initialize_tournament_weights()

    def _initialize_tournament_weights(self) -> Dict[str, float]:
        """Initialize tournament weights based on importance"""
        return {
            "VCT": 1.2,         # Most important tournaments
            "VCL": 1.0,         # Standard importance
            "WDG VCL": 1.0,     # Standard importance
            "Challengers": 0.9,  # Slightly less important
            "Open": 0.8,         # Less important
        }

    def load_prediction(self, prediction_file: str) -> Dict:
        """Load a prediction from a JSON file with validation"""
        try:
            with open(prediction_file, 'r') as f:
                prediction_data = json.load(f)
            
            # Validate required fields
            required_fields = ["prediction", "team_stats"]
            if not all(field in prediction_data for field in required_fields):
                logger.error(f"Missing required fields in {prediction_file}")
                raise ValueError(f"Missing required fields in {prediction_file}")
            
            # Apply Bayesian updating if we have new information
            prediction_data = self._apply_bayesian_update(prediction_data)
            
            # Apply probability calibration if we have a calibration model
            if self.calibration_model is not None:
                prediction_data = self._calibrate_prediction(prediction_data)
            
            self.predictions.append(prediction_data)
            logger.info(f"Successfully loaded prediction from {prediction_file}")
            return prediction_data
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {prediction_file}")
            raise ValueError(f"Invalid JSON in {prediction_file}")
        except Exception as e:
            logger.error(f"Error loading {prediction_file}: {str(e)}")
            raise

    def _apply_bayesian_update(self, prediction_data: Dict) -> Dict:
        """Apply Bayesian updating to predictions based on recent information"""
        # This is a simplified implementation - a real one would fetch actual new information
        
        # Extract initial probabilities
        team1 = prediction_data["prediction"]["match"].split(" vs ")[0]
        team2 = prediction_data["prediction"]["match"].split(" vs ")[1]
        team1_win_prob = prediction_data["prediction"]["team1_win_probability"]
        team2_win_prob = prediction_data["prediction"]["team2_win_probability"]
        
        # Apply tournament weight adjustments
        tournament_name = next(
            (tournament for tournament in self.tournament_weights if tournament in prediction_data.get("tournament", "")),
            None
        )
        if tournament_name:
            tournament_weight = self.tournament_weights[tournament_name]
            
            # Apply tournament weight as a modifier to the confidence
            # This increases or decreases the polarity of the prediction based on tournament importance
            if team1_win_prob > 0.5:
                # Team 1 is favored, apply weight
                team1_win_prob = team1_win_prob ** (1 / tournament_weight)
                team2_win_prob = 1 - team1_win_prob
            else:
                # Team 2 is favored, apply weight
                team2_win_prob = team2_win_prob ** (1 / tournament_weight)
                team1_win_prob = 1 - team2_win_prob
        
        # Check for recent H2H matchups in betting history
        recent_matches = [
            bet for bet in self.betting_history.bets 
            if (team1 in bet.match and team2 in bet.match) and bet.result is not None
        ]
        
        if recent_matches:
            # Simple Bayesian update based on recent H2H results
            for match in recent_matches:
                # Get outcome
                team1_won = team1 in match.pick and match.result == "win"
                team2_won = team2 in match.pick and match.result == "win"
                
                # Apply a small Bayesian update - this is simplified
                if team1_won:
                    # Increase team1's probability slightly
                    team1_win_prob = (team1_win_prob * 1.05) / ((team1_win_prob * 1.05) + team2_win_prob)
                    team2_win_prob = 1 - team1_win_prob
                elif team2_won:
                    # Increase team2's probability slightly
                    team2_win_prob = (team2_win_prob * 1.05) / ((team2_win_prob * 1.05) + team1_win_prob)
                    team1_win_prob = 1 - team2_win_prob
        
        # Adjust for recency bias (form)
        team1_form = prediction_data["team_stats"][team1].get("recent_form", 0.5)
        team2_form = prediction_data["team_stats"][team2].get("recent_form", 0.5)
        
        # Apply a small form adjustment
        form_weight = 0.1  # 10% weight to form
        team1_win_prob = (team1_win_prob * (1 - form_weight)) + (team1_form * form_weight)
        team2_win_prob = (team2_win_prob * (1 - form_weight)) + (team2_form * form_weight)
        
        # Normalize probabilities to ensure they sum to 1
        total_prob = team1_win_prob + team2_win_prob
        team1_win_prob = team1_win_prob / total_prob
        team2_win_prob = team2_win_prob / total_prob
        
        # Update the prediction
        prediction_data["prediction"]["team1_win_probability"] = team1_win_prob
        prediction_data["prediction"]["team2_win_probability"] = team2_win_prob
        prediction_data["prediction"]["confidence"] = max(team1_win_prob, team2_win_prob)
        
        # Log the adjustments
        logger.info(f"Applied Bayesian updates to prediction for {team1} vs {team2}")
        
        return prediction_data

    def _calibrate_prediction(self, prediction_data: Dict) -> Dict:
        """Apply probability calibration to the prediction"""
        if self.calibration_model is None:
            return prediction_data
        
        try:
            team1_win_prob = prediction_data["prediction"]["team1_win_probability"]
            # Use an isotonic regression model to calibrate the probability
            calibrated_prob = self.calibration_model.predict([[team1_win_prob]])[0]
            
            prediction_data["prediction"]["team1_win_probability"] = calibrated_prob
            prediction_data["prediction"]["team2_win_probability"] = 1 - calibrated_prob
            prediction_data["prediction"]["confidence"] = max(calibrated_prob, 1 - calibrated_prob)
            
            logger.info(f"Applied probability calibration: original={team1_win_prob:.4f}, calibrated={calibrated_prob:.4f}")
            
            return prediction_data
        except Exception as e:
            logger.error(f"Error applying calibration: {str(e)}")
            return prediction_data

    def fit_calibration_model(self) -> None:
        """Fit a calibration model based on historical predictions and outcomes"""
        if not self.betting_history.bets or not any(b.result is not None for b in self.betting_history.bets):
            logger.warning("Not enough historical data to fit calibration model")
            return
        
        probabilities = []
        outcomes = []
        
        for bet in self.betting_history.bets:
            if bet.result is not None:
                probabilities.append(bet.model_probability)
                outcomes.append(1 if bet.result == "win" else 0)
        
        if len(probabilities) < 10:
            logger.warning("Not enough data points to fit a reliable calibration model")
            return
        
        try:
            # Reshape for scikit-learn
            probabilities = np.array(probabilities).reshape(-1, 1)
            outcomes = np.array(outcomes)
            
            # Use isotonic regression for calibration
            self.calibration_model = IsotonicRegression(out_of_bounds='clip')
            self.calibration_model.fit(probabilities, outcomes)
            
            # Save the model
            joblib.dump(self.calibration_model, 'calibration_model.pkl')
            
            logger.info(f"Fitted calibration model on {len(probabilities)} historical predictions")
        except Exception as e:
            logger.error(f"Error fitting calibration model: {str(e)}")

    def load_calibration_model(self, model_path: str = 'calibration_model.pkl') -> bool:
        """Load a previously saved calibration model"""
        if not os.path.exists(model_path):
            logger.info(f"No calibration model found at {model_path}")
            return False
        
        try:
            self.calibration_model = joblib.load(model_path)
            logger.info(f"Loaded calibration model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading calibration model: {str(e)}")
            return False

    def load_predictions_from_directory(self, directory: str) -> List[Dict]:
        """Load all prediction files from a directory"""
        prediction_files = glob.glob(os.path.join(directory, "*.json"))
        if not prediction_files:
            logger.warning(f"No prediction files found in {directory}")
            return []
        
        logger.info(f"Found {len(prediction_files)} prediction files")
        
        # Use ThreadPoolExecutor for parallel loading
        with ThreadPoolExecutor(max_workers=min(10, len(prediction_files))) as executor:
            futures = []
            for file_path in prediction_files:
                futures.append(executor.submit(self.load_prediction, file_path))
            
            # Collect results, handling exceptions
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in thread: {e}")
        
        return self.predictions

    def add_odds(self, match_id: str, team1: str, team2: str, 
                 ml_team1: float, ml_team2: float, 
                 handicap_team1_plus: float, handicap_team1_minus: float, 
                 handicap_team2_plus: float, handicap_team2_minus: float, 
                 map_over: float, map_under: float) -> Dict:
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
        logger.info(f"Added odds for {team1} vs {team2}")
        return odds_entry

    def collect_odds_interactively(self) -> bool:
        """Collect odds for each match prediction interactively with automated scraping option"""
        if not self.predictions:
            logger.warning("No predictions loaded. Please load predictions first.")
            return False
        
        logger.info("\nCollecting odds for each match prediction...\n")
        use_scraper = input("Would you like to try auto-scraping odds? (y/n): ").lower() == 'y'
        
        for idx, prediction in enumerate(self.predictions):
            match = prediction["prediction"]["match"]
            team1 = match.split(" vs ")[0]
            team2 = match.split(" vs ")[1]
            
            print(f"\nMatch {idx+1}/{len(self.predictions)}: {match}")
            
            if use_scraper:
                try:
                    logger.info(f"Attempting to scrape odds for {match}")
                    scraped_odds = self.odds_scraper.scrape_odds(team1, team2)
                    
                    # Display scraped odds and ask for confirmation
                    print("\nScraped odds:")
                    for key, value in scraped_odds.items():
                        print(f"{key}: {value}")
                    
                    use_these_odds = input("\nUse these odds? (y/n): ").lower() == 'y'
                    
                    if use_these_odds:
                        self.add_odds(
                            match_id=f"match_{idx}",
                            team1=team1,
                            team2=team2,
                            ml_team1=scraped_odds["moneyline_team1"],
                            ml_team2=scraped_odds["moneyline_team2"],
                            handicap_team1_plus=scraped_odds["handicap_team1_plus_1.5"],
                            handicap_team1_minus=scraped_odds["handicap_team1_minus_1.5"],
                            handicap_team2_plus=scraped_odds["handicap_team2_plus_1.5"],
                            handicap_team2_minus=scraped_odds["handicap_team2_minus_1.5"],
                            map_over=scraped_odds["total_maps_over_2.5"],
                            map_under=scraped_odds["total_maps_under_2.5"]
                        )
                        print(f"Added scraped odds for {match}")
                        continue
                    
                    print("Manual odds entry:")
                except Exception as e:
                    logger.error(f"Error scraping odds: {e}")
                    print("Failed to scrape odds. Falling back to manual entry.")
            
            try:
                # Manual odds entry
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
                logger.error(f"Invalid input: {e}")
                print(f"Error: Invalid input. Please enter odds as decimals (e.g. 2.40).")
                retry = input("Would you like to retry this match? (y/n): ")
                if retry.lower() == 'y':
                    idx -= 1  # Retry this match
                else:
                    print(f"Skipping {match}")
            
            except KeyboardInterrupt:
                logger.info("Odds collection interrupted")
                print("\nOdds collection interrupted. Processing available data...")
                break
        
        return True

    def _decimal_to_implied_probability(self, decimal_odds: float) -> float:
        """Convert decimal odds to implied probability"""
        return 1 / decimal_odds

    def analyze_value_bets(self) -> List[Dict]:
        """Find value betting opportunities with enhanced analysis"""
        self.value_bets = []
        
        # Load calibration model if available
        self.load_calibration_model()
        
        # First, convert predictions and odds to DataFrames for more efficient processing
        pred_data = []
        for prediction in self.predictions:
            match = prediction["prediction"]["match"]
            team1 = match.split(" vs ")[0]
            team2 = match.split(" vs ")[1]
            
            pred_data.append({
                "match": match,
                "team1": team1,
                "team2": team2,
                "team1_win_prob": prediction["prediction"]["team1_win_probability"],
                "team2_win_prob": prediction["prediction"]["team2_win_probability"],
                "confidence": prediction["prediction"]["confidence"],
                "prediction_data": prediction  # Store the full prediction
            })
        
        predictions_df = pd.DataFrame(pred_data)
        odds_df = pd.DataFrame(self.odds_data)
        
        # Merge predictions with odds
        if odds_df.empty or predictions_df.empty:
            logger.warning("No odds data or predictions available")
            return []
        
        # Merge on team names
        analysis_df = pd.merge(
            predictions_df, 
            odds_df,
            how='inner',
            left_on=['team1', 'team2'],
            right_on=['team1', 'team2']
        )
        
        if analysis_df.empty:
            logger.warning("No matches found between predictions and odds")
            return []
        
        logger.info(f"Analyzing {len(analysis_df)} matches for value bets")
        
        # Process each match
        for _, row in analysis_df.iterrows():
            prediction = row['prediction_data']
            match = row['match']
            team1 = row['team1']
            team2 = row['team2']
            team1_win_prob = row['team1_win_prob']
            team2_win_prob = row['team2_win_prob']
            confidence = row['confidence']
            match_id = row['match_id']
            
            # Calculate map distribution probabilities with enhanced model
            map_distribution = self._calculate_map_distribution(
                team1_win_prob, team2_win_prob, team1, team2, prediction
            )
            
            # Extract probabilities from the distribution
            p_team1_2_0 = map_distribution['team1_2_0']
            p_team1_2_1 = map_distribution['team1_2_1']
            p_team2_2_0 = map_distribution['team2_2_0']
            p_team2_2_1 = map_distribution['team2_2_1']
            
            # Calculate derivatives
            p_over_2_5 = p_team1_2_1 + p_team2_2_1  # Matches that go to 3 maps
            p_under_2_5 = p_team1_2_0 + p_team2_2_0  # Matches that end in 2 maps
            
            # Calculate odds for handicaps
            p_team1_plus_1_5 = p_team1_2_0 + p_team1_2_1 + p_team2_2_1  # Team 1 wins at least 1 map
            p_team1_minus_1_5 = p_team1_2_0  # Team 1 wins 2-0
            p_team2_plus_1_5 = p_team2_2_0 + p_team2_2_1 + p_team1_2_1  # Team 2 wins at least 1 map
            p_team2_minus_1_5 = p_team2_2_0  # Team 2 wins 2-0
            
            # Convert bookmaker odds to probabilities
            ml_team1_decimal = row["moneyline_team1"]
            ml_team2_decimal = row["moneyline_team2"]
            handicap_team1_plus_decimal = row["handicap_team1_plus_1.5"]
            handicap_team1_minus_decimal = row["handicap_team1_minus_1.5"]
            handicap_team2_plus_decimal = row["handicap_