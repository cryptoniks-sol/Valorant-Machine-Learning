#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import norm

# Make sure our work integrates with your existing code
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

#-------------------------------------------------------------------------
# DATA PREPARATION
#-------------------------------------------------------------------------

def load_ensemble_models(model_paths, scaler_path, features_path):
    """
    Load ensemble models, scaler, and stable feature list.
    
    Args:
        model_paths (list): List of file paths to the trained models
        scaler_path (str): Path to the feature scaler
        features_path (str): Path to the stable features list
        
    Returns:
        tuple: (loaded_models, scaler, stable_features)
    """
    print("Loading ensemble models and data preprocessing artifacts...")
    
    # Load models
    loaded_models = []
    for path in model_paths:
        try:
            model = load_model(path)
            loaded_models.append(model)
            print(f"Loaded model from {path}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
    
    # Load scaler
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded feature scaler from {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        scaler = None
    
    # Load stable features
    try:
        with open(features_path, 'rb') as f:
            stable_features = pickle.load(f)
        print(f"Loaded {len(stable_features)} stable features")
    except Exception as e:
        print(f"Error loading stable features: {e}")
        stable_features = None
    
    return loaded_models, scaler, stable_features

def prepare_match_features(team1_stats, team2_stats, stable_features, scaler):
    """
    Prepare features for a single match for prediction.
    
    Args:
        team1_stats (dict): Statistics for team 1
        team2_stats (dict): Statistics for team 2
        stable_features (list): List of feature names to use
        scaler (object): Trained feature scaler
        
    Returns:
        numpy.ndarray: Scaled feature vector ready for model prediction
    """
    # Use the existing prepare_data_for_model function to get base features
    features = prepare_data_for_model(team1_stats, team2_stats)
    
    if not features:
        print("Failed to prepare match features.")
        return None
    
    # Convert to DataFrame for easier manipulation
    features_df = pd.DataFrame([features])
    
    # Select only stable features that were identified during cross-validation
    available_features = [f for f in stable_features if f in features_df.columns]
    
    if len(available_features) < len(stable_features):
        missing = len(stable_features) - len(available_features)
        print(f"Warning: {missing} stable features are missing from the input data.")
    
    # Select and order features to match the training data
    X = features_df[available_features].values
    
    # Scale features
    if scaler:
        X_scaled = scaler.transform(X)
        return X_scaled
    else:
        print("Warning: No scaler provided. Using unscaled features.")
        return X

#-------------------------------------------------------------------------
# PREDICTION AND BETTING FUNCTIONS
#-------------------------------------------------------------------------

def ensemble_predict(models, X, calibrate=True, confidence=True):
    """
    Get predictions from ensemble of models with confidence scores.
    
    Args:
        models (list): List of trained models
        X (numpy.ndarray): Scaled feature vector
        calibrate (bool): Whether to calibrate probabilities
        confidence (bool): Whether to return confidence intervals
        
    Returns:
        dict: Prediction results with probabilities and confidence intervals
    """
    if not models or X is None:
        return None
    
    # Get predictions from each model
    predictions = []
    for i, model in enumerate(models):
        try:
            pred = model.predict(X)
            predictions.append(pred[0][0])  # Extract raw probability
        except Exception as e:
            print(f"Error getting prediction from model {i+1}: {e}")
    
    # Calculate ensemble statistics
    if predictions:
        mean_prob = np.mean(predictions)
        median_prob = np.median(predictions)
        std_dev = np.std(predictions)
        
        # Apply calibration to correct for overconfidence
        if calibrate:
            # Simple calibration: push extreme probabilities toward the center
            # More sophisticated methods like Platt scaling could be used with more data
            calibrated_prob = 0.5 + (mean_prob - 0.5) * 0.9  # Shrink by 10%
        else:
            calibrated_prob = mean_prob
        
        result = {
            'team1_win_probability': float(calibrated_prob),
            'team2_win_probability': float(1 - calibrated_prob),
            'raw_predictions': predictions,
            'std_dev': float(std_dev),
            'ensemble_size': len(predictions)
        }
        
        # Add confidence intervals if requested
        if confidence:
            # 95% confidence interval
            alpha = 0.05
            lower_bound = max(0, mean_prob - 1.96 * std_dev / np.sqrt(len(predictions)))
            upper_bound = min(1, mean_prob + 1.96 * std_dev / np.sqrt(len(predictions)))
            
            result['confidence_interval'] = {
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'confidence_level': 1 - alpha
            }
        
        return result
    
    return None

def predict_series_outcome(team1_win_prob, format_type="bo3"):
    """
    Calculate probabilities for different match outcomes in a series.
    
    Args:
        team1_win_prob (float): Probability of team1 winning a single map
        format_type (str): Format of the series ("bo1", "bo3", "bo5")
        
    Returns:
        dict: Probabilities for different outcomes
    """
    team2_win_prob = 1 - team1_win_prob
    
    if format_type == "bo1":
        # Best of 1 (single map)
        return {
            'team1_win_series': team1_win_prob,
            'team2_win_series': team2_win_prob,
            'team1_win_2_0': 0,
            'team2_win_2_0': 0,
            'team1_win_2_1': 0,
            'team2_win_2_1': 0,
            'maps_over_1.5': 0,  # Always under in a BO1
            'maps_under_1.5': 1.0
        }
    
    elif format_type == "bo3":
        # Best of 3 (first to 2 maps)
        team1_2_0 = team1_win_prob * team1_win_prob  # Win first two maps
        team2_2_0 = team2_win_prob * team2_win_prob  # Win first two maps
        
        team1_2_1 = 2 * team1_win_prob * team2_win_prob * team1_win_prob  # Win, lose, win or lose, win, win
        team2_2_1 = 2 * team2_win_prob * team1_win_prob * team2_win_prob  # Win, lose, win or lose, win, win
        
        team1_win_series = team1_2_0 + team1_2_1
        team2_win_series = team2_2_0 + team2_2_1
        
        maps_over_2_5 = team1_2_1 + team2_2_1  # 3 maps played
        maps_under_2_5 = team1_2_0 + team2_2_0  # 2 maps played
        
        # Maps_over_1.5 means at least 2 maps played, which is always true in BO3
        
        return {
            'team1_win_series': team1_win_series,
            'team2_win_series': team2_win_series,
            'team1_win_2_0': team1_2_0,
            'team2_win_2_0': team2_2_0,
            'team1_win_2_1': team1_2_1,
            'team2_win_2_1': team2_2_1,
            'maps_over_2.5': maps_over_2_5,
            'maps_under_2.5': maps_under_2_5,
            'maps_over_1.5': 1.0,
            'maps_under_1.5': 0.0
        }
    
    elif format_type == "bo5":
        # Best of 5 (first to 3 maps)
        team1_3_0 = team1_win_prob ** 3
        team2_3_0 = team2_win_prob ** 3
        
        team1_3_1 = 3 * (team1_win_prob ** 3) * team2_win_prob
        team2_3_1 = 3 * (team2_win_prob ** 3) * team1_win_prob
        
        team1_3_2 = 6 * (team1_win_prob ** 3) * (team2_win_prob ** 2)
        team2_3_2 = 6 * (team2_win_prob ** 3) * (team1_win_prob ** 2)
        
        team1_win_series = team1_3_0 + team1_3_1 + team1_3_2
        team2_win_series = team2_3_0 + team2_3_1 + team2_3_2
        
        maps_over_3_5 = team1_3_1 + team2_3_1 + team1_3_2 + team2_3_2
        maps_under_3_5 = team1_3_0 + team2_3_0
        
        maps_over_4_5 = team1_3_2 + team2_3_2
        maps_under_4_5 = team1_3_0 + team2_3_0 + team1_3_1 + team2_3_1
        
        return {
            'team1_win_series': team1_win_series,
            'team2_win_series': team2_win_series,
            'team1_win_3_0': team1_3_0,
            'team2_win_3_0': team2_3_0,
            'team1_win_3_1': team1_3_1,
            'team2_win_3_1': team2_3_1,
            'team1_win_3_2': team1_3_2,
            'team2_win_3_2': team2_3_2,
            'maps_over_3.5': maps_over_3_5,
            'maps_under_3.5': maps_under_3_5,
            'maps_over_4.5': maps_over_4_5,
            'maps_under_4.5': maps_under_4_5
        }
    
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

def calculate_implied_probability(odds):
    """
    Calculate implied probability from decimal odds.
    
    Args:
        odds (float): Decimal odds
        
    Returns:
        float: Implied probability
    """
    return 1 / odds

def calculate_expected_value(probability, odds):
    """
    Calculate expected value of a bet.
    
    Args:
        probability (float): Estimated probability of winning
        odds (float): Decimal odds offered by bookmaker
        
    Returns:
        float: Expected value (positive = profitable)
    """
    return probability * (odds - 1) - (1 - probability)

def kelly_criterion(probability, odds, bankroll, fraction=1.0):
    """
    Calculate optimal bet size using the Kelly Criterion.
    
    Args:
        probability (float): Estimated probability of winning
        odds (float): Decimal odds offered by bookmaker
        bankroll (float): Current bankroll size
        fraction (float): Fraction of full Kelly to use (0.0-1.0), lower = more conservative
        
    Returns:
        float: Recommended bet size
    """
    # Calculate implied probability from odds
    implied_prob = calculate_implied_probability(odds)
    
    # Kelly formula: f* = (bp - q) / b
    # where f* is the fraction of bankroll to bet
    # p is the probability of winning
    # q is the probability of losing (1-p)
    # b is the odds - 1
    
    b = odds - 1  # Net odds received on the bet
    p = probability
    q = 1 - p
    
    if p <= 0 or p >= 1:
        return 0
    
    # Standard Kelly formula
    if p > implied_prob:  # Only bet when we have an edge
        kelly_stake = (b * p - q) / b
    else:
        kelly_stake = 0
    
    # Apply the fraction (fractional Kelly)
    kelly_stake *= fraction
    
    # Limit the stake to a reasonable amount (e.g., max 20% of bankroll)
    kelly_stake = min(kelly_stake, 0.2)
    
    # Calculate the actual bet size
    bet_size = bankroll * kelly_stake
    
    return bet_size

def find_value_bets(prediction_results, available_odds, format_type='bo3', bankroll=1000.0, kelly_fraction=0.25, min_edge=0.05):
    """
    Identify value bets with positive expected value and calculate optimal stakes.
    
    Args:
        prediction_results (dict): Model predictions for the match
        available_odds (dict): Available odds from bookmakers
        format_type (str): Match format (bo1, bo3, bo5)
        bankroll (float): Current bankroll size
        kelly_fraction (float): Conservative fraction of Kelly stake to use
        min_edge (float): Minimum edge required to place a bet
        
    Returns:
        dict: Value bets with optimal stake sizes and expected values
    """
    # Calculate series outcome probabilities
    map_win_prob = prediction_results['team1_win_probability']
    series_probs = predict_series_outcome(map_win_prob, format_type)
    
    # Initialize value bets container
    value_bets = {}
    
    # Check each bet type for value
    for bet_type, odds in available_odds.items():
        if bet_type in series_probs:
            our_prob = series_probs[bet_type]
            implied_prob = calculate_implied_probability(odds)
            edge = our_prob - implied_prob
            
            # Additional edge check: make sure edge meets minimum threshold
            edge_percent = edge / implied_prob * 100.0
            
            if edge > min_edge:
                ev = calculate_expected_value(our_prob, odds)
                stake = kelly_criterion(our_prob, odds, bankroll, kelly_fraction)
                
                if stake > 0:
                    value_bets[bet_type] = {
                        'odds': odds,
                        'our_probability': our_prob,
                        'implied_probability': implied_prob,
                        'edge': edge,
                        'edge_percent': edge_percent,
                        'expected_value': ev,
                        'recommended_stake': stake,
                        'profit_if_win': stake * (odds - 1),
                        'confidence': prediction_results.get('confidence_interval', {})
                    }
    
    return value_bets

def evaluate_models_profitability(models, test_data, odds_data, initial_bankroll=1000.0, fixed_stake=False, plot=True):
    """
    Evaluate model profitability on historical data with real odds.
    
    Args:
        models (list): List of trained models
        test_data (DataFrame): Test dataset with features and outcomes
        odds_data (DataFrame): Historical odds data for the matches
        initial_bankroll (float): Starting bankroll
        fixed_stake (bool/float): Use fixed stake instead of Kelly (False or stake amount)
        plot (bool): Whether to plot results
        
    Returns:
        dict: Profitability metrics including ROI and final bankroll
    """
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bet_history = []
    dates = []
    
    for i, match in test_data.iterrows():
        # Prepare match features
        X = match['features'].reshape(1, -1)
        
        # Get model prediction
        prediction = ensemble_predict(models, X)
        
        # Get actual outcome
        actual = match['outcome']
        
        # Get odds for this match
        match_odds = odds_data.loc[odds_data['match_id'] == match['match_id']].to_dict('records')[0]
        
        # Calculate recommended bets
        value_bets = find_value_bets(
            prediction, 
            match_odds['odds'], 
            format_type=match_odds['format'],
            bankroll=bankroll
        )
        
        # Simulate betting on all value bets
        for bet_type, bet_info in value_bets.items():
            # Determine stake amount
            if fixed_stake:
                stake = fixed_stake if isinstance(fixed_stake, (int, float)) else 50.0
            else:
                stake = bet_info['recommended_stake']
            
            # Ensure we don't bet more than what we have
            stake = min(stake, bankroll * 0.2)  # Max 20% per bet
            
            # Record bet
            bet_record = {
                'match_id': match['match_id'],
                'date': match['date'],
                'bet_type': bet_type,
                'odds': bet_info['odds'],
                'our_probability': bet_info['our_probability'],
                'implied_probability': bet_info['implied_probability'],
                'edge': bet_info['edge'],
                'stake': stake,
                'won': match[f'{bet_type}_result'] == 1,
                'profit': stake * (bet_info['odds'] - 1) if match[f'{bet_type}_result'] == 1 else -stake,
                'bankroll_before': bankroll
            }
            
            # Update bankroll
            bankroll += bet_record['profit']
            bet_record['bankroll_after'] = bankroll
            
            bet_history.append(bet_record)
            bankroll_history.append(bankroll)
            dates.append(match['date'])
    
    # Calculate profitability metrics
    bet_df = pd.DataFrame(bet_history)
    total_bets = len(bet_df)
    
    if total_bets > 0:
        total_stakes = bet_df['stake'].sum()
        total_profit = bet_df['profit'].sum()
        win_rate = (bet_df['won'] == True).mean()
        roi = total_profit / total_stakes if total_stakes > 0 else 0
        
        # Calculate win rate by bet type
        bet_type_analysis = bet_df.groupby('bet_type').agg({
            'stake': 'sum',
            'profit': 'sum',
            'won': ['count', 'mean']
        })
        
        # Plot results if requested
        if plot:
            plt.figure(figsize=(12, 8))
            
            # Bankroll evolution
            plt.subplot(2, 1, 1)
            plt.plot(range(len(bankroll_history)), bankroll_history, 'b-')
            plt.axhline(y=initial_bankroll, color='r', linestyle='--')
            plt.title('Bankroll Evolution')
            plt.xlabel('Bet Number')
            plt.ylabel('Bankroll ($)')
            plt.grid(True)
            
            # ROI by bet type
            plt.subplot(2, 1, 2)
            bet_types = bet_type_analysis.index
            rois = bet_type_analysis[('profit', 'sum')] / bet_type_analysis[('stake', 'sum')]
            
            plt.bar(bet_types, rois)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('ROI by Bet Type')
            plt.xlabel('Bet Type')
            plt.ylabel('ROI')
            plt.xticks(rotation=45)
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('profitability_analysis.png')
            plt.close()
        
        return {
            'initial_bankroll': initial_bankroll,
            'final_bankroll': bankroll,
            'profit': total_profit,
            'roi': roi,
            'win_rate': win_rate,
            'total_bets': total_bets,
            'total_stakes': total_stakes,
            'bankroll_history': bankroll_history,
            'bet_type_analysis': bet_type_analysis.to_dict(),
            'bet_history': bet_df
        }
    
    return {
        'initial_bankroll': initial_bankroll,
        'final_bankroll': bankroll,
        'profit': 0,
        'roi': 0,
        'win_rate': 0,
        'total_bets': 0,
        'error': 'No valid bets found'
    }

#-------------------------------------------------------------------------
# BACKTESTING AND CALIBRATION
#-------------------------------------------------------------------------

def calibrate_model_probabilities(models, calibration_data):
    """
    Calibrate model probabilities to match empirical outcomes.
    
    Args:
        models (list): List of trained models
        calibration_data (DataFrame): Data with features and known outcomes
        
    Returns:
        function: Calibration function to correct raw probabilities
    """
    # Get raw predictions for calibration set
    raw_probs = []
    actual_outcomes = []
    
    for i, match in calibration_data.iterrows():
        X = match['features'].reshape(1, -1)
        prediction = ensemble_predict(models, X, calibrate=False)
        
        if prediction:
            raw_probs.append(prediction['team1_win_probability'])
            actual_outcomes.append(match['outcome'])
    
    # Convert to numpy arrays
    raw_probs = np.array(raw_probs)
    actual_outcomes = np.array(actual_outcomes)
    
    # Break probabilities into bins and check actual win rates
    bins = 10
    bin_edges = np.linspace(0, 1, bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(raw_probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins-1)
    
    # Calculate actual win rates per bin
    bin_win_rates = []
    bin_counts = []
    
    for i in range(bins):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            win_rate = np.mean(actual_outcomes[mask])
            bin_win_rates.append(win_rate)
            bin_counts.append(np.sum(mask))
        else:
            bin_win_rates.append(np.nan)
            bin_counts.append(0)
    
    # Create calibration mapping
    bin_centers_valid = []
    bin_win_rates_valid = []
    
    for i in range(bins):
        if not np.isnan(bin_win_rates[i]) and bin_counts[i] >= 5:
            bin_centers_valid.append(bin_centers[i])
            bin_win_rates_valid.append(bin_win_rates[i])
    
    # If not enough data points, use a simple scaling approach
    if len(bin_centers_valid) < 3:
        print("Not enough data for full calibration, using simple scaling.")
        
        # Fit a simple scaling factor
        if raw_probs.size > 0 and actual_outcomes.size > 0:
            mean_raw_prob = np.mean(raw_probs)
            mean_actual = np.mean(actual_outcomes)
            
            if mean_raw_prob > 0:
                scaling_factor = mean_actual / mean_raw_prob
            else:
                scaling_factor = 1.0
                
            # Calibration function with simple scaling
            def calibrate_prob(raw_prob):
                calibrated = raw_prob * scaling_factor
                return max(0.01, min(0.99, calibrated))
        else:
            # Default no-op calibration
            def calibrate_prob(raw_prob):
                return raw_prob
    else:
        # Fit a logistic regression or isotonic regression for calibration
        from sklearn.isotonic import IsotonicRegression
        
        calibration_model = IsotonicRegression(out_of_bounds='clip')
        calibration_model.fit(np.array(bin_centers_valid), np.array(bin_win_rates_valid))
        
        def calibrate_prob(raw_prob):
            return max(0.01, min(0.99, calibration_model.predict([raw_prob])[0]))
    
    # Plot calibration curve if we have enough data
    if len(bin_centers_valid) >= 3:
        plt.figure(figsize=(10, 6))
        
        # Plot ideal calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        
        # Plot actual calibration points
        plt.scatter(bin_centers, bin_win_rates, s=[max(20, min(400, c*20)) for c in bin_counts], alpha=0.7)
        
        # Add point counts
        for i, (x, y, count) in enumerate(zip(bin_centers, bin_win_rates, bin_counts)):
            if not np.isnan(y) and count > 0:
                plt.annotate(f'{count}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Plot fitted calibration curve
        x_range = np.linspace(0, 1, 100)
        y_calibrated = [calibrate_prob(x) for x in x_range]
        plt.plot(x_range, y_calibrated, 'r-', label='Calibration Curve')
        
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Predicted Probability')
        plt.ylabel('Actual Win Rate')
        plt.title('Probability Calibration Curve')
        plt.legend(loc='best')
        plt.grid(True)
        
        plt.savefig('probability_calibration.png')
        plt.close()
    
    return calibrate_prob

def validate_calibration(models, validation_data, calibration_func=None):
    """
    Validate the calibration of model probabilities.
    
    Args:
        models (list): List of trained models
        validation_data (DataFrame): Data with features and known outcomes
        calibration_func (function): Function to calibrate raw probabilities
        
    Returns:
        dict: Calibration metrics including reliability and resolution
    """
    raw_probs = []
    calibrated_probs = []
    actual_outcomes = []
    
    for i, match in validation_data.iterrows():
        X = match['features'].reshape(1, -1)
        prediction = ensemble_predict(models, X, calibrate=False)
        
        if prediction:
            raw_prob = prediction['team1_win_probability']
            raw_probs.append(raw_prob)
            
            if calibration_func:
                calibrated_prob = calibration_func(raw_prob)
            else:
                calibrated_prob = raw_prob
                
            calibrated_probs.append(calibrated_prob)
            actual_outcomes.append(match['outcome'])
    
    # Convert to numpy arrays
    raw_probs = np.array(raw_probs)
    calibrated_probs = np.array(calibrated_probs)
    actual_outcomes = np.array(actual_outcomes)
    
    # Calculate Brier score (lower is better)
    raw_brier = np.mean((raw_probs - actual_outcomes) ** 2)
    calibrated_brier = np.mean((calibrated_probs - actual_outcomes) ** 2)
    
    # Calculate log loss
    def safe_log(x):
        return np.log(max(0.001, min(0.999, x)))
    
    raw_log_loss = -np.mean(actual_outcomes * safe_log(raw_probs) + 
                           (1 - actual_outcomes) * safe_log(1 - raw_probs))
    
    calibrated_log_loss = -np.mean(actual_outcomes * safe_log(calibrated_probs) + 
                                  (1 - actual_outcomes) * safe_log(1 - calibrated_probs))
    
    # Calculate reliability diagram data
    bins = 10
    bin_edges = np.linspace(0, 1, bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    raw_bin_indices = np.digitize(raw_probs, bin_edges) - 1
    raw_bin_indices = np.clip(raw_bin_indices, 0, bins-1)
    
    calibrated_bin_indices = np.digitize(calibrated_probs, bin_edges) - 1
    calibrated_bin_indices = np.clip(calibrated_bin_indices, 0, bins-1)
    
    # Calculate actual win rates per bin
    raw_bin_win_rates = []
    raw_bin_counts = []
    
    calibrated_bin_win_rates = []
    calibrated_bin_counts = []
    
    for i in range(bins):
        # Raw probabilities
        mask = (raw_bin_indices == i)
        if np.sum(mask) > 0:
            win_rate = np.mean(actual_outcomes[mask])
            raw_bin_win_rates.append(win_rate)
            raw_bin_counts.append(np.sum(mask))
        else:
            raw_bin_win_rates.append(np.nan)
            raw_bin_counts.append(0)
        
        # Calibrated probabilities
        mask = (calibrated_bin_indices == i)
        if np.sum(mask) > 0:
            win_rate = np.mean(actual_outcomes[mask])
            calibrated_bin_win_rates.append(win_rate)
            calibrated_bin_counts.append(np.sum(mask))
        else:
            calibrated_bin_win_rates.append(np.nan)
            calibrated_bin_counts.append(0)
    
    # Plot calibration curves
    plt.figure(figsize=(12, 8))
    
    # Raw probabilities calibration curve
    plt.subplot(2, 1, 1)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    # Plot raw probability calibration points
    plt.scatter(bin_centers, raw_bin_win_rates, 
                s=[max(20, min(400, c*20)) for c in raw_bin_counts], 
                alpha=0.7, 
                label='Raw Probabilities')
    
    # Add counts for raw probabilities
    for i, (x, y, count) in enumerate(zip(bin_centers, raw_bin_win_rates, raw_bin_counts)):
        if not np.isnan(y) and count > 0:
            plt.annotate(f'{count}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Win Rate')
    plt.title('Raw Probability Calibration')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Calibrated probabilities curve
    plt.subplot(2, 1, 2)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    # Plot calibrated probability points
    plt.scatter(bin_centers, calibrated_bin_win_rates, 
                s=[max(20, min(400, c*20)) for c in calibrated_bin_counts], 
                alpha=0.7, 
                label='Calibrated Probabilities')
    
    # Add counts for calibrated probabilities
    for i, (x, y, count) in enumerate(zip(bin_centers, calibrated_bin_win_rates, calibrated_bin_counts)):
        if not np.isnan(y) and count > 0:
            plt.annotate(f'{count}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Win Rate')
    plt.title('Calibrated Probability Calibration')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('calibration_validation.png')
    plt.close()
    
    return {
        'raw_brier_score': float(raw_brier),
        'calibrated_brier_score': float(calibrated_brier),
        'raw_log_loss': float(raw_log_loss),
        'calibrated_log_loss': float(calibrated_log_loss),
        'improvement': {
            'brier_score': float(raw_brier - calibrated_brier),
            'log_loss': float(raw_log_loss - calibrated_log_loss)
        },
        'reliability_diagram': {
            'bin_centers': bin_centers.tolist(),
            'raw_win_rates': [float(x) if not np.isnan(x) else None for x in raw_bin_win_rates],
            'calibrated_win_rates': [float(x) if not np.isnan(x) else None for x in calibrated_bin_win_rates],
            'raw_counts': raw_bin_counts,
            'calibrated_counts': calibrated_bin_counts
        }
    }

def backtest_betting_strategy(models, historical_data, odds_data, scaler=None, stable_features=None,
                             initial_bankroll=1000.0, kelly_fraction=0.25, min_edge=0.05,
                             calibration_data=None, use_calibration=True):
    """
    Backtest a betting strategy on historical match data.
    
    Args:
        models (list): List of trained models
        historical_data (DataFrame): Historical match data
        odds_data (DataFrame): Historical odds data
        scaler (object): Feature scaler
        stable_features (list): List of stable features
        initial_bankroll (float): Starting bankroll
        kelly_fraction (float): Conservative fraction of Kelly criterion
        min_edge (float): Minimum required edge for a bet
        calibration_data (DataFrame): Data to use for calibration
        use_calibration (bool): Whether to use probability calibration
        
    Returns:
        dict: Backtest results including profitability metrics
    """
    print("Starting betting strategy backtest...")
    
    # Split data into calibration, validation, and test sets if needed
    if calibration_data is None and use_calibration:
        # Use 20% of historical data for calibration if not provided
        data_size = len(historical_data)
        calibration_size = int(data_size * 0.2)
        validation_size = int(data_size * 0.2)
        test_size = data_size - calibration_size - validation_size
        
        # Shuffle data
        shuffled_indices = np.random.permutation(data_size)
        
        calibration_indices = shuffled_indices[:calibration_size]
        validation_indices = shuffled_indices[calibration_size:calibration_size+validation_size]
        test_indices = shuffled_indices[calibration_size+validation_size:]
        
        calibration_data = historical_data.iloc[calibration_indices].reset_index(drop=True)
        validation_data = historical_data.iloc[validation_indices].reset_index(drop=True)
        test_data = historical_data.iloc[test_indices].reset_index(drop=True)
    else:
        # Use all data for testing if no calibration needed
        test_data = historical_data
    
    # Calibrate probabilities if requested
    calibration_func = None
    if use_calibration and calibration_data is not None:
        print("Calibrating model probabilities...")
        calibration_func = calibrate_model_probabilities(models, calibration_data)
        
        # Validate calibration if we have validation data
        if 'validation_data' in locals():
            print("Validating probability calibration...")
            calibration_metrics = validate_calibration(models, validation_data, calibration_func)
            print(f"Calibration improved Brier score by {calibration_metrics['improvement']['brier_score']:.6f}")
    
    # Define a custom prediction function that applies calibration
    def predict_with_calibration(X):
        raw_prediction = ensemble_predict(models, X, calibrate=False)
        
        if raw_prediction and calibration_func:
            raw_prob = raw_prediction['team1_win_probability']
            calibrated_prob = calibration_func(raw_prob)
            
            # Update the prediction with calibrated probability
            raw_prediction['team1_win_probability'] = calibrated_prob
            raw_prediction['team2_win_probability'] = 1 - calibrated_prob
        
        return raw_prediction
    
    # Initialize tracking variables
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bet_history = []
    dates = []
    
    print(f"Starting backtest with {len(test_data)} matches...")
    
    # Iterate through test data chronologically
    for i, match in test_data.iterrows():
        if i % 10 == 0:
            print(f"Processing match {i+1}/{len(test_data)}...")
        
        try:
            # Prepare match features
            team1_name = match['team1_name']
            team2_name = match['team2_name']
            match_date = match['date']
            match_id = match['match_id']
            
            # Get model prediction
            X = match['features'].reshape(1, -1)
            prediction = predict_with_calibration(X)
            
            if not prediction:
                print(f"Failed to get prediction for match {match_id}.")
                continue
            
            # Get match odds
            match_odds = odds_data[odds_data['match_id'] == match_id].to_dict('records')
            
            if not match_odds:
                print(f"No odds data found for match {match_id}.")
                continue
                
            match_odds = match_odds[0]
            
            # Find value bets
            value_bets = find_value_bets(
                prediction, 
                match_odds['odds'], 
                format_type=match_odds['format'],
                bankroll=bankroll,
                kelly_fraction=kelly_fraction,
                min_edge=min_edge
            )
            
            # Record match data regardless of whether we bet
            match_record = {
                'match_id': match_id,
                'date': match_date,
                'team1': team1_name,
                'team2': team2_name,
                'predicted_prob': prediction['team1_win_probability'],
                'actual_outcome': match['outcome'],
                'format': match_odds['format'],
                'num_value_bets': len(value_bets)
            }
            
            # Process each value bet
            for bet_type, bet_info in value_bets.items():
                # Extract real outcome for this bet type
                bet_result = match.get(f'{bet_type}_result', 0)
                
                # Calculate stake based on Kelly criterion
                stake = bet_info['recommended_stake']
                
                # Cap stake at 5% of bankroll as a safety measure
                stake = min(stake, bankroll * 0.05)
                
                # Calculate profit/loss
                if bet_result == 1:  # Bet won
                    profit = stake * (bet_info['odds'] - 1)
                else:  # Bet lost
                    profit = -stake
                
                # Update bankroll
                bankroll += profit
                
                # Record bet details
                bet_record = {
                    'match_id': match_id,
                    'date': match_date,
                    'team1': team1_name,
                    'team2': team2_name,
                    'bet_type': bet_type,
                    'odds': bet_info['odds'],
                    'our_probability': bet_info['our_probability'],
                    'implied_probability': bet_info['implied_probability'],
                    'edge': bet_info['edge'],
                    'edge_percent': bet_info['edge_percent'],
                    'stake': stake,
                    'result': bet_result,
                    'profit': profit,
                    'bankroll_after': bankroll
                }
                
                bet_history.append(bet_record)
                bankroll_history.append(bankroll)
                dates.append(match_date)
                
                # Add bet details to match record
                match_record[f'bet_{bet_type}_odds'] = bet_info['odds']
                match_record[f'bet_{bet_type}_stake'] = stake
                match_record[f'bet_{bet_type}_profit'] = profit
        
        except Exception as e:
            print(f"Error processing match {match_id if 'match_id' in locals() else i}: {e}")
            continue
    
    # Calculate overall metrics
    if bet_history:
        bet_df = pd.DataFrame(bet_history)
        
        total_bets = len(bet_df)
        total_matches_bet = len(bet_df['match_id'].unique())
        total_matches = len(test_data)
        
        if total_bets > 0:
            total_stake = bet_df['stake'].sum()
            total_profit = bet_df['profit'].sum()
            
            overall_roi = total_profit / total_stake if total_stake > 0 else 0
            profit_per_bet = total_profit / total_bets
            profit_per_match = total_profit / total_matches_bet
            
            win_rate = (bet_df['result'] == 1).mean()
            
            # Calculate average odds
            avg_odds = bet_df['odds'].mean()
            
            # Calculate metrics by bet type
            bet_type_metrics = bet_df.groupby('bet_type').agg({
                'stake': 'sum',
                'profit': 'sum',
                'result': ['count', 'mean'],
                'odds': 'mean'
            })
            
            # Calculate ROI by bet type
            bet_type_roi = bet_type_metrics[('profit', 'sum')] / bet_type_metrics[('stake', 'sum')]
            bet_type_metrics['roi'] = bet_type_roi
            
            # Generate visualizations
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Bankroll evolution
            plt.subplot(2, 2, 1)
            plt.plot(range(len(bankroll_history)), bankroll_history, 'b-')
            plt.axhline(y=initial_bankroll, color='r', linestyle='--')
            plt.title('Bankroll Evolution')
            plt.xlabel('Bet Number')
            plt.ylabel('Bankroll ($)')
            plt.grid(True)
            
            # Plot 2: ROI by bet type
            plt.subplot(2, 2, 2)
            bet_types = bet_type_roi.index
            
            plt.bar(bet_types, bet_type_roi)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('ROI by Bet Type')
            plt.xlabel('Bet Type')
            plt.ylabel('ROI')
            plt.xticks(rotation=45)
            plt.grid(True)
            
            # Plot 3: Bet count by bet type
            plt.subplot(2, 2, 3)
            bet_counts = bet_type_metrics[('result', 'count')]
            
            plt.bar(bet_types, bet_counts)
            plt.title('Number of Bets by Type')
            plt.xlabel('Bet Type')
            plt.ylabel('Number of Bets')
            plt.xticks(rotation=45)
            plt.grid(True)
            
            # Plot 4: Win rate vs. predicted probability
            plt.subplot(2, 2, 4)
            
            # Create bins for predicted probabilities
            prob_bins = np.linspace(0, 1, 11)
            bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
            
            # Group bets by predicted probability
            bet_df['prob_bin'] = pd.cut(bet_df['our_probability'], prob_bins)
            bin_stats = bet_df.groupby('prob_bin').agg({
                'result': ['count', 'mean']
            })
            
            bin_win_rates = bin_stats[('result', 'mean')]
            bin_counts = bin_stats[('result', 'count')]
            
            # Plot win rates by probability bin
            plt.scatter(bin_centers, bin_win_rates, s=bin_counts*5, alpha=0.7)
            plt.plot([0, 1], [0, 1], 'k--')
            
            plt.title('Win Rate vs. Predicted Probability')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Actual Win Rate')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('backtest_results.png')
            plt.close()
            
            # Return backtest results
            return {
                'summary': {
                    'initial_bankroll': initial_bankroll,
                    'final_bankroll': bankroll,
                    'total_profit': total_profit,
                    'overall_roi': overall_roi,
                    'win_rate': win_rate,
                    'total_bets': total_bets,
                    'total_matches_bet': total_matches_bet,
                    'percentage_of_matches_bet': total_matches_bet / total_matches,
                    'total_stake': total_stake,
                    'avg_odds': avg_odds,
                    'profit_per_bet': profit_per_bet,
                    'profit_per_match': profit_per_match
                },
                'bet_type_metrics': bet_type_metrics.to_dict(),
                'bankroll_history': bankroll_history,
                'bet_history': bet_df.to_dict('records'),
                'final_bankroll_multiple': bankroll / initial_bankroll
            }
        else:
            return {
                'summary': {
                    'initial_bankroll': initial_bankroll,
                    'final_bankroll': initial_bankroll,
                    'total_profit': 0,
                    'overall_roi': 0,
                    'win_rate': 0,
                    'total_bets': 0,
                    'total_matches_bet': 0,
                    'percentage_of_matches_bet': 0,
                    'total_stake': 0
                },
                'error': 'No bets were placed during the backtest'
            }
    else:
        return {
            'summary': {
                'initial_bankroll': initial_bankroll,
                'final_bankroll': initial_bankroll,
                'total_profit': 0,
                'overall_roi': 0,
                'total_bets': 0
            },
            'error': 'No bet history was created during the backtest'
        }

#-------------------------------------------------------------------------
# BETTING STRATEGY APPLICATION
#-------------------------------------------------------------------------

def get_best_format_for_match(team1_stats, team2_stats):
    """
    Determine the best match format for optimal betting based on team characteristics.
    
    Args:
        team1_stats (dict): Team 1 statistics
        team2_stats (dict): Team 2 statistics
        
    Returns:
        str: Recommended format ("bo1", "bo3", "bo5")
    """
    # Calculate relevant metrics
    team1_consistency = team1_stats.get('team_consistency', 0.5)
    team2_consistency = team2_stats.get('team_consistency', 0.5)
    
    skill_gap = abs(team1_stats.get('win_rate', 0.5) - team2_stats.get('win_rate', 0.5))
    
    # Bo1 is better for underdogs since there's more variance
    # Bo3/Bo5 better for favorites since skill wins out over more maps
    
    if skill_gap > 0.2:
        # Large skill difference
        if team1_stats.get('win_rate', 0.5) > team2_stats.get('win_rate', 0.5):
            # Team 1 is significantly better
            if team1_consistency > 0.7:
                return "bo3"  # Team 1 is consistent, bet on them in bo3
            else:
                return "bo1"  # Team 1 is inconsistent, bo1 is riskier but higher odds
        else:
            # Team 2 is significantly better
            if team2_consistency < 0.7:
                return "bo1"  # Team 2 is inconsistent, opportunity for upset in bo1
            else:
                return "bo5"  # Team 2 is consistent, bo5 may offer value on underdog maps
    else:
        # Teams are close in skill
        avg_consistency = (team1_consistency + team2_consistency) / 2
        
        if avg_consistency > 0.8:
            return "bo5"  # Both teams consistent, likely to have competitive maps
        elif avg_consistency > 0.6:
            return "bo3"  # Moderate consistency, bo3 offers balanced variance
        else:
            return "bo1"  # Both teams inconsistent, high variance match

def apply_betting_strategy(team1_name, team2_name, api_options=None, bookmaker_odds=None,
                          format_type="bo3", bankroll=1000.0, kelly_fraction=0.25,
                          min_edge=0.05, confidence_threshold=0.05, max_bet_percentage=0.05,
                          models=None, scaler=None, stable_features=None, calibration_func=None):
    """
    Apply the betting strategy to a specific match.
    
    Args:
        team1_name (str): Name of team 1
        team2_name (str): Name of team 2
        api_options (dict): Options for API calls
        bookmaker_odds (dict): Odds from bookmakers
        format_type (str): Match format (bo1, bo3, bo5)
        bankroll (float): Current bankroll
        kelly_fraction (float): Conservative Kelly stake fraction
        min_edge (float): Minimum edge required for a bet
        confidence_threshold (float): Required confidence margin
        max_bet_percentage (float): Maximum percentage of bankroll to bet
        models (list): Loaded ensemble models
        scaler (object): Feature scaler
        stable_features (list): Stable feature names
        calibration_func (function): Probability calibration function
        
    Returns:
        dict: Recommended bets and analysis
    """
    # Load models if not provided
    if models is None:
        model_paths = [
            'valorant_model_fold_1.h5', 
            'valorant_model_fold_2.h5', 
            'valorant_model_fold_3.h5', 
            'valorant_model_fold_4.h5', 
            'valorant_model_fold_5.h5'
        ]
        scaler_path = 'ensemble_scaler.pkl'
        features_path = 'stable_features.pkl'
        
        models, scaler, stable_features = load_ensemble_models(model_paths, scaler_path, features_path)
    
    print(f"Analyzing match: {team1_name} vs {team2_name} ({format_type})")
    
    try:
        # Get team stats
        team1_id = get_team_id(team1_name)
        team2_id = get_team_id(team2_name)
        
        if not team1_id or not team2_id:
            return {
                'error': 'Failed to find one or both teams',
                'team1_id_found': team1_id is not None,
                'team2_id_found': team2_id is not None
            }
        
        # Fetch team stats
        team1_stats = {}
        team2_stats = {}
        
        # Load team match history
        team1_history = fetch_team_match_history(team1_id)
        team2_history = fetch_team_match_history(team2_id)
        
        if team1_history and team2_history:
            # Parse match data
            team1_matches = parse_match_data(team1_history, team1_name)
            team2_matches = parse_match_data(team2_history, team2_name)
            
            # Get player stats
            team1_player_stats = fetch_team_player_stats(team1_id)
            team2_player_stats = fetch_team_player_stats(team2_id)
            
            # Calculate team stats
            team1_stats = calculate_team_stats(team1_matches, team1_player_stats, include_economy=True)
            team2_stats = calculate_team_stats(team2_matches, team2_player_stats, include_economy=True)
            
            # Add team names
            team1_stats['team_name'] = team1_name
            team2_stats['team_name'] = team2_name
        else:
            return {
                'error': 'Failed to fetch match history for one or both teams',
                'team1_history_found': team1_history is not None,
                'team2_history_found': team2_history is not None
            }
        
        # Prepare features for prediction
        match_features = prepare_match_features(team1_stats, team2_stats, stable_features, scaler)
        
        if match_features is None:
            return {
                'error': 'Failed to prepare match features',
                'team1_stats_available': team1_stats != {},
                'team2_stats_available': team2_stats != {}
            }
        
        # Get prediction from ensemble
        prediction = ensemble_predict(models, match_features, calibrate=True, confidence=True)
        
        if not prediction:
            return {
                'error': 'Failed to get prediction from models'
            }
        
        # Apply calibration if available
        if calibration_func:
            raw_prob = prediction['team1_win_probability']
            calibrated_prob = calibration_func(raw_prob)
            
            prediction['raw_probability'] = raw_prob
            prediction['team1_win_probability'] = calibrated_prob
            prediction['team2_win_probability'] = 1 - calibrated_prob
        
        # Calculate series outcome probabilities
        series_probs = predict_series_outcome(prediction['team1_win_probability'], format_type)
        
        # If no bookmaker odds provided, give best format to bet
        if not bookmaker_odds:
            suggested_format = get_best_format_for_match(team1_stats, team2_stats)
            
            return {
                'match': {
                    'team1': team1_name,
                    'team2': team2_name,
                    'current_format': format_type,
                    'suggested_format': suggested_format
                },
                'prediction': prediction,
                'series_probabilities': series_probs,
                'message': 'No bookmaker odds provided. Please input odds to get betting recommendations.'
            }
        
        # Find value bets
        value_bets = find_value_bets(
            prediction, 
            bookmaker_odds, 
            format_type=format_type,
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            min_edge=min_edge
        )
        
        # Additional filtering based on confidence
        filtered_bets = {}
        for bet_type, bet_info in value_bets.items():
            # Check if confidence interval supports the value assessment
            confidence = bet_info.get('confidence', {})
            lower_bound = confidence.get('lower_bound', 0)
            
            implied_prob = bet_info['implied_probability']
            
            # Only recommend bet if the lower bound of our confidence interval still gives us an edge
            edge_with_confidence = lower_bound - implied_prob
            
            if edge_with_confidence >= confidence_threshold:
                # Cap bet at maximum percentage
                max_bet = bankroll * max_bet_percentage
                recommended_stake = min(bet_info['recommended_stake'], max_bet)
                
                bet_info['recommended_stake'] = recommended_stake
                bet_info['edge_with_confidence'] = edge_with_confidence
                
                filtered_bets[bet_type] = bet_info
        
        # Prepare team comparison for output
        team_comparison = []
        
        # Add win rate comparison
        team_comparison.append({
            'metric': 'Win Rate',
            'team1_value': team1_stats.get('win_rate', 0) * 100,
            'team2_value': team2_stats.get('win_rate', 0) * 100,
            'difference': (team1_stats.get('win_rate', 0) - team2_stats.get('win_rate', 0)) * 100
        })
        
        # Add recent form comparison
        team_comparison.append({
            'metric': 'Recent Form',
            'team1_value': team1_stats.get('recent_form', 0) * 100,
            'team2_value': team2_stats.get('recent_form', 0) * 100,
            'difference': (team1_stats.get('recent_form', 0) - team2_stats.get('recent_form', 0)) * 100
        })
        
        # Add player ratings comparison if available
        if 'avg_player_rating' in team1_stats and 'avg_player_rating' in team2_stats:
            team_comparison.append({
                'metric': 'Player Rating',
                'team1_value': team1_stats.get('avg_player_rating', 0),
                'team2_value': team2_stats.get('avg_player_rating', 0),
                'difference': team1_stats.get('avg_player_rating', 0) - team2_stats.get('avg_player_rating', 0)
            })
        
        # Add economy stats comparison if available
        if 'pistol_win_rate' in team1_stats and 'pistol_win_rate' in team2_stats:
            team_comparison.append({
                'metric': 'Pistol Win Rate',
                'team1_value': team1_stats.get('pistol_win_rate', 0) * 100,
                'team2_value': team2_stats.get('pistol_win_rate', 0) * 100,
                'difference': (team1_stats.get('pistol_win_rate', 0) - team2_stats.get('pistol_win_rate', 0)) * 100
            })
        
        # Return comprehensive analysis and recommendations
        return {
            'match': {
                'team1': team1_name,
                'team2': team2_name,
                'format': format_type,
            },
            'prediction': {
                'team1_win_probability': prediction['team1_win_probability'],
                'team2_win_probability': prediction['team2_win_probability'],
                'confidence_interval': prediction.get('confidence_interval', {}),
                'model_confidence': 1 - prediction.get('std_dev', 0)
            },
            'series_probabilities': series_probs,
            'team_comparison': team_comparison,
            'recommended_bets': filtered_bets,
            'all_value_bets': value_bets,
            'bankroll': bankroll,
            'max_exposure': sum(bet['recommended_stake'] for bet in filtered_bets.values())
        }
    
    except Exception as e:
        print(f"Error in apply_betting_strategy: {e}")
        return {
            'error': f'An error occurred: {str(e)}'
        }

def build_historical_dataset(team_data_collection, odds_data=None):
    """
    Build a dataset for backtesting from team data and historical odds.
    
    Args:
        team_data_collection (dict): Team statistics collection
        odds_data (DataFrame): Historical odds data
        
    Returns:
        DataFrame: Dataset for backtesting
    """
    historical_data = []
    
    for team_name, team_data in team_data_collection.items():
        matches = team_data.get('matches', [])
        
        for match in matches:
            opponent_name = match.get('opponent_name')
            
            # Skip if we don't have data for the opponent
            if opponent_name not in team_data_collection:
                continue
            
            # Get match ID
            match_id = match.get('match_id')
            if not match_id:
                continue
            
            # Get match date
            match_date = match.get('date')
            if not match_date:
                continue
            
            # Get match result (from team1's perspective)
            team1_won = match.get('team_won', False)
            
            # Get team stats
            team1_stats = team_data
            team2_stats = team_data_collection[opponent_name]
            
            # Prepare features for this match
            features = prepare_data_for_model(team1_stats, team2_stats)
            
            if not features:
                continue
                
            # Convert features to a numpy array
            X = np.array(list(features.values())).reshape(1, -1)
            
            # Get map count if available
            map_score = match.get('map_score', '')
            if map_score and ':' in map_score:
                try:
                    team1_maps, team2_maps = map_score.split(':')
                    team1_maps = int(team1_maps)
                    team2_maps = int(team2_maps)
                    
                    # Determine format (bo1, bo3, bo5) from map count
                    if team1_maps + team2_maps == 1:
                        match_format = 'bo1'
                    elif team1_maps + team2_maps <= 3:
                        match_format = 'bo3'
                    else:
                        match_format = 'bo5'
                        
                    # Calculate result types
                    if match_format == 'bo1':
                        team1_win_match = team1_maps > team2_maps
                        team2_win_match = team2_maps > team1_maps
                        
                        maps_over_0_5 = team1_maps + team2_maps > 0.5
                        
                        # Add results for different bet types
                        result_dict = {
                            'team1_win_series': 1 if team1_win_match else 0,
                            'team2_win_series': 1 if team2_win_match else 0,
                            'maps_over_0.5': 1 if maps_over_0_5 else 0,
                            'maps_under_0.5': 1 if not maps_over_0_5 else 0
                        }
                    
                    elif match_format == 'bo3':
                        team1_win_match = team1_maps > team2_maps
                        team2_win_match = team2_maps > team1_maps
                        
                        team1_win_2_0 = team1_maps == 2 and team2_maps == 0
                        team2_win_2_0 = team2_maps == 2 and team1_maps == 0
                        
                        team1_win_2_1 = team1_maps == 2 and team2_maps == 1
                        team2_win_2_1 = team2_maps == 2 and team1_maps == 1
                        
                        maps_over_2_5 = team1_maps + team2_maps > 2.5
                        maps_under_2_5 = team1_maps + team2_maps < 2.5
                        
                        maps_over_1_5 = team1_maps + team2_maps > 1.5
                        maps_under_1_5 = team1_maps + team2_maps < 1.5
                        
                        # Add results for different bet types
                        result_dict = {
                            'team1_win_series': 1 if team1_win_match else 0,
                            'team2_win_series': 1 if team2_win_match else 0,
                            'team1_win_2_0': 1 if team1_win_2_0 else 0,
                            'team2_win_2_0': 1 if team2_win_2_0 else 0,
                            'team1_win_2_1': 1 if team1_win_2_1 else 0,
                            'team2_win_2_1': 1 if team2_win_2_1 else 0,
                            'maps_over_2.5': 1 if maps_over_2_5 else 0,
                            'maps_under_2.5': 1 if maps_under_2_5 else 0,
                            'maps_over_1.5': 1 if maps_over_1_5 else 0,
                            'maps_under_1.5': 1 if maps_under_1_5 else 0
                        }
                    
                    elif match_format == 'bo5':
                        team1_win_match = team1_maps > team2_maps
                        team2_win_match = team2_maps > team1_maps
                        
                        team1_win_3_0 = team1_maps == 3 and team2_maps == 0
                        team2_win_3_0 = team2_maps == 3 and team1_maps == 0
                        
                        team1_win_3_1 = team1_maps == 3 and team2_maps == 1
                        team2_win_3_1 = team2_maps == 3 and team1_maps == 1
                        
                        team1_win_3_2 = team1_maps == 3 and team2_maps == 2
                        team2_win_3_2 = team2_maps == 3 and team1_maps == 2
                        
                        maps_over_3_5 = team1_maps + team2_maps > 3.5
                        maps_under_3_5 = team1_maps + team2_maps < 3.5
                        
                        maps_over_4_5 = team1_maps + team2_maps > 4.5
                        maps_under_4_5 = team1_maps + team2_maps < 4.5
                        
                        # Add results for different bet types
                        result_dict = {
                            'team1_win_series': 1 if team1_win_match else 0,
                            'team2_win_series': 1 if team2_win_match else 0,
                            'team1_win_3_0': 1 if team1_win_3_0 else 0,
                            'team2_win_3_0': 1 if team2_win_3_0 else 0,
                            'team1_win_3_1': 1 if team1_win_3_1 else 0,
                            'team2_win_3_1': 1 if team2_win_3_1 else 0,
                            'team1_win_3_2': 1 if team1_win_3_2 else 0,
                            'team2_win_3_2': 1 if team2_win_3_2 else 0,
                            'maps_over_3.5': 1 if maps_over_3_5 else 0,
                            'maps_under_3.5': 1 if maps_under_3_5 else 0,
                            'maps_over_4.5': 1 if maps_over_4_5 else 0,
                            'maps_under_4.5': 1 if maps_under_4_5 else 0
                        }
                    
                    else:
                        # Unknown format
                        result_dict = {
                            'team1_win_series': 1 if team1_won else 0,
                            'team2_win_series': 1 if not team1_won else 0
                        }
                    
                    # Prepare match entry for the dataset
                    match_entry = {
                        'match_id': match_id,
                        'date': match_date,
                        'team1_name': team_name,
                        'team2_name': opponent_name,
                        'features': X[0],  # Save the features as a numpy array
                        'outcome': 1 if team1_won else 0,  # 1 if team1 won, 0 if team2 won
                        'format': match_format
                    }
                    
                    # Add results for each bet type
                    for bet_type, result in result_dict.items():
                        match_entry[f'{bet_type}_result'] = result
                    
                    # Add odds if available
                    if odds_data is not None:
                        match_odds = odds_data[odds_data['match_id'] == match_id]
                        
                        if not match_odds.empty:
                            match_odds = match_odds.iloc[0]
                            
                            for bet_type in result_dict.keys():
                                if bet_type in match_odds:
                                    match_entry[f'{bet_type}_odds'] = match_odds[bet_type]
                    
                    historical_data.append(match_entry)
                    
                except (ValueError, TypeError) as e:
                    # Skip this match if there's an error parsing the map score
                    print(f"Error parsing map score for match {match_id}: {e}")
                    continue
    
    # Convert to DataFrame
    if historical_data:
        historical_df = pd.DataFrame(historical_data)
        return historical_df
    else:
        return pd.DataFrame()

def prepare_new_matches_for_prediction(upcoming_matches, team_data_collection, models, scaler, stable_features, calibration_func=None):
    """
    Prepare upcoming matches for prediction and betting.
    
    Args:
        upcoming_matches (list): List of upcoming matches
        team_data_collection (dict): Team statistics collection
        models (list): Loaded ensemble models
        scaler (object): Feature scaler
        stable_features (list): Stable feature names
        calibration_func (function): Probability calibration function
        
    Returns:
        list: Prepared matches with predictions
    """
    prepared_matches = []
    
    for match in upcoming_matches:
        try:
            team1_name = match.get('team1', {}).get('name')
            team2_name = match.get('team2', {}).get('name')
            
            match_id = match.get('id')
            match_date = match.get('date')
            
            # Skip if missing team names
            if not team1_name or not team2_name:
                print(f"Skipping match {match_id}: Missing team names")
                continue
            
            # Skip if teams are not in our data collection
            if team1_name not in team_data_collection or team2_name not in team_data_collection:
                print(f"Skipping match {match_id}: One or both teams not in data collection")
                continue
            
            # Get team stats
            team1_stats = team_data_collection[team1_name]
            team2_stats = team_data_collection[team2_name]
            
            # Prepare features
            match_features = prepare_match_features(team1_stats, team2_stats, stable_features, scaler)
            
            if match_features is None:
                print(f"Skipping match {match_id}: Failed to prepare features")
                continue
            
            # Get prediction
            prediction = ensemble_predict(models, match_features, calibrate=True, confidence=True)
            
            if not prediction:
                print(f"Skipping match {match_id}: Failed to get prediction")
                continue
            
            # Apply calibration if available
            if calibration_func:
                raw_prob = prediction['team1_win_probability']
                calibrated_prob = calibration_func(raw_prob)
                
                prediction['raw_probability'] = raw_prob
                prediction['team1_win_probability'] = calibrated_prob
                prediction['team2_win_probability'] = 1 - calibrated_prob
            
            # Determine match format
            # Default to bo3 if not specified
            format_type = match.get('format', 'bo3')
            
            # Calculate series outcome probabilities
            series_probs = predict_series_outcome(prediction['team1_win_probability'], format_type)
            
            # Prepare match entry
            match_entry = {
                'match_id': match_id,
                'date': match_date,
                'team1': team1_name,
                'team2': team2_name,
                'format': format_type,
                'prediction': prediction,
                'series_probabilities': series_probs
            }
            
            prepared_matches.append(match_entry)
            
        except Exception as e:
            print(f"Error preparing match {match.get('id', 'unknown')}: {e}")
            continue
    
    return prepared_matches

def simulate_betting_strategy(prediction_model, initial_bankroll=1000.0, num_bets=1000, 
                             win_rate=0.56, avg_odds=2.0, kelly_fraction=0.25, bankrupt_threshold=10.0,
                             plot=True):
    """
    Simulate a betting strategy over a large number of bets to assess long-term profitability.
    
    Args:
        prediction_model (object): Prediction model or function
        initial_bankroll (float): Starting bankroll
        num_bets (int): Number of bets to simulate
        win_rate (float): Average win rate (model accuracy)
        avg_odds (float): Average odds for bets
        kelly_fraction (float): Fraction of Kelly to use
        bankrupt_threshold (float): Minimum bankroll to continue betting
        plot (bool): Whether to generate plots
        
    Returns:
        dict: Simulation results
    """
    print(f"Simulating betting strategy with {num_bets} bets...")
    print(f"Parameters: Win Rate = {win_rate:.2f}, Avg Odds = {avg_odds:.2f}, Kelly Fraction = {kelly_fraction:.2f}")
    
    # Initialize simulation variables
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    
    bet_results = []
    bet_sizes = []
    
    # Track metrics
    num_wins = 0
    num_losses = 0
    total_stake = 0
    max_bankroll = bankroll
    min_bankroll = bankroll
    max_drawdown = 0
    current_drawdown = 0
    
    # Generate random outcomes based on win rate
    np.random.seed(42)  # For reproducibility
    outcomes = np.random.random(num_bets) < win_rate
    
    # Simulate each bet
    for i in range(num_bets):
        # Skip if bankroll is too low (effectively bankrupt)
        if bankroll < bankrupt_threshold:
            print(f"Simulation stopped at bet {i} due to insufficient bankroll (${bankroll:.2f})")
            break
        
        # Vary the odds slightly for realism
        odds_variation = 0.2  # Vary by +/- 20%
        bet_odds = avg_odds * (1 + (np.random.random() - 0.5) * odds_variation)
        
        # Vary the model's predicted probability
        true_prob = win_rate
        model_error = 0.05  # 5% standard error in probability estimation
        predicted_prob = true_prob + np.random.normal(0, model_error)
        predicted_prob = max(0.01, min(0.99, predicted_prob))  # Keep within valid range
        
        # Calculate Kelly stake
        kelly_stake = kelly_criterion(predicted_prob, bet_odds, bankroll, kelly_fraction)
        
        # Cap stake at 5% of bankroll as safety measure
        stake = min(kelly_stake, bankroll * 0.05)
        bet_sizes.append(stake)
        total_stake += stake
        
        # Determine outcome
        won = outcomes[i]
        
        # Update bankroll
        if won:
            profit = stake * (bet_odds - 1)
            bankroll += profit
            num_wins += 1
            current_drawdown = 0
        else:
            bankroll -= stake
            num_losses += 1
            current_drawdown += stake
        
        # Track bankroll history
        bankroll_history.append(bankroll)
        
        # Update max/min bankroll
        max_bankroll = max(max_bankroll, bankroll)
        min_bankroll = min(min_bankroll, bankroll)
        
        # Update max drawdown
        max_drawdown = max(max_drawdown, current_drawdown)
        
        # Record bet result
        bet_results.append({
            'bet_number': i + 1,
            'predicted_probability': predicted_prob,
            'true_probability': true_prob,
            'odds': bet_odds,
            'stake': stake,
            'won': won,
            'profit': profit if won else -stake,
            'bankroll': bankroll
        })
    
    # Calculate final metrics
    final_bankroll = bankroll
    total_bets = num_wins + num_losses
    
    if total_bets > 0:
        actual_win_rate = num_wins / total_bets
        roi = (final_bankroll - initial_bankroll) / total_stake if total_stake > 0 else 0
        
        # Calculate statistical significance of results
        expected_win_rate = 1 / avg_odds  # Break-even win rate for average odds
        p_value = calculate_betting_significance(actual_win_rate, expected_win_rate, total_bets)
        
        # Plot results if requested
        if plot:
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Bankroll evolution
            plt.subplot(2, 2, 1)
            plt.plot(range(len(bankroll_history)), bankroll_history, 'b-')
            plt.axhline(y=initial_bankroll, color='r', linestyle='--')
            plt.title('Bankroll Evolution')
            plt.xlabel('Bet Number')
            plt.ylabel('Bankroll ($)')
            plt.grid(True)
            
            # Plot 2: Bet size distribution
            plt.subplot(2, 2, 2)
            plt.hist(bet_sizes, bins=20, alpha=0.7)
            plt.title('Bet Size Distribution')
            plt.xlabel('Bet Size ($)')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # Plot 3: Cumulative ROI
            plt.subplot(2, 2, 3)
            
            cumulative_stake = np.cumsum([b['stake'] for b in bet_results])
            cumulative_profit = np.cumsum([b['profit'] for b in bet_results])
            cumulative_roi = cumulative_profit / cumulative_stake
            
            plt.plot(range(1, len(cumulative_roi) + 1), cumulative_roi, 'g-')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Cumulative ROI')
            plt.xlabel('Bet Number')
            plt.ylabel('ROI')
            plt.grid(True)
            
            # Plot 4: Win rate confidence interval
            plt.subplot(2, 2, 4)
            
            # Calculate confidence intervals over time
            window_size = 50
            win_rates = []
            confidence_intervals = []
            x_values = []
            
            for i in range(window_size, len(bet_results), window_size//2):
                window = bet_results[max(0, i-window_size):i]
                wins = sum(1 for b in window if b['won'])
                window_win_rate = wins / len(window)
                
                # 95% confidence interval
                ci = 1.96 * np.sqrt(window_win_rate * (1 - window_win_rate) / len(window))
                
                win_rates.append(window_win_rate)
                confidence_intervals.append(ci)
                x_values.append(i)
            
            plt.errorbar(x_values, win_rates, yerr=confidence_intervals, fmt='o-')
            plt.axhline(y=win_rate, color='r', linestyle='--', label='Target Win Rate')
            plt.axhline(y=expected_win_rate, color='g', linestyle='--', label='Break-even Win Rate')
            
            plt.title('Win Rate with 95% Confidence Interval')
            plt.xlabel('Bet Number')
            plt.ylabel('Win Rate')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('simulation_results.png')
            plt.close()
            
        # Generate summary
        return {
            'summary': {
                'initial_bankroll': initial_bankroll,
                'final_bankroll': final_bankroll,
                'profit': final_bankroll - initial_bankroll,
                'total_bets': total_bets,
                'wins': num_wins,
                'losses': num_losses,
                'actual_win_rate': actual_win_rate,
                'target_win_rate': win_rate,
                'break_even_win_rate': expected_win_rate,
                'total_stake': total_stake,
                'roi': roi,
                'max_bankroll': max_bankroll,
                'min_bankroll': min_bankroll,
                'max_drawdown': max_drawdown,
                'max_drawdown_percentage': max_drawdown / max_bankroll if max_bankroll > 0 else 0,
                'p_value': p_value,
                'statistically_significant': p_value < 0.05
            },
            'bankroll_history': bankroll_history,
            'bet_results': bet_results
        }
    else:
        return {
            'error': 'No bets were placed during simulation',
            'initial_bankroll': initial_bankroll,
            'final_bankroll': final_bankroll
        }

def calculate_betting_significance(observed_win_rate, expected_win_rate, sample_size):
    """
    Calculate the statistical significance (p-value) of betting results.
    
    Args:
        observed_win_rate (float): Actual win rate achieved
        expected_win_rate (float): Expected win rate (e.g., break-even rate)
        sample_size (int): Number of bets
        
    Returns:
        float: p-value (lower = more significant)
    """
    # Calculate the standard error of the observed win rate
    std_error = np.sqrt(expected_win_rate * (1 - expected_win_rate) / sample_size)
    
    # Calculate the z-score
    z_score = (observed_win_rate - expected_win_rate) / std_error
    
    # Calculate the p-value (one-tailed test since we care if we beat the expected win rate)
    p_value = 1 - norm.cdf(z_score)
    
    return p_value

def run_monte_carlo_simulation(team1_stats, team2_stats, format_type="bo3", num_simulations=10000,
                              models=None, scaler=None, stable_features=None, calibration_func=None):
    """
    Run Monte Carlo simulations to estimate probabilities for different match outcomes.
    
    Args:
        team1_stats (dict): Team 1 statistics
        team2_stats (dict): Team 2 statistics
        format_type (str): Match format (bo1, bo3, bo5)
        num_simulations (int): Number of simulations to run
        models (list): Trained models
        scaler (object): Feature scaler
        stable_features (list): Stable feature names
        calibration_func (function): Probability calibration function
        
    Returns:
        dict: Simulation results
    """
    print(f"Running {num_simulations} Monte Carlo simulations for {team1_stats['team_name']} vs {team2_stats['team_name']} ({format_type})")
    
    # Prepare features
    match_features = prepare_match_features(team1_stats, team2_stats, stable_features, scaler)
    
    if match_features is None:
        return {'error': 'Failed to prepare match features'}
    
    # Get base prediction
    base_prediction = ensemble_predict(models, match_features, calibrate=True, confidence=True)
    
    if not base_prediction:
        return {'error': 'Failed to get base prediction'}
    
    # Extract win probability and standard deviation
    base_prob = base_prediction['team1_win_probability']
    std_dev = base_prediction.get('std_dev', 0.1)  # Default to 0.1 if not available
    
    # Apply calibration if available
    if calibration_func:
        base_prob = calibration_func(base_prob)
    
    # Initialize simulation results
    simulation_results = {
        'team1_win_series': 0,
        'team2_win_series': 0
    }
    
    # Add format-specific outcomes
    if format_type == "bo3":
        simulation_results.update({
            'team1_win_2_0': 0,
            'team2_win_2_0': 0,
            'team1_win_2_1': 0,
            'team2_win_2_1': 0,
            'maps_over_2.5': 0,
            'maps_under_2.5': 0
        })
    elif format_type == "bo5":
        simulation_results.update({
            'team1_win_3_0': 0,
            'team2_win_3_0': 0,
            'team1_win_3_1': 0,
            'team2_win_3_1': 0,
            'team1_win_3_2': 0,
            'team2_win_3_2': 0,
            'maps_over_3.5': 0,
            'maps_over_4.5': 0,
            'maps_under_3.5': 0,
            'maps_under_4.5': 0
        })
    
    # Run simulations
    for i in range(num_simulations):
        # Add random variation to the base probability to simulate uncertainty
        simulated_prob = np.random.normal(base_prob, std_dev)
        simulated_prob = max(0.01, min(0.99, simulated_prob))  # Keep within valid range
        
        # Simulate series outcome
        if format_type == "bo1":
            # Simple coin flip for bo1
            team1_wins = np.random.random() < simulated_prob
            
            if team1_wins:
                simulation_results['team1_win_series'] += 1
            else:
                simulation_results['team2_win_series'] += 1
                
        elif format_type == "bo3":
            # Simulate up to 3 maps
            map_results = []
            
            # First map
            map_results.append(np.random.random() < simulated_prob)
            
            # Second map (with slight adjustment based on first map result)
            momentum_factor = 0.05  # Small momentum factor
            map2_prob = simulated_prob
            
            if map_results[0]:  # Team 1 won first map
                map2_prob = min(0.99, simulated_prob + momentum_factor)
            else:  # Team 2 won first map
                map2_prob = max(0.01, simulated_prob - momentum_factor)
                
            map_results.append(np.random.random() < map2_prob)
            
            # Check if we need a third map
            if map_results[0] == map_results[1]:
                # One team won both maps
                team1_wins = map_results[0]
                maps_played = 2
                
                if team1_wins:
                    simulation_results['team1_win_series'] += 1
                    simulation_results['team1_win_2_0'] += 1
                else:
                    simulation_results['team2_win_series'] += 1
                    simulation_results['team2_win_2_0'] += 1
            else:
                # Need a third map
                # Further adjust probability based on map 2 result
                map3_prob = simulated_prob
                
                if map_results[1]:  # Team 1 won second map
                    map3_prob = min(0.99, simulated_prob + momentum_factor)
                else:  # Team 2 won second map
                    map3_prob = max(0.01, simulated_prob - momentum_factor)
                
                map_results.append(np.random.random() < map3_prob)
                maps_played = 3
                
                # Determine winner
                team1_maps = sum(map_results)
                team1_wins = team1_maps > 1
                
                if team1_wins:
                    simulation_results['team1_win_series'] += 1
                    simulation_results['team1_win_2_1'] += 1
                else:
                    simulation_results['team2_win_series'] += 1
                    simulation_results['team2_win_2_1'] += 1
            
            # Update map count stats
            if maps_played > 2.5:
                simulation_results['maps_over_2.5'] += 1
            else:
                simulation_results['maps_under_2.5'] += 1
                
        elif format_type == "bo5":
            # Simulate up to 5 maps
            map_results = []
            team1_maps = 0
            team2_maps = 0
            maps_played = 0
            
            # Simulate until one team gets 3 wins
            while team1_maps < 3 and team2_maps < 3 and maps_played < 5:
                # Adjust probability based on momentum
                momentum_factor = 0.03 * (team1_maps - team2_maps)  # Momentum increases with map difference
                current_prob = max(0.01, min(0.99, simulated_prob + momentum_factor))
                
                # Simulate map
                team1_wins_map = np.random.random() < current_prob
                
                if team1_wins_map:
                    team1_maps += 1
                else:
                    team2_maps += 1
                    
                map_results.append(team1_wins_map)
                maps_played += 1
            
            # Determine series winner
            team1_wins = team1_maps > team2_maps
            
            if team1_wins:
                simulation_results['team1_win_series'] += 1
                
                if team2_maps == 0:
                    simulation_results['team1_win_3_0'] += 1
                elif team2_maps == 1:
                    simulation_results['team1_win_3_1'] += 1
                elif team2_maps == 2:
                    simulation_results['team1_win_3_2'] += 1
            else:
                simulation_results['team2_win_series'] += 1
                
                if team1_maps == 0:
                    simulation_results['team2_win_3_0'] += 1
                elif team1_maps == 1:
                    simulation_results['team2_win_3_1'] += 1
                elif team1_maps == 2:
                    simulation_results['team2_win_3_2'] += 1
            
            # Update map count stats
            if maps_played > 3.5:
                simulation_results['maps_over_3.5'] += 1
            else:
                simulation_results['maps_under_3.5'] += 1
                
            if maps_played > 4.5:
                simulation_results['maps_over_4.5'] += 1
            else:
                simulation_results['maps_under_4.5'] += 1
    
    # Calculate probabilities
    for key in simulation_results:
        simulation_results[key] = simulation_results[key] / num_simulations
    
    # Add confidence intervals
    confidence_intervals = {}
    for key, prob in simulation_results.items():
        # 95% confidence interval using normal approximation
        std_error = np.sqrt(prob * (1 - prob) / num_simulations)
        margin_of_error = 1.96 * std_error
        
        confidence_intervals[key] = {
            'lower': max(0, prob - margin_of_error),
            'upper': min(1, prob + margin_of_error),
            'std_error': std_error
        }
    
    return {
        'base_probability': base_prob,
        'simulation_results': simulation_results,
        'confidence_intervals': confidence_intervals,
        'num_simulations': num_simulations
    }

#-------------------------------------------------------------------------
# BETTING CLI INTERFACE
#-------------------------------------------------------------------------

def load_betting_models(model_paths=None, scaler_path=None, features_path=None):
    """
    Load all models and data needed for betting predictions.
    
    Args:
        model_paths (list): Paths to trained models
        scaler_path (str): Path to scaler
        features_path (str): Path to stable features
        
    Returns:
        dict: Loaded betting models and data
    """
    # Default paths if not provided
    if model_paths is None:
        model_paths = [
            'valorant_model_fold_1.h5',
            'valorant_model_fold_2.h5',
            'valorant_model_fold_3.h5',
            'valorant_model_fold_4.h5',
            'valorant_model_fold_5.h5'
        ]
    
    if scaler_path is None:
        scaler_path = 'ensemble_scaler.pkl'
    
    if features_path is None:
        features_path = 'stable_features.pkl'
    
    # Load models, scaler and features
    models, scaler, stable_features = load_ensemble_models(model_paths, scaler_path, features_path)
    
    # Check if we have all components
    if not models:
        print("Failed to load ensemble models.")
        return None
    
    if scaler is None:
        print("Warning: No feature scaler loaded. Predictions may be inaccurate.")
    
    if stable_features is None or len(stable_features) == 0:
        print("Warning: No stable features loaded. Using all features.")
    
    return {
        'models': models,
        'scaler': scaler,
        'stable_features': stable_features
    }

def parse_odds_input(odds_string):
    """
    Parse odds input string into a dictionary.
    
    Args:
        odds_string (str): String containing odds in format "bet_type:odds,bet_type:odds,..."
        
    Returns:
        dict: Parsed odds dictionary
    """
    # Example input: "team1_win_series:1.75,team2_win_series:2.05,maps_over_2.5:1.90,maps_under_2.5:1.85"
    odds_dict = {}
    
    if not odds_string:
        return odds_dict
    
    # Split by commas to get each bet type
    bet_parts = odds_string.split(',')
    
    for part in bet_parts:
        if ':' in part:
            bet_type, odds_str = part.split(':', 1)
            bet_type = bet_type.strip()
            
            try:
                odds = float(odds_str.strip())
                odds_dict[bet_type] = odds
            except ValueError:
                print(f"Warning: Could not parse odds value '{odds_str}' for bet type '{bet_type}'")
    
    return odds_dict

def betting_cli():
    """
    Command-line interface for the betting strategy.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Valorant Betting Strategy CLI")
    
    # Add command line arguments
    parser.add_argument("--predict", action="store_true", help="Predict a specific match")
    parser.add_argument("--team1", type=str, help="First team name")
    parser.add_argument("--team2", type=str, help="Second team name")
    parser.add_argument("--format", type=str, default="bo3", choices=["bo1", "bo3", "bo5"], 
                      help="Match format (bo1, bo3, bo5)")
    parser.add_argument("--odds", type=str, help="Bookmaker odds in format 'bet_type:odds,bet_type:odds,...'")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Current bankroll")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (0.0-1.0)")
    parser.add_argument("--edge", type=float, default=0.05, help="Minimum edge required for bet")
    
    parser.add_argument("--backtest", action="store_true", help="Run betting strategy backtest")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of historical matches to use")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate probabilities in backtest")
    
    parser.add_argument("--simulation", action="store_true", help="Run Monte Carlo simulation for a match")
    parser.add_argument("--num-sims", type=int, default=10000, help="Number of simulations to run")
    
    parser.add_argument("--analyze-future", action="store_true", help="Analyze all upcoming matches")
    
    args = parser.parse_args()
    
    # Load betting models
    print("Loading betting models...")
    betting_models = load_betting_models()
    
    if betting_models is None:
        print("Failed to load betting models. Exiting.")
        return
    
    models = betting_models['models']
    scaler = betting_models['scaler']
    stable_features = betting_models['stable_features']
    
    # Load team data
    print("Loading team data...")
    team_data_collection = collect_team_data(include_player_stats=True, include_economy=True)
    
    if not team_data_collection:
        print("Failed to collect team data. Exiting.")
        return
    
    print(f"Loaded data for {len(team_data_collection)} teams.")
    
    # Process command
    if args.predict and args.team1 and args.team2:
        print(f"Predicting match: {args.team1} vs {args.team2} ({args.format})")
        
        # Parse odds if provided
        bookmaker_odds = None
        if args.odds:
            bookmaker_odds = parse_odds_input(args.odds)
        
        # Apply betting strategy
        result = apply_betting_strategy(
            args.team1, 
            args.team2, 
            format_type=args.format, 
            bankroll=args.bankroll,
            kelly_fraction=args.kelly,
            min_edge=args.edge,
            bookmaker_odds=bookmaker_odds,
            models=models,
            scaler=scaler,
            stable_features=stable_features
        )
        
        # Print results
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print("\n--- MATCH PREDICTION ---")
            print(f"Teams: {result['match']['team1']} vs {result['match']['team2']}")
            print(f"Format: {result['match']['format']}")
            print(f"Prediction: {result['match']['team1']} win probability: {result['prediction']['team1_win_probability']:.2%}")
            print(f"Confidence interval: {result['prediction']['confidence_interval'].get('lower_bound', 0):.2%} - {result['prediction']['confidence_interval'].get('upper_bound', 1):.2%}")
            
            print("\n--- SERIES PROBABILITIES ---")
            for outcome, prob in result['series_probabilities'].items():
                print(f"{outcome}: {prob:.2%}")
            
            if 'recommended_bets' in result and result['recommended_bets']:
                print("\n--- RECOMMENDED BETS ---")
                for bet_type, bet_info in result['recommended_bets'].items():
                    print(f"{bet_type}:")
                    print(f"  Odds: {bet_info['odds']:.2f}")
                    print(f"  Our probability: {bet_info['our_probability']:.2%}")
                    print(f"  Implied probability: {bet_info['implied_probability']:.2%}")
                    print(f"  Edge: {bet_info['edge']:.2%} ({bet_info['edge_percent']:.2f}%)")
                    print(f"  Recommended stake: ${bet_info['recommended_stake']:.2f}")
                    print(f"  Profit if win: ${bet_info['profit_if_win']:.2f}")
            elif 'all_value_bets' in result and result['all_value_bets']:
                print("\n--- POTENTIAL VALUE BETS (BELOW CONFIDENCE THRESHOLD) ---")
                for bet_type, bet_info in result['all_value_bets'].items():
                    if bet_type not in result.get('recommended_bets', {}):
                        print(f"{bet_type}:")
                        print(f"  Odds: {bet_info['odds']:.2f}")
                        print(f"  Our probability: {bet_info['our_probability']:.2%}")
                        print(f"  Implied probability: {bet_info['implied_probability']:.2%}")
                        print(f"  Edge: {bet_info['edge']:.2%}")
            else:
                print("\nNo value bets found with the current bookmaker odds and confidence threshold.")
                
            if 'team_comparison' in result:
                print("\n--- TEAM COMPARISON ---")
                for comp in result['team_comparison']:
                    print(f"{comp['metric']}: {result['match']['team1']} {comp['team1_value']:.2f} vs {result['match']['team2']} {comp['team2_value']:.2f} (Diff: {comp['difference']:.2f})")
    
    elif args.backtest:
        print("Running betting strategy backtest...")
        
        # Build historical dataset
        historical_data = build_historical_dataset(team_data_collection)
        
        if historical_data.empty:
            print("No historical data available for backtest.")
            return
        
        # Run backtest
        backtest_results = backtest_betting_strategy(
            models,
            historical_data,
            None,  # No odds data available, will use model-derived odds
            scaler=scaler,
            stable_features=stable_features,
            initial_bankroll=args.bankroll,
            kelly_fraction=args.kelly,
            min_edge=args.edge,
            use_calibration=args.calibrate
        )
        
        # Print results
        if 'error' in backtest_results:
            print(f"Backtest error: {backtest_results['error']}")
        else:
            print("\n--- BACKTEST RESULTS ---")
            print(f"Initial bankroll: ${backtest_results['summary']['initial_bankroll']:.2f}")
            print(f"Final bankroll: ${backtest_results['summary']['final_bankroll']:.2f}")
            print(f"Total profit: ${backtest_results['summary']['total_profit']:.2f}")
            print(f"ROI: {backtest_results['summary']['overall_roi']:.2%}")
            print(f"Win rate: {backtest_results['summary']['win_rate']:.2%}")
            print(f"Total bets: {backtest_results['summary']['total_bets']}")
            print(f"Total matches bet: {backtest_results['summary']['total_matches_bet']}")
            print(f"Percentage of matches bet: {backtest_results['summary']['percentage_of_matches_bet']:.2%}")
            
            if 'bet_type_metrics' in backtest_results:
                print("\n--- BET TYPE PERFORMANCE ---")
                for bet_type, metrics in backtest_results['bet_type_metrics'].items():
                    if isinstance(metrics, dict) and ('roi' in metrics):
                        print(f"{bet_type}:")
                        print(f"  ROI: {metrics['roi']:.2%}")
                        print(f"  Bets: {metrics[('result', 'count')]}")
                        print(f"  Win rate: {metrics[('result', 'mean')]:.2%}")
                        print(f"  Avg odds: {metrics[('odds', 'mean')]:.2f}")
            
            print("\nBacktest visualization saved as 'backtest_results.png'")
    
    elif args.simulation and args.team1 and args.team2:
        print(f"Running Monte Carlo simulation for: {args.team1} vs {args.team2} ({args.format})")
        
        # Find teams in data collection
        if args.team1 not in team_data_collection:
            print(f"Team '{args.team1}' not found in data collection.")
            return
            
        if args.team2 not in team_data_collection:
            print(f"Team '{args.team2}' not found in data collection.")
            return
            
        team1_stats = team_data_collection[args.team1]
        team2_stats = team_data_collection[args.team2]
        
        # Run simulation
        simulation_results = run_monte_carlo_simulation(
            team1_stats,
            team2_stats,
            format_type=args.format,
            num_simulations=args.num_sims,
            models=models,
            scaler=scaler,
            stable_features=stable_features
        )
        
        # Print results
        if 'error' in simulation_results:
            print(f"Simulation error: {simulation_results['error']}")
        else:
            print("\n--- MONTE CARLO SIMULATION RESULTS ---")
            print(f"Base win probability for {args.team1}: {simulation_results['base_probability']:.2%}")
            print(f"Number of simulations: {simulation_results['num_simulations']}")
            
            print("\nOutcome probabilities:")
            for outcome, prob in simulation_results['simulation_results'].items():
                ci = simulation_results['confidence_intervals'][outcome]
                print(f"{outcome}: {prob:.2%} (95% CI: {ci['lower']:.2%} - {ci['upper']:.2%})")
    
    elif args.analyze_future:
        print("Analyzing upcoming matches...")
        
        # Fetch upcoming matches
        upcoming_matches = fetch_upcoming_matches()
        
        if not upcoming_matches:
            print("No upcoming matches found.")
            return
            
        # Prepare matches for prediction
        prepared_matches = prepare_new_matches_for_prediction(
            upcoming_matches,
            team_data_collection,
            models,
            scaler,
            stable_features
        )
        
        if not prepared_matches:
            print("Failed to prepare any matches for prediction.")
            return
            
        print(f"\nAnalyzed {len(prepared_matches)} upcoming matches:")
        
        # List matches by date
        sorted_matches = sorted(prepared_matches, key=lambda x: x.get('date', ''))
        
        for match in sorted_matches:
            team1 = match['team1']
            team2 = match['team2']
            format_type = match['format']
            
            team1_prob = match['prediction']['team1_win_probability']
            suggested_format = get_best_format_for_match(
                team_data_collection[team1], team_data_collection[team2]
            )
            
            print(f"\n{match.get('date', 'TBD')} - {team1} vs {team2} ({format_type})")
            print(f"Prediction: {team1} {team1_prob:.2%} vs {team2} {1-team1_prob:.2%}")
            print(f"Suggested betting format: {suggested_format}")
            
            # Show key series probabilities
            series_probs = match['series_probabilities']
            
            if format_type == 'bo3':
                print(f"Maps over 2.5: {series_probs.get('maps_over_2.5', 0):.2%}")
                print(f"{team1} win 2-0: {series_probs.get('team1_win_2_0', 0):.2%}")
                print(f"{team2} win 2-0: {series_probs.get('team2_win_2_0', 0):.2%}")
            elif format_type == 'bo5':
                print(f"Maps over 3.5: {series_probs.get('maps_over_3.5', 0):.2%}")
    
    else:
        print("No valid command specified. Use --predict, --backtest, --simulation, or --analyze-future.")
        print("For help, use --help")

#-------------------------------------------------------------------------
# ENTRY POINT
#-------------------------------------------------------------------------

if __name__ == "__main__":
    betting_cli()