import re
import os
import shutil
from datetime import datetime

def apply_realistic_betting_patches():
    """
    Automatically patch the existing valorant predictor script with realistic betting functions.
    """
    
    # Backup original file
    original_file = "paste.txt"  # Your current script name
    backup_file = f"paste_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    if not os.path.exists(original_file):
        print(f"Error: {original_file} not found!")
        return False
    
    # Create backup
    shutil.copy2(original_file, backup_file)
    print(f"Created backup: {backup_file}")
    
    # Read original file
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define all the function replacements
    function_patches = {
        'predict_with_ensemble': '''def predict_with_ensemble(ensemble_models, X):
    if not ensemble_models:
        raise ValueError("No models provided for prediction")
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    raw_predictions = []
    model_weights = []
    model_types = []
    for i, (model_type, model, model_scaler) in enumerate(ensemble_models):
        try:
            X_pred = X.copy()
            if model_scaler is not None:
                try:
                    X_pred = model_scaler.transform(X_pred)
                except Exception as e:
                    pass
            if model_type == 'nn':
                raw_pred = model.predict(X_pred, verbose=0)[0][0]
                raw_pred = min(0.75, max(0.25, raw_pred))
                calibrated_pred = 0.5 + (raw_pred - 0.5) * 0.4
                pred = calibrated_pred
            else:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_pred)[0][1]
                    pred = min(0.75, max(0.25, pred))
                    if model_type in ['gb', 'rf']:
                        pred = 0.5 + (pred - 0.5) * 0.35
                    else:
                        pred = 0.5 + (pred - 0.5) * 0.3
                else:
                    pred = model.predict(X_pred)[0]
                    pred = min(0.75, max(0.25, pred))
            if np.isnan(pred) or not np.isfinite(pred):
                pred = 0.5
            if model_type == 'nn':
                base_weight = 0.6
            elif model_type == 'gb':
                base_weight = 0.8
            elif model_type == 'rf':
                base_weight = 0.8
            else:
                base_weight = 0.4
            extremeness = abs(pred - 0.5) / 0.5
            weight = base_weight * (1.0 - extremeness * 0.3)
            raw_predictions.append(pred)
            model_weights.append(weight)
            model_types.append(model_type)
        except Exception as e:
            continue
    if not raw_predictions:
        return 0.5, [0.5], 0.0
    if raw_predictions and sum(model_weights) > 0:
        weighted_sum = sum(p * w for p, w in zip(raw_predictions, model_weights))
        total_weight = sum(model_weights)
        mean_pred = weighted_sum / total_weight
    else:
        mean_pred = 0.5
    std_pred = np.std(raw_predictions)
    disagreement_penalty = min(0.4, std_pred * 3)
    raw_confidence = 1 - min(0.8, std_pred * 2)
    adjusted_confidence = max(0.1, raw_confidence - disagreement_penalty)
    final_pred = 0.5 + (mean_pred - 0.5) * 0.6
    final_pred = min(0.65, max(0.35, final_pred))
    raw_predictions_str = [f'{p:.4f}' for p in raw_predictions]
    return final_pred, raw_predictions_str, adjusted_confidence''',

        'analyze_betting_edge_for_backtesting': '''def analyze_betting_edge_for_backtesting(team1_win_prob, team2_win_prob, odds_data, confidence_score, bankroll=1000.0):
    betting_analysis = {}
    if not (0 < team1_win_prob < 1) or not (0 < team2_win_prob < 1):
        team1_win_prob = min(0.65, max(0.35, team1_win_prob))
        team2_win_prob = 1 - team1_win_prob
    if abs(team1_win_prob + team2_win_prob - 1) > 0.001:
        total = team1_win_prob + team2_win_prob
        team1_win_prob = team1_win_prob / total
        team2_win_prob = team2_win_prob / total
    base_min_edge = 0.015
    confidence_factor = 0.8 + (0.2 * confidence_score)
    adjusted_threshold = base_min_edge / confidence_factor
    map_scale = 0.6 + (confidence_score * 0.1)
    single_map_prob = 0.5 + (team1_win_prob - 0.5) * map_scale
    single_map_prob = max(0.35, min(0.65, single_map_prob))
    team1_plus_prob = 1 - (1 - single_map_prob) ** 2