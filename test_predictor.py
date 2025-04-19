# test_predictor.py
import os
import numpy as np
from valorant_predictor.predictor_system import ValorantPredictorSystem

def test_prediction():
    """Test the predictor with mock data for a specific match."""
    print("Initializing Valorant Predictor System...")
    system = ValorantPredictorSystem()
    
    # Create manual feature data for a match between two teams
    features = {
        # Basic team info (not used in prediction but included for reference)
        "team1_id": "team1",
        "team2_id": "team2",
        
        # Performance metrics
        "team1_win_rate": 0.65,
        "team2_win_rate": 0.55,
        "team1_matches_played": 50,
        "team2_matches_played": 45,
        
        # Recent performance
        "team1_win_rate_5": 0.8,  # Last 5 matches
        "team1_win_rate_10": 0.7,  # Last 10 matches
        "team1_win_rate_20": 0.65, # Last 20 matches
        "team2_win_rate_5": 0.6,
        "team2_win_rate_10": 0.5,
        "team2_win_rate_20": 0.55,
        
        # Score statistics
        "team1_avg_score": 13.2,
        "team1_avg_opp_score": 10.1,
        "team1_score_diff": 3.1,
        "team2_avg_score": 12.8,
        "team2_avg_opp_score": 11.5,
        "team2_score_diff": 1.3,
        
        # Form (weighted recent performance)
        "team1_form": 0.75,
        "team2_form": 0.6,
        
        # Map performance
        "team1_map_win_rate": 0.7,
        "team2_map_win_rate": 0.5,
        "team1_map_matches": 20,
        "team2_map_matches": 18,
        
        # Head-to-head
        "h2h_team1_wins": 3,
        "h2h_team2_wins": 1,
        "h2h_total_matches": 4,
        "h2h_team1_win_rate": 0.75,
        
        # Player statistics
        "team1_avg_acs": 240.0,
        "team1_avg_kd": 1.2,
        "team1_max_acs": 280.0,
        "team1_max_kd": 1.5,
        "team2_avg_acs": 225.0,
        "team2_avg_kd": 1.1,
        "team2_max_acs": 260.0,
        "team2_max_kd": 1.3,
        
        # Rankings and ratings
        "team1_ranking": 3,
        "team2_ranking": 8,
        "team1_rating": 85.0,
        "team2_rating": 78.0,
        
        # Comparative features
        "ranking_diff": -5,  # team1 - team2
        "rating_diff": 7.0,
        "win_rate_diff": 0.1,
        "recent_form_diff": 0.15,
        "score_diff_diff": 1.8,
        "map_win_rate_diff": 0.2
    }
    
    # If the models directory doesn't exist, create it
    os.makedirs("models", exist_ok=True)
    
    # Train a simple model with mock data
    print("Creating temporary model for testing...")
    
    # Get feature columns from the features dictionary
    feature_cols = [key for key in features.keys() if key not in ["team1_id", "team2_id"]]
    
    # Create a small mock dataset for training
    mock_data = []
    for i in range(100):
        mock_features = features.copy()
        # Add some random variation
        for key in mock_features:
            if key not in ["team1_id", "team2_id"]:
                if isinstance(mock_features[key], float):
                    mock_features[key] += np.random.normal(0, 0.1)
                elif isinstance(mock_features[key], int):
                    mock_features[key] += np.random.randint(-2, 3)
        
        # Generate win result (team with better features has higher probability)
        team1_win = 1 if np.random.random() < 0.65 else 0  # 65% chance of team1 winning
        mock_features["team1_win"] = team1_win
        mock_features["map"] = "Haven"  # Example map
        mock_data.append(mock_features)
    
    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame(mock_data)
    
    # Train model
    system.model_trainer.feature_cols = feature_cols
    system.model_trainer.train_models(df)
    
    # Predict match outcome
    print("\nPredicting match outcome...")
    prediction = system.model_trainer.predict_match(features)
    
    print(f"\nPrediction for Team 1 vs Team 2:")
    print(f"Predicted winner: {'Team 1' if prediction['predicted_winner'] == 'team1' else 'Team 2'}")
    print(f"Team 1 win probability: {prediction['team1_win_probability']:.4f}")
    print(f"Team 2 win probability: {prediction['team2_win_probability']:.4f}")
    print(f"Confidence: {prediction['confidence']:.4f} ({prediction['confidence_level']})")
    print(f"Model used: {prediction['model_used']}")
    
    # Print top feature importances
    print("\nTop features influencing the prediction:")
    for feature, importance in prediction.get('top_features', []):
        print(f"- {feature}: {importance:.4f}")
    
    return prediction

if __name__ == "__main__":
    test_prediction()