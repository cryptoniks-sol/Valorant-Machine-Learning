# predict.py
from valorant_predictor.predictor_system import ValorantPredictorSystem
import os
import datetime
from datetime import timedelta

def predict_match(team1_id, team2_id):
    """
    Predict a match between two specific teams, building models from real API data if needed.
    """
    print("Initializing Valorant Predictor System...")
    system = ValorantPredictorSystem()
    
    # Check if models exist, if not, collect data and train
    models_exist = os.path.exists("models") and os.path.exists(os.path.join("models", "best_model.txt"))
    if not models_exist:
        print("No pre-trained models found. Collecting data and training models from API...")
        # Get data from the last 90 days for training
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        system.collect_and_train(start_date, end_date)
    
    print(f"Predicting match between {team1_id} and {team2_id}...")
    
    # Get prediction for specific match
    prediction = system.predict_specific_match(team1_id, team2_id)
    
    # Display prediction
    team1_name = prediction.get('team1_name', f'Team {team1_id}')
    team2_name = prediction.get('team2_name', f'Team {team2_id}')
    predicted_winner = prediction.get('predicted_winner', 'Unknown')
    winner_name = team1_name if predicted_winner == 'team1' else team2_name
    team1_win_prob = prediction.get('team1_win_probability', 0.0)
    team2_win_prob = prediction.get('team2_win_probability', 0.0)
    confidence = prediction.get('confidence', 0.0)
    confidence_level = prediction.get('confidence_level', 'Unknown')
    
    print(f"\nPrediction for {team1_name} vs {team2_name}:")
    print(f"Predicted winner: {winner_name}")
    print(f"Win probabilities: {team1_name}: {team1_win_prob:.2f}, {team2_name}: {team2_win_prob:.2f}")
    print(f"Confidence: {confidence:.2f} ({confidence_level})")
    
    # Get feature importances if available
    if 'top_features' in prediction and prediction['top_features']:
        print("\nTop features influencing the prediction:")
        for feature, importance in prediction['top_features']:
            print(f"- {feature}: {importance:.4f}")
    
    # Add match score prediction
    try:
        print("\nPredicting match score (BO3)...")
        score_prediction = system.predict_match_score(team1_id, team2_id, "bo3")
        
        print(f"Predicted score: {score_prediction['match_score']}")
        print(f"Maps: {', '.join(score_prediction['maps_to_play'])}")
        
        for i, map_pred in enumerate(score_prediction['map_predictions']):
            map_winner = team1_name if map_pred['predicted_winner'] == 'team1' else team2_name
            print(f"Map {i+1} ({map_pred['map']}): {map_winner} ({map_pred['confidence']:.2f} confidence)")
    except Exception as e:
        print(f"Could not predict match score: {e}")
    
    return prediction

if __name__ == "__main__":
    # Replace with actual team IDs from your API
    team1_id = "624"  # Change these to match your actual team IDs
    team2_id = "918"
    predict_match(team1_id, team2_id)