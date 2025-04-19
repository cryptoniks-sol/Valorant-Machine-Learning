from valorant_predictor import ValorantPredictorSystem
import pandas as pd
from datetime import datetime
import json

def predict_all_upcoming_matches():
    """
    Predict all upcoming Valorant matches and save results to a CSV and JSON file.
    """
    print("Initializing Valorant Predictor System...")
    system = ValorantPredictorSystem()
    
    print("Fetching upcoming matches...")
    upcoming_predictions = system.predict_upcoming_matches()
    
    if not upcoming_predictions:
        print("No upcoming matches found.")
        return
    
    print(f"Found {len(upcoming_predictions)} upcoming matches.")
    
    # Create a list to store prediction data
    prediction_data = []
    
    for prediction in upcoming_predictions:
        team1_name = prediction.get('team1_name', 'Unknown')
        team2_name = prediction.get('team2_name', 'Unknown')
        predicted_winner = prediction.get('predicted_winner', 'Unknown')
        winner_name = team1_name if predicted_winner == 'team1' else team2_name
        confidence = prediction.get('confidence', 0.0)
        confidence_level = prediction.get('confidence_level', 'Unknown')
        team1_win_prob = prediction.get('team1_win_probability', 0.0)
        team2_win_prob = prediction.get('team2_win_probability', 0.0)
        event = prediction.get('event', 'Unknown')
        match_date = prediction.get('date', 'Unknown')
        
        # Print prediction summary
        print(f"\n{team1_name} vs {team2_name}")
        print(f"Event: {event}, Date: {match_date}")
        print(f"Predicted winner: {winner_name}")
        print(f"Win probabilities: {team1_name}: {team1_win_prob:.2f}, {team2_name}: {team2_win_prob:.2f}")
        print(f"Confidence: {confidence:.2f} ({confidence_level})")
        
        # Append to data list
        prediction_data.append({
            'match_id': prediction.get('match_id', ''),
            'team1_id': prediction.get('team1_id', ''),
            'team2_id': prediction.get('team2_id', ''),
            'team1_name': team1_name,
            'team2_name': team2_name,
            'event': event,
            'date': match_date,
            'predicted_winner': predicted_winner,
            'winner_name': winner_name,
            'team1_win_probability': team1_win_prob,
            'team2_win_probability': team2_win_prob,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'model_used': prediction.get('model_used', 'Unknown')
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(prediction_data)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"valorant_predictions_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved predictions to {csv_filename}")
    
    # Save detailed predictions to JSON
    json_filename = f"valorant_predictions_detailed_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(upcoming_predictions, f, indent=4)
    print(f"Saved detailed predictions to {json_filename}")
    
    # Print overall statistics
    print("\nPrediction Statistics:")
    print(f"Total matches: {len(prediction_data)}")
    
    confidence_groups = df.groupby('confidence_level').size()
    print("\nConfidence level distribution:")
    for level, count in confidence_groups.items():
        print(f"{level}: {count} matches")
    
    # Print closest matches (around 50-50 odds)
    closest_matches = df.iloc[(df['confidence'] - 0.5).abs().argsort()[:3]]
    print("\nClosest matches (near 50-50 odds):")
    for _, match in closest_matches.iterrows():
        print(f"{match['team1_name']} vs {match['team2_name']}: {match['team1_win_probability']:.2f} - {match['team2_win_probability']:.2f}")
    
    # Print most confident predictions
    most_confident = df.sort_values('confidence', ascending=False).head(3)
    print("\nMost confident predictions:")
    for _, match in most_confident.iterrows():
        print(f"{match['winner_name']} to win against {match['team1_name'] if match['winner_name'] == match['team2_name'] else match['team2_name']} (Confidence: {match['confidence']:.2f})")
    
    return df, upcoming_predictions

# Execute the function if run directly
if __name__ == "__main__":
    predict_all_upcoming_matches()