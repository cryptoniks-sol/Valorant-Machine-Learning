from valorant_predictor import ValorantPredictorSystem

# Create system instance
system = ValorantPredictorSystem()

# Predict a specific match
team1_id = "12345"  # Replace with actual team ID
team2_id = "67890"  # Replace with actual team ID

# Basic prediction
prediction = system.predict_specific_match(team1_id, team2_id)
print(f"Predicted winner: {prediction['predicted_winner']}")
print(f"Confidence: {prediction['confidence']:.2f}")

# Prediction with map
prediction_map = system.predict_specific_match(team1_id, team2_id, map_name="Ascent")
print(f"Predicted winner on Ascent: {prediction_map['predicted_winner']}")

# Prediction with betting odds for betting analysis
prediction_with_odds = system.predict_specific_match(
    team1_id, team2_id,
    team1_odds=1.85, team2_odds=1.95,
    handicap_odds={"team1_plus": 1.5, "team1_minus": 2.5},
    over_under_odds={"over": 1.9, "under": 1.9}
)

# Access betting recommendations
if "betting_advice" in prediction_with_odds:
    recommendations = prediction_with_odds["betting_advice"].get("recommendations", [])
    for rec in recommendations:
        print(rec["recommendation"])