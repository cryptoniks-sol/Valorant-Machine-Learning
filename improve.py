# Learn from past mistakes
learning_results = system.learn_from_mistakes(start_date, end_date)

# Print learning insights
team_loss_patterns = learning_results.get("team_loss_patterns", [])
print("Teams most often mispredicted:")
for team, count in team_loss_patterns:
    print(f"{team}: {count} incorrect predictions")

# Tune specific model
tuning_results = system.retrain_model("random_forest", start_date, end_date)
print(f"New best parameters: {tuning_results.get('tuning_results', {}).get('best_params')}")