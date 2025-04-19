import datetime
from datetime import timedelta

# Define date range for backtesting
end_date = datetime.datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

# Run backtest
backtest_results = system.backtest_model(start_date, end_date)

# Print overall metrics
metrics = backtest_results.get("overall_metrics", {})
print(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
print(f"Precision: {metrics.get('precision', 0.0):.4f}")

# Print accuracy by confidence level
conf_acc = backtest_results.get("confidence_accuracy", {})
for bin_name, data in conf_acc.items():
    print(f"{bin_name}: {data.get('accuracy', 0.0):.4f} ({data.get('match_count', 0)} matches)")