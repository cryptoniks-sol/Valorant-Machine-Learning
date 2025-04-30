# Enhanced Valorant Match Prediction System

An advanced machine learning system for predicting the outcomes of Valorant esports matches with high accuracy. This system integrates KD ratios and other performance metrics, uses deep learning models, provides comprehensive visualizations, and features continuous learning from match outcomes.

## 🔥 Key Features

- **Advanced Prediction Models**: Multiple ML models including Random Forest, Ensemble, and Deep Neural Networks
- **Performance Metrics Integration**: Analyzes KD ratios, ACS, ADR, first bloods and more from the API
- **Symmetrical Feature Engineering**: Balanced approach to team comparisons for unbiased predictions
- **Continuous Learning**: System improves by learning from prediction outcomes
- **Comprehensive Backtesting**: Evaluate model performance against historical match data
- **Detailed Visualizations**: Visual representations of predictions and model performance
- **Retrainable Models**: Easily update models with new match data
- **Command-line and Interactive Interfaces**: Flexible usage options

## 📋 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/valorant-predictor.git
cd valorant-predictor
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the Valorant API server as described in the API documentation.

## 🚀 Usage

### Command Line Interface

The system provides a command-line interface for various operations:

#### Predict a Match

```bash
python predictor.py predict --team1 "Team Liquid" --team2 "Fnatic" --team1-region "eu" --team2-region "eu" --visualize --lan
```

#### Train a Model

```bash
python predictor.py train --num-teams 20 --method advanced --save
```

Available methods: `basic`, `ensemble`, `advanced`, `deep`

#### Backtest Model

```bash
python predictor.py backtest --num-teams 20 --min-date "2023-01-01" --max-date "2023-12-31" --visualize
```

#### Load a Specific Model

```bash
python predictor.py load --model-prefix test1_advanced_20240418_123456
```

### Interactive Mode

Run the script without arguments to enter interactive mode:

```bash
python maintest.py
```

This will guide you through the available options with prompts.

## 📊 Model Performance

The system includes several model types, each with different performance characteristics:

- **Basic**: Simple Random Forest model - fastest to train, good baseline performance
- **Ensemble**: Combines multiple models for improved accuracy
- **Advanced**: Optimized ML pipeline with hyperparameter tuning and feature selection
- **Deep Learning**: Neural network model that can capture complex patterns (requires more data)

The advanced model typically achieves 65-75% prediction accuracy on professional matches.

## 📈 Feature Importance

The system analyzes numerous factors to make predictions, including:

1. Team win rates and recent performance trends
2. Head-to-head history between teams
3. Player KD ratios and ACS statistics
4. Map-specific performance metrics
5. Performance against top-ranked opponents
6. LAN vs online performance differences
7. Team role balance and composition metrics

## 📁 Directory Structure

```
valorant-predictor/
├── models/             # Saved prediction models
├── data/               # Dataset storage
│   └── historical_matches/  # Historical match data for backtesting
├── predictions/        # Saved match predictions
│   ├── evaluations/    # Prediction evaluations
│   └── visualizations/ # Prediction visualizations
├── backtests/          # Backtest results
│   └── visualizations/ # Backtest visualizations
├── logs/               # System logs
└── requirements.txt    # Required dependencies
```

## 🔄 Continuous Learning

The system can learn from its predictions by recording actual match outcomes and retraining models. Use the following workflow:

1. Make a prediction before a match
2. After the match, provide the actual outcome:
   ```python
   # In code
   system.learn_from_outcome(prediction_result, actual_winner)
   ```
3. Periodically retrain the model to incorporate new learnings

## 🌐 API Integration

The system interfaces with a local Valorant API server that provides:

- Team and player data
- Match history and statistics
- KD ratios and performance metrics

Make sure your API server is running at the specified URL before using the system.

## 📊 Visualization Examples

The system generates several types of visualizations:

- **Prediction Visualization**: Win probability, team comparison, key differentials
- **Backtest Results**: Accuracy by confidence level, prediction distribution
- **Feature Importance**: Top factors influencing predictions
- **Performance Trends**: Tracking model improvement over time

## 🛠️ Customization

You can customize various aspects of the system:

- **API URL**: Change the `API_URL` constant in the code
- **Model Parameters**: Adjust hyperparameters in the model training functions
- **Feature Engineering**: Modify the `FeatureEngineering` class to add new features

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The Valorant API server implementation
- scikit-learn, TensorFlow, and pandas libraries
- The esports analytics community

---

For questions or issues, please open an issue on the GitHub repository or contact the maintainer.

python test1.py --predict --team1 "T1 Academy" --team2 "SLT"

python test1.py --train --players --economy --learning-curves

python test1.py --train --players --economy --maps --cross-validate --folds 10 --learning-curves

python test1.py --backtest --cutoff-date 2025/01/01 --bet-amount 100 

python betscache.py betting_predictions/SHERPA_vs_FN_Pocheon_20250423_225441.json --bankroll 60

python bets.py betting_predictions/Zero_Tenacity_vs_PIXEL_LUMINA_20250421_160309.json --bankroll 100

Basic Training
bash# Train a standard model with 30 teams
python test1.py --train --players --economy --test-teams "TSM" "Cloud9" "100 Thieves" "LOUD" "Fnatic" "Team Liquid" "G2 Esports" "NRG" "Evil Geniuses" "KRÜ Esports" "FURIA" "MIBR" "Team Heretics" "FUT Esports" "Gen.G" "DRX" "ZETA DIVISION" "DetonatioN" "Paper Rex" "T1" "NAVI" "FPX" "BBL Esports" "Team Secret" "Talon Esports" "Global Esports" "Team Vitality" "Karmine Corp" "Leviatán" "Team Liquid Brazil"

# Train with top 100 teams automatically
python test1.py --train --players --economy
Advanced Training with Optimization
bash# Train with learning curves for overfitting diagnosis
python test1.py --train --players --economy --learning-curves

# Run complete optimization pipeline
python test1.py --train --players --economy --optimize

# Train with cross-validation and ensemble modeling (5 folds)

python test1.py --train --cross-validate --folds 5 --players --economy --maps --learning-curves 

# Train with cross-validation and 10 folds

python test1.py --train --players --economy --maps --cross-validate --folds 10 --learning-curves

Match Prediction
bash# Predict a specific match (will automatically use the best available model)
python ml.py --predict --team1 "Nightblood Gaming" --team2 "Funhavers"

Analysis and Evaluation
bash# Analyze all upcoming matches
python test1.py --analyze

# Run backtesting
python test1.py --backtest --cutoff-date 2023/06/01 --confidence 0.60 --bet-amount 100

# Generate learning curves for an existing model
python test1.py --learning-curves

# Optimize an existing model
python test1.py --optimize
Additional Options
bash# Show detailed progress during operations
python test1.py --train --players --economy --verbose --optimize

# Use specific test teams
python test1.py --train --players --economy --test-teams "Team1" "Team2"


Cross-validation training is likely your best option:
python your_script.py --train --cross-validate --folds 5
This creates an ensemble of models which should be more robust than a single model.
Include relevant data sources based on your needs:

For most accurate but potentially slowest training: include all data
python ml.py --train --cross-validate --folds 5 --players --economy --maps

For balanced approach: use player stats and economy (this appears to be the default)
python your_script.py --train --players --economy

For fastest training: use minimal data
python your_script.py --train --players



Optimize your model by testing different confidence thresholds:
python ml.py --backtest --cutoff-date 2023/06/01 --confidence 0.7
Try different confidence values (0.6, 0.7, 0.8) to find the optimal balance between accuracy and number of bets.
Find the right dataset size by adjusting team limits:
python your_script.py --train --team-limit 50
A smaller team limit might train faster but with less data, while a larger limit provides more comprehensive data.