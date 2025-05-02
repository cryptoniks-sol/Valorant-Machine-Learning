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
python test.py --predict --team1 "Akave Esports GC" --team2 "MYVRA GC"

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









Epoch 42/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7152 - loss: 0.5848 - val_accuracy: 0.7065 - val_loss: 0.5877
Epoch 43/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7281 - loss: 0.5825 - val_accuracy: 0.7038 - val_loss: 0.5869
Epoch 44/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7340 - loss: 0.5864 - val_accuracy: 0.7038 - val_loss: 0.5840
Epoch 45/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7139 - loss: 0.5806 - val_accuracy: 0.6929 - val_loss: 0.5829
Epoch 46/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7205 - loss: 0.5753 - val_accuracy: 0.6984 - val_loss: 0.5857
Epoch 47/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7234 - loss: 0.5798 - val_accuracy: 0.7011 - val_loss: 0.5848
Epoch 48/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7209 - loss: 0.5702 - val_accuracy: 0.6984 - val_loss: 0.5755
Epoch 49/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7269 - loss: 0.5667 - val_accuracy: 0.6929 - val_loss: 0.5780
Epoch 50/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7209 - loss: 0.5753 - val_accuracy: 0.7065 - val_loss: 0.5729
Epoch 51/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7329 - loss: 0.5696 - val_accuracy: 0.7011 - val_loss: 0.5760
Epoch 52/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7090 - loss: 0.5696 - val_accuracy: 0.7011 - val_loss: 0.5759
Epoch 53/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7138 - loss: 0.5686 - val_accuracy: 0.7120 - val_loss: 0.5723
Epoch 54/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7180 - loss: 0.5632 - val_accuracy: 0.7120 - val_loss: 0.5652
Epoch 55/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7260 - loss: 0.5542 - val_accuracy: 0.7120 - val_loss: 0.5631
Epoch 56/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7266 - loss: 0.5504 - val_accuracy: 0.7120 - val_loss: 0.5619
Epoch 57/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7272 - loss: 0.5509 - val_accuracy: 0.7092 - val_loss: 0.5629
Epoch 58/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7109 - loss: 0.5686 - val_accuracy: 0.7065 - val_loss: 0.5591
Epoch 59/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7310 - loss: 0.5539 - val_accuracy: 0.7038 - val_loss: 0.5641
Epoch 60/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7344 - loss: 0.5366 - val_accuracy: 0.7065 - val_loss: 0.5681
Epoch 61/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7274 - loss: 0.5491 - val_accuracy: 0.7092 - val_loss: 0.5574
Epoch 62/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7257 - loss: 0.5434 - val_accuracy: 0.7038 - val_loss: 0.5606
Epoch 63/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7308 - loss: 0.5371 - val_accuracy: 0.6984 - val_loss: 0.5583
Epoch 64/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7222 - loss: 0.5453 - val_accuracy: 0.7120 - val_loss: 0.5521
Epoch 65/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7349 - loss: 0.5361 - val_accuracy: 0.7065 - val_loss: 0.5473
Epoch 66/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7219 - loss: 0.5450 - val_accuracy: 0.7065 - val_loss: 0.5443
Epoch 67/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7246 - loss: 0.5439 - val_accuracy: 0.7038 - val_loss: 0.5487
Epoch 68/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7225 - loss: 0.5407 - val_accuracy: 0.7120 - val_loss: 0.5427
Epoch 69/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7229 - loss: 0.5434 - val_accuracy: 0.7120 - val_loss: 0.5443
Epoch 70/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7296 - loss: 0.5389 - val_accuracy: 0.7065 - val_loss: 0.5491
Epoch 71/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7213 - loss: 0.5362 - val_accuracy: 0.7147 - val_loss: 0.5404
Epoch 72/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7269 - loss: 0.5268 - val_accuracy: 0.7038 - val_loss: 0.5374
Epoch 73/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7129 - loss: 0.5382 - val_accuracy: 0.7092 - val_loss: 0.5434
Epoch 74/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7223 - loss: 0.5315 - val_accuracy: 0.7174 - val_loss: 0.5453
Epoch 75/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7222 - loss: 0.5322 - val_accuracy: 0.6929 - val_loss: 0.5441
Epoch 76/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7158 - loss: 0.5267 - val_accuracy: 0.7120 - val_loss: 0.5362
Epoch 77/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7328 - loss: 0.5195 - val_accuracy: 0.6957 - val_loss: 0.5377
Epoch 78/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7346 - loss: 0.5247 - val_accuracy: 0.7038 - val_loss: 0.5396
Epoch 79/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7339 - loss: 0.5212 - val_accuracy: 0.7092 - val_loss: 0.5367
Epoch 80/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7357 - loss: 0.5190 - val_accuracy: 0.7038 - val_loss: 0.5356
Epoch 81/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7383 - loss: 0.5101 - val_accuracy: 0.6902 - val_loss: 0.5466
Epoch 82/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7112 - loss: 0.5321 - val_accuracy: 0.6984 - val_loss: 0.5396
Epoch 83/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7122 - loss: 0.5336 - val_accuracy: 0.7120 - val_loss: 0.5289
Epoch 84/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7342 - loss: 0.5171 - val_accuracy: 0.6902 - val_loss: 0.5314
Epoch 85/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7163 - loss: 0.5359 - val_accuracy: 0.7065 - val_loss: 0.5355
Epoch 86/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7311 - loss: 0.5126 - val_accuracy: 0.7120 - val_loss: 0.5338
Epoch 87/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7257 - loss: 0.5197 - val_accuracy: 0.7147 - val_loss: 0.5259
Epoch 88/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7112 - loss: 0.5266 - val_accuracy: 0.7147 - val_loss: 0.5328
Epoch 89/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7147 - loss: 0.5237 - val_accuracy: 0.7174 - val_loss: 0.5334
Epoch 90/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7134 - loss: 0.5328 - val_accuracy: 0.7201 - val_loss: 0.5270
Epoch 91/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7307 - loss: 0.5073 - val_accuracy: 0.7120 - val_loss: 0.5197
Epoch 92/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7275 - loss: 0.5204 - val_accuracy: 0.7147 - val_loss: 0.5270
Epoch 93/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7229 - loss: 0.5185 - val_accuracy: 0.7065 - val_loss: 0.5308
Epoch 94/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7190 - loss: 0.5430 - val_accuracy: 0.7038 - val_loss: 0.5339
Epoch 95/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7159 - loss: 0.5251 - val_accuracy: 0.7092 - val_loss: 0.5281
Epoch 96/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7291 - loss: 0.5237 - val_accuracy: 0.7147 - val_loss: 0.5190
Epoch 97/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7166 - loss: 0.5130 - val_accuracy: 0.7092 - val_loss: 0.5242
Epoch 98/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7381 - loss: 0.5048 - val_accuracy: 0.7120 - val_loss: 0.5163
Epoch 99/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7126 - loss: 0.5204 - val_accuracy: 0.7092 - val_loss: 0.5180
Epoch 100/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7179 - loss: 0.5302 - val_accuracy: 0.7147 - val_loss: 0.5298
Restoring model weights from the end of the best epoch: 98.
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
Fold 6 Results:
  Accuracy: 0.7120
  Precision: 0.7200
  Recall: 0.6885
  F1 Score: 0.7039
  AUC: 0.8110
  Selected Features: 39

----- Training Fold 7/10 -----
Epoch 1/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.5921 - loss: 1.0105 - val_accuracy: 0.7038 - val_loss: 0.7510
Epoch 2/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6849 - loss: 0.8078 - val_accuracy: 0.7092 - val_loss: 0.7240
Epoch 3/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6890 - loss: 0.8007 - val_accuracy: 0.7065 - val_loss: 0.7200
Epoch 4/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6864 - loss: 0.7744 - val_accuracy: 0.7092 - val_loss: 0.7114
Epoch 5/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7028 - loss: 0.7274 - val_accuracy: 0.6984 - val_loss: 0.7084
Epoch 6/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7088 - loss: 0.7306 - val_accuracy: 0.6984 - val_loss: 0.7026
Epoch 7/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7120 - loss: 0.7241 - val_accuracy: 0.7065 - val_loss: 0.6957
Epoch 8/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7018 - loss: 0.7201 - val_accuracy: 0.7038 - val_loss: 0.6919
Epoch 9/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7027 - loss: 0.7049 - val_accuracy: 0.6957 - val_loss: 0.6929
Epoch 10/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7023 - loss: 0.7127 - val_accuracy: 0.6957 - val_loss: 0.6915
Epoch 11/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6946 - loss: 0.7018 - val_accuracy: 0.6902 - val_loss: 0.6841
Epoch 12/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7274 - loss: 0.6679 - val_accuracy: 0.7011 - val_loss: 0.6837
Epoch 13/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7191 - loss: 0.6816 - val_accuracy: 0.6984 - val_loss: 0.6777
Epoch 14/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7158 - loss: 0.6826 - val_accuracy: 0.6929 - val_loss: 0.6773
Epoch 15/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6998 - loss: 0.6913 - val_accuracy: 0.7092 - val_loss: 0.6713
Epoch 16/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7353 - loss: 0.6730 - val_accuracy: 0.7011 - val_loss: 0.6705
Epoch 17/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7141 - loss: 0.6596 - val_accuracy: 0.6984 - val_loss: 0.6700
Epoch 18/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7167 - loss: 0.6643 - val_accuracy: 0.7038 - val_loss: 0.6637
Epoch 19/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7193 - loss: 0.6749 - val_accuracy: 0.7011 - val_loss: 0.6649
Epoch 20/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7171 - loss: 0.6614 - val_accuracy: 0.6984 - val_loss: 0.6587
Epoch 21/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7248 - loss: 0.6618 - val_accuracy: 0.7120 - val_loss: 0.6553
Epoch 22/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7183 - loss: 0.6398 - val_accuracy: 0.7092 - val_loss: 0.6485
Epoch 23/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7303 - loss: 0.6451 - val_accuracy: 0.7038 - val_loss: 0.6526
Epoch 24/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7178 - loss: 0.6445 - val_accuracy: 0.7011 - val_loss: 0.6446
Epoch 25/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7216 - loss: 0.6422 - val_accuracy: 0.7038 - val_loss: 0.6437
Epoch 26/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7225 - loss: 0.6436 - val_accuracy: 0.7092 - val_loss: 0.6494
Epoch 27/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7125 - loss: 0.6417 - val_accuracy: 0.6957 - val_loss: 0.6461
Epoch 28/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7147 - loss: 0.6346 - val_accuracy: 0.6984 - val_loss: 0.6397
Epoch 29/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7107 - loss: 0.6405 - val_accuracy: 0.7038 - val_loss: 0.6338
Epoch 30/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7326 - loss: 0.6270 - val_accuracy: 0.6984 - val_loss: 0.6328
Epoch 31/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7158 - loss: 0.6302 - val_accuracy: 0.7120 - val_loss: 0.6289
Epoch 32/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7357 - loss: 0.6118 - val_accuracy: 0.7065 - val_loss: 0.6259
Epoch 33/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7234 - loss: 0.6050 - val_accuracy: 0.7065 - val_loss: 0.6258
Epoch 34/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7243 - loss: 0.6169 - val_accuracy: 0.6984 - val_loss: 0.6241
Epoch 35/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7225 - loss: 0.6130 - val_accuracy: 0.6957 - val_loss: 0.6223
Epoch 36/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7395 - loss: 0.6061 - val_accuracy: 0.6929 - val_loss: 0.6182
Epoch 37/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7213 - loss: 0.6080 - val_accuracy: 0.7038 - val_loss: 0.6170
Epoch 38/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7127 - loss: 0.6017 - val_accuracy: 0.7038 - val_loss: 0.6157
Epoch 39/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7266 - loss: 0.6052 - val_accuracy: 0.7011 - val_loss: 0.6197
Epoch 40/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7208 - loss: 0.6063 - val_accuracy: 0.7092 - val_loss: 0.6146
Epoch 41/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7251 - loss: 0.5958 - val_accuracy: 0.7038 - val_loss: 0.6093
Epoch 42/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7291 - loss: 0.5852 - val_accuracy: 0.7065 - val_loss: 0.6061
Epoch 43/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7194 - loss: 0.5881 - val_accuracy: 0.7065 - val_loss: 0.6030
Epoch 44/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7160 - loss: 0.5801 - val_accuracy: 0.7011 - val_loss: 0.6026
Epoch 45/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7284 - loss: 0.5726 - val_accuracy: 0.7038 - val_loss: 0.5990
Epoch 46/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7291 - loss: 0.5716 - val_accuracy: 0.7038 - val_loss: 0.5959
Epoch 47/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7166 - loss: 0.5837 - val_accuracy: 0.6984 - val_loss: 0.5926
Epoch 48/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7437 - loss: 0.5642 - val_accuracy: 0.7011 - val_loss: 0.5968
Epoch 49/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7271 - loss: 0.5789 - val_accuracy: 0.6957 - val_loss: 0.5905
Epoch 50/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7330 - loss: 0.5707 - val_accuracy: 0.6957 - val_loss: 0.5875
Epoch 51/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7297 - loss: 0.5681 - val_accuracy: 0.6984 - val_loss: 0.5869
Epoch 52/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7372 - loss: 0.5626 - val_accuracy: 0.7038 - val_loss: 0.5850
Epoch 53/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7272 - loss: 0.5585 - val_accuracy: 0.6902 - val_loss: 0.5841
Epoch 54/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7283 - loss: 0.5534 - val_accuracy: 0.7011 - val_loss: 0.5839
Epoch 55/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7244 - loss: 0.5682 - val_accuracy: 0.6929 - val_loss: 0.5760
Epoch 56/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7297 - loss: 0.5644 - val_accuracy: 0.6984 - val_loss: 0.5696
Epoch 57/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7158 - loss: 0.5550 - val_accuracy: 0.6902 - val_loss: 0.5774
Epoch 58/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7175 - loss: 0.5593 - val_accuracy: 0.7065 - val_loss: 0.5749
Epoch 59/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7117 - loss: 0.5588 - val_accuracy: 0.7011 - val_loss: 0.5667
Epoch 60/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7313 - loss: 0.5377 - val_accuracy: 0.7065 - val_loss: 0.5651
Epoch 61/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7326 - loss: 0.5521 - val_accuracy: 0.7092 - val_loss: 0.5633
Epoch 62/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7395 - loss: 0.5334 - val_accuracy: 0.7038 - val_loss: 0.5616
Epoch 63/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7315 - loss: 0.5523 - val_accuracy: 0.7092 - val_loss: 0.5634
Epoch 64/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7215 - loss: 0.5502 - val_accuracy: 0.7092 - val_loss: 0.5549
Epoch 65/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7074 - loss: 0.5463 - val_accuracy: 0.6957 - val_loss: 0.5624
Epoch 66/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7306 - loss: 0.5526 - val_accuracy: 0.7065 - val_loss: 0.5561
Epoch 67/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7209 - loss: 0.5436 - val_accuracy: 0.6957 - val_loss: 0.5547
Epoch 68/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7215 - loss: 0.5456 - val_accuracy: 0.7038 - val_loss: 0.5561
Epoch 69/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7222 - loss: 0.5490 - val_accuracy: 0.7011 - val_loss: 0.5539
Epoch 70/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7444 - loss: 0.5402 - val_accuracy: 0.7038 - val_loss: 0.5617
Epoch 71/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7304 - loss: 0.5305 - val_accuracy: 0.6929 - val_loss: 0.5504
Epoch 72/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7236 - loss: 0.5398 - val_accuracy: 0.7092 - val_loss: 0.5477
Epoch 73/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7212 - loss: 0.5432 - val_accuracy: 0.7092 - val_loss: 0.5495
Epoch 74/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7172 - loss: 0.5470 - val_accuracy: 0.7120 - val_loss: 0.5472
Epoch 75/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7273 - loss: 0.5224 - val_accuracy: 0.7092 - val_loss: 0.5501
Epoch 76/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7328 - loss: 0.5246 - val_accuracy: 0.7011 - val_loss: 0.5433
Epoch 77/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7264 - loss: 0.5387 - val_accuracy: 0.6984 - val_loss: 0.5480
Epoch 78/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7247 - loss: 0.5361 - val_accuracy: 0.7092 - val_loss: 0.5452
Epoch 79/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7189 - loss: 0.5236 - val_accuracy: 0.7065 - val_loss: 0.5448
Epoch 80/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7263 - loss: 0.5321 - val_accuracy: 0.7065 - val_loss: 0.5368
Epoch 81/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7236 - loss: 0.5235 - val_accuracy: 0.7120 - val_loss: 0.5412
Epoch 82/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7341 - loss: 0.5267 - val_accuracy: 0.7147 - val_loss: 0.5416
Epoch 83/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7240 - loss: 0.5292 - val_accuracy: 0.7065 - val_loss: 0.5401
Epoch 84/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7404 - loss: 0.5072 - val_accuracy: 0.7038 - val_loss: 0.5447
Epoch 85/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7087 - loss: 0.5490 - val_accuracy: 0.7092 - val_loss: 0.5362
Epoch 86/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7328 - loss: 0.5260 - val_accuracy: 0.7174 - val_loss: 0.5337
Epoch 87/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7292 - loss: 0.5248 - val_accuracy: 0.7147 - val_loss: 0.5314
Epoch 88/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7394 - loss: 0.5156 - val_accuracy: 0.7038 - val_loss: 0.5324
Epoch 89/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7334 - loss: 0.5107 - val_accuracy: 0.7147 - val_loss: 0.5330
Epoch 90/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7400 - loss: 0.5140 - val_accuracy: 0.7147 - val_loss: 0.5354
Epoch 91/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7216 - loss: 0.5317 - val_accuracy: 0.7038 - val_loss: 0.5317
Epoch 92/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7391 - loss: 0.5060 - val_accuracy: 0.7201 - val_loss: 0.5399
Epoch 93/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7261 - loss: 0.5218 - val_accuracy: 0.6957 - val_loss: 0.5344
Epoch 94/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7303 - loss: 0.5122 - val_accuracy: 0.7120 - val_loss: 0.5333
Epoch 95/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7297 - loss: 0.5210 - val_accuracy: 0.7065 - val_loss: 0.5260
Epoch 96/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7161 - loss: 0.5294 - val_accuracy: 0.6984 - val_loss: 0.5273
Epoch 97/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7233 - loss: 0.5251 - val_accuracy: 0.7065 - val_loss: 0.5263
Epoch 98/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7300 - loss: 0.5237 - val_accuracy: 0.7174 - val_loss: 0.5242
Epoch 99/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7351 - loss: 0.5027 - val_accuracy: 0.6957 - val_loss: 0.5278
Epoch 100/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7346 - loss: 0.5089 - val_accuracy: 0.7065 - val_loss: 0.5236
Restoring model weights from the end of the best epoch: 100.
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
Fold 7 Results:
  Accuracy: 0.7065
  Precision: 0.7246
  Recall: 0.6612
  F1 Score: 0.6914
  AUC: 0.8048
  Selected Features: 39

----- Training Fold 8/10 -----
Epoch 1/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.6193 - loss: 0.9582 - val_accuracy: 0.7120 - val_loss: 0.7328
Epoch 2/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6897 - loss: 0.8049 - val_accuracy: 0.6929 - val_loss: 0.7230
Epoch 3/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7266 - loss: 0.7334 - val_accuracy: 0.6821 - val_loss: 0.7202
Epoch 4/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6829 - loss: 0.7701 - val_accuracy: 0.6793 - val_loss: 0.7188
Epoch 5/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7015 - loss: 0.7345 - val_accuracy: 0.6848 - val_loss: 0.7221
Epoch 6/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7154 - loss: 0.7079 - val_accuracy: 0.6902 - val_loss: 0.7163
Epoch 7/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7031 - loss: 0.7238 - val_accuracy: 0.6630 - val_loss: 0.7174
Epoch 8/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7225 - loss: 0.6901 - val_accuracy: 0.6712 - val_loss: 0.7186
Epoch 9/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6886 - loss: 0.7135 - val_accuracy: 0.6739 - val_loss: 0.7140
Epoch 10/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6983 - loss: 0.6965 - val_accuracy: 0.6793 - val_loss: 0.7092
Epoch 11/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7056 - loss: 0.6954 - val_accuracy: 0.6902 - val_loss: 0.7012
Epoch 12/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7163 - loss: 0.6897 - val_accuracy: 0.6766 - val_loss: 0.7023
Epoch 13/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7192 - loss: 0.6926 - val_accuracy: 0.6848 - val_loss: 0.7006
Epoch 14/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7010 - loss: 0.6980 - val_accuracy: 0.6875 - val_loss: 0.6975
Epoch 15/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7083 - loss: 0.6919 - val_accuracy: 0.6793 - val_loss: 0.6902
Epoch 16/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7276 - loss: 0.6751 - val_accuracy: 0.6739 - val_loss: 0.6940
Epoch 17/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7267 - loss: 0.6553 - val_accuracy: 0.6902 - val_loss: 0.6895
Epoch 18/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7169 - loss: 0.6657 - val_accuracy: 0.6848 - val_loss: 0.6881
Epoch 19/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7165 - loss: 0.6733 - val_accuracy: 0.6848 - val_loss: 0.6864
Epoch 20/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7025 - loss: 0.6725 - val_accuracy: 0.6984 - val_loss: 0.6830
Epoch 21/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7185 - loss: 0.6590 - val_accuracy: 0.6875 - val_loss: 0.6825
Epoch 22/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7264 - loss: 0.6392 - val_accuracy: 0.6739 - val_loss: 0.6794
Epoch 23/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7159 - loss: 0.6552 - val_accuracy: 0.7065 - val_loss: 0.6718
Epoch 24/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7208 - loss: 0.6375 - val_accuracy: 0.6766 - val_loss: 0.6748
Epoch 25/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7089 - loss: 0.6451 - val_accuracy: 0.6821 - val_loss: 0.6723
Epoch 26/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7183 - loss: 0.6333 - val_accuracy: 0.6902 - val_loss: 0.6692
Epoch 27/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7211 - loss: 0.6435 - val_accuracy: 0.6984 - val_loss: 0.6704
Epoch 28/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7233 - loss: 0.6274 - val_accuracy: 0.6929 - val_loss: 0.6584
Epoch 29/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7300 - loss: 0.6301 - val_accuracy: 0.7038 - val_loss: 0.6559
Epoch 30/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7211 - loss: 0.6246 - val_accuracy: 0.6875 - val_loss: 0.6592
Epoch 31/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7201 - loss: 0.6246 - val_accuracy: 0.6875 - val_loss: 0.6519
Epoch 32/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7292 - loss: 0.6131 - val_accuracy: 0.6984 - val_loss: 0.6514
Epoch 33/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7126 - loss: 0.6269 - val_accuracy: 0.6957 - val_loss: 0.6529
Epoch 34/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7158 - loss: 0.6158 - val_accuracy: 0.6875 - val_loss: 0.6486
Epoch 35/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7234 - loss: 0.6160 - val_accuracy: 0.6875 - val_loss: 0.6441
Epoch 36/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7181 - loss: 0.6199 - val_accuracy: 0.6929 - val_loss: 0.6411
Epoch 37/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7345 - loss: 0.5997 - val_accuracy: 0.6957 - val_loss: 0.6420
Epoch 38/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7127 - loss: 0.6290 - val_accuracy: 0.6875 - val_loss: 0.6356
Epoch 39/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7414 - loss: 0.5858 - val_accuracy: 0.6902 - val_loss: 0.6332
Epoch 40/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7296 - loss: 0.5890 - val_accuracy: 0.6929 - val_loss: 0.6348
Epoch 41/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7155 - loss: 0.6016 - val_accuracy: 0.6821 - val_loss: 0.6300
Epoch 42/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7251 - loss: 0.5875 - val_accuracy: 0.6766 - val_loss: 0.6281
Epoch 43/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7339 - loss: 0.5868 - val_accuracy: 0.6821 - val_loss: 0.6278
Epoch 44/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7115 - loss: 0.5957 - val_accuracy: 0.6821 - val_loss: 0.6226
Epoch 45/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7253 - loss: 0.5789 - val_accuracy: 0.6739 - val_loss: 0.6183
Epoch 46/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7184 - loss: 0.5838 - val_accuracy: 0.6766 - val_loss: 0.6152
Epoch 47/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7322 - loss: 0.5764 - val_accuracy: 0.6929 - val_loss: 0.6148
Epoch 48/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7357 - loss: 0.5863 - val_accuracy: 0.6793 - val_loss: 0.6155
Epoch 49/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7332 - loss: 0.5741 - val_accuracy: 0.6875 - val_loss: 0.6099
Epoch 50/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7345 - loss: 0.5715 - val_accuracy: 0.6848 - val_loss: 0.6124
Epoch 51/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7141 - loss: 0.5872 - val_accuracy: 0.6793 - val_loss: 0.6059
Epoch 52/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7398 - loss: 0.5660 - val_accuracy: 0.6821 - val_loss: 0.6035
Epoch 53/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7252 - loss: 0.5484 - val_accuracy: 0.6848 - val_loss: 0.6014
Epoch 54/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7326 - loss: 0.5514 - val_accuracy: 0.6984 - val_loss: 0.5974
Epoch 55/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7208 - loss: 0.5568 - val_accuracy: 0.6712 - val_loss: 0.5993
Epoch 56/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7202 - loss: 0.5647 - val_accuracy: 0.6902 - val_loss: 0.5994
Epoch 57/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7224 - loss: 0.5716 - val_accuracy: 0.6848 - val_loss: 0.5947
Epoch 58/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7241 - loss: 0.5538 - val_accuracy: 0.6902 - val_loss: 0.5966
Epoch 59/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7327 - loss: 0.5434 - val_accuracy: 0.6848 - val_loss: 0.5916
Epoch 60/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7240 - loss: 0.5570 - val_accuracy: 0.6929 - val_loss: 0.5965
Epoch 61/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7389 - loss: 0.5476 - val_accuracy: 0.6929 - val_loss: 0.5888
Epoch 62/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7325 - loss: 0.5494 - val_accuracy: 0.6902 - val_loss: 0.5870
Epoch 63/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7308 - loss: 0.5348 - val_accuracy: 0.6875 - val_loss: 0.5812
Epoch 64/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7088 - loss: 0.5559 - val_accuracy: 0.6902 - val_loss: 0.5821
Epoch 65/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7334 - loss: 0.5380 - val_accuracy: 0.6929 - val_loss: 0.5806
Epoch 66/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7417 - loss: 0.5247 - val_accuracy: 0.6902 - val_loss: 0.5840
Epoch 67/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7254 - loss: 0.5658 - val_accuracy: 0.6848 - val_loss: 0.5718
Epoch 68/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7226 - loss: 0.5387 - val_accuracy: 0.6848 - val_loss: 0.5792
Epoch 69/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7182 - loss: 0.5426 - val_accuracy: 0.6875 - val_loss: 0.5796
Epoch 70/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7312 - loss: 0.5382 - val_accuracy: 0.6929 - val_loss: 0.5728
Epoch 71/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7367 - loss: 0.5355 - val_accuracy: 0.6902 - val_loss: 0.5821
Epoch 72/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7255 - loss: 0.5300 - val_accuracy: 0.6848 - val_loss: 0.5773
Epoch 73/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7288 - loss: 0.5389 - val_accuracy: 0.6929 - val_loss: 0.5697
Epoch 74/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7415 - loss: 0.5321 - val_accuracy: 0.6929 - val_loss: 0.5681
Epoch 75/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7264 - loss: 0.5381 - val_accuracy: 0.6929 - val_loss: 0.5709
Epoch 76/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7158 - loss: 0.5294 - val_accuracy: 0.6902 - val_loss: 0.5691
Epoch 77/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7432 - loss: 0.5254 - val_accuracy: 0.6929 - val_loss: 0.5617
Epoch 78/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7364 - loss: 0.5193 - val_accuracy: 0.6984 - val_loss: 0.5639
Epoch 79/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7247 - loss: 0.5202 - val_accuracy: 0.6957 - val_loss: 0.5592
Epoch 80/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7194 - loss: 0.5283 - val_accuracy: 0.6875 - val_loss: 0.5653
Epoch 81/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7148 - loss: 0.5292 - val_accuracy: 0.6984 - val_loss: 0.5593
Epoch 82/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7260 - loss: 0.5338 - val_accuracy: 0.6957 - val_loss: 0.5610
Epoch 83/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7295 - loss: 0.5217 - val_accuracy: 0.6929 - val_loss: 0.5683
Epoch 84/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7260 - loss: 0.5116 - val_accuracy: 0.6848 - val_loss: 0.5605
Epoch 85/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7357 - loss: 0.5205 - val_accuracy: 0.6793 - val_loss: 0.5516
Epoch 86/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7266 - loss: 0.5293 - val_accuracy: 0.6902 - val_loss: 0.5468
Epoch 87/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7294 - loss: 0.5255 - val_accuracy: 0.6821 - val_loss: 0.5573
Epoch 88/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7291 - loss: 0.5303 - val_accuracy: 0.6984 - val_loss: 0.5534
Epoch 89/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7253 - loss: 0.5266 - val_accuracy: 0.6821 - val_loss: 0.5602
Epoch 90/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7314 - loss: 0.5265 - val_accuracy: 0.6821 - val_loss: 0.5493
Epoch 91/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7342 - loss: 0.5177 - val_accuracy: 0.6875 - val_loss: 0.5532
Epoch 92/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7252 - loss: 0.5201 - val_accuracy: 0.7011 - val_loss: 0.5487
Epoch 93/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7370 - loss: 0.5132 - val_accuracy: 0.6984 - val_loss: 0.5500
Epoch 94/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7253 - loss: 0.5400 - val_accuracy: 0.6821 - val_loss: 0.5500
Epoch 95/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7321 - loss: 0.5154 - val_accuracy: 0.6957 - val_loss: 0.5513
Epoch 96/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7245 - loss: 0.5210 - val_accuracy: 0.6821 - val_loss: 0.5475
Epoch 97/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7166 - loss: 0.5221 - val_accuracy: 0.6957 - val_loss: 0.5474
Epoch 98/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7364 - loss: 0.5133 - val_accuracy: 0.6848 - val_loss: 0.5425
Epoch 99/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7332 - loss: 0.5001 - val_accuracy: 0.6929 - val_loss: 0.5472
Epoch 100/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7369 - loss: 0.4996 - val_accuracy: 0.6821 - val_loss: 0.5427
Restoring model weights from the end of the best epoch: 98.
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
Fold 8 Results:
  Accuracy: 0.6848
  Precision: 0.6936
  Recall: 0.6557
  F1 Score: 0.6742
  AUC: 0.7913
  Selected Features: 39

----- Training Fold 9/10 -----
Epoch 1/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.6135 - loss: 0.9092 - val_accuracy: 0.6984 - val_loss: 0.7463
Epoch 2/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7033 - loss: 0.7616 - val_accuracy: 0.6957 - val_loss: 0.7101
Epoch 3/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7020 - loss: 0.7453 - val_accuracy: 0.6875 - val_loss: 0.7033
Epoch 4/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7102 - loss: 0.7336 - val_accuracy: 0.6902 - val_loss: 0.7005
Epoch 5/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7087 - loss: 0.7292 - val_accuracy: 0.6929 - val_loss: 0.7011
Epoch 6/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6903 - loss: 0.7315 - val_accuracy: 0.6875 - val_loss: 0.7043
Epoch 7/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7193 - loss: 0.6885 - val_accuracy: 0.6848 - val_loss: 0.7025
Epoch 8/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7100 - loss: 0.7086 - val_accuracy: 0.6902 - val_loss: 0.6989
Epoch 9/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7120 - loss: 0.6972 - val_accuracy: 0.6875 - val_loss: 0.6969
Epoch 10/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7077 - loss: 0.6860 - val_accuracy: 0.6712 - val_loss: 0.6967
Epoch 11/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7096 - loss: 0.6838 - val_accuracy: 0.6821 - val_loss: 0.6919
Epoch 12/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7206 - loss: 0.6715 - val_accuracy: 0.6712 - val_loss: 0.6880
Epoch 13/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7169 - loss: 0.6840 - val_accuracy: 0.6630 - val_loss: 0.6879
Epoch 14/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7195 - loss: 0.6658 - val_accuracy: 0.6793 - val_loss: 0.6851
Epoch 15/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7345 - loss: 0.6512 - val_accuracy: 0.6848 - val_loss: 0.6835
Epoch 16/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7111 - loss: 0.6650 - val_accuracy: 0.6848 - val_loss: 0.6834
Epoch 17/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7135 - loss: 0.6730 - val_accuracy: 0.6821 - val_loss: 0.6823
Epoch 18/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7373 - loss: 0.6374 - val_accuracy: 0.6793 - val_loss: 0.6810
Epoch 19/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7247 - loss: 0.6391 - val_accuracy: 0.6766 - val_loss: 0.6759
Epoch 20/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7216 - loss: 0.6505 - val_accuracy: 0.6685 - val_loss: 0.6737
Epoch 21/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7276 - loss: 0.6423 - val_accuracy: 0.6793 - val_loss: 0.6743
Epoch 22/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7165 - loss: 0.6400 - val_accuracy: 0.6766 - val_loss: 0.6697
Epoch 23/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7283 - loss: 0.6363 - val_accuracy: 0.6658 - val_loss: 0.6673
Epoch 24/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7302 - loss: 0.6298 - val_accuracy: 0.6630 - val_loss: 0.6626
Epoch 25/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7284 - loss: 0.6315 - val_accuracy: 0.6685 - val_loss: 0.6589
Epoch 26/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7201 - loss: 0.6263 - val_accuracy: 0.6712 - val_loss: 0.6582
Epoch 27/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7162 - loss: 0.6200 - val_accuracy: 0.6685 - val_loss: 0.6552
Epoch 28/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7165 - loss: 0.6254 - val_accuracy: 0.6685 - val_loss: 0.6549
Epoch 29/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7286 - loss: 0.6113 - val_accuracy: 0.6712 - val_loss: 0.6505
Epoch 30/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7088 - loss: 0.6245 - val_accuracy: 0.6793 - val_loss: 0.6469
Epoch 31/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7217 - loss: 0.6211 - val_accuracy: 0.6739 - val_loss: 0.6444
Epoch 32/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7337 - loss: 0.5956 - val_accuracy: 0.6685 - val_loss: 0.6430
Epoch 33/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7275 - loss: 0.6056 - val_accuracy: 0.6658 - val_loss: 0.6388
Epoch 34/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7205 - loss: 0.6134 - val_accuracy: 0.6630 - val_loss: 0.6373
Epoch 35/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7090 - loss: 0.6146 - val_accuracy: 0.6685 - val_loss: 0.6320
Epoch 36/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7334 - loss: 0.5924 - val_accuracy: 0.6685 - val_loss: 0.6318
Epoch 37/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7243 - loss: 0.5892 - val_accuracy: 0.6603 - val_loss: 0.6298
Epoch 38/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7299 - loss: 0.5901 - val_accuracy: 0.6603 - val_loss: 0.6287
Epoch 39/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7174 - loss: 0.6015 - val_accuracy: 0.6766 - val_loss: 0.6220
Epoch 40/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7257 - loss: 0.5913 - val_accuracy: 0.6685 - val_loss: 0.6220
Epoch 41/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7289 - loss: 0.5765 - val_accuracy: 0.6658 - val_loss: 0.6172
Epoch 42/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7216 - loss: 0.5796 - val_accuracy: 0.6658 - val_loss: 0.6155
Epoch 43/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7275 - loss: 0.5717 - val_accuracy: 0.6712 - val_loss: 0.6105
Epoch 44/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7252 - loss: 0.5673 - val_accuracy: 0.6712 - val_loss: 0.6126
Epoch 45/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7260 - loss: 0.5708 - val_accuracy: 0.6793 - val_loss: 0.6101
Epoch 46/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7394 - loss: 0.5561 - val_accuracy: 0.6685 - val_loss: 0.6041
Epoch 47/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7256 - loss: 0.5744 - val_accuracy: 0.6848 - val_loss: 0.6053
Epoch 48/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7310 - loss: 0.5530 - val_accuracy: 0.6630 - val_loss: 0.6038
Epoch 49/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7325 - loss: 0.5597 - val_accuracy: 0.6712 - val_loss: 0.6023
Epoch 50/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7344 - loss: 0.5577 - val_accuracy: 0.6685 - val_loss: 0.6096
Epoch 51/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7300 - loss: 0.5528 - val_accuracy: 0.6821 - val_loss: 0.6042
Epoch 52/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7226 - loss: 0.5541 - val_accuracy: 0.6793 - val_loss: 0.6036
Epoch 53/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7230 - loss: 0.5423 - val_accuracy: 0.6549 - val_loss: 0.5938
Epoch 54/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7266 - loss: 0.5576 - val_accuracy: 0.6766 - val_loss: 0.5955
Epoch 55/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7425 - loss: 0.5433 - val_accuracy: 0.6658 - val_loss: 0.5946
Epoch 56/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7319 - loss: 0.5443 - val_accuracy: 0.6712 - val_loss: 0.5922
Epoch 57/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7350 - loss: 0.5499 - val_accuracy: 0.6766 - val_loss: 0.5916
Epoch 58/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7316 - loss: 0.5298 - val_accuracy: 0.6739 - val_loss: 0.5942
Epoch 59/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7385 - loss: 0.5446 - val_accuracy: 0.6793 - val_loss: 0.5867
Epoch 60/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7281 - loss: 0.5376 - val_accuracy: 0.6685 - val_loss: 0.5821
Epoch 61/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7302 - loss: 0.5348 - val_accuracy: 0.6739 - val_loss: 0.5842
Epoch 62/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7225 - loss: 0.5480 - val_accuracy: 0.6739 - val_loss: 0.5833
Epoch 63/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7028 - loss: 0.5506 - val_accuracy: 0.6766 - val_loss: 0.5857
Epoch 64/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7429 - loss: 0.5296 - val_accuracy: 0.6712 - val_loss: 0.5780
Epoch 65/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7258 - loss: 0.5371 - val_accuracy: 0.6793 - val_loss: 0.5774
Epoch 66/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7117 - loss: 0.5491 - val_accuracy: 0.6739 - val_loss: 0.5754
Epoch 67/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7203 - loss: 0.5374 - val_accuracy: 0.6848 - val_loss: 0.5817
Epoch 68/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7024 - loss: 0.5426 - val_accuracy: 0.6902 - val_loss: 0.5828
Epoch 69/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7283 - loss: 0.5300 - val_accuracy: 0.6685 - val_loss: 0.5773
Epoch 70/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7157 - loss: 0.5383 - val_accuracy: 0.6875 - val_loss: 0.5771
Epoch 71/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7287 - loss: 0.5268 - val_accuracy: 0.6685 - val_loss: 0.5769
Epoch 72/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7170 - loss: 0.5328 - val_accuracy: 0.6875 - val_loss: 0.5728
Epoch 73/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7324 - loss: 0.5119 - val_accuracy: 0.6848 - val_loss: 0.5791
Epoch 74/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7266 - loss: 0.5226 - val_accuracy: 0.6766 - val_loss: 0.5756
Epoch 75/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7241 - loss: 0.5304 - val_accuracy: 0.6821 - val_loss: 0.5776
Epoch 76/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7445 - loss: 0.5038 - val_accuracy: 0.6875 - val_loss: 0.5752
Epoch 77/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7216 - loss: 0.5275 - val_accuracy: 0.6793 - val_loss: 0.5685
Epoch 78/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7089 - loss: 0.5334 - val_accuracy: 0.6766 - val_loss: 0.5659
Epoch 79/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7277 - loss: 0.5240 - val_accuracy: 0.6821 - val_loss: 0.5634
Epoch 80/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7300 - loss: 0.5244 - val_accuracy: 0.6793 - val_loss: 0.5735
Epoch 81/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7281 - loss: 0.5239 - val_accuracy: 0.6793 - val_loss: 0.5686
Epoch 82/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7251 - loss: 0.5332 - val_accuracy: 0.6875 - val_loss: 0.5629
Epoch 83/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7248 - loss: 0.5273 - val_accuracy: 0.6766 - val_loss: 0.5617
Epoch 84/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7462 - loss: 0.5106 - val_accuracy: 0.6739 - val_loss: 0.5563
Epoch 85/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7318 - loss: 0.5227 - val_accuracy: 0.6793 - val_loss: 0.5534
Epoch 86/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7376 - loss: 0.5244 - val_accuracy: 0.6929 - val_loss: 0.5607
Epoch 87/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7201 - loss: 0.5213 - val_accuracy: 0.6929 - val_loss: 0.5638
Epoch 88/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7224 - loss: 0.5351 - val_accuracy: 0.6821 - val_loss: 0.5623
Epoch 89/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7303 - loss: 0.5155 - val_accuracy: 0.6821 - val_loss: 0.5600
Epoch 90/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7366 - loss: 0.5051 - val_accuracy: 0.6848 - val_loss: 0.5590
Epoch 91/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7309 - loss: 0.5156 - val_accuracy: 0.6793 - val_loss: 0.5605
Epoch 92/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7454 - loss: 0.5027 - val_accuracy: 0.6739 - val_loss: 0.5578
Epoch 93/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7401 - loss: 0.5058 - val_accuracy: 0.6821 - val_loss: 0.5561
Epoch 94/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7314 - loss: 0.5099 - val_accuracy: 0.6766 - val_loss: 0.5553
Epoch 95/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7245 - loss: 0.5168 - val_accuracy: 0.6793 - val_loss: 0.5576
Epoch 96/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7440 - loss: 0.4993 - val_accuracy: 0.6929 - val_loss: 0.5537
Epoch 97/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7272 - loss: 0.5153 - val_accuracy: 0.6766 - val_loss: 0.5577
Epoch 98/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7305 - loss: 0.5169 - val_accuracy: 0.6793 - val_loss: 0.5601
Epoch 99/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7205 - loss: 0.5315 - val_accuracy: 0.6929 - val_loss: 0.5583
Epoch 100/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7223 - loss: 0.5227 - val_accuracy: 0.6848 - val_loss: 0.5592
Epoch 100: early stopping
Restoring model weights from the end of the best epoch: 85.
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
Fold 9 Results:
  Accuracy: 0.6793
  Precision: 0.6796
  Recall: 0.6721
  F1 Score: 0.6758
  AUC: 0.7825
  Selected Features: 40

----- Training Fold 10/10 -----
Epoch 1/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.6217 - loss: 0.9102 - val_accuracy: 0.7418 - val_loss: 0.7377
Epoch 2/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6807 - loss: 0.7843 - val_accuracy: 0.7527 - val_loss: 0.6893
Epoch 3/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6851 - loss: 0.7717 - val_accuracy: 0.7446 - val_loss: 0.6780
Epoch 4/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7027 - loss: 0.7363 - val_accuracy: 0.7337 - val_loss: 0.6809
Epoch 5/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6855 - loss: 0.7370 - val_accuracy: 0.7065 - val_loss: 0.6838
Epoch 6/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7140 - loss: 0.7244 - val_accuracy: 0.7201 - val_loss: 0.6766
Epoch 7/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7170 - loss: 0.7038 - val_accuracy: 0.7174 - val_loss: 0.6692
Epoch 8/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6943 - loss: 0.7281 - val_accuracy: 0.7228 - val_loss: 0.6677
Epoch 9/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7024 - loss: 0.7055 - val_accuracy: 0.7065 - val_loss: 0.6734
Epoch 10/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7041 - loss: 0.7063 - val_accuracy: 0.7092 - val_loss: 0.6683
Epoch 11/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7125 - loss: 0.6860 - val_accuracy: 0.7065 - val_loss: 0.6682
Epoch 12/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7054 - loss: 0.6817 - val_accuracy: 0.7147 - val_loss: 0.6660
Epoch 13/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7029 - loss: 0.6972 - val_accuracy: 0.7065 - val_loss: 0.6635
Epoch 14/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7112 - loss: 0.6669 - val_accuracy: 0.7201 - val_loss: 0.6587
Epoch 15/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7105 - loss: 0.6744 - val_accuracy: 0.7201 - val_loss: 0.6537
Epoch 16/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7088 - loss: 0.6782 - val_accuracy: 0.7228 - val_loss: 0.6540
Epoch 17/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7129 - loss: 0.6553 - val_accuracy: 0.7092 - val_loss: 0.6511
Epoch 18/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7214 - loss: 0.6498 - val_accuracy: 0.7065 - val_loss: 0.6505
Epoch 19/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7063 - loss: 0.6654 - val_accuracy: 0.7174 - val_loss: 0.6463
Epoch 20/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7047 - loss: 0.6470 - val_accuracy: 0.7228 - val_loss: 0.6486
Epoch 21/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7210 - loss: 0.6425 - val_accuracy: 0.7038 - val_loss: 0.6419
Epoch 22/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7152 - loss: 0.6394 - val_accuracy: 0.7065 - val_loss: 0.6388
Epoch 23/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.6961 - loss: 0.6538 - val_accuracy: 0.7201 - val_loss: 0.6344
Epoch 24/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7084 - loss: 0.6453 - val_accuracy: 0.7228 - val_loss: 0.6302
Epoch 25/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7163 - loss: 0.6322 - val_accuracy: 0.7201 - val_loss: 0.6275
Epoch 26/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7238 - loss: 0.6237 - val_accuracy: 0.7283 - val_loss: 0.6266
Epoch 27/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7164 - loss: 0.6219 - val_accuracy: 0.7201 - val_loss: 0.6237
Epoch 28/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7317 - loss: 0.6069 - val_accuracy: 0.7120 - val_loss: 0.6196
Epoch 29/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7285 - loss: 0.6207 - val_accuracy: 0.7228 - val_loss: 0.6172
Epoch 30/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7082 - loss: 0.6246 - val_accuracy: 0.7092 - val_loss: 0.6147
Epoch 31/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7067 - loss: 0.6233 - val_accuracy: 0.7201 - val_loss: 0.6079
Epoch 32/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7233 - loss: 0.6084 - val_accuracy: 0.7065 - val_loss: 0.6054
Epoch 33/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7148 - loss: 0.6177 - val_accuracy: 0.7174 - val_loss: 0.6018
Epoch 34/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7261 - loss: 0.5998 - val_accuracy: 0.7011 - val_loss: 0.5995
Epoch 35/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7204 - loss: 0.6158 - val_accuracy: 0.7092 - val_loss: 0.6011
Epoch 36/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7139 - loss: 0.5991 - val_accuracy: 0.7201 - val_loss: 0.5959
Epoch 37/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7210 - loss: 0.5880 - val_accuracy: 0.7255 - val_loss: 0.5932
Epoch 38/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7208 - loss: 0.6033 - val_accuracy: 0.7201 - val_loss: 0.5894
Epoch 39/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7205 - loss: 0.6041 - val_accuracy: 0.7337 - val_loss: 0.5889
Epoch 40/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7216 - loss: 0.5858 - val_accuracy: 0.7174 - val_loss: 0.5873
Epoch 41/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7212 - loss: 0.5862 - val_accuracy: 0.7310 - val_loss: 0.5826
Epoch 42/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7287 - loss: 0.5863 - val_accuracy: 0.7283 - val_loss: 0.5790
Epoch 43/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7228 - loss: 0.5654 - val_accuracy: 0.7337 - val_loss: 0.5774
Epoch 44/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7205 - loss: 0.5749 - val_accuracy: 0.7310 - val_loss: 0.5805
Epoch 45/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7185 - loss: 0.5726 - val_accuracy: 0.7255 - val_loss: 0.5768
Epoch 46/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7118 - loss: 0.5727 - val_accuracy: 0.7310 - val_loss: 0.5703
Epoch 47/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7250 - loss: 0.5781 - val_accuracy: 0.7310 - val_loss: 0.5724
Epoch 48/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7318 - loss: 0.5792 - val_accuracy: 0.7120 - val_loss: 0.5675
Epoch 49/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7255 - loss: 0.5618 - val_accuracy: 0.7120 - val_loss: 0.5661
Epoch 50/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7086 - loss: 0.5712 - val_accuracy: 0.7038 - val_loss: 0.5654
Epoch 51/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7261 - loss: 0.5559 - val_accuracy: 0.7038 - val_loss: 0.5619
Epoch 52/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7209 - loss: 0.5582 - val_accuracy: 0.7310 - val_loss: 0.5615
Epoch 53/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7429 - loss: 0.5481 - val_accuracy: 0.7065 - val_loss: 0.5572
Epoch 54/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7365 - loss: 0.5441 - val_accuracy: 0.7120 - val_loss: 0.5551
Epoch 55/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7177 - loss: 0.5556 - val_accuracy: 0.7065 - val_loss: 0.5560
Epoch 56/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7264 - loss: 0.5459 - val_accuracy: 0.7120 - val_loss: 0.5509
Epoch 57/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7214 - loss: 0.5531 - val_accuracy: 0.7228 - val_loss: 0.5496
Epoch 58/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7109 - loss: 0.5563 - val_accuracy: 0.7147 - val_loss: 0.5474
Epoch 59/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7295 - loss: 0.5368 - val_accuracy: 0.7201 - val_loss: 0.5438
Epoch 60/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7189 - loss: 0.5515 - val_accuracy: 0.7147 - val_loss: 0.5471
Epoch 61/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7341 - loss: 0.5383 - val_accuracy: 0.7310 - val_loss: 0.5450
Epoch 62/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7261 - loss: 0.5350 - val_accuracy: 0.7283 - val_loss: 0.5451
Epoch 63/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7096 - loss: 0.5448 - val_accuracy: 0.7228 - val_loss: 0.5466
Epoch 64/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7229 - loss: 0.5414 - val_accuracy: 0.7283 - val_loss: 0.5450
Epoch 65/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7259 - loss: 0.5382 - val_accuracy: 0.7228 - val_loss: 0.5362
Epoch 66/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7188 - loss: 0.5448 - val_accuracy: 0.7201 - val_loss: 0.5303
Epoch 67/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7329 - loss: 0.5306 - val_accuracy: 0.7147 - val_loss: 0.5306
Epoch 68/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7136 - loss: 0.5493 - val_accuracy: 0.7310 - val_loss: 0.5314
Epoch 69/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7230 - loss: 0.5384 - val_accuracy: 0.7283 - val_loss: 0.5321
Epoch 70/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7329 - loss: 0.5240 - val_accuracy: 0.7337 - val_loss: 0.5310
Epoch 71/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7389 - loss: 0.5228 - val_accuracy: 0.7283 - val_loss: 0.5307
Epoch 72/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7220 - loss: 0.5408 - val_accuracy: 0.7174 - val_loss: 0.5261
Epoch 73/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7069 - loss: 0.5409 - val_accuracy: 0.7228 - val_loss: 0.5261
Epoch 74/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7271 - loss: 0.5205 - val_accuracy: 0.7120 - val_loss: 0.5260
Epoch 75/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7330 - loss: 0.5261 - val_accuracy: 0.7201 - val_loss: 0.5237
Epoch 76/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7445 - loss: 0.5248 - val_accuracy: 0.7092 - val_loss: 0.5234
Epoch 77/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7247 - loss: 0.5328 - val_accuracy: 0.7174 - val_loss: 0.5290
Epoch 78/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7177 - loss: 0.5270 - val_accuracy: 0.7174 - val_loss: 0.5286
Epoch 79/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7147 - loss: 0.5324 - val_accuracy: 0.7201 - val_loss: 0.5248
Epoch 80/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7168 - loss: 0.5274 - val_accuracy: 0.7092 - val_loss: 0.5222
Epoch 81/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7176 - loss: 0.5135 - val_accuracy: 0.7174 - val_loss: 0.5223
Epoch 82/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7342 - loss: 0.5208 - val_accuracy: 0.7228 - val_loss: 0.5262
Epoch 83/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7323 - loss: 0.5275 - val_accuracy: 0.7201 - val_loss: 0.5250
Epoch 84/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7120 - loss: 0.5348 - val_accuracy: 0.7174 - val_loss: 0.5211
Epoch 85/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7228 - loss: 0.5283 - val_accuracy: 0.7201 - val_loss: 0.5189
Epoch 86/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7264 - loss: 0.5198 - val_accuracy: 0.7092 - val_loss: 0.5177
Epoch 87/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7214 - loss: 0.5270 - val_accuracy: 0.7147 - val_loss: 0.5224
Epoch 88/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7062 - loss: 0.5403 - val_accuracy: 0.7147 - val_loss: 0.5145
Epoch 89/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7364 - loss: 0.5085 - val_accuracy: 0.7228 - val_loss: 0.5197
Epoch 90/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7284 - loss: 0.5189 - val_accuracy: 0.7092 - val_loss: 0.5186
Epoch 91/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7114 - loss: 0.5301 - val_accuracy: 0.7174 - val_loss: 0.5158
Epoch 92/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7200 - loss: 0.5207 - val_accuracy: 0.7174 - val_loss: 0.5157
Epoch 93/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7220 - loss: 0.5230 - val_accuracy: 0.7201 - val_loss: 0.5108
Epoch 94/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7089 - loss: 0.5230 - val_accuracy: 0.7174 - val_loss: 0.5165
Epoch 95/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7176 - loss: 0.5173 - val_accuracy: 0.7228 - val_loss: 0.5175
Epoch 96/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7364 - loss: 0.5024 - val_accuracy: 0.7065 - val_loss: 0.5201
Epoch 97/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7168 - loss: 0.5176 - val_accuracy: 0.7255 - val_loss: 0.5176
Epoch 98/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7253 - loss: 0.5164 - val_accuracy: 0.6984 - val_loss: 0.5190
Epoch 99/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7195 - loss: 0.5284 - val_accuracy: 0.7147 - val_loss: 0.5104
Epoch 100/100
104/104 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7240 - loss: 0.5272 - val_accuracy: 0.7174 - val_loss: 0.5105
Restoring model weights from the end of the best epoch: 99.
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
Fold 10 Results:
  Accuracy: 0.7147
  Precision: 0.7120
  Recall: 0.7158
  F1 Score: 0.7139
  AUC: 0.8227
  Selected Features: 39

Average Metrics Across 10 Folds:
  Accuracy: 0.6965 ± 0.0220
  Precision: 0.6988 ± 0.0183
  Recall: 0.6849 ± 0.0563
  F1: 0.6906 ± 0.0319
  Auc: 0.7964 ± 0.0209

Stable Feature Set: 24 features
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
Ensemble model training complete.