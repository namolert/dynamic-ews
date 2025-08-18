# Dynamic Early Warning System for Financial Crashes

## Abstract

This study investigates the role of market-based and sentiment-based volatility indicators in predicting financial crashes and develops a dynamic early warning system (EWS) to improve crisis detection. The research evaluates 48 models, comparing traditional static logit approaches with dynamic architectures including dynamic logit, CNN, and LSTM, across multiple volatility windows. Models were filtered based on false negative rate (FNR ≤ 0.5) and noise-to-signal ratio (NSR ≤ 0.34), resulting in 6 models that met both criteria. The CNN model using sentiment features along with 22-day volatility window achieved the highest weighted score (Score = 0.8971) and true positive rate (TPR = 0.9394), demonstrating strong sensitivity while maintaining reasonable specificity. The analysis shows that sentiment-based indicators provide anticipatory signals that complement market-based volatility, enhancing early crash detection. While some models with higher specificity exhibited lower sensitivity, prioritizing models with high sensitivity is preferable due to the asymmetric cost of missed crises. An additional finding is that shorter volatility windows (5-day and 22-day) outperform longer windows (66-day and 132-day), suggesting the importance of timely information.

## Features

- Combines market volatility and sentiment volatility features
- Implements dynamic EWS models using Lags, CNN and LSTM architectures
- Supports multiple volatility window sizes (22, 66, 132 days)
- Evaluates models using TPR, TNR, FPR, FNR, PPV, FOR, NSR, ACC, AUC, F1, and weighted scoring
- Provides visualizations and evaluation tables for model performance

## Usage

- After clone the repository, open `dynamic_ews.ipynb`. This notebook contains the full workflow for running all models and generating results.
- The plotting functions are available in `plot_utils.py`. You can modify them if needed, but make sure to restart the IDE and run the entire notebook again to ensure all changes take effect.
- Results, including evaluation tables and visualizations, will be generated and saved in the csv files.

## Key Findings

- The CNN model with sentiment features (CNN_Sentiment_22) achieved the highest weighted score (0.8971) and TPR (0.9394), showing strong sensitivity to crash events.
- Static logit models with combined features also performed well, suggesting feature combination can improve traditional approaches.
- Shorter volatility windows (22 days) were generally more effective than longer windows for early warning detection.
- Sentiment-based indicators provide complementary anticipatory signals to market-based volatility.

## Challenges

- Financial crashes are rare events (~5.87% in the dataset), making model training and evaluation difficult.
- Complex models like CNN and LSTM may produce low probability outputs, affecting confidence in rare event predictions.

## Future Work

- Explore intermediate-complexity models, such as XGBoost or shallow neural networks, to balance flexibility and reliability.
- Include additional volatility or alternative indicators to enrich model inputs.
- Experiment with advanced volatility estimators (Parkinson, Garman-Klass) and customized scoring functions to improve predictive performance.
