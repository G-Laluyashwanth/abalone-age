## Results

| Metric | Score |
|---|---|
| Validation MAE (5-fold CV) | 1.251 rings |
| Kaggle Public Score (RMSLE) | 0.14913 |
| Kaggle Private Score (RMSLE) | 0.14931 |

**Model:** Random Forest with tuned hyperparameters inside a scikit-learn Pipeline.

**Best hyperparameters:** `n_estimators=200, max_depth=20, max_features='sqrt', min_samples_split=2, min_samples_leaf=2`

## Key Takeaways

- Reused the preprocessing + modelling template from my [Ames Housing project](https://github.com/G-Laluyashwanth/ames-housing) with minimal changes
- Dataset is ~60x larger than House Prices (90k vs 1.5k rows) — forced me to tune the search space for speed
- Random Forest defaults are strong; tuning provided marginal gains
- Same `id` column trap as House Prices — muscle memory now engaged