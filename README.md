# DATA ANALYTICS Final

Small greyhound-racing modeling/backtesting workflow with:
- feature building
- LightGBM training + isotonic calibration
- strategy ROI summaries
- bankroll simulation
- calibration plotting

All scripts are configured by variables at the top of each file (no CLI flags needed).

## Project Layout

- `datasets/`  
  Source/input datasets used by scripts.
- `outputs/`  
  Generated predictions, matrices, and plots.
- `train_model.py`  
  Trains LightGBM on `datasets/feature_dataset.csv`, calibrates probabilities, writes predictions.
- `find_profitable_strategy.py`  
  Builds strategy columns from predictions and outputs edge-threshold ROI/bet-count table.
- `simulate_bankroll.py`  
  Sequential compounding bankroll simulation + ROI matrix + bankroll plot.
- `plot_p_win_calibration.py`  
  Calibration scatter plot (`predicted p` vs empirical win rate).
- `build_feature_dataset.py`  
  Builds feature dataset from race/form data.

## Data + Output Paths

Current defaults in code:
- Inputs mostly read from `datasets/...`
- Outputs write to `outputs/...`

If you change filenames/locations, edit the constants near the top of each script.

### Dataset Availability

The full raw/working datasets are too large to include directly in this repository.
If you need the datasets used for this project, they are available upon request.

## Typical Workflow

1. Build/refresh features (if needed):
   - Run `build_feature_dataset.py`
2. Train model + create predictions:
   - Run `train_model.py`
   - Output: `outputs/test_predictions.csv`
3. Strategy summary table:
   - Run `find_profitable_strategy.py`
   - Outputs include `outputs/test_predictions_strategy.csv` and `outputs/kelly_edge_matrix.csv`
4. Bankroll simulation:
   - Run `simulate_bankroll.py`
   - Outputs include `outputs/simulate_roi_matrix.csv` and `outputs/bankroll_sim.png`
5. Calibration plot:
   - Run `plot_p_win_calibration.py`
   - Output: `outputs/p_win_calibration_scatter.png`

## Notes

- `TOP_PICK_ONLY` options in strategy/simulation scripts control whether analysis is one model pick per race or all qualifying runners.
- ROI values in current strategy/simulation tables are decimal returns unless otherwise stated (e.g., `0.05 = +5%`).
