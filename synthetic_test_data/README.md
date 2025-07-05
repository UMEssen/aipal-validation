# Synthetic Test Data for AIPAL Validation

Generate and test synthetic medical data for the AIPAL (AI-based Pediatric Acute Leukemia) validation system.

## Quick Start

```bash
# Install dependencies
poetry install

# Generate synthetic data and run complete pipeline
cd synthetic_test_data
poetry run python generate_syntetic_data.py
cd ..

# Run the complete validation pipeline
poetry run aipal_validation \
    --config aipal_validation/config/config_synthetic.yaml \
    --task aipal \
    --step sampling+test \
    --debug
```

## What This Does

1. **Generates synthetic medical data** with realistic parameters (MCV, MCHC, Platelets, WBC, etc.)
2. **Processes the data** through the AIPAL model pipeline
3. **Runs predictions** using the R-based XGBoost model
4. **Evaluates performance** with multiple metrics and confidence cutoffs

## Output Files

After running, you'll find these files in `synthetic_test_data/aipal/`:

- `data.csv` - Original synthetic data
- `samples.csv` - Processed data for ML pipeline
- `predict.csv` - Model predictions
- `predictions_*.csv` - Results with different confidence cutoffs
- `results.csv` - Final evaluation metrics

## Requirements

- Python 3.10+
- R with required packages (dplyr, tidyr, yaml, xgboost)
- Poetry for dependency management

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Run `poetry install` to ensure all packages are installed
2. **R Script Errors**: Verify R is installed with required packages: `install.packages(c('dplyr', 'tidyr', 'yaml', 'xgboost'))`
3. **Path Issues**: Always run commands from the project root directory
4. **Permission Errors**: If you see "Read-only file system" errors, check your directory permissions

### Debug Mode
Always run with `--debug` flag during development:
- Reduces bootstrap iterations for faster execution
- Enables verbose logging
- Disables Weights & Biases logging

## Notes
- Synthetic data is for validation purposes only
- The pipeline generates realistic medical test data with proper statistical distributions