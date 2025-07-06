# AIPAL Validator

AIPAL Validator is a tool designed to streamline the validation process for [AIPAL](https://github.com/VincentAlcazer/AIPAL). This software provides a comprehensive FHIR-based validation pipeline for pediatric acute leukemia prediction models.

**Repository**: https://github.com/UMEssen/aipal-validation  
**Version**: 0.2.0  
**License**: MIT License

## System Requirements

### Software Dependencies
- **Python**: 3.10 or higher
- **R**: 4.0 or higher
- **Poetry**: 1.0 or higher (for dependency management)
- **Docker**: Optional, for containerized deployment

### Required R Packages
- dplyr
- tidyr 
- yaml
- caret
- xgboost

### Python Dependencies (managed by Poetry)
- fhir-pyrate (from GitHub)
- wandb ^0.18.0
- pyyaml ^6.0.1
- python-dotenv ^1.0.0
- psycopg2-binary ^2.9.7
- matplotlib ^3.8.0
- scikit-learn ^1.4.2
- openpyxl ^3.1.3
- xgboost ^2.1.0
- shap ^0.46.0
- seaborn ^0.13.2
- tabulate ^0.9.0

### Operating Systems Tested
- Linux (Ubuntu 20.04+)
- macOS (10.15+)
- Windows 10 (via Docker)

### Hardware Requirements
- Minimum: 8GB RAM, 2 CPU cores
- Recommended: 16GB RAM, 4+ CPU cores
- Storage: 2GB free space

## Installation Guide

### Local Installation

1. **Install R and required packages**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install r-base
   
   # macOS
   brew install r
   
   # Install R packages
   R -e "install.packages(c('dplyr', 'tidyr', 'yaml', 'caret', 'xgboost'))"
   ```

2. **Install Poetry**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Clone and install dependencies**:
   ```bash
   git clone https://github.com/UMEssen/aipal-validation.git
   cd aipal-validation
   poetry install
   ```

4. **Verify installation**:
   ```bash
   poetry run aipal_validation --help
   ```

**Typical installation time**: 5-10 minutes on a standard desktop computer

### Docker Installation

1. **Install Docker and Docker Compose**
2. **Clone repository**:
   ```bash
   git clone https://github.com/UMEssen/aipal-validation.git
   cd aipal-validation
   ```

3. **Build and run container**:
   ```bash
   docker compose build
   docker compose run aipal bash
   ```

**Typical installation time**: 5-10 minutes (including Docker image build)

## Demo

### Quick Demo with Synthetic Data

The software includes synthetic test data for demonstration purposes.

1. **Generate synthetic data and run complete pipeline**:
   ```bash
   # Local installation
   cd synthetic_test_data
   poetry run python generate_syntetic_data.py
   cd ..
   poetry run aipal_validation \
       --config aipal_validation/config/config_synthetic.yaml \
       --task aipal \
       --step sampling+test \
       --debug
   ```

   ```bash
   # Docker installation
   docker compose run aipal bash
   cd synthetic_test_data
   python generate_syntetic_data.py
   cd ..
   python -m aipal_validation \
       --config aipal_validation/config/config_synthetic.yaml \
       --task aipal \
       --step sampling+test \
       --debug
   ```

### Expected Output

The demo will generate the following files in `synthetic_test_data/aipal/`:
- `data.csv` - Original synthetic medical data (1000 samples)
- `samples.csv` - Processed data for ML pipeline
- `predict.csv` - Model predictions with probabilities
- `predictions_*.csv` - Results with different confidence cutoffs
- `results.csv` - Final evaluation metrics (AUC, sensitivity, specificity)

Expected console output includes:
- Data processing statistics
- Model prediction results
- Performance metrics
- Confidence interval calculations

**Expected run time**: 2-5 minutes on a standard desktop computer

## Instructions for Use

### Running on Your Own Data

#### Option 1: Excel Data Import

1. **Prepare your data**:
   - Create a directory structure: `your_cohort_name/aipal/`
   - Place your Excel file in the `aipal` folder
   - Ensure column names match expected format (see configuration files)

2. **Set configuration**:
   ```bash
   # Update run_id in config to match your cohort name
   vim aipal_validation/config/config_training.yaml
   ```

3. **Generate samples and run validation**:
   ```bash
   poetry run aipal_validation --task aipal --step sampling
   poetry run aipal_validation --task aipal --step test
   ```

#### Option 2: FHIR Data Processing

1. **Configure FHIR connection**:
   - Update database connection settings in configuration files
   - Set up environment variables for database credentials

2. **Run complete pipeline**:
   ```bash
   poetry run aipal_validation --task aipal --step all
   ```

### Additional Tasks

#### Outlier Detection (Adult Subset)
```bash
poetry run aipal_validation --task outlier --step detect
```

#### Model Retraining (Pediatric Subset)
```bash
poetry run aipal_validation --task retrain --step all
```

## Project Structure

```
aipal_validation/
├── r/                    # R scripts for prediction and model training
├── config/              # Configuration files
├── data_preprocessing/  # Data preprocessing modules
├── eval/               # Evaluation modules
├── fhir/               # FHIR-related modules
├── ml/                 # Machine learning modules
├── outlier/            # Outlier detection modules
└── helper/             # Utility functions
```

## Reproduction Instructions

To reproduce the results from the associated manuscript:

1. **Follow the installation guide above**
2. **Use the provided synthetic data or your own dataset**
3. **Run the complete validation pipeline**:
   ```bash
   poetry run aipal_validation --task aipal --step all
   ```
4. **Evaluation scripts** are available in the `eval/` directory for detailed analysis

## License

This software is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:
```
[Citation information to be added when manuscript is published]
```

## Support

For issues and questions, please open an issue on the [GitHub repository](https://github.com/UMEssen/aipal-validation/issues).
