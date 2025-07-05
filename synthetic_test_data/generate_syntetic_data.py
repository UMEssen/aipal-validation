import json
import csv
import random
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_sample_config(config_path: str) -> dict:
    """Load the sample configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Sample configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        raise


def generate_realistic_ranges(sample_data: dict) -> dict:
    """
    Generate realistic ranges for each medical parameter based on sample values.
    Returns a dictionary with min, max, and distribution type for each parameter.
    """
    ranges = {}
    
    # Define parameter-specific ranges based on medical knowledge
    parameter_configs = {
        'MCV_fL': {'factor': 0.3, 'min_val': 60, 'max_val': 120},  # Mean corpuscular volume
        'MCHC_g_L': {'factor': 0.15, 'min_val': 280, 'max_val': 380},  # Mean corpuscular hemoglobin concentration
        'Platelets_G_L': {'factor': 0.8, 'min_val': 10, 'max_val': 800},  # Platelet count
        'age': {'factor': 0.4, 'min_val': 18, 'max_val': 90},  # Age
        'WBC_G_L': {'factor': 0.7, 'min_val': 2, 'max_val': 20},  # White blood cell count
        'Monocytes_G_L': {'factor': 0.8, 'min_val': 0.2, 'max_val': 15},  # Monocyte count
        'PT_percent': {'factor': 0.6, 'min_val': 2, 'max_val': 25},  # Prothrombin time
        'Fibrinogen_g_L': {'factor': 0.7, 'min_val': 1, 'max_val': 12},  # Fibrinogen level
        'LDH_UI_L': {'factor': 0.4, 'min_val': 100, 'max_val': 500},  # Lactate dehydrogenase
        'Lymphocytes_G_L': {'factor': 0.8, 'min_val': 0.5, 'max_val': 15},  # Lymphocyte count
    }
    
    for param, value in sample_data.items():
        if param in parameter_configs:
            config = parameter_configs[param]
            
            # Calculate range based on sample value and factor
            variation = value * config['factor']
            min_val = max(config['min_val'], value - variation)
            max_val = min(config['max_val'], value + variation)
            
            ranges[param] = {
                'min': min_val,
                'max': max_val,
                'sample_value': value,
                'distribution': 'normal'  # Use normal distribution for most medical parameters
            }
        else:
            logger.warning(f"Unknown parameter: {param}, using default range")
            # Default range for unknown parameters
            ranges[param] = {
                'min': value * 0.5,
                'max': value * 1.5,
                'sample_value': value,
                'distribution': 'normal'
            }
    
    return ranges


def normal_random(mean: float, std: float) -> float:
    """Generate a random number from a normal distribution using Box-Muller transform."""
    import math
    
    # Simple Box-Muller transform
    u1 = random.random()
    u2 = random.random()
    
    # Avoid log(0)
    while u1 <= 0:
        u1 = random.random()
    
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mean + std * z0


def generate_test_data(ranges: dict, n_samples: int = 1000, random_seed: int = 42) -> list:
    """Generate test data based on the defined ranges."""
    random.seed(random_seed)
    
    # Get column names in consistent order (original medical parameters)
    medical_columns = list(ranges.keys())
    
    # Add required pipeline columns
    all_columns = medical_columns + ['class', 'sex', 'ID', 'Monocytes_percent']
    data = []
    
    # Add header row
    data.append(all_columns)
    
    # Define choices for categorical columns
    class_choices = ['ALL', 'AML', 'APL']
    sex_choices = ['Male', 'Female']
    
    for i in range(n_samples):
        row = []
        
        # Generate medical parameters
        medical_values = {}
        for param in medical_columns:
            param_config = ranges[param]
            min_val = param_config['min']
            max_val = param_config['max']
            sample_val = param_config['sample_value']
            
            if param_config['distribution'] == 'normal':
                # Use normal distribution centered around sample value
                mean = sample_val
                std = (max_val - min_val) / 6  # 99.7% of values within range
                
                # Generate normal random value
                value = normal_random(mean, std)
                
                # Clip values to ensure they're within the specified range
                value = max(min_val, min(max_val, value))
            else:
                # Use uniform distribution
                value = random.uniform(min_val, max_val)
            
            # Round values appropriately based on parameter type
            if param == 'age':
                value = int(round(value))
            elif param in ['MCV_fL', 'LDH_UI_L', 'MCHC_g_L']:
                value = round(value, 1)
            else:
                value = round(value, 2)
            
            medical_values[param] = value
            row.append(value)
        
        # Add required pipeline columns
        # Class: random choice from leukemia types
        class_value = random.choice(class_choices)
        row.append(class_value)
        
        # Sex: random choice from Male/Female
        sex_value = random.choice(sex_choices)
        row.append(sex_value)
        
        # ID: formatted synthetic ID
        id_value = f'SYN_{i:04d}'
        row.append(id_value)
        
        # Monocytes_percent: calculated from Monocytes_G_L and WBC_G_L
        if 'Monocytes_G_L' in medical_values and 'WBC_G_L' in medical_values:
            monocytes_percent = (medical_values['Monocytes_G_L'] / medical_values['WBC_G_L']) * 100
            monocytes_percent = round(monocytes_percent, 2)
        else:
            # Fallback if these columns don't exist
            monocytes_percent = round(random.uniform(2, 15), 2)
        row.append(monocytes_percent)
        
        data.append(row)
    
    return data


def calculate_statistics(data: list) -> dict:
    """Calculate basic statistics for the generated data."""
    if len(data) <= 1:
        return {}
    
    columns = data[0]
    stats = {}
    
    # Define which columns are categorical vs numeric
    categorical_columns = {'class', 'sex', 'ID'}
    
    for col_idx, col_name in enumerate(columns):
        values = [row[col_idx] for row in data[1:]]  # Skip header
        n = len(values)
        
        if col_name in categorical_columns:
            # Calculate categorical statistics
            unique_values = list(set(values))
            value_counts = {val: values.count(val) for val in unique_values}
            most_common = max(value_counts, key=value_counts.get)
            
            stats[col_name] = {
                'count': n,
                'unique_values': len(unique_values),
                'most_common': most_common,
                'most_common_count': value_counts[most_common],
                'values': unique_values,
                'type': 'categorical'
            }
        else:
            # Calculate numeric statistics
            numeric_values = [float(v) for v in values]  # We know these are numeric
            numeric_values_sorted = sorted(numeric_values)
            stats[col_name] = {
                'count': n,
                'mean': sum(numeric_values) / n,
                'min': min(numeric_values),
                'max': max(numeric_values),
                'median': numeric_values_sorted[n // 2] if n % 2 == 1 else (numeric_values_sorted[n // 2 - 1] + numeric_values_sorted[n // 2]) / 2,
                'type': 'numeric'
            }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Generate test dataset from sample configuration')
    parser.add_argument('--config', '-c', default='../aipal_validation/config/sample.json',
                        help='Path to sample configuration file')
    parser.add_argument('--output', '-o', default='data.csv',
                        help='Output CSV file name')
    parser.add_argument('--samples', '-n', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load sample configuration
    logger.info(f"Loading sample configuration from: {args.config}")
    sample_data = load_sample_config(args.config)
    
    # Generate ranges
    logger.info("Generating realistic parameter ranges...")
    ranges = generate_realistic_ranges(sample_data)
    
    # Log the ranges for each parameter
    for param, param_range in ranges.items():
        logger.info(f"{param}: {param_range['min']:.2f} - {param_range['max']:.2f} "
                   f"(sample: {param_range['sample_value']})")
    
    # Generate test data
    logger.info(f"Generating {args.samples} test samples...")
    test_data = generate_test_data(ranges, args.samples, args.seed)
    
    # Save to CSV
    logger.info(f"Saving test dataset to: {args.output}")
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(test_data)
    
    # Calculate and display summary statistics
    stats = calculate_statistics(test_data)
    
    logger.info("Test dataset summary:")
    logger.info(f"Shape: ({len(test_data)-1}, {len(test_data[0])})")
    logger.info(f"Columns: {test_data[0]}")
    
    print("\nDataset Summary:")
    print(f"{'Parameter':<20} {'Type':<12} {'Count':<10} {'Details':<50}")
    print("-" * 92)
    for param, stat in stats.items():
        if stat['type'] == 'numeric':
            details = f"Mean: {stat['mean']:.2f}, Min: {stat['min']:.2f}, Max: {stat['max']:.2f}, Median: {stat['median']:.2f}"
        else:
            details = f"Unique: {stat['unique_values']}, Most common: {stat['most_common']} ({stat['most_common_count']}x)"
        print(f"{param:<20} {stat['type']:<12} {stat['count']:<10} {details:<50}")
    
    logger.info(f"Test dataset successfully generated: {args.output}")


if __name__ == "__main__":
    main() 