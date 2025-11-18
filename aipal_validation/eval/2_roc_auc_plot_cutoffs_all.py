import matplotlib.pyplot as plt
import pandas as pd
import yaml
import os
from pathlib import Path
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from .util import load_data, extract_roc_source_data, save_roc_source_data_to_excel

# Add proper font settings for better text rendering
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Helper function to prune prediction data based on missing values
def prediction_data_pruner(df, threshold=0):
    """Remove data rows based on the threshold of missing values."""
    data = df.copy()
    mandatory_columns = [
        'Fibrinogen_g_L', 'LDH_UI_L', 'WBC_G_L', 'Lymphocytes_G_L',
        'MCHC_g_L', 'MCV_fL', 'Monocytes_G_L', 'Platelets_G_L',
        'PT_percent'
    ]

    data["nan_percentage"] = data[mandatory_columns].isna().mean(axis=1)

    # Prune data where the percentage of NaN values is above the threshold
    data = data[data["nan_percentage"] <= threshold]
    data.drop(columns=["nan_percentage"], inplace=True)
    return data

# Load the data using util.py's load_data function with analysis config
analysis_config_path = str(Path(__file__).parent.parent / "config" / "config_analysis.yaml")

# Load config first to get root_dir
with open(analysis_config_path, 'r') as f:
    temp_config = yaml.safe_load(f)

df, config, features = load_data(config_path=analysis_config_path, root_path=temp_config['root_dir'] + '/', is_adult=True)

# Also load training config for cutoffs
training_config_path = str(Path(__file__).parent.parent / "config" / "config_training.yaml")
with open(training_config_path, 'r') as f:
    training_config = yaml.safe_load(f)

# Ensure predictions are in float format
df[["prediction.AML", "prediction.APL", "prediction.ALL"]] = df[["prediction.AML", "prediction.APL", "prediction.ALL"]].astype(float)

# Prune the prediction data based on the threshold
df = prediction_data_pruner(df, threshold=0.2)

# Identify the column with the maximum prediction value
df["max_pred"] = df[[
    "prediction.AML",
    "prediction.APL",
    "prediction.ALL"
]].idxmax(axis=1)
df["max_pred"] = df["max_pred"].str.split(".").str[1]  # Extract the class name

# Extract cutoffs from training config
cutoffs = {item['category']: item for item in training_config['cutoffs']}

# Overall cutoff (ACC)
acc_cutoff = {
    "AML": cutoffs['AML']['ACC'],
    "APL": cutoffs['APL']['ACC'],
    "ALL": cutoffs['ALL']['ACC']
}

# Confident cutoff (PPV)
ppv_cutoff = {
    "AML": cutoffs['AML']['PPV'],
    "APL": cutoffs['APL']['PPV'],
    "ALL": cutoffs['ALL']['PPV']
}

# Filter data using cutoffs
df_acc = df[(df["prediction.AML"] > acc_cutoff["AML"]) |
            (df["prediction.APL"] > acc_cutoff["APL"]) |
            (df["prediction.ALL"] > acc_cutoff["ALL"])]

df_ppv = df[(df["prediction.AML"] > ppv_cutoff["AML"]) |
            (df["prediction.APL"] > ppv_cutoff["APL"]) |
            (df["prediction.ALL"] > ppv_cutoff["ALL"])]

# Get sample sizes by class for each dataset
class_counts = {
    'No Cutoff': df['class'].value_counts().to_dict(),
    'Overall Cutoff': df_acc['class'].value_counts().to_dict(),
    'Confident Cutoff': df_ppv['class'].value_counts().to_dict()
}

# Function to calculate ROC curves
def calculate_roc_curves(data, classes=["AML", "APL", "ALL"]):
    """Calculate ROC curves for all classes."""
    y = label_binarize(data['class'], classes=classes)

    # Convert predictions to a similar format for ROC calculation
    y_score = data[["prediction.AML", "prediction.APL", "prediction.ALL"]].values

    # Compute ROC curve and ROC area for each class
    results = {}
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        results[class_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }

    return results

# Calculate ROC curves for each dataset
roc_results_all = calculate_roc_curves(df)
roc_results_acc = calculate_roc_curves(df_acc)
roc_results_ppv = calculate_roc_curves(df_ppv)

# Create the plots directory if it doesn't exist
tag = "Adult" if config['is_adult'] else "Pediatric"
plots_dir = os.path.join(config['root_results'], '2_roc_auc_plot_cutoffs_all')
os.makedirs(plots_dir, exist_ok=True)
print(f"Saving plots to: {plots_dir}")

# Extract and save ROC source data to Excel
classes = ["AML", "APL", "ALL"]
datasets = {
    'no_cutoff': ('No Cutoff', df),
    'overall_cutoff': ('Overall Cutoff', df_acc),
    'confident_cutoff': ('Confident Cutoff', df_ppv)
}
save_roc_source_data_to_excel(datasets, plots_dir, tag, classes)

# Create a combined figure with subplots for each class
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
classes = ["AML", "APL", "ALL"]
class_colors = {'AML': 'red', 'APL': 'green', 'ALL': 'blue'}
cutoff_styles = {
    'No Cutoff': {'linestyle': '-', 'alpha': 0.9},
    'Overall Cutoff': {'linestyle': '--', 'alpha': 0.9},
    'Confident Cutoff': {'linestyle': ':', 'alpha': 0.9}
}

# Plot ROC curves for each class in separate subplots
for i, class_name in enumerate(classes):
    ax = axes[i]

    # Get sample sizes for this class
    n_all = class_counts['No Cutoff'].get(class_name, 0)
    n_acc = class_counts['Overall Cutoff'].get(class_name, 0)
    n_ppv = class_counts['Confident Cutoff'].get(class_name, 0)

    # Plot ROC curve for No Cutoff
    ax.plot(
        roc_results_all[class_name]['fpr'],
        roc_results_all[class_name]['tpr'],
        color=class_colors[class_name],
        linestyle=cutoff_styles['No Cutoff']['linestyle'],
        alpha=cutoff_styles['No Cutoff']['alpha'],
        label=f'No Cutoff (AUC = {roc_results_all[class_name]["auc"]:.2f}, n={n_all})'
    )

    # Plot ROC curve for Overall Cutoff (ACC)
    ax.plot(
        roc_results_acc[class_name]['fpr'],
        roc_results_acc[class_name]['tpr'],
        color=class_colors[class_name],
        linestyle=cutoff_styles['Overall Cutoff']['linestyle'],
        alpha=cutoff_styles['Overall Cutoff']['alpha'],
        label=f'Overall Cutoff (AUC = {roc_results_acc[class_name]["auc"]:.2f}, n={n_acc})'
    )

    # Plot ROC curve for Confident Cutoff (PPV)
    ax.plot(
        roc_results_ppv[class_name]['fpr'],
        roc_results_ppv[class_name]['tpr'],
        color=class_colors[class_name],
        linestyle=cutoff_styles['Confident Cutoff']['linestyle'],
        alpha=cutoff_styles['Confident Cutoff']['alpha'],
        label=f'Confident Cutoff (AUC = {roc_results_ppv[class_name]["auc"]:.2f}, n={n_ppv})'
    )

    # Plot the diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=1)

    # Set labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve for {class_name}')
    ax.legend(loc="lower right")

    # Print AUC values and sample sizes
    print(f"AUC for {class_name}:")
    print(f"  No Cutoff: {roc_results_all[class_name]['auc']:.4f} (n={n_all})")
    print(f"  Overall Cutoff: {roc_results_acc[class_name]['auc']:.4f} (n={n_acc})")
    print(f"  Confident Cutoff: {roc_results_ppv[class_name]['auc']:.4f} (n={n_ppv})")

# Add a main title to the figure
tag = "Adult" if config['is_adult'] else "Pediatric"
plt.suptitle(f'ROC Curves for {tag} Cohorts by Leukemia Type and Cutoff Method', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the main title

# Save the combined figure
# combined_path = os.path.join(plots_dir, f'roc_{tag}_combined.png')
# fig.savefig(combined_path, dpi=300, bbox_inches='tight')
# save to svg
combined_path_svg = os.path.join(plots_dir, f'roc_{tag}_combined.svg')
fig.savefig(combined_path_svg, format='svg', bbox_inches='tight')
print(f"Saved combined plot to: {combined_path_svg}")
plt.close(fig)

# Print summary statistics
print("\nData Summary:")
print(f"Total samples: {len(df)}")
print(f"Samples with Overall Cutoff: {len(df_acc)} ({len(df_acc)/len(df)*100:.1f}%)")
print(f"Samples with Confident Cutoff: {len(df_ppv)} ({len(df_ppv)/len(df)*100:.1f}%)")

print("\nClass distribution in all data:")
print(df['class'].value_counts())

print("\nClass distribution with Overall Cutoff:")
print(df_acc['class'].value_counts())

print("\nClass distribution with Confident Cutoff:")
print(df_ppv['class'].value_counts())

print("\nCutoff values:")
print("Overall Cutoff (ACC):", acc_cutoff)
print("Confident Cutoff (PPV):", ppv_cutoff)

def run_analysis(is_adult: bool):
    """Runs the ROC analysis for a specific cohort."""
    tag = "Adult" if is_adult else "Pediatric"
    print(f"--- Running analysis for {tag} Cohort ---")

    # Load the data using util.py's load_data function with analysis config
    analysis_config_path = str(Path(__file__).parent.parent / "config" / "config_analysis.yaml")

    # Load config first to get root_dir
    with open(analysis_config_path, 'r') as f:
        temp_config = yaml.safe_load(f)

    df, config, features = load_data(config_path=analysis_config_path, root_path=temp_config['root_dir'] + '/', is_adult=is_adult)

    # Ensure config reflects the current cohort (important for output path)
    config['is_adult'] = is_adult

    # Also load training config for cutoffs
    training_config_path = str(Path(__file__).parent.parent / "config" / "config_training.yaml")
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)

    # Ensure predictions are in float format
    df[["prediction.AML", "prediction.APL", "prediction.ALL"]] = df[["prediction.AML", "prediction.APL", "prediction.ALL"]].astype(float)

    # Prune the prediction data based on the threshold
    df = prediction_data_pruner(df, threshold=0.2)

    # Identify the column with the maximum prediction value
    df["max_pred"] = df[[
        "prediction.AML",
        "prediction.APL",
        "prediction.ALL"
    ]].idxmax(axis=1)
    df["max_pred"] = df["max_pred"].str.split(".").str[1]  # Extract the class name

    # Extract cutoffs from training config
    cutoffs = {item['category']: item for item in training_config['cutoffs']}

    # Overall cutoff (ACC)
    acc_cutoff = {
        "AML": cutoffs['AML']['ACC'],
        "APL": cutoffs['APL']['ACC'],
        "ALL": cutoffs['ALL']['ACC']
    }

    # Confident cutoff (PPV)
    ppv_cutoff = {
        "AML": cutoffs['AML']['PPV'],
        "APL": cutoffs['APL']['PPV'],
        "ALL": cutoffs['ALL']['PPV']
    }

    # Filter data using cutoffs
    df_acc = df[(df["prediction.AML"] > acc_cutoff["AML"]) |
                (df["prediction.APL"] > acc_cutoff["APL"]) |
                (df["prediction.ALL"] > acc_cutoff["ALL"])]

    df_ppv = df[(df["prediction.AML"] > ppv_cutoff["AML"]) |
                (df["prediction.APL"] > ppv_cutoff["APL"]) |
                (df["prediction.ALL"] > ppv_cutoff["ALL"])]

    # Get sample sizes by class for each dataset
    class_counts = {
        'No Cutoff': df['class'].value_counts().to_dict(),
        'Overall Cutoff': df_acc['class'].value_counts().to_dict(),
        'Confident Cutoff': df_ppv['class'].value_counts().to_dict()
    }

    # Calculate ROC curves for each dataset
    roc_results_all = calculate_roc_curves(df)
    roc_results_acc = calculate_roc_curves(df_acc)
    roc_results_ppv = calculate_roc_curves(df_ppv)

    # Create the plots directory if it doesn't exist
    # Use tag in directory name
    plots_dir = os.path.join(config['root_results'], f'2_roc_auc_plot_cutoffs_{tag.lower()}')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to: {plots_dir}")

    # Extract and save ROC source data to Excel
    print("Extracting ROC source data...")
    roc_source_dir = os.path.join(plots_dir, 'roc_source_data')
    os.makedirs(roc_source_dir, exist_ok=True)

    classes = ["AML", "APL", "ALL"]
    datasets = {
        'no_cutoff': ('No Cutoff', df),
        'overall_cutoff': ('Overall Cutoff', df_acc),
        'confident_cutoff': ('Confident Cutoff', df_ppv)
    }

    for dataset_key, (dataset_name, df_data) in datasets.items():
        if df_data is None or df_data.empty:
            print(f"Skipping {dataset_name} - no data available")
            continue

        print(f"Processing {dataset_name} ({len(df_data)} samples)")

        # Create Excel file with multiple sheets for each class
        excel_path = os.path.join(roc_source_dir, f'roc_source_{tag.lower()}_{dataset_key}.xlsx')

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for class_name in classes:
                try:
                    # Extract source data for this class
                    source_df = extract_roc_source_data(df_data, class_name)
                    # Save to Excel sheet
                    source_df.to_excel(writer, sheet_name=class_name, index=False)
                    print(f"Saved {class_name} data for {dataset_name}: {len(source_df)} samples")
                except Exception as e:
                    print(f"Error extracting ROC source data for {class_name} in {dataset_name}: {e}")

        print(f"Saved ROC source data to: {excel_path}")

    # Create a combined figure with subplots for each class
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    classes = ["AML", "APL", "ALL"]
    class_colors = {'AML': 'red', 'APL': 'green', 'ALL': 'blue'}
    cutoff_styles = {
        'No Cutoff': {'linestyle': '-', 'alpha': 0.9},
        'Overall Cutoff': {'linestyle': '--', 'alpha': 0.9},
        'Confident Cutoff': {'linestyle': ':', 'alpha': 0.9}
    }

    # Plot ROC curves for each class in separate subplots
    for i, class_name in enumerate(classes):
        ax = axes[i]

        # Get sample sizes for this class
        n_all = class_counts['No Cutoff'].get(class_name, 0)
        n_acc = class_counts['Overall Cutoff'].get(class_name, 0)
        n_ppv = class_counts['Confident Cutoff'].get(class_name, 0)

        # Plot ROC curve for No Cutoff
        ax.plot(
            roc_results_all[class_name]['fpr'],
            roc_results_all[class_name]['tpr'],
            color=class_colors[class_name],
            linestyle=cutoff_styles['No Cutoff']['linestyle'],
            alpha=cutoff_styles['No Cutoff']['alpha'],
            label=f'No Cutoff (AUC = {roc_results_all[class_name]["auc"]:.2f}, n={n_all})'
        )

        # Plot ROC curve for Overall Cutoff (ACC)
        ax.plot(
            roc_results_acc[class_name]['fpr'],
            roc_results_acc[class_name]['tpr'],
            color=class_colors[class_name],
            linestyle=cutoff_styles['Overall Cutoff']['linestyle'],
            alpha=cutoff_styles['Overall Cutoff']['alpha'],
            label=f'Overall Cutoff (AUC = {roc_results_acc[class_name]["auc"]:.2f}, n={n_acc})'
        )

        # Plot ROC curve for Confident Cutoff (PPV)
        ax.plot(
            roc_results_ppv[class_name]['fpr'],
            roc_results_ppv[class_name]['tpr'],
            color=class_colors[class_name],
            linestyle=cutoff_styles['Confident Cutoff']['linestyle'],
            alpha=cutoff_styles['Confident Cutoff']['alpha'],
            label=f'Confident Cutoff (AUC = {roc_results_ppv[class_name]["auc"]:.2f}, n={n_ppv})'
        )

        # Plot the diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1)

        # Set labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve for {class_name}')
        ax.legend(loc="lower right")

        # Print AUC values and sample sizes
        print(f"AUC for {class_name} ({tag}):")
        print(f"  No Cutoff: {roc_results_all[class_name]['auc']:.4f} (n={n_all})")
        print(f"  Overall Cutoff: {roc_results_acc[class_name]['auc']:.4f} (n={n_acc})")
        print(f"  Confident Cutoff: {roc_results_ppv[class_name]['auc']:.4f} (n={n_ppv})")

    # Add a main title to the figure
    plt.suptitle(f'ROC Curves for {tag} Cohorts by Leukemia Type and Cutoff Method', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the main title

    # Save the combined figure
    # Use tag in filename
    combined_path_svg = os.path.join(plots_dir, f'roc_{tag.lower()}_combined.svg')
    fig.savefig(combined_path_svg, format='svg', bbox_inches='tight')
    print(f"Saved combined plot to: {combined_path_svg}")
    plt.close(fig)

    # Print summary statistics
    print(f"\nData Summary ({tag}):")
    print(f"Total samples: {len(df)}")
    print(f"Samples with Overall Cutoff: {len(df_acc)} ({len(df_acc)/len(df)*100:.1f}%)")
    print(f"Samples with Confident Cutoff: {len(df_ppv)} ({len(df_ppv)/len(df)*100:.1f}%)")

    print(f"\nClass distribution in all data ({tag}):")
    print(df['class'].value_counts())

    print(f"\nClass distribution with Overall Cutoff ({tag}):")
    print(df_acc['class'].value_counts())

    print(f"\nClass distribution with Confident Cutoff ({tag}):")
    print(df_ppv['class'].value_counts())

    print("\nCutoff values:")
    print("Overall Cutoff (ACC):", acc_cutoff)
    print("Confident Cutoff (PPV):", ppv_cutoff)

def main():
    """Runs the analysis for both adult and pediatric cohorts."""
    print("Starting ROC analysis...")
    run_analysis(is_adult=True)
    run_analysis(is_adult=False)
    print("\nROC analysis finished for both cohorts.")

if __name__ == "__main__":
    main()
