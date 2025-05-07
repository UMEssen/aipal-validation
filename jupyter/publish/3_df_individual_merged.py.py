# %%
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from utils.data_loader import load_data
import warnings
warnings.filterwarnings("ignore")

df, config, features = load_data()


# %%
def prediction_data_pruner(df, threshold=0):
    """Remove data rows based on the threshold of missing values."""
    data = df
    mandatory_columns = ['Fibrinogen_g_L', 'LDH_UI_L', 'WBC_G_L', 'Lymphocytes_G_L',
    'MCHC_g_L', 'MCV_fL', 'Monocytes_G_L', 'Platelets_G_L',
    'PT_percent']

    ['MCV_fL', 'PT_percent', 'LDH_UI_L', 'MCHC_g_L', 'WBC_G_L', 'Fibrinogen_g_L', 'Monocytes_G_L', 'Platelets_G_L', 'Lymphocytes_G_L']

    data["nan_percentage"] = data[mandatory_columns].isna().mean(axis=1)

    # Prune data where the percentage of NaN values is above the threshold
    data = data[data["nan_percentage"] <= threshold]
    data.drop(columns=["nan_percentage"], inplace=True)
    return data


def get_confusion_matrix(df, ax, title, do_normalize='true', min_samples=10):
    df = df.copy()

    # Check if we have enough samples
    if len(df) < min_samples:
        ax.text(0.5, 0.5, f'Insufficient samples\n(n={len(df)})',
                ha='center', va='center')
        ax.set_title(title)
        return

    # Convert prediction columns to float
    df[["prediction.ALL", "prediction.AML", "prediction.APL"]] = df[["prediction.ALL", "prediction.AML", "prediction.APL"]].astype(float)

    # if not title == 'Antananarivo':
    #     df = prediction_data_pruner(df, threshold=0.2)

    # Check if we still have samples after pruning
    if len(df) == 0:
        ax.text(0.5, 0.5, 'No samples after pruning',
                ha='center', va='center')
        ax.set_title(title)
        return

    # Get the predicted class by finding the column name with max value
    max_pred_cols = df[["prediction.ALL", "prediction.AML", "prediction.APL"]].idxmax(axis=1)
    df["max_pred"] = max_pred_cols.apply(lambda x: x.split(".")[1])

    cm_display = ConfusionMatrixDisplay.from_predictions(df["class"], df["max_pred"],
                                                       ax=ax, colorbar=False,
                                                       normalize=do_normalize)
    ax.set_title(f"{title}\n(n={len(df)})")

# %%
import matplotlib.pyplot as plt

# Define the order of cities by region
europe = ["Hannover", "Bochum", "Barcelona", "Salamanca", "Milano", "Rome", "Maastricht", "Wroclaw"]
asia_oceania = ["Suzhou", "Kolkata", "Turkey", "Melbourne"]
americas_africa = ["Sao paulo", "Buenos aires", "Dallas", "Antananarivo", "Lagos"]

# Combine all regions in the desired order
ordered_cities = europe + asia_oceania + americas_africa

# Create the figure with subplots
n_cities = df['city_country'].nunique()
n_rows = (n_cities + 2) // 3  # Calculate rows needed for 3 columns
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()
sum_of_samples = 0

# Filter and order the cities that exist in the dataset
available_cities = df['city_country'].unique()
ordered_cities_available = [city for city in ordered_cities if city in available_cities]

for i, city_country in enumerate(ordered_cities_available):
    df_small = df[df['city_country'] == city_country]
    print(f"size of {city_country}: {df_small.shape[0]}")
    sum_of_samples += df_small.shape[0]
    get_confusion_matrix(df_small, axes[i], title=city_country)

print(f"sum of samples: {sum_of_samples}")

# Hide any unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig('plots/cm_relative_all.svg')
plt.show()

# %%
import matplotlib.pyplot as plt

# Create the figure with subplots
n_cities = df['city_country'].nunique()
# filter country with less than 10 samples
n_rows = (n_cities + 2) // 3  # Calculate rows needed for 3 columns
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, city_country in enumerate(ordered_cities_available):
    df_small = df[df['city_country'] == city_country]
    get_confusion_matrix(df_small, axes[i], title=city_country, do_normalize=None)

# Hide any unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# %%
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Create the figure with subplots
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))

# Add main title at the top
kids_adult = 'Adult' if config['is_adult'] else 'Pediatric'
fig.suptitle(f'ROC Curves for {kids_adult} Cohorts', fontsize=16)

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, city_country in enumerate(ordered_cities_available):
    df_small = df[df['city_country'] == city_country].copy()
    df_small[["prediction.ALL", "prediction.AML", "prediction.APL"]] = df_small[["prediction.ALL", "prediction.AML", "prediction.APL"]].astype(float)

    # Binarize the class labels
    classes = ["ALL", "AML", "APL"]
    y = label_binarize(df_small['class'], classes=classes)
    n_classes = y.shape[1]

    # Convert predictions to a similar format for ROC calculation
    y_score = df_small[["prediction.ALL", "prediction.AML", "prediction.APL"]].values

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for j in range(n_classes):
        fpr[j], tpr[j], _ = roc_curve(y[:, j], y_score[:, j])
        roc_auc[j] = auc(fpr[j], tpr[j])

    # Plot all ROC curves
    ax = axes[i]
    colors = ['blue', 'red', 'green']
    for j, color in zip(range(n_classes), colors):
        ax.plot(fpr[j], tpr[j], color=color, lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(classes[j], roc_auc[j]))

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve for {city_country}')
    ax.legend(loc="lower right")

# Hide any unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust to make room for the suptitle
plt.savefig('Fig2_ROC_curves.svg')
plt.show()



# %%
