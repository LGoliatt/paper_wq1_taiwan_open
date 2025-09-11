# eda_wq_taiwan.py
# Exploratory Data Analysis (EDA) of Taiwan Water Quality dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import requests
from sklearn.model_selection import train_test_split

# -------------------------------
# Matplotlib & Seaborn settings
# -------------------------------
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')

sns.set_style("white")
sns.set_context("paper", font_scale=1.2, rc={
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# -------------------------------
# Function to load dataset
# -------------------------------
def read_wq_taiwan(test_size=0.25, seed=42):
    """
    Load and preprocess Taiwan water quality dataset.
    """
    key = '1a5DReajqstsnUSUdTcRm8pZqeIP9ZmOct834UcOLmjg'
    link = 'https://docs.google.com/spreadsheet/ccc?key=' + key + '&output=csv'
    r = requests.get(link)
    data = r.content
    df = pd.read_csv(BytesIO(data), header=0)

    cols = ['siteid', 'sampledate', 'itemengabbreviation', 'itemvalue']
    data = df[cols]
    data = data.pivot(index=['siteid', 'sampledate'], 
                      columns='itemengabbreviation', 
                      values='itemvalue')
    data['site'] = [data.index[i][0] for i in range(len(data))]
    data = data[data['site'] < 1008]

    cols = ['EC', 'RPI', 'SS', 'WT', 'pH']
    X = data[cols]

    for c in cols:
        X[c] = pd.to_numeric(X[c], errors='coerce')

    X.dropna(inplace=True)

    variable_names = ['EC', 'SS', 'WT', 'pH']
    target_names = ['RPI']
    X_train, X_test, y_train, y_test = train_test_split(
        X[variable_names], X[target_names], 
        test_size=test_size, random_state=seed
    )

    dataset = {
        'task': 'regression',
        'name': 'WQ Taiwan',
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'X_train': X_train.values,
        'y_train': y_train.values.ravel(),
        'X_test': X_test.values,
        'y_test': y_test.values.ravel(),
    }
    return dataset['X_train'], dataset['X_test'], dataset['y_train'], dataset['y_test']

# -------------------------------
# Load dataset
# -------------------------------
X_train, X_test, y_train, y_test = read_wq_taiwan(seed=42)

feature_names = ['EC', 'SS', 'WT', 'pH']
X_full = np.vstack((X_train, X_test))
y_full = np.concatenate((y_train, y_test))
df = pd.DataFrame(X_full, columns=feature_names)
df['RPI'] = y_full

# -------------------------------
# Basic statistics
# -------------------------------
stats = df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
stats.columns = ['Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
stats = stats.round(2)

print("Table: Basic statistics of the dataset.")
print(stats)

with open('basic_statistics.tex', 'w') as f:
    f.write(stats.to_latex(caption='Basic statistics of the WQ Taiwan dataset.', 
                           label='tab:basic_stats', float_format="%.2f"))

# -------------------------------
# Correlation with RPI (barplot)
# -------------------------------
correlations = df.corr()['RPI'].drop('RPI').sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(5, 3.5))
sns.barplot(x=correlations.values, y=correlations.index, 
            palette='RdBu_r', ax=ax, edgecolor='black')
ax.set_title(r'Pearson Correlations with $RPI$', pad=10)
ax.set_xlabel('Correlation Coefficient')
ax.set_ylabel('Variables')

for i, v in enumerate(correlations.values):
    ax.text(v + (0.02 if v > 0 else -0.02), i, f"{v:.2f}", 
            va='center', ha='left' if v > 0 else 'right', fontsize=9)

sns.despine()
plt.tight_layout()
plt.savefig('correlation_barplot.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# Correlation matrix table
# -------------------------------
corr_matrix = df.corr().round(2)
print("\nTable: Correlation matrix.")
print(corr_matrix)
with open('correlation_matrix.tex', 'w') as f:
    f.write(corr_matrix.to_latex(caption='Pearson correlation matrix of variables.', 
                                 label='tab:corr_matrix', float_format="%.2f"))

# -------------------------------
# Histograms
# -------------------------------
fig, axes = plt.subplots(2, 3, figsize=(9, 5))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    sns.histplot(df[col], kde=True, bins=30, color='steelblue',
                 line_kws={"linewidth": 1.2}, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Frequency')

sns.despine()
plt.tight_layout()
plt.savefig('histograms.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# Boxplots
# -------------------------------
fig, ax = plt.subplots(figsize=(6, 3.5))
sns.boxplot(data=df, palette='pastel', fliersize=2, linewidth=1.2, ax=ax)

ax.set_title('Box Plots of Variables', pad=10)
ax.set_ylabel('Values')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

sns.despine()
plt.tight_layout()
plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# Pair plot
# -------------------------------
pair = sns.pairplot(df, corner=True, diag_kind='kde',
                    plot_kws={'alpha': 0.6, 's': 20, 'edgecolor': 'k'})
pair.fig.suptitle('Pair Plot of Variables', y=1.02)

plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# Heatmap (lower triangle)
# -------------------------------
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap='RdBu_r', vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.7, "label": "Correlation"},
            linewidths=0.5, ax=ax)

ax.set_title('Correlation Heatmap (Lower Triangle)', pad=10)
plt.tight_layout()
plt.savefig('correlation_heatmap_triangular.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# Additional dataset info
# -------------------------------
print(f"\nNumber of samples: {len(df)}")
print(f"Number of duplicates: {df.duplicated().sum()}")
print("Percentage of missing values:\n", df.isnull().mean() * 100)
