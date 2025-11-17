import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE  # For embeddings analysis (t-SNE)
import ast  # Para convertir los embeddings de string a lista
import warnings

# --- Configuración General ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['figure.autolayout'] = True
warnings.filterwarnings('ignore')

print("Libraries loaded. Starting EDA...")

from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
PATH_TRAIN = DATA_DIR / 'train.csv'
PATH_TEST = DATA_DIR / 'test.csv'

# --- STEP 0: LOAD AND AGGREGATE DATA ---
try:
    # Load training dataset (semicolon delimited)
    df_train = pd.read_csv(PATH_TRAIN, sep=';')
    print(f"train.csv loaded. Shape: {df_train.shape}")
except FileNotFoundError:
    print(f"Error: train.csv not found at {PATH_TRAIN}")
    exit()
except Exception as e:
    print(f"An error occurred while reading the CSV: {e}")
    exit()

# Define the columns that are static (product-level)
static_cols = [
    'ID', 'id_season', 'aggregated_family', 'family', 'category', 'fabric',
    'color_name', 'color_rgb', 'image_embedding', 'length_type', 'silhouette_type',
    'waist_type', 'sleeve_length_type', 'heel_shape_type', 'toecap_type',
    'woven_structure', 'knit_structure', 'print_type', 'archetype', 'moment',
    'ocassion', 'phase_in', 'phase_out', 'life_cycle_length', 'num_stores',
    'num_sizes', 'has_plus_size', 'price', 'year', 'Production'
]

# Ensure we only reference static columns that exist
static_cols = [col for col in static_cols if col in df_train.columns]

if 'ID' not in df_train.columns:
    print("Error: The 'ID' column is missing from train.csv. Is the separator correct (sep=';')?")
    exit()
if 'weekly_demand' not in df_train.columns:
    print("Error: The 'weekly_demand' column is missing from train.csv.")
    exit()

print("Aggregating weekly data to product level (ID)...")

# Aggregate weekly rows into a single product-level record (ID)
df_agg = df_train.groupby('ID').agg(
    Total_Demand=('weekly_demand', 'sum'),
    Total_Sales=('weekly_sales', 'sum'),
    Avg_Weekly_Demand=('weekly_demand', 'mean'),
    Max_Weekly_Demand=('weekly_demand', 'max')
).reset_index()

# Obtenemos los datos estáticos (únicos por ID)
df_static = df_train[static_cols].drop_duplicates(subset='ID').set_index('ID')

# Join aggregated stats with static product attributes
df_producto = df_static.join(df_agg.set_index('ID')).reset_index()

print(f"Aggregated data. New shape (unique products): {df_producto.shape}")


# --- STEP 1: TARGET ANALYSIS (Total_Demand) ---
print("\n--- Step 1: Target Analysis ---")

plt.figure(figsize=(16, 6))
plt.suptitle('Target analysis (Total_Demand)', fontsize=18, y=1.03)

# 1. Histograma de Total_Demand
plt.subplot(1, 2, 1)
sns.histplot(df_producto['Total_Demand'], kde=True, bins=50, color='blue')
plt.title('Distribution of Total_Demand')
plt.xlabel('Total Demand')
plt.ylabel('Frequency')
plt.text(0.6, 0.9, 'Right-skewed (long tail to the right)', transform=plt.gca().transAxes, style='italic')

# 2. Histograma de log(Total_Demand)
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df_producto['Total_Demand']), kde=True, bins=50, color='salmon')
plt.title('Distribution of log(1 + Total_Demand)')
plt.xlabel('log(Total Demand)')
plt.ylabel('Frequency')
plt.text(0.6, 0.9, 'Closer to a normal distribution', transform=plt.gca().transAxes, style='italic')

plt.tight_layout()
plt.show()


# --- STEP 2: NUMERIC VARIABLES ANALYSIS ---
print("\n--- Step 2: Numeric Variables Analysis ---")

# 1. Matriz de Correlación
num_features = ['num_stores', 'price', 'life_cycle_length', 'num_sizes', 'Production', 'Total_Sales', 'Total_Demand']
num_features = [col for col in num_features if col in df_producto.columns] # Only existing features
corr_matrix = df_producto[num_features].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Key Numeric Features', fontsize=16)
plt.show()

# 2. Scatter plots contra Total_Demand
print("Generating key scatter plots...")
plt.figure(figsize=(20, 6))
plt.suptitle('Bivariate analysis (Numeric vs. Total_Demand)', fontsize=18, y=1.03)

if 'num_stores' in df_producto.columns:
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df_producto, x='num_stores', y='Total_Demand', alpha=0.3, s=20)
    plt.title('num_stores vs Total_Demand')
    plt.xlabel('Number of stores')
    plt.ylabel('Total Demand')
else:
    print("No 'num_stores' column found; skipping plot.")

if 'price' in df_producto.columns:
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df_producto, x='price', y='Total_Demand', alpha=0.3, s=20)
    plt.title('price vs Total_Demand')
    plt.xlabel('Price')
    plt.ylabel('Total Demand')
    plt.ylabel('Total Demand')
else:
    print("No 'price' column found; skipping plot.")

if 'life_cycle_length' in df_producto.columns:
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df_producto, x='life_cycle_length', y='Total_Demand', alpha=0.3, s=20)
    plt.title('life_cycle_length vs Total_Demand')
    plt.xlabel('Weeks on sale')
    plt.ylabel('Total Demand')
    plt.ylabel('Total Demand')
else:
    print("No 'life_cycle_length' column found; skipping plot.")

plt.tight_layout()
plt.show()


# --- STEP 3: CATEGORICAL VARIABLES ANALYSIS ---
print("\n--- Step 3: Categorical Variables Analysis ---")

# 1. Boxplot para 'family'
try:
    top_15_families = df_producto['family'].value_counts().nlargest(15).index
    df_plot_family = df_producto[df_producto['family'].isin(top_15_families)]
    family_order = df_plot_family.groupby('family')['Total_Demand'].median().sort_values().index

    plt.figure(figsize=(18, 8))
    sns.boxplot(data=df_plot_family, x='family', y='Total_Demand', order=family_order, showfliers=False)
    plt.title('Total_Demand (log scale) by Product Family (Top 15)', fontsize=16)
    plt.xlabel('Family')
    plt.ylabel('Total_Demand')
    plt.yscale('log') # Usamos escala logarítmica
    plt.yscale('log') # Use logarithmic scale
    plt.xticks(rotation=45, ha='right')
    plt.show()
except KeyError:
    print("The 'family' column was not found; skipping this plot.")


# 2. Boxplot for 'ocassion'
try:
    plt.figure(figsize=(12, 7))
    ocassion_order = df_producto.groupby('ocassion')['Total_Demand'].median().sort_values().index
    sns.boxplot(data=df_producto, x='ocassion', y='Total_Demand', order=ocassion_order, showfliers=False)
    plt.title('Total_Demand (log scale) by Occasion', fontsize=16)
    plt.xlabel('Occasion')
    plt.ylabel('Total_Demand')
    plt.yscale('log') # Usamos escala logarítmica
    plt.xticks(rotation=45, ha='right')
    plt.show()
except KeyError:
    print("The 'ocassion' column was not found; skipping this plot.")


# --- STEP 4: SPECIAL FEATURES ANALYSIS ---
print("\n--- Step 4: Special Feature Analysis ---")

# 1. Date analysis (seasonality)
try:
    # !!! AÑADIMOS dayfirst=True !!!
    df_producto['phase_in_dt'] = pd.to_datetime(df_producto['phase_in'], dayfirst=True)
    df_producto['phase_in_month'] = df_producto['phase_in_dt'].dt.month

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_producto, x='phase_in_month', y='Total_Demand', showfliers=False)
    plt.title('Demand (log scale) by Launch Month (phase_in_month)', fontsize=16)
    plt.xlabel('Launch month')
    plt.ylabel('Total_Demand')
    plt.yscale('log')
    plt.gca().set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.show()
except KeyError:
    print("'phase_in' column not found; skipping seasonality plot.")
except Exception as e:
    print(f"Error processing 'phase_in': {e}. Skipping seasonality plot.")


# 2. Production gap analysis (Business problem exploration)
print("Generating the key plot: Business problem (Production vs Demand)...")
try:
    plt.figure(figsize=(10, 10))
    df_sample_plot = df_producto.sample(n=min(10000, len(df_producto)), random_state=42)
    
    sns.scatterplot(data=df_sample_plot, x='Production', y='Total_Demand', alpha=0.3, s=20)

    max_val = max(df_producto['Production'].max(), df_producto['Total_Demand'].max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Production = Demand')

    plt.title('Business Problem: Production vs Potential Demand', fontsize=16)
    plt.xlabel('Production (Ordered stock)')
    plt.ylabel('Total Demand (Potential sales)')
    plt.legend()
    plt.text(max_val*0.1, max_val*0.8, 'Lost sales\n(Demand > Production)', color='red', fontsize=12, weight='bold')
    plt.text(max_val*0.6, max_val*0.2, 'Excess stock\n(Production > Demand)', color='blue', fontsize=12, weight='bold')
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.show()
except KeyError:
    print("'Production' or 'Total_Demand' not found; skipping the production gap plot.")


# 3. Embeddings analysis (can be slow)
print("\nStarting embeddings analysis with t-SNE...")
print("This can take a few minutes. We'll sample up to 5,000 products.")

try:
    def parse_embedding(embed_str):
        try:
            return ast.literal_eval(embed_str)
        except (ValueError, SyntaxError):
            try:
                return [float(x) for x in embed_str.strip('[]').split()]
            except:
                return None

    sample_size = min(5000, len(df_producto))
    df_sample = df_producto.sample(n=sample_size, random_state=42)
    embeddings_list = df_sample['image_embedding'].apply(parse_embedding).tolist()

    valid_embeddings = [emb for emb in embeddings_list if emb is not None]
    
    if valid_embeddings:
        dim = len(valid_embeddings[0])
        embeddings_matrix = []
        for emb in embeddings_list:
            if emb is not None and len(emb) == dim:
                embeddings_matrix.append(emb)
            else:
                embeddings_matrix.append([0.0] * dim)
                
        embeddings_matrix = np.array(embeddings_matrix)

        if embeddings_matrix.shape[1] > 0:
            tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=42, init='pca', learning_rate='auto')
            tsne_results = tsne.fit_transform(embeddings_matrix)

            df_sample['tsne_1'] = tsne_results[:, 0]
            df_sample['tsne_2'] = tsne_results[:, 1]
            df_sample['log_Total_Demand'] = np.log1p(df_sample['Total_Demand'])

            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=df_sample,
                x='tsne_1',
                y='tsne_2',
                hue='log_Total_Demand',
                palette='coolwarm',
                alpha=0.7,
                s=25
            )
            plt.title('Embeddings visualization (t-SNE) colored by log(Total_Demand)', fontsize=16)
            plt.xlabel('Componente t-SNE 1')
            plt.ylabel('Componente t-SNE 2')
            plt.legend(title='log(Demand)')
            plt.show()
        else:
            print("La matriz de embeddings está vacía o tiene dimensión 0.")
    else:
        print("Could not parse valid embeddings; skipping t-SNE.")
except KeyError:
    print("No 'image_embedding' column found; skipping embeddings analysis.")
except Exception as e:
    print(f"An unexpected error occurred while analyzing embeddings: {e}")

print("\n--- EDA completed ---")
print("Inspect the graphs generated to understand the dataset. Next: feature engineering and modeling (e.g., CatBoost).")