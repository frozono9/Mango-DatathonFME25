import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE  # Para el análisis de embeddings
import ast  # Para convertir los embeddings de string a lista
import warnings

# --- Configuración General ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['figure.autolayout'] = True
warnings.filterwarnings('ignore')

print("Librerías cargadas. Iniciando EDA...")

# --- CONFIGURACIÓN DE RUTAS ---
# !!! Usamos r'...' (raw strings) para manejar las rutas de Windows correctamente !!!
PATH_TRAIN = r'C:\Users\fabia\OneDrive\Escritorio\UNI\FME2025\data\train.csv'
PATH_TEST = r'C:\Users\fabia\OneDrive\Escritorio\UNI\FME2025\data\test.csv'

# --- PASO 0: CARGA Y AGREGACIÓN DE DATOS ---
try:
    # Cargamos el dataset de entrenamiento
    # !!! AÑADIMOS sep=';' !!!
    df_train = pd.read_csv(PATH_TRAIN, sep=';')
    print(f"train.csv cargado con éxito. Shape: {df_train.shape}")
except FileNotFoundError:
    print(f"Error: No se encontró train.csv en la ruta: {PATH_TRAIN}")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el CSV: {e}")
    exit()

# Definimos las columnas que son estáticas (a nivel de producto)
static_cols = [
    'ID', 'id_season', 'aggregated_family', 'family', 'category', 'fabric',
    'color_name', 'color_rgb', 'image_embedding', 'length_type', 'silhouette_type',
    'waist_type', 'sleeve_length_type', 'heel_shape_type', 'toecap_type',
    'woven_structure', 'knit_structure', 'print_type', 'archetype', 'moment',
    'ocassion', 'phase_in', 'phase_out', 'life_cycle_length', 'num_stores',
    'num_sizes', 'has_plus_size', 'price', 'year', 'Production'
]

# Asegurarnos de que las columnas estáticas existen antes de usarlas
static_cols = [col for col in static_cols if col in df_train.columns]

if 'ID' not in df_train.columns:
    print("Error: La columna 'ID' no se encuentra en train.csv. ¿Estás seguro de que el separador es ';'? ")
    exit()
if 'weekly_demand' not in df_train.columns:
    print("Error: La columna 'weekly_demand' no se encuentra en train.csv.")
    exit()

print("Agregando datos semanales a nivel de producto (ID)...")

# Agregamos los datos semanales para tener un solo registro por producto (ID)
df_agg = df_train.groupby('ID').agg(
    Total_Demand=('weekly_demand', 'sum'),
    Total_Sales=('weekly_sales', 'sum'),
    Avg_Weekly_Demand=('weekly_demand', 'mean'),
    Max_Weekly_Demand=('weekly_demand', 'max')
).reset_index()

# Obtenemos los datos estáticos (únicos por ID)
df_static = df_train[static_cols].drop_duplicates(subset='ID').set_index('ID')

# Unimos los datos agregados con los estáticos
df_producto = df_static.join(df_agg.set_index('ID')).reset_index()

print(f"Datos agregados. Nuevo shape (productos únicos): {df_producto.shape}")


# --- PASO 1: ANÁLISIS DE LA VARIABLE OBJETIVO (Total_Demand) ---
print("\n--- Iniciando Paso 1: Análisis del Objetivo ---")

plt.figure(figsize=(16, 6))
plt.suptitle('Análisis de la Variable Objetivo (Total_Demand)', fontsize=18, y=1.03)

# 1. Histograma de Total_Demand
plt.subplot(1, 2, 1)
sns.histplot(df_producto['Total_Demand'], kde=True, bins=50, color='blue')
plt.title('Distribución de Total_Demand')
plt.xlabel('Demanda Total')
plt.ylabel('Frecuencia')
plt.text(0.6, 0.9, 'Muy sesgada a la derecha\n(Right-skewed)', transform=plt.gca().transAxes, style='italic')

# 2. Histograma de log(Total_Demand)
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df_producto['Total_Demand']), kde=True, bins=50, color='salmon')
plt.title('Distribución de log(1 + Total_Demand)')
plt.xlabel('log(Demanda Total)')
plt.ylabel('Frecuencia')
plt.text(0.6, 0.9, 'Distribución más "Normal"', transform=plt.gca().transAxes, style='italic')

plt.tight_layout()
plt.show()


# --- PASO 2: ANÁLISIS DE VARIABLES NUMÉRICAS ---
print("\n--- Iniciando Paso 2: Análisis Numérico ---")

# 1. Matriz de Correlación
num_features = ['num_stores', 'price', 'life_cycle_length', 'num_sizes', 'Production', 'Total_Sales', 'Total_Demand']
num_features = [col for col in num_features if col in df_producto.columns] # Solo columnas existentes
corr_matrix = df_producto[num_features].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlación de Variables Numéricas Clave', fontsize=16)
plt.show()

# 2. Scatter plots contra Total_Demand
print("Generando scatter plots clave...")
plt.figure(figsize=(20, 6))
plt.suptitle('Análisis Bivariante (Numérico vs. Total_Demand)', fontsize=18, y=1.03)

if 'num_stores' in df_producto.columns:
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df_producto, x='num_stores', y='Total_Demand', alpha=0.3, s=20)
    plt.title('num_stores vs Total_Demand')
    plt.xlabel('Número de Tiendas')
    plt.ylabel('Demanda Total')
else:
    print("No se encontró 'num_stores' para plotear.")

if 'price' in df_producto.columns:
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df_producto, x='price', y='Total_Demand', alpha=0.3, s=20)
    plt.title('price vs Total_Demand')
    plt.xlabel('Precio')
    plt.ylabel('Demanda Total')
else:
    print("No se encontró 'price' para plotear.")

if 'life_cycle_length' in df_producto.columns:
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df_producto, x='life_cycle_length', y='Total_Demand', alpha=0.3, s=20)
    plt.title('life_cycle_length vs Total_Demand')
    plt.xlabel('Semanas de Venta')
    plt.ylabel('Demanda Total')
else:
    print("No se encontró 'life_cycle_length' para plotear.")

plt.tight_layout()
plt.show()


# --- PASO 3: ANÁLISIS DE VARIABLES CATEGÓRICAS ---
print("\n--- Iniciando Paso 3: Análisis Categórico ---")

# 1. Boxplot para 'family'
try:
    top_15_families = df_producto['family'].value_counts().nlargest(15).index
    df_plot_family = df_producto[df_producto['family'].isin(top_15_families)]
    family_order = df_plot_family.groupby('family')['Total_Demand'].median().sort_values().index

    plt.figure(figsize=(18, 8))
    sns.boxplot(data=df_plot_family, x='family', y='Total_Demand', order=family_order, showfliers=False)
    plt.title('Total_Demand (log scale) por Familia de Producto (Top 15)', fontsize=16)
    plt.xlabel('Familia')
    plt.ylabel('Total_Demand')
    plt.yscale('log') # Usamos escala logarítmica
    plt.xticks(rotation=45, ha='right')
    plt.show()
except KeyError:
    print("No se encontró la columna 'family'. Saltando este gráfico.")


# 2. Boxplot para 'ocassion'
try:
    plt.figure(figsize=(12, 7))
    ocassion_order = df_producto.groupby('ocassion')['Total_Demand'].median().sort_values().index
    sns.boxplot(data=df_producto, x='ocassion', y='Total_Demand', order=ocassion_order, showfliers=False)
    plt.title('Total_Demand (log scale) por Ocasión', fontsize=16)
    plt.xlabel('Ocasión')
    plt.ylabel('Total_Demand')
    plt.yscale('log') # Usamos escala logarítmica
    plt.xticks(rotation=45, ha='right')
    plt.show()
except KeyError:
    print("No se encontró la columna 'ocassion'. Saltando este gráfico.")


# --- PASO 4: ANÁLISIS DE FEATURES "ESPECIALES" ---
print("\n--- Iniciando Paso 4: Análisis de Features Especiales ---")

# 1. Análisis de Fechas (Estacionalidad)
try:
    # !!! AÑADIMOS dayfirst=True !!!
    df_producto['phase_in_dt'] = pd.to_datetime(df_producto['phase_in'], dayfirst=True)
    df_producto['phase_in_month'] = df_producto['phase_in_dt'].dt.month

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_producto, x='phase_in_month', y='Total_Demand', showfliers=False)
    plt.title('Demanda (log scale) por Mes de Lanzamiento (phase_in_month)', fontsize=16)
    plt.xlabel('Mes de Lanzamiento')
    plt.ylabel('Total_Demand')
    plt.yscale('log')
    plt.gca().set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    plt.show()
except KeyError:
    print("No se encontró la columna 'phase_in'. Saltando gráfico de estacionalidad.")
except Exception as e:
    print(f"Error al procesar 'phase_in': {e}. Saltando gráfico de estacionalidad.")


# 2. Análisis del Gap de Producción (El Problema de Negocio)
print("Generando el gráfico CLAVE: Problema de Negocio...")
try:
    plt.figure(figsize=(10, 10))
    df_sample_plot = df_producto.sample(n=min(10000, len(df_producto)), random_state=42)
    
    sns.scatterplot(data=df_sample_plot, x='Production', y='Total_Demand', alpha=0.3, s=20)

    max_val = max(df_producto['Production'].max(), df_producto['Total_Demand'].max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Producción = Demanda')

    plt.title('El Problema de Negocio: Producción vs. Demanda Potencial', fontsize=16)
    plt.xlabel('Producción (Stock Pedido)')
    plt.ylabel('Demanda Total (Potencial de Venta)')
    plt.legend()
    plt.text(max_val*0.1, max_val*0.8, 'Ventas Perdidas\n(Demanda > Producción)', color='red', fontsize=12, weight='bold')
    plt.text(max_val*0.6, max_val*0.2, 'Exceso de Stock\n(Producción > Demanda)', color='blue', fontsize=12, weight='bold')
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.show()
except KeyError:
    print("No se encontraron 'Production' o 'Total_Demand'. Saltando gráfico de Gap de Producción.")


# 3. Análisis de Embeddings (¡Puede ser lento!)
print("\nIniciando análisis de Embeddings con t-SNE...")
print("Esto puede tardar varios minutos. Usaremos una muestra de 5000 productos.")

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
            plt.title('Visualización de Embeddings (t-SNE) coloreados por log(Total_Demand)', fontsize=16)
            plt.xlabel('Componente t-SNE 1')
            plt.ylabel('Componente t-SNE 2')
            plt.legend(title='log(Demanda)')
            plt.show()
        else:
            print("La matriz de embeddings está vacía o tiene dimensión 0.")
    else:
        print("No se pudieron parsear los embeddings válidos. Saltando este paso.")
except KeyError:
    print("No se encontró la columna 'image_embedding'. Saltando análisis de embeddings.")
except Exception as e:
    print(f"Ocurrió un error inesperado en el análisis de embeddings: {e}")

print("\n--- EDA Completado ---")
print("Observa los gráficos generados para entender los datos.")
print("El siguiente paso sería la Ingeniería de Características y el Modelado (ej. CatBoost).")