# Oink Oink – Datathon FME 2025 – Mango

## 📖 Descripción

Este repositorio contiene nuestro **pipeline final de predicción de demanda para Mango**, desarrollado para el **Datathon FME 2025**.

Objetivo: **predecir la cantidad óptima de producción de prendas para la próxima temporada** usando:

* Embeddings de imágenes de productos 🖼️
* Atributos de las prendas 👗
* Historial de ventas y producción 📊

La versión **`8.py`** es la final que nos permitió alcanzar **55.57900 de accuracy**, combinando los mejores modelos en un **ensemble ponderado**.

---

## Estructura del Repositorio

```
.
├── data/
│   ├── train.csv        # Datos históricos de entrenamiento
│   └── test.csv         # Datos de test para predicción 
├── 1.py … 7.py          # Versiones previas de experimentos
└── 8.py                 # Pipeline final (ensemble de finalistas)
```

---

## Pipeline Final (`8.py`)

### Pasos principales:

1. **Importación de librerías**

   * pandas, numpy, sklearn, catboost, etc.

2. **Configuración global**

   * Paths, parámetros PCA, cross-validation, pesos del ensemble

3. **Ingeniería de características**

   * Limpieza y agregación de datos
   * Parsing y PCA de embeddings de imagen
   * Features agregadas por familia, categoría y atributos
   * Normalización logarítmica de features numéricas

4. **Entrenamiento de modelos finalistas**

   * **Modelo A**: Alpha=0.78, learning_rate=0.01 (más estable)
   * **Modelo B**: Alpha=0.75, learning_rate=0.03 (más agresivo)
   * CatBoost con **K-Fold CV** para seleccionar iteraciones óptimas

5. **Ensemble ponderado**

   * 60% Modelo A + 40% Modelo B
   * Transformación inversa log1p para obtener predicciones reales

6. **Generación de submission**

   * Archivo `submission_catboost_V18_EnsembleFinalists.csv` listo para Kaggle/Datathon

---

## Requisitos

* Python >= 3.9
* pandas
* numpy
* scikit-learn
* catboost

```bash
pip install pandas numpy scikit-learn catboost
```

---

## Uso

1. Coloca `train.csv` y `test.csv` en la carpeta `data/`
2. Ejecuta el pipeline final:

```bash
python 8.py
```

3. Obtendrás `submission_catboost_V18_EnsembleFinalists.csv` con las predicciones finales.

---

## Logros y Aprendizajes

* Ensemble de modelos CatBoost alcanzó **55.57900 de accuracy**
* Feature engineering robusto fue más determinante que hiperajustar modelos complejos
* Combinación de embeddings de imagen, atributos categóricos y datos históricos multi-temporada fue clave
* Validación temporal (TimeSeriesSplit) evitó fugas de información y permitió modelos generalizables

---

## Próximos pasos

* Entrenar embeddings visuales propios
* Explorar TabNet o LightGBM con tuning automático
* Añadir interpretabilidad al pipeline para entender qué atributos generan más demanda
* Automatizar todo el flujo para producción real

---

## Créditos

Equipo **Oink Oink** – Estudiantes de Inteligencia Artificial UPC, Datathon FME 2025.
