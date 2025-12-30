import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Iniciando Fase 4: Modelado Predictivo y Explicabilidad (TF-IDF)...")

# 1. Cargar Datos Enriquecidos
input_file = "phase3_features_dataset.parquet"
try:
    df = pd.read_parquet(input_file)
    print(f"Dataset cargado: {len(df)} registros.")
except Exception as e:
    print("Error: No se encuentra el archivo de la Fase 3.")
    exit()

# 2. Preparación de Datos
# Usaremos solo el texto limpio para predecir.
X = df['clean_text'].fillna('')
y = df['label'].map({'Control': 0, 'Burnout': 1}) # 1 = Burnout (Clase Positiva)

# División Train/Test (80% entrenamiento, 20% evaluación)
print("Dividiendo dataset (Train/Test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Vectorización (Convertir texto a números)
# TF-IDF: Da peso a palabras únicas y baja importancia a palabras muy comunes (como "the", "is")
print("Vectorizando textos (esto puede tardar un poco)...")
tfidf = TfidfVectorizer(
    max_features=5000,      # Nos quedamos solo con las 5,000 palabras más importantes para no saturar RAM
    stop_words='english',   # Eliminamos palabras vacías en inglés
    ngram_range=(1, 2)      # Consideramos palabras sueltas y pares de palabras (bigramas)
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 4. Entrenamiento del Modelo
print("Entrenando Regresión Logística...")
model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
model.fit(X_train_vec, y_train)

# 5. Evaluación
print("\n--- RESULTADOS DEL MODELO ---")
y_pred = model.predict(X_test_vec)
y_prob = model.predict_proba(X_test_vec)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Control', 'Burnout']))
auc = roc_auc_score(y_test, y_prob)
print(f"AUC-ROC Score: {auc:.4f} (0.5 es azar, 1.0 es perfecto)")

# 6. EXPLICABILIDAD CIENTÍFICA (La parte más importante para el paper)
# Extraemos qué palabras tienen los coeficientes más altos para cada clase
print("\n--- ANÁLISIS LINGÜÍSTICO (Palabras Clave) ---")
feature_names = tfidf.get_feature_names_out()
coefs = model.coef_[0]

# Crear un dataframe de importancia
word_importance = pd.DataFrame({'word': feature_names, 'coef': coefs})
word_importance = word_importance.sort_values(by='coef', ascending=False)

top_burnout = word_importance.head(20) # Coeficientes positivos altos predicen Burnout (1)
top_control = word_importance.tail(20) # Coeficientes negativos altos predicen Control (0)

print("\nTOP 20 PALABRAS ASOCIADAS A BURNOUT (Usuario Abandonó):")
print(top_burnout['word'].tolist())

print("\nTOP 20 PALABRAS ASOCIADAS A CONTROL (Usuario Activo):")
# Invertimos el orden para ver las más negativas primero
print(top_control.sort_values(by='coef')['word'].tolist())

# 7. Guardar métricas clave si es necesario
# (Opcional: guardar df con predicciones para análisis de error)
df_test = df.loc[X_test.index].copy()
df_test['pred_prob'] = y_prob
df_test['actual'] = y_test
df_test.to_csv("phase4_predictions.csv", index=False)
print("\nPredicciones guardadas en 'phase4_predictions.csv' para análisis de error.")