"""
===========================================================
Fase 4: Evaluación y Análisis del Modelo LSTM
===========================================================
Este script:

 - Ejecuta Fase 1, Fase 2 y Fase 3 automáticamente
 - Evalúa el modelo LSTM con:
        ✔ classification_report
        ✔ confusion_matrix
 - Genera visualizaciones con seaborn y matplotlib
 - Aplica SHAP para interpretabilidad

Compatible con:
 - TensorFlow 2.20
 - Keras 3.12
===========================================================
"""

# ==========================================================
# 0. Importar Fases 1, 2 y 3
# ==========================================================
import logging
import importlib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

logger.info("Cargando fases previas...")

fase_1 = importlib.import_module("fase_1")
fase_2 = importlib.import_module("fase_2_extraccion_caracteristicas")
fase_3 = importlib.import_module("fase_3")

logger.info("Ejecutando Fase 1, 2 y 3...")

# Fase 1
labeled_df = fase_1.main_processing(fase_1.final_df)

# Fase 2
full_features_df = fase_2.main_feature_extraction(labeled_df)
user_sequences = fase_2.create_time_series(full_features_df)

# Fase 3 – Necesitamos re-ejecutar su pipeline para obtener:
# X_train, X_test, y_train, y_test, le y model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---------- Preparación igual que en Fase 3 ----------
sequences = [u["sequence"] for u in user_sequences.values()]
labels = [u["label"] for u in user_sequences.values()]

# Filtrar secuencias vacías
valid_idx = [i for i, seq in enumerate(sequences) if len(seq) > 0]
sequences = [sequences[i] for i in valid_idx]
labels = [labels[i] for i in valid_idx]

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)

from keras.src.utils import pad_sequences
from keras.src.utils import to_categorical

max_len = max(len(seq) for seq in sequences)

X_padded = pad_sequences(
    sequences,
    maxlen=max_len,
    dtype="float32",
    padding="pre",
    value=0.0
)

X_train, X_test, y_train, y_test = train_test_split(
    X_padded,
    labels_encoded,
    test_size=0.2,
    stratify=labels_encoded,
    random_state=42
)

y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# ---------- Cargar modelo de Fase 3 ----------
model = fase_3.model  # el modelo definido en fase_3.py

logger.info("Datos listos. Ejecutando evaluación...")


# ==========================================================
# 1. Evaluación del modelo
# ==========================================================
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger.info("Generando predicciones...")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# -------- Reporte de clasificación --------
print("\n===================================================")
print("   REPORTE DE CLASIFICACIÓN DEL MODELO LSTM")
print("===================================================\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -------- Matriz de confusión --------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Matriz de Confusión")
plt.xlabel("Predicciones")
plt.ylabel("Etiquetas Reales")
plt.tight_layout()
plt.show()


# ==========================================================
# 2. Interpretabilidad con SHAP
# ==========================================================
logger.info("Calculando interpretabilidad SHAP...")

import shap

# Seleccionar 100 muestras para background (si existen)
bg_size = min(100, X_train.shape[0])
background = X_train[np.random.choice(X_train.shape[0], bg_size, replace=False)]

# Crear DeepExplainer
explainer = shap.DeepExplainer(model, background)

logger.info("Calculando valores SHAP (esto puede tardar)...")

shap_values = explainer.shap_values(X_test)

# Obtener nombres reales de las características (solo una vez)
feature_cols = [
    col for col in full_features_df.columns
    if any(k in col for k in ["sentiment", "lexical", "bert"])
]

# El remuestreo creó columnas multi-índice → generamos nombres:
expanded_feature_names = []
agg_stats = ["mean", "std", "max"]

for stat in agg_stats:
    for feat in feature_cols:
        expanded_feature_names.append(f"{feat}_{stat}")

# -------- Plot SHAP por clase --------
for class_idx, class_name in enumerate(le.classes_):
    print(f"\n === SHAP Summary Plot para clase: {class_name} ===\n")
    shap.summary_plot(
        shap_values[class_idx],
        feature_names=expanded_feature_names,
        plot_type="bar"
    )

print("\n===================================================")
print("           FASE 4 COMPLETADA CON ÉXITO")
print("===================================================\n")
