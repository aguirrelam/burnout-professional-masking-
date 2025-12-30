"""
===========================================================
Fase 3: Modelado y Clasificación Secuencial (Keras 3.12)
===========================================================
"""

# ==========================================================
# 0. Importar fases previas
# ==========================================================
import logging
import importlib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

logger.info("Cargando Fase 1 y Fase 2...")

fase_1 = importlib.import_module("fase_1")
fase_2 = importlib.import_module("fase_2_extraccion_caracteristicas")

logger.info("Ejecutando Fase 1 y Fase 2...")

# Ejecutar Fase 1
labeled_df = fase_1.main_processing(fase_1.final_df)

# Ejecutar Fase 2
full_features_df = fase_2.main_feature_extraction(labeled_df)
user_sequences = fase_2.create_time_series(full_features_df)

if len(user_sequences) < 2:
    raise SystemExit("ERROR: No hay suficientes usuarios para entrenar LSTM.")

logger.info("Datos listos. Iniciando Fase 3...")


# ==========================================================
# 1. Importaciones Keras 3.12
# ==========================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Rutas correctas en Keras 3.12
from keras.src.utils import pad_sequences, to_categorical
from keras.src.utils import to_categorical
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout, Masking


# ==========================================================
# 2. Preparar datos
# ==========================================================
logger.info("Preparando secuencias...")

sequences = [u["sequence"] for u in user_sequences.values()]
labels = [u["label"] for u in user_sequences.values()]

# Filtrar secuencias vacías
valid_idx = [i for i, seq in enumerate(sequences) if len(seq) > 0]

sequences = [sequences[i] for i in valid_idx]
labels = [labels[i] for i in valid_idx]

# Codificar etiquetas
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)

logger.info(f"Clases detectadas: {list(le.classes_)}")

# Padding
max_len = max(len(seq) for seq in sequences)

X_padded = pad_sequences(
    sequences,
    maxlen=max_len,
    dtype="float32",
    padding="pre",
    value=0.0
)

logger.info(f"Shape final de X_padded: {X_padded.shape}")

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_padded,
    labels_encoded,
    test_size=0.2,
    stratify=labels_encoded,
    random_state=42
)

# One-hot
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)


# ==========================================================
# 3. Construcción del modelo LSTM
# ==========================================================
logger.info("Construyendo modelo LSTM...")

input_shape = (X_train.shape[1], X_train.shape[2])

model = Sequential([
    Masking(mask_value=0.0, input_shape=input_shape),

    LSTM(128, return_sequences=True),
    Dropout(0.3),

    LSTM(64),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================================
# 4. Entrenamiento
# ==========================================================
logger.info("Entrenando modelo...")

history = model.fit(
    X_train,
    y_train_cat,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

logger.info("Entrenamiento completado.")

# Guardar modelo
model.save("modelo_lstm_burnout.keras")


# ==========================================================
# 5. Evaluación
# ==========================================================
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)

print("\n===================================================")
print("          RESULTADOS DEL MODELO LSTM")
print("===================================================")
print(f"Accuracy en test: {acc:.4f}")
print(f"Pérdida en test: {loss:.4f}")
print("===================================================\n")

