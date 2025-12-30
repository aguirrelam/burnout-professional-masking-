import pandas as pd
from textblob import TextBlob
import numpy as np
from tqdm import tqdm

# Configuración de pandas para ver bien las columnas
pd.set_option('display.max_columns', None)
tqdm.pandas() # Habilitar barra de progreso en operaciones pandas

input_file = "phase2_clean_dataset.parquet"
output_file = "phase3_features_dataset.parquet"

print("Iniciando Fase 3: Ingeniería de Características Psicométricas...")

# 1. Cargar el Golden Dataset
try:
    df = pd.read_parquet(input_file)
    print(f"Dataset cargado. Dimensiones: {df.shape}")
except Exception as e:
    print(f"Error cargando el archivo: {e}")
    exit()

# 2. Definición de Funciones Psicolingüísticas

def get_sentiment_stats(text):
    """
    Retorna polaridad (-1 muy negativo a +1 muy positivo)
    y subjetividad (0 objetivo a 1 muy subjetivo/emocional).
    """
    if not text:
        return 0.0, 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def get_structural_features(text):
    """
    Analiza la estructura: longitud, conteo de palabras, 
    y promedio de longitud de palabra (indicador de complejidad cognitiva).
    """
    if not text:
        return 0, 0
    words = text.split()
    word_count = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    return word_count, avg_word_len

# 3. Aplicación Vectorizada (Optimizada)
print("Extrayendo características (Sentimiento y Estructura)...")

# Usamos apply con una barra de progreso
# Sentimiento
df[['polarity', 'subjectivity']] = df['clean_text'].progress_apply(
    lambda x: pd.Series(get_sentiment_stats(x))
)

# Estructura
df[['word_count', 'avg_word_len']] = df['clean_text'].progress_apply(
    lambda x: pd.Series(get_structural_features(x))
)

# 4. Análisis Preliminar (Validación Estadística Rápida)
print("\n--- ANÁLISIS PRELIMINAR DE DIFERENCIAS ---")
stats = df.groupby('label')[['polarity', 'subjectivity', 'word_count']].mean()
print(stats)
print("------------------------------------------")

# Interpretación automática para el usuario
diff_polarity = stats.loc['Control', 'polarity'] - stats.loc['Burnout', 'polarity']
if diff_polarity > 0:
    print(f"Insight: El grupo Control es {diff_polarity:.4f} puntos más positivo que Burnout.")
else:
    print(f"Insight: El grupo Burnout parece más positivo (Inesperado, revisar ironía).")

# 5. Guardado Final
print(f"\nGuardando dataset enriquecido en {output_file}...")
df.to_parquet(output_file, index=False)
print("Fase 3 Completada. Listo para Modelado Predictivo.")