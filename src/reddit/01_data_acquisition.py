"""
===========================================================
Fase 1: Adquisición y Preparación de Datos
Versión: Gold Standard para Reproducibilidad (Snapshot Zenodo)
===========================================================

Esta fase realiza:
1. Verifica si existe un snapshot local (para reproducibilidad).
2. Si no existe, recolecta publicaciones desde Reddit (PRAW).
3. Realiza limpieza, lematización y normalización.
4. Genera el dataset con etiqueta: Burnout | Control.
5. Guarda una copia estática 'reddit_burnout_dataset_snapshot.csv'.

IMPORTANTE:
Credenciales incluidas para ejecución inmediata.
===========================================================
"""

# ==============================
# 1. Importación de bibliotecas
# ==============================
import re
import os
import logging
from typing import Union
from datetime import datetime

import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
import praw

# ==============================
# 2. Configuración del logging
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Nombre del archivo para Zenodo
SNAPSHOT_FILENAME = "reddit_burnout_dataset_snapshot.csv"

# ==============================
# 3. Configuración de la API
# ==============================
# Credenciales activas para ejecución inmediata
reddit = praw.Reddit(
    client_id="IsipRVzS46SkCchu4LHwdw",
    client_secret="5ha9RIM9PSy6_x7KJO7cXjNk8ceHjg",
    user_agent="tesis_reddit_scraper_v1"
)

# ==============================
# 4. Palabras clave por cohorte
# ==============================
query_burnout = ["burnout", "mental health", "stress"]
query_control = ["travel", "cooking", "gardening"]

subreddits = ["all"]

# ==============================
# 5. Funciones
# ==============================
def get_posts(keywords, limit=200):
    """Descarga posts de Reddit usando PRAW"""
    posts_data = []
    try:
        for term in keywords:
            for subreddit in subreddits:
                logger.info("Buscando '%s' en r/%s...", term, subreddit)
                # search() devuelve un generador, iteramos sobre él
                for post in reddit.subreddit(subreddit).search(term, limit=limit, time_filter="month"):
                    posts_data.append({
                        "id": post.id,
                        "subreddit": post.subreddit.display_name,
                        "author": str(post.author),
                        "title": post.title,
                        "text": post.selftext,
                        "score": post.score,
                        "url": post.url,
                        "created_utc": datetime.utcfromtimestamp(post.created_utc)
                    })
    except Exception as e:
        logger.error(f"Error conectando con Reddit API: {e}")
        return pd.DataFrame()
        
    return pd.DataFrame(posts_data)

def ensure_nltk_stopwords(language: str = "english") -> None:
    try:
        stopwords.words(language)
    except LookupError:
        logger.info("Descargando 'stopwords' de NLTK...")
        nltk.download("stopwords")

def load_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        nlp_model = spacy.load(model_name)
        return nlp_model
    except OSError:
        logger.error(f"Modelo spaCy no encontrado. Ejecuta: python -m spacy download {model_name}")
        raise

def preprocess_text(text: Union[str, None], nlp, stop_words_set: set) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # Normalización básica
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|\#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Lematización
    doc = nlp(text)
    lemmas = [
        token.lemma_
        for token in doc
        if token.is_alpha and token.lemma_ not in stop_words_set
    ]

    return " ".join(lemmas)

# =====================================================
# 6. Pipeline de Procesamiento
# =====================================================
def main_processing(df_raw: pd.DataFrame, spacy_model_name: str = "en_core_web_sm") -> pd.DataFrame:
    ensure_nltk_stopwords("english")
    stop_words_set = set(stopwords.words("english"))
    nlp = load_spacy_model(spacy_model_name)

    df = df_raw.copy()

    logger.info("Realizando limpieza y lematización de texto...")
    
    # Intenta usar tqdm para barra de progreso, si no usa apply normal
    try:
        from tqdm import tqdm
        tqdm.pandas()
        df["clean_text"] = df["text"].progress_apply(lambda t: preprocess_text(t, nlp, stop_words_set))
    except ImportError:
        df["clean_text"] = df["text"].apply(lambda t: preprocess_text(t, nlp, stop_words_set))

    # Asegurar formato de fecha
    if 'created_utc' in df.columns and not pd.api.types.is_datetime64_any_dtype(df["created_utc"]):
         df["date"] = pd.to_datetime(df["created_utc"])
    elif 'date' in df.columns:
         df["date"] = pd.to_datetime(df["date"])

    logger.info("Procesamiento completado: %d registros.", len(df))
    return df

# =====================================================
# 7. Ejecución Principal
# =====================================================
# Definimos final_df como None inicialmente para evitar errores de scope
final_df = None

if __name__ == "__main__":
    
    # --- PASO 1: Verificar Snapshot (Zenodo Mode) ---
    if os.path.exists(SNAPSHOT_FILENAME):
        logger.info(f"✅ Archivo snapshot encontrado: {SNAPSHOT_FILENAME}")
        logger.info("Cargando datos estáticos para garantizar reproducibilidad...")
        labeled_df = pd.read_csv(SNAPSHOT_FILENAME)
        
        # Ajuste de fechas al cargar desde CSV
        if 'date' in labeled_df.columns:
            labeled_df['date'] = pd.to_datetime(labeled_df['date'])
            
    else:
        logger.info(f"⚠️ Snapshot no encontrado. Iniciando descarga EN VIVO desde Reddit...")
        
        # --- PASO 2: Recolección (Solo si no hay snapshot) ---
        burnout_df = get_posts(query_burnout)
        control_df = get_posts(query_control)

        if burnout_df.empty and control_df.empty:
            logger.error("No se pudieron descargar datos. Verifica tu conexión o límites de API.")
            exit()

        burnout_df["label"] = "Burnout"
        control_df["label"] = "Control"

        final_df = pd.concat([burnout_df, control_df], ignore_index=True)
        final_df["created_utc"] = pd.to_datetime(final_df["created_utc"]) # Asegurar datetime antes de procesar
        
        logger.info("Datos recopilados: %d publicaciones.", len(final_df))
        logger.info("Resumen → Burnout: %d | Control: %d", len(burnout_df), len(control_df))
        
        # --- PASO 3: Procesamiento NLP ---
        labeled_df = main_processing(final_df)
        
        # --- PASO 4: Guardado Crítico (SNAPSHOT) ---
        logger.info(f"💾 Guardando snapshot para Zenodo: {SNAPSHOT_FILENAME}")
        labeled_df.to_csv(SNAPSHOT_FILENAME, index=False, encoding='utf-8')
        print(f"\n[ATENCIÓN] Se ha generado el archivo '{SNAPSHOT_FILENAME}'.")
        print(">> SUBE ESTE ARCHIVO A ZENODO PARA ASEGURAR LA REPRODUCIBILIDAD <<")

    print("\n✅ Fase 1 completada correctamente.")
    print(labeled_df[['label', 'clean_text']].head())
