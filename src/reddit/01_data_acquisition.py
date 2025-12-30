"""
===========================================================
Fase 1: Adquisición y Preparación de Datos
Versión final para clasificación Burnout vs Control
===========================================================

Esta fase realiza:
1. Recolección de publicaciones desde Reddit (PRAW)
2. Limpieza, lematización y normalización del texto
3. Generación del dataset con etiqueta: Burnout | Control
4. Prepara 'labeled_df' para la Fase 2

IMPORTANTE:
No se usa etiqueta temporal (label_proxy) en esta versión.
===========================================================
"""

# ==============================
# 1. Importación de bibliotecas
# ==============================
import re
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

# ==============================
# 3. Configuración de la API
# ==============================
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",      
    client_secret="YOUR_SECRET_KEY", 
    user_agent="research_scraper_v1"
)

# ==============================
# 4. Palabras clave por cohorte
# ==============================
query_burnout = ["burnout", "mental health", "stress"]
query_control = ["travel", "cooking", "gardening"]

subreddits = ["all"]

# ==============================
# 5. Función de recolección
# ==============================
def get_posts(keywords, limit=200):
    posts_data = []
    for term in keywords:
        for subreddit in subreddits:
            logger.info("Buscando '%s' en r/%s...", term, subreddit)
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
    return pd.DataFrame(posts_data)

# ==============================
# 6. Recolección y etiquetado simple
# ==============================
logger.info("Iniciando recolección de datos desde Reddit...")

burnout_df = get_posts(query_burnout)
control_df = get_posts(query_control)

burnout_df["label"] = "Burnout"
control_df["label"] = "Control"

final_df = pd.concat([burnout_df, control_df], ignore_index=True)
final_df["date"] = pd.to_datetime(final_df["created_utc"])

logger.info("Datos recopilados: %d publicaciones.", len(final_df))
logger.info("Resumen → Burnout: %d | Control: %d", len(burnout_df), len(control_df))

print(final_df.head())

# =====================================================
# 7. Preprocesamiento de texto
# =====================================================
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
        logger.error("Descargar modelo con: python -m spacy download en_core_web_sm")
        raise

def preprocess_text(text: Union[str, None], nlp, stop_words_set: set) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|\#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)

    lemmas = [
        token.lemma_
        for token in doc
        if token.is_alpha and token.lemma_ not in stop_words_set
    ]

    return " ".join(lemmas)

# =====================================================
# 8. Pipeline principal
# =====================================================
def main_processing(final_df: pd.DataFrame, spacy_model_name: str = "en_core_web_sm") -> pd.DataFrame:
    ensure_nltk_stopwords("english")
    stop_words_set = set(stopwords.words("english"))
    nlp = load_spacy_model(spacy_model_name)

    df = final_df.copy()

    logger.info("Realizando limpieza de texto...")
    df["clean_text"] = df["text"].apply(lambda t: preprocess_text(t, nlp, stop_words_set))

    logger.info("Procesamiento completado: %d registros.", len(df))
    return df

# =====================================================
# 9. Ejecución
# =====================================================
if __name__ == "__main__":
    labeled_df = main_processing(final_df)
    print("\n✅ Fase 1 completada correctamente.")
    print(labeled_df.head())
