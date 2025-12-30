"""
===========================================================
Fase 2: Extracción de Características y Creación de Series Temporales
===========================================================
"""

# ==============================
# 0. Importar Fase 1 automáticamente
# ==============================
import logging
import importlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

logger.info("Importando y ejecutando Fase 1...")

fase_1 = importlib.import_module("fase_1")

try:
    labeled_df = fase_1.main_processing(fase_1.final_df)
except Exception as e:
    logger.error("Error al ejecutar Fase 1: %s", e)
    raise SystemExit("Fase 1 falló. Verifica fase_1.py.")

logger.info("Fase 1 completada. Continuando con Fase 2...")

# ==============================
# 1. Importación de librerías
# ==============================
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

# ==============================
# 2. Inicializar modelos
# ==============================
logger.info("Cargando BERT ('bert-base-uncased')...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model_bert = BertModel.from_pretrained("bert-base-uncased")
model_bert.eval()

logger.info("Inicializando VADER...")
sentiment_analyzer = SentimentIntensityAnalyzer()

# ==============================
# 3. BERT embeddings
# ==============================
def get_bert_embedding(text: str):
    if not isinstance(text, str) or text.strip() == "":
        return torch.zeros(768).numpy()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model_bert(**inputs)

    cls_emb = outputs.last_hidden_state[:, 0, :].squeeze()
    return cls_emb.cpu().numpy()

# ==============================
# 4. Extracción de características
# ==============================
def extract_all_features(row):
    text = row.get("clean_text", "")
    features = {}

    # Sentimiento
    sentiment = sentiment_analyzer.polarity_scores(text)
    features.update({
        "sentiment_pos": sentiment["pos"],
        "sentiment_neg": sentiment["neg"],
        "sentiment_neu": sentiment["neu"],
        "sentiment_compound": sentiment["compound"]
    })

    # Complejidad
    try:
        features["lexical_diversity"] = (
            textstat.lexicon_count(text) /
            max(1, textstat.sentence_count(text))
        )
    except:
        features["lexical_diversity"] = 0.0

    # Embeddings BERT
    bert_vec = get_bert_embedding(text)
    for i, val in enumerate(bert_vec):
        features[f"bert_{i}"] = val

    return pd.Series(features)

# ==============================
# 5. Pipeline de características
# ==============================
def main_feature_extraction(labeled_df):
    logger.info("Extrayendo características de texto...")

    features_df = labeled_df.apply(extract_all_features, axis=1)
    full_features = pd.concat([labeled_df.reset_index(drop=True),
                               features_df], axis=1)

    logger.info("Extracción completada (%d columnas).", full_features.shape[1])
    return full_features

# ==============================
# 6. Series temporales
# ==============================
def create_time_series(full_features_df):
    logger.info("Creando series temporales...")

    # Índice temporal
    full_features_df = full_features_df.set_index("date")

    feature_cols = [
        col for col in full_features_df.columns
        if any(k in col for k in ["sentiment", "lexical", "bert"])
    ]

    user_sequences = {}

    for author, group in full_features_df.groupby("author"):   # ← CORREGIDO AQUÍ
        seq = group[feature_cols].resample("W").agg(["mean", "std", "max"]).fillna(0)

        final_label = group["label"].iloc[-1]

        user_sequences[author] = {
            "sequence": seq.values,
            "label": final_label
        }

    logger.info("Series temporales generadas para %d usuarios.", len(user_sequences))
    return user_sequences

# ==============================
# 7. Ejecución
# ==============================
if __name__ == "__main__":
    full_features_df = main_feature_extraction(labeled_df)
    user_sequences = create_time_series(full_features_df)

    print("\n✅ Fase 2 completada con éxito.")
    print("Usuarios procesados:", len(user_sequences))
    print("Ejemplos:", list(user_sequences.keys())[:5])






