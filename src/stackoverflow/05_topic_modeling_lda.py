import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns

print("Iniciando Fase 5: Modelado de Tópicos (LDA) y Riesgo por Tema...")

# 1. Cargar Dataset Limpio (Fase 2)
# Usamos el de la fase 2 porque necesitamos el texto, las features de la fase 3 ya no las necesitamos
input_file = "phase2_clean_dataset.parquet"
try:
    df = pd.read_parquet(input_file)
    print(f"Dataset cargado: {len(df)} registros.")
except Exception:
    # Fallback a CSV si parquet no existe
    try:
        df = pd.read_csv("phase2_clean_dataset.csv")
    except:
        print("Error: No se encuentra el dataset de la Fase 2.")
        exit()

# Tomamos una muestra para que LDA no tarde horas (LDA es pesado)
# 20,000 registros es suficiente para encontrar temas generales
df_sample = df.sample(n=20000, random_state=42).copy()
print("Muestra de 20,000 registros seleccionada para modelado de temas.")

# 2. Vectorización (Bag of Words)
# LDA necesita conteos simples, no TF-IDF
print("Vectorizando para LDA...")
tf_vectorizer = CountVectorizer(
    max_features=2000,      # Vocabulario limitado a 2k palabras más frecuentes
    stop_words='english',
    ngram_range=(1, 1)      # Solo palabras simples
)
X_tf = tf_vectorizer.fit_transform(df_sample['clean_text'].fillna(''))

# 3. Entrenamiento LDA (Encontrar 5 Temas Principales)
NUM_TOPICS = 5
print(f"Buscando {NUM_TOPICS} temas latentes en las discusiones...")
lda = LatentDirichletAllocation(
    n_components=NUM_TOPICS, 
    max_iter=10, 
    learning_method='online', 
    random_state=42,
    n_jobs=-1
)
lda.fit(X_tf)

# 4. Visualización de Temas (¿De qué hablan?)
print("\n--- TEMAS IDENTIFICADOS ---")
feature_names = tf_vectorizer.get_feature_names_out()

topic_dict = {}
for topic_idx, topic in enumerate(lda.components_):
    top_features_ind = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_features_ind]
    topic_name = f"Tema {topic_idx+1}"
    topic_dict[topic_idx] = top_words
    print(f"{topic_name}: {', '.join(top_words)}")

# 5. Asignar Tema a cada Post y Calcular Riesgo
# Transformamos la muestra para ver qué tema predomina en cada post
topic_values = lda.transform(X_tf)
df_sample['dominante_topic'] = topic_values.argmax(axis=1)

# Cruzamos con la etiqueta de Burnout/Control
print("\n--- ANÁLISIS DE RIESGO POR TEMA ---")
# Agrupamos por tema y calculamos el % de Burnout
# (Recordemos: label 'Burnout' vs 'Control')

# Convertir label a binario para calcular promedio (Burnout=1)
df_sample['is_burnout'] = df_sample['label'].apply(lambda x: 1 if x == 'Burnout' else 0)

risk_analysis = df_sample.groupby('dominante_topic')['is_burnout'].agg(['count', 'mean'])
risk_analysis['mean'] = risk_analysis['mean'] * 100 # Convertir a porcentaje
risk_analysis.columns = ['Volumen de Posts', '% Tasa de Burnout']

# Mostramos resultados con las palabras clave para contexto
for topic_idx in risk_analysis.index:
    words = ", ".join(topic_dict[topic_idx][:5])
    print(f"\nTema {topic_idx+1} ({words}):")
    print(f"   -> Volumen: {risk_analysis.loc[topic_idx, 'Volumen de Posts']} posts")
    print(f"   -> Tasa de Abandono (Burnout): {risk_analysis.loc[topic_idx, '% Tasa de Burnout']:.2f}%")

# 6. Conclusión Automática
max_risk = risk_analysis['% Tasa de Burnout'].max()
min_risk = risk_analysis['% Tasa de Burnout'].min()
print(f"\n[INSIGHT]: La diferencia entre el tema más 'quemante' y el más 'seguro' es de {max_risk - min_risk:.2f} puntos porcentuales.")

# Guardar resultados
risk_analysis.to_csv("phase5_topic_risk.csv")
print("Análisis de tópicos guardado en 'phase5_topic_risk.csv'")