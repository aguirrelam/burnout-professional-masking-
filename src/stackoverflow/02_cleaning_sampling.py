import json
import pandas as pd
import re
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
from tqdm import tqdm
import os

# --- 0. SILENCIAR ADVERTENCIAS MOLESTAS ---
# Esto elimina el mensaje "XMLParsedAsHTMLWarning" de la consola
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# --- Configuración ---
input_path = "filtered_posts.jsonl"
output_path_parquet = "phase2_clean_dataset.parquet"
output_path_csv = "phase2_clean_dataset.csv"

# Ajusta esto según tu capacidad de RAM. 50k por clase = 100k filas (manejable).
SAMPLE_SIZE_PER_CLASS = 50000 
MIN_TEXT_LENGTH = 50 

print("Iniciando Fase 2 (FINAL): Limpieza Profunda y Muestreo Estratificado...")

def clean_html_and_code(text):
    """
    Limpia el texto eliminando HTML y bloques de código.
    """
    if not text: return ""
    
    # Usamos 'lxml' porque es rápido, el warning ya está silenciado arriba.
    soup = BeautifulSoup(text, "lxml")
    
    # 1. Eliminar bloques de código (<pre> y <code>)
    # Queremos analizar la psicología del usuario, no su código C++/Python.
    for code in soup.find_all(['pre', 'code']):
        code.decompose()
        
    # 2. Obtener texto limpio
    text_clean = soup.get_text(separator=' ')
    
    # 3. Normalizar espacios (eliminar saltos de línea múltiples)
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()
    return text_clean

# Estructuras para el muestreo equilibrado
burnout_data = []
control_data = []

print(f"Leyendo {input_path} y buscando {SAMPLE_SIZE_PER_CLASS} muestras por clase...")

try:
    with open(input_path, 'r', encoding='utf-8') as f:
        # Usamos tqdm para barra de progreso. 
        # Si la Fase 1 regeneró el archivo, el total puede variar, pero 33M es la referencia.
        for line in tqdm(f, total=33013075, desc="Procesando y Limpiando"):
            
            # Condición de parada: Si ya tenemos 50k de CADA UNO, paramos para ahorrar tiempo.
            if len(burnout_data) >= SAMPLE_SIZE_PER_CLASS and len(control_data) >= SAMPLE_SIZE_PER_CLASS:
                print("\n¡Objetivo de muestreo alcanzado antes de terminar el archivo!")
                break

            try:
                record = json.loads(line)
                original_label = record['label']
                
                # OPTIMIZACIÓN: Si ya llenamos el cupo de esta etiqueta, saltamos la limpieza (es costosa)
                if original_label == 'Burnout' and len(burnout_data) >= SAMPLE_SIZE_PER_CLASS:
                    continue
                if original_label == 'Control' and len(control_data) >= SAMPLE_SIZE_PER_CLASS:
                    continue

                # --- Lógica de Limpieza ---
                clean_text = clean_html_and_code(record['text'])
                
                # Filtro de calidad: Textos muy cortos no sirven para NLP psicológico
                if len(clean_text) < MIN_TEXT_LENGTH:
                    continue
                
                processed_record = {
                    'user_id': record['user'],
                    'label': original_label,
                    'date': record['date'],
                    'clean_text': clean_text,
                    'char_count': len(clean_text) # Útil para estadísticas posteriores
                }

                # --- Llenado de Listas ---
                if original_label == 'Burnout':
                    burnout_data.append(processed_record)
                elif original_label == 'Control':
                    control_data.append(processed_record)
                    
            except Exception:
                continue

except FileNotFoundError:
    print(f"Error CRÍTICO: No se encuentra {input_path}")
    print("Asegúrate de haber ejecutado la Fase 1 primero.")
    exit()

print(f"\nRecolección finalizada.")
print(f"Burnout recolectados: {len(burnout_data)}")
print(f"Control recolectados: {len(control_data)}")

# --- DIAGNÓSTICO DE CALIDAD DE DATOS ---
if len(control_data) == 0:
    print("\n[ALERTA ROJA] -----------------------------------------")
    print("Seguimos sin encontrar usuarios 'Control'.")
    print("Verifica que en la Fase 1 hayas cambiado la fecha 'now' por '2024-03-31'.")
    print("-------------------------------------------------------\n")
else:
    print("\n[OK] Balance de clases exitoso. Dataset listo para análisis.")

# Combinar datos y crear DataFrame
all_data = burnout_data + control_data
df = pd.DataFrame(all_data)

# --- GUARDADO INTELIGENTE (Parquet preferido, CSV respaldo) ---
if not df.empty:
    # Mezclar aleatoriamente las filas para no tener todo Burnout al principio y Control al final
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    try:
        print(f"Intentando guardar en Parquet (Optimizado): {output_path_parquet}...")
        df.to_parquet(output_path_parquet, index=False)
        print("¡Éxito! Archivo Parquet generado correctamente.")
    except ImportError:
        print("Aviso: Librería 'pyarrow' no detectada.")
        print(f"Guardando como CSV (Estándar): {output_path_csv}...")
        df.to_csv(output_path_csv, index=False, encoding='utf-8')
        print("¡Éxito! Archivo CSV generado.")
    except Exception as e:
        print(f"Error inesperado al guardar: {e}")
else:
    print("No hay datos suficientes para guardar.")

print("Fase 2 Finalizada.")