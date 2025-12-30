import json
from datetime import datetime
import pandas as pd

input_path = "filtered_posts.jsonl"
max_date = datetime(2000, 1, 1)
min_date = datetime(2100, 1, 1)

print(f"Escaneando {input_path} para encontrar la fecha de corte real...")

try:
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0: print(f"Leídas {i} líneas...")
            
            try:
                record = json.loads(line)
                # Parsear fecha. El formato usual de SO es '2020-05-12T14:30:00.000'
                dt_str = record['date']
                # Cortamos milisegundos si molestan o usamos pd.to_datetime que es flexible
                dt = pd.to_datetime(dt_str)
                
                if dt > max_date: max_date = dt
                if dt < min_date: min_date = dt
            except:
                continue

    print("-" * 30)
    print(f"RESULTADO DEL DIAGNÓSTICO:")
    print(f"Primera fecha en el dataset: {min_date}")
    print(f"ÚLTIMA FECHA EN EL DATASET: {max_date}")
    print("-" * 30)
    print("Usa esta 'ÚLTIMA FECHA' como tu punto de referencia 'now' en la Fase 1.")

except FileNotFoundError:
    print("No encuentro el archivo filtered_posts.jsonl")