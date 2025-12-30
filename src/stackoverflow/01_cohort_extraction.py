import pandas as pd
from lxml import etree
from datetime import datetime, timedelta
import os
import json

print("Iniciando Pre-Procesamiento Local (Fase 1 - CORREGIDA)...")

# --- 1. Definir rutas a tus archivos descomprimidos ---
# Asegúrate de que los archivos XML estén en la misma carpeta que este script
# o ajusta base_dir según corresponda.
base_dir = "" 
users_xml_path = os.path.join(base_dir, "Users.xml")
posts_xml_path = os.path.join(base_dir, "Posts.xml")
output_path = os.path.join(base_dir, "filtered_posts.jsonl")

# --- 2. Definir cohortes y fechas (CRÍTICO: AJUSTE TEMPORAL) ---
# Diagnóstico previo indicó que el dataset termina el 2024-03-31.
# Usamos esa fecha como nuestro "Ahora" para el experimento.
DUMP_DATE = datetime(2024, 3, 31) 
one_year_ago = DUMP_DATE - timedelta(days=365)

print(f"Fecha de referencia del dataset (Presente simulado): {DUMP_DATE}")
print(f"Umbral de corte para Burnout (Inactividad desde): {one_year_ago}")

# --- 3. Procesar Users.xml PRIMERO ---
print(f"Procesando {users_xml_path} para identificar cohortes...")

users_data = []
# Usamos iterparse para no cargar todo el XML en RAM
context = etree.iterparse(users_xml_path, events=('end',), tag='row')

for event, elem in context:
    try:
        rep = int(elem.get('Reputation', 0))
        if rep > 1000: # Filtro de reputación para usuarios comprometidos
            
            # Parseo seguro de fecha
            last_access_str = elem.get('LastAccessDate')
            if last_access_str:
                last_access = pd.to_datetime(last_access_str)
                
                users_data.append({
                    'Id': elem.get('Id'),
                    'LastAccessDate': last_access,
                    'Reputation': rep
                })
    except Exception:
        continue 
    
    # Limpieza de memoria del elemento XML procesado
    elem.clear() 
    while elem.getprevious() is not None:
        del elem.getparent()[0]

users_df = pd.DataFrame(users_data)
print(f"Total de usuarios procesados con >1000 rep: {len(users_df)}")

# Etiquetado basado en la fecha corregida
users_df['label'] = users_df.apply(
    lambda row: 'Burnout' if row['LastAccessDate'] < one_year_ago else 'Control',
    axis=1
)

# Estadísticas rápidas de las cohortes
print("Distribución de Cohortes (Preliminar):")
print(users_df['label'].value_counts())

# Crear mapa hash para búsqueda rápida O(1)
cohort_users_map = users_df.set_index('Id')['label'].to_dict()
print("Mapa de usuarios cargado en memoria.")

# --- 4. Procesar Posts.xml (Streaming de ENTRADA y SALIDA) ---
print(f"Iniciando streaming de {posts_xml_path}...")
print(f"Escribiendo resultados en {output_path}...")

count = 0
found_count = 0

# Abrimos el archivo de salida
with open(output_path, 'w', encoding='utf-8') as f_out:
    context = etree.iterparse(posts_xml_path, events=('end',), tag='row')

    for event, elem in context:
        count += 1
        if count % 1000000 == 0: 
            print(f"  ... {count:,} posts procesados | {found_count:,} posts guardados")

        try:
            user_id = elem.get('OwnerUserId')
            
            # Solo guardamos si el usuario pertenece a una de nuestras cohortes
            if user_id in cohort_users_map:
                post_label = cohort_users_map[user_id]
                
                # Preparamos el registro JSON
                record = {
                    'user': user_id,
                    'label': post_label,
                    'text': elem.get('Title', '') + " " + elem.get('Body', ''),
                    'date': elem.get('CreationDate')
                }
                
                f_out.write(json.dumps(record) + '\n')
                found_count += 1
                
        except Exception:
            continue 
        
        # Limpieza vital de memoria
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

print(f"Streaming completado.")
print(f"Total de posts procesados: {count:,}")
print(f"Total de posts relevantes guardados: {found_count:,}")
print(f"DATOS FILTRADOS GUARDADOS EN: {output_path}")
print("Fase 1 Completada Exitosamente.")