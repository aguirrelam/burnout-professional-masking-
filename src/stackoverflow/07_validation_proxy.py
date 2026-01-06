import pandas as pd
from lxml import etree
from datetime import datetime, timedelta
import os
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

print("--- INICIANDO VALIDACIÓN DE HIPÓTESIS DE BURNOUT (PROXY) ---")

# 1. CONFIGURACIÓN (Misma que tu script 01)
base_dir = "" 
users_xml_path = os.path.join(base_dir, "Users.xml")
DUMP_DATE = datetime(2024, 3, 31) 
one_year_ago = DUMP_DATE - timedelta(days=365)

print(f"Fecha de Corte: {DUMP_DATE}")
print(f"Umbral de Inactividad: {one_year_ago}")

# 2. EXTRACCIÓN DE DATOS DE USUARIOS
# Necesitamos comparar tus usuarios 'Burnout' (>1000 rep) contra 'Casual Churn' (<50 rep)
print(f"Leyendo {users_xml_path} para extraer métricas de antigüedad...")

users_data = []
context = etree.iterparse(users_xml_path, events=('end',), tag='row')

for event, elem in context:
    try:
        rep = int(elem.get('Reputation', 0))
        
        # Parsear fechas
        last_access_str = elem.get('LastAccessDate')
        creation_date_str = elem.get('CreationDate')
        
        if last_access_str and creation_date_str:
            last_access = pd.to_datetime(last_access_str)
            creation_date = pd.to_datetime(creation_date_str)
            
            # CATEGORIZACIÓN (La clave de la validación)
            group = None
            
            # Tu Cohorte de Estudio (Burnout)
            if rep > 1000 and last_access < one_year_ago:
                group = 'Burnout (Target)'
            
            # Grupo de Control (Activos)
            elif rep > 1000 and last_access >= one_year_ago:
                group = 'Active (Control)'
                
            # Grupo de Comparación (Abandonos Casuales/Novatos)
            # Usuarios que se fueron pero tenían poca reputación
            elif rep < 50 and last_access < one_year_ago:
                group = 'Casual Churn'
            
            if group:
                # Calcular Tenencia (Días entre creación y último acceso)
                tenure_days = (last_access - creation_date).days
                
                # Guardamos solo lo necesario para no saturar RAM
                # (Muestreamos el grupo casual porque son millones)
                if group == 'Casual Churn':
                    if hash(elem.get('Id')) % 100 == 0: # Tomar 1% de casuales
                        users_data.append({'group': group, 'tenure_days': tenure_days, 'rep': rep})
                else:
                    users_data.append({'group': group, 'tenure_days': tenure_days, 'rep': rep})

    except Exception:
        continue
    
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]

df = pd.DataFrame(users_data)
print(f"Datos extraídos: {len(df)} usuarios.")
print(df['group'].value_counts())

# 3. ANÁLISIS ESTADÍSTICO (La prueba para el Paper)
print("\n--- RESULTADOS DE LA VALIDACIÓN ---")

target_group = df[df['group'] == 'Burnout (Target)']['tenure_days']
casual_group = df[df['group'] == 'Casual Churn']['tenure_days']

mean_target = target_group.mean()
mean_casual = casual_group.mean()

print(f"Promedio de días activos antes de renunciar (Target): {mean_target:.1f} días ({mean_target/365:.1f} años)")
print(f"Promedio de días activos antes de renunciar (Casual): {mean_casual:.1f} días ({mean_casual/365:.1f} años)")

# T-Test independiente
t_stat, p_val = stats.ttest_ind(target_group, casual_group, equal_var=False)

print(f"\nPrueba T de Student:")
print(f"Statistic: {t_stat:.4f}")
print(f"P-Value: {p_val:.2e}")  # Notación científica

if p_val < 0.05:
    print("\n[CONCLUSIÓN EXITOSA]: La diferencia es estadísticamente significativa.")
    print("Se demuestra que el grupo 'Burnout' invirtió significativamente más tiempo")
    print("en la plataforma antes de irse que los usuarios casuales.")
    print("Esto valida el uso de la reputación como filtro para 'Fatiga de Voluntariado'.")
else:
    print("\n[ADVERTENCIA]: No se encontró diferencia significativa.")

# 4. GRAFICAR PARA EL PAPER (Opcional pero recomendado)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='tenure_days', hue='group', fill=True, common_norm=False, palette='viridis')
plt.title('Distribution of Account Tenure Before Abandonment')
plt.xlabel('Days Active on Platform')
plt.xlim(0, 4000) # Limitar a 10 años aprox para ver mejor
plt.savefig("Validation_Proxy_Tenure.png")
print("\nGráfico guardado como 'Validation_Proxy_Tenure.png'")