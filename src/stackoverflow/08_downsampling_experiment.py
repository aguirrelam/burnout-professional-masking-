import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import gc

# ==========================================
# CONFIGURACIÓN DEL EXPERIMENTO (Peer Review)
# ==========================================
INPUT_FILE = "phase2_clean_dataset.parquet"
MODEL_NAME = "distilbert-base-uncased"
REDDIT_SIZE_PROXY = 1200  # Tamaño del dataset de Reddit para igualar condiciones
NUM_ITERATIONS = 5        # Número de repeticiones (Bootstrapping)
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3                # Mismas épocas que el experimento principal
LEARNING_RATE = 1e-4

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- EXPERIMENTO DE ROBUSTEZ (DOWNSAMPLING) ---")
print(f"Dispositivo: {device}")
print(f"Objetivo: Validar si la baja precisión persiste con N={REDDIT_SIZE_PROXY}")

# ==========================================
# 1. DEFINICIÓN DEL MODELO (Idéntico a Fase 6)
# ==========================================
class BertLstmClassifier(nn.Module):
    def __init__(self, transformer_name):
        super(BertLstmClassifier, self).__init__()
        # Capa BERT (Congelada para velocidad y estabilidad en small data)
        self.bert = AutoModel.from_pretrained(transformer_name)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Capa LSTM
        self.lstm = nn.LSTM(input_size=768, hidden_size=64, batch_first=True)
        
        # Clasificador
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        
        # Global Max Pooling
        out, _ = torch.max(lstm_out, dim=1) 
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)

# ==========================================
# 2. FUNCIONES UTILITARIAS
# ==========================================
def tokenize_data(texts, tokenizer, max_len):
    encoded = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']

def train_iteration(iteration_idx, df_sample, tokenizer):
    print(f"\n[Iteración {iteration_idx+1}/{NUM_ITERATIONS}] Preparando datos...")
    
    # A. Split Train/Test (80/20)
    X_text = df_sample['clean_text'].astype(str).tolist()
    y = df_sample['label'].map({'Control': 0, 'Burnout': 1}).values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # B. Tokenización
    ids_train, masks_train = tokenize_data(X_train, tokenizer, MAX_LEN)
    ids_test, masks_test = tokenize_data(X_test, tokenizer, MAX_LEN)
    
    labels_train = torch.tensor(y_train, dtype=torch.float)
    labels_test = torch.tensor(y_test, dtype=torch.float)
    
    # C. DataLoaders
    train_data = TensorDataset(ids_train, masks_train, labels_train)
    test_data = TensorDataset(ids_test, masks_test, labels_test)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # D. Inicializar Modelo
    model = BertLstmClassifier(MODEL_NAME)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # E. Loop de Entrenamiento
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            b_ids, b_masks, b_lbls = [t.to(device) for t in batch]
            b_lbls = b_lbls.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(b_ids, b_masks)
            loss = criterion(outputs, b_lbls)
            loss.backward()
            optimizer.step()
            
    # F. Evaluación
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in test_loader:
            b_ids, b_masks, b_lbls = [t.to(device) for t in batch]
            outputs = model(b_ids, b_masks)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_true.extend(b_lbls.cpu().numpy().flatten())
            
    # Métricas
    pred_binary = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_true, pred_binary)
    f1 = f1_score(all_true, pred_binary)
    
    print(f"   -> Resultado Iteración {iteration_idx+1}: Accuracy={acc:.4f} | F1={f1:.4f}")
    
    # Limpieza de memoria GPU
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()
    
    return acc, f1

# ==========================================
# 3. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # A. Cargar Dataset Masivo
    if not os.path.exists(INPUT_FILE):
        print(f"Error: No se encuentra {INPUT_FILE}. Ejecuta Fase 2 primero.")
        exit()
        
    print(f"Cargando dataset completo ({INPUT_FILE})...")
    df_full = pd.read_parquet(INPUT_FILE)
    print(f"Total registros disponibles: {len(df_full)}")
    
    # B. Cargar Tokenizer (una sola vez)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    results_acc = []
    results_f1 = []
    
    # C. Bucle de Bootstrapping
    for i in range(NUM_ITERATIONS):
        # Muestreo aleatorio (Downsampling)
        # random_state cambia con 'i' para que cada muestra sea diferente
        df_sample = df_full.sample(n=REDDIT_SIZE_PROXY, random_state=i).copy()
        
        acc, f1 = train_iteration(i, df_sample, tokenizer)
        results_acc.append(acc)
        results_f1.append(f1)
        
    # D. Reporte Final para el Paper
    mean_acc = np.mean(results_acc)
    std_acc = np.std(results_acc)
    mean_f1 = np.mean(results_f1)
    
    print("\n" + "="*60)
    print("RESULTADOS DEL EXPERIMENTO DE ROBUSTEZ (para incluir en Paper)")
    print("="*60)
    print(f"Dataset: Stack Overflow (Downsampled a N={REDDIT_SIZE_PROXY})")
    print(f"Iteraciones: {NUM_ITERATIONS}")
    print(f"Modelo: BERT (Frozen) + LSTM")
    print("-" * 30)
    print(f"ACCURACY PROMEDIO: {mean_acc:.4f}  (+/- {std_acc:.4f})")
    print(f"F1-SCORE PROMEDIO: {mean_f1:.4f}")
    print("-" * 30)
    
    if mean_acc < 0.65:
        print(">> INTERPRETACIÓN: La precisión se mantiene baja incluso con muestras pequeñas.")
        print("   Esto VALIDA la hipótesis: El fallo no es por 'Big Data Noise',")
        print("   sino por 'Enmascaramiento Semántico' intrínseco.")
    else:
        print(">> INTERPRETACIÓN: La precisión mejoró. Revisar si el dataset grande tenía ruido.")