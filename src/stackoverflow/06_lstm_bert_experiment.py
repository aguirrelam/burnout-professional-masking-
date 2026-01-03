import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# --- 1. CONFIGURACIÓN ---
INPUT_FILE = "phase2_clean_dataset.parquet"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16  # Reducido un poco para estabilidad en PyTorch Windows
EPOCHS = 3
LEARNING_RATE = 1e-4 # Tasa de aprendizaje estándar para fine-tuning

# Configuración de Dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

print("Iniciando Fase 6: Experimento Deep Learning (BERT + LSTM) con PyTorch...")

# --- 2. CARGAR DATOS ---
try:
    df = pd.read_parquet(INPUT_FILE)
    print(f"Dataset cargado: {len(df)} registros.")
except Exception:
    print("Error: No se encuentra 'phase2_clean_dataset.parquet'.")
    exit()

# Preparación de etiquetas
X_text = df['clean_text'].astype(str).tolist()
y = df['label'].map({'Control': 0, 'Burnout': 1}).values

# DIVISIÓN IDÉNTICA (Random State 42)
print("Dividiendo dataset (Train/Test)...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. TOKENIZACIÓN ---
print(f"Cargando Tokenizer ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_data(texts, tokenizer, max_len):
    encoded = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt' # Retorna tensores de PyTorch
    )
    return encoded['input_ids'], encoded['attention_mask']

print("Tokenizando datos...")
input_ids_train, attention_masks_train = tokenize_data(X_train_text, tokenizer, MAX_LEN)
input_ids_test, attention_masks_test = tokenize_data(X_test_text, tokenizer, MAX_LEN)

labels_train = torch.tensor(y_train, dtype=torch.float) # Float para BCEWithLogitsLoss
labels_test = torch.tensor(y_test, dtype=torch.float)

# Crear DataLoaders
train_data = TensorDataset(input_ids_train, attention_masks_train, labels_train)
test_data = TensorDataset(input_ids_test, attention_masks_test, labels_test)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. ARQUITECTURA DEL MODELO (BERT + LSTM) ---
class BertLstmClassifier(nn.Module):
    def __init__(self, transformer_name):
        super(BertLstmClassifier, self).__init__()
        # Capa BERT (Congelada)
        self.bert = AutoModel.from_pretrained(transformer_name)
        for param in self.bert.parameters():
            param.requires_grad = False # Congelar pesos de BERT
            
        # Capa LSTM
        # input_size=768 (dimensión standard de BERT/DistilBERT)
        self.lstm = nn.LSTM(input_size=768, hidden_size=64, batch_first=True)
        
        # Clasificador
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1) # Salida binaria
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # Pasar por BERT
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Obtener last_hidden_state (Batch, Sequence, Features) -> (32, 128, 768)
        sequence_output = outputs.last_hidden_state
        
        # Pasar por LSTM
        # lstm_out: (Batch, Sequence, Hidden) -> (32, 128, 64)
        lstm_out, _ = self.lstm(sequence_output)
        
        # Global Max Pooling (equivalente a lo que hicimos en Keras)
        # Tomamos el máximo valor sobre la dimensión de la secuencia
        out, _ = torch.max(lstm_out, dim=1) 
        
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)

print("Construyendo modelo PyTorch...")
model = BertLstmClassifier(MODEL_NAME)
model.to(device)

# Optimizador y Función de Pérdida
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

# --- 5. BUCLE DE ENTRENAMIENTO ---
print("\n--- INICIANDO ENTRENAMIENTO ---")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    # Barra de progreso
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    
    for batch in loop:
        # Mover datos a GPU/CPU
        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device).unsqueeze(1) # [Batch] -> [Batch, 1]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(b_input_ids, b_masks)
        loss = criterion(outputs, b_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

# --- 6. EVALUACIÓN ---
print("\n--- EVALUACIÓN FINAL ---")
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluando"):
        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device).unsqueeze(1)
        
        outputs = model(b_input_ids, b_masks)
        
        # Mover a CPU para métricas de sklearn
        predictions.extend(outputs.cpu().numpy())
        true_labels.extend(b_labels.cpu().numpy())

# Convertir a arrays planos
predictions = np.array(predictions).flatten()
true_labels = np.array(true_labels).flatten()
pred_binary = (predictions > 0.5).astype(int)

# Reporte
print(classification_report(true_labels, pred_binary, target_names=['Control', 'Burnout']))
auc = roc_auc_score(true_labels, predictions)
print(f"AUC-ROC Score: {auc:.4f}")

# Guardar Modelo
torch.save(model.state_dict(), "phase6_lstm_bert_model.pth")
print("Modelo guardado en 'phase6_lstm_bert_model.pth'")