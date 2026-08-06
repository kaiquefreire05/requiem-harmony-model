"""
train_model.py
==============
Script de treinamento do modelo LSTM de harmonia Requiem.

COMO EXECUTAR:
  python3 train_model.py

SAÍDAS GERADAS:
  saved_model/      — Modelo treinado (TensorFlow SavedModel)
  saved_model/vocab.json  — Vocabulário de acordes
  training_history.json   — Curvas de loss/accuracy por época

O QUE ACONTECE DURANTE O TREINO:
  1. Carrega os 51 JSONs de music_data/
  2. Aplica data augmentation (×12 tonalidades) → 612 progressões
  3. Cria amostras de treino por janela deslizante
  4. Divide em treino (80%) e validação (20%)
  5. Treina o modelo LSTM por até MAX_EPOCHS épocas
  6. Usa Early Stopping: para automaticamente quando a
     accuracy de validação não melhora por PATIENCE épocas
  7. Salva o melhor modelo (checkpoint)

MÉTRICAS ESPERADAS (com 51 músicas):
  - Accuracy de treino:      > 70%
  - Accuracy de validação:   > 55%
  - Se overfitting (train>>val): adicione mais JSONs ou reduza o modelo

GLOSSÁRIO:
  Época (epoch):   uma passagem completa por todos os dados de treino
  Batch:           subconjunto dos dados processado de uma vez
  Loss:            erro médio do modelo (menor = melhor)
  Accuracy:        % de acordes previstos corretamente
  Overfitting:     modelo decorou o treino mas não generaliza (val_loss sobe)
  Early Stopping:  para o treino quando a val_loss para de melhorar
  Checkpoint:      salva os pesos do melhor modelo durante o treino
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras

from chord_dataset import prepare_dataset, WINDOW_SIZE
from lstm_harmony_model import build_lstm_model, save_model, print_model_summary

#  Configurações de treinamento

# Caminhos de entrada e saída
DATA_DIR       = "music_data"
MODEL_DIR      = "saved_model"
HISTORY_FILE   = "training_history.json"

# Hiperparâmetros do treinamento
BATCH_SIZE     = 32     # Número de amostras processadas por passo de gradiente.
                        # 32 é padrão e funciona bem para datasets pequenos.

MAX_EPOCHS     = 300    # Número máximo de épocas. O Early Stopping geralmente
                        # para antes disso.

PATIENCE       = 25     # Quantas épocas sem melhora antes de parar.
                        # 25 é generoso para dar tempo ao modelo de escapar
                        # de mínimos locais.

VALIDATION_SPLIT = 0.2  # 20% dos dados reservados para validação.
                        # Validação mede a capacidade de generalização.

LEARNING_RATE  = 1e-3   # Taxa de aprendizado. 1e-3 (0.001) é o padrão do Adam.

# Semente aleatória para reprodutibilidade
SEED = 42

#  Função principal de treinamento

def train(
    data_dir: str = DATA_DIR,
    model_dir: str = MODEL_DIR,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    validation_split: float = VALIDATION_SPLIT,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
) -> dict:
    """
    Pipeline completo de treinamento.

    Retorna o histórico de métricas (loss e accuracy por época).
    """
    # Reprodutibilidade
    tf.random.set_seed(seed)
    np.random.seed(seed)

    print("\n" + "=" * 55)
    print("   REQUIEM — Treinamento do Modelo LSTM")
    print("=" * 55)

    # 1. Preparar Dataset
    print("\n[1/5] Preparando dataset...")
    data = prepare_dataset(data_dir=data_dir, window_size=WINDOW_SIZE, augment=True)

    # Salva o vocabulário junto com o modelo
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    data.vocab.save(Path(model_dir) / "vocab.json")
    print(f"      Vocabulário salvo em {model_dir}/vocab.json")

    # 2. Embaralhar os dados
    # Importante: embaralhar garante que cada batch tenha diversidade
    # e que a divisão treino/validação seja representativa.
    print("\n[2/5] Embaralhando dados...")
    indices = np.random.permutation(len(data.X))
    X_shuffled = data.X[indices]
    y_shuffled = data.y[indices]

    # 3. Construir Modelo
    print("\n[3/5] Construindo modelo LSTM...")
    vocab_size = len(data.vocab)
    model = build_lstm_model(
        vocab_size=vocab_size,
        window_size=WINDOW_SIZE,
        embedding_dim=32,
        lstm_units_1=128,
        lstm_units_2=64,
        dense_units=64,
        dropout_rate=0.3,
        learning_rate=learning_rate,
    )
    print_model_summary(model)

    # 4. Callbacks
    # Callbacks são funções chamadas automaticamente pelo Keras durante o treino.

    # EarlyStopping: monitora a loss de validação e para quando não melhora.
    # restore_best_weights=True: ao final, restaura os pesos da melhor época.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",       # monitora a loss de validação (não de treino)
        patience=patience,        # espera N épocas sem melhora antes de parar
        restore_best_weights=True,# usa os pesos da melhor época ao final
        verbose=1,
    )

    # ModelCheckpoint: salva o modelo a cada vez que val_loss melhora.
    # Garante que o melhor modelo seja preservado mesmo se o treino travasse.
    checkpoint_path = str(Path(model_dir) / "checkpoint.keras")
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,      # só salva se for melhor que antes
        save_weights_only=False,  # salva arquitetura + pesos
        verbose=0,
    )

    # ReduceLROnPlateau: reduz o learning rate quando a loss estagna.
    # Isso permite "afinar" o modelo quando ele já convergiu parcialmente.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,      # multiplica o LR por 0.5 (ex: 1e-3 → 5e-4)
        patience=10,     # espera 10 épocas antes de reduzir
        min_lr=1e-6,     # não reduz abaixo deste valor
        verbose=1,
    )

    # 5. Treinar
    print(f"\n[4/5] Treinando por até {max_epochs} épocas "
          f"(early stop com patience={patience})...")
    print(f"      Batch size: {batch_size}  |  Validação: {int(validation_split*100)}%")
    print("-" * 55)

    history = model.fit(
        X_shuffled,              # sequências de contexto (X)
        y_shuffled,              # próximo acorde (y)
        batch_size=batch_size,
        epochs=max_epochs,
        validation_split=validation_split,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=1,               # mostra progresso por época
    )

    # 6. Salvar Modelo e Histórico
    print(f"\n[5/5] Salvando modelo em {model_dir}/...")
    save_model(model, model_dir)

    # Salva histórico de métricas para análise posterior
    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    Path(HISTORY_FILE).write_text(json.dumps(history_data, indent=2))

    # Relatório Final
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))
    best_val_acc  = float(history.history["val_accuracy"][best_epoch - 1])
    total_epochs  = len(history.history["loss"])

    print("\n" + "=" * 55)
    print("   TREINAMENTO CONCLUÍDO")
    print("=" * 55)
    print(f"  Épocas treinadas        : {total_epochs}")
    print(f"  Melhor época            : {best_epoch}")
    print(f"  Melhor val_loss         : {best_val_loss:.4f}")
    print(f"  Melhor val_accuracy     : {best_val_acc:.1%}")
    print(f"  Modelo salvo em         : {model_dir}/")
    print(f"  Histórico salvo em      : {HISTORY_FILE}")
    print("=" * 55)

    if best_val_acc < 0.4:
        print("\n⚠  Accuracy abaixo de 40%. Dicas:")
        print("   - Adicione mais JSONs em music_data/")
        print("   - Aumente MAX_EPOCHS")
        print("   - Reduza dropout_rate no lstm_harmony_model.py")
    elif best_val_acc > 0.7:
        print("\n✓ Excelente! Accuracy > 70%.")

    return history_data


#  Ponto de entrada

if __name__ == "__main__":
    train()
