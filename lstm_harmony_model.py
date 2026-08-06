"""
lstm_harmony_model.py
=====================
Define a arquitetura do modelo LSTM com TensorFlow/Keras.

ARQUITETURA:
  ┌──────────────────────────────────────────────────────┐
  │  Entrada: sequência de 4 acordes (índices inteiros)  │
  │  [PAD, PAD, C, Am]  →  shape: (batch, 4)            │
  └─────────────────┬────────────────────────────────────┘
                    │
          ┌─────────▼─────────┐
          │  Embedding Layer  │   chord_idx → vetor 32-dim
          │  (vocab_size, 32) │   aprende representação musical
          └─────────┬─────────┘
                    │  shape: (batch, 4, 32)
          ┌─────────▼─────────┐
          │  LSTM  128 units  │   processa a sequência temporal
          │  return_seq=True  │   mantém estado para próxima camada
          │  Dropout 0.3      │   regularização (evita overfitting)
          └─────────┬─────────┘
                    │  shape: (batch, 4, 128)
          ┌─────────▼─────────┐
          │  LSTM  64 units   │   extrai padrões de nível mais alto
          │  return_seq=False │   retorna apenas o estado final
          │  Dropout 0.3      │
          └─────────┬─────────┘
                    │  shape: (batch, 64)
          ┌─────────▼─────────┐
          │  Dense  64 units  │   camada de fusão
          │  ReLU activation  │
          │  Dropout 0.2      │
          └─────────┬─────────┘
                    │  shape: (batch, 64)
          ┌─────────▼─────────┐
          │  Dense  vocab_sz  │   uma saída por acorde possível
          │  Softmax          │   converte em probabilidades (0-1)
          └─────────┬─────────┘
                    │  shape: (batch, vocab_size)
                    ▼
          Distribuição de probabilidade
          sobre todos os acordes do vocabulário.
          O acorde mais provável é o "próximo acorde".

POR QUE LSTM?
  - Recurrent Neural Network: processa sequências em ordem
  - Mantém "memória" dos acordes anteriores via estado oculto
  - Captura padrões de longo prazo (ex: I→IV→V→I ao longo do compasso)
  - Mais adequado que Markov, que só olha 2 acordes atrás

POR QUE 2 CAMADAS LSTM?
  - 1ª camada: aprende padrões locais (progressões de 2-3 notas)
  - 2ª camada: aprende padrões de nível mais alto (frases harmônicas)

POR QUE EMBEDDING?
  - Diferente de one-hot encoding, aprende representações densas
  - Acordes musicalmente similares ficam próximos no espaço vetorial
  - Ex: C e G (dominante/tônica) devem ter vetores similares após treino
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#  1. Construção do modelo

def build_lstm_model(
    vocab_size: int,
    window_size: int = 4,
    embedding_dim: int = 32,
    lstm_units_1: int = 128,
    lstm_units_2: int = 64,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Constrói e compila o modelo LSTM para predição de acordes.

    Parâmetros
    ----------
    vocab_size : int
        Número de acordes únicos no vocabulário (incluindo PAD).
    window_size : int
        Número de acordes de contexto na entrada (padrão: 4).
    embedding_dim : int
        Dimensão do espaço de embedding. 32 é suficiente para ~30 acordes.
        Valores maiores capturam mais nuances mas precisam de mais dados.
    lstm_units_1 : int
        Número de unidades da 1ª camada LSTM.
        128 é um bom equilíbrio para dataset pequeno.
    lstm_units_2 : int
        Número de unidades da 2ª camada LSTM.
        Metade da 1ª para criar um "funil" de abstração.
    dense_units : int
        Unidades da camada densa intermediária.
    dropout_rate : float
        Taxa de dropout (0.0 = sem dropout, 1.0 = tudo zerado).
        0.3 é conservador para datasets pequenos.
    learning_rate : float
        Taxa de aprendizado do otimizador Adam.

    Retorna
    -------
    keras.Model compilado, pronto para .fit()
    """
    # Entrada
    # Sequência de window_size índices inteiros de acordes
    inputs = keras.Input(shape=(window_size,), name="chord_sequence")

    # Embedding
    # Transforma índices inteiros em vetores densos de embedding_dim dimensões.
    # mask_zero=True: ignora tokens PAD (índice 0) nos cálculos da LSTM.
    # Isso é importante para não "confundir" o modelo com o padding artificial.
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,         # PAD (idx=0) é ignorado pelo LSTM
        name="chord_embedding",
    )(inputs)
    # shape: (batch_size, window_size, embedding_dim)

    # LSTM Camada 1
    # return_sequences=True: retorna o estado oculto em CADA passo de tempo.
    # Isso é necessário para alimentar a 2ª camada LSTM.
    x = layers.LSTM(
        units=lstm_units_1,
        return_sequences=True,  # mantém saída em cada timestep
        name="lstm_layer_1",
    )(x)
    # shape: (batch_size, window_size, lstm_units_1)

    # Dropout entre as camadas LSTM para regularização
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    # LSTM Camada 2
    # return_sequences=False: retorna apenas o estado final.
    # Aqui capturamos o "resumo" de toda a sequência.
    x = layers.LSTM(
        units=lstm_units_2,
        return_sequences=False, # retorna apenas o último estado
        name="lstm_layer_2",
    )(x)
    # shape: (batch_size, lstm_units_2)

    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    # Camada Densa Intermediária
    # Transforma a representação LSTM em features de alto nível.
    # ReLU é a ativação padrão para camadas intermediárias.
    x = layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)
    # shape: (batch_size, dense_units)

    x = layers.Dropout(dropout_rate * 0.67, name="dropout_3")(x)  # dropout menor no final

    # Camada de Saída
    # Uma unidade por acorde no vocabulário.
    # Softmax garante que as saídas são probabilidades que somam 1.0.
    # Acorde com maior probabilidade = próxima previsão do modelo.
    outputs = layers.Dense(
        vocab_size,
        activation="softmax",
        name="chord_probabilities",
    )(x)
    # shape: (batch_size, vocab_size)

    # Montar modelo
    model = keras.Model(inputs=inputs, outputs=outputs, name="RequiemLSTM")

    # Compilação
    # Loss: sparse_categorical_crossentropy
    #   → usada quando os rótulos (y) são índices inteiros (não one-hot)
    #   → mede o quão errada está a distribuição de probabilidade prevista
    #
    # Optimizer: Adam
    #   → adaptativo, lida bem com gradientes esparsos (comuns em embeddings)
    #   → learning_rate=1e-3 é o padrão recomendado pela literatura
    #
    # Metrics: accuracy
    #   → "acertou o acorde exato?" — fácil de interpretar
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


#  2. Salvar e Carregar

def save_model(model: keras.Model, path: str | Path) -> None:
    """
    Salva o modelo treinado no formato nativo Keras (.keras).
    Inclui arquitetura + pesos + configuração de compilação.
    O Keras 3 exige a extensão .keras para o formato nativo.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    keras_path = str(path / "model.keras")
    model.save(keras_path)
    print(f"[Model] Modelo salvo em: {keras_path}")


def load_model(path: str | Path) -> keras.Model:
    """
    Carrega um modelo previamente salvo.
    Retorna modelo pronto para inferência (.predict()).
    """
    path = Path(path)
    # Suporta tanto o diretório (novo) quanto o arquivo .keras diretamente
    keras_path = path / "model.keras" if path.is_dir() else path
    model = keras.models.load_model(str(keras_path))
    print(f"[Model] Modelo carregado de: {keras_path}")
    return model


#  3. Sumário e inspeção

def print_model_summary(model: keras.Model) -> None:
    """Imprime o resumo detalhado da arquitetura do modelo."""
    model.summary(expand_nested=True)

    total_params = model.count_params()
    print(f"\n  Total de parâmetros: {total_params:,}")
    print(f"  Parâmetros treináveis: {sum(w.numpy().size for w in model.trainable_weights):,}")
