"""
export_weights_json.py
======================
Exporta os pesos do modelo LSTM como JSON puro para uso no frontend.

O TF.js tem problemas de compatibilidade com o Keras 3 (.keras format).
Esta abordagem exporta os pesos como arrays JSON que podem ser carregados
diretamente pelo TypeScript sem depender do TF.js loadLayersModel.

A inferência LSTM é então implementada em TS puro (forward pass manual).

SAÍDA:
  public/model/weights.json  — pesos do modelo em JSON
  public/model/vocab.json    — vocabulário (já existia)
"""

from __future__ import annotations
import os
import json
import struct
import base64
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras


def main():
    keras_path = Path("saved_model/model.keras")
    output_dir = Path("../requiem-app-source/requiem-app/public/model")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  REQUIEM — Exportação de Pesos para JSON")
    print("=" * 55)

    print(f"\nCarregando modelo de {keras_path}...")
    model = keras.models.load_model(str(keras_path))
    print(f"Parâmetros: {model.count_params():,}")

    # Coleta pesos por camada
    weights_data = {}

    for layer in model.layers:
        layer_weights = layer.get_weights()
        if not layer_weights:
            continue

        layer_name = layer.name
        weights_data[layer_name] = {
            "class": layer.__class__.__name__,
            "weights": [],
        }

        for i, w in enumerate(layer_weights):
            w_np = np.array(w, dtype=np.float32)
            weights_data[layer_name]["weights"].append({
                "shape": list(w_np.shape),
                "dtype": "float32",
                # Codifica em base64 para compacidade
                "data_b64": base64.b64encode(w_np.tobytes()).decode("ascii"),
            })

        print(f"  Layer '{layer_name}' ({layer.__class__.__name__}): "
              f"{sum(np.prod(w['shape']) for w in weights_data[layer_name]['weights']):,} params")

    # Metadados do modelo
    output = {
        "model_name": "RequiemLSTM",
        "vocab_size": 97,
        "window_size": 4,
        "embedding_dim": 32,
        "lstm_units": [128, 64],
        "dense_units": [64],
        "total_params": model.count_params(),
        "layers": weights_data,
    }

    # Salva JSON
    out_path = output_dir / "weights.json"
    out_path.write_text(json.dumps(output, separators=(",", ":")))

    size_kb = out_path.stat().st_size / 1024
    print(f"\n✓ weights.json salvo: {out_path}")
    print(f"  Tamanho: {size_kb:.1f} KB")

    # Copia vocab.json
    import shutil
    vocab_src = Path("saved_model/vocab.json")
    vocab_dst = output_dir / "vocab.json"
    if vocab_src.exists():
        shutil.copy2(vocab_src, vocab_dst)
        print(f"  vocab.json copiado para: {vocab_dst}")

    print("\nArquivos em public/model/:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name:40s} {f.stat().st_size/1024:8.1f} KB")


if __name__ == "__main__":
    main()
