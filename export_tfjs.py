"""
export_tfjs.py
==============
Converte o modelo LSTM treinado (.keras) para o formato TF.js.

USO:
  python3 export_tfjs.py

SAÍDA:
  /public/model/model.json          — grafo do modelo
  /public/model/group1-shard1of1.bin — pesos em binário
  /public/model/vocab.json          — vocabulário de acordes

ESTRATÉGIA:
  Usa tensorflowjs.converters diretamente (sem importar o módulo raiz
  que depende de tensorflow_decision_forests).
"""

from __future__ import annotations
import sys
import json
import struct
import base64
import shutil
from pathlib import Path

# Tenta importar o converter do TF.js sem o __init__ quebrado
try:
    # Importação seletiva evita o crash do tensorflow_decision_forests
    import importlib
    import types

    # Cria módulo fake para evitar import do tensorflowjs.__init__
    # Estratégia: importar o submodulo diretamente via importlib
    spec = importlib.util.find_spec("tensorflowjs")
    if spec is None:
        raise ImportError("tensorflowjs não instalado")

    # Usa o convertor de keras direto
    from tensorflowjs.converters import keras_tfjs_loader
    HAS_TFJS = True
except Exception as e:
    print(f"[warn] tensorflowjs não disponível via import direto: {e}")
    HAS_TFJS = False


def convert_via_tfjs_converter(keras_path: str, output_dir: str) -> bool:
    """Tenta conversão via CLI do tensorflowjs."""
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "tensorflowjs.converters.converter",
            "--input_format=keras",
            "--output_format=tfjs_layers_model",
            str(keras_path),
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    print(f"[warn] converter CLI falhou: {result.stderr[:300]}")
    return False


def convert_via_subprocess(keras_path: str, output_dir: str) -> bool:
    """Chama tensorflowjs_converter como processo externo."""
    import subprocess
    import shutil
    converter_cmd = shutil.which("tensorflowjs_converter")
    if not converter_cmd:
        return False

    result = subprocess.run(
        [
            converter_cmd,
            "--input_format=keras",
            "--output_format=tfjs_layers_model",
            str(keras_path),
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    print(f"[warn] tensorflowjs_converter CLI falhou: {result.stderr[:500]}")
    return False


def manual_keras_to_tfjs(keras_path: str, output_dir: str) -> bool:
    """
    Converte manualmente um modelo Keras para o formato TF.js LayersModel.
    
    Processo:
      1. Carrega o modelo Keras
      2. Exporta pesos como arquivo binário (.bin)
      3. Gera model.json compatível com tf.loadLayersModel()
    """
    print("[manual] Iniciando conversão manual Keras → TF.js...")
    
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    
    # Carrega modelo
    print(f"[manual] Carregando {keras_path}...")
    model = keras.models.load_model(keras_path)
    print(f"[manual] Modelo carregado. Parâmetros: {model.count_params():,}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Coleta todos os pesos
    weight_manifests = []
    all_weights_bytes = bytearray()
    offset = 0
    
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if not layer_weights:
            continue
        
        layer_weight_names = [f"{layer.name}/{w.name}" for w in layer.weights]
        layer_weight_manifest = []
        
        for weight_array, weight_name in zip(layer_weights, layer_weight_names):
            weight_np = np.array(weight_array, dtype=np.float32)
            weight_bytes = weight_np.tobytes()
            byte_len = len(weight_bytes)
            
            # Adiciona ao buffer binário
            all_weights_bytes.extend(weight_bytes)
            
            layer_weight_manifest.append({
                "name": weight_name,
                "shape": list(weight_np.shape),
                "dtype": "float32",
                "quantization": None,
            })
            offset += byte_len
        
        weight_manifests.extend(layer_weight_manifest)
    
    # Salva arquivo binário
    bin_filename = "group1-shard1of1.bin"
    bin_path = output_path / bin_filename
    bin_path.write_bytes(bytes(all_weights_bytes))
    print(f"[manual] Pesos salvos em: {bin_path} ({len(all_weights_bytes):,} bytes)")
    
    # Gera model.json
    # Serializa configuração do modelo como TF.js espera
    model_config = model.get_config()
    
    # Recria o manifesto de pesos agrupados
    grouped_manifest = {
        "paths": [bin_filename],
        "weights": weight_manifests,
    }
    
    model_json = {
        "format": "layers-model",
        "generatedBy": "requiem-export_tfjs.py",
        "convertedBy": "manual",
        "modelTopology": {
            "class_name": model.__class__.__name__,
            "config": model_config,
            "keras_version": keras.__version__,
            "backend": "tensorflow",
        },
        "weightsManifest": [grouped_manifest],
        "userDefinedMetadata": {
            "modelName": "RequiemLSTM",
        },
    }
    
    model_json_path = output_path / "model.json"
    model_json_path.write_text(json.dumps(model_json, indent=2))
    print(f"[manual] model.json salvo em: {model_json_path}")
    
    return True


def convert_via_saved_model(keras_path: str, output_dir: str) -> bool:
    """
    Alternativa: salva como SavedModel TF2 primeiro, depois converte.
    Usa o comando tensorflowjs_converter que pode funcionar mesmo
    com o import quebrado do pacote Python.
    """
    import tensorflow as tf
    from tensorflow import keras
    from pathlib import Path
    import tempfile, subprocess, shutil
    
    print("[saved_model] Convertendo via SavedModel intermediário...")
    
    model = keras.models.load_model(keras_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = Path(tmpdir) / "saved_model"
        model.export(str(saved_model_path))
        print(f"[saved_model] SavedModel exportado em: {saved_model_path}")
        
        converter_bin = shutil.which("tensorflowjs_converter")
        if not converter_bin:
            print("[saved_model] tensorflowjs_converter não encontrado no PATH")
            return False
        
        result = subprocess.run(
            [
                converter_bin,
                "--input_format=tf_saved_model",
                "--output_format=tfjs_graph_model",
                "--signature_name=serving_default",
                "--saved_model_tags=serve",
                str(saved_model_path),
                str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            print(f"[saved_model] Conversão bem-sucedida!")
            return True
        else:
            print(f"[saved_model] FALHOU: {result.stderr[:500]}")
            return False


def main():
    keras_path = Path("saved_model/model.keras")
    output_dir = Path("../requiem-app-source/requiem-app/public/model")
    
    if not keras_path.exists():
        print(f"ERRO: Modelo não encontrado em {keras_path}")
        print("Execute primeiro: python3 train_model.py")
        sys.exit(1)
    
    print("=" * 55)
    print("  REQUIEM — Exportação do modelo para TF.js")
    print("=" * 55)
    print(f"  Modelo: {keras_path.resolve()}")
    print(f"  Saída : {output_dir.resolve()}")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tenta as estratégias em ordem
    success = False
    
    # Estratégia 1: CLI externo com SavedModel intermediário  
    if not success:
        print("[1/3] Tentando via SavedModel + tensorflowjs_converter...")
        try:
            success = convert_via_saved_model(str(keras_path), str(output_dir))
        except Exception as e:
            print(f"  Falhou: {e}")
    
    # Estratégia 2: subprocess com tensorflowjs_converter
    if not success:
        print("[2/3] Tentando via tensorflowjs_converter (keras direto)...")
        try:
            success = convert_via_subprocess(str(keras_path), str(output_dir))
        except Exception as e:
            print(f"  Falhou: {e}")
    
    # Estratégia 3: conversão manual
    if not success:
        print("[3/3] Tentando conversão manual...")
        try:
            success = manual_keras_to_tfjs(str(keras_path), str(output_dir))
        except Exception as e:
            print(f"  Falhou: {e}")
    
    if not success:
        print("\nERRO: Nenhuma estratégia de conversão funcionou.")
        sys.exit(1)
    
    # Copia vocab.json
    vocab_src = Path("saved_model/vocab.json")
    vocab_dst = output_dir / "vocab.json"
    if vocab_src.exists():
        shutil.copy2(vocab_src, vocab_dst)
        print(f"[ok] vocab.json copiado para: {vocab_dst}")
    
    # Verifica saída
    print("\n" + "=" * 55)
    print("  Arquivos gerados:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:40s} {size_kb:8.1f} KB")
    print("=" * 55)
    print("\n✓ Modelo exportado com sucesso!")
    print(f"  O frontend pode carregar em: /model/model.json")


if __name__ == "__main__":
    main()
