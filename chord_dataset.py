"""
chord_dataset.py
================
Prepara o dataset de treinamento para o modelo LSTM de harmonia.

PIPELINE:
  1. Carrega os 51 JSONs de music_data/
  2. Normaliza cada progressão para Dó Maior (espaço local)
  3. Aplica data augmentation: transpõe para todas as 12 tonalidades
     → 51 × 12 = 612 progressões de treino
  4. Constrói o vocabulário de acordes
  5. Gera janelas deslizantes (sliding windows) para treino supervisionado

FORMATO DO PROBLEMA (supervised learning):
  - Entrada (X): sequência de N acordes anteriores (janela de contexto)
  - Saída (y): próximo acorde (índice no vocabulário)

  Exemplo com window_size=4:
    X = [C, Am, F, G]  →  y = C     (índice de C no vocabulário)
    X = [Am, F, G, C]  →  y = Am

VOCABULÁRIO:
  Todos os acordes únicos encontrados nos dados + token especial <PAD>
  para preencher janelas no início das sequências.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import NamedTuple
import numpy as np

from tonality_adapter import (
    normalize_progression_to_c,
    transpose_chord,
    NOTE_NAMES_SHARP,
    ROOT_OFFSETS,
)

# ─────────────────────────────────────────────────────────
#  1. Constantes
# ─────────────────────────────────────────────────────────

# Tamanho da janela de contexto:
# O modelo vê os últimos WINDOW_SIZE acordes para prever o próximo.
# Valor 4 captura padrões de 1 compasso (I-IV-V-I) sem explodir o modelo.
WINDOW_SIZE = 4

# Token especial para preencher o início das sequências (padding)
PAD_TOKEN = "<PAD>"

# ─────────────────────────────────────────────────────────
#  2. Carregamento dos JSONs
# ─────────────────────────────────────────────────────────

def load_raw_progressions(data_dir: str | Path = "music_data") -> list[dict]:
    """
    Carrega todos os JSONs do diretório de dados.
    Retorna lista de dicts com title, originalTonality, normalizedProgression.
    """
    path = Path(data_dir)
    raw_data = []
    for f in sorted(path.glob("*.json")):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(raw.get("normalizedProgression"), list):
                raw_data.append(raw)
        except Exception:
            pass
    return raw_data


# ─────────────────────────────────────────────────────────
#  3. Data Augmentation: transposição para todas as 12 tonalidades
# ─────────────────────────────────────────────────────────

def _transpose_chord_by_semitones(chord: str, semitones: int) -> str:
    """
    Transpõe um acorde (em Dó Maior) por N semitons.
    Usado para gerar versões augmentadas do dataset.

    Exemplo: _transpose_chord_by_semitones("C", 2) → "D"
    """
    from tonality_adapter import CHORD_REGEX, ROOT_OFFSETS, NOTE_NAMES_SHARP
    import re
    m = re.match(r"^([A-G][#b]?)(.*)$", chord)
    if not m:
        return chord
    root, suffix = m.group(1), m.group(2)
    root_idx = ROOT_OFFSETS.get(root)
    if root_idx is None:
        return chord
    new_idx = (root_idx + semitones) % 12
    return NOTE_NAMES_SHARP[new_idx] + suffix


def augment_progressions(raw_data: list[dict]) -> list[list[str]]:
    """
    Gera progressões aumentadas por transposição cromática.

    Para cada progressão do dataset:
      1. Normaliza para Dó Maior
      2. Transpõe para cada uma das 12 tonalidades (shift 0 a 11)
      → Resultado: 51 músicas × 12 = 612 progressões

    Isso é fundamental porque nosso dataset é pequeno (51 músicas).
    A transposição é musicalmente válida: I→IV→V em Dó é
    exatamente o mesmo padrão que I→IV→V em Ré.
    """
    all_progressions: list[list[str]] = []

    for item in raw_data:
        tonality = item["originalTonality"]
        prog = item["normalizedProgression"]

        # Normaliza a progressão original para Dó Maior
        normalized = normalize_progression_to_c(prog, tonality)

        # Transpõe para as 12 tonalidades (incluindo a original = shift 0)
        for semitones in range(12):
            transposed = [_transpose_chord_by_semitones(c, semitones) for c in normalized]
            all_progressions.append(transposed)

    return all_progressions


# ─────────────────────────────────────────────────────────
#  4. Vocabulário
# ─────────────────────────────────────────────────────────

class Vocabulary:
    """
    Mapeia acordes (strings) ↔ índices inteiros.

    O token PAD tem sempre índice 0 e é usado para preencher
    o início das sequências (antes de haver contexto suficiente).

    Exemplo:
      vocab["C"]   → 1
      vocab["Am"]  → 7
      vocab.decode(7) → "Am"
    """

    def __init__(self, chords: list[str]) -> None:
        # Garante que PAD é sempre o índice 0
        unique = [PAD_TOKEN] + sorted(set(chords))
        self.idx_to_chord: list[str] = unique
        self.chord_to_idx: dict[str, int] = {c: i for i, c in enumerate(unique)}

    def __len__(self) -> int:
        return len(self.idx_to_chord)

    def encode(self, chord: str) -> int:
        """Converte nome de acorde → índice. Usa 0 (PAD) para desconhecidos."""
        return self.chord_to_idx.get(chord, 0)

    def decode(self, idx: int) -> str:
        """Converte índice → nome de acorde."""
        if 0 <= idx < len(self.idx_to_chord):
            return self.idx_to_chord[idx]
        return PAD_TOKEN

    def save(self, path: str | Path) -> None:
        """Salva o vocabulário em JSON para reprodutibilidade."""
        import json
        Path(path).write_text(
            json.dumps({"vocab": self.idx_to_chord}, indent=2),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Carrega vocabulário salvo anteriormente."""
        import json
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = cls.__new__(cls)
        vocab.idx_to_chord = data["vocab"]
        vocab.chord_to_idx = {c: i for i, c in enumerate(vocab.idx_to_chord)}
        return vocab


def build_vocabulary(progressions: list[list[str]]) -> Vocabulary:
    """Constrói o vocabulário a partir de todas as progressões."""
    all_chords = [chord for prog in progressions for chord in prog]
    return Vocabulary(all_chords)


# ─────────────────────────────────────────────────────────
#  5. Sliding Window → Amostras de Treino
# ─────────────────────────────────────────────────────────

class TrainingData(NamedTuple):
    """Resultado do pré-processamento: arrays NumPy prontos para o Keras."""
    X: np.ndarray      # shape: (n_samples, WINDOW_SIZE) — sequências de contexto
    y: np.ndarray      # shape: (n_samples,)             — próximo acorde (índice)
    vocab: Vocabulary  # vocabulário usado


def create_training_data(
    progressions: list[list[str]],
    vocab: Vocabulary,
    window_size: int = WINDOW_SIZE,
) -> TrainingData:
    """
    Gera amostras de treino por janela deslizante.

    Para uma progressão [C, Am, F, G, C] com window_size=4:

      Janela 1: X=[PAD, PAD, PAD,  C]  y=Am
      Janela 2: X=[PAD, PAD,   C, Am]  y=F
      Janela 3: X=[PAD,   C,  Am,  F]  y=G
      Janela 4: X=[  C,  Am,   F,  G]  y=C

    O padding no início permite que o modelo aprenda também
    a partir do início das progressões (sem histórico prévio).

    Parâmetros
    ----------
    progressions : list[list[str]]
        Lista de progressões (já aumentadas e normalizadas).
    vocab : Vocabulary
        Vocabulário para codificação.
    window_size : int
        Tamanho da janela de contexto (padrão: 4).

    Retorna
    -------
    TrainingData com arrays X (contexto) e y (alvo).
    """
    X_list: list[list[int]] = []
    y_list: list[int] = []
    pad_idx = vocab.encode(PAD_TOKEN)  # = 0

    for prog in progressions:
        # Codifica toda a progressão em índices
        encoded = [vocab.encode(chord) for chord in prog]

        # Desliza a janela sobre a progressão
        for i in range(len(encoded)):
            # Pega os window_size acordes anteriores (com padding à esquerda)
            start = i - window_size
            context = []
            for j in range(start, i):
                if j < 0:
                    context.append(pad_idx)   # padding
                else:
                    context.append(encoded[j])

            target = encoded[i]  # próximo acorde a prever

            X_list.append(context)
            y_list.append(target)

    X = np.array(X_list, dtype=np.int32)   # (n_samples, window_size)
    y = np.array(y_list, dtype=np.int32)   # (n_samples,)

    return TrainingData(X=X, y=y, vocab=vocab)


# ─────────────────────────────────────────────────────────
#  6. Função principal de preparação
# ─────────────────────────────────────────────────────────

def prepare_dataset(
    data_dir: str | Path = "music_data",
    window_size: int = WINDOW_SIZE,
    augment: bool = True,
    verbose: bool = True,
) -> TrainingData:
    """
    Pipeline completo de preparação do dataset.

    1. Carrega os JSONs
    2. Normaliza para Dó Maior
    3. Aplica data augmentation (×12 tonalidades)
    4. Constrói vocabulário
    5. Gera amostras de treino (sliding window)

    Parâmetros
    ----------
    data_dir : str | Path
        Diretório com os JSONs de treinamento.
    window_size : int
        Tamanho da janela de contexto.
    augment : bool
        Se True, aplica augmentation (recomendado).
    verbose : bool
        Se True, imprime estatísticas do dataset.

    Retorna
    -------
    TrainingData pronto para alimentar o modelo Keras.
    """
    # 1. Carrega dados brutos
    raw = load_raw_progressions(data_dir)

    # 2. Aumenta (ou não)
    if augment:
        progressions = augment_progressions(raw)
    else:
        progressions = [
            normalize_progression_to_c(item["normalizedProgression"], item["originalTonality"])
            for item in raw
        ]

    # 3. Vocabulário
    vocab = build_vocabulary(progressions)

    # 4. Amostras de treino
    data = create_training_data(progressions, vocab, window_size)

    if verbose:
        print(f"[Dataset] Progressões originais : {len(raw)}")
        print(f"[Dataset] Progressões (augment) : {len(progressions)}")
        print(f"[Dataset] Tamanho do vocabulário: {len(vocab)} acordes")
        print(f"[Dataset] Amostras de treino    : {len(data.X)}")
        print(f"[Dataset] Janela de contexto    : {window_size} acordes")
        print(f"[Dataset] Shape X               : {data.X.shape}")
        print(f"[Dataset] Shape y               : {data.y.shape}")

    return data
