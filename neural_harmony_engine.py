"""
neural_harmony_engine.py
========================
Motor de harmonia baseado em LSTM — substituto drop-in do HarmonyEngine.ts.

COMO FUNCIONA A INFERÊNCIA:

  Dado um histórico de acordes e as notas detectadas pelo microfone,
  o motor combina duas fontes de conhecimento:

  ┌─────────────────────────────────────────────────────┐
  │  SCORE FINAL = (α × score_neural) + (β × score_notas) │
  └─────────────────────────────────────────────────────┘

  1. score_neural: probabilidade do LSTM para cada acorde candidato
     → O modelo aprendeu padrões harmônicos a partir das 612 progressões
     → Reflete o estilo musical aprendido

  2. score_notas: compatibilidade acústica do acorde com as notas tocadas
     → Idêntico ao scoring do HarmonyEngine.ts original
     → Reflete o que o usuário está tocando no momento

  A combinação garante que o motor é simultaneamente:
  - Musicalmente coerente (LSTM respeita progressões aprendidas)
  - Responsivo ao usuário (scoring acústico reage às notas reais)

DIFERENÇAS vs. MOTOR MARKOV (harmony_engine.py):
  ┌────────────────┬─────────────────────┬──────────────────────┐
  │ Aspecto        │ Markov (original)   │ LSTM (este arquivo)  │
  ├────────────────┼─────────────────────┼──────────────────────┤
  │ Contexto       │ 2 acordes           │ 4 acordes (ajustável)│
  │ Treinamento    │ Contagem + suaviz.  │ Backpropagation      │
  │ Generalização  │ Limitada            │ Melhor               │
  │ Requisitos     │ Nenhum              │ TensorFlow           │
  │ Velocidade     │ Instantânea         │ < 10ms por previsão  │
  └────────────────┴─────────────────────┴──────────────────────┘

USO:
  # Treine primeiro:
  #   python3 train_model.py
  #
  # Depois use:
  from neural_harmony_engine import NeuralHarmonyEngine, generate_progression_neural

  engine = NeuralHarmonyEngine()   # carrega modelo salvo
  result = engine.generate(notes, bpm=120)
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Optional
import numpy as np

import tensorflow as tf
from tensorflow import keras

from tonality_adapter import (
    DetectedNote,
    normalize_notes,
    transpose_chord,
    get_chord_pitch_classes,
    chord_to_roman,
)
from harmony_engine import (
    HARMONY_GRAPH,
    ChordResult,
    split_cinematic,
    # constantes de scoring acústico (reutilizamos do motor original)
    WEIGHT_IN_CHORD,
    WEIGHT_ROOT_BONUS,
    WEIGHT_OUT_PENALTY,
    WEIGHT_STRONG_BEAT,
    WEIGHT_WEAK_BEAT,
    BEAT_MARGIN_SEC,
    REGISTER_LOW_THRESHOLD,
    REGISTER_HIGH_THRESHOLD,
    REGISTER_PENALTY,
    REGISTER_BONUS,
    DENSITY_THRESHOLD,
    ORNAMENT_PENALTY_MULTIPLIER,
    LEAP_THRESHOLD,
    TENSION_BONUS,
)
from chord_dataset import Vocabulary, WINDOW_SIZE
from audio_analyzer import detect_key, estimate_bpm

# Caminhos padrão do modelo salvo
DEFAULT_MODEL_DIR = "saved_model"

# Peso relativo do modelo neural vs. scoring acústico.
# NEURAL_WEIGHT=15.0 dá mais importância ao estilo aprendido;
# reduza para 5.0 se quiser que o modelo reaja mais às notas tocadas.
NEURAL_WEIGHT = 15.0

# Pesos do scoring acústico (mesmo do motor original)
ACOUSTIC_WEIGHT = 1.0

# Limiar para aceitar acordes não permitidos pelo grafo harmônico.
# Se a probabilidade neural for > NEURAL_OVERRIDE_THRESHOLD,
# o modelo pode sugerir acordes fora das transições padrão.
NEURAL_OVERRIDE_THRESHOLD = 0.15


#  1. Engine Neural

class NeuralHarmonyEngine:
    """
    Motor de harmonia baseado em LSTM.

    Substitui identicamente o HarmonyEngine.ts, mas usa uma
    rede neural treinada em vez de cadeias de Markov.

    Parâmetros
    ----------
    model_dir : str | Path
        Diretório do modelo salvo (output de train_model.py).
        Deve conter: saved_model.pb, vocab.json, variables/
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
        model_dir = Path(model_dir)

        # Carrega o modelo TensorFlow salvo
        model_path = Path(model_dir)
        keras_file = model_path / "model.keras"
        if not keras_file.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em: {keras_file}\n"
                f"Execute primeiro: python3 train_model.py"
            )
        self._model: keras.Model = keras.models.load_model(str(keras_file))

        # Carrega o vocabulário
        vocab_path = model_dir / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulário não encontrado: {vocab_path}")
        self._vocab = Vocabulary.load(vocab_path)

        # Pré-calcula o índice de todos os acordes do HARMONY_GRAPH
        # para evitar overhead durante a inferência em tempo real
        self._graph_chords = list(HARMONY_GRAPH.keys())

        print(f"[NeuralHarmonyEngine] Modelo carregado.")
        print(f"  Vocabulário: {len(self._vocab)} acordes")
        print(f"  Parâmetros : {self._model.count_params():,}")

    # Predição Neural

    def _predict_probabilities(self, chord_history: list[str]) -> np.ndarray:
        """
        Dado o histórico de acordes, retorna a distribuição de
        probabilidade do LSTM sobre todo o vocabulário.

        Parâmetros
        ----------
        chord_history : list[str]
            Acordes anteriores (mais recente = último elemento).
            Ex: ["C", "Am", "F", "G"]

        Retorna
        -------
        np.ndarray de shape (vocab_size,) com probabilidades (somam 1.0).
        """
        pad_idx = 0  # PAD sempre tem índice 0

        # Pega os últimos WINDOW_SIZE acordes do histórico
        # Se o histórico for menor, preenche com PAD à esquerda
        context = chord_history[-WINDOW_SIZE:]
        while len(context) < WINDOW_SIZE:
            context = ["<PAD>"] + context

        # Codifica para índices inteiros
        encoded = np.array(
            [[self._vocab.encode(c) for c in context]],
            dtype=np.int32,
        )  # shape: (1, WINDOW_SIZE)

        # Inferência: passa pelo modelo e obtém probabilidades
        # O modelo retorna shape (1, vocab_size)
        probs: np.ndarray = self._model.predict(encoded, verbose=0)[0]
        return probs  # shape: (vocab_size,)

    # Scoring Acústico (mesmo do motor original)

    def _compute_note_score(
        self,
        chord_notes: list[int],
        played_notes: list[DetectedNote],
        window_start: float,
        seconds_per_beat: float,
        is_high_density: bool,
        avg_amplitude: float,
    ) -> float:
        """
        Calcula o score de compatibilidade acústica entre um acorde
        e as notas tocadas. Idêntico ao computeNoteScore() do TS original.
        """
        score = 0.0
        root_note = chord_notes[0]
        out_penalty = (
            WEIGHT_OUT_PENALTY * ORNAMENT_PENALTY_MULTIPLIER
            if is_high_density else WEIGHT_OUT_PENALTY
        )

        for note in played_notes:
            pc = round(note.pitch) % 12
            is_chord_tone = pc in chord_notes
            is_root = (pc == root_note)

            # Peso rítmico: notas em tempo forte valem mais
            rel_time = note.start_time - window_start
            beat_phase = (
                (rel_time % seconds_per_beat) / seconds_per_beat
                if seconds_per_beat > 0 else 0
            )
            is_strong = beat_phase < BEAT_MARGIN_SEC or beat_phase > 1 - BEAT_MARGIN_SEC
            rhythmic_weight = WEIGHT_STRONG_BEAT if is_strong else WEIGHT_WEAK_BEAT

            # Peso de amplitude: notas mais fortes têm mais influência
            amp_weight = (note.amplitude or 0.7) / avg_amplitude

            # Modificador de registro (oitava)
            register_mod = 1.0
            if note.pitch < REGISTER_LOW_THRESHOLD and is_root:
                register_mod = REGISTER_BONUS
            if note.pitch > REGISTER_HIGH_THRESHOLD and not is_chord_tone:
                register_mod = REGISTER_PENALTY

            base_val = (
                (WEIGHT_IN_CHORD + (WEIGHT_ROOT_BONUS if is_root else 0))
                if is_chord_tone else -out_penalty
            )
            score += base_val * rhythmic_weight * amp_weight * register_mod

        return score

    # Determinação do Próximo Acorde

    def determine_next_chord(
        self,
        chord_history: list[str],
        played_notes: list[DetectedNote],
        window_start: float,
        seconds_per_beat: float,
    ) -> ChordResult:
        """
        Elege o próximo acorde combinando LSTM + scoring acústico.

        Equivalente exato a determineNextChord() do HarmonyEngine.ts,
        mas usando o modelo neural em vez da cadeia de Markov.

        Parâmetros
        ----------
        chord_history : list[str]
            Histórico de acordes gerados (ex: ["C", "Am", "F"]).
        played_notes : list[DetectedNote]
            Notas detectadas no compasso atual.
        window_start : float
            Início do compasso atual em segundos.
        seconds_per_beat : float
            Duração de uma semínima em segundos (60/BPM).

        Retorna
        -------
        ChordResult com o acorde eleito e a velocity média.
        """
        current_chord = chord_history[-1] if chord_history else "C"

        # Sem notas → mantém acorde atual (igual ao motor original)
        if not played_notes:
            return ChordResult(chord=current_chord, velocity=0.7)

        # Obtém distribuição de probabilidade do LSTM
        neural_probs = self._predict_probabilities(chord_history)

        # Contexto acústico
        density = len(played_notes)
        is_high_density = density > DENSITY_THRESHOLD
        sum_amp = sum(n.amplitude or 0.7 for n in played_notes)
        avg_amplitude = max(0.2, sum_amp / density)

        has_tension_leap = any(
            abs(played_notes[i].pitch - played_notes[i-1].pitch) >= LEAP_THRESHOLD
            for i in range(1, density)
        )

        # Nó atual no grafo harmônico
        current_base_roman, _ = split_cinematic(chord_to_roman(current_chord, "C"))
        current_base_name = next(
            (c for c in HARMONY_GRAPH if chord_to_roman(c, "C") == current_base_roman),
            "C",
        )
        current_node = HARMONY_GRAPH.get(current_base_name)

        best_chord = current_chord
        best_score = -math.inf

        # Itera sobre todos os acordes do HARMONY_GRAPH
        for candidate in self._graph_chords:
            # Probabilidade neural para este acorde
            cand_idx = self._vocab.encode(candidate)
            neural_prob = float(neural_probs[cand_idx]) if cand_idx > 0 else 0.0

            # Filtro harmônico: aceita se:
            # (a) está nas transições permitidas do grafo, OU
            # (b) o LSTM tem probabilidade alta o suficiente
            is_allowed_graph = (
                current_node is not None
                and candidate in current_node.allowed_transitions
            )
            is_allowed_neural = neural_prob > NEURAL_OVERRIDE_THRESHOLD

            if not is_allowed_graph and not is_allowed_neural:
                continue

            # Bônus de tensão para acordes de dominante/diminuto
            if has_tension_leap and ("7" in candidate or "dim" in candidate):
                neural_prob *= TENSION_BONUS

            # Scoring acústico
            chord_notes = get_chord_pitch_classes(candidate)
            acoustic_score = self._compute_note_score(
                chord_notes, played_notes, window_start,
                seconds_per_beat, is_high_density, avg_amplitude,
            )

            # Score final: combinação linear ponderada
            #   neural_prob  → quanto o modelo "quer" este acorde (estilo)
            #   acoustic_score → quanto as notas "pedem" este acorde (contexto)
            final_score = (
                ACOUSTIC_WEIGHT * acoustic_score
                + NEURAL_WEIGHT * neural_prob
            )

            if final_score > best_score:
                best_score = final_score
                best_chord = candidate

        return ChordResult(chord=best_chord, velocity=avg_amplitude)

    # Geração de Progressão Completa

    def generate(
        self,
        played_notes: list[DetectedNote],
        bpm: float = 120,
        time_signature: tuple[int, int] = (4, 4),
        start_chord: str = "C",
        tonality: Optional[str] = None,
        auto_detect: bool = True,
    ) -> list[ChordResult]:
        """
        Gera uma progressão harmônica completa para as notas tocadas.

        Equivalente exato a generateProgression() do HarmonyEngine.ts.

        Parâmetros
        ----------
        played_notes : list[DetectedNote]
            Notas captadas pelo microfone.
        bpm : float
            BPM da música.
        time_signature : (int, int)
            Fórmula de compasso (numerador, denominador). Ex: (4, 4).
        start_chord : str
            Acorde inicial (padrão: "C").
        tonality : str, opcional
            Tonalidade. Se None e auto_detect=True, detecta automaticamente.
        auto_detect : bool
            Se True, detecta tonalidade e BPM automaticamente.

        Retorna
        -------
        list[ChordResult] — um acorde por compasso.
        """
        if not played_notes:
            return [ChordResult(chord=start_chord, velocity=0.7)]

        # Auto-detecção de tonalidade e BPM
        effective_tonality = tonality or "C"
        effective_bpm = bpm

        if auto_detect:
            if tonality is None:
                effective_tonality = detect_key(played_notes)
            effective_bpm = estimate_bpm(played_notes) if bpm == 120 else bpm

        # Normaliza notas para Dó Maior (espaço local do motor)
        local_notes = normalize_notes(played_notes, effective_tonality)

        # Parâmetros de tempo
        numerator, denominator = time_signature
        seconds_per_beat = (4 / denominator) * (60 / effective_bpm)
        seconds_per_measure = seconds_per_beat * numerator

        # Define janelas de compasso
        total_start = min(n.start_time for n in local_notes)
        total_end = max(n.end_time for n in local_notes)
        total_duration = total_end - total_start
        window_count = max(1, math.ceil(total_duration / seconds_per_measure))

        # Gera acorde por compasso
        progression: list[ChordResult] = []
        history = [start_chord]

        for w in range(window_count):
            window_start = total_start + w * seconds_per_measure
            window_end = window_start + seconds_per_measure

            # Filtra notas do compasso atual
            window_notes = [
                n for n in local_notes
                if n.start_time < window_end and n.end_time > window_start
            ]
            # Recorta as notas para os limites do compasso
            clipped = [
                DetectedNote(
                    pitch=n.pitch,
                    start_time=max(n.start_time, window_start),
                    end_time=min(n.end_time, window_end),
                    amplitude=n.amplitude,
                )
                for n in window_notes
            ]

            # Determina próximo acorde
            result = self.determine_next_chord(
                history, clipped, window_start, seconds_per_beat
            )
            # Transpõe de Dó Maior para a tonalidade real do usuário
            transposed_chord = transpose_chord(result.chord, effective_tonality)
            final_result = ChordResult(chord=transposed_chord, velocity=result.velocity)

            progression.append(final_result)
            history.append(result.chord)  # histórico em Dó Maior (espaço local)

        return progression


#  2. Função de conveniência (drop-in para harmony_engine.py)

_engine_cache: Optional[NeuralHarmonyEngine] = None


def get_neural_engine(model_dir: str | Path = DEFAULT_MODEL_DIR) -> NeuralHarmonyEngine:
    """Retorna o engine singleton (instanciado uma única vez)."""
    global _engine_cache
    if _engine_cache is None:
        _engine_cache = NeuralHarmonyEngine(model_dir)
    return _engine_cache


def generate_progression_neural(
    played_notes: list[DetectedNote],
    bpm: float = 120,
    time_signature_numerator: int = 4,
    time_signature_denominator: int = 4,
    start_chord: str = "C",
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> list[ChordResult]:
    """
    Wrapper de conveniência com assinatura idêntica a generate_progression()
    do harmony_engine.py — facilita troca drop-in entre os dois motores.
    """
    engine = get_neural_engine(model_dir)
    return engine.generate(
        played_notes=played_notes,
        bpm=bpm,
        time_signature=(time_signature_numerator, time_signature_denominator),
        start_chord=start_chord,
        auto_detect=False,
    )
