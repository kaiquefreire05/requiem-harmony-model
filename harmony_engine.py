"""
harmony_engine.py
=================
Porta Python fiel ao HarmonyEngine.ts do Requiem App.

Pipeline:
  1. Carrega JSONs de music_data → extrai progressões
  2. Constrói modelo Two-Tier de Markov (base + sufixos)
  3. determineNextChord() — elege o próximo acorde
  4. generateProgression() — gera progressão completa
"""

from __future__ import annotations
import re
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from tonality_adapter import (
    DetectedNote,
    chord_to_roman,
    get_chord_pitch_classes,
    TONALITY_OFFSETS,
)

#  1. Tipos

@dataclass
class ChordNode:
    notes: list[int]
    allowed_transitions: list[str]

@dataclass
class ProgressionData:
    title: str
    original_tonality: str
    normalized_progression: list[str]

TransitionMatrix = dict[str, dict[str, float]]

@dataclass
class TwoTierMarkovModel:
    base_matrix: TransitionMatrix = field(default_factory=dict)
    suffix_matrix: TransitionMatrix = field(default_factory=dict)

@dataclass
class ChordResult:
    chord: str
    velocity: float

#  2. Grafo Harmônico (Knowledge Base)

HARMONY_GRAPH: dict[str, ChordNode] = {
    # Maiores
    "C":   ChordNode([0,4,7],   ["C","G","F","Am","Dm","Em","G7","C7","E7","A7","D7","Bdim"]),
    "G":   ChordNode([7,11,2],  ["G","C","D","Em","Am","Bm","D7","G7","B7","E7","F#dim"]),
    "D":   ChordNode([2,6,9],   ["D","G","A","Bm","Em","F#m","A7","D7","F#dim","E7"]),
    "A":   ChordNode([9,1,4],   ["A","D","E","F#m","Bm","C#m","E7","A7","C#dim","B7"]),
    "E":   ChordNode([4,8,11],  ["E","A","B","C#m","F#m","G#m","B7","E7","G#dim"]),
    "B":   ChordNode([11,3,6],  ["B","E","F#","G#m","C#m","D#m","F#7","B7"]),
    "F#":  ChordNode([6,10,1],  ["F#","B","C#","D#m","G#m","F#7","C#7"]),
    "F":   ChordNode([5,9,0],   ["F","C","Bb","Dm","Am","Gm","C7","F7","D7","A7"]),
    "Bb":  ChordNode([10,2,5],  ["Bb","F","Eb","Gm","Dm","Cm","F7","Bb7","D7"]),
    "Eb":  ChordNode([3,7,10],  ["Eb","Bb","Ab","Cm","Gm","Fm","Bb7","Eb7","G7"]),
    "Ab":  ChordNode([8,0,3],   ["Ab","Eb","Db","Fm","Cm","Bbm","Eb7","Ab7"]),
    "Db":  ChordNode([1,5,8],   ["Db","Ab","Gb","Bbm","Fm","Ab7","Db7"]),
    # Menores
    "Am":  ChordNode([9,0,4],   ["Am","C","Dm","Em","F","G","E7","G7","A7","Bdim","D7"]),
    "Em":  ChordNode([4,7,11],  ["Em","G","Am","C","D","Bm","B7","D7","F#dim","E7"]),
    "Bm":  ChordNode([11,2,6],  ["Bm","D","Em","G","A","F#m","F#7","A7","F#dim"]),
    "F#m": ChordNode([6,9,1],   ["F#m","A","Bm","D","E","C#m","C#7","E7","C#dim"]),
    "C#m": ChordNode([1,4,8],   ["C#m","E","F#m","A","B","G#m","G#7","B7","G#dim"]),
    "G#m": ChordNode([8,11,3],  ["G#m","B","C#m","E","F#","D#m","D#7","F#7"]),
    "D#m": ChordNode([3,6,10],  ["D#m","F#","G#m","B","D#7","A#7"]),
    "Dm":  ChordNode([2,5,9],   ["Dm","F","Am","C","Bb","Gm","A7","C7","D7","Bdim"]),
    "Gm":  ChordNode([7,10,2],  ["Gm","Bb","Dm","F","Eb","Cm","D7","F7","Bb7"]),
    "Cm":  ChordNode([0,3,7],   ["Cm","Eb","Gm","Bb","Ab","Fm","G7","Bb7","Eb7"]),
    "Fm":  ChordNode([5,8,0],   ["Fm","Ab","Cm","Eb","Db","Bbm","C7","Eb7","Ab7"]),
    "Bbm": ChordNode([10,1,5],  ["Bbm","Db","Fm","Ab","F7","Ab7","Db7"]),
    # Dominantes
    "G7":  ChordNode([7,11,2,5],  ["C","Am","Cm","G7","C7","F","Dm"]),
    "D7":  ChordNode([2,6,9,0],   ["G","Em","Gm","D7","G7","C","Am"]),
    "A7":  ChordNode([9,1,4,7],   ["D","Bm","Dm","A7","D7","G","Em"]),
    "E7":  ChordNode([4,8,11,2],  ["A","F#m","Am","E7","A7","D","Bm"]),
    "B7":  ChordNode([11,3,6,9],  ["E","C#m","Em","B7","E7","A","F#m"]),
    "F#7": ChordNode([6,10,1,4],  ["B","G#m","Bm","F#7","B7","E","C#m"]),
    "C#7": ChordNode([1,5,8,11],  ["F#","D#m","F#m","C#7","F#7","B"]),
    "C7":  ChordNode([0,4,7,10],  ["F","Dm","Fm","C7","F7","Bb","Gm"]),
    "F7":  ChordNode([5,9,0,3],   ["Bb","Gm","Bbm","F7","Bb7","Eb","Cm"]),
    "Bb7": ChordNode([10,2,5,8],  ["Eb","Cm","Bb7","Eb7","Ab","Fm"]),
    "Eb7": ChordNode([3,7,10,1],  ["Ab","Fm","Eb7","Ab7","Db"]),
    "Ab7": ChordNode([8,0,3,6],   ["Db","Bbm","Ab7","Db7"]),
    "Db7": ChordNode([1,5,8,11],  ["Gb","Db7","Ab","Fm"]),
    "D#7": ChordNode([3,7,10,1],  ["G#m","D#7","G#","F"]),
    "G#7": ChordNode([8,0,3,6],   ["C#m","G#7","C#","F#m"]),
    "A#7": ChordNode([10,2,5,8],  ["D#m","A#7","D#","G#m"]),
    # Diminutos
    "Bdim":  ChordNode([11,2,5], ["C","Am","G7","Dm","F","Em"]),
    "F#dim": ChordNode([6,9,0],  ["G","Em","D7","Am","C","Bm"]),
    "C#dim": ChordNode([1,4,7],  ["D","Bm","A7","Em","G","F#m"]),
    "G#dim": ChordNode([8,11,2], ["A","F#m","E7","Bm","D","C#m"]),
    "D#dim": ChordNode([3,6,9],  ["E","C#m","B7","F#m","A"]),
    "A#dim": ChordNode([10,1,4], ["B","G#m","F#7","C#m","E"]),
    "Edim":  ChordNode([4,7,10], ["F","Dm","C7","Am","Bb","Gm"]),
    "Adim":  ChordNode([9,0,3],  ["Bb","Gm","F7","Dm","Eb","Cm"]),
    "Ddim":  ChordNode([2,5,8],  ["Eb","Cm","Bb7","Gm","Ab","Fm"]),
    "Gdim":  ChordNode([7,10,1], ["Ab","Fm","Eb7","Cm","Db"]),
    "Cdim":  ChordNode([0,3,6],  ["Db","Bbm","Ab7","Fm"]),
    "Fdim":  ChordNode([5,8,11], ["Gb","Ebm","Db7","Bbm"]),
    # Aliases enarmônicos
    "C#":  ChordNode([1,5,8],  ["Db","Ab","Gb","Bbm","Fm","Ab7","Db7"]),
    "Gb":  ChordNode([6,10,1], ["F#","B","C#","D#m","G#m","F#7","C#7"]),
    "G#":  ChordNode([8,0,3],  ["Ab","Eb","Db","Fm","Cm","Bbm","Eb7","Ab7"]),
    "D#":  ChordNode([3,7,10], ["Eb","Bb","Ab","Cm","Gm","Fm","Bb7","Eb7","G7"]),
    "Ebm": ChordNode([3,6,10], ["D#m","F#","G#m","B","D#7","A#7"]),
}

#  3. Pesos e constantes

WEIGHT_IN_CHORD = 2.0
WEIGHT_ROOT_BONUS = 1.5
WEIGHT_OUT_PENALTY = 0.5
WEIGHT_STRONG_BEAT = 3.0
WEIGHT_WEAK_BEAT = 0.5
BEAT_MARGIN_SEC = 0.15
REGISTER_LOW_THRESHOLD = 48
REGISTER_HIGH_THRESHOLD = 72
REGISTER_PENALTY = 0.2
REGISTER_BONUS = 1.5
DENSITY_THRESHOLD = 8
ORNAMENT_PENALTY_MULTIPLIER = 0.2
LEAP_THRESHOLD = 7
TENSION_BONUS = 1.5
WEIGHT_MARKOV = 20.0
THRESHOLD_MODAL_INTERCHANGE = 0.05
ALPHA = 0.1  # Laplace smoothing

#  4. Carregamento dos JSONs

def load_music_data(data_dir: str | Path = "music_data") -> list[ProgressionData]:
    """Carrega todos os JSONs do diretório music_data."""
    path = Path(data_dir)
    datasets: list[ProgressionData] = []
    for f in sorted(path.glob("*.json")):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(raw.get("normalizedProgression"), list) and isinstance(raw.get("originalTonality"), str):
                datasets.append(ProgressionData(
                    title=raw.get("title", f.stem),
                    original_tonality=raw["originalTonality"],
                    normalized_progression=raw["normalizedProgression"],
                ))
        except Exception:
            pass
    return datasets

#  5. Separação base/sufixo cinematográfico

HARMONY_GRAPH_ROMANS: set[str] = set()  # preenchido após definir grafo

_ROMAN_SPLIT_RE = re.compile(
    r"^([b#]?(?:III|iii|II|ii|IV|iv|VIII|viii|VII|vii|VI|vi|V|v|I|i))(.*)$"
)

def _init_romans():
    global HARMONY_GRAPH_ROMANS
    HARMONY_GRAPH_ROMANS = {chord_to_roman(c, "C") for c in HARMONY_GRAPH}

_init_romans()


def split_cinematic(roman: str) -> tuple[str, str]:
    """
    Separa o numeral romano base do sufixo (ex: 'Iadd9' → ('I','add9')).
    Equivalente a splitCinematic() do HarmonyEngine.ts.
    """
    if roman in HARMONY_GRAPH_ROMANS:
        return roman, "none"
    m = _ROMAN_SPLIT_RE.match(roman)
    if not m:
        return roman, "none"
    base = m.group(1)
    suffix = m.group(2) or "none"
    return base, suffix

#  6. Construção do modelo Two-Tier de Markov

def build_two_tier_markov_model(datasets: list[ProgressionData]) -> TwoTierMarkovModel:
    """
    Constrói o modelo Two-Tier a partir dos datasets.
    Equivalente a buildTwoTierMarkovModel() do HarmonyEngine.ts.
    """
    all_base_degrees = list({chord_to_roman(c, "C") for c in HARMONY_GRAPH})

    # Matriz Base
    base_counts: dict[str, dict[str, float]] = {}

    for mod in datasets:
        prog = [chord_to_roman(c, mod.original_tonality) for c in mod.normalized_progression]
        base_prog = [split_cinematic(r)[0] for r in prog]

        for i in range(len(base_prog) - 1):
            from1 = base_prog[i]
            to = base_prog[i + 1]
            base_counts.setdefault(from1, {})[to] = base_counts.get(from1, {}).get(to, 0) + 1

            if i >= 1:
                from2 = base_prog[i - 1]
                state2 = f"{from2},{from1}"
                base_counts.setdefault(state2, {})[to] = base_counts.get(state2, {}).get(to, 0) + 1

    base_matrix: TransitionMatrix = {}
    for state, destinations in base_counts.items():
        row: dict[str, float] = {}
        total = 0.0
        for deg in all_base_degrees:
            count = destinations.get(deg, 0) + ALPHA
            row[deg] = count
            total += count
        for deg in all_base_degrees:
            row[deg] /= total
        base_matrix[state] = row

    # Matriz de Sufixos
    all_extensions: set[str] = {"none","add9","madd9","sus4","sus2","maj7","m7b5","aug","m7","7","11","m11","5"}
    suffix_counts: dict[str, dict[str, float]] = {}

    for mod in datasets:
        prog = [chord_to_roman(c, mod.original_tonality) for c in mod.normalized_progression]
        for r in prog:
            base, ext = split_cinematic(r)
            suffix_counts.setdefault(base, {})[ext] = suffix_counts.get(base, {}).get(ext, 0) + 1
            all_extensions.add(ext)

    extensions_list = list(all_extensions)
    suffix_matrix: TransitionMatrix = {}
    for base_state, extensions in suffix_counts.items():
        row: dict[str, float] = {}
        total = 0.0
        for ext in extensions_list:
            count = extensions.get(ext, 0) + ALPHA
            row[ext] = count
            total += count
        for ext in extensions_list:
            row[ext] /= total
        suffix_matrix[base_state] = row

    return TwoTierMarkovModel(base_matrix=base_matrix, suffix_matrix=suffix_matrix)

#  7. Singleton do modelo (carregado uma vez)

_model_cache: Optional[TwoTierMarkovModel] = None

def get_model(data_dir: str | Path = "music_data") -> TwoTierMarkovModel:
    """Retorna o modelo singleton, construindo-o na primeira chamada."""
    global _model_cache
    if _model_cache is None:
        datasets = load_music_data(data_dir)
        _model_cache = build_two_tier_markov_model(datasets)
    return _model_cache

#  8. Avaliação de notas

def _compute_note_score(
    chord_notes: list[int],
    played_notes: list[DetectedNote],
    window_start: float,
    seconds_per_beat: float,
    is_high_density: bool,
    avg_amplitude: float,
) -> float:
    score = 0.0
    root_note = chord_notes[0]
    out_penalty = WEIGHT_OUT_PENALTY * ORNAMENT_PENALTY_MULTIPLIER if is_high_density else WEIGHT_OUT_PENALTY

    for note in played_notes:
        pc = round(note.pitch) % 12
        is_chord_tone = pc in chord_notes
        is_root = pc == root_note

        rel_time = note.start_time - window_start
        beat_phase = (rel_time % seconds_per_beat) / seconds_per_beat if seconds_per_beat > 0 else 0
        is_strong = beat_phase < BEAT_MARGIN_SEC or beat_phase > 1 - BEAT_MARGIN_SEC
        rhythmic_weight = WEIGHT_STRONG_BEAT if is_strong else WEIGHT_WEAK_BEAT

        amp_weight = (note.amplitude or 0.7) / avg_amplitude

        register_mod = 1.0
        if note.pitch < REGISTER_LOW_THRESHOLD and is_root:
            register_mod = REGISTER_BONUS
        if note.pitch > REGISTER_HIGH_THRESHOLD and not is_chord_tone:
            register_mod = REGISTER_PENALTY

        base_val = (WEIGHT_IN_CHORD + (WEIGHT_ROOT_BONUS if is_root else 0)) if is_chord_tone else -out_penalty
        score += base_val * rhythmic_weight * amp_weight * register_mod

    return score

#  9. Probabilidade de estilo

def _lookup_style_probability(
    matrix: TransitionMatrix,
    state1: str,
    state2: Optional[str],
    target: str,
) -> float:
    if state2 and matrix.get(state2, {}).get(target) is not None:
        return matrix[state2][target]
    return matrix.get(state1, {}).get(target, 0.0)

#  10. determineNextChord

def determine_next_chord(
    chord_history: list[str],
    played_notes: list[DetectedNote],
    window_start: float,
    seconds_per_beat: float,
    model: Optional[TwoTierMarkovModel] = None,
) -> ChordResult:
    """
    Elege o próximo acorde dado o histórico e as notas tocadas.
    Equivalente a determineNextChord() do HarmonyEngine.ts.
    """
    if model is None:
        model = get_model()

    current_chord = chord_history[-1] if chord_history else "C"

    if not played_notes:
        return ChordResult(chord=current_chord, velocity=0.7)

    current_base_roman, _ = split_cinematic(chord_to_roman(current_chord, "C"))
    current_base_name = next(
        (c for c in HARMONY_GRAPH if chord_to_roman(c, "C") == current_base_roman), "C"
    )
    current_node = HARMONY_GRAPH.get(current_base_name)
    if not current_node:
        return ChordResult(chord=current_chord, velocity=0.7)

    roman_history = [split_cinematic(chord_to_roman(c, "C"))[0] for c in chord_history]
    curr_roman = roman_history[-1] if roman_history else "I"
    prev_roman = roman_history[-2] if len(roman_history) > 1 else None
    state1 = curr_roman
    state2 = f"{prev_roman},{curr_roman}" if prev_roman else None

    density = len(played_notes)
    is_high_density = density > DENSITY_THRESHOLD

    sum_pitch = sum(n.pitch for n in played_notes)
    sum_amp = sum(n.amplitude or 0.7 for n in played_notes)
    avg_pitch = sum_pitch / density if density > 0 else 0
    avg_amplitude = max(0.2, sum_amp / density) if density > 0 else 0.7

    has_tension_leap = any(
        abs(played_notes[i].pitch - played_notes[i-1].pitch) >= LEAP_THRESHOLD
        for i in range(1, density)
    )

    all_extensions = list(model.suffix_matrix.get(curr_roman, {"none": 1.0}).keys()) or ["none"]

    best_chord = current_chord
    best_score = -math.inf

    for candidate_base in HARMONY_GRAPH:
        candidate_roman = chord_to_roman(candidate_base, "C")
        base_roman, _ = split_cinematic(candidate_roman)

        base_style_prob = _lookup_style_probability(model.base_matrix, state1, state2, base_roman)

        is_allowed_graph = candidate_base in current_node.allowed_transitions
        is_allowed_markov = base_style_prob > THRESHOLD_MODAL_INTERCHANGE
        if not is_allowed_graph and not is_allowed_markov:
            continue

        if has_tension_leap and ("7" in candidate_base or "dim" in candidate_base):
            base_style_prob *= TENSION_BONUS

        is_triad = "7" not in candidate_base and "dim" not in candidate_base
        valid_extensions = all_extensions if is_triad else ["none"]

        for ext in valid_extensions:
            full_chord = candidate_base if ext == "none" else candidate_base + ext
            full_notes = get_chord_pitch_classes(full_chord)

            suffix_prob = model.suffix_matrix.get(base_roman, {}).get(ext, 1.0 / len(all_extensions))
            combined_prob = base_style_prob * suffix_prob

            note_score = _compute_note_score(
                full_notes, played_notes, window_start,
                seconds_per_beat, is_high_density, avg_amplitude
            )

            final_score = note_score + combined_prob * WEIGHT_MARKOV
            if final_score > best_score:
                best_score = final_score
                best_chord = full_chord

    return ChordResult(chord=best_chord, velocity=avg_amplitude)

#  11. generateProgression

def generate_progression(
    played_notes: list[DetectedNote],
    bpm: float,
    time_signature_numerator: int,
    time_signature_denominator: int,
    start_chord: str = "C",
    model: Optional[TwoTierMarkovModel] = None,
) -> list[ChordResult]:
    """
    Gera uma progressão completa para as notas tocadas.
    Equivalente a generateProgression() do HarmonyEngine.ts.
    """
    if model is None:
        model = get_model()

    if not played_notes:
        return [ChordResult(chord=start_chord, velocity=0.7)]

    seconds_per_beat = (4 / time_signature_denominator) * (60 / bpm)
    seconds_per_measure = seconds_per_beat * time_signature_numerator

    total_start = min(n.start_time for n in played_notes)
    total_end = max(n.end_time for n in played_notes)
    total_duration = total_end - total_start

    window_count = max(1, math.ceil(total_duration / seconds_per_measure))

    progression: list[ChordResult] = []
    history = [start_chord]

    for w in range(window_count):
        window_start = total_start + w * seconds_per_measure
        window_end = window_start + seconds_per_measure

        window_notes = [n for n in played_notes if n.start_time < window_end and n.end_time > window_start]
        clipped: list[DetectedNote] = [
            DetectedNote(
                pitch=n.pitch,
                start_time=max(n.start_time, window_start),
                end_time=min(n.end_time, window_end),
                amplitude=n.amplitude,
            )
            for n in window_notes
        ]

        result = determine_next_chord(history, clipped, window_start, seconds_per_beat, model)
        progression.append(result)
        history.append(result.chord)

    return progression
