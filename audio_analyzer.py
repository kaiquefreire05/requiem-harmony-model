"""
audio_analyzer.py
=================
Porta Python fiel ao AudioAnalyzer.ts do Requiem App.

Funções:
  - detect_key()    — Krumhansl-Schmuckler: detecta tonalidade das notas
  - estimate_bpm()  — Estima BPM a partir dos onsets
"""

from __future__ import annotations
import math
from tonality_adapter import DetectedNote

# Perfis de Krumhansl-Schmuckler
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

PITCH_CLASS_NAMES_MAJOR = ["C","Db","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
PITCH_CLASS_NAMES_MINOR = ["C","C#","D","D#","E","F","F#","G","G#","A","Bb","B"]


def detect_key(notes: list[DetectedNote]) -> str:
    """
    Detecta a tonalidade predominante por correlação de Pearson.
    Equivalente a detectKey() do AudioAnalyzer.ts.
    """
    if not notes or len(notes) < 3:
        return "C"

    durations = [0.0] * 12
    for note in notes:
        if note.pitch < 0:
            continue
        dur = note.end_time - note.start_time
        pc = int(note.pitch) % 12
        durations[pc] += dur

    total = sum(durations)
    if total == 0:
        return "C"

    normalized = [d / total for d in durations]

    def pearson(inp: list[float], profile: list[float]) -> float:
        avg_inp = sum(inp) / 12
        avg_prof = sum(profile) / 12
        num = sum((inp[i] - avg_inp) * (profile[i] - avg_prof) for i in range(12))
        den1 = sum((inp[i] - avg_inp) ** 2 for i in range(12))
        den2 = sum((profile[i] - avg_prof) ** 2 for i in range(12))
        denom = math.sqrt(den1 * den2)
        return num / denom if denom != 0 else 0.0

    best_key = "C"
    max_corr = -math.inf

    for shift in range(12):
        shifted = normalized[shift:] + normalized[:shift]

        corr_major = pearson(shifted, MAJOR_PROFILE)
        if corr_major > max_corr:
            max_corr = corr_major
            best_key = PITCH_CLASS_NAMES_MAJOR[shift]

        corr_minor = pearson(shifted, MINOR_PROFILE)
        if corr_minor > max_corr:
            max_corr = corr_minor
            best_key = f"{PITCH_CLASS_NAMES_MINOR[shift]}m"

    return best_key


def estimate_bpm(notes: list[DetectedNote]) -> int:
    """
    Estima o BPM a partir dos deltas de onset.
    Equivalente a estimateBPM() do AudioAnalyzer.ts.
    """
    if not notes or len(notes) < 2:
        return 120

    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    deltas: list[float] = []
    for i in range(1, len(sorted_notes)):
        delta = sorted_notes[i].start_time - sorted_notes[i - 1].start_time
        if 0.1 < delta < 2.0:
            deltas.append(delta)

    if not deltas:
        return 120

    deltas.sort()
    if len(deltas) > 5:
        trim = max(1, int(len(deltas) * 0.1))
        deltas = deltas[trim: len(deltas) - trim]

    avg_delta = sum(deltas) / len(deltas)
    bpm = round(60 / avg_delta)

    while bpm < 60:
        bpm *= 2
    while bpm > 180:
        bpm = bpm // 2

    return max(60, min(180, bpm))
