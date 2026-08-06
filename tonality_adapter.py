"""
tonality_adapter.py
===================
Porta Python fiel ao TonalityAdapter.ts do Requiem App.

Responsabilidades:
  1. Normalizar notas de entrada (microfone) → Dó Maior
  2. Transpor acordes de saída (Dó Maior) → tonalidade alvo
  3. Normalizar progressões do dataset para treinar Markov
  4. Converter acorde ↔ numeral romano
  5. Obter pitch-classes reais de qualquer acorde
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

#  1. Tipos de dados

@dataclass
class DetectedNote:
    """Representa uma nota detectada pelo microfone."""
    pitch: float          # MIDI pitch (ex: 60 = C4)
    start_time: float     # Tempo de início em segundos
    end_time: float       # Tempo de fim em segundos
    amplitude: float = 0.7

#  2. Offsets de tonalidade (semitons a partir de Dó)

ROOT_OFFSETS: dict[str, int] = {
    "C": 0,  "C#": 1,  "Db": 1,
    "D": 2,  "D#": 3,  "Eb": 3,
    "E": 4,  "Fb": 4,
    "F": 5,  "F#": 6,  "Gb": 6,
    "G": 7,  "G#": 8,  "Ab": 8,
    "A": 9,  "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

# Maiores usam offset direto; Menores usam offset da relativa maior
TONALITY_OFFSETS: dict[str, int] = {
    # Maiores
    "C": 0,   "C#": 1,  "Db": 1,
    "D": 2,   "D#": 3,  "Eb": 3,
    "E": 4,
    "F": 5,   "F#": 6,  "Gb": 6,
    "G": 7,   "G#": 8,  "Ab": 8,
    "A": 9,   "A#": 10, "Bb": 10,
    "B": 11,  "Cb": 11,
    # Menores (offset = relativa maior)
    "Am": 0,   "A#m": 1,  "Bbm": 1,
    "Bm": 2,   "Cm": 3,
    "C#m": 4,  "Dm": 5,
    "D#m": 6,  "Ebm": 6,  "Em": 7,
    "Fm": 8,   "F#m": 9,
    "Gm": 10,  "G#m": 11, "Abm": 11,
}

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_FLAT  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

FLAT_TONALITIES: set[str] = {
    "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb",
    "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm", "Abm",
}

CHORD_REGEX = re.compile(r"^([A-G][#b]?)(.*)$")

# Intervalos → Numeral Romano
INTERVAL_TO_ROMAN = ["I", "bII", "II", "bIII", "III", "IV", "bV", "V", "bVI", "VI", "bVII", "VII"]

# Dicionário de sufixos → intervalos em semitons
SUFFIX_DICT: dict[str, list[int]] = {
    "":        [0, 4, 7],
    "m":       [0, 3, 7],
    "dim":     [0, 3, 6],
    "aug":     [0, 4, 8],
    "+":       [0, 4, 8],
    "sus4":    [0, 5, 7],
    "sus2":    [0, 2, 7],
    "5":       [0, 7],
    "maj7":    [0, 4, 7, 11],
    "7":       [0, 4, 7, 10],
    "m7":      [0, 3, 7, 10],
    "m7b5":    [0, 3, 6, 10],
    "dim7":    [0, 3, 6, 9],
    "add9":    [0, 4, 7, 2],
    "madd9":   [0, 3, 7, 2],
    "sus4add9":[0, 5, 7, 2],
    "11":      [0, 4, 7, 10, 2, 5],
    "m11":     [0, 3, 7, 10, 2, 5],
}

# Tonalidades com opções de UI (mesmo catálogo do TonalityAdapter.ts)
MAJOR_TONALITIES = [
    {"value": "C",   "label": "Dó Maior (C)"},
    {"value": "C#",  "label": "Dó# Maior (C#)"},
    {"value": "Db",  "label": "Réb Maior (Db)"},
    {"value": "D",   "label": "Ré Maior (D)"},
    {"value": "Eb",  "label": "Mib Maior (Eb)"},
    {"value": "E",   "label": "Mi Maior (E)"},
    {"value": "F",   "label": "Fá Maior (F)"},
    {"value": "F#",  "label": "Fá# Maior (F#)"},
    {"value": "Gb",  "label": "Solb Maior (Gb)"},
    {"value": "G",   "label": "Sol Maior (G)"},
    {"value": "Ab",  "label": "Láb Maior (Ab)"},
    {"value": "A",   "label": "Lá Maior (A)"},
    {"value": "Bb",  "label": "Sib Maior (Bb)"},
    {"value": "B",   "label": "Si Maior (B)"},
]

MINOR_TONALITIES = [
    {"value": "Am",   "label": "Lá Menor (Am)"},
    {"value": "A#m",  "label": "Lá# Menor (A#m)"},
    {"value": "Bbm",  "label": "Sib Menor (Bbm)"},
    {"value": "Bm",   "label": "Si Menor (Bm)"},
    {"value": "Cm",   "label": "Dó Menor (Cm)"},
    {"value": "C#m",  "label": "Dó# Menor (C#m)"},
    {"value": "Dm",   "label": "Ré Menor (Dm)"},
    {"value": "D#m",  "label": "Ré# Menor (D#m)"},
    {"value": "Ebm",  "label": "Mib Menor (Ebm)"},
    {"value": "Em",   "label": "Mi Menor (Em)"},
    {"value": "Fm",   "label": "Fá Menor (Fm)"},
    {"value": "F#m",  "label": "Fá# Menor (F#m)"},
    {"value": "Gm",   "label": "Sol Menor (Gm)"},
    {"value": "G#m",  "label": "Sol# Menor (G#m)"},
    {"value": "Abm",  "label": "Láb Menor (Abm)"},
]

ALL_TONALITIES = MAJOR_TONALITIES + MINOR_TONALITIES

UT_OPTIONS = [2, 4, 8]
QT_OPTIONS = [2, 3, 4, 6, 9, 12]

#  3. Funções auxiliares

def _parse_chord(chord_name: str) -> Optional[tuple[str, str]]:
    """Separa a tônica (root) do sufixo de um nome de acorde."""
    m = CHORD_REGEX.match(chord_name)
    if not m:
        return None
    return m.group(1), m.group(2)


def _note_names_for(tonality: str) -> list[str]:
    return NOTE_NAMES_FLAT if tonality in FLAT_TONALITIES else NOTE_NAMES_SHARP


def get_tonality_offset(tonality: str) -> int:
    """Retorna o offset em semitons da tonalidade (fallback Dó Maior = 0)."""
    return TONALITY_OFFSETS.get(tonality, 0)


#  4. Transposição de notas (World → Local / Dó Maior)

def normalize_notes(notes: list[DetectedNote], target_tonality: str) -> list[DetectedNote]:
    """
    Normaliza notas detectadas transpondo-as para Dó Maior.
    Equivalente a normalizeNotes() do TonalityAdapter.ts.
    """
    offset = get_tonality_offset(target_tonality)
    if offset == 0:
        return [DetectedNote(n.pitch, n.start_time, n.end_time, n.amplitude) for n in notes]
    return [
        DetectedNote(n.pitch - offset, n.start_time, n.end_time, n.amplitude)
        for n in notes
    ]


#  5. Transposição de acordes (Local → World)

def transpose_chord(chord_name: str, target_tonality: str) -> str:
    """
    Transpõe um acorde de Dó Maior para a tonalidade alvo.
    Equivalente a transposeChord() do TonalityAdapter.ts.
    """
    parsed = _parse_chord(chord_name)
    if not parsed:
        return chord_name
    root, suffix = parsed

    offset = get_tonality_offset(target_tonality)
    if offset == 0:
        return chord_name

    root_index = ROOT_OFFSETS.get(root)
    if root_index is None:
        return chord_name

    new_index = (root_index + offset) % 12
    note_names = _note_names_for(target_tonality)
    return note_names[new_index] + suffix


def transpose_progression(progression: list[str], target_tonality: str) -> list[str]:
    """Transpõe toda uma progressão para a tonalidade alvo."""
    return [transpose_chord(c, target_tonality) for c in progression]


#  6. Normalização de dataset (World → Local)

def normalize_progression_to_c(progression: list[str], original_tonality: str) -> list[str]:
    """
    Normaliza progressão da tonalidade original para Dó Maior.
    Equivalente a normalizeProgressionToC() do TonalityAdapter.ts.
    """
    offset = get_tonality_offset(original_tonality)
    if offset == 0:
        return list(progression)

    result = []
    for chord_name in progression:
        parsed = _parse_chord(chord_name)
        if not parsed:
            result.append(chord_name)
            continue
        root, suffix = parsed
        root_index = ROOT_OFFSETS.get(root)
        if root_index is None:
            result.append(chord_name)
            continue
        new_index = ((root_index - offset) % 12 + 12) % 12
        result.append(NOTE_NAMES_SHARP[new_index] + suffix)
    return result


#  7. Conversão acorde ↔ numeral romano

def chord_to_roman(chord: str, tonality: str) -> str:
    """
    Converte acorde absoluto → numeral romano na tonalidade.
    Equivalente a chordToRoman() do TonalityAdapter.ts.
    """
    parsed = _parse_chord(chord)
    if not parsed:
        return chord
    root, suffix = parsed

    root_offset = ROOT_OFFSETS.get(root)
    if root_offset is None:
        return chord

    tonality_offset = get_tonality_offset(tonality)
    interval = (root_offset - tonality_offset + 12) % 12
    roman = INTERVAL_TO_ROMAN[interval]

    is_minor = suffix.startswith("m") and not suffix.startswith("maj")
    is_dim = suffix.startswith("dim")
    if is_minor or is_dim:
        roman = roman.lower()

    # Remover "m" redundante (igual ao TS)
    if suffix == "m":
        suffix = ""
    elif suffix.startswith("m") and not suffix.startswith("maj"):
        suffix = suffix[1:]

    return roman + suffix


ROMAN_REGEX = re.compile(
    r"^([b#]?(?:III|iii|II|ii|IV|iv|VIII|viii|VII|vii|VI|vi|V|v|I|i))((?:.*))$"
)

def roman_to_chord(roman: str, tonality: str) -> str:
    """
    Converte numeral romano → acorde absoluto na tonalidade.
    Equivalente a romanToChord() do TonalityAdapter.ts.
    """
    m = ROMAN_REGEX.match(roman)
    if not m:
        return roman

    base_roman = m.group(1)
    suffix = m.group(2)

    search_base = base_roman.upper()
    if search_base.startswith("B"):
        search_base = "b" + search_base[1:]
    if search_base.startswith("#"):
        search_base = "#" + search_base[1:]

    try:
        interval = INTERVAL_TO_ROMAN.index(search_base)
    except ValueError:
        return roman

    is_lower = base_roman == base_roman.lower()
    if is_lower and not suffix.startswith("dim"):
        suffix = "m" + suffix

    tonality_offset = get_tonality_offset(tonality)
    root_index = (tonality_offset + interval) % 12
    note_names = _note_names_for(tonality)
    return note_names[root_index] + suffix


#  8. Pitch-classes de um acorde

def get_chord_pitch_classes(chord_name: str) -> list[int]:
    """
    Retorna os pitch-classes (0-11) de qualquer acorde válido.
    Equivalente a getChordPitchClasses() do TonalityAdapter.ts.
    """
    parsed = _parse_chord(chord_name)
    if not parsed:
        return [0, 4, 7]
    root, suffix = parsed

    root_index = ROOT_OFFSETS.get(root, 0)

    intervals = SUFFIX_DICT.get(suffix)
    if intervals is None:
        if suffix.startswith("m") and not suffix.startswith("maj"):
            intervals = [0, 3, 7]
        elif suffix.startswith("sus"):
            intervals = [0, 5, 7]
        else:
            intervals = [0, 4, 7]

    return [(root_index + i) % 12 for i in intervals]
