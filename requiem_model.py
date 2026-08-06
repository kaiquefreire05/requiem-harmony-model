"""
requiem_model.py
================
Ponto de entrada principal do modelo Requiem.

Expõe a interface pública de alto nível que substitui
identicamente o motor JS do Requiem App.

Uso básico:
    from requiem_model import RequiemModel, DetectedNote

    model = RequiemModel()                 # Carrega e treina automaticamente
    result = model.run(notes, bpm=120)     # Gera progressão
    print(result)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from tonality_adapter import (
    DetectedNote,
    normalize_notes,
    transpose_chord,
    transpose_progression,
    normalize_progression_to_c,
    chord_to_roman,
    roman_to_chord,
    get_chord_pitch_classes,
    ALL_TONALITIES,
    MAJOR_TONALITIES,
    MINOR_TONALITIES,
    UT_OPTIONS,
    QT_OPTIONS,
)
from audio_analyzer import detect_key, estimate_bpm
from harmony_engine import (
    HARMONY_GRAPH,
    ChordResult,
    TwoTierMarkovModel,
    load_music_data,
    build_two_tier_markov_model,
    generate_progression,
    determine_next_chord,
    get_model,
)


class RequiemModel:
    """
    Modelo Requiem — substituto drop-in do HarmonyEngine.ts.

    Parâmetros
    ----------
    data_dir : str | Path
        Diretório com os JSONs de treinamento (padrão: music_data/).
    tonality : str
        Tonalidade de trabalho (padrão: "C"). Ex: "D", "Am", "F#".
    time_signature : tuple[int, int]
        Fórmula de compasso (numerador, denominador). Ex: (4, 4).
    bpm : int
        BPM padrão (pode ser sobrescrito por estimate_bpm).
    """

    def __init__(
        self,
        data_dir: str | Path = "music_data",
        tonality: str = "C",
        time_signature: tuple[int, int] = (4, 4),
        bpm: int = 120,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.tonality = tonality
        self.time_signature = time_signature
        self.bpm = bpm
        self._model: Optional[TwoTierMarkovModel] = None
        self._load()

    # Treinamento

    def _load(self) -> None:
        """Carrega os JSONs e treina o modelo Two-Tier de Markov."""
        datasets = load_music_data(self.data_dir)
        if not datasets:
            raise FileNotFoundError(
                f"Nenhum JSON de treinamento encontrado em: {self.data_dir}"
            )
        self._model = build_two_tier_markov_model(datasets)
        print(f"[RequiemModel] Modelo treinado com {len(datasets)} progressões.")

    # Interface principal

    def run(
        self,
        notes: list[DetectedNote],
        bpm: Optional[int] = None,
        tonality: Optional[str] = None,
        time_signature: Optional[tuple[int, int]] = None,
        start_chord: str = "C",
        auto_detect: bool = True,
    ) -> list[ChordResult]:
        """
        Gera uma progressão harmônica a partir das notas tocadas.

        Parâmetros
        ----------
        notes : list[DetectedNote]
            Notas captadas pelo microfone.
        bpm : int, opcional
            BPM (se None e auto_detect=True, estima automaticamente).
        tonality : str, opcional
            Tonalidade (se None e auto_detect=True, detecta automaticamente).
        time_signature : (int, int), opcional
            Fórmula de compasso.
        start_chord : str
            Acorde inicial.
        auto_detect : bool
            Se True, detecta tonalidade e BPM automaticamente.

        Retorna
        -------
        list[ChordResult]
            Lista de acordes com velocity para cada compasso.
        """
        # 1. Detectar/usar tonalidade e BPM
        effective_tonality = tonality or self.tonality
        effective_bpm = bpm or self.bpm
        effective_ts = time_signature or self.time_signature

        if auto_detect and notes:
            if tonality is None:
                effective_tonality = detect_key(notes)
            if bpm is None:
                effective_bpm = estimate_bpm(notes)

        # 2. Normalizar notas para Dó Maior (Local Space)
        local_notes = normalize_notes(notes, effective_tonality)

        # 3. Gerar progressão em Dó Maior
        raw_progression = generate_progression(
            played_notes=local_notes,
            bpm=effective_bpm,
            time_signature_numerator=effective_ts[0],
            time_signature_denominator=effective_ts[1],
            start_chord=start_chord,
            model=self._model,
        )

        # 4. Transpor para a tonalidade real do usuário
        transposed = [
            ChordResult(
                chord=transpose_chord(r.chord, effective_tonality),
                velocity=r.velocity,
            )
            for r in raw_progression
        ]

        return transposed

    def next_chord(
        self,
        history: list[str],
        notes: list[DetectedNote],
        window_start: float,
        seconds_per_beat: float,
        tonality: str = "C",
    ) -> ChordResult:
        """
        Determina o próximo acorde dado histórico e notas atuais.
        Equivalente direto a determineNextChord() do HarmonyEngine.ts.
        """
        local_notes = normalize_notes(notes, tonality)
        result = determine_next_chord(history, local_notes, window_start, seconds_per_beat, self._model)
        return ChordResult(
            chord=transpose_chord(result.chord, tonality),
            velocity=result.velocity,
        )

    # Utilitários exportados

    @staticmethod
    def detect_key(notes: list[DetectedNote]) -> str:
        """Detecta tonalidade predominante (Krumhansl-Schmuckler)."""
        return detect_key(notes)

    @staticmethod
    def estimate_bpm(notes: list[DetectedNote]) -> int:
        """Estima BPM a partir dos onsets."""
        return estimate_bpm(notes)

    @staticmethod
    def chord_pitch_classes(chord: str) -> list[int]:
        """Retorna os pitch-classes (0-11) de um acorde."""
        return get_chord_pitch_classes(chord)

    @staticmethod
    def chord_to_roman(chord: str, tonality: str) -> str:
        return chord_to_roman(chord, tonality)

    @staticmethod
    def roman_to_chord(roman: str, tonality: str) -> str:
        return roman_to_chord(roman, tonality)

    @property
    def harmony_graph(self) -> dict:
        return HARMONY_GRAPH

    @property
    def model(self) -> Optional[TwoTierMarkovModel]:
        return self._model


#  Re-exportações para facilitar importação direta
__all__ = [
    "RequiemModel",
    "DetectedNote",
    "ChordResult",
    "TwoTierMarkovModel",
    "detect_key",
    "estimate_bpm",
    "generate_progression",
    "determine_next_chord",
    "chord_to_roman",
    "roman_to_chord",
    "get_chord_pitch_classes",
    "normalize_notes",
    "transpose_chord",
    "transpose_progression",
    "ALL_TONALITIES",
    "MAJOR_TONALITIES",
    "MINOR_TONALITIES",
    "HARMONY_GRAPH",
]
