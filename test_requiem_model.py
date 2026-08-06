"""
test_requiem_model.py
=====================
Testes unitários que verificam paridade com o HarmonyEngine.ts.
Execute com: python -m pytest test_requiem_model.py -v
"""

import pytest
from tonality_adapter import (
    DetectedNote,
    chord_to_roman,
    roman_to_chord,
    get_chord_pitch_classes,
    normalize_notes,
    transpose_chord,
    normalize_progression_to_c,
    get_tonality_offset,
)
from audio_analyzer import detect_key, estimate_bpm
from harmony_engine import (
    load_music_data,
    build_two_tier_markov_model,
    generate_progression,
    determine_next_chord,
    HARMONY_GRAPH,
)
from requiem_model import RequiemModel

DATA_DIR = "music_data"


#  TonalityAdapter tests

class TestTonalityOffsets:
    def test_c_major_zero(self):
        assert get_tonality_offset("C") == 0

    def test_d_major(self):
        assert get_tonality_offset("D") == 2

    def test_am_zero(self):
        assert get_tonality_offset("Am") == 0

    def test_em_offset(self):
        assert get_tonality_offset("Em") == 7

    def test_fm_offset(self):
        assert get_tonality_offset("F#m") == 9


class TestChordToRoman:
    def test_c_is_I(self):
        assert chord_to_roman("C", "C") == "I"

    def test_am_is_vi(self):
        assert chord_to_roman("Am", "C") == "vi"

    def test_g7_is_V7(self):
        assert chord_to_roman("G7", "C") == "V7"

    def test_dm_in_c(self):
        assert chord_to_roman("Dm", "C") == "ii"

    def test_fm_in_c(self):
        assert chord_to_roman("Fm", "C") == "iv"

    def test_d_in_d(self):
        assert chord_to_roman("D", "D") == "I"

    def test_bm_in_d(self):
        assert chord_to_roman("Bm", "D") == "vi"


class TestRomanToChord:
    def test_I_in_D(self):
        assert roman_to_chord("I", "D") == "D"

    def test_vi_in_C(self):
        assert roman_to_chord("vi", "C") == "Am"

    def test_V7_in_C(self):
        assert roman_to_chord("V7", "C") == "G7"

    def test_ii_in_C(self):
        assert roman_to_chord("ii", "C") == "Dm"


class TestGetChordPitchClasses:
    def test_c_major(self):
        assert get_chord_pitch_classes("C") == [0, 4, 7]

    def test_am(self):
        assert get_chord_pitch_classes("Am") == [9, 0, 4]

    def test_g7(self):
        assert get_chord_pitch_classes("G7") == [7, 11, 2, 5]

    def test_bdim(self):
        assert get_chord_pitch_classes("Bdim") == [11, 2, 5]


class TestTransposeChord:
    def test_noop_c(self):
        assert transpose_chord("C", "C") == "C"

    def test_dm7_to_d(self):
        # Dm7 em C + offset 2 → Em7
        assert transpose_chord("Dm7", "D") == "Em7"

    def test_g7_to_f(self):
        # G7 em C + offset 5 → C7
        assert transpose_chord("G7", "F") == "C7"

    def test_am_to_g(self):
        # Am em C + offset 7 → Em
        assert transpose_chord("Am", "G") == "Em"


class TestNormalizeNotes:
    def test_no_offset(self):
        notes = [DetectedNote(60, 0, 1), DetectedNote(64, 1, 2)]
        result = normalize_notes(notes, "C")
        assert [n.pitch for n in result] == [60, 64]

    def test_d_major_offset_2(self):
        notes = [DetectedNote(62, 0, 1)]  # D
        result = normalize_notes(notes, "D")
        assert result[0].pitch == 60  # D - 2 = C


#  AudioAnalyzer tests

class TestDetectKey:
    def test_few_notes_returns_c(self):
        assert detect_key([]) == "C"
        assert detect_key([DetectedNote(60, 0, 1)]) == "C"

    def test_c_major_scale(self):
        # Escala de Dó com durações iguais — deve retornar tonalidade próxima de C
        notes = [
            DetectedNote(60+i, i, i+1) for i in [0, 2, 4, 5, 7, 9, 11]
        ]
        key = detect_key(notes)
        assert key in {"C", "Am"}  # ambas são candidatas fortes


class TestEstimateBPM:
    def test_few_notes_returns_120(self):
        assert estimate_bpm([]) == 120
        assert estimate_bpm([DetectedNote(60, 0, 1)]) == 120

    def test_regular_120_bpm(self):
        # Notas a cada 0.5s = 120 BPM (semínima)
        notes = [DetectedNote(60, i * 0.5, i * 0.5 + 0.1) for i in range(10)]
        bpm = estimate_bpm(notes)
        assert 100 <= bpm <= 140


#  HarmonyEngine tests

class TestLoadMusicData:
    def test_loads_datasets(self):
        datasets = load_music_data(DATA_DIR)
        assert len(datasets) > 0

    def test_dataset_fields(self):
        datasets = load_music_data(DATA_DIR)
        for d in datasets:
            assert isinstance(d.title, str)
            assert isinstance(d.original_tonality, str)
            assert isinstance(d.normalized_progression, list)
            assert len(d.normalized_progression) > 0


class TestBuildMarkovModel:
    def test_model_has_matrices(self):
        datasets = load_music_data(DATA_DIR)
        model = build_two_tier_markov_model(datasets)
        assert model.base_matrix
        assert model.suffix_matrix

    def test_probabilities_sum_to_one(self):
        datasets = load_music_data(DATA_DIR)
        model = build_two_tier_markov_model(datasets)
        for row in model.base_matrix.values():
            total = sum(row.values())
            assert abs(total - 1.0) < 1e-6, f"Soma inválida: {total}"


class TestDetermineNextChord:
    def setup_method(self):
        datasets = load_music_data(DATA_DIR)
        self.model = build_two_tier_markov_model(datasets)

    def test_no_notes_returns_current(self):
        result = determine_next_chord(["C"], [], 0.0, 0.5, self.model)
        assert result.chord == "C"

    def test_returns_valid_chord(self):
        notes = [
            DetectedNote(60, 0.0, 0.5, 0.8),
            DetectedNote(64, 0.2, 0.7, 0.7),
            DetectedNote(67, 0.4, 1.0, 0.9),
        ]
        result = determine_next_chord(["C"], notes, 0.0, 0.5, self.model)
        assert isinstance(result.chord, str)
        assert 0.0 <= result.velocity <= 1.5

    def test_returns_allowed_transition(self):
        notes = [DetectedNote(60, 0.0, 1.0, 0.8), DetectedNote(67, 0.2, 1.0, 0.7)]
        result = determine_next_chord(["C", "G"], notes, 0.0, 0.5, self.model)
        assert isinstance(result.chord, str)


class TestGenerateProgression:
    def setup_method(self):
        datasets = load_music_data(DATA_DIR)
        self.model = build_two_tier_markov_model(datasets)

    def test_empty_notes(self):
        result = generate_progression([], 120, 4, 4, "C", self.model)
        assert len(result) == 1
        assert result[0].chord == "C"

    def test_generates_chords(self):
        # 2 compassos de notas a 120 BPM (4/4) = 4s por compasso
        notes = [
            DetectedNote(60, i * 0.5, i * 0.5 + 0.4, 0.8)
            for i in range(16)  # 8s de notas
        ]
        result = generate_progression(notes, 120, 4, 4, "C", self.model)
        assert len(result) >= 1
        for r in result:
            assert isinstance(r.chord, str)

    def test_waltz_time_signature(self):
        notes = [DetectedNote(60, i * 0.5, i * 0.5 + 0.4) for i in range(12)]
        result = generate_progression(notes, 120, 3, 4, "C", self.model)
        assert len(result) >= 1


#  RequiemModel (high-level API) tests

class TestRequiemModel:
    def setup_method(self):
        self.model = RequiemModel(data_dir=DATA_DIR)

    def test_init_loads_model(self):
        assert self.model.model is not None

    def test_run_empty_returns_start(self):
        result = self.model.run([], auto_detect=False)
        assert len(result) == 1
        assert result[0].chord == "C"

    def test_run_with_notes(self):
        notes = [DetectedNote(60 + i, i * 0.5, i * 0.5 + 0.4, 0.8) for i in range(8)]
        result = self.model.run(notes, bpm=120, tonality="C", auto_detect=False)
        assert len(result) >= 1
        for r in result:
            assert isinstance(r.chord, str)

    def test_detect_key(self):
        notes = [DetectedNote(60, i, i+1) for i in range(4)]
        key = RequiemModel.detect_key(notes)
        assert isinstance(key, str)

    def test_estimate_bpm(self):
        notes = [DetectedNote(60, i * 0.5, i * 0.5 + 0.1) for i in range(10)]
        bpm = RequiemModel.estimate_bpm(notes)
        assert 60 <= bpm <= 180

    def test_chord_pitch_classes(self):
        pcs = RequiemModel.chord_pitch_classes("C")
        assert pcs == [0, 4, 7]

    def test_harmony_graph_not_empty(self):
        assert len(self.model.harmony_graph) > 0

    def test_auto_detect_run(self):
        # Simula escala de Dó com durações
        notes = [DetectedNote(60 + i, i * 0.5, i * 0.5 + 0.4) for i in [0, 2, 4, 5, 7, 9, 11, 12]]
        result = self.model.run(notes, auto_detect=True)
        assert len(result) >= 1
