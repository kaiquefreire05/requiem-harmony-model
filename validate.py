"""
validate.py
===========
Script de validação rápida sem dependência de pytest.
Execute com: python3 validate.py
"""

import sys
import traceback

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

results = []

def check(name: str, condition: bool, detail: str = ""):
    ok = bool(condition)
    icon = PASS if ok else FAIL
    msg = f"  {icon} {name}"
    if not ok and detail:
        msg += f"\n      → {detail}"
    print(msg)
    results.append(ok)

print("\n=== Requiem Harmony Model — Validação ===\n")

# ─ Imports ─
try:
    from tonality_adapter import (
        DetectedNote, chord_to_roman, roman_to_chord,
        get_chord_pitch_classes, normalize_notes, transpose_chord,
        get_tonality_offset,
    )
    from audio_analyzer import detect_key, estimate_bpm
    from harmony_engine import (
        load_music_data, build_two_tier_markov_model,
        generate_progression, determine_next_chord, HARMONY_GRAPH,
    )
    from requiem_model import RequiemModel
    print("[1] TonalityAdapter")
except Exception as e:
    print(f"  {FAIL} Falha no import: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────
print("\n[1] TonalityAdapter")
check("C major offset = 0",  get_tonality_offset("C") == 0)
check("D major offset = 2",  get_tonality_offset("D") == 2)
check("Am offset = 0",       get_tonality_offset("Am") == 0)
check("Em offset = 7",       get_tonality_offset("Em") == 7)
check("F#m offset = 9",      get_tonality_offset("F#m") == 9)

check("C → I (C major)",     chord_to_roman("C",  "C") == "I")
check("Am → vi (C major)",   chord_to_roman("Am", "C") == "vi")
check("G7 → V7 (C major)",   chord_to_roman("G7", "C") == "V7")
check("Dm → ii (C major)",   chord_to_roman("Dm", "C") == "ii")
check("D → I (D major)",     chord_to_roman("D",  "D") == "I")

check("I in D → D",          roman_to_chord("I",   "D") == "D")
check("vi in C → Am",        roman_to_chord("vi",  "C") == "Am")
check("V7 in C → G7",        roman_to_chord("V7",  "C") == "G7")
check("ii in C → Dm",        roman_to_chord("ii",  "C") == "Dm")

check("C pitch classes",     get_chord_pitch_classes("C") == [0, 4, 7])
check("Am pitch classes",    get_chord_pitch_classes("Am") == [9, 0, 4])
check("G7 pitch classes",    get_chord_pitch_classes("G7") == [7, 11, 2, 5])
check("Bdim pitch classes",  get_chord_pitch_classes("Bdim") == [11, 2, 5])

check("transpose noop (C→C)",  transpose_chord("C", "C") == "C")
check("transpose Am→G = Em",   transpose_chord("Am", "G") == "Em")
check("transpose G7→F = C7",   transpose_chord("G7", "F") == "C7")
check("transpose Dm7→D = Em7", transpose_chord("Dm7", "D") == "Em7")

notes_d = [DetectedNote(62, 0, 1)]
norm = normalize_notes(notes_d, "D")
check("normalize D offset: pitch 62→60", norm[0].pitch == 60)

# ─────────────────────────────────────────────────────────
print("\n[2] AudioAnalyzer")
check("detect_key([]) = C", detect_key([]) == "C")
check("estimate_bpm([]) = 120", estimate_bpm([]) == 120)

scale_notes = [DetectedNote(60+i, i, i+1) for i in [0,2,4,5,7,9,11]]
key = detect_key(scale_notes)
check(f"detect_key(C scale) ∈ {{C, Am}}: got {key}", key in {"C", "Am"})

regular = [DetectedNote(60, i*0.5, i*0.5+0.1) for i in range(10)]
bpm = estimate_bpm(regular)
check(f"estimate_bpm(120bpm notes) ≈ 120: got {bpm}", 100 <= bpm <= 140)

# ─────────────────────────────────────────────────────────
print("\n[3] HarmonyEngine — Dados")
try:
    datasets = load_music_data("music_data")
    check(f"Carregou {len(datasets)} progressões", len(datasets) > 40)
    check("Todos têm originalTonality", all(d.original_tonality for d in datasets))
    check("Todos têm normalizedProgression", all(d.normalized_progression for d in datasets))
except Exception as e:
    check("load_music_data() sem erro", False, str(e))

# ─────────────────────────────────────────────────────────
print("\n[4] HarmonyEngine — Modelo Markov")
try:
    model = build_two_tier_markov_model(datasets)
    check("base_matrix não vazia", bool(model.base_matrix))
    check("suffix_matrix não vazia", bool(model.suffix_matrix))

    # Verifica soma de probabilidades
    bad = [(s, sum(r.values())) for s, r in model.base_matrix.items() if abs(sum(r.values()) - 1.0) > 1e-5]
    check("Probabilidades base somam 1.0", len(bad) == 0,
          f"Falhou em: {bad[:3]}" if bad else "")
except Exception as e:
    check("build_two_tier_markov_model() sem erro", False, str(e))

# ─────────────────────────────────────────────────────────
print("\n[5] HarmonyEngine — determineNextChord")
try:
    result_empty = determine_next_chord(["C"], [], 0.0, 0.5, model)
    check("Sem notas → mantém acorde atual", result_empty.chord == "C")

    notes3 = [
        DetectedNote(60, 0.0, 0.5, 0.8),
        DetectedNote(64, 0.2, 0.7, 0.7),
        DetectedNote(67, 0.4, 1.0, 0.9),
    ]
    res = determine_next_chord(["C"], notes3, 0.0, 0.5, model)
    check(f"Retorna acorde válido: {res.chord}", isinstance(res.chord, str) and len(res.chord) > 0)
    check(f"Velocity razoável: {res.velocity:.2f}", 0 <= res.velocity <= 2.0)
except Exception as e:
    check("determine_next_chord() sem erro", False, str(e))
    traceback.print_exc()

# ─────────────────────────────────────────────────────────
print("\n[6] HarmonyEngine — generateProgression")
try:
    empty_prog = generate_progression([], 120, 4, 4, "C", model)
    check("Vazia → 1 acorde", len(empty_prog) == 1 and empty_prog[0].chord == "C")

    long_notes = [DetectedNote(60+i%12, i*0.5, i*0.5+0.4, 0.8) for i in range(16)]
    prog = generate_progression(long_notes, 120, 4, 4, "C", model)
    check(f"8s de notas → {len(prog)} compassos", len(prog) >= 1)
    check("Todos acordes são strings", all(isinstance(r.chord, str) for r in prog))

    waltz = generate_progression(long_notes, 120, 3, 4, "C", model)
    check(f"Compasso 3/4 → {len(waltz)} compassos", len(waltz) >= 1)
except Exception as e:
    check("generate_progression() sem erro", False, str(e))
    traceback.print_exc()

# ─────────────────────────────────────────────────────────
print("\n[7] RequiemModel (API de alto nível)")
try:
    rm = RequiemModel(data_dir="music_data")
    check("Inicializou sem erro", rm.model is not None)

    empty_r = rm.run([], auto_detect=False)
    check("run([]) → [C]", len(empty_r) == 1 and empty_r[0].chord == "C")

    with_notes = rm.run(long_notes, bpm=120, tonality="C", auto_detect=False)
    check(f"run(notas) → {len(with_notes)} resultado(s)", len(with_notes) >= 1)

    pcs = RequiemModel.chord_pitch_classes("C")
    check("chord_pitch_classes('C') = [0,4,7]", pcs == [0, 4, 7])

    check("harmony_graph não vazio", len(rm.harmony_graph) > 0)
except Exception as e:
    check("RequiemModel sem erro", False, str(e))
    traceback.print_exc()

# ─────────────────────────────────────────────────────────
total = len(results)
passed = sum(results)
failed = total - passed
print(f"\n{'='*42}")
print(f"  Resultado: {passed}/{total} passou  |  {failed} falhou")
print(f"{'='*42}\n")
sys.exit(0 if failed == 0 else 1)
