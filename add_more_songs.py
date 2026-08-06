#!/usr/bin/env python3
"""
add_more_songs.py
=================
Adiciona ~35 novas músicas ao dataset music_data/
para enriquecer o treinamento do modelo LSTM.

Inclui:
  - Jazz standards diversificados
  - Pop brasileiro (MPB)
  - Rock clássico / Metal
  - Música erudita / cinematográfica
  - Eletrônica / ambient
  - R&B / Soul / Funk
  - Bossa Nova
"""
import json
from pathlib import Path

DATA_DIR = Path("music_data")
DATA_DIR.mkdir(exist_ok=True)

new_songs = [
    # ── Jazz Standards ────────────────────────────────────────
    {
        "title": "Autumn Leaves (Jazz Reharmonized)",
        "originalTonality": "Am",
        "normalizedProgression": ["Cm7","F7","Bbmaj7","Ebmaj7","Am7b5","D7","Gm","Gm","Cm7","F7","Bbmaj7","Ebmaj7","Am7b5","D7","Gm7","Gm7"]
    },
    {
        "title": "Summertime",
        "originalTonality": "Am",
        "normalizedProgression": ["Am","E7","Am","Am","Dm","Am","E7","Am","Am","E7","Am","C","Dm","F","E7","Am"]
    },
    {
        "title": "Misty",
        "originalTonality": "F",
        "normalizedProgression": ["Fmaj7","Cm7","F7","Bbmaj7","Bbm7","Eb7","Fmaj7","Dm7","Gm7","C7","Am7","D7","Gm7","C7"]
    },
    {
        "title": "Satin Doll",
        "originalTonality": "C",
        "normalizedProgression": ["Dm7","G7","Dm7","G7","Em7","A7","Em7","A7","Am7","D7","Abm7","Db7","Cmaj7","Gm7","C7"]
    },
    {
        "title": "Round Midnight",
        "originalTonality": "Cm",
        "normalizedProgression": ["Cm","Ebdim","Dm7b5","G7b9","Cm","Cm7","Fm","Dm7b5","G7","Cm","Abmaj7","G7","Cm"]
    },
    {
        "title": "Bye Bye Blackbird",
        "originalTonality": "F",
        "normalizedProgression": ["F","Fm","C","A7","Dm","G7","Gm7","C7","F","Fm","C","A7","Dm","G7","C7","F"]
    },
    {
        "title": "There Will Never Be Another You",
        "originalTonality": "Eb",
        "normalizedProgression": ["Ebmaj7","Ab7","Gm7","C7","Fm7","Bb7","Ebmaj7","Cm7","Fm7","Bb7","Ebmaj7","Gm7","C7","Fm7","Bb7","Ebmaj7"]
    },

    # ── MPB / Bossa Nova ──────────────────────────────────────
    {
        "title": "Desafinado",
        "originalTonality": "F",
        "normalizedProgression": ["Fmaj7","G7","Gm7","C7","Fmaj7","F7","Bbmaj7","Bb6","Am7","D7","Gm7","C7","Am7","D7","Gm7","C7","Fmaj7"]
    },
    {
        "title": "Corcovado",
        "originalTonality": "D",
        "normalizedProgression": ["Dm7","G7","Cmaj7","C6","Cm7","F7","Bbmaj7","E7","A7","D","Em7","A7","D","D7","G","Gm","D"]
    },
    {
        "title": "Mas Que Nada",
        "originalTonality": "Bm",
        "normalizedProgression": ["Bm","Em","Bm","Em","Bm","F#7","Bm","Bm","Em","Bm","Em","Bm","F#7","Bm"]
    },
    {
        "title": "Triste",
        "originalTonality": "Bb",
        "normalizedProgression": ["Bbmaj7","Bbm7","Eb7","Abmaj7","Dm7","G7","Cm7","F7","Bbmaj7","Bbm7","Eb7","Abmaj7","Am7","D7","Gm7","C7","Cm7","F7","Bb"]
    },
    {
        "title": "Chega de Saudade",
        "originalTonality": "D",
        "normalizedProgression": ["Dm","E7","Am","Dm","E7","Am","Gm","A7","Dm","C#7","F#","Bm","E7","A7","D","A7","D"]
    },

    # ── Rock / Metal ──────────────────────────────────────────
    {
        "title": "Paranoid Android",
        "originalTonality": "Am",
        "normalizedProgression": ["Am","G","F","Em","Am","G","F","Em","C","G","Em","C","G","D","Am","E7"]
    },
    {
        "title": "Under the Bridge",
        "originalTonality": "E",
        "normalizedProgression": ["E","B","C#m","G#m","A","E","B","C#m","G#m","A","E","B","F#m","E","B","C#m","A","E","B"]
    },
    {
        "title": "Wish You Were Here (verse)",
        "originalTonality": "G",
        "normalizedProgression": ["Em","G","Em","G","Em","A","Em","A","G","D","Am","G","D","Am","G"]
    },
    {
        "title": "Black",
        "originalTonality": "E",
        "normalizedProgression": ["E","A","E","A","E","A","B","C#m","A","B","E","A","E","A"]
    },
    {
        "title": "Fade to Black",
        "originalTonality": "Bm",
        "normalizedProgression": ["Bm","G","Bm","G","Bm","A","E","Bm","Bm","G","A","Bm","G","A","E","Bm"]
    },
    {
        "title": "Welcome Home (Coheed)",
        "originalTonality": "Am",
        "normalizedProgression": ["Am","G","F","C","Am","G","F","G","Am","Em","F","G","C","G","Am","F"]
    },

    # ── Música Cinematográfica ────────────────────────────────
    {
        "title": "Concerning Hobbits",
        "originalTonality": "D",
        "normalizedProgression": ["D","G","D","G","D","A","G","D","Bm","G","A","D","G","D","A","D"]
    },
    {
        "title": "Requiem for a Dream",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm","A7","Dm","Gm","A7","Dm","Dm","Gm","Dm","A7","Dm"]
    },
    {
        "title": "The Last of Us Theme",
        "originalTonality": "Am",
        "normalizedProgression": ["Am","C","G","Am","F","C","G","Am","Em","F","C","G","Am","F","G","Am"]
    },
    {
        "title": "Ori and the Blind Forest",
        "originalTonality": "C",
        "normalizedProgression": ["C","Am","F","G","C","Em","F","G","Am","F","C","G","Am","F","G","C"]
    },
    {
        "title": "Skyrim Main Theme (Dragonborn)",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm","C","Dm","C","Bb","C","Dm","Dm","Gm","Dm","A7","Dm","Bb","C","Dm"]
    },

    # ── R&B / Soul / Funk ─────────────────────────────────────
    {
        "title": "I Will Always Love You",
        "originalTonality": "G",
        "normalizedProgression": ["G","Bm","C","D","G","Bm","C","D","G","Em","C","D","G","Am","D","G"]
    },
    {
        "title": "Superstition (full)",
        "originalTonality": "Cm",
        "normalizedProgression": ["Cm","Cm","Cm","Cm","Eb","Eb","Cm","Cm","Fm","Fm","Cm","Cm","G7","F7","Cm","Cm"]
    },
    {
        "title": "Ain't No Sunshine",
        "originalTonality": "Am",
        "normalizedProgression": ["Am","Am","Am","Em","G7","Am","Am","Am","Dm","Dm","Am","Am","G7","Am"]
    },
    {
        "title": "What's Going On",
        "originalTonality": "E",
        "normalizedProgression": ["Emaj7","A6","Emaj7","A6","C#m7","F#m7","C#m7","F#m7","Emaj7","A6","Emaj7","A6"]
    },

    # ── Pop / Indie ───────────────────────────────────────────
    {
        "title": "Dancing Queen",
        "originalTonality": "A",
        "normalizedProgression": ["A","E","D","A","A","E","D","A","Bm","E7","A","D","A","Bm","E7","A"]
    },
    {
        "title": "Africa (Toto)",
        "originalTonality": "F#m",
        "normalizedProgression": ["F#m","D","A","E","F#m","D","A","E","Bm","E","A","D","Bm","E","A","F#m"]
    },
    {
        "title": "Mr. Brightside (chorus)",
        "originalTonality": "Bb",
        "normalizedProgression": ["Bb","F","Gm","Eb","Bb","F","Gm","Eb","Bb","F","Gm","Eb"]
    },
    {
        "title": "Somebody That I Used to Know",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm","C","Bb","Dm","C","Bb","Dm","C","Bb","A7","Dm","C","Bb","F","C","Dm"]
    },
    {
        "title": "Shape of My Heart (Sting)",
        "originalTonality": "F#m",
        "normalizedProgression": ["F#m","C#m","D","A","F#m","C#m","D","E","F#m","D","A","E","F#m","D","A","E","F#m"]
    },

    # ── Clássicos / Erudito ───────────────────────────────────
    {
        "title": "Prelude Op 28 No 4 Chopin",
        "originalTonality": "Em",
        "normalizedProgression": ["Em","Em","B7","B7","Em","C","G","Am","B7","Em","Fm","C","G","B7","Em"]
    },
    {
        "title": "Sonatina in G Major (Clementi)",
        "originalTonality": "G",
        "normalizedProgression": ["G","D7","G","D","G","D7","G","C","G","D7","G","Em","A7","D","G","D7","G"]
    },
    {
        "title": "Für Elise (extended)",
        "originalTonality": "Am",
        "normalizedProgression": ["Am","E7","Am","E7","Am","C","G","Am","E7","Am","F","C","G","Am","E7","Am"]
    },
    {
        "title": "Waltz Op 69 No 1 Chopin",
        "originalTonality": "Ab",
        "normalizedProgression": ["Ab","Eb7","Ab","Db","Ab","Eb7","Ab","Fm","Bbm","Eb7","Ab","Bbm","Eb7","Ab"]
    },
]


def save_song(song: dict, overwrite: bool = False):
    title = song["title"]
    # Gera nome de arquivo seguro
    fname = title.lower()
    for ch in " ()-,.'!?/":
        fname = fname.replace(ch, "_")
    while "__" in fname:
        fname = fname.replace("__", "_")
    fname = fname.strip("_") + ".json"
    path = DATA_DIR / fname

    if path.exists() and not overwrite:
        print(f"  [skip] {fname} (já existe)")
        return False

    path.write_text(json.dumps(song, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [add]  {fname}")
    return True


def main():
    print(f"Adicionando {len(new_songs)} músicas ao dataset...")
    print(f"Diretório: {DATA_DIR.resolve()}\n")

    added = 0
    skipped = 0
    for song in new_songs:
        if save_song(song):
            added += 1
        else:
            skipped += 1

    total = len(list(DATA_DIR.glob("*.json")))
    print(f"\n✓ {added} músicas adicionadas, {skipped} ignoradas (já existiam)")
    print(f"  Total no dataset: {total} músicas")


if __name__ == "__main__":
    main()
