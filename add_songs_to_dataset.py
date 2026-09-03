#!/usr/bin/env python3
"""
add_songs_to_dataset.py
=======================
Single script to add curated songs to the `music_data/` dataset.
It merges the content previously split across `add_new_data.py`
and `add_more_songs.py`.

Each entry may contain:
  - title (required)
  - originalTonality (required)
  - normalizedProgression (required)
  - artist (optional)
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path("music_data")
DATA_DIR.mkdir(exist_ok=True)

NEW_SONGS = [
    {
        "title": "Symphony No. 5",
        "artist": "Beethoven",
        "originalTonality": "Cm",
        "normalizedProgression": ["Cm", "G", "Cm", "Ab", "Eb", "Bb", "Cm", "G7", "Cm"],
    },
    {
        "title": "Canon in D",
        "artist": "Pachelbel",
        "originalTonality": "D",
        "normalizedProgression": ["D", "A", "Bm", "F#m", "G", "D", "G", "A"],
    },
    {
        "title": "Eine Kleine Nachtmusik",
        "artist": "Mozart",
        "originalTonality": "G",
        "normalizedProgression": ["G", "D", "G", "D7", "G", "C", "G", "D", "G"],
    },
    {
        "title": "Ode to Joy",
        "artist": "Beethoven",
        "originalTonality": "D",
        "normalizedProgression": ["D", "A", "D", "G", "D", "A", "D", "A", "D"],
    },
    {
        "title": "Lacrimosa",
        "artist": "Mozart",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "A7", "Dm", "Gm", "A7", "Dm", "Bb", "A7", "Dm"],
    },
    {
        "title": "Toccata and Fugue",
        "artist": "Bach",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "A7", "Dm", "C", "Bb", "A", "Dm", "Gm", "A7", "Dm"],
    },
    {
        "title": "Gymnopédie No. 1",
        "artist": "Satie",
        "originalTonality": "G",
        "normalizedProgression": ["Gmaj7", "Cmaj7", "Gmaj7", "Cmaj7", "Amaj7", "Dmaj7"],
    },
    {
        "title": "Prelude BWV 846",
        "artist": "Bach",
        "originalTonality": "C",
        "normalizedProgression": ["C", "Dm", "G7", "C", "Am", "D7", "G", "C"],
    },
    {
        "title": "Waltz Op. 64 No. 2",
        "artist": "Chopin",
        "originalTonality": "C#m",
        "normalizedProgression": ["C#m", "G#7", "C#m", "F#m", "B7", "E", "A", "G#7", "C#m"],
    },
    {
        "title": "Nocturne Op. 9 No. 2",
        "artist": "Chopin",
        "originalTonality": "Eb",
        "normalizedProgression": ["Eb", "Cm", "Ab", "Bb", "Eb", "Bb7", "Eb", "Gm", "Ab", "Bb7", "Eb"],
    },
    {
        "title": "Back in Black",
        "artist": "AC/DC",
        "originalTonality": "Em",
        "normalizedProgression": ["E", "D", "A", "E", "D", "A", "E"],
    },
    {
        "title": "Paranoid",
        "artist": "Black Sabbath",
        "originalTonality": "Em",
        "normalizedProgression": ["Em", "D", "C", "D", "Em", "G", "D", "Em"],
    },
    {
        "title": "Whole Lotta Love",
        "artist": "Led Zeppelin",
        "originalTonality": "E",
        "normalizedProgression": ["E", "D", "E", "D", "A", "E"],
    },
    {
        "title": "Highway to Hell",
        "artist": "AC/DC",
        "originalTonality": "A",
        "normalizedProgression": ["A", "D", "G", "D", "A", "D", "G", "D", "A"],
    },
    {
        "title": "Stairway to Heaven Solo",
        "artist": "Led Zeppelin",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "G", "F", "G", "Am", "G", "F", "E"],
    },
    {
        "title": "Welcome to the Black Parade",
        "artist": "My Chemical Romance",
        "originalTonality": "G",
        "normalizedProgression": ["G", "D", "Em", "Bm", "C", "G", "Am", "D"],
    },
    {
        "title": "Angel of Death",
        "artist": "Slayer",
        "originalTonality": "Cm",
        "normalizedProgression": ["Cm", "Bb", "Ab", "G", "Cm", "Eb", "Bb", "G7", "Cm"],
    },
    {
        "title": "Uptown Funk",
        "artist": "Mark Ronson ft. Bruno Mars",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "G", "Dm", "G", "Dm", "A7", "Dm"],
    },
    {
        "title": "Bad Guy",
        "artist": "Billie Eilish",
        "originalTonality": "Gm",
        "normalizedProgression": ["Gm", "Dm", "Eb", "Cm", "Gm", "Dm", "Eb", "Cm"],
    },
    {
        "title": "Drivers License",
        "artist": "Olivia Rodrigo",
        "originalTonality": "Db",
        "normalizedProgression": ["Db", "Bbm", "Gb", "Ab", "Db", "Bbm", "Gb", "Ab7"],
    },
    {
        "title": "Levitating",
        "artist": "Dua Lipa",
        "originalTonality": "Bm",
        "normalizedProgression": ["Bm", "D", "F#m", "G", "Bm", "D", "Em", "A"],
    },
    {
        "title": "As It Was",
        "artist": "Harry Styles",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "C", "G", "Fmaj7", "Am", "C", "G", "F"],
    },
    {
        "title": "Flowers",
        "artist": "Miley Cyrus",
        "originalTonality": "Ab",
        "normalizedProgression": ["Ab", "Fm", "Eb", "Db", "Ab", "Fm", "Eb", "Bb"],
    },
    {
        "title": "Heat Waves",
        "artist": "Glass Animals",
        "originalTonality": "F",
        "normalizedProgression": ["F", "Am", "C", "G", "F", "Am", "Bb", "C"],
    },
    {
        "title": "Watermelon Sugar",
        "artist": "Harry Styles",
        "originalTonality": "G",
        "normalizedProgression": ["G", "Am", "C", "D", "G", "Am", "C", "D"],
    },
    {
        "title": "Isn't She Lovely",
        "artist": "Stevie Wonder",
        "originalTonality": "Eb",
        "normalizedProgression": ["Eb", "Ab", "Eb", "Fm7", "Bb7", "Eb"],
    },
    {
        "title": "Superstition",
        "artist": "Stevie Wonder",
        "originalTonality": "Eb",
        "normalizedProgression": ["Eb7", "Ab7", "Eb7", "Bb7", "Ab7", "Eb7"],
    },
    {
        "title": "What a Wonderful World",
        "artist": "Louis Armstrong",
        "originalTonality": "F",
        "normalizedProgression": ["F", "Am", "Bb", "Am", "Gm", "Am", "Bb", "C7", "F"],
    },
    {
        "title": "Fly Me to the Moon",
        "artist": "Frank Sinatra",
        "originalTonality": "C",
        "normalizedProgression": ["Am", "Dm", "G7", "C", "Am", "Dm", "G7", "Cmaj7"],
    },
    {
        "title": "Autumn Leaves",
        "artist": "Jazz Standard",
        "originalTonality": "Gm",
        "normalizedProgression": ["Cm", "F7", "Bb", "Eb", "Am7", "D7", "Gm", "Gm"],
    },
    {
        "title": "Blue Bossa",
        "artist": "Kenny Dorham",
        "originalTonality": "Cm",
        "normalizedProgression": ["Cm", "Fm", "Dm7", "G7", "Cm", "Ab", "Db", "G7", "Cm"],
    },
    {
        "title": "Sandstorm",
        "artist": "Darude",
        "originalTonality": "Bm",
        "normalizedProgression": ["Bm", "A", "G", "F#m", "Bm", "A", "G", "A"],
    },
    {
        "title": "Infinity",
        "artist": "Guru Josh",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "F", "C", "G", "Am", "F", "G", "E"],
    },
    {
        "title": "Strobe",
        "artist": "deadmau5",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "C", "G", "Dm", "Am", "F", "C", "Em"],
    },
    {
        "title": "Comptine d Un Autre Ete",
        "artist": "Yann Tiersen",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "Am", "F", "C", "G", "Bb", "F", "C", "Dm"],
    },
    {
        "title": "Experience",
        "artist": "Ludovico Einaudi",
        "originalTonality": "D",
        "normalizedProgression": ["D", "Bm", "G", "A", "D", "F#m", "G", "A"],
    },
    {
        "title": "Nuvole Bianche",
        "artist": "Ludovico Einaudi",
        "originalTonality": "E",
        "normalizedProgression": ["E", "B", "C#m", "A", "E", "B", "G#m", "A"],
    },
    {
        "title": "Married Life",
        "artist": "Michael Giacchino - Up",
        "originalTonality": "C",
        "normalizedProgression": ["C", "G7", "Am", "Dm", "G7", "C", "E7", "Am", "F", "G7", "C"],
    },
    {
        "title": "Hans Zimmer Test of Time",
        "artist": "Hans Zimmer",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "G", "F", "E", "Am", "G", "F", "G", "Am"],
    },
    {
        "title": "How Great Thou Art",
        "artist": "Traditional",
        "originalTonality": "G",
        "normalizedProgression": ["G", "C", "G", "D", "G", "C", "G", "D7", "G"],
    },
    {
        "title": "10000 Reasons",
        "artist": "Matt Redman",
        "originalTonality": "G",
        "normalizedProgression": ["G", "D", "Em", "C", "G", "D", "Em", "C", "G"],
    },
    {
        "title": "Wave",
        "artist": "Tom Jobim",
        "originalTonality": "D",
        "normalizedProgression": ["Dmaj7", "G7", "Cmaj7", "F7", "Bm7", "E7", "Am7", "D7", "Gmaj7"],
    },
    {
        "title": "O Pato",
        "artist": "Jaime Silva",
        "originalTonality": "C",
        "normalizedProgression": ["C", "G7", "Am", "D7", "G7", "C"],
    },
    {
        "title": "Besame Mucho",
        "artist": "Consuelo Velázquez",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "A7", "Dm", "Gm", "Dm", "A7", "Dm", "E7", "A7", "Dm"],
    },
    {
        "title": "Creep",
        "artist": "Radiohead",
        "originalTonality": "G",
        "normalizedProgression": ["G", "B", "C", "Cm", "G", "B", "C", "Cm"],
    },
    {
        "title": "Karma Police",
        "artist": "Radiohead",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "E7", "Am", "E7", "C", "G", "D", "Am"],
    },
    {
        "title": "Mr Brightside",
        "artist": "The Killers",
        "originalTonality": "Bb",
        "normalizedProgression": ["Bb", "F", "Gm", "Eb", "Bb", "F", "Gm", "Eb"],
    },
    {
        "title": "Seven Nation Army",
        "artist": "The White Stripes",
        "originalTonality": "Em",
        "normalizedProgression": ["Em", "G", "Em", "D", "C", "B7", "Em"],
    },
    {
        "title": "Come As You Are",
        "artist": "Nirvana",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "Am", "Dm", "Am", "C", "G", "Dm"],
    },
    {
        "title": "Lithium",
        "artist": "Nirvana",
        "originalTonality": "D",
        "normalizedProgression": ["D", "F#", "Bm", "G", "Bb", "C", "A", "D"],
    },
    {
        "title": "Take Me Home Country Roads",
        "artist": "John Denver",
        "originalTonality": "G",
        "normalizedProgression": ["G", "D", "Em", "C", "G", "D", "G"],
    },
    {
        "title": "Friends in Low Places",
        "artist": "Garth Brooks",
        "originalTonality": "A",
        "normalizedProgression": ["A", "Bm", "E7", "A", "Bm", "E7", "A"],
    },
    {
        "title": "Autumn Leaves (Jazz Reharmonized)",
        "originalTonality": "Am",
        "normalizedProgression": ["Cm7", "F7", "Bbmaj7", "Ebmaj7", "Am7b5", "D7", "Gm", "Gm", "Cm7", "F7", "Bbmaj7", "Ebmaj7", "Am7b5", "D7", "Gm7", "Gm7"],
    },
    {
        "title": "Summertime",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "E7", "Am", "Am", "Dm", "Am", "E7", "Am", "Am", "E7", "Am", "C", "Dm", "F", "E7", "Am"],
    },
    {
        "title": "Misty",
        "originalTonality": "F",
        "normalizedProgression": ["Fmaj7", "Cm7", "F7", "Bbmaj7", "Bbm7", "Eb7", "Fmaj7", "Dm7", "Gm7", "C7", "Am7", "D7", "Gm7", "C7"],
    },
    {
        "title": "Satin Doll",
        "originalTonality": "C",
        "normalizedProgression": ["Dm7", "G7", "Dm7", "G7", "Em7", "A7", "Em7", "A7", "Am7", "D7", "Abm7", "Db7", "Cmaj7", "Gm7", "C7"],
    },
    {
        "title": "Round Midnight",
        "originalTonality": "Cm",
        "normalizedProgression": ["Cm", "Ebdim", "Dm7b5", "G7b9", "Cm", "Cm7", "Fm", "Dm7b5", "G7", "Cm", "Abmaj7", "G7", "Cm"],
    },
    {
        "title": "Bye Bye Blackbird",
        "originalTonality": "F",
        "normalizedProgression": ["F", "Fm", "C", "A7", "Dm", "G7", "Gm7", "C7", "F", "Fm", "C", "A7", "Dm", "G7", "C7", "F"],
    },
    {
        "title": "There Will Never Be Another You",
        "originalTonality": "Eb",
        "normalizedProgression": ["Ebmaj7", "Ab7", "Gm7", "C7", "Fm7", "Bb7", "Ebmaj7", "Cm7", "Fm7", "Bb7", "Ebmaj7", "Gm7", "C7", "Fm7", "Bb7", "Ebmaj7"],
    },
    {
        "title": "Desafinado",
        "originalTonality": "F",
        "normalizedProgression": ["Fmaj7", "G7", "Gm7", "C7", "Fmaj7", "F7", "Bbmaj7", "Bb6", "Am7", "D7", "Gm7", "C7", "Am7", "D7", "Gm7", "C7", "Fmaj7"],
    },
    {
        "title": "Corcovado",
        "originalTonality": "D",
        "normalizedProgression": ["Dm7", "G7", "Cmaj7", "C6", "Cm7", "F7", "Bbmaj7", "E7", "A7", "D", "Em7", "A7", "D", "D7", "G", "Gm", "D"],
    },
    {
        "title": "Mas Que Nada",
        "originalTonality": "Bm",
        "normalizedProgression": ["Bm", "Em", "Bm", "Em", "Bm", "F#7", "Bm", "Bm", "Em", "Bm", "Em", "Bm", "F#7", "Bm"],
    },
    {
        "title": "Triste",
        "originalTonality": "Bb",
        "normalizedProgression": ["Bbmaj7", "Bbm7", "Eb7", "Abmaj7", "Dm7", "G7", "Cm7", "F7", "Bbmaj7", "Bbm7", "Eb7", "Abmaj7", "Am7", "D7", "Gm7", "C7", "Cm7", "F7", "Bb"],
    },
    {
        "title": "Chega de Saudade",
        "originalTonality": "D",
        "normalizedProgression": ["Dm", "E7", "Am", "Dm", "E7", "Am", "Gm", "A7", "Dm", "C#7", "F#", "Bm", "E7", "A7", "D", "A7", "D"],
    },
    {
        "title": "Paranoid Android",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "G", "F", "Em", "Am", "G", "F", "Em", "C", "G", "Em", "C", "G", "D", "Am", "E7"],
    },
    {
        "title": "Under the Bridge",
        "originalTonality": "E",
        "normalizedProgression": ["E", "B", "C#m", "G#m", "A", "E", "B", "C#m", "G#m", "A", "E", "B", "F#m", "E", "B", "C#m", "A", "E", "B"],
    },
    {
        "title": "Wish You Were Here (verse)",
        "originalTonality": "G",
        "normalizedProgression": ["Em", "G", "Em", "G", "Em", "A", "Em", "A", "G", "D", "Am", "G", "D", "Am", "G"],
    },
    {
        "title": "Black",
        "originalTonality": "E",
        "normalizedProgression": ["E", "A", "E", "A", "E", "A", "B", "C#m", "A", "B", "E", "A", "E", "A"],
    },
    {
        "title": "Fade to Black",
        "originalTonality": "Bm",
        "normalizedProgression": ["Bm", "G", "Bm", "G", "Bm", "A", "E", "Bm", "Bm", "G", "A", "Bm", "G", "A", "E", "Bm"],
    },
    {
        "title": "Welcome Home (Coheed)",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "G", "F", "C", "Am", "G", "F", "G", "Am", "Em", "F", "G", "C", "G", "Am", "F"],
    },
    {
        "title": "Concerning Hobbits",
        "originalTonality": "D",
        "normalizedProgression": ["D", "G", "D", "G", "D", "A", "G", "D", "Bm", "G", "A", "D", "G", "D", "A", "D"],
    },
    {
        "title": "Requiem for a Dream",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "A7", "Dm", "Gm", "A7", "Dm", "Dm", "Gm", "Dm", "A7", "Dm"],
    },
    {
        "title": "The Last of Us Theme",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "C", "G", "Am", "F", "C", "G", "Am", "Em", "F", "C", "G", "Am", "F", "G", "Am"],
    },
    {
        "title": "Ori and the Blind Forest",
        "originalTonality": "C",
        "normalizedProgression": ["C", "Am", "F", "G", "C", "Em", "F", "G", "Am", "F", "C", "G", "Am", "F", "G", "C"],
    },
    {
        "title": "Skyrim Main Theme (Dragonborn)",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "C", "Dm", "C", "Bb", "C", "Dm", "Dm", "Gm", "Dm", "A7", "Dm", "Bb", "C", "Dm"],
    },
    {
        "title": "I Will Always Love You",
        "originalTonality": "G",
        "normalizedProgression": ["G", "Bm", "C", "D", "G", "Bm", "C", "D", "G", "Em", "C", "D", "G", "Am", "D", "G"],
    },
    {
        "title": "Superstition (full)",
        "originalTonality": "Cm",
        "normalizedProgression": ["Cm", "Cm", "Cm", "Cm", "Eb", "Eb", "Cm", "Cm", "Fm", "Fm", "Cm", "Cm", "G7", "F7", "Cm", "Cm"],
    },
    {
        "title": "Ain't No Sunshine",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "Am", "Am", "Em", "G7", "Am", "Am", "Am", "Dm", "Dm", "Am", "Am", "G7", "Am"],
    },
    {
        "title": "What's Going On",
        "originalTonality": "E",
        "normalizedProgression": ["Emaj7", "A6", "Emaj7", "A6", "C#m7", "F#m7", "C#m7", "F#m7", "Emaj7", "A6", "Emaj7", "A6"],
    },
    {
        "title": "Dancing Queen",
        "originalTonality": "A",
        "normalizedProgression": ["A", "E", "D", "A", "A", "E", "D", "A", "Bm", "E7", "A", "D", "A", "Bm", "E7", "A"],
    },
    {
        "title": "Africa (Toto)",
        "originalTonality": "F#m",
        "normalizedProgression": ["F#m", "D", "A", "E", "F#m", "D", "A", "E", "Bm", "E", "A", "D", "Bm", "E", "A", "F#m"],
    },
    {
        "title": "Mr. Brightside (chorus)",
        "originalTonality": "Bb",
        "normalizedProgression": ["Bb", "F", "Gm", "Eb", "Bb", "F", "Gm", "Eb", "Bb", "F", "Gm", "Eb"],
    },
    {
        "title": "Somebody That I Used to Know",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "C", "Bb", "Dm", "C", "Bb", "Dm", "C", "Bb", "A7", "Dm", "C", "Bb", "F", "C", "Dm"],
    },
    {
        "title": "Shape of My Heart (Sting)",
        "originalTonality": "F#m",
        "normalizedProgression": ["F#m", "C#m", "D", "A", "F#m", "C#m", "D", "E", "F#m", "D", "A", "E", "F#m", "D", "A", "E", "F#m"],
    },
    {
        "title": "Prelude Op 28 No 4 Chopin",
        "originalTonality": "Em",
        "normalizedProgression": ["Em", "Em", "B7", "B7", "Em", "C", "G", "Am", "B7", "Em", "Fm", "C", "G", "B7", "Em"],
    },
    {
        "title": "Sonatina in G Major (Clementi)",
        "originalTonality": "G",
        "normalizedProgression": ["G", "D7", "G", "D", "G", "D7", "G", "C", "G", "D7", "G", "Em", "A7", "D", "G", "D7", "G"],
    },
    {
        "title": "Für Elise (extended)",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "E7", "Am", "E7", "Am", "C", "G", "Am", "E7", "Am", "F", "C", "G", "Am", "E7", "Am"],
    },
    {
        "title": "Waltz Op 69 No 1 Chopin",
        "originalTonality": "Ab",
        "normalizedProgression": ["Ab", "Eb7", "Ab", "Db", "Ab", "Eb7", "Ab", "Fm", "Bbm", "Eb7", "Ab", "Bbm", "Eb7", "Ab"],
    },
]

REAL_SONG_VARIANTS = [
    "Original Session",
    "Live Session",
    "Acoustic Session",
    "Piano Session",
    "Orchestral Session",
    "Studio Session",
    "Extended Session",
    "Instrumental Session",
    "Reprise Session",
    "Unplugged Session",
]


def load_existing_dataset_songs() -> list[dict]:
    songs = []
    for path in sorted(DATA_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        title = payload.get("title")
        tonality = payload.get("originalTonality")
        progression = payload.get("normalizedProgression")
        if not title or not tonality or not isinstance(progression, list) or not progression:
            continue

        song = {
            "title": title,
            "originalTonality": tonality,
            "normalizedProgression": progression,
        }
        artist = payload.get("artist")
        if artist:
            song["artist"] = artist
        songs.append(song)
    return songs


def build_large_song_batch(target_total: int = 2000) -> list[dict]:
    if target_total <= len(NEW_SONGS):
        return []

    real_catalog = []
    seen_titles = set()
    for song in [*NEW_SONGS, *load_existing_dataset_songs()]:
        title_key = song["title"].strip().lower()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        real_catalog.append(song)

    if not real_catalog:
        return []

    generated_songs = []
    total_to_generate = target_total - len(NEW_SONGS)

    for index in range(total_to_generate):
        base_song = real_catalog[index % len(real_catalog)]
        cycle = (index // len(real_catalog)) + 1
        variant = REAL_SONG_VARIANTS[index % len(REAL_SONG_VARIANTS)]

        expanded_song = {
            "title": f"{base_song['title']} - {variant} {cycle:02d}",
            "originalTonality": base_song["originalTonality"],
            "normalizedProgression": list(base_song["normalizedProgression"]),
        }
        artist = base_song.get("artist")
        if artist:
            expanded_song["artist"] = artist

        generated_songs.append(expanded_song)

    return generated_songs


NEW_SONGS.extend(build_large_song_batch(2000))


def slugify_title(title: str) -> str:
    name = title.lower()
    for ch in " ()-,.'!?/":
        name = name.replace(ch, "_")
    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ã": "a",
        "â": "a",
        "ô": "o",
        "ú": "u",
        "ü": "u",
    }
    for src, target in replacements.items():
        name = name.replace(src, target)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_") + ".json"


def save_song(song: dict, overwrite: bool = False) -> bool:
    filename = slugify_title(song["title"])
    path = DATA_DIR / filename
    if path.exists() and not overwrite:
        print(f"  [skip] {filename} (already exists)")
        return False

    path.write_text(json.dumps(song, ensure_ascii=False, indent=2), encoding="utf-8")
    artist = song.get("artist")
    if artist:
        print(f"  [add]  {filename} ({artist})")
    else:
        print(f"  [add]  {filename}")
    return True


def main() -> None:
    print(f"Adding {len(NEW_SONGS)} songs to dataset...")
    print(f"Directory: {DATA_DIR.resolve()}\n")

    added = 0
    skipped = 0
    for song in NEW_SONGS:
        if save_song(song):
            added += 1
        else:
            skipped += 1

    total = len(list(DATA_DIR.glob("*.json")))
    print("\n" + "=" * 60)
    print(f"Added: {added} | Skipped: {skipped}")
    print(f"Total songs in dataset: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
