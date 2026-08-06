"""
add_new_data.py
===============
Adiciona novos JSONs de treinamento ao dataset de harmonia.
Cada entrada tem: title, artist, originalTonality, normalizedProgression
"""
import json
from pathlib import Path

DATA_DIR = Path("music_data")
DATA_DIR.mkdir(exist_ok=True)

NEW_SONGS = [
    # ── Clássicos / Erudito ──────────────────────────────────
    {
        "title": "Symphony No. 5",
        "artist": "Beethoven",
        "originalTonality": "Cm",
        "normalizedProgression": ["Cm", "G", "Cm", "Ab", "Eb", "Bb", "Cm", "G7", "Cm"]
    },
    {
        "title": "Canon in D",
        "artist": "Pachelbel",
        "originalTonality": "D",
        "normalizedProgression": ["D", "A", "Bm", "F#m", "G", "D", "G", "A"]
    },
    {
        "title": "Eine Kleine Nachtmusik",
        "artist": "Mozart",
        "originalTonality": "G",
        "normalizedProgression": ["G", "D", "G", "D7", "G", "C", "G", "D", "G"]
    },
    {
        "title": "Ode to Joy",
        "artist": "Beethoven",
        "originalTonality": "D",
        "normalizedProgression": ["D", "A", "D", "G", "D", "A", "D", "A", "D"]
    },
    {
        "title": "Lacrimosa",
        "artist": "Mozart",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "A7", "Dm", "Gm", "A7", "Dm", "Bb", "A7", "Dm"]
    },
    {
        "title": "Toccata and Fugue",
        "artist": "Bach",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "A7", "Dm", "C", "Bb", "A", "Dm", "Gm", "A7", "Dm"]
    },
    {
        "title": "Gymnopédie No. 1",
        "artist": "Satie",
        "originalTonality": "G",
        "normalizedProgression": ["Gmaj7", "Cmaj7", "Gmaj7", "Cmaj7", "Amaj7", "Dmaj7"]
    },
    {
        "title": "Prelude BWV 846",
        "artist": "Bach",
        "originalTonality": "C",
        "normalizedProgression": ["C", "Dm", "G7", "C", "Am", "D7", "G", "C"]
    },
    {
        "title": "Waltz Op. 64 No. 2",
        "artist": "Chopin",
        "originalTonality": "C#m",
        "normalizedProgression": ["C#m", "G#7", "C#m", "F#m", "B7", "E", "A", "G#7", "C#m"]
    },
    {
        "title": "Nocturne Op. 9 No. 2",
        "artist": "Chopin",
        "originalTonality": "Eb",
        "normalizedProgression": ["Eb", "Cm", "Ab", "Bb", "Eb", "Bb7", "Eb", "Gm", "Ab", "Bb7", "Eb"]
    },

    # ── Rock / Metal ─────────────────────────────────────────
    {
        "title": "Back in Black",
        "artist": "AC/DC",
        "originalTonality": "Em",
        "normalizedProgression": ["E", "D", "A", "E", "D", "A", "E"]
    },
    {
        "title": "Paranoid",
        "artist": "Black Sabbath",
        "originalTonality": "Em",
        "normalizedProgression": ["Em", "D", "C", "D", "Em", "G", "D", "Em"]
    },
    {
        "title": "Whole Lotta Love",
        "artist": "Led Zeppelin",
        "originalTonality": "E",
        "normalizedProgression": ["E", "D", "E", "D", "A", "E"]
    },
    {
        "title": "Highway to Hell",
        "artist": "AC/DC",
        "originalTonality": "A",
        "normalizedProgression": ["A", "D", "G", "D", "A", "D", "G", "D", "A"]
    },
    {
        "title": "Stairway to Heaven Solo",
        "artist": "Led Zeppelin",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "G", "F", "G", "Am", "G", "F", "E"]
    },
    {
        "title": "Welcome to the Black Parade",
        "artist": "My Chemical Romance",
        "originalTonality": "G",
        "normalizedProgression": ["G", "D", "Em", "Bm", "C", "G", "Am", "D"]
    },
    {
        "title": "Angel of Death",
        "artist": "Slayer",
        "originalTonality": "Cm",
        "normalizedProgression": ["Cm", "Bb", "Ab", "G", "Cm", "Eb", "Bb", "G7", "Cm"]
    },

    # ── Pop ──────────────────────────────────────────────────
    {
        "title": "Uptown Funk",
        "artist": "Mark Ronson ft. Bruno Mars",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "G", "Dm", "G", "Dm", "A7", "Dm"]
    },
    {
        "title": "Bad Guy",
        "artist": "Billie Eilish",
        "originalTonality": "Gm",
        "normalizedProgression": ["Gm", "Dm", "Eb", "Cm", "Gm", "Dm", "Eb", "Cm"]
    },
    {
        "title": "Drivers License",
        "artist": "Olivia Rodrigo",
        "originalTonality": "Db",
        "normalizedProgression": ["Db", "Bbm", "Gb", "Ab", "Db", "Bbm", "Gb", "Ab7"]
    },
    {
        "title": "Levitating",
        "artist": "Dua Lipa",
        "originalTonality": "Bm",
        "normalizedProgression": ["Bm", "D", "F#m", "G", "Bm", "D", "Em", "A"]
    },
    {
        "title": "As It Was",
        "artist": "Harry Styles",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "C", "G", "Fmaj7", "Am", "C", "G", "F"]
    },
    {
        "title": "Flowers",
        "artist": "Miley Cyrus",
        "originalTonality": "Ab",
        "normalizedProgression": ["Ab", "Fm", "Eb", "Db", "Ab", "Fm", "Eb", "Bb"]
    },
    {
        "title": "Heat Waves",
        "artist": "Glass Animals",
        "originalTonality": "F",
        "normalizedProgression": ["F", "Am", "C", "G", "F", "Am", "Bb", "C"]
    },
    {
        "title": "Watermelon Sugar",
        "artist": "Harry Styles",
        "originalTonality": "G",
        "normalizedProgression": ["G", "Am", "C", "D", "G", "Am", "C", "D"]
    },

    # ── Soul / R&B / Jazz ────────────────────────────────────
    {
        "title": "Isn't She Lovely",
        "artist": "Stevie Wonder",
        "originalTonality": "Eb",
        "normalizedProgression": ["Eb", "Ab", "Eb", "Fm7", "Bb7", "Eb"]
    },
    {
        "title": "Superstition",
        "artist": "Stevie Wonder",
        "originalTonality": "Eb",
        "normalizedProgression": ["Eb7", "Ab7", "Eb7", "Bb7", "Ab7", "Eb7"]
    },
    {
        "title": "What a Wonderful World",
        "artist": "Louis Armstrong",
        "originalTonality": "F",
        "normalizedProgression": ["F", "Am", "Bb", "Am", "Gm", "Am", "Bb", "C7", "F"]
    },
    {
        "title": "Fly Me to the Moon",
        "artist": "Frank Sinatra",
        "originalTonality": "C",
        "normalizedProgression": ["Am", "Dm", "G7", "C", "Am", "Dm", "G7", "Cmaj7"]
    },
    {
        "title": "Autumn Leaves",
        "artist": "Jazz Standard",
        "originalTonality": "Gm",
        "normalizedProgression": ["Cm", "F7", "Bb", "Eb", "Am7", "D7", "Gm", "Gm"]
    },
    {
        "title": "Blue Bossa",
        "artist": "Kenny Dorham",
        "originalTonality": "Cm",
        "normalizedProgression": ["Cm", "Fm", "Dm7", "G7", "Cm", "Ab", "Db", "G7", "Cm"]
    },

    # ── Eletrônica / Dance ───────────────────────────────────
    {
        "title": "Sandstorm",
        "artist": "Darude",
        "originalTonality": "Bm",
        "normalizedProgression": ["Bm", "A", "G", "F#m", "Bm", "A", "G", "A"]
    },
    {
        "title": "Infinity",
        "artist": "Guru Josh",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "F", "C", "G", "Am", "F", "G", "E"]
    },
    {
        "title": "Strobe",
        "artist": "deadmau5",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "C", "G", "Dm", "Am", "F", "C", "Em"]
    },

    # ── Trilhas Sonoras / Cinematográfico ────────────────────
    {
        "title": "Comptine d Un Autre Ete",
        "artist": "Yann Tiersen",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "Am", "F", "C", "G", "Bb", "F", "C", "Dm"]
    },
    {
        "title": "Experience",
        "artist": "Ludovico Einaudi",
        "originalTonality": "D",
        "normalizedProgression": ["D", "Bm", "G", "A", "D", "F#m", "G", "A"]
    },
    {
        "title": "Nuvole Bianche",
        "artist": "Ludovico Einaudi",
        "originalTonality": "E",
        "normalizedProgression": ["E", "B", "C#m", "A", "E", "B", "G#m", "A"]
    },
    {
        "title": "Married Life",
        "artist": "Michael Giacchino - Up",
        "originalTonality": "C",
        "normalizedProgression": ["C", "G7", "Am", "Dm", "G7", "C", "E7", "Am", "F", "G7", "C"]
    },
    {
        "title": "Hans Zimmer Test of Time",
        "artist": "Hans Zimmer",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "G", "F", "E", "Am", "G", "F", "G", "Am"]
    },

    # ── Gospel / Spiritual ───────────────────────────────────
    {
        "title": "How Great Thou Art",
        "artist": "Traditional",
        "originalTonality": "G",
        "normalizedProgression": ["G", "C", "G", "D", "G", "C", "G", "D7", "G"]
    },
    {
        "title": "10000 Reasons",
        "artist": "Matt Redman",
        "originalTonality": "G",
        "normalizedProgression": ["G", "D", "Em", "C", "G", "D", "Em", "C", "G"]
    },

    # ── Bossa Nova / MPB ─────────────────────────────────────
    {
        "title": "Wave",
        "artist": "Tom Jobim",
        "originalTonality": "D",
        "normalizedProgression": ["Dmaj7", "G7", "Cmaj7", "F7", "Bm7", "E7", "Am7", "D7", "Gmaj7"]
    },
    {
        "title": "O Pato",
        "artist": "Jaime Silva",
        "originalTonality": "C",
        "normalizedProgression": ["C", "G7", "Am", "D7", "G7", "C"]
    },
    {
        "title": "Besame Mucho",
        "artist": "Consuelo Velázquez",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "A7", "Dm", "Gm", "Dm", "A7", "Dm", "E7", "A7", "Dm"]
    },

    # ── Alternativo / Indie ──────────────────────────────────
    {
        "title": "Creep",
        "artist": "Radiohead",
        "originalTonality": "G",
        "normalizedProgression": ["G", "B", "C", "Cm", "G", "B", "C", "Cm"]
    },
    {
        "title": "Karma Police",
        "artist": "Radiohead",
        "originalTonality": "Am",
        "normalizedProgression": ["Am", "E7", "Am", "E7", "C", "G", "D", "Am"]
    },
    {
        "title": "Mr Brightside",
        "artist": "The Killers",
        "originalTonality": "Bb",
        "normalizedProgression": ["Bb", "F", "Gm", "Eb", "Bb", "F", "Gm", "Eb"]
    },
    {
        "title": "Seven Nation Army",
        "artist": "The White Stripes",
        "originalTonality": "Em",
        "normalizedProgression": ["Em", "G", "Em", "D", "C", "B7", "Em"]
    },
    {
        "title": "Come As You Are",
        "artist": "Nirvana",
        "originalTonality": "Dm",
        "normalizedProgression": ["Dm", "Am", "Dm", "Am", "C", "G", "Dm"]
    },
    {
        "title": "Lithium",
        "artist": "Nirvana",
        "originalTonality": "D",
        "normalizedProgression": ["D", "F#", "Bm", "G", "Bb", "C", "A", "D"]
    },

    # ── Country ──────────────────────────────────────────────
    {
        "title": "Take Me Home Country Roads",
        "artist": "John Denver",
        "originalTonality": "G",
        "normalizedProgression": ["G", "D", "Em", "C", "G", "D", "G"]
    },
    {
        "title": "Friends in Low Places",
        "artist": "Garth Brooks",
        "originalTonality": "A",
        "normalizedProgression": ["A", "Bm", "E7", "A", "Bm", "E7", "A"]
    },
]


def save_song(song: dict, data_dir: Path) -> bool:
    """Salva uma música como JSON. Retorna True se criou, False se já existia."""
    filename = (
        song["title"]
        .lower()
        .replace(" ", "_")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(".", "")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("ã", "a")
        .replace("â", "a")
        .replace("ô", "o")
        .replace("ú", "u")
        .replace("ü", "u")
        + ".json"
    )
    filepath = data_dir / filename
    if filepath.exists():
        return False
    filepath.write_text(json.dumps(song, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def main():
    created = 0
    skipped = 0
    for song in NEW_SONGS:
        if save_song(song, DATA_DIR):
            print(f"  ✓  {song['title']} ({song['artist']})")
            created += 1
        else:
            print(f"  ⟳  Já existe: {song['title']}")
            skipped += 1

    total = len(list(DATA_DIR.glob("*.json")))
    print(f"\n{'='*55}")
    print(f"  Criados: {created}  |  Ignorados: {skipped}")
    print(f"  Total de JSONs no dataset: {total}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
