#!/usr/bin/env python3
"""
import_chordonomicon.py
======================
Importa músicas do dataset Chordonomicon (chordonomicon_v2.csv) para o
formato JSON utilizado pelo requiem-harmony-model.

Fonte:
  https://huggingface.co/datasets/ailsntua/Chordonomicon/resolve/main/chordonomicon_v2.csv

Cada linha do CSV contém:
  id, chords, release_date, genres, decade, rock_genre,
  artist_id, main_genre, spotify_song_id, spotify_artist_id

A coluna `chords` usa notação especial:
  - Seções entre <...>  → removidas, usamos apenas os acordes
  - "min"   → "m"   (ex: Amin  → Am)
  - "s"     → "#"   (ex: Cs    → C#, Fsmin → F#m)
  - "no3d"  → "5"   (power chord, ex: Bno3d → B5)
  - "/"     → "/"   (slash chords, ex: A/Cs → A/C#)
  - Bemóis como "b" dentro do símbolo → mantidos

O script:
  1. Baixa o CSV diretamente de HuggingFace
  2. Converte a notação de acordes para o padrão do dataset
  3. Deduplica progressões idênticas
  4. Salva cada entrada como {title}.json em music_data/
  5. Gera um relatório final
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

CSV_URL = (
    "https://huggingface.co/datasets/ailsntua/Chordonomicon"
    "/resolve/main/chordonomicon_v2.csv"
)
DATA_DIR = Path("music_data")
DATA_DIR.mkdir(exist_ok=True)

# Número máximo de entradas a importar (None = todas)
MAX_ENTRIES: Optional[int] = None

# Tamanho mínimo e máximo de acordes únicos na progressão
# A mediana do Chordonomicon é ~75 acordes (progressão completa da música)
MIN_CHORDS = 4
MAX_CHORDS = 200

# Gêneros a excluir (progressões muito simples/ruidosas)
EXCLUDE_GENRES: set[str] = set()  # ex: {"children's music"}

# ---------------------------------------------------------------------------
# Conversão de notação Chordonomicon → padrão do dataset
# ---------------------------------------------------------------------------

# Mapa de enharmonics / abreviações do Chordonomicon
_SECTION_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _convert_chord(chord: str) -> str:
    """
    Converte um acorde individual da notação Chordonomicon para notação padrão.

    Exemplos:
      Amin    → Am
      Csmin   → C#m
      Fsmin   → F#m
      Fs7     → F#7
      Dsmin   → D#m
      Bno3d   → B5
      A/Cs    → A/C#
      Dmaj7   → Dmaj7  (sem mudança)
      Emin    → Em
      G7      → G7
      D7      → D7
      Bbmin   → Bbm
    """
    if not chord:
        return chord

    # Trata slash chords recursivamente
    if "/" in chord:
        parts = chord.split("/", 1)
        return _convert_chord(parts[0]) + "/" + _convert_chord(parts[1])

    # "no3d" → power chord "5"
    if "no3d" in chord:
        root = chord.replace("no3d", "")
        root = _sharpen(root)
        return root + "5"

    # Primeiro aplica sustenido (s → #) para que Csmin → C#min → C#m
    chord = _sharpen(chord)

    # Depois converte "min" → "m"
    chord = chord.replace("min", "m")

    return chord


def _sharpen(chord: str) -> str:
    """
    Substitui o sufixo 's' da nota raiz por '#'.
    Apenas substitui 's' imediatamente após a letra da nota raiz (A-G),
    diferenciando de sufixos como 'sus', 'maj', 'dim', etc.

    Exemplos:
      Cs   → C#
      Fs7  → F#7
      Gsm  → G#m (já convertido de Gsmin)
      Dsm  → D#m
      Bbs  → Bbs  (NÃO converte — Bb + s não é padrão, mas é raro)
    """
    # Nota raiz: uma letra A-G opcionalmente seguida de 'b' (bemol)
    # O 's' de sustenido aparece logo após a letra raiz sem 'b'
    # Pattern: ^([A-G])(s)  mas não se for 'sus'
    def replace_sharp(m: re.Match) -> str:
        note = m.group(1)
        return note + "#"

    return re.sub(r"^([A-G])s(?!us|dim|maj|m\b)", replace_sharp, chord)


def parse_chords(raw: str) -> list[str]:
    """
    Extrai a lista de acordes de uma string Chordonomicon.
    Remove marcadores de seção, converte notação e deduplica adjacentes.
    """
    # Remove seções <...>
    clean = _SECTION_RE.sub(" ", raw)
    tokens = _WHITESPACE_RE.split(clean.strip())

    chords = []
    for tok in tokens:
        if not tok:
            continue
        converted = _convert_chord(tok)
        if converted:
            chords.append(converted)

    # Remove duplicatas consecutivas (ex: "Am Am Am" → "Am")
    deduped: list[str] = []
    for c in chords:
        if not deduped or c != deduped[-1]:
            deduped.append(c)

    return deduped


def infer_tonality(chords: list[str]) -> str:
    """
    Infere uma tonalidade aproximada a partir da progressão.
    Usa heurística simples: o acorde mais comum que começa a progressão,
    com preferência pelo primeiro e último acorde.
    """
    if not chords:
        return "C"

    # Peso: primeiro e último acordes têm peso maior
    from collections import Counter
    counts: Counter[str] = Counter(chords)
    counts[chords[0]] += 3
    counts[chords[-1]] += 2

    return counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Utilitários de arquivo
# ---------------------------------------------------------------------------

def slugify(title: str) -> str:
    """Converte um título em nome de arquivo seguro."""
    name = title.lower()
    for ch in " ()-,.'!?/\\:;\"":
        name = name.replace(ch, "_")
    replacements = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "ã": "a", "â": "a", "à": "a", "á": "a", "ä": "a",
        "ô": "o", "ó": "o", "ö": "o",
        "ú": "u", "ü": "u", "û": "u",
        "ñ": "n", "ç": "c",
        "#": "sharp", "&": "and",
    }
    for src, tgt in replacements.items():
        name = name.replace(src, tgt)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_") + ".json"


def save_entry(entry: dict, overwrite: bool = False) -> bool:
    """Salva uma entrada no diretório music_data/. Retorna True se adicionada."""
    filename = slugify(entry["title"])
    path = DATA_DIR / filename
    if path.exists() and not overwrite:
        return False
    path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Download do CSV
# ---------------------------------------------------------------------------

def download_csv(url: str, chunk_size: int = 1024 * 64) -> str:
    """Baixa o CSV e retorna o conteúdo como string."""
    print(f"Baixando CSV de:\n  {url}\n")
    req = urllib.request.Request(url, headers={"User-Agent": "python/requiem-importer"})
    total = 0
    chunks: list[bytes] = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        content_length = resp.headers.get("Content-Length")
        if content_length:
            print(f"  Tamanho do arquivo: {int(content_length) / 1024 / 1024:.1f} MB")
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            mb = total / 1024 / 1024
            print(f"\r  Baixados: {mb:.2f} MB", end="", flush=True)
    print(f"\r  Download concluído: {total / 1024 / 1024:.2f} MB\n")
    raw = b"".join(chunks)
    # Detecta encoding
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


# ---------------------------------------------------------------------------
# Processamento principal
# ---------------------------------------------------------------------------

def build_title(row_id: str, main_genre: str, genres: str, decade: str) -> str:
    """
    Cria um título descritivo para a entrada, já que o dataset é anonimizado.
    Formato: Chordonomicon_{id}_{genre}_{decade}
    """
    genre_slug = main_genre.strip() if main_genre.strip() else "unknown"
    decade_slug = decade.strip().replace(".0", "") if decade.strip() else "unknown"
    return f"Chordonomicon_{row_id}_{genre_slug}_{decade_slug}"


def process_csv(content: str) -> tuple[int, int, int]:
    """
    Processa o CSV e salva os arquivos JSON.
    Retorna (adicionadas, ignoradas, erros).
    """
    reader = csv.DictReader(io.StringIO(content))
    added = skipped = errors = 0
    seen_progressions: set[tuple[str, ...]] = set()

    # Carrega progressões já existentes para evitar duplicatas
    print("Carregando progressões existentes...")
    existing = set()
    for f in DATA_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            prog = data.get("normalizedProgression", [])
            if prog:
                existing.add(tuple(prog))
        except Exception:
            pass
    print(f"  {len(existing)} progressões já existem no dataset.\n")

    print("Processando entradas do Chordonomicon...")
    print("-" * 60)

    start_time = time.time()

    for i, row in enumerate(reader):
        if MAX_ENTRIES and i >= MAX_ENTRIES:
            break

        try:
            row_id = row.get("id", str(i))
            raw_chords = row.get("chords", "").strip()
            main_genre = row.get("main_genre", "").strip()
            genres = row.get("genres", "").strip()
            decade = row.get("decade", "").strip()
            spotify_id = row.get("spotify_song_id", "").strip()

            # Filtra gêneros excluídos
            if main_genre in EXCLUDE_GENRES:
                skipped += 1
                continue

            # Converte acordes
            chords = parse_chords(raw_chords)

            # Filtra progressões muito curtas ou muito longas
            if len(chords) < MIN_CHORDS or len(chords) > MAX_CHORDS:
                skipped += 1
                continue

            # Deduplica progressões idênticas
            prog_key = tuple(chords)
            if prog_key in existing or prog_key in seen_progressions:
                skipped += 1
                continue
            seen_progressions.add(prog_key)

            # Infere tonalidade
            tonality = infer_tonality(chords)

            # Monta título e entrada
            title = build_title(row_id, main_genre, genres, decade)

            entry = {
                "title": title,
                "artist": f"Chordonomicon/{main_genre}" if main_genre else "Chordonomicon",
                "originalTonality": tonality,
                "normalizedProgression": chords,
                "source": "chordonomicon_v2",
            }
            if spotify_id:
                entry["spotifyId"] = spotify_id

            if save_entry(entry):
                added += 1
                if added % 500 == 0:
                    elapsed = time.time() - start_time
                    print(f"  [{added} adicionadas | {skipped} ignoradas] — {elapsed:.1f}s")
            else:
                skipped += 1

        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  [ERRO] linha {i}: {exc}")

    return added, skipped, errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Importador Chordonomicon → Requiem Dataset")
    print("=" * 60)
    print()

    # Download
    try:
        content = download_csv(CSV_URL)
    except Exception as exc:
        print(f"Falha no download: {exc}", file=sys.stderr)
        sys.exit(1)

    # Processa
    added, skipped, errors = process_csv(content)

    # Relatório final
    total = len(list(DATA_DIR.glob("*.json")))
    elapsed_total = time.time()

    print()
    print("=" * 60)
    print("  RELATÓRIO FINAL")
    print("=" * 60)
    print(f"  Adicionadas  : {added}")
    print(f"  Ignoradas    : {skipped}  (duplicatas, muito curtas/longas, etc.)")
    print(f"  Erros        : {errors}")
    print(f"  Total no dataset: {total} músicas")
    print("=" * 60)


if __name__ == "__main__":
    main()
