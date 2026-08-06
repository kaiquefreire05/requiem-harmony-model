# Requiem Harmony Model

A Python project for training and validating a neural harmony model based on chord progressions.

## Overview

This repository contains the full pipeline to:

- Build a chord-progression dataset from JSON songs
- Train an LSTM-based harmony model
- Validate generation quality and TypeScript parity
- Export model artifacts for frontend/browser inference

Core idea: the model learns the next chord from a short harmonic context and can generate coherent progressions.

## Repository Structure

- `music_data/` — song JSON files used as training data
- `chord_dataset.py` — dataset loading, encoding, and augmentation utilities
- `lstm_harmony_model.py` — model architecture
- `train_model.py` — model training script
- `requiem_model.py` — high-level model usage
- `harmony_engine.py` / `neural_harmony_engine.py` — generation logic
- `validate.py` — validation checks
- `test_requiem_model.py` — Python tests for model behavior
- `test_ts_parity.py` — parity checks between Python and TypeScript inference
- `export_weights_json.py` / `export_tfjs.py` — export scripts
- `add_songs_to_dataset.py` — unified script to add curated songs into `music_data/`

## Data Format

Each song in `music_data/` must follow this format:

```json
{
  "title": "Song Name",
  "originalTonality": "C",
  "normalizedProgression": ["C", "Am", "F", "G", "C", "G", "Am", "F"]
}
```

Optional field:

- `artist`: string

Requirements:

- `normalizedProgression` should have valid chord symbols
- Keep progressions musically meaningful (typically 8–16 chords)

## Prerequisites

- Python 3.10+
- pip

## Installation

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How It Works

1. **Dataset Preparation**
   - Reads all JSON songs from `music_data/`
   - Builds vocabulary of chord tokens
   - Creates input/output training samples (sliding window)
   - Applies transposition-based augmentation (if enabled)

2. **Model Training**
   - Trains an LSTM neural network on chord sequences
   - Learns probability distribution of the next chord
   - Saves trained model and metadata

3. **Generation**
   - Receives a seed progression
   - Iteratively predicts next chords
   - Applies harmony constraints to keep results coherent

4. **Validation & Export**
   - Runs tests and parity checks
   - Exports weights/vocabulary for frontend use

## Usage

### 1) Add Songs to the Dataset

Run the unified dataset script:

```bash
python3 add_songs_to_dataset.py
```

This script:

- Merges the previously split curated song lists
- Creates new files in `music_data/`
- Skips files that already exist

### 2) Train the Model

```bash
python3 train_model.py
```

Expected outputs include updated trained artifacts and training history.

### 3) Validate

```bash
python3 validate.py
python3 test_requiem_model.py
python3 test_ts_parity.py
```

### 4) Export Artifacts

```bash
python3 export_weights_json.py
python3 export_tfjs.py
```

Use these exports to run the same model behavior in TypeScript/browser environments.

## End-to-End Recommended Workflow

```bash
python3 add_songs_to_dataset.py
python3 train_model.py
python3 validate.py
python3 test_requiem_model.py
python3 test_ts_parity.py
python3 export_weights_json.py
python3 export_tfjs.py
```

## Notes

- Re-training after adding songs is required for changes to affect predictions.
- Keep dataset quality high; noisy chord annotations degrade generation quality.
- Commit new `music_data/*.json` files only when they are valid and intentional.

## Developed by Kaíque Freire
