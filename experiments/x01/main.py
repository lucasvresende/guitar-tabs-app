from enum import Enum
from pathlib import Path

import polars as pl

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
FILE_PATH = Path(__file__)
PARENT_PATH = FILE_PATH.parent
DATA_PATH = PARENT_PATH / "data"
OUT_PATH = PARENT_PATH / "out"


# -----------------------------------------------------------------------------
# Note Conversions
# -----------------------------------------------------------------------------
import math

NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

SEMITONE_TO_NOTE_SHARP = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}

SEMITONE_TO_NOTE_FLAT = {
    0: "C",
    1: "Db",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "Gb",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}


def note_name_to_midi_number(note: str) -> int:
    """Convert a note name (e.g., 'C4', 'A#3', 'Db5') to a MIDI number."""
    note = note.strip()

    if len(note) >= 3 and (note[1] in "#b"):
        name = note[:2]
        octave = int(note[2:])
    else:
        name = note[0]
        octave = int(note[1:])

    semitone = NOTE_TO_SEMITONE[name]
    return semitone + (octave + 1) * 12


def midi_number_to_note_name(midi: int, *, sharp: bool = True) -> str:
    """Convert MIDI number to note name."""
    octave = midi // 12 - 1
    semitone = midi % 12

    name = SEMITONE_TO_NOTE_SHARP[semitone] if sharp else SEMITONE_TO_NOTE_FLAT[semitone]
    return f"{name}{octave}"


def midi_number_to_pitch(midi: int) -> float:
    """Convert MIDI note to frequency in Hz using A4=440 Hz."""
    return 440.0 * (2 ** ((midi - 69) / 12))


def pitch_to_midi_number(freq: float) -> tuple[int, float]:
    """Convert frequency to (nearest_midi_note, cents_off)."""
    midi_real = 69 + 12 * math.log2(freq / 440.0)
    midi_int = round(midi_real)
    cents = (midi_real - midi_int) * 100
    return midi_int, cents


# -----------------------------------------------------------------------------
# Section
# -----------------------------------------------------------------------------
speed = 4 # dashes/s
dt = 1 / speed

df = (pl.read_csv(DATA_PATH / "tab_01.csv", has_header=False))
df = df.rename({old_name: f"string_{i}" for i, old_name in enumerate(df.columns, start=1)})

tuning = df.row(0)
tuning_midi = tuple(note_name_to_midi_number(note) for note in tuning)

df = df.slice(1)
df = df.with_columns(pl.all().replace("-", None).cast(pl.UInt8))

midi_df = df.with_columns(
    [
        (pl.col(f"string_{i}") + open_midi).cast(pl.UInt8).alias(f"string_{i}")
        for i, open_midi in enumerate(tuning_midi, start=1)
    ]
)


print(
    df.lazy()
    .with_row_index("index")
    .with_columns((pl.col("index") * dt).alias("start_time"))
    .collect()
)







    
    

    
    