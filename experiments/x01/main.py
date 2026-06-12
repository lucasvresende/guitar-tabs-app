from enum import Enum
from pathlib import Path
import json

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


def name_to_midi(note: str) -> int:
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


def midi_to_name(midi: int, *, sharp: bool = True) -> str:
    """Convert MIDI number to note name."""
    octave = midi // 12 - 1
    semitone = midi % 12

    name = SEMITONE_TO_NOTE_SHARP[semitone] if sharp else SEMITONE_TO_NOTE_FLAT[semitone]
    return f"{name}{octave}"


def midi_to_frequency(midi: int) -> float:
    """Convert MIDI note to frequency in Hz using A4=440 Hz."""
    return 440.0 * (2 ** ((midi - 69) / 12))


def frequency_to_midi(freq: float) -> float:
    """Convert frequency in Hz to MIDI value (float)."""
    return 69 + 12 * math.log2(freq / 440.0)


# -----------------------------------------------------------------------------
# Section
# -----------------------------------------------------------------------------
speed = 4 # dashes/s
dt = 1 / speed

tuning = json.load(open(DATA_PATH / "tuning_01.json"))
open_string_midis = tuple(name_to_midi(note) for note in tuning)
string_columns = tuple(f"string_{i}" for i in range(1, len(tuning)))
df = (
    pl.read_csv(
        DATA_PATH / "tab_01.csv",
        has_header=False,
        new_columns=string_columns
    )
    .with_row_index("index")
    .with_columns(pl.col(string_columns).replace("-", None).cast(pl.UInt8))
    .with_columns((pl.col("index") * dt).alias("start_time"))
    .select("index", "start_time", pl.all()) 
)
midi_df = df.with_columns(
    [
        (pl.col(string_col) + open_midi).cast(pl.UInt8).alias(string_col)
        for string_col, open_midi in zip(string_columns, open_string_midis)
    ],
)
freq_df = midi_df.with_columns(
    [
        (440.0 * 2 ** ((pl.col(string_col).cast(pl.Float64) - 69.0) / 12.0)).alias(string_col)
        for string_col in string_columns
    ],
)




    
    

    
    