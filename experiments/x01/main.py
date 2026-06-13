from enum import Enum
from pathlib import Path
import json
from sqlite3 import SQLITE_ERROR_MISSING_COLLSEQ
from typing import Literal

import polars as pl
import polars.selectors as cs
from numba import njit
import numpy as np
import numpy.typing as npt

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
# Extraction
# -----------------------------------------------------------------------------
speed = 4 # dashes/s
dt = 1 / speed
max_note_duration = 3
tuning = json.load(open(DATA_PATH / "tuning_01.json"))
open_string_midi_values = tuple(name_to_midi(note) for note in tuning)
instrument = "guitar"
sample_rate = 44100

tab_df = (
    pl.scan_csv(
        DATA_PATH / "tab_01.csv",
        has_header=False,
        new_columns=[f"string_{i}" for i in range(1, len(tuning) + 1)],
    )
    .cast(pl.String)
    .with_row_index("index")
    .select(
        pl.col("index").mul(dt).alias("start_time"),
        cs.starts_with("string"),
    )
    .collect()
)

notes_df_schema = pl.Schema({
    "id": pl.UInt32,
    "start_time": pl.Float64,
    "duration": pl.Float64,
    "string_number": pl.UInt16,
    "fret": pl.UInt16,
    "midi": pl.Int16,
    "frequency": pl.Float64,
    "instrument": pl.Enum(["guitar"]),
})
notes_df = (
    pl.concat(
        [
            tab_df.lazy()
            .select(pl.col("start_time"), pl.col(f"string_{string_num}").alias("raw"))
            .filter(pl.col("raw") != "-")
            .with_columns(pl.lit(string_num).cast(pl.UInt16).alias("string_number"))
            .with_columns(pl.col("raw").cast(pl.UInt16).alias("fret"))
            .with_columns((pl.col("fret") + open_midi).cast(pl.Int16).alias("midi"))
            .with_columns((440.0 * 2 ** ((pl.col("midi").cast(pl.Float64) - 69.0) / 12.0)).alias("frequency"))
            .with_columns((pl.col("start_time").shift(-1) - pl.col("start_time")).alias("duration"))
            .with_columns(pl.min_horizontal(pl.col("duration"), pl.lit(max_note_duration)))
            .with_columns(pl.lit(instrument).alias("instrument"))
            for string_num, open_midi
            in enumerate(open_string_midi_values, start=1)
        ],
    )
    .sort(["start_time", "string_number"])
    .with_row_index("id")
    .select(notes_df_schema.keys())
    .cast(notes_df_schema)
    .collect()
)
notes_df.write_ndjson(OUT_PATH / "notes.jsonl")


# -----------------------------------------------------------------------------
# Synthesis
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _karplus_strong_core(
    *,
    number_of_samples: int,
    samples_per_cycle: int,
    buffer: np.ndarray,
    decay: float,
) -> npt.NDArray[np.float32]:
    """Karplus-Strong core algorithm for simulating plucked strings."""
    out = np.empty(number_of_samples, dtype=np.float32)

    i0 = 0  # “read” index
    i1 = 1  # next sample index (i0+1 wrapped)

    for t in range(number_of_samples):
        x0 = buffer[i0]
        x1 = buffer[i1]
        out[t] = x0

        # write back into the slot we just read
        buffer[i0] = decay * 0.5 * (x0 + x1)

        i0 += 1
        if i0 == samples_per_cycle:
            i0 = 0
        i1 = i0 + 1
        if i1 == samples_per_cycle:
            i1 = 0
    return out


def _karplus_strong_ring(
    frequency: float,
    duration: float,
    *,
    sample_rate: int = 44100,
    decay: float = 0.996,
    base_volume: float = 0.8,
) -> npt.NDArray[np.float16]:
    """Generate a plucked string sound (numpy array) using the Karplus-Strong algorithm."""
    samples_per_cycle = int(sample_rate / frequency)
    number_of_samples = int(duration * sample_rate)
    rng = np.random.default_rng()
    buffer = (2.0 * rng.uniform(0.0, 1.0, samples_per_cycle) - 1.0).astype(np.float32)
    signal = _karplus_strong_core(
        number_of_samples=number_of_samples,
        samples_per_cycle=samples_per_cycle,
        buffer=buffer,
        decay=decay,
    )

    return signal * base_volume


def synthesize_notes(
    config,
    notes_df: pl.DataFrame,
):
    pass