from enum import Enum, StrEnum, auto
from functools import lru_cache
from pathlib import Path
import json
from sqlite3 import SQLITE_ERROR_MISSING_COLLSEQ
from typing import Literal
from math import log2
from collections.abc import Sequence

from dataclasses import dataclass
# from pedalboard import Compressor, Gain, HighpassFilter, Reverb
# from pedalboard._pedalboard import Pedalboard
# from pedalboard.io import AudioFile
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

TAB_01_CSV_DATA_PATH = DATA_PATH / "tab_01.csv"
TUNING_01_JSON_DATA_PATH = DATA_PATH / "tuning_01.json"
NOTES_01_JSONL_OUT_PATH = OUT_PATH / "notes_01.jsonl"
TAB_01_WAV_OUT_PATH = OUT_PATH / "audio_01.wav"
TAB_01_MP3_OUT_PATH = OUT_PATH / "audio_01.mp3"

# -----------------------------------------------------------------------------
# Note Conversions
# -----------------------------------------------------------------------------
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
    return 69 + 12 * log2(freq / 440.0)

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------
class SynthAlgorithms(StrEnum):
    KARPLUS_STRONG = auto()


@dataclass(frozen=True, slots=True)
class TabConfig:
    speed: float = 4.0 # dashes/s
    max_note_duration:float = 3.0
    tuning: tuple[str, ...] = ("E4", "B3", "G3", "D3", "A2", "E2")
    instrument_name: str = "guitar"
    sample_rate: int = 44100
    synthesis_algorithm: SynthAlgorithms = SynthAlgorithms.KARPLUS_STRONG
    base_volume: float = 0.5
    
    @property
    def dt(self) -> float:
        return 1 / self.speed
    
    @property
    def midi_tuning(self) -> tuple[int, ...]:
        return tuple(name_to_midi(note) for note in self.tuning)
    
    
# class Pedalboards(Enum):
#     EMPTY = Pedalboard()
#     ACOUSTIC = Pedalboard([
#         HighpassFilter(cutoff_frequency_hz=80),
#         Compressor(
#             threshold_db=-20,
#             ratio=2.5,
#             attack_ms=10,
#             release_ms=120,
#         ),
#         Reverb(
#             room_size=0.18,
#             damping=0.55,
#             wet_level=0.08,
#             dry_level=0.92,
#         ),
#         Gain(gain_db=1.5),
#     ])


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------
cfg = TabConfig(
    speed=4,
    max_note_duration = 3,
    tuning=json.load(open(TUNING_01_JSON_DATA_PATH)),
    instrument_name="guitar",
    sample_rate=44100,
)

tab_df = (
    pl.scan_csv(
        TAB_01_CSV_DATA_PATH,
        has_header=False,
        new_columns=[f"string_{i}" for i in range(1, len(cfg.tuning) + 1)],
    )
    .cast(pl.String)
    .with_row_index("index")
    .select(
        (pl.col("index") * cfg.dt).alias("start_time"),
        cs.starts_with("string"),
    )
    .collect()
)

NOTES_DF_SCHEMA = pl.Schema({
    "id": pl.UInt32,
    "start_time": pl.Float64,
    "duration": pl.Float64,
    "string_number": pl.UInt16,
    "fret": pl.UInt16,
    "midi": pl.Int16,
    "frequency": pl.Float64,
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
            .with_columns(pl.min_horizontal(pl.col("duration"), pl.lit(cfg.max_note_duration)))
            for string_num, open_midi
            in enumerate(cfg.midi_tuning, start=1)
        ],
    )
    .sort(["start_time", "string_number"])
    .with_row_index("id")
    .select(NOTES_DF_SCHEMA.keys())
    .cast(NOTES_DF_SCHEMA)
    .collect()
)
notes_df.write_ndjson(NOTES_01_JSONL_OUT_PATH)


# -----------------------------------------------------------------------------
# Synthesis
# -----------------------------------------------------------------------------
@njit(cache=True)
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


def karplus_strong_ring(
    frequency: float,
    duration: float,
    *,
    sample_rate: int = 44100,
    decay: float = 0.996,
    base_volume: float = 1,
) -> npt.NDArray[np.float32]:
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


def synthesize_note(
    config: TabConfig,
    frequency: float,
    duration: float,
) -> npt.NDArray[np.float32]:
    match config.synthesis_algorithm:
        case SynthAlgorithms.KARPLUS_STRONG:
            return karplus_strong_ring(
                frequency=frequency,
                duration=duration,
                sample_rate=config.sample_rate,
                base_volume=config.base_volume,
            )
        case _:
            raise ValueError("Note implemented")


def synthesize_notes(
    config: TabConfig,
    notes_df: pl.DataFrame,
) -> npt.NDArray[np.float32]:
    try:
        notes_df.cast(NOTES_DF_SCHEMA)
    except Exception as e:
        raise ValueError(f"Invalid notes_df: {e}")
    
    song_length: float = notes_df.select(
        (pl.col("start_time") + pl.col("duration")).alias("end_time"),
    ).max().item()
    
    song_array = np.zeros(int(song_length * config.sample_rate), dtype=np.float32)
    for note in notes_df.iter_rows(named=True):
        synthesized = synthesize_note(
            config=config,
            frequency=note["frequency"],
            duration=note["duration"],
        )
        start_idx = int(note["start_time"] * config.sample_rate)
        song_array[start_idx : start_idx + len(synthesized)] += synthesized
    
    peak = np.max(np.abs(song_array))
    return song_array / peak if peak > 1.0 else song_array


# -----------------------------------------------------------------------------
# Conversion
# -----------------------------------------------------------------------------
# def save_array(
#     array: npt.NDArray[np.float32],
#     sample_rate: int,
#     path: Path,
#     *,
#     quality: int = 128,
#     board: Pedalboards = Pedalboards.EMPTY,
# ) -> None:
#     if not path.suffix.lower() in {".mp3", ".ogg"}:
#         raise ValueError("File path should end with `.mp3` or `.ogg`")
    
#     processed = board.value(array, sample_rate)
#     peak = np.max(np.abs(processed))
#     normalized = processed / peak if peak > 1.0 else processed

#     num_channels = 1 if array.ndim == 1 else array.shape[0]
#     with AudioFile(
#         str(path),
#         mode="w",
#         samplerate=sample_rate,
#         num_channels=num_channels,
#         quality=quality,
#     ) as f:
#         f.write(normalized)

