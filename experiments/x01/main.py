from enum import Enum, StrEnum, auto
from functools import lru_cache
from pathlib import Path
import json
from sqlite3 import SQLITE_ERROR_MISSING_COLLSEQ
from typing import Literal
from math import log2
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
import polars.selectors as cs
from numba import njit
import numpy as np
import numpy.typing as npt
from pydub import AudioSegment
from scipy.io import wavfile

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
TAB_01_JSONL_OUT_PATH = OUT_PATH / "tab_01.jsonl"
TAB_01_WAV_OUT_PATH = OUT_PATH / "audio_01.wav"
TAB_01_MP3_OUT_PATH = OUT_PATH / "audio_01.mp3"
TAB_01_OGG_OUT_PATH = OUT_PATH / "audio_01.ogg"

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
    base_volume: float = 1.0
    
    @property
    def dt(self) -> float:
        return 1 / self.speed
    
    @property
    def midi_tuning(self) -> tuple[int, ...]:
        return tuple(name_to_midi(note) for note in self.tuning)


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------
def get_tab_df_schema(config: TabConfig):
    return pl.Schema(
        {"start_time": pl.Float64} 
        | {f"string_{i}": pl.String for i in range(1, len(config.tuning) + 1)}
    )


def csv_to_tab_df(
    in_filepath: Path,
    config: TabConfig,
    *,
    out_filepath: Path | None = None
) -> pl.DataFrame:
    tab_df_schema = get_tab_df_schema(config)
    tab_df = (
        pl.scan_csv(
            in_filepath,
            has_header=False,
            new_columns=[f"string_{i}" for i in range(1, len(config.tuning) + 1)],
        )
        .cast(pl.String)
        .with_row_index("index")
        .select(
            (pl.col("index") * config.dt).cast(pl.Float64).alias("start_time"),
            cs.starts_with("string"),
        )
        .cast(tab_df_schema)
        .collect()
    )
    
    if out_filepath is not None:
        tab_df.write_ndjson(out_filepath)
        
    return tab_df
    
    
def get_notes_df_schema():
    return pl.Schema({
        "id": pl.UInt32,
        "start_time": pl.Float64,
        "duration": pl.Float64,
        "string_number": pl.UInt16,
        "fret": pl.UInt16,
        "midi": pl.Int16,
        "frequency": pl.Float64,
    })


def tab_df_to_notes_df(
    tab_df: pl.DataFrame,
    config: TabConfig,
    *,
    out_filepath: Path | None = None
) -> pl.DataFrame:
    notes_df_schema = get_notes_df_schema()
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
                .with_columns(pl.min_horizontal(pl.col("duration"), pl.lit(config.max_note_duration)))
                for string_num, open_midi
                in enumerate(config.midi_tuning, start=1)
            ],
        )
        .sort(["start_time", "string_number"])
        .with_row_index("id")
        .select(notes_df_schema.keys())
        .cast(notes_df_schema)
        .collect()
    )

    if out_filepath is not None:
        notes_df.write_ndjson(out_filepath)
        
    return notes_df


# -----------------------------------------------------------------------------
# Synthesis
# -----------------------------------------------------------------------------
@njit(cache=True)
def _karplus_strong_core(
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
    base_volume: float = 1.0,
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
    notes_df: pl.DataFrame,
    config: TabConfig,
) -> npt.NDArray[np.float32]:
    try:
        notes_df.cast(get_notes_df_schema())
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
    
    return np.tanh(song_array)


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------
def save_numpy_as_mp3(array: npt.NDArray[np.floating], file_path: Path, sample_rate: int=44100) -> None:
    np.clip(array, -1.0, 1.0)
    audio_int16 = np.int16(array * 32767)
    
    segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_int16.dtype.itemsize,
        channels=1,
    )
    
    segment.export(str(file_path), format="mp3", bitrate="192k")

    
def save_numpy_as_ogg(array: npt.NDArray[np.floating], file_path: Path, sample_rate: int=44100) -> None:
    np.clip(array, -1.0, 1.0)
    audio_int16 = np.int16(array * 32767)
    
    segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_int16.dtype.itemsize,
        channels=1,
    )
    
    segment.export(
        file_path,
        format="ogg",
        codec="libvorbis",
        bitrate="192k",
    )
    
    
def save_numpy_as_wav(array: npt.NDArray[np.floating], file_path: Path, sample_rate: int=44100) -> None:
    array = np.clip(array, -1.0, 1.0)
    int16_array = (array * 32767).astype(np.int16)

    wavfile.write(file_path, sample_rate, int16_array)
    

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
config = TabConfig(
    speed=6.0,
    max_note_duration = 2.5,
    tuning=json.load(open(TUNING_01_JSON_DATA_PATH)),
    instrument_name="guitar",
    sample_rate=44100,
    base_volume=0.5,
)

tab_df = csv_to_tab_df(TAB_01_CSV_DATA_PATH, config, out_filepath=TAB_01_JSONL_OUT_PATH)
notes_df = tab_df_to_notes_df(tab_df, config, out_filepath=NOTES_01_JSONL_OUT_PATH)
audio_array = synthesize_notes(notes_df, config)
save_numpy_as_mp3(audio_array, TAB_01_MP3_OUT_PATH, config.sample_rate)
save_numpy_as_ogg(audio_array, TAB_01_OGG_OUT_PATH, config.sample_rate)
save_numpy_as_wav(audio_array, TAB_01_WAV_OUT_PATH, config.sample_rate)
