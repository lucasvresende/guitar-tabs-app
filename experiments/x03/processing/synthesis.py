from numba import njit
import numpy as np
import numpy.typing as npt
import polars as pl
import dataframely as dy

from experiments.x03.processing.types import NotesDfSchema, TabConfig

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
        case "karplus-strong":
            return karplus_strong_ring(
                frequency=frequency,
                duration=duration,
                sample_rate=config.sample_rate,
                base_volume=config.base_volume,
            )
        case _:
            raise ValueError("Not implemented")


def synthesize_notes(
    notes_df: dy.DataFrame[NotesDfSchema],
    config: TabConfig,
) -> npt.NDArray[np.float32]:
    if notes_df.is_empty():
        raise ValueError("Notes df is empty")
    
    song_length: float = (
        notes_df.select(
            (pl.col("start_time") + pl.col("duration")).alias("end_time"),
        )
        .max()
        .item()
    )
    
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
