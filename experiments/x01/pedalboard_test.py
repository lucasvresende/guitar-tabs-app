from enum import Enum
from pathlib import Path

from pedalboard import Compressor, Gain, HighpassFilter, Reverb
from pedalboard._pedalboard import Pedalboard
from pedalboard.io import AudioFile
import numpy as np
import numpy.typing as npt


class Pedalboards(Enum):
    EMPTY = Pedalboard()
    ACOUSTIC = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        Compressor(
            threshold_db=-20,
            ratio=2.5,
            attack_ms=10,
            release_ms=120,
        ),
        Reverb(
            room_size=0.18,
            damping=0.55,
            wet_level=0.08,
            dry_level=0.92,
        ),
        Gain(gain_db=1.5),
    ])


def save_array(
    array: npt.NDArray[np.float32],
    sample_rate: int,
    path: Path,
    *,
    quality: int = 128,
    board: Pedalboards = Pedalboards.EMPTY,
) -> None:
    if not path.suffix.lower() in {".mp3", ".ogg"}:
        raise ValueError("File path should end with `.mp3` or `.ogg`")
    
    processed = board.value(array, sample_rate)
    peak = np.max(np.abs(processed))
    normalized = processed / peak if peak > 1.0 else processed

    num_channels = 1 if array.ndim == 1 else array.shape[0]
    with AudioFile(
        str(path),
        mode="w",
        samplerate=sample_rate,
        num_channels=num_channels,
        quality=quality,
    ) as f:
        f.write(normalized)
