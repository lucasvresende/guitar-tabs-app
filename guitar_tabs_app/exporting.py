"""Audio export helpers for synthesized tab audio."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from pydub import AudioSegment
from scipy.io import wavfile

if TYPE_CHECKING:
    from pathlib import Path

# NOTE: save_numpy_as_mp3 and save_numpy_as_ogg require ffmpeg


def save_numpy_as_mp3(
    array: npt.NDArray[np.floating],
    file_path: Path,
    sample_rate: int = 44100,
) -> None:
    """Save a mono floating-point audio array as an MP3 file."""
    np.clip(array, -1.0, 1.0)
    audio_int16 = np.int16(array * 32767)

    segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_int16.dtype.itemsize,
        channels=1,
    )

    segment.export(str(file_path), format="mp3", bitrate="192k")


def save_numpy_as_ogg(
    array: npt.NDArray[np.floating],
    file_path: Path,
    sample_rate: int = 44100,
) -> None:
    """Save a mono floating-point audio array as an Ogg Vorbis file."""
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


def save_numpy_as_wav(
    array: npt.NDArray[np.floating],
    file_path: Path,
    sample_rate: int = 44100,
) -> None:
    """Save a mono floating-point audio array as a WAV file."""
    array = np.clip(array, -1.0, 1.0)
    int16_array = (array * 32767).astype(np.int16)

    wavfile.write(file_path, sample_rate, int16_array)
