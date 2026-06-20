from pathlib import Path

import numpy as np
import numpy.typing as npt

from pathlib import Path
from pydub import AudioSegment
from scipy.io import wavfile


# NOTE: save_numpy_as_mp3 and save_numpy_as_ogg require ffmpeg

def save_numpy_as_mp3(
    array: npt.NDArray[np.floating], 
    file_path: Path,
    sample_rate: int=44100
) -> None:
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
    sample_rate: int=44100
) -> None:
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
    sample_rate: int=44100
) -> None:
    array = np.clip(array, -1.0, 1.0)
    int16_array = (array * 32767).astype(np.int16)

    wavfile.write(file_path, sample_rate, int16_array)
