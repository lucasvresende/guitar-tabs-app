from collections.abc import Sequence
from typing import Literal

import dataframely as dy
import polars as pl
from pydantic import BaseModel

from experiments.x03.processing.note_conversions import name_to_midi


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STANDARD_TUNING = ("E4", "B3", "G3", "D3", "A2", "E2")
NOTE_PATTERN = r"^(?:[ACDFG](?:#|b)?|[BE]b?)(?:-1|[0-9])$"
FRET_PATTERN = r"^(?:[0-9]|[12][0-9]|30)?$"

type SynthAlgorithms = Literal["karplus-strong"]


# -----------------------------------------------------------------------------
# Pydantic
# -----------------------------------------------------------------------------
class TabConfig(BaseModel, frozen=True):
    speed: float = 4.0 # dashes/s
    max_note_duration:float = 2.0
    tuning: Sequence[str] = ("E4", "B3", "G3", "D3", "A2", "E2")
    number_of_beats: int = 16
    max_fret: int = 30
    instrument_name: str = "guitar"
    sample_rate: int = 44100
    synthesis_algorithm: SynthAlgorithms = "karplus-strong"
    base_volume: float = 1.0
    
    @property
    def dt(self) -> float:
        return 1 / self.speed
    
    @property
    def midi_tuning(self) -> tuple[int, ...]:
        return tuple(name_to_midi(note) for note in self.tuning)
    
    @property
    def number_of_strings(self) -> int:
        return len(self.tuning)
    
    
# -----------------------------------------------------------------------------
# Dataframely
# -----------------------------------------------------------------------------
class NotesDfSchema(dy.Schema):
    id = dy.UInt32(primary_key=True)
    start_time = dy.Float64(min=0)
    duration = dy.Float64(min_exclusive=0)
    string_number = dy.UInt16(min=1)
    fret = dy.UInt16(min=1)
    midi = dy.Int16(min=-1, max=9)
    frequency = dy.Float64(min_exclusive=0)
    
    
def make_tab_schema(df: pl.DataFrame, config: TabConfig) -> type[dy.Schema]:
    start_time_col = {"start_time": dy.Float64(min=0)}
    string_cols = {
        f"string_{i}": dy.String(nullable=True, regex=FRET_PATTERN) 
        for i in range(1, len(config.tuning) + 1)
    }
    attrs = start_time_col | string_cols

    return type("TabDfSchema", (dy.Schema,), attrs)