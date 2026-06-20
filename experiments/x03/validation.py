from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal

import dataframely as dy
import polars as pl
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from note_conversions import name_to_midi


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STANDARD_TUNING = ("E4", "B3", "G3", "D3", "A2", "E2")

class RegexPatterns(StrEnum):
    NOTE_PATTERN = r"^(?:Ab|A|A#|Bb|B|C|C#|Db|D|D#|Eb|E|F|F#|Gb|G|G#)(?:-1|[0-9])$"
    FRET_PATTERN = r"^(?:[0-9]|[1-9][0-9])$"


# -----------------------------------------------------------------------------
# Pydantic
# -----------------------------------------------------------------------------
type SynthAlgorithms = Literal["karplus-strong"]


class TabConfig(BaseModel, frozen=True):
    speed: PositiveFloat = 4.0 # dashes/s
    max_note_duration: PositiveFloat = 2.0
    tuning: Sequence[str] = STANDARD_TUNING
    number_of_beats: PositiveInt = 16
    max_fret: PositiveInt = 30
    instrument_name: str = "guitar"
    sample_rate: PositiveInt = 44100
    synthesis_algorithm: SynthAlgorithms = "karplus-strong"
    base_volume: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    
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
    fret = dy.UInt16(min=0)
    midi = dy.Int16(min=-1, max=9)
    frequency = dy.Float64(min_exclusive=0)


def validate_notes_df(df: pl.DataFrame, config: TabConfig) -> dy.DataFrame[NotesDfSchema]:
    class NotesDfSchemaWithExtraRules(NotesDfSchema):
        @dy.rule()
        def validate_max_string_number(cls) -> pl.Expr:
            return pl.col("string_number") <= config.number_of_strings
        
        @dy.rule()
        def validate_max_fret(cls) -> pl.Expr:
            return pl.col("fret") <= config.number_of_strings
        
        @dy.rule()
        def validate_max_note_duration(cls) -> pl.Expr:
            return pl.col("duration") <= config.max_note_duration
        
    return NotesDfSchemaWithExtraRules.validate(df, cast=True)


def make_tab_schema(config: TabConfig) -> type[dy.Schema]:
    start_time_col = {"start_time": dy.Float64(min=0)}
    string_cols = {
        f"string_{i}": dy.String(nullable=True, regex=RegexPatterns.FRET_PATTERN) 
        for i in range(1, len(config.tuning) + 1)
    }
    attrs = start_time_col | string_cols

    return type("TabDfSchema", (dy.Schema,), attrs)


def validate_tab_df(df: pl.DataFrame, config: TabConfig) -> dy.DataFrame:
    TabDfSchema = make_tab_schema(config)
    return TabDfSchema.validate(df)
    


