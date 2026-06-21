"""Validation schemas and configuration models for guitar tabs."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

import dataframely as dy
import polars as pl
from note_conversions import name_to_midi
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STANDARD_TUNING = ("E4", "B3", "G3", "D3", "A2", "E2")


class RegexPatterns(StrEnum):
    """Regular expression patterns used by UI and dataframe validators."""

    NOTE_PATTERN = r"^(?:Ab|A|A#|Bb|B|C|C#|Db|D|D#|Eb|E|F|F#|Gb|G|G#)(?:-1|[0-9])$"
    FRET_PATTERN = r"^(?:[0-9]|[1-2][0-9]|[30])$"


# -----------------------------------------------------------------------------
# Pydantic
# -----------------------------------------------------------------------------
type SynthAlgorithms = Literal["karplus-strong"]


class TabConfig(BaseModel, frozen=True):
    """Tab editing and audio synthesis configuration."""

    speed: PositiveFloat = 4.0  # dashes/s
    max_note_duration: PositiveFloat = 2.0
    tuning: tuple[str, ...] = STANDARD_TUNING
    number_of_beats: PositiveInt = 16
    instrument_name: str = "guitar"
    sample_rate: PositiveInt = 44100
    synthesis_algorithm: SynthAlgorithms = "karplus-strong"
    base_volume: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0

    @property
    def dt(self) -> float:
        """Beat interval in seconds."""
        return 1 / self.speed

    @property
    def midi_tuning(self) -> tuple[int, ...]:
        """Open-string tuning converted to MIDI note numbers."""
        return tuple(name_to_midi(note) for note in self.tuning)

    @property
    def number_of_strings(self) -> int:
        """Number of strings defined by the tuning."""
        return len(self.tuning)


# -----------------------------------------------------------------------------
# Dataframely
# -----------------------------------------------------------------------------
class NotesDfSchema(dy.Schema):
    """Schema for note events produced from a tab dataframe."""

    id = dy.UInt32(primary_key=True)
    start_time = dy.Float64(min=0)
    duration = dy.Float64(min_exclusive=0)
    string_number = dy.UInt16(min=1)
    fret = dy.UInt16(min=0)
    midi = dy.Int16()
    frequency = dy.Float64(min_exclusive=0)


def validate_notes_df(df: pl.DataFrame, config: TabConfig) -> dy.DataFrame[NotesDfSchema]:
    """Validate note events against static and config-dependent rules.

    Args:
        df: Notes dataframe to validate.
        config: Tab configuration that defines string and duration limits.

    Returns:
        Validated and cast notes dataframe.

    Raises:
        ValueError: If the dataframe violates the notes schema or config-dependent rules.

    """

    class NotesDfSchemaWithExtraRules(NotesDfSchema):
        @dy.rule()
        def validate_max_string_number(self) -> pl.Expr:
            del self
            return pl.col("string_number") <= config.number_of_strings

        @dy.rule()
        def validate_max_note_duration(self) -> pl.Expr:
            del self
            return pl.col("duration") <= config.max_note_duration

    return NotesDfSchemaWithExtraRules.validate(df, cast=True)


def make_tab_schema(config: TabConfig) -> type[dy.Schema]:
    """Create a tab dataframe schema for the configured string count.

    Args:
        config: Tab configuration that defines the string columns.

    Returns:
        Dataframely schema class for tab rows.

    """
    start_time_col = {"start_time": dy.Float64(min=0)}
    string_cols = {
        f"string_{i}": dy.String(nullable=True, regex=RegexPatterns.FRET_PATTERN)
        for i in range(1, len(config.tuning) + 1)
    }
    attrs = start_time_col | string_cols

    return type("TabDfSchema", (dy.Schema,), attrs)


def validate_tab_df(df: pl.DataFrame, config: TabConfig) -> dy.DataFrame:
    """Validate a tab dataframe.

    Raises:
        ValueError: If the dataframe violates the generated tab schema.

    """
    tab_df_schema = make_tab_schema(config)
    return tab_df_schema.validate(df)


# -----------------------------------------------------------------------------
# Custom
# -----------------------------------------------------------------------------
def validate_ui_df(ui_df: pl.DataFrame, config: TabConfig) -> pl.DataFrame:
    """Validate the editable UI dataframe shape.

    Args:
        ui_df: Editable UI dataframe to validate.
        config: Tab configuration that defines expected beat columns.

    Returns:
        The input dataframe when validation succeeds.

    Raises:
        ValueError: If required UI dataframe columns are missing or invalid.

    """
    if "String" not in ui_df.columns:
        raise ValueError("`String` column missing from ui_df")

    if not all(f"{i}" for i in range(1, config.number_of_beats)):
        raise ValueError("Beats columns do not match the number of beats set in config")

    return ui_df
