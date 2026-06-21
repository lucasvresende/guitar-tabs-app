"""Data transformations between UI tables, tab rows, notes, text, and audio."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs
from synthesis import synthesize_notes
from validation import NotesDfSchema, TabConfig, validate_notes_df, validate_tab_df

if TYPE_CHECKING:
    from pathlib import Path

    import dataframely as dy
    import numpy as np
    import numpy.typing as npt


def make_empty_tab(config: TabConfig) -> pl.DataFrame:
    """Create an empty editable tab dataframe."""
    string_series = pl.Series(name="String", values=config.tuning, dtype=pl.String)
    beats_series_list = [
        pl.Series(name=str(i), values=[None] * config.number_of_strings, dtype=pl.String)
        for i in range(1, config.number_of_beats + 1)
    ]
    data = [string_series, *beats_series_list]
    return pl.DataFrame(data)


def check_ui_df_not_filled(ui_df: pl.DataFrame) -> bool:
    """Check whether the editable tab has no fret entries."""
    return ui_df.select(
        pl.all_horizontal(
            pl.all().exclude("String").is_null().all(),
        ),
    ).item()


def ui_df_to_tab_df(
    ui_df: pl.DataFrame,
    config: TabConfig,
    *,
    out_jsonl_filepath: Path | None = None,
) -> dy.DataFrame:
    """Convert the editable UI dataframe to a validated tab dataframe.

    Args:
        ui_df: Editable UI dataframe with string labels and beat columns.
        config: Tab configuration that defines timing and strings.
        out_jsonl_filepath: Optional path where the tab dataframe is written as JSONL.

    Returns:
        Validated tab dataframe with start times and string columns.

    """
    tab_df = (
        ui_df
        .drop("String")
        .transpose(column_names=(f"string_{i}" for i in range(1, config.number_of_strings + 1)))
        .lazy()
        .with_row_index("index")
        .select(
            (pl.col("index") * config.dt).cast(pl.Float64).alias("start_time"),
            cs.starts_with("string"),
        )
        .collect()
    )
    tab_df = validate_tab_df(tab_df, config)

    if out_jsonl_filepath is not None:
        tab_df.write_ndjson(out_jsonl_filepath)

    return tab_df


def tab_df_to_notes_df(
    tab_df: pl.DataFrame,
    config: TabConfig,
    *,
    out_jsonl_filepath: Path | None = None,
) -> dy.DataFrame[NotesDfSchema]:
    """Convert a validated tab dataframe to note events.

    Args:
        tab_df: Tab dataframe with start times and string fret columns.
        config: Tab configuration that defines tuning and note duration limits.
        out_jsonl_filepath: Optional path where note events are written as JSONL.

    Returns:
        Validated notes dataframe ready for synthesis.

    """
    notes_df = (
        pl
        .concat(
            [
                tab_df
                .lazy()
                .select(pl.col("start_time"), pl.col(f"string_{string_num}").alias("raw"))
                .filter(pl.col("raw").is_not_null(), ~pl.col("raw").str.contains(r"^[ -]*$"))
                .with_columns(pl.lit(string_num).cast(pl.UInt16).alias("string_number"))
                .with_columns(pl.col("raw").cast(pl.UInt16).alias("fret"))
                .with_columns((pl.col("fret") + open_midi).cast(pl.Int16).alias("midi"))
                .with_columns(
                    (440.0 * 2 ** ((pl.col("midi").cast(pl.Float64) - 69.0) / 12.0)).alias(
                        "frequency",
                    ),
                )
                .with_columns(
                    (pl.col("start_time").shift(-1) - pl.col("start_time")).alias("duration"),
                )
                .with_columns(
                    pl.min_horizontal(pl.col("duration"), pl.lit(config.max_note_duration)),
                )
                for string_num, open_midi in enumerate(config.midi_tuning, start=1)
            ],
        )
        .sort(["start_time", "string_number"])
        .with_row_index("id")
        .select(NotesDfSchema.column_names())
        .collect()
    )
    notes_df = validate_notes_df(notes_df, config)

    if out_jsonl_filepath is not None:
        notes_df.write_ndjson(out_jsonl_filepath)

    return notes_df


def tab_to_text(ui_df: pl.DataFrame) -> str:
    """Convert the editable table into plain text guitar tab."""

    def render_frets(row: tuple[str, ...]) -> str:
        line_frets = [fret.center(3, "─") if fret else "───" for fret in row]
        return "".join(line_frets)

    lines = [f"{row[0]}├{render_frets(row[1:])}┤" for row in ui_df.iter_rows()]
    return "\n".join(lines)


def ui_df_to_numpy_audio(ui_df: pl.DataFrame, config: TabConfig) -> npt.NDArray[np.float32]:
    """Convert the editable UI dataframe directly to synthesized audio."""
    tab_df = ui_df_to_tab_df(ui_df, config)
    notes_df = tab_df_to_notes_df(tab_df, config)
    return synthesize_notes(notes_df, config)
