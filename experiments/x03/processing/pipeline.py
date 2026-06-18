from pathlib import Path

import dataframely as dy
import polars as pl

from experiments.x03.processing.types import NotesDfSchema, TabConfig


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------   
def _validate_notes_df(df: pl.DataFrame, config: TabConfig) -> pl.DataFrame:
    schema = pl.Schema({
        "id": pl.UInt32,
        "start_time": pl.Float64,
        "duration": pl.Float64,
        "string_number": pl.UInt16,
        "fret": pl.UInt16,
        "midi": pl.Int16,
        "frequency": pl.Float64,
    })
    
    df = df.cast(schema)
    
    if df.null_count().sum_horizontal().item() > 0:
        raise ValueError("`notes_df` must not contain null values")

    if df.select(pl.col("start_time").lt(0).any()).item():
        raise ValueError("`start_time` column must not contain negative values.")
    
    if df.select(pl.col("duration").lt(0).any()).item():
        raise ValueError("`duration` column must not contain negative values.")
    
    if df.select(~pl.col("string_number").is_between(1, config.number_of_strings).any()).item():
        msg = f"`string_number` column must be between 0 and {config.number_of_strings}."
        raise ValueError(msg)
    
    if df.select(~pl.col("fret").is_between(1, config.max_fret).any()).item():
        msg = f"`fret` column must be between 0 and {config.max_fret}."
        raise ValueError(msg)
    
    if df.select(pl.col("frequency").lt(0).any()).item():
        raise ValueError("`frequency` column must not contain negative values.")
    
    return df


def validate_notes_df(df: pl.DataFrame, config: TabConfig) -> dy.DataFrame[NotesDfSchema]:
    validated_df = NotesDfSchema.validate(df, cast=True)
    
    if validated_df.select(~pl.col("string_number").is_between(1, config.number_of_strings).any()).item():
        msg = f"`string_number` column must be between 0 and {config.number_of_strings}."
        raise ValueError(msg)
    
    if validated_df.select(~pl.col("fret").is_between(1, config.max_fret).any()).item():
        msg = f"`fret` column must be between 0 and {config.max_fret}."
        raise ValueError(msg)
    
    return validated_df
    

# -----------------------------------------------------------------------------
# Logic
# -----------------------------------------------------------------------------
def make_empty_tab(config: TabConfig) -> pl.DataFrame:
    """Create an empty guitar tab table."""
    string_series = pl.Series(name="String", values=config.tuning, dtype=pl.String)
    beats_series_list = [
        pl.Series(name=str(i), values=[None] * config.number_of_strings, dtype=pl.String) 
        for i in range(1, config.number_of_beats + 1)
    ]
    data = [string_series] + beats_series_list
    return pl.DataFrame(data)


def tab_df_to_notes_df(
    tab_df: pl.DataFrame,
    config: TabConfig,
    *,
    out_filepath: Path | None = None
) -> dy.DataFrame[NotesDfSchema]:
    notes_df = (
        pl.concat(
            [
                tab_df.lazy()
                .select(pl.col("start_time"), pl.col(f"string_{string_num}").alias("raw"))
                .filter(pl.col("raw").is_not_null(), ~pl.col("raw").str.contains(r"^[ -]*$"))
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
        .select(NotesDfSchema.column_names())
        .collect()
    )

    if out_filepath is not None:
        notes_df.write_ndjson(out_filepath)
    
    return validate_notes_df(notes_df, config)