from pathlib import Path

import dataframely as dy
import polars as pl
import polars.selectors as cs

from exporting import save_numpy_as_ogg
from synthesis import synthesize_notes
from validation import NotesDfSchema, TabConfig, validate_notes_df, validate_tab_df
    

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
    tab_df = validate_tab_df(tab_df, config)

    tab_df_transposed = (
        tab_df.drop("String")
        .transpose(column_names=(f"string_{i}" for i in range(1, len(config.tuning) + 1)))
        .lazy()
        .with_row_index("index")
        .select(
            (pl.col("index") * config.dt).cast(pl.Float64).alias("start_time"),
            cs.starts_with("string"),
        )
    )

    notes_df = (
        pl.concat(
            [
                tab_df_transposed.lazy()
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


def tab_to_text(df: pl.DataFrame) -> str:
    """Convert the editable table into plain text guitar tab."""
    def render_frets(row: tuple[str, ...]):
        line_frets = [fret.center(3, "─") if fret else "───" for fret in row]
        return "".join(line_frets)
    
    lines = [f"{row[0]}├{render_frets(row[1:])}┤" for row in df.iter_rows()]
    return "\n".join(lines)


def tab_to_ogg(tab_df: pl.DataFrame, path: Path):
    tuning = tab_df.get_column("String").to_list()
    config = TabConfig(tuning=tuning)
    notes_df = tab_df_to_notes_df(tab_df, config, out_filepath=None)
    audio_array = synthesize_notes(notes_df, config)
    save_numpy_as_ogg(audio_array, path)