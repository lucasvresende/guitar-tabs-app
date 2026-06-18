from collections.abc import Sequence
from pathlib import Path

import streamlit as st
import polars as pl
import polars.selectors as cs

from data_pipeline import TabConfig, get_tab_df_schema, save_numpy_as_ogg, synthesize_notes, tab_df_to_notes_df

# FIXME: Validate input values in the df
# FIXME: Raise error if the audio couldn't be generated (e.g. empty dataframe, internal error)
# FIXME: Handle exceptions so they don't reach the app UI

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STANDARD_TUNING = ("E4", "B3", "G3", "D3", "A2", "E2")
NOTE_PATTERN = r"^(?:[ACDFG](?:#|b)?|[BE]b?)(?:-1|[0-9])$"
FRET_PATTERN = r"^(?:[0-9]|[12][0-9]|30)?$"

FILE_PATH = Path(__file__)
PARENT_PATH = FILE_PATH.parent
DATA_PATH = PARENT_PATH / "data"
OUT_PATH = PARENT_PATH / "out"
TAB_01_OGG_OUT_PATH = OUT_PATH / "audio_01.ogg"


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def make_empty_tab(num_beats: int = 16, tuning: Sequence[str]=STANDARD_TUNING) -> pl.DataFrame:
    """Create an empty guitar tab table."""
    string_series = pl.Series(name="String", values=tuning, dtype=pl.String)
    beats_series_list = [
        pl.Series(name=str(i), values=[None] * len(tuning), dtype=pl.String) 
        for i in range(1, num_beats + 1)
    ]
    data = [string_series] + beats_series_list
    return pl.DataFrame(data)


def tab_to_text(df: pl.DataFrame) -> str:
    """Convert the editable table into plain text guitar tab."""
    def render_frets(row: tuple[str, ...]):
        line_frets = [fret.center(3, "─") if fret else "───" for fret in row]
        return "".join(line_frets)
    
    lines = [f"{row[0]}├{render_frets(row[1:])}┤" for row in df.iter_rows()]
    return "\n".join(lines)


def save_tab_as_ogg(df: pl.DataFrame, path: Path):
    tuning = df.get_column("String").to_list()
    config = TabConfig(tuning=tuning)
    tab_df_schema = get_tab_df_schema(config)
    tab_df = (
        df.drop("String")
        .transpose(column_names=(f"string_{i}" for i in range(1, len(config.tuning) + 1)))
        .with_row_index("index")
        .select(
            (pl.col("index") * config.dt).cast(pl.Float64).alias("start_time"),
            cs.starts_with("string"),
        )
        .cast(tab_df_schema)
    )
    print(tab_df)
    notes_df = tab_df_to_notes_df(tab_df, config, out_filepath=None)
    audio_array = synthesize_notes(notes_df, config)
    save_numpy_as_ogg(audio_array, path)
    

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Guitar Tab Editor", page_icon="🎸", layout="wide")

st.title("Guitar Tab Editor")
st.caption("Create, edit, and play guitar tabs.")

if "tab_df" not in st.session_state:
    st.session_state.tab_df = make_empty_tab()

with st.sidebar:
    st.header("Settings")

    beats =  st.number_input(
        "Number of beat columns",
        min_value=1,
        max_value=64,
        value=16,
        step=1,
    )

    if st.button("New blank tab"):
        st.session_state.tab_df = make_empty_tab(beats)

st.subheader("Edit tab")


edited_df = pl.from_pandas(
    st.data_editor(
        st.session_state.tab_df.to_pandas(),
        hide_index=True,
        num_rows="fixed",
        on_change=None,
        placeholder="-",
        column_config={
            "String": st.column_config.TextColumn(
                "String",
                disabled=True,
                width="small",
                default="E2",
                validate=NOTE_PATTERN,
            ),
            **{
                col: st.column_config.TextColumn(validate=FRET_PATTERN) 
                for col in st.session_state.tab_df.columns 
                if col != "String"
            },
        },
    ),
)

# st.session_state.tab_df = edited_df

st.subheader("Plain-text tab preview")
tab_text = tab_to_text(edited_df)
st.code(tab_text, language="text")

st.download_button(
    "Download tab as .txt",
    data=tab_text,
    file_name="guitar_tab.txt",
    mime="text/plain",
)

if st.button("Generate audio"):
    ogg_path = TAB_01_OGG_OUT_PATH
    save_tab_as_ogg(edited_df, path=ogg_path)
    st.audio(
        data=open(ogg_path, 'rb').read(),
        format='audio/ogg',
    )
