from pathlib import Path
import tempfile

import streamlit as st

import polars as pl

from experiments.x03.transformations import make_empty_tab, tab_to_ogg, tab_to_text
from experiments.x03.validation import RegexPatterns, TabConfig

st.set_page_config(page_title="Guitar Tab Editor", page_icon="🎸", layout="wide")

st.title("Guitar Tab Editor")
st.caption("Create, edit, and play guitar tabs.")

if "tab_df" not in st.session_state:
    tab_config = TabConfig()
    tab_df = make_empty_tab(tab_config)
    st.session_state.tab_config = tab_config
    st.session_state.tab_df = tab_df

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
                validate=RegexPatterns.NOTE_PATTERN,
            ),
            **{
                col: st.column_config.TextColumn(validate=RegexPatterns.FRET_PATTERN) 
                for col in st.session_state.tab_df.columns
                if col != "String"
            },
        },
    ),
)


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
    with tempfile.TemporaryFile(suffix=".ogg", mode='w+t') as ogg_file:
        ogg_file_path = Path(ogg_file.name)
        tab_to_ogg(edited_df, path=ogg_file_path)
        st.audio(
            data=ogg_file_path.open('rb').read(),
            format='audio/ogg',
        )