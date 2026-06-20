from pathlib import Path
import tempfile

import streamlit as st
import polars as pl

from transformations import make_empty_tab, tab_to_ogg, tab_to_text
from validation import RegexPatterns, TabConfig

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Guitar Tab Editor", page_icon="🎸", layout="wide")

if "tab_config" not in st.session_state:
    st.session_state.tab_config = TabConfig()
    st.session_state.tab_df = make_empty_tab(st.session_state.tab_config)

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
st.title("Guitar Tab Editor")
st.caption("Create, edit, and play guitar tabs.")

with st.container(border=True):
    st.header("Tab")
    
    # Tab data editor
    
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

    
    with st.expander("⚙️ Config"):
        pass
    
    with st.expander("📔 Text Preview"):
        tab_text = tab_to_text(edited_df)
        st.code(tab_text, language="text")
        st.download_button(
            "⤓ Download tab as .txt",
            data=tab_text,
            file_name="tab.txt",
            mime="text/plain",
        )
    
    with st.expander("▶️ Audio Preview"):
        with tempfile.TemporaryDirectory() as tmpdir:
            ogg_file_path = Path(tmpdir) / "audio.ogg"
            tab_to_ogg(edited_df, path=ogg_file_path)
            audio_data = ogg_file_path.read_bytes()

        st.audio(
            data=audio_data,
            format='audio/ogg',
        )
        st.download_button(
            "⤓ Download audio as .ogg",
            data=audio_data,
            file_name="tab.ogg",
            mime="audio/ogg",
        )
        
    

        
    
    
    
    
    
    
    
        