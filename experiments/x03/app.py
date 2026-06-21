from pathlib import Path
import tempfile

import streamlit as st
import polars as pl

from exporting import save_numpy_as_ogg
from transformations import check_ui_df_not_filled, make_empty_tab, tab_to_text, ui_df_to_numpy_audio, validate_ui_df
from validation import RegexPatterns, TabConfig

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Guitar Tab Editor", page_icon="🎸", layout="wide")

if "tab_config" not in st.session_state:
    st.session_state.tab_config = TabConfig()

if "ui_df" not in st.session_state:
    st.session_state.ui_df = make_empty_tab(st.session_state.tab_config)

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
st.title("Guitar Tab Editor")
st.caption("Create, edit, and play guitar tabs.")

with st.container(border=True):
    st.header("Tab")
    
    # Tab data editor
    pd_ui_df = st.data_editor(
        st.session_state.ui_df.to_pandas(),
        hide_index=True,
        num_rows="fixed",
        on_change=None,
        placeholder="-",
        width="content",
        column_config={
            "String": st.column_config.TextColumn(
                "String",
                disabled=True,
                width=50,
                default="E2",
                validate=RegexPatterns.NOTE_PATTERN,
            ),
            **{
                col: st.column_config.TextColumn(
                    validate=RegexPatterns.FRET_PATTERN, # TODO: Adapt to config max frets
                    width=50,
                )
                for col in st.session_state.ui_df.columns
                if col != "String"
            },
        },
    )
    ui_df = validate_ui_df(
        ui_df=pl.from_pandas(pd_ui_df), 
        config=st.session_state.tab_config,
    )

    with st.expander("⚙️ Config"):
        pass
    
    with st.expander("📔 Text Preview"):
        tab_text = tab_to_text(ui_df)
        st.code(tab_text, language="text")
        st.download_button(
            "⤓ Download tab as .txt",
            data=tab_text,
            file_name="tab.txt",
            mime="text/plain",
        )
    
    with st.expander("▶️ Audio Preview"):   
        if check_ui_df_not_filled(ui_df):
            st.text("The tab has no notes.")
        else:
            audio_array = ui_df_to_numpy_audio(ui_df, st.session_state.tab_config)
            with tempfile.TemporaryDirectory() as tmpdir:
                ogg_file_path = Path(tmpdir) / "audio.ogg"
                save_numpy_as_ogg(audio_array, Path(ogg_file_path))
                audio_data = ogg_file_path.read_bytes()
                
            st.audio(audio_data, format='audio/ogg')
            st.download_button(
                "⤓ Download audio as .ogg",
                data=audio_data,
                file_name="tab.ogg",
                mime="audio/ogg",
            )
            
    

        
    
    
    
    
    
    
    
        