"""Streamlit application for editing, previewing, and exporting guitar tabs."""

import tempfile
from pathlib import Path

import polars as pl
import streamlit as st
from exporting import save_numpy_as_ogg
from transformations import (
    check_ui_df_not_filled,
    make_empty_tab,
    tab_to_text,
    ui_df_to_numpy_audio,
)
from validation import RegexPatterns, TabConfig, validate_ui_df

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Guitar Tab Editor", page_icon="🎸", layout="wide")

if "tab_config" not in st.session_state:
    st.session_state.tab_config = TabConfig()

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
st.title("Guitar Tab Editor")
st.caption("Create, edit, and play guitar tabs.")

with st.container(border=True):
    st.header("Tab")

    # Tab data editor
    empty_df = make_empty_tab(st.session_state.tab_config)
    ui_df = st.data_editor(
        empty_df,
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
                    validate=RegexPatterns.FRET_PATTERN,
                    width=50,
                )
                for col in empty_df.columns
                if col != "String"
            },
        },
    )
    ui_df = validate_ui_df(
        ui_df=pl.DataFrame(ui_df),
        config=st.session_state.tab_config,
    )

    with st.expander("⚙️ Config"):
        col_1, col_2 = st.columns(2)

        with col_1:
            with st.container(border=True):
                speed = st.slider(
                    "Speed",
                    min_value=1.0,
                    max_value=20.0,
                    value=st.session_state.tab_config.speed,
                    step=0.5,
                )

            with st.container(border=True):
                default_number_of_beats = st.session_state.tab_config.number_of_beats
                number_of_beats = st.number_input(
                    "Number of beats",
                    min_value=1,
                    max_value=100,
                    value=default_number_of_beats,
                )

        with col_2, st.container(border=True):
            st.text("Tuning")
            default_number_of_strings = st.session_state.tab_config.number_of_strings
            default_tuning = st.session_state.tab_config.tuning
            number_of_strings = st.number_input(
                "Number of strings",
                min_value=1,
                max_value=12,
                value=default_number_of_strings,
            )
            tuning_df = st.data_editor(
                data=pl.DataFrame({
                    str(i + 1): default_tuning[i % default_number_of_strings]
                    for i in range(number_of_strings)
                }),
                hide_index=True,
                column_config={
                    col: st.column_config.SelectboxColumn(
                        options=[
                            f"{note}{octave}"
                            for note in [
                                "A",
                                "A#",
                                "B",
                                "C",
                                "C#",
                                "D",
                                "D#",
                                "E",
                                "F",
                                "F#",
                                "G",
                                "G#",
                            ]
                            for octave in [1, 2, 3, 4, 5]
                        ],
                        required=True,
                    )
                    for col in ui_df.columns
                },
            )
            if not isinstance(tuning_df, pl.DataFrame):
                raise TypeError("Expected Streamlit tuning editor to return a Polars DataFrame")
            tuning = tuple(tuning_df.row(0))

        def update_tab_config() -> None:
            """Update the tab configuration from the current form values."""
            st.session_state.tab_config = TabConfig(
                speed=speed,
                number_of_beats=number_of_beats,
                tuning=tuning,
            )

        st.button("Apply", on_click=update_tab_config)

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

            st.audio(audio_data, format="audio/ogg")
            st.download_button(
                "⤓ Download audio as .ogg",
                data=audio_data,
                file_name="tab.ogg",
                mime="audio/ogg",
            )
