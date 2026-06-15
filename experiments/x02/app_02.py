"""
Streamlit Guitar Tab Editor

Run:
    pip install streamlit pandas
    streamlit run app.py

This app uses:
- st.data_editor for editable tab measures
- st.dataframe for read-only preview tables
- st.column_config for table column labels, widths, help text, and validation
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pandas as pd
import streamlit as st


STRINGS_STANDARD = ["e", "B", "G", "D", "A", "E"]
CELL_PATTERN = r"^[-0-9hHpPbBrR/\\~xXmMsS().| ]*$"


@dataclass(frozen=True)
class TabSettings:
    title: str
    artist: str
    tuning: list[str]
    beats_per_measure: int
    measures: int


def make_empty_tab(tuning: list[str], measures: int, beats_per_measure: int) -> pd.DataFrame:
    """Create a blank tab grid with one row per string and one column per measure."""
    cols = [f"Measure {i}" for i in range(1, measures + 1)]
    width_hint = "-" * max(4, beats_per_measure * 4)
    data = {
        "String": tuning,
        **{col: [width_hint for _ in tuning] for col in cols},
    }
    return pd.DataFrame(data)


def normalize_tab_df(df: pd.DataFrame, tuning: list[str], measures: int, beats_per_measure: int) -> pd.DataFrame:
    """Keep uploaded/imported/session data aligned to the current tuning and measure count."""
    measure_cols = [f"Measure {i}" for i in range(1, measures + 1)]
    width_hint = "-" * max(4, beats_per_measure * 4)

    normalized = pd.DataFrame({"String": tuning})
    for col in measure_cols:
        if col in df.columns:
            values = df[col].astype(str).tolist()[: len(tuning)]
            values += [width_hint] * (len(tuning) - len(values))
            normalized[col] = values
        else:
            normalized[col] = [width_hint for _ in tuning]

    return normalized


def validate_tab_cells(df: pd.DataFrame) -> list[str]:
    """Return validation issues for editable tab cells."""
    issues: list[str] = []
    measure_cols = [c for c in df.columns if c.startswith("Measure ")]

    for row_idx, row in df.iterrows():
        string_name = row["String"]
        for col in measure_cols:
            value = "" if pd.isna(row[col]) else str(row[col])
            if not re.match(CELL_PATTERN, value):
                issues.append(
                    f"{string_name}, {col}: contains unsupported characters. "
                    "Allowed: frets, hyphens, spaces, h, p, b, r, /, \\, ~, x, m, s, parentheses, dots, and |."
                )

    return issues


def render_ascii_tab(settings: TabSettings, df: pd.DataFrame) -> str:
    """Convert the table into common monospaced ASCII guitar tab."""
    title_lines = []
    if settings.title:
        title_lines.append(settings.title)
    if settings.artist:
        title_lines.append(f"by {settings.artist}")
    title_lines.append(f"Tuning: {' '.join(settings.tuning)}")
    title_lines.append("")

    measure_cols = [c for c in df.columns if c.startswith("Measure ")]
    body_lines = []

    for _, row in df.iterrows():
        string_label = str(row["String"]).rjust(2)
        cells = [str(row[col]).strip() for col in measure_cols]
        body_lines.append(f"{string_label}|{'|'.join(cells)}|")

    return "\n".join(title_lines + body_lines)


def parse_uploaded_csv(uploaded_file, tuning: list[str], measures: int, beats_per_measure: int) -> pd.DataFrame:
    """Read an uploaded CSV tab and align it to current editor settings."""
    uploaded_df = pd.read_csv(uploaded_file)
    if "String" not in uploaded_df.columns:
        st.warning("Uploaded CSV must contain a 'String' column. A blank tab was created instead.")
        return make_empty_tab(tuning, measures, beats_per_measure)

    return normalize_tab_df(uploaded_df, tuning, measures, beats_per_measure)


def reset_tab(settings: TabSettings) -> None:
    st.session_state.tab_df = make_empty_tab(
        settings.tuning,
        settings.measures,
        settings.beats_per_measure,
    )


st.set_page_config(page_title="Guitar Tab Editor", page_icon="🎸", layout="wide")

st.title("Guitar Tab Editor")
st.caption("Create, edit, validate, preview, and export guitar tabs in a table.")

with st.sidebar:
    st.header("Tab setup")

    title = st.text_input("Song title", value=st.session_state.get("song_title", "Untitled Tab"))
    artist = st.text_input("Artist", value=st.session_state.get("artist", ""))

    tuning_preset = st.selectbox(
        "Tuning preset",
        ["Standard (E A D G B e)", "Drop D (D A D G B e)", "Custom"],
        index=0,
    )

    if tuning_preset == "Standard (E A D G B e)":
        tuning = ["e", "B", "G", "D", "A", "E"]
    elif tuning_preset == "Drop D (D A D G B e)":
        tuning = ["e", "B", "G", "D", "A", "D"]
    else:
        custom = st.text_input("Custom tuning, high-to-low", value="e B G D A E")
        tuning = [part.strip() for part in custom.split() if part.strip()]
        if not tuning:
            tuning = STRINGS_STANDARD

    beats_per_measure = st.number_input("Beats per measure", min_value=1, max_value=16, value=4, step=1)
    measures = st.number_input("Number of measures", min_value=1, max_value=64, value=8, step=1)

    settings = TabSettings(
        title=title,
        artist=artist,
        tuning=tuning,
        beats_per_measure=int(beats_per_measure),
        measures=int(measures),
    )

    uploaded_file = st.file_uploader("Import tab CSV", type=["csv"])

    c1, c2 = st.columns(2)
    with c1:
        reset_clicked = st.button("New blank tab", use_container_width=True)
    with c2:
        normalize_clicked = st.button("Apply setup", use_container_width=True)

if "tab_df" not in st.session_state:
    reset_tab(settings)

if reset_clicked:
    reset_tab(settings)

if uploaded_file is not None:
    st.session_state.tab_df = parse_uploaded_csv(
        uploaded_file,
        settings.tuning,
        settings.measures,
        settings.beats_per_measure,
    )

if normalize_clicked:
    st.session_state.tab_df = normalize_tab_df(
        st.session_state.tab_df,
        settings.tuning,
        settings.measures,
        settings.beats_per_measure,
    )

measure_cols = [f"Measure {i}" for i in range(1, settings.measures + 1)]
st.session_state.tab_df = normalize_tab_df(
    st.session_state.tab_df,
    settings.tuning,
    settings.measures,
    settings.beats_per_measure,
)

column_config = {
    "String": st.column_config.TextColumn(
        "String",
        help="Instrument string label, ordered high-to-low.",
        width="small",
        disabled=True,
    ),
}

for col in measure_cols:
    column_config[col] = st.column_config.TextColumn(
        col,
        help="Enter fret/tab notation. Example: --0-2h3--5/7--",
        width="medium",
        validate=CELL_PATTERN,
        default="-" * max(4, settings.beats_per_measure * 4),
    )

editor_tab, preview_tab, export_tab = st.tabs(["Editor", "Preview", "Export"])

with editor_tab:
    st.subheader("Editable tab table")
    st.write(
        "Edit each string/measure cell directly. Use common tab notation such as "
        "`0`, `2h3`, `5p3`, `7b9`, `5/7`, `7\\5`, `x`, `~`, and `|`."
    )

    edited_df = st.data_editor(
        st.session_state.tab_df,
        key="tab_editor",
        column_config=column_config,
        column_order=["String", *measure_cols],
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        height=360,
    )

    st.session_state.tab_df = normalize_tab_df(
        edited_df,
        settings.tuning,
        settings.measures,
        settings.beats_per_measure,
    )

    issues = validate_tab_cells(st.session_state.tab_df)
    if issues:
        st.error("Some cells need attention.")
        for issue in issues[:10]:
            st.write(f"- {issue}")
        if len(issues) > 10:
            st.write(f"- ...and {len(issues) - 10} more.")
    else:
        st.success("Tab notation looks valid.")

with preview_tab:
    st.subheader("Read-only table preview")
    st.dataframe(
        st.session_state.tab_df,
        column_config=column_config,
        column_order=["String", *measure_cols],
        hide_index=True,
        width="stretch",
        height=280,
    )

    st.subheader("ASCII tab preview")
    ascii_tab = render_ascii_tab(settings, st.session_state.tab_df)
    st.code(ascii_tab, language="text")

with export_tab:
    st.subheader("Download")

    csv_bytes = st.session_state.tab_df.to_csv(index=False).encode("utf-8")
    ascii_tab = render_ascii_tab(settings, st.session_state.tab_df)
    txt_bytes = ascii_tab.encode("utf-8")

    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="guitar_tab.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "Download ASCII tab",
            data=txt_bytes,
            file_name="guitar_tab.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.subheader("Copyable ASCII")
    st.text_area("ASCII tab", value=ascii_tab, height=260)