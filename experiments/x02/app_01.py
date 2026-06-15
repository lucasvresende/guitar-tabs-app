import streamlit as st
import pandas as pd
import json

st.set_page_config(
    page_title="Guitar Tab Editor",
    page_icon="🎸",
    layout="wide",
)

STRINGS_STANDARD = ("E4", "B3", "G3", "D3", "A2", "E2")


def make_blank_tab(num_beats: int) -> pd.DataFrame:
    return pd.DataFrame(
        data=[["-" for _ in range(num_beats)] for _ in STRINGS_STANDARD],
        index=STRINGS_STANDARD,
        columns=[str(i + 1) for i in range(num_beats)],
        dtype="string[pyarrow]"
    )


def render_ascii_tab(df: pd.DataFrame) -> str:
    lines = []

    for string_name in df.index:
        row = df.loc[string_name].fillna("").astype(str).tolist()

        line = f"{string_name}|"
        for cell in row:
            value = cell.strip()
            if value == "":
                value = "-"
            line += f"{value:^3}-"

        lines.append(line)

    return "\n".join(lines)


def dataframe_to_json(df: pd.DataFrame) -> str:
    return json.dumps(
        {
            "strings": list(df.index),
            "columns": list(df.columns),
            "data": df.values.tolist(),
        },
        indent=2,
    )


def json_to_dataframe(uploaded_file) -> pd.DataFrame:
    payload = json.load(uploaded_file)

    return pd.DataFrame(
        payload["data"],
        index=payload["strings"],
        columns=payload["columns"],
    )


st.title("Guitar Tab Editor")

with st.sidebar:
    st.header("Tab Settings")

    num_beats = st.number_input(
        "Number of columns / beats",
        min_value=4,
        max_value=128,
        value=16,
        step=4,
    )

    uploaded_file = st.file_uploader(
        "Import tab JSON",
        type=["json"],
    )

    reset = st.button("Create Blank Tab")

if "tab_df" not in st.session_state:
    st.session_state.tab_df = make_blank_tab(num_beats)

if reset:
    st.session_state.tab_df = make_blank_tab(num_beats)

if uploaded_file is not None:
    try:
        st.session_state.tab_df = json_to_dataframe(uploaded_file)
        st.sidebar.success("Tab imported.")
    except Exception as exc:
        st.sidebar.error(f"Could not import file: {exc}")

st.subheader("Edit Tab")

edited_df = st.data_editor(
    st.session_state.tab_df,
    use_container_width=True,
    num_rows="fixed",
    key="tab_editor",
)

st.session_state.tab_df = edited_df

st.subheader("ASCII Preview")

ascii_tab = render_ascii_tab(st.session_state.tab_df)

st.code(ascii_tab, language="text")

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="Download ASCII Tab",
        data=ascii_tab,
        file_name="guitar_tab.txt",
        mime="text/plain",
    )

with col2:
    json_tab = dataframe_to_json(st.session_state.tab_df)

    st.download_button(
        label="Download Editable JSON",
        data=json_tab,
        file_name="guitar_tab.json",
        mime="application/json",
    )

st.subheader("How to Enter Notes")

st.markdown(
    """
Use each cell as one rhythmic position.

Examples:

| Entry | Meaning |
|---|---|
| `0` | Open string |
| `3` | 3rd fret |
| `3h5` | Hammer-on |
| `5p3` | Pull-off |
| `7/9` | Slide up |
| `9\\7` | Slide down |
| `5~` | Vibrato |
| `x` | Muted note |
| `3,5` | Multiple notes in one position |
"""
)