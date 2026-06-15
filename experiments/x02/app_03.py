import streamlit as st
import pandas as pd

st.set_page_config(page_title="Guitar Tab Editor", layout="wide")

st.title("Guitar Tab Editor")

STANDARD_TUNING = ("E4", "B3", "G3", "D3", "A2", "E2")


def make_empty_tab(num_beats: int = 16) -> pd.DataFrame:
    """Create an empty guitar tab table."""
    return pd.DataFrame(
        data={
            "String": STANDARD_TUNING,
            **{str(i + 1): [""] * len(STANDARD_TUNING) for i in range(num_beats)},
        },
        dtype="string[pyarrow]",
    )


def tab_to_text(df: pd.DataFrame) -> str:
    """Convert the editable table into plain text guitar tab."""
    beat_columns = [col for col in df.columns if col != "String"]

    lines = []
    for _, row in df.iterrows():
        string_name = row["String"]
        notes = [str(row[col]) if pd.notna(row[col]) else "" for col in beat_columns]

        # Keep each beat cell visually aligned.
        rendered_notes = "".join(note.center(3, "─") if note else "───" for note in notes)
        lines.append(f"{string_name}├{rendered_notes}┤")

    return "\n".join(lines)


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


edited_df = st.data_editor(
    st.session_state.tab_df,
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    on_change=None,
    column_config={
        "String": st.column_config.TextColumn(
            "String",
            disabled=True,
            width="small",
            default="E2",
            validate=r"^(?:[ACDFG](?:#|b)?|[BE]b?)(?:-1|[0-9])$",
        ),
        **{
            col: st.column_config.TextColumn(
                validate=r"^(?:[0-9]|[12][0-9]|30)$",
            ) for col in st.session_state.tab_df.columns if col != "String"
        }
    },
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