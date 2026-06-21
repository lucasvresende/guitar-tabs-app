# Guitar Tabs App

Guitar Tabs App is a Streamlit app for writing simple guitar tablature, previewing it as
plain-text tab, synthesizing it into audio, and downloading the result.

The maintained application code lives in `guitar_tabs_app/`.

## Features

- Edit a tab in a fixed Streamlit table, with one row per string and one column per beat.
- Configure playback speed, beat count, string count, and tuning.
- Validate note names, fret numbers, tab tables, and generated note events.
- Render the editable table as plain-text guitar tab.
- Synthesize notes with a Karplus-Strong plucked-string algorithm.
- Preview and download generated Ogg audio.

## Requirements

- Python `>=3.14`
- `uv` for dependency management
- `ffmpeg` for MP3/Ogg export through `pydub`

On Debian/Ubuntu-style systems, install `ffmpeg` with:

```bash
sudo apt install ffmpeg
```

## Setup

Install the project dependencies:

```bash
uv sync
```

## Run The App

Start Streamlit from the repository root:

```bash
uv run streamlit run guitar_tabs_app
```

Then open the local URL printed by Streamlit.

## How To Use

1. Enter fret numbers in the tab grid.
2. Open **Config** to change speed, number of beats, number of strings, or tuning.
3. Click **Apply** after changing configuration values.
4. Open **Text Preview** to view or download a `.txt` tab.
5. Open **Audio Preview** to synthesize, play, or download an `.ogg` audio file.

## Package Overview

### `guitar_tabs_app/streamlit_app.py`

Defines the Streamlit UI. It creates the editable tab grid, configuration controls, text preview,
audio preview, and download buttons.

### `guitar_tabs_app/validation.py`

Defines configuration and validation rules.

- `RegexPatterns`: Shared regex patterns for note names and fret numbers.
- `TabConfig`: Frozen Pydantic model for tab timing, tuning, synthesis, and audio settings.
- `NotesDfSchema`: Dataframely schema for synthesized note events.
- `validate_notes_df()`: Validates note events against schema and config-dependent limits.
- `make_tab_schema()`: Builds a dynamic Dataframely schema for the configured string count.
- `validate_tab_df()`: Validates the normalized tab dataframe.
- `validate_ui_df()`: Validates the editable UI dataframe shape.

### `guitar_tabs_app/transformations.py`

Converts data between UI, normalized tab, note events, text, and audio.

- `make_empty_tab()`: Creates the initial editable tab dataframe.
- `check_ui_df_not_filled()`: Checks whether the tab has no fret entries.
- `ui_df_to_tab_df()`: Converts the Streamlit table into a validated tab dataframe.
- `tab_df_to_notes_df()`: Converts tab rows into validated note events.
- `tab_to_text()`: Renders the editable tab as plain text.
- `ui_df_to_numpy_audio()`: Converts the editable tab directly into synthesized samples.

### `guitar_tabs_app/synthesis.py`

Generates audio samples from note events.

- `_karplus_strong_core()`: Numba-compiled Karplus-Strong inner loop.
- `karplus_strong_ring()`: Generates one plucked-string waveform.
- `synthesize_note()`: Dispatches synthesis for one note using `TabConfig`.
- `synthesize_notes()`: Mixes all note events into one mono audio array.

### `guitar_tabs_app/exporting.py`

Writes synthesized NumPy audio arrays to files.

- `save_numpy_as_mp3()`: Exports MP3 audio.
- `save_numpy_as_ogg()`: Exports Ogg Vorbis audio.
- `save_numpy_as_wav()`: Exports WAV audio.

### `guitar_tabs_app/note_conversions.py`

Converts between note names, MIDI values, and frequencies.

- `name_to_midi()`: Converts note names such as `E2` or `C#4` to MIDI note numbers.
- `midi_to_name()`: Converts MIDI note numbers to note names.
- `midi_to_frequency()`: Converts MIDI note numbers to frequency in hertz.
- `frequency_to_midi()`: Converts frequency in hertz to a MIDI note value.

## Development Checks

Run Ruff and Pyright:

```bash
uv run ruff check .
uv run pyright
```

The configured checks target `guitar_tabs_app/`.

## Next Steps

- Add effects like slide, hammer-on, pull-off, vibrato, and bend.
- Add support for exporting and importing tabs as csv.
- Add support for multiple tabs in a single page, forming a single final audio output.
