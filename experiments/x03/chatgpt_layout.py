# app.py
import streamlit as st

st.set_page_config(page_title="Hierarchical UI Demo", layout="wide")

st.title("Hierarchical Streamlit UI")

# Root level
with st.container(border=True):
    st.header("1. App Shell")

    # Level 1: Sidebar controls
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Page", ["Dashboard", "Settings", "Help"])

        with st.expander("Filters"):
            date_range = st.date_input("Date range", [])
            category = st.selectbox("Category", ["All", "A", "B", "C"])

    # Level 1: Main content
    with st.container(border=True):
        st.subheader(f"2. Main Content: {page}")

        # Level 2: Tabs
        tab_overview, tab_details, tab_actions = st.tabs(
            ["Overview", "Details", "Actions"]
        )

        # Level 3: Overview tab
        with tab_overview:
            st.markdown("### 3. Overview Section")

            col1, col2, col3 = st.columns(3)

            with col1:
                with st.container(border=True):
                    st.metric("Users", "1,240", "+12%")

            with col2:
                with st.container(border=True):
                    st.metric("Revenue", "$48K", "+8%")

            with col3:
                with st.container(border=True):
                    st.metric("Errors", "23", "-4%")

        # Level 3: Details tab
        with tab_details:
            st.markdown("### 3. Details Section")

            with st.expander("Nested Form"):
                with st.form("details_form"):
                    name = st.text_input("Name")
                    email = st.text_input("Email")
                    role = st.selectbox("Role", ["Admin", "Editor", "Viewer"])
                    submitted = st.form_submit_button("Save")

                    if submitted:
                        st.success(f"Saved {name} as {role}")

            with st.expander("Nested Data Table"):
                st.dataframe(
                    {
                        "Component": ["Sidebar", "Tabs", "Columns", "Form"],
                        "Level": [1, 2, 3, 4],
                        "Purpose": [
                            "Navigation",
                            "Content grouping",
                            "Horizontal layout",
                            "Input collection",
                        ],
                    },
                    use_container_width=True,
                )

        # Level 3: Actions tab
        with tab_actions:
            st.markdown("### 3. Actions Section")

            with st.container(border=True):
                st.write("Grouped action buttons")

                action_col1, action_col2 = st.columns(2)

                with action_col1:
                    if st.button("Run Process"):
                        st.info("Process started")

                with action_col2:
                    if st.button("Reset"):
                        st.warning("State reset")

# Footer level
with st.container(border=True):
    st.caption("Hierarchy: App → Sidebar/Main → Tabs → Containers → Columns/Form/Widgets")