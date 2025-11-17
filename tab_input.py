"""Input tab for test data and erg results."""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional


def watts_to_split(watts: float) -> str:
    """
    Convert watts to split time using Concept2 formula.
    pace = ³√(2.80/watts), where pace is m:ss.0 per 500m
    """
    if watts <= 0:
        return ""

    pace_seconds = (2.80 / watts) ** (1/3) * 500
    minutes = int(pace_seconds // 60)
    seconds = pace_seconds % 60
    return f"{minutes}:{seconds:04.1f}"


def split_to_watts(split_str: str) -> Optional[float]:
    """
    Convert split time to watts using Concept2 formula.
    watts = 2.80 / pace³, where pace is seconds per 500m
    """
    if not split_str or split_str.strip() == "":
        return None

    try:
        parts = split_str.strip().split(":")
        if len(parts) != 2:
            return None

        minutes = int(parts[0])
        seconds = float(parts[1])

        total_seconds = minutes * 60 + seconds
        watts = 2.80 / (total_seconds / 500) ** 3

        return watts
    except (ValueError, ZeroDivisionError):
        return None


def render():
    """Render the input tab."""
    st.header("Test Data Input")

    # Add delete checkboxes column
    display_df = st.session_state.step_data.copy()
    display_df.insert(0, 'Delete', False)

    # Create data editor
    edited_df = st.data_editor(
        display_df,
        hide_index=True,
        width='stretch',
        num_rows="fixed",
        column_config={
            "Delete": st.column_config.CheckboxColumn(
                "Delete",
                help="Select rows to delete",
                default=False,
            ),
            "Duration (min)": st.column_config.NumberColumn(
                "Duration (min)",
                min_value=0.0,
                step=0.5,
                format="%.1f"
            ),
            "Power (W)": st.column_config.NumberColumn(
                "Power (W)",
                min_value=0.0,
                step=5.0,
                format="%.1f"
            ),
            "Split (m:ss.0)": st.column_config.TextColumn(
                "Split (m:ss.0)"
            ),
            "Heart Rate (bpm)": st.column_config.NumberColumn(
                "Heart Rate (bpm)",
                min_value=0,
                max_value=250,
                step=1
            ),
            "Lactate (mmol/L)": st.column_config.NumberColumn(
                "Lactate (mmol/L)",
                min_value=0.0,
                step=0.1,
                format="%.2f"
            ),
            "RPE": st.column_config.NumberColumn(
                "RPE",
                help="Rating of Perceived Exertion (6-20)",
                min_value=6,
                max_value=20,
                step=1
            )
        }
    )

    # Row management buttons
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Add Row"):
            new_row = pd.DataFrame({
                'Duration (min)': [3.0],
                'Power (W)': [0.0],
                'Split (m:ss.0)': [''],
                'Heart Rate (bpm)': [0],
                'Lactate (mmol/L)': [0.0],
                'RPE': [6]
            })
            st.session_state.step_data = pd.concat([st.session_state.step_data, new_row], ignore_index=True)
            st.rerun()

    with col2:
        if st.button("Delete Selected Rows"):
            # Remove Delete column and filter out selected rows
            data_without_delete = edited_df.drop('Delete', axis=1)
            filtered_df = data_without_delete[~edited_df['Delete']].reset_index(drop=True)
            if len(filtered_df) > 0:
                st.session_state.step_data = filtered_df
                st.session_state.previous_data = filtered_df.copy()
                st.rerun()

    # Sync power and split - detect which changed and update the other
    needs_rerun = False
    prev_df = st.session_state.previous_data

    # Remove Delete column from edited_df for processing
    data_df = edited_df.drop('Delete', axis=1)

    # Handle rows that exist in both old and new data
    for idx in range(min(len(data_df), len(prev_df))):
        power_changed = data_df.at[idx, 'Power (W)'] != prev_df.at[idx, 'Power (W)']
        split_changed = data_df.at[idx, 'Split (m:ss.0)'] != prev_df.at[idx, 'Split (m:ss.0)']

        if power_changed and data_df.at[idx, 'Power (W)'] > 0:
            # Power was edited, update split
            new_split = watts_to_split(data_df.at[idx, 'Power (W)'])
            data_df.at[idx, 'Split (m:ss.0)'] = new_split
            needs_rerun = True
        elif split_changed and data_df.at[idx, 'Split (m:ss.0)']:
            # Split was edited, update power
            new_watts = split_to_watts(data_df.at[idx, 'Split (m:ss.0)'])
            if new_watts is not None:
                data_df.at[idx, 'Power (W)'] = new_watts
                needs_rerun = True

    # Update session state
    st.session_state.step_data = data_df
    st.session_state.previous_data = data_df.copy()

    # Rerun to show conversions in the same table
    if needs_rerun:
        st.rerun()

    # Erg Test Results section
    st.header("Erg Test Results")

    # Add delete checkboxes column for erg tests
    erg_display_df = st.session_state.erg_test_data.copy()
    erg_display_df.insert(0, 'Delete', False)

    # Create erg test data editor
    edited_erg_df = st.data_editor(
        erg_display_df,
        hide_index=True,
        width='stretch',
        num_rows="fixed",
        column_config={
            "Delete": st.column_config.CheckboxColumn(
                "Delete",
                help="Select rows to delete",
                default=False,
            ),
            "Distance (m)": st.column_config.NumberColumn(
                "Distance (m)",
                min_value=0,
                step=100,
                format="%d"
            ),
            "Time (m:ss.s)": st.column_config.TextColumn(
                "Time (m:ss.s)",
                help="Format: m:ss.s (e.g., 2:54.2)"
            )
        }
    )

    # Erg test row management buttons
    erg_col1, erg_col2 = st.columns([1, 5])
    with erg_col1:
        if st.button("Add Row", key="add_erg_row"):
            new_erg_row = pd.DataFrame({
                'Distance (m)': [2000],
                'Time (m:ss.s)': ['6:00.0']
            })
            st.session_state.erg_test_data = pd.concat([st.session_state.erg_test_data, new_erg_row], ignore_index=True)
            st.rerun()

    with erg_col2:
        if st.button("Delete Selected Rows", key="delete_erg_rows"):
            erg_data_without_delete = edited_erg_df.drop('Delete', axis=1)
            filtered_erg_df = erg_data_without_delete[~edited_erg_df['Delete']].reset_index(drop=True)
            if len(filtered_erg_df) > 0:
                st.session_state.erg_test_data = filtered_erg_df
                st.session_state.previous_erg_data = filtered_erg_df.copy()
                st.rerun()

    # Update erg test session state
    erg_data_df = edited_erg_df.drop('Delete', axis=1)
    st.session_state.erg_test_data = erg_data_df
    st.session_state.previous_erg_data = erg_data_df.copy()
