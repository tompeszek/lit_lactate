"""Metrics tab for lactate threshold analysis."""
import streamlit as st
import pandas as pd


def render(results):
    """Render the metrics tab."""
    st.header("Lactate Threshold Metrics")

    if results is not None:
        # Calculate height to show all rows (35px per row + 38px header)
        row_height = 35
        header_height = 38
        total_height = len(results) * row_height + header_height

        # Apply row coloring based on Metric
        def highlight_metric(row):
            if row['Metric'] == 'LT1 (Aerobic Threshold)':
                color = 'background-color: rgba(144, 238, 144, 0.3)'  # light green
            elif row['Metric'] == 'Between LT1 and LT2':
                color = 'background-color: rgba(255, 255, 0, 0.3)'  # light yellow
            elif row['Metric'] == 'LT2/MLSS (Anaerobic Threshold)':
                color = 'background-color: rgba(255, 182, 193, 0.3)'  # light red
            else:
                color = ''
            return [color] * len(row)

        # Format columns: intensity/HR as whole numbers, lactate to 1 decimal
        format_dict = {}
        if 'Intensity (W)' in results.columns:
            format_dict['Intensity (W)'] = '{:.0f}'
        if 'Lambda Intensity (W)' in results.columns:
            format_dict['Lambda Intensity (W)'] = '{:.0f}'
        if 'Lactate (mmol/L)' in results.columns:
            format_dict['Lactate (mmol/L)'] = '{:.1f}'
        if 'Lambda Lactate (mmol/L)' in results.columns:
            format_dict['Lambda Lactate (mmol/L)'] = '{:.1f}'
        if 'Heart Rate (bpm)' in results.columns:
            format_dict['Heart Rate (bpm)'] = '{:.0f}'
        if 'Lambda HR (bpm)' in results.columns:
            format_dict['Lambda HR (bpm)'] = '{:.0f}'

        styled_results = results.style.apply(highlight_metric, axis=1).format(format_dict, na_rep='')

        # Configure columns to be hidden by default
        column_config = {}
        if 'Category' in results.columns:
            column_config['Category'] = None
        if 'Fitting' in results.columns:
            column_config['Fitting'] = None

        st.dataframe(
            styled_results,
            hide_index=True,
            width='stretch',
            height=total_height,
            column_config=column_config
        )
    else:
        st.info("Add at least 3 rows of test data to see threshold metrics")
