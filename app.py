import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="Rowing Lactate Analyzer",
    page_icon="🚣",
    layout="wide"
)

def watts_to_split(watts: float) -> str:
    """
    Convert watts to split time using Concept2 formula.
    pace = ³√(2.80/watts), where pace is m:ss.0 per 500m

    Args:
        watts: Power in watts

    Returns:
        Split time as string in format m:ss.0
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

    Args:
        split_str: Split time as string in format m:ss.s or m:ss

    Returns:
        Power in watts, or None if invalid input
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


def initialize_session_state():
    """Initialize session state variables."""
    if 'step_data' not in st.session_state:
        powers = [0.0, 169.0, 214.0, 260.0, 305.0, 395.0, 484.0]
        st.session_state.step_data = pd.DataFrame({
            'Duration (min)': [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            'Power (W)': powers,
            'Split (m:ss.0)': [watts_to_split(p) if p > 0 else '' for p in powers],
            'Heart Rate (bpm)': [65, 91, 104, 120, 141, 170, 180],
            'Lactate (mmol/L)': [2.4, 2.4, 2.4, 1.8, 2.3, 5.8, 19.5],
            'RPE': [6, 8, 9, 9, 11, 14, 16]
        })
    if 'previous_data' not in st.session_state:
        st.session_state.previous_data = st.session_state.step_data.copy()

    if 'erg_test_data' not in st.session_state:
        st.session_state.erg_test_data = pd.DataFrame({
            'Distance (m)': [1000, 2000, 5000, 6000, 17785],
            'Time (m:ss.s)': ['2:54.2', '5:59.5', '15:59.3', '19:07.6', '60:00.0']
        })
    if 'previous_erg_data' not in st.session_state:
        st.session_state.previous_erg_data = st.session_state.erg_test_data.copy()


def analyze_lactate_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Analyze lactate data using Python implementation of lactate threshold methods.

    Args:
        df: DataFrame with columns for intensity, lactate, heart_rate

    Returns:
        DataFrame with analysis results or None if analysis fails
    """
    try:
        from lactate_analysis import analyze_lactate_thresholds

        # Extract arrays - include baseline (0 watts) as some methods need it
        intensity = df['intensity'].values
        lactate = df['lactate'].values
        heart_rate = df['heart_rate'].values

        # Validate we have enough exercise steps (non-zero intensity)
        exercise_steps = intensity > 0
        if exercise_steps.sum() < 3:
            st.error("At least 3 exercise steps (intensity > 0) are required for analysis")
            return None

        # Check if this is the default data - if so, skip Lambda API call
        default_intensity = np.array([0.0, 169.0, 214.0, 260.0, 305.0, 395.0, 484.0])
        default_lactate = np.array([2.4, 2.4, 2.4, 1.8, 2.3, 5.8, 19.5])
        default_hr = np.array([65, 91, 104, 120, 141, 170, 180])

        is_default_data = (
            np.array_equal(intensity, default_intensity) and
            np.array_equal(lactate, default_lactate) and
            np.array_equal(heart_rate, default_hr)
        )

        # Run comprehensive lactate threshold analysis
        # Skip Lambda API for default data (use cached results)
        results = analyze_lactate_thresholds(
            intensity, lactate, heart_rate,
            sport='cycling',
            include_lambda=not is_default_data  # Only call Lambda for non-default data
        )

        # If default data, add cached Lambda results
        if is_default_data:
            # Cached Lambda results for default data
            cached_lambda = {
                'OBLA 2.0': {'intensity': 306.3, 'lactate': 2.0, 'heart_rate': 135},
                'OBLA 2.5': {'intensity': 317.2, 'lactate': 2.5, 'heart_rate': 138},
                'OBLA 3.0': {'intensity': 328.2, 'lactate': 3.0, 'heart_rate': 141},
                'OBLA 3.5': {'intensity': 339.1, 'lactate': 3.5, 'heart_rate': 145},
                'OBLA 4.0': {'intensity': 350.0, 'lactate': 4.0, 'heart_rate': 148},
                'Dmax': {'intensity': 366.7, 'lactate': 3.9, 'heart_rate': 153},
                'ModDmax': {'intensity': 387.7, 'lactate': 5.3, 'heart_rate': 159},
                'Exp-Dmax': {'intensity': 386.7, 'lactate': 5.1, 'heart_rate': 159},
                'Bsln + 0.5': {'intensity': 326.0, 'lactate': 2.9, 'heart_rate': 141},
                'Bsln + 1.0': {'intensity': 336.9, 'lactate': 3.4, 'heart_rate': 144},
                'Bsln + 1.5': {'intensity': 347.9, 'lactate': 3.9, 'heart_rate': 147},
                'Log-log': {'intensity': 301.4, 'lactate': 1.9, 'heart_rate': 133},
                'LTP1': {'intensity': 294.4, 'lactate': 1.9, 'heart_rate': 131},
                'LTP2': {'intensity': 394.8, 'lactate': 6.0, 'heart_rate': 161},
                'LTratio': {'intensity': 320.9, 'lactate': 1.7, 'heart_rate': 139}
            }

            # Add Lambda columns manually
            results['Lambda Intensity (W)'] = results['Method'].map(
                lambda m: cached_lambda.get(m, {}).get('intensity', np.nan)
            )
            results['Lambda Split (/500m)'] = results['Lambda Intensity (W)'].apply(
                lambda p: '' if pd.isna(p) or p <= 0 else
                f"{int(((2.80/p)**(1/3)*500)//60)}:{((2.80/p)**(1/3)*500)%60:04.1f}"
            )
            results['Lambda Lactate (mmol/L)'] = results['Method'].map(
                lambda m: cached_lambda.get(m, {}).get('lactate', np.nan)
            )
            results['Lambda HR (bpm)'] = results['Method'].map(
                lambda m: cached_lambda.get(m, {}).get('heart_rate', np.nan)
            )

        return results

    except Exception as e:
        st.error(f"Error analyzing data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def main():
    """Main application."""
    initialize_session_state()

    st.title("Rowing Lactate Test Analyzer")
    st.markdown("Analyze rowing lactate test results using comprehensive threshold detection methods")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Input", "Metrics", "Zones", "Ranges"])

    with tab1:
        # Main data input section
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

    # Run analysis once for all tabs
    results = None
    if len(st.session_state.step_data) >= 3:
        analysis_df = st.session_state.step_data[['Power (W)', 'Lactate (mmol/L)', 'Heart Rate (bpm)']].copy()
        analysis_df.columns = ['intensity', 'lactate', 'heart_rate']
        results = analyze_lactate_data(analysis_df)

    with tab2:
        # Metrics section - LT Analysis table
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

    with tab3:
        # Zones section - Training Zones table
        st.header("Training Zones")

        if results is not None and len(st.session_state.erg_test_data) > 0:
            from training_zones import calculate_training_zones, calculate_pace_from_power
        
            # Parse erg test data from the existing erg_test_data table
            erg_tests = []
            for _, row in st.session_state.erg_test_data.iterrows():
                meters = row['Distance (m)']
                time_str = row['Time (m:ss.s)']
                if meters > 0 and time_str:
                    # Parse time string (format: "m:ss.s" or "mm:ss.s")
                    try:
                        parts = time_str.split(':')
                        if len(parts) == 2:
                            minutes = float(parts[0])
                            seconds = float(parts[1])
                            total_seconds = minutes * 60 + seconds
                            erg_tests.append({'time': total_seconds, 'meters': meters})
                    except:
                        pass
        
            # Default values for max HR and power (could make these configurable later)
            max_hr = 200
            max_power = 600
        
            if len(erg_tests) >= 1:
                # Calculate zones
                zones = calculate_training_zones(results, erg_tests, max_hr, max_power)
        
                # Zone details table
                zone_data = []
                for zone in zones:
                    pr = zone['power_range']
                    hr = zone.get('hr_range')
                    lac = zone.get('lactate_range')
        
                    # Calculate pace
                    pace_bottom = calculate_pace_from_power(pr['median_bottom'])
                    pace_top = calculate_pace_from_power(pr['median_top'])
        
                    # Format pace as M:SS.S, handle NaN
                    if pd.notna(pace_bottom) and pd.notna(pace_top):
                        pace_bottom_str = f"{int(pace_bottom//60)}:{pace_bottom%60:04.1f}"
                        pace_top_str = f"{int(pace_top//60)}:{pace_top%60:04.1f}"
                        pace_str = f"{pace_top_str}-{pace_bottom_str}"
                    else:
                        pace_str = 'N/A'
        
                    # Format power range, handle NaN
                    if pd.notna(pr['median_bottom']) and pd.notna(pr['median_top']):
                        power_str = f"{int(pr['median_bottom'])}-{int(pr['median_top'])}"
                    else:
                        power_str = 'N/A'
        
                    # Format HR range, handle NaN
                    if hr and pd.notna(hr['median_bottom']) and pd.notna(hr['median_top']):
                        hr_str = f"{int(hr['median_bottom'])}-{int(hr['median_top'])}"
                    else:
                        hr_str = 'N/A'
        
                    # Format lactate range, handle NaN
                    if lac and pd.notna(lac['median_bottom']) and pd.notna(lac['median_top']):
                        lac_str = f"{lac['median_bottom']:.1f}-{lac['median_top']:.1f}"
                    else:
                        lac_str = 'N/A'
        
                    zone_data.append({
                        'Zone': zone['zone'],
                        'Description': zone['description'],
                        'Power (W)': power_str,
                        'Pace (/500m)': pace_str,
                        'HR (bpm)': hr_str,
                        'Lactate (mmol/L)': lac_str
                    })
        
                zones_df = pd.DataFrame(zone_data)
                st.dataframe(zones_df, hide_index=True, width='stretch')
        else:
            st.info("Add test data and erg test results to see training zones")

    with tab4:
        # Ranges section - Zone range graphs
        st.header("Training Zone Ranges")

        if results is not None and len(st.session_state.erg_test_data) > 0:
            from training_zones import calculate_training_zones, calculate_pace_from_power

            # Parse erg test data
            erg_tests = []
            for _, row in st.session_state.erg_test_data.iterrows():
                meters = row['Distance (m)']
                time_str = row['Time (m:ss.s)']
                if meters > 0 and time_str:
                    try:
                        parts = time_str.split(':')
                        if len(parts) == 2:
                            minutes = float(parts[0])
                            seconds = float(parts[1])
                            total_seconds = minutes * 60 + seconds
                            erg_tests.append({'time': total_seconds, 'meters': meters})
                    except:
                        pass

            max_hr = 200
            max_power = 600

            if len(erg_tests) >= 1:
                zones = calculate_training_zones(results, erg_tests, max_hr, max_power)

                # Create bar charts for each metric
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
        
                # Prepare data for charts
                zone_names = [z['zone'] for z in zones]
        
                # Power data
                power_data = []
                for z in zones:
                    pr = z['power_range']
                    if all(pd.notna(pr[k]) for k in ['min_bottom', 'median_bottom', 'median_top', 'max_top']):
                        power_data.append(pr)
                    else:
                        power_data.append(None)
        
                # Pace data (convert from power)
                pace_data = []
                for pr in power_data:
                    if pr:
                        pace_data.append({
                            'min_bottom': calculate_pace_from_power(pr['max_top']),  # inverted
                            'median_bottom': calculate_pace_from_power(pr['median_top']),  # inverted
                            'median_top': calculate_pace_from_power(pr['median_bottom']),  # inverted
                            'max_top': calculate_pace_from_power(pr['min_bottom'])  # inverted
                        })
                    else:
                        pace_data.append(None)
        
                # HR data
                hr_data = []
                for z in zones:
                    hr = z.get('hr_range')
                    if hr and all(pd.notna(hr[k]) for k in ['min_bottom', 'median_bottom', 'median_top', 'max_top']):
                        hr_data.append(hr)
                    else:
                        hr_data.append(None)
        
                # Lactate data
                lac_data = []
                for z in zones:
                    lac = z.get('lactate_range')
                    if lac and all(pd.notna(lac[k]) for k in ['min_bottom', 'median_bottom', 'median_top', 'max_top']):
                        lac_data.append(lac)
                    else:
                        lac_data.append(None)
        
                # Create subplots: 4 rows, 1 column
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Power (W)', 'Pace (/500m)', 'Heart Rate (bpm)', 'Lactate (mmol/L)'),
                    vertical_spacing=0.08,
                    row_heights=[0.25, 0.25, 0.25, 0.25]
                )
        
                colors = ['#9370DB', '#90EE90', '#FFB6C1', '#87CEEB']  # Purple, Green, Pink, Blue
        
                # Power chart
                if any(power_data):
                    for i, (zone, pd_val) in enumerate(zip(zone_names, power_data)):
                        if pd_val:
                            # Full range (light)
                            fig.add_trace(go.Bar(
                                name=f'{zone} range',
                                x=[zone],
                                y=[pd_val['max_top'] - pd_val['min_bottom']],
                                base=[pd_val['min_bottom']],
                                marker_color=f"rgba(147, 112, 219, 0.3)",
                                showlegend=False,
                                hovertemplate=f"{zone}<br>Min: {pd_val['min_bottom']:.0f}W<br>Max: {pd_val['max_top']:.0f}W<extra></extra>"
                            ), row=1, col=1)
        
                            # Target range (dark)
                            fig.add_trace(go.Bar(
                                name=f'{zone} target',
                                x=[zone],
                                y=[pd_val['median_top'] - pd_val['median_bottom']],
                                base=[pd_val['median_bottom']],
                                marker_color=colors[0],
                                showlegend=False,
                                text=f"{pd_val['median_bottom']:.0f}-{pd_val['median_top']:.0f}",
                                textposition='inside',
                                textfont=dict(size=10, color='white'),
                                hovertemplate=f"{zone}<br>Target: {pd_val['median_bottom']:.0f}-{pd_val['median_top']:.0f}W<extra></extra>"
                            ), row=1, col=1)
        
                # Pace chart
                if any(pace_data):
                    for i, (zone, pace_val) in enumerate(zip(zone_names, pace_data)):
                        if pace_val:
                            # Full range (light)
                            fig.add_trace(go.Bar(
                                name=f'{zone} range',
                                x=[zone],
                                y=[pace_val['max_top'] - pace_val['min_bottom']],
                                base=[pace_val['min_bottom']],
                                marker_color=f"rgba(144, 238, 144, 0.3)",
                                showlegend=False,
                                hovertemplate=f"{zone}<br>Min: {int(pace_val['min_bottom']//60)}:{pace_val['min_bottom']%60:04.1f}<br>Max: {int(pace_val['max_top']//60)}:{pace_val['max_top']%60:04.1f}<extra></extra>"
                            ), row=2, col=1)
        
                            # Target range (dark)
                            fig.add_trace(go.Bar(
                                name=f'{zone} target',
                                x=[zone],
                                y=[pace_val['median_top'] - pace_val['median_bottom']],
                                base=[pace_val['median_bottom']],
                                marker_color=colors[1],
                                showlegend=False,
                                text=f"{int(pace_val['median_bottom']//60)}:{pace_val['median_bottom']%60:04.1f}-{int(pace_val['median_top']//60)}:{pace_val['median_top']%60:04.1f}",
                                textposition='inside',
                                textfont=dict(size=9, color='white'),
                                hovertemplate=f"{zone}<br>Target: {int(pace_val['median_bottom']//60)}:{pace_val['median_bottom']%60:04.1f}-{int(pace_val['median_top']//60)}:{pace_val['median_top']%60:04.1f}<extra></extra>"
                            ), row=2, col=1)
        
                # HR chart
                if any(hr_data):
                    for i, (zone, hr_val) in enumerate(zip(zone_names, hr_data)):
                        if hr_val:
                            # Full range (light)
                            fig.add_trace(go.Bar(
                                name=f'{zone} range',
                                x=[zone],
                                y=[hr_val['max_top'] - hr_val['min_bottom']],
                                base=[hr_val['min_bottom']],
                                marker_color=f"rgba(255, 182, 193, 0.3)",
                                showlegend=False,
                                hovertemplate=f"{zone}<br>Min: {hr_val['min_bottom']:.0f}bpm<br>Max: {hr_val['max_top']:.0f}bpm<extra></extra>"
                            ), row=3, col=1)
        
                            # Target range (dark)
                            fig.add_trace(go.Bar(
                                name=f'{zone} target',
                                x=[zone],
                                y=[hr_val['median_top'] - hr_val['median_bottom']],
                                base=[hr_val['median_bottom']],
                                marker_color=colors[2],
                                showlegend=False,
                                text=f"{hr_val['median_bottom']:.0f}-{hr_val['median_top']:.0f}",
                                textposition='inside',
                                textfont=dict(size=10, color='white'),
                                hovertemplate=f"{zone}<br>Target: {hr_val['median_bottom']:.0f}-{hr_val['median_top']:.0f}bpm<extra></extra>"
                            ), row=3, col=1)
        
                # Lactate chart
                if any(lac_data):
                    for i, (zone, lac_val) in enumerate(zip(zone_names, lac_data)):
                        if lac_val:
                            # Full range (light)
                            fig.add_trace(go.Bar(
                                name=f'{zone} range',
                                x=[zone],
                                y=[lac_val['max_top'] - lac_val['min_bottom']],
                                base=[lac_val['min_bottom']],
                                marker_color=f"rgba(135, 206, 235, 0.3)",
                                showlegend=False,
                                hovertemplate=f"{zone}<br>Min: {lac_val['min_bottom']:.1f}<br>Max: {lac_val['max_top']:.1f}<extra></extra>"
                            ), row=4, col=1)
        
                            # Target range (dark)
                            fig.add_trace(go.Bar(
                                name=f'{zone} target',
                                x=[zone],
                                y=[lac_val['median_top'] - lac_val['median_bottom']],
                                base=[lac_val['median_bottom']],
                                marker_color=colors[3],
                                showlegend=False,
                                text=f"{lac_val['median_bottom']:.1f}-{lac_val['median_top']:.1f}",
                                textposition='inside',
                                textfont=dict(size=10, color='white'),
                                hovertemplate=f"{zone}<br>Target: {lac_val['median_bottom']:.1f}-{lac_val['median_top']:.1f}<extra></extra>"
                            ), row=4, col=1)
        
                # Update layout
                fig.update_layout(
                    height=1000,
                    showlegend=False,
                    barmode='overlay',
                    title_text="Training Zone Ranges",
                    title_x=0.5
                )
        
                # Update all y-axes
                fig.update_yaxes(title_text="Power (W)", row=1, col=1)
                fig.update_yaxes(title_text="Pace (s/500m)", row=2, col=1)
                fig.update_yaxes(title_text="HR (bpm)", row=3, col=1)
                fig.update_yaxes(title_text="Lactate (mmol/L)", row=4, col=1)

                st.plotly_chart(fig)
        else:
            st.info("Add test data and erg test results to see training zone ranges")


if __name__ == "__main__":
    main()
