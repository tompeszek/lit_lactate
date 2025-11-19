import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

# Import tab modules
import tab_input
import tab_metrics
import tab_zones
import tab_zones2 as tab_ranges

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

    # Run analysis once for all tabs
    results = None
    if len(st.session_state.step_data) >= 3:
        analysis_df = st.session_state.step_data[['Power (W)', 'Lactate (mmol/L)', 'Heart Rate (bpm)']].copy()
        analysis_df.columns = ['intensity', 'lactate', 'heart_rate']
        results = analyze_lactate_data(analysis_df)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Input", "Metrics", "Zones", "Ranges"])

    with tab1:
        tab_input.render()

    with tab2:
        tab_metrics.render(results)

    with tab3:
        tab_zones.render(results)

    with tab4:
        tab_ranges.render(results)


if __name__ == "__main__":
    main()
