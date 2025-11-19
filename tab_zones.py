"""Zones tab for training zones table."""
import streamlit as st
import pandas as pd
from training_zones import calculate_training_zones, calculate_pace_from_power


def render(results):
    """Render the zones tab."""
    st.header("Training Zones")

    if results is not None and len(st.session_state.erg_test_data) > 0:
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

        # Default values for max HR and power
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

                # Calculate pace for all values
                pace_min = calculate_pace_from_power(pr['max_top'])  # inverted
                pace_median_bottom = calculate_pace_from_power(pr['median_top'])  # inverted
                pace_median_top = calculate_pace_from_power(pr['median_bottom'])  # inverted
                pace_max = calculate_pace_from_power(pr['min_bottom'])  # inverted

                # Format pace median range
                if all(pd.notna(v) for v in [pace_median_bottom, pace_median_top]):
                    pace_median_bottom_str = f"{int(pace_median_bottom//60)}:{pace_median_bottom%60:04.1f}"
                    pace_median_top_str = f"{int(pace_median_top//60)}:{pace_median_top%60:04.1f}"
                    pace_median_str = f"{pace_median_bottom_str}-{pace_median_top_str}"
                else:
                    pace_median_str = 'N/A'

                # Format pace min/max range (blank if same as median)
                pace_minmax_str = ''
                if all(pd.notna(v) for v in [pace_min, pace_max]):
                    pace_min_str = f"{int(pace_min//60)}:{pace_min%60:04.1f}"
                    pace_max_str = f"{int(pace_max//60)}:{pace_max%60:04.1f}"
                    # Check if different from median
                    if pace_min_str != pace_median_bottom_str or pace_max_str != pace_median_top_str:
                        pace_minmax_str = f"{pace_min_str}-{pace_max_str}"

                # Format power median range
                if all(pd.notna(pr[k]) for k in ['median_bottom', 'median_top']):
                    power_median_str = f"{int(pr['median_bottom'])}-{int(pr['median_top'])}"
                else:
                    power_median_str = 'N/A'

                # Format power min/max range (blank if same as median)
                power_minmax_str = ''
                if all(pd.notna(pr[k]) for k in ['min_bottom', 'max_top']):
                    if int(pr['min_bottom']) != int(pr['median_bottom']) or int(pr['max_top']) != int(pr['median_top']):
                        power_minmax_str = f"{int(pr['min_bottom'])}-{int(pr['max_top'])}"

                # Format HR median range
                if hr and all(pd.notna(hr[k]) for k in ['median_bottom', 'median_top']):
                    hr_median_str = f"{int(hr['median_bottom'])}-{int(hr['median_top'])}"
                else:
                    hr_median_str = 'N/A'

                # Format HR min/max range (blank if same as median)
                hr_minmax_str = ''
                if hr and all(pd.notna(hr[k]) for k in ['min_bottom', 'max_top']):
                    if int(hr['min_bottom']) != int(hr['median_bottom']) or int(hr['max_top']) != int(hr['median_top']):
                        hr_minmax_str = f"{int(hr['min_bottom'])}-{int(hr['max_top'])}"

                # Format lactate median range
                if lac and all(pd.notna(lac[k]) for k in ['median_bottom', 'median_top']):
                    lac_median_str = f"{lac['median_bottom']:.1f}-{lac['median_top']:.1f}"
                else:
                    lac_median_str = 'N/A'

                # Format lactate min/max range (blank if same as median)
                lac_minmax_str = ''
                if lac and all(pd.notna(lac[k]) for k in ['min_bottom', 'max_top']):
                    if abs(lac['min_bottom'] - lac['median_bottom']) > 0.05 or abs(lac['max_top'] - lac['median_top']) > 0.05:
                        lac_minmax_str = f"{lac['min_bottom']:.1f}-{lac['max_top']:.1f}"

                zone_data.append({
                    'Zone': zone['zone'],
                    'Description': zone['description'],
                    'Power (W)': power_median_str,
                    'Power Min/Max': power_minmax_str,
                    'Pace (/500m)': pace_median_str,
                    'Pace Min/Max': pace_minmax_str,
                    'HR (bpm)': hr_median_str,
                    'HR Min/Max': hr_minmax_str,
                    'Lactate (mmol/L)': lac_median_str,
                    'Lactate Min/Max': lac_minmax_str
                })

            zones_df = pd.DataFrame(zone_data)
            st.dataframe(zones_df, hide_index=True, width='stretch')
    else:
        st.info("Add test data and erg test results to see training zones")
