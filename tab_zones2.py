"""Zones2 tab for detailed training zone visualizations."""
import streamlit as st
import pandas as pd
import numpy as np
from training_zones import calculate_training_zones


def render(results):
    """Render the zones2 tab with detailed zone visualizations."""
    st.header("Training Zones Visualization")

    if results is not None and len(st.session_state.erg_test_data) > 0:
        # Store results for method lookup
        global _results_df
        _results_df = results
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

            # Zone colors
            zone_colors = [
                '#4ade80',  # Z1 - bright green
                '#86efac',  # Z2 - light green
                '#fbbf24',  # Z3 - yellow/gold
                '#fb923c',  # Z4 - orange
                '#f87171',  # Z5 - light red
                '#dc2626',  # Z6 - red
                '#991b1b'   # Z7 - dark red
            ]

            zone_names = [
                'Recovery',
                'Endurance',
                'Tempo',
                'Lactate Threshold',
                'VO2 Max',
                'Anaerobic',
                'Neuromuscular'
            ]

            # Extract zone data for each metric
            metrics = ['Heart Rate', 'Power', 'Pace', 'Lactate', 'RPE']
            metric_units = ['(bpm)', '(W)', '(/500m)', '(mmol/L)', '(6-20)']

            # Prepare data structure
            zone_data = {
                'Heart Rate': [],
                'Power': [],
                'Pace': [],
                'Lactate': [],
                'RPE': []
            }

            for i, zone in enumerate(zones):
                # Heart Rate
                hr = zone.get('hr_range')
                if hr and all(pd.notna(hr[k]) for k in ['min_bottom', 'median_bottom', 'median_top', 'max_top']):
                    zone_data['Heart Rate'].append({
                        'min': hr['min_bottom'],
                        'lower_median': hr['median_bottom'],
                        'upper_median': hr['median_top'],
                        'max': hr['max_top']
                    })
                else:
                    zone_data['Heart Rate'].append(None)

                # Power
                pr = zone['power_range']
                if all(pd.notna(pr[k]) for k in ['min_bottom', 'median_bottom', 'median_top', 'max_top']):
                    zone_data['Power'].append({
                        'min': pr['min_bottom'],
                        'lower_median': pr['median_bottom'],
                        'upper_median': pr['median_top'],
                        'max': pr['max_top']
                    })
                else:
                    zone_data['Power'].append(None)

                # Pace (convert from power, inverted)
                if all(pd.notna(pr[k]) for k in ['min_bottom', 'median_bottom', 'median_top', 'max_top']):
                    from training_zones import calculate_pace_from_power
                    zone_data['Pace'].append({
                        'min': calculate_pace_from_power(pr['max_top']),  # inverted
                        'lower_median': calculate_pace_from_power(pr['median_top']),  # inverted
                        'upper_median': calculate_pace_from_power(pr['median_bottom']),  # inverted
                        'max': calculate_pace_from_power(pr['min_bottom'])  # inverted
                    })
                else:
                    zone_data['Pace'].append(None)

                # Lactate
                lac = zone.get('lactate_range')
                if lac and all(pd.notna(lac[k]) for k in ['min_bottom', 'median_bottom', 'median_top', 'max_top']):
                    zone_data['Lactate'].append({
                        'min': lac['min_bottom'],
                        'lower_median': lac['median_bottom'],
                        'upper_median': lac['median_top'],
                        'max': lac['max_top']
                    })
                else:
                    zone_data['Lactate'].append(None)

                # RPE - we don't have this data, so create placeholder
                zone_data['RPE'].append(None)

            # Create HTML/CSS visualization
            html = generate_zones_html(zone_data, zone_colors, zone_names, metrics, metric_units, zones, results)
            st.components.v1.html(html, height=1400, scrolling=True)
        else:
            st.info("Add erg test results to see training zones visualization")
    else:
        st.info("Add test data and erg test results to see training zones visualization")


def find_method_for_value(results_df, metric_col, value, zone_methods):
    """Find which method contributed a specific value."""
    if pd.isna(value):
        return "Unknown"

    # Filter to methods in this zone
    filtered = results_df[results_df['Method'].isin(zone_methods)]

    # Find the closest match
    closest = filtered.iloc[(filtered[metric_col] - value).abs().argsort()[:1]]
    if not closest.empty and abs(closest[metric_col].iloc[0] - value) < 0.1:  # Tolerance
        return closest['Method'].iloc[0]
    return "Unknown"


def generate_zones_html(zone_data, zone_colors, zone_names, metrics, metric_units, zones, results):
    """Generate HTML/CSS for the zones visualization."""

    # Define method groups for each zone (from training_zones.py)
    zone_method_map = {
        0: ['LTP1', 'Bsln + 0.5', 'Bsln + 1.0', 'OBLA 2.0', 'Log-log'],  # Z1-Z2 use LT1
        1: ['LTP1', 'Bsln + 0.5', 'Bsln + 1.0', 'OBLA 2.0', 'Log-log'],
        2: ['LTP1', 'Bsln + 0.5', 'Bsln + 1.0', 'OBLA 2.0', 'Log-log'],  # Z3 uses LT1 for min
        3: ['LTP2', 'Dmax', 'ModDmax', 'Exp-Dmax', 'OBLA 4.0'],  # Z4-Z5 use LT2
        4: ['LTP2', 'Dmax', 'ModDmax', 'Exp-Dmax', 'OBLA 4.0'],
        5: [],  # Z6-Z7 use max values
        6: []
    }

    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', Oxygen, Ubuntu, Cantarell, sans-serif;
            background: transparent;
            padding: 0;
            margin: 0;
        }

        .container {
            max-width: 100%;
            background: transparent;
            padding: 0;
        }

        .metric-section {
            margin-bottom: 40px;
            background: #ffffff;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
            transition: box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .metric-section:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.12), 0 2px 4px rgba(0,0,0,0.08);
        }

        .metric-title {
            font-size: 14px;
            font-weight: 500;
            color: rgba(0, 0, 0, 0.87);
            margin-bottom: 12px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .metric-unit {
            font-size: 12px;
            color: rgba(0, 0, 0, 0.6);
            margin-left: 8px;
            font-weight: 400;
        }

        .zones-container {
            display: flex;
            width: 100%;
            margin-top: 12px;
            position: relative;
        }

        .zone {
            display: flex;
            flex-direction: column;
            position: relative;
            gap: 3px;
        }

        .bleed-bar-top {
            height: 10px;
            display: flex;
            justify-content: flex-start;
            position: relative;
        }

        .main-bar {
            height: 56px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2), 0 1px 2px rgba(0,0,0,0.12);
            color: white;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            padding: 4px 8px;
            gap: 2px;
            z-index: 2;
        }

        .zone-number {
            font-size: 11px;
            font-weight: 500;
            line-height: 1;
            letter-spacing: 0.5px;
            opacity: 0.95;
        }

        .zone-range {
            line-height: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 100%;
            font-weight: 600;
        }

        .main-bar:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.24), 0 2px 4px rgba(0,0,0,0.16);
            z-index: 10;
        }

        .bleed-bar-bottom {
            height: 10px;
            display: flex;
            justify-content: flex-end;
            position: relative;
        }

        .bleed-segment {
            height: 100%;
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 2px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.15), 0 0 1px rgba(0,0,0,0.1);
            position: relative;
        }

        .bleed-segment::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(180deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0) 100%);
            border-radius: 2px;
            pointer-events: none;
        }

        .bleed-segment:hover {
            opacity: 0.75 !important;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2), 0 1px 3px rgba(0,0,0,0.15);
        }

        .bleed-segment:active {
            transform: translateY(0);
            box-shadow: 0 1px 2px rgba(0,0,0,0.15);
        }

        .scale-markers {
            display: flex;
            justify-content: space-between;
            margin-top: 12px;
            font-size: 11px;
            color: rgba(0, 0, 0, 0.54);
            font-weight: 400;
        }

        .legend {
            margin-top: 40px;
            padding: 24px;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
        }

        .legend-title {
            font-size: 14px;
            font-weight: 500;
            color: rgba(0, 0, 0, 0.87);
            margin-bottom: 16px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .legend-items {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
            transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .legend-item:hover {
            transform: translateX(2px);
        }

        .legend-color {
            width: 24px;
            height: 24px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }

        .legend-label {
            font-size: 13px;
            color: rgba(0, 0, 0, 0.87);
            font-weight: 400;
        }

        .tooltip {
            position: fixed;
            background: rgba(33, 33, 33, 0.95);
            color: white;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 12px;
            max-width: 250px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 1000;
            line-height: 1.6;
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        }

        .tooltip.visible {
            opacity: 1;
        }

        .tooltip-header {
            font-weight: 600;
            margin-bottom: 6px;
            font-size: 13px;
        }

        .tooltip-line {
            margin-bottom: 3px;
        }
    </style>
    </head>
    <body>
    <div class="container">
    """

    # Generate each metric section
    for metric_idx, metric in enumerate(metrics):
        data = zone_data[metric]
        unit = metric_units[metric_idx]

        # Skip if no valid data
        if not any(data):
            continue

        # Calculate total range
        valid_data = [d for d in data if d is not None]
        if not valid_data:
            continue

        min_val = min(d['min'] for d in valid_data)
        max_val = max(d['max'] for d in valid_data)
        total_range = max_val - min_val

        if total_range == 0:
            continue

        html += f"""
        <div class="metric-section">
            <div class="metric-title">{metric} <span class="metric-unit">{unit}</span></div>
            <div class="zones-container" id="zones-{metric_idx}">
        """

        # Generate each zone
        for zone_idx, zone_info in enumerate(data):
            if zone_info is None:
                continue

            # Calculate widths
            median_range = zone_info['upper_median'] - zone_info['lower_median']
            width_pct = (median_range / total_range) * 100

            # Calculate bleed widths (as percentage of this zone's width)
            left_bleed = zone_info['lower_median'] - zone_info['min']
            right_bleed = zone_info['max'] - zone_info['upper_median']

            if median_range > 0:
                left_bleed_pct = (left_bleed / median_range) * 100
                right_bleed_pct = (right_bleed / median_range) * 100
            else:
                left_bleed_pct = 0
                right_bleed_pct = 0

            center_gap_pct = 100 - left_bleed_pct - right_bleed_pct

            # Format display values
            if metric == 'Pace':
                lower_display = f"{int(zone_info['lower_median']//60)}:{zone_info['lower_median']%60:04.1f}"
                upper_display = f"{int(zone_info['upper_median']//60)}:{zone_info['upper_median']%60:04.1f}"
                min_display = f"{int(zone_info['min']//60)}:{zone_info['min']%60:04.1f}"
                max_display = f"{int(zone_info['max']//60)}:{zone_info['max']%60:04.1f}"
                range_text = f"{lower_display}-{upper_display}"
            elif metric == 'Lactate':
                range_text = f"{zone_info['lower_median']:.1f}-{zone_info['upper_median']:.1f}"
                min_display = f"{zone_info['min']:.1f}"
                max_display = f"{zone_info['max']:.1f}"
            else:
                range_text = f"{int(zone_info['lower_median'])}-{int(zone_info['upper_median'])}"
                min_display = f"{int(zone_info['min'])}"
                max_display = f"{int(zone_info['max'])}"

            # Calculate font size for range based on zone width
            if width_pct > 5:
                font_size = "13px"
            elif width_pct > 3:
                font_size = "11px"
            elif width_pct > 2:
                font_size = "9px"
            else:
                font_size = "8px"

            zone_number = f"Z{zone_idx + 1}"

            color = zone_colors[zone_idx]
            prev_color = zone_colors[zone_idx - 1] if zone_idx > 0 else color
            next_color = zone_colors[zone_idx + 1] if zone_idx < 6 else color

            # Define metric column mapping for method lookup
            metric_col_map = {
                'Heart Rate': 'Heart Rate (bpm)',
                'Power': 'Intensity (W)',
                'Lactate': 'Lactate (mmol/L)'
            }

            html += f"""
                <div class="zone" style="width: {width_pct}%;">
                    <div class="bleed-bar-top">
            """

            # Top bar: Left bleed (previous zone's max extending into this zone)
            if zone_idx > 0 and left_bleed_pct > 0:
                # This shows the current zone's min, but it represents the previous zone's max overlapping
                # Find which method from the PREVIOUS zone caused this
                col_name = metric_col_map.get(metric)
                prev_zone_methods = zone_method_map.get(zone_idx - 1, [])
                method_name = find_method_for_value(results, col_name, zone_info['min'], prev_zone_methods) if col_name else "Unknown"

                html += f"""
                        <div class="bleed-segment"
                             style="width: {left_bleed_pct}%; background-color: {prev_color}; opacity: 0.5;"
                             data-zone-from="{zone_idx}"
                             data-zone-to="{zone_idx + 1}"
                             data-data-type="{metric}"
                             data-method="{method_name}"
                             data-outlier="{min_display}">
                        </div>
                """

            html += f"""
                    </div>
                    <div class="main-bar" style="background-color: {color};"
                         data-zone="{zone_idx + 1}"
                         data-name="{zone_names[zone_idx]}"
                         data-median-range="{range_text}"
                         data-full-range="{min_display}-{max_display}"
                         data-metric="{metric}">
                        <div class="zone-number">{zone_number}</div>
                        <div class="zone-range" style="font-size: {font_size};">{range_text}</div>
                    </div>
                    <div class="bleed-bar-bottom">
            """

            # Bottom bar: Right bleed (next zone's min extending into this zone)
            if zone_idx < 6 and right_bleed_pct > 0:
                # This shows the current zone's max, but it represents the next zone's min overlapping
                # Find which method from the NEXT zone caused this
                col_name = metric_col_map.get(metric)
                next_zone_methods = zone_method_map.get(zone_idx + 1, [])
                method_name = find_method_for_value(results, col_name, zone_info['max'], next_zone_methods) if col_name else "Unknown"

                html += f"""
                        <div class="bleed-segment"
                             style="width: {right_bleed_pct}%; background-color: {next_color}; opacity: 0.5;"
                             data-zone-from="{zone_idx + 2}"
                             data-zone-to="{zone_idx + 1}"
                             data-data-type="{metric}"
                             data-method="{method_name}"
                             data-outlier="{max_display}">
                        </div>
                """

            html += """
                    </div>
                </div>
            """

        # Scale markers
        if metric == 'Pace':
            min_display = f"{int(min_val//60)}:{min_val%60:04.1f}"
            max_display = f"{int(max_val//60)}:{max_val%60:04.1f}"
        elif metric == 'Lactate':
            min_display = f"{min_val:.1f}"
            max_display = f"{max_val:.1f}"
        else:
            min_display = f"{int(min_val)}"
            max_display = f"{int(max_val)}"

        html += f"""
            </div>
            <div class="scale-markers">
                <span>{min_display}</span>
                <span>{max_display}</span>
            </div>
        </div>
        """

    # Legend
    html += """
        <div class="legend">
            <div class="legend-title">Training Zones</div>
            <div class="legend-items">
    """

    for i, (color, name) in enumerate(zip(zone_colors, zone_names)):
        html += f"""
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {color};"></div>
                    <div class="legend-label">Zone {i + 1}: {name}</div>
                </div>
        """

    html += """
            </div>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script>
        const tooltip = document.getElementById('tooltip');

        // Main bar hover
        document.querySelectorAll('.main-bar').forEach(bar => {
            bar.addEventListener('mouseenter', (e) => {
                const zone = bar.dataset.zone;
                const name = bar.dataset.name;
                const medianRange = bar.dataset.medianRange;
                const fullRange = bar.dataset.fullRange;
                const metric = bar.dataset.metric;

                tooltip.innerHTML = `
                    <div class="tooltip-header">Zone ${zone}: ${name}</div>
                    <div class="tooltip-line">Median Range: ${medianRange}</div>
                    <div class="tooltip-line">Full Range: ${fullRange}</div>
                `;
                tooltip.classList.add('visible');
            });

            bar.addEventListener('mousemove', (e) => {
                tooltip.style.left = (e.clientX + 15) + 'px';
                tooltip.style.top = (e.clientY + 15) + 'px';
            });

            bar.addEventListener('mouseleave', () => {
                tooltip.classList.remove('visible');
            });
        });

        // Bleed segment hover
        document.querySelectorAll('.bleed-segment').forEach(segment => {
            segment.addEventListener('mouseenter', (e) => {
                const zoneFrom = segment.dataset.zoneFrom;
                const zoneTo = segment.dataset.zoneTo;
                const dataType = segment.dataset.dataType;
                const method = segment.dataset.method;
                const outlier = segment.dataset.outlier;

                // Determine if this is a min or max outlier
                // If zoneFrom > zoneTo, it's zoneFrom's minimum extending left into zoneTo
                // If zoneFrom < zoneTo, it's zoneFrom's maximum extending right into zoneTo
                const outlierType = parseInt(zoneFrom) > parseInt(zoneTo) ? 'minimum' : 'maximum';

                tooltip.innerHTML = `
                    <div class="tooltip-header">Potential Zone Overlap</div>
                    <div class="tooltip-line">Zone ${zoneFrom} ${outlierType} might extend into Zone ${zoneTo}</div>
                    <div class="tooltip-line" style="margin-top: 6px;"><strong>${method}</strong> method suggests ${dataType} ${outlierType}: ${outlier}</div>
                `;
                tooltip.classList.add('visible');
            });

            segment.addEventListener('mousemove', (e) => {
                tooltip.style.left = (e.clientX + 15) + 'px';
                tooltip.style.top = (e.clientY + 15) + 'px';
            });

            segment.addEventListener('mouseleave', () => {
                tooltip.classList.remove('visible');
            });
        });
    </script>
    </body>
    </html>
    """

    return html
