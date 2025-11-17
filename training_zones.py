#!/usr/bin/env python3
"""
Training Zone Calculation
Ported from old_site/js/tom-zones.js

Based primarily on Coggan Power Zones
https://www.trainingpeaks.com/blog/power-training-levels/
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.interpolate import interp1d


def calculate_power_from_time(time_seconds: float, meters: float) -> float:
    """
    Calculate power from time and distance.

    Power = 2.80 / pace³ where pace is time per meter

    Args:
        time_seconds: Time in seconds
        meters: Distance in meters

    Returns:
        Power in watts
    """
    pace = time_seconds / meters  # Time per meter
    return 2.80 / (pace ** 3)


def calculate_pace_from_power(power: float) -> float:
    """
    Calculate pace per 500m from power.

    Args:
        power: Power in watts

    Returns:
        Pace in seconds per 500m
    """
    pace_per_meter = (2.80 / power) ** (1/3)
    return pace_per_meter * 500


def calculate_critical_power(erg_tests: List[Dict]) -> Optional[float]:
    """
    Calculate Critical Power from erg test results.

    Uses all pairs of tests and averages the results.
    CP = (time1*power1 - time2*power2) / (time1 - time2)

    Args:
        erg_tests: List of dicts with 'time' (seconds) and 'meters' keys

    Returns:
        Average Critical Power in watts, or None if insufficient data
    """
    if len(erg_tests) < 2:
        return None

    # Calculate power for each test
    tests_with_power = []
    for test in erg_tests:
        power = calculate_power_from_time(test['time'], test['meters'])
        tests_with_power.append({
            **test,
            'power': power
        })

    # Calculate CP for all pairs
    critical_powers = []
    for i in range(len(tests_with_power)):
        for j in range(i + 1, len(tests_with_power)):
            test1 = tests_with_power[i]
            test2 = tests_with_power[j]

            t1, p1 = test1['time'], test1['power']
            t2, p2 = test2['time'], test2['power']

            cp = (t1 * p1 - t2 * p2) / (t1 - t2)
            critical_powers.append(cp)

    # Return average
    return np.mean(critical_powers)


def safe_round(value: float, decimals: int = 0) -> float:
    """
    Safely round a value, handling infinity and NaN.

    Args:
        value: Value to round
        decimals: Number of decimal places

    Returns:
        Rounded value, or NaN if input is infinity or NaN
    """
    if np.isnan(value) or np.isinf(value):
        return np.nan
    return round(value, decimals)


def interpolate_at_value(lactate_results: pd.DataFrame, field: str, target_value: float) -> Dict[str, float]:
    """
    Interpolate power, HR, and lactate at a given field value.

    Args:
        lactate_results: DataFrame with analysis results
        field: Field to interpolate on ('power', 'hr', or 'lactate')
        target_value: Target value to interpolate at

    Returns:
        Dict with 'power', 'hr', 'lactate' at the target value
    """
    # Map column names
    field_map = {
        'power': 'Intensity (W)',
        'hr': 'Heart Rate (bpm)',
        'lactate': 'Lactate (mmol/L)'
    }

    # Get column name
    x_col = field_map.get(field, field)

    # Sort by the field we're interpolating on
    df = lactate_results.sort_values(x_col).copy()

    # Create interpolation functions
    x_values = df[x_col].values

    result = {}

    # Interpolate each metric
    for key, col in field_map.items():
        if col in df.columns:
            y_values = df[col].values

            # Remove NaN values
            mask = ~(np.isnan(x_values) | np.isnan(y_values))
            if mask.sum() < 2:
                result[key] = np.nan
                continue

            x_clean = x_values[mask]
            y_clean = y_values[mask]

            # Check if target is within bounds
            if target_value < x_clean.min() or target_value > x_clean.max():
                # Extrapolate
                f = interp1d(x_clean, y_clean, kind='linear', fill_value='extrapolate')
            else:
                f = interp1d(x_clean, y_clean, kind='linear')

            result[key] = float(f(target_value))
        else:
            result[key] = np.nan

    return result


def set_lt_values(lactate_results: pd.DataFrame, target_method: str, methods: List[str]) -> Dict:
    """
    Calculate LT values (target, min, max) from lactate threshold results.

    Args:
        lactate_results: DataFrame with analysis results
        target_method: Primary method to use for target values
        methods: List of method names to consider

    Returns:
        Dict with 'target', 'min', 'max' sub-dicts containing power, hr, lactate
    """
    # Filter to relevant methods
    filtered = lactate_results[lactate_results['Method'].isin(methods)].copy()

    if filtered.empty:
        return {
            'target': {'power': np.nan, 'hr': np.nan, 'lactate': np.nan},
            'min': {'power': np.nan, 'hr': np.nan, 'lactate': np.nan},
            'max': {'power': np.nan, 'hr': np.nan, 'lactate': np.nan}
        }

    # Get target values
    target_row = filtered[filtered['Method'] == target_method]
    if not target_row.empty:
        target = {
            'power': float(target_row['Intensity (W)'].iloc[0]),
            'hr': float(target_row['Heart Rate (bpm)'].iloc[0]),
            'lactate': float(target_row['Lactate (mmol/L)'].iloc[0])
        }
    else:
        # Use mean if target method not found
        target = {
            'power': float(filtered['Intensity (W)'].mean()),
            'hr': float(filtered['Heart Rate (bpm)'].mean()),
            'lactate': float(filtered['Lactate (mmol/L)'].mean())
        }

    # Get min and max
    lt_values = {
        'target': target,
        'min': {
            'power': float(filtered['Intensity (W)'].min()),
            'hr': float(filtered['Heart Rate (bpm)'].min()),
            'lactate': float(filtered['Lactate (mmol/L)'].min())
        },
        'max': {
            'power': float(filtered['Intensity (W)'].max()),
            'hr': float(filtered['Heart Rate (bpm)'].max()),
            'lactate': float(filtered['Lactate (mmol/L)'].max())
        }
    }

    return lt_values


def calculate_training_zones(lactate_results: pd.DataFrame,
                            erg_tests: List[Dict],
                            max_hr: float,
                            max_power: float) -> List[Dict]:
    """
    Calculate 7 training zones based on lactate test results and erg tests.

    Args:
        lactate_results: DataFrame from analyze_lactate_thresholds()
        erg_tests: List of dicts with 'time' (seconds) and 'meters' keys
                   e.g., [{'time': 180, 'meters': 1000}, {'time': 900, 'meters': 5000}]
        max_hr: Maximum heart rate (bpm)
        max_power: Maximum power output (watts)

    Returns:
        List of zone dicts with zone, description, calculation, and ranges
    """
    # Define LT1 methods (Aerobic Threshold)
    LT1_methods = ['LTP1', 'Bsln + 0.5', 'Bsln + 1.0', 'OBLA 2.0', 'Log-log']
    LT1 = set_lt_values(lactate_results, 'LTP1', LT1_methods)

    # Calculate 85% and 55% of LT1 HR
    LT1_85 = interpolate_at_value(lactate_results, 'hr', LT1['target']['hr'] * 0.85)
    LT1_85_min = interpolate_at_value(lactate_results, 'hr', LT1['min']['hr'] * 0.85)
    LT1_85_max = interpolate_at_value(lactate_results, 'hr', LT1['max']['hr'] * 0.85)

    LT1_55 = interpolate_at_value(lactate_results, 'hr', LT1['target']['hr'] * 0.55)
    LT1_55_min = interpolate_at_value(lactate_results, 'hr', LT1['min']['hr'] * 0.55)
    LT1_55_max = interpolate_at_value(lactate_results, 'hr', LT1['max']['hr'] * 0.55)

    # Define LT2 methods (Anaerobic Threshold)
    LT2_methods = ['LTP2', 'OBLA 4.0', 'Dmax', 'ModDmax', 'Exp-Dmax',
                   'Log-Poly-ModDmax', 'Log-Exp-ModDmax']
    LT2 = set_lt_values(lactate_results, 'Log-Poly-ModDmax', LT2_methods)

    # Calculate Critical Power
    critical_power = calculate_critical_power(erg_tests)
    if critical_power:
        CP = interpolate_at_value(lactate_results, 'power', critical_power)
    else:
        CP = {'power': np.nan, 'hr': np.nan, 'lactate': np.nan}

    # Calculate 92.5% of max HR
    HR925 = interpolate_at_value(lactate_results, 'hr', 0.925 * max_hr)

    # 1k test power
    test_1k = [t for t in erg_tests if t['meters'] == 1000]
    if test_1k:
        test_1k_power = calculate_power_from_time(test_1k[0]['time'], 1000)
        Test1k = interpolate_at_value(lactate_results, 'power', test_1k_power)
    else:
        Test1k = {'power': max_power, 'hr': max_hr, 'lactate': np.nan}

    # Define 7 training zones
    zones = [
        {
            'zone': 'Z1',
            'description': 'Easy recovery. Not hard enough to be productive',
            'calculation': 'Between 55% and 85% of LT1, based on HR',
            'power_range': {
                'min_bottom': safe_round(LT1_55_min['power']),
                'median_bottom': safe_round(LT1_55['power']),
                'median_top': safe_round(LT1_85['power']),
                'max_top': safe_round(LT1_85_max['power'])
            },
            'hr_range': {
                'min_bottom': safe_round(LT1_55_min['hr']),
                'median_bottom': safe_round(LT1_55['hr']),
                'median_top': safe_round(LT1_85['hr']),
                'max_top': safe_round(LT1_85_max['hr'])
            },
            'lactate_range': {
                'min_bottom': max(safe_round(LT1_55_min['lactate'], 1), 0),
                'median_bottom': max(safe_round(LT1_55['lactate'], 1), 0),
                'median_top': safe_round(LT1_85['lactate'], 1),
                'max_top': safe_round(LT1_85_max['lactate'], 1)
            }
        },
        {
            'zone': 'Z2',
            'description': 'Maximally productive zone. Hard as possible without going beyond LT1',
            'calculation': 'Between 85% and 100% of LT1',
            'power_range': {
                'min_bottom': safe_round(LT1_85_min['power']),
                'median_bottom': safe_round(LT1_85['power']),
                'median_top': safe_round(LT1['target']['power']),
                'max_top': safe_round(LT1['max']['power'])
            },
            'hr_range': {
                'min_bottom': safe_round(LT1_85_min['hr']),
                'median_bottom': safe_round(LT1_85['hr']),
                'median_top': safe_round(LT1['target']['hr']),
                'max_top': safe_round(LT1['max']['hr'])
            },
            'lactate_range': {
                'min_bottom': safe_round(LT1_85_min['lactate'], 1),
                'median_bottom': safe_round(LT1_85['lactate'], 1),
                'median_top': safe_round(LT1['target']['lactate'], 1),
                'max_top': safe_round(LT1['max']['lactate'], 1)
            }
        },
        {
            'zone': 'Z3',
            'description': 'Between LT1 and LT2. Lactate in steady state but elevated',
            'calculation': 'Up to the lesser of LT2 and Critical Power (usually LT2)',
            'power_range': {
                'min_bottom': safe_round(LT1['min']['power']),
                'median_bottom': safe_round(LT1['target']['power']),
                'median_top': safe_round(min(CP['power'], LT2['target']['power'])),
                'max_top': safe_round(min(CP['power'], LT2['max']['power']))
            },
            'hr_range': {
                'min_bottom': safe_round(LT1['min']['hr']),
                'median_bottom': safe_round(LT1['target']['hr']),
                'median_top': safe_round(min(CP['hr'], LT2['target']['hr'])),
                'max_top': safe_round(min(CP['hr'], LT2['max']['hr']))
            },
            'lactate_range': {
                'min_bottom': safe_round(LT1['min']['lactate'], 1),
                'median_bottom': safe_round(LT1['target']['lactate'], 1),
                'median_top': safe_round(min(CP['lactate'], LT2['target']['lactate']), 1),
                'max_top': safe_round(min(CP['lactate'], LT2['max']['lactate']), 1)
            }
        },
        {
            'zone': 'Z4',
            'description': 'Over MLSS. Lactate not steady and will increase over time',
            'calculation': 'Up to the greater of LT2 and Critical Power (usually Critical Power)',
            'power_range': {
                'min_bottom': safe_round(min(CP['power'], LT2['min']['power'])),
                'median_bottom': safe_round(min(CP['power'], LT2['target']['power'])),
                'median_top': safe_round(max(CP['power'], LT2['target']['power'])),
                'max_top': safe_round(max(CP['power'], LT2['max']['power']))
            },
            'hr_range': {
                'min_bottom': safe_round(min(CP['hr'], LT2['min']['hr'])),
                'median_bottom': safe_round(min(CP['hr'], LT2['target']['hr'])),
                'median_top': safe_round(max(CP['hr'], LT2['target']['hr'])),
                'max_top': safe_round(max(CP['hr'], LT2['max']['hr']))
            },
            'lactate_range': {
                'min_bottom': safe_round(min(CP['lactate'], LT2['min']['lactate']), 1),
                'median_bottom': safe_round(min(CP['lactate'], LT2['target']['lactate']), 1),
                'median_top': safe_round(max(CP['lactate'], LT2['target']['lactate']), 1),
                'max_top': safe_round(max(CP['lactate'], LT2['max']['lactate']), 1)
            }
        },
        {
            'zone': 'Z5',
            'description': 'Over LT2/CP, aiming around 90% of VO2max/HR (87.5%-92.5%). Very productive zone',
            'calculation': 'Over LT2 and Critical Power. Up to 92.5% max HR',
            'power_range': {
                'min_bottom': safe_round(min(CP['power'], LT2['target']['power'])),
                'median_bottom': safe_round(max(CP['power'], LT2['target']['power'])),
                'median_top': safe_round(HR925['power']),
                'max_top': safe_round(HR925['power'])
            },
            'hr_range': {
                'min_bottom': safe_round(max(CP['hr'], LT2['min']['hr'])),
                'median_bottom': safe_round(max(CP['hr'], LT2['min']['hr'])),
                'median_top': safe_round(HR925['hr']),
                'max_top': safe_round(HR925['hr'])
            },
            'lactate_range': {
                'min_bottom': safe_round(max(CP['lactate'], LT2['min']['lactate']), 1),
                'median_bottom': safe_round(max(CP['lactate'], LT2['min']['lactate']), 1),
                'median_top': safe_round(HR925['lactate'], 1),
                'max_top': safe_round(HR925['lactate'], 1)
            }
        },
        {
            'zone': 'Z6',
            'description': 'Above 92.5% max HR, but below 1k max pace',
            'calculation': '92.5% max HR up to 1k test score',
            'power_range': {
                'min_bottom': safe_round(HR925['power']),
                'median_bottom': safe_round(HR925['power']),
                'median_top': safe_round(Test1k['power']),
                'max_top': safe_round(Test1k['power'])
            },
            'hr_range': {
                'min_bottom': safe_round(HR925['hr']),
                'median_bottom': safe_round(HR925['hr']),
                'median_top': min(safe_round(Test1k['hr']), max_hr),
                'max_top': min(safe_round(Test1k['hr']), max_hr)
            },
            'lactate_range': {
                'min_bottom': safe_round(HR925['lactate'], 1),
                'median_bottom': safe_round(HR925['lactate'], 1),
                'median_top': safe_round(Test1k['lactate'], 1),
                'max_top': safe_round(Test1k['lactate'], 1)
            }
        },
        {
            'zone': 'Z7',
            'description': 'Anaerobic - sustainable for no more than 3 minutes. HR and Lactate not useful',
            'calculation': '1k test score and harder',
            'power_range': {
                'min_bottom': safe_round(Test1k['power']),
                'median_bottom': safe_round(Test1k['power']),
                'median_top': safe_round(max_power),
                'max_top': safe_round(max_power)
            },
            'hr_range': None,  # Not useful in Z7
            'lactate_range': None  # Not useful in Z7
        }
    ]

    return zones
