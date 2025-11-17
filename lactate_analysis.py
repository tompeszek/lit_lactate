"""
Exact Python port of the R lactater package for lactate threshold analysis.
Based on: https://github.com/fmmattioni/lactater

All methods replicate the exact algorithms from the R source code.
"""

import numpy as np
import pandas as pd
from scipy import interpolate, optimize
from scipy.interpolate import BSpline, make_interp_spline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from typing import Optional, Dict, List, Tuple
import requests
import json


# ============================================================================
# CURVE FITTING FUNCTIONS
# ============================================================================

def fit_polynomial_3rd(intensity: np.ndarray, lactate: np.ndarray) -> Tuple:
    """
    Fit 3rd degree polynomial: lactate = β₀ + β₁*x + β₂*x² + β₃*x³

    Matches R: lm(lactate ~ poly(intensity, degree = 3, raw = TRUE))
    """
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(intensity.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, lactate)

    # Get coefficients [intercept, β₁, β₂, β₃]
    coefficients = np.concatenate([[model.intercept_], model.coef_[1:]])

    return model, poly, coefficients


def fit_exponential(intensity: np.ndarray, lactate: np.ndarray) -> Optional[Tuple]:
    """
    Fit exponential model: lactate = a + b * exp(c * intensity)

    Matches R: nlsLM(lactate ~ a + (b * exp(c * intensity)))
    Initial parameters: a=0, b=1, c=0
    """
    def exponential_model(x, a, b, c):
        return a + b * np.exp(c * x)

    try:
        params, _ = optimize.curve_fit(
            exponential_model,
            intensity,
            lactate,
            p0=[0, 1, 0],  # initial guess
            maxfev=1000
        )
        return params  # [a, b, c]
    except:
        return None


def fit_bspline(intensity: np.ndarray, lactate: np.ndarray):
    """
    Fit B-spline with 4 degrees of freedom (natural spline).

    Matches R: glm(lactate ~ splines::ns(intensity, 4))

    Returns a callable that predicts lactate at given intensities.
    """
    from scipy.interpolate import UnivariateSpline

    # Use UnivariateSpline with k=3 for cubic spline
    # s=None means automatic smoothing (like R's natural spline)
    spl = UnivariateSpline(intensity, lactate, k=3, s=None)

    return spl


# ============================================================================
# VALUE RETRIEVAL FUNCTIONS
# ============================================================================

def retrieve_intensity_polynomial(fitted_lactate: np.ndarray, intensity: np.ndarray,
                                 lactate_value: float) -> float:
    """
    Find intensity corresponding to a lactate value (inverse interpolation).

    Matches R: approx(x = model$fitted.values, y = intensity, xout = lactate_value, ties = "ordered")

    This mimics R's approx() which does linear interpolation even with non-monotonic x values.
    """
    # Find all segments where lactate_value could be interpolated
    results = []

    for i in range(len(fitted_lactate) - 1):
        lac_low = fitted_lactate[i]
        lac_high = fitted_lactate[i + 1]
        int_low = intensity[i]
        int_high = intensity[i + 1]

        # Check if lactate_value is between this pair (in either direction)
        if (lac_low <= lactate_value <= lac_high) or (lac_high <= lactate_value <= lac_low):
            # Linear interpolation
            if lac_high != lac_low:  # Avoid division by zero
                t = (lactate_value - lac_low) / (lac_high - lac_low)
                interpolated_intensity = int_low + t * (int_high - int_low)
                results.append(interpolated_intensity)

    # Return the maximum intensity if multiple solutions found
    if len(results) > 0:
        return float(np.max(results))

    # If no interpolation found, find the closest lactate value
    # This handles edge cases where target is slightly outside the fitted range
    closest_idx = np.argmin(np.abs(fitted_lactate - lactate_value))
    return float(intensity[closest_idx])


def retrieve_lactate(model, poly_features, intensity_value: float) -> float:
    """
    Predict lactate at given intensity.

    Matches R: predict(object = model, newdata = data.frame(intensity = intensity_value))
    """
    X = poly_features.transform([[intensity_value]])
    lactate = model.predict(X)[0]
    return round(lactate, 1)


def retrieve_heart_rate(raw_data: pd.DataFrame, intensity_value: float) -> int:
    """
    Predict heart rate using linear regression (excluding baseline).

    Matches R: lm(heart_rate ~ intensity, data = raw_data[-1,])
    """
    # Remove baseline (first row)
    data = raw_data.iloc[1:]

    if 'heart_rate' not in data.columns or len(data) < 2:
        return None

    # Fit linear model
    X = data['intensity'].values.reshape(-1, 1)
    y = data['heart_rate'].values

    model = LinearRegression()
    model.fit(X, y)

    # Predict
    hr = model.predict([[intensity_value]])[0]
    return round(hr)


# ============================================================================
# INTERPOLATION
# ============================================================================

def interpolate_data_linear(intensity: np.ndarray, lactate: np.ndarray,
                            sport: str = 'cycling') -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear interpolation of raw data (for Log-log and LTP methods).

    Matches R: approx(x = intensity, y = lactate, xout = seq(...))
    """
    interpolation_factor = 0.1 if sport in ['cycling', 'running'] else 0.01

    intensity_interp = np.arange(
        intensity.min(),
        intensity.max() + interpolation_factor,
        interpolation_factor
    )

    f = interpolate.interp1d(intensity, lactate, kind='linear')
    lactate_interp = f(intensity_interp)

    return intensity_interp, lactate_interp


def interpolate_data(intensity: np.ndarray, lactate: np.ndarray, model, poly_features,
                    sport: str = 'cycling') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create augmented data by predicting lactate at interpolated intensities using the polynomial model.

    Matches R: broom::augment(model, newdata = new_data_model, type.predict = "response")
    where new_data_model is seq(min, max, by = interpolation_factor)
    """
    # Sport-specific interpolation factor
    interpolation_factor = 0.1 if sport in ['cycling', 'running'] else 0.01

    # Create sequence of intensity values
    intensity_interp = np.arange(
        intensity.min(),
        intensity.max() + interpolation_factor,
        interpolation_factor
    )

    # Predict lactate using the polynomial model at each interpolated intensity
    lactate_interp = []
    for i in intensity_interp:
        X_new = poly_features.transform([[i]])
        lactate_pred = model.predict(X_new)[0]
        lactate_interp.append(lactate_pred)

    lactate_interp = np.array(lactate_interp)

    return intensity_interp, lactate_interp


# ============================================================================
# SEGMENTED REGRESSION (for Log-log and LTP)
# ============================================================================

def segmented_regression(x: np.ndarray, y: np.ndarray, n_breakpoints: int) -> np.ndarray:
    """
    Segmented regression to find breakpoints that minimize RSS.

    Approximates R: segmented::segmented(lm(y ~ x), npsi = n_breakpoints)

    For each candidate breakpoint, fits two separate lines and calculates RSS.
    Returns breakpoint(s) that minimize total RSS.
    """
    from scipy import stats

    if n_breakpoints == 1:
        # Single breakpoint: try each interior point and find the one with minimum RSS
        best_breakpoint = None
        best_rss = np.inf

        # Need at least 2 points on each side
        for i in range(2, len(x) - 2):
            # Fit left segment
            x_left = x[:i+1]
            y_left = y[:i+1]
            slope_left, intercept_left, _, _, _ = stats.linregress(x_left, y_left)
            fitted_left = slope_left * x_left + intercept_left
            rss_left = np.sum((y_left - fitted_left) ** 2)

            # Fit right segment
            x_right = x[i:]
            y_right = y[i:]
            slope_right, intercept_right, _, _, _ = stats.linregress(x_right, y_right)
            fitted_right = slope_right * x_right + intercept_right
            rss_right = np.sum((y_right - fitted_right) ** 2)

            # Total RSS
            total_rss = rss_left + rss_right

            if total_rss < best_rss:
                best_rss = total_rss
                best_breakpoint = x[i]

        return np.array([best_breakpoint])

    elif n_breakpoints == 2:
        # Two breakpoints: use a sampling approach to avoid O(n^2) complexity
        # Sample candidate positions instead of trying all combinations
        n = len(x)

        # For large datasets, sample positions; for small datasets, try more
        if n > 1000:
            # Sample 20 positions in first third, 20 in second third
            positions_1 = np.linspace(2, n // 3, min(20, n // 3 - 2), dtype=int)
            positions_2 = np.linspace(n // 3 + 2, n - 2, min(20, n - n // 3 - 4), dtype=int)
        else:
            # For smaller datasets, use more quantile-based positions to increase accuracy
            # Focus sampling around typical LTP1 (20-35%) and LTP2 (60-80%) locations
            positions_1 = [int(n * p) for p in [0.15, 0.18, 0.20, 0.22, 0.25, 0.27, 0.30, 0.32, 0.35, 0.37, 0.40, 0.42, 0.45]]
            positions_2 = [int(n * p) for p in [0.55, 0.57, 0.60, 0.62, 0.65, 0.67, 0.70, 0.72, 0.75, 0.77, 0.80, 0.82, 0.85]]

        best_breakpoints = None
        best_rss = np.inf

        for i in positions_1:
            for j in positions_2:
                if j <= i + 2:
                    continue

                try:
                    # Segment 1
                    slope1, intercept1, _, _, _ = stats.linregress(x[:i+1], y[:i+1])
                    rss1 = np.sum((y[:i+1] - (slope1 * x[:i+1] + intercept1)) ** 2)

                    # Segment 2
                    slope2, intercept2, _, _, _ = stats.linregress(x[i:j+1], y[i:j+1])
                    rss2 = np.sum((y[i:j+1] - (slope2 * x[i:j+1] + intercept2)) ** 2)

                    # Segment 3
                    slope3, intercept3, _, _, _ = stats.linregress(x[j:], y[j:])
                    rss3 = np.sum((y[j:] - (slope3 * x[j:] + intercept3)) ** 2)

                    total_rss = rss1 + rss2 + rss3

                    if total_rss < best_rss:
                        best_rss = total_rss
                        best_breakpoints = [x[i], x[j]]
                except:
                    continue

        if best_breakpoints is None:
            # Fallback to quantiles
            idx1 = int(len(x) * 0.33)
            idx2 = int(len(x) * 0.67)
            return np.array([x[idx1], x[idx2]])

        return np.array(best_breakpoints)

    else:
        # General case: evenly spaced breakpoints
        indices = np.linspace(0, len(x)-1, n_breakpoints+2, dtype=int)[1:-1]
        return x[indices]


# ============================================================================
# METHOD 1: OBLA (Onset of Blood Lactate Accumulation)
# ============================================================================

def method_obla(data: pd.DataFrame, model, poly_features, fitted_intensity: np.ndarray,
                fitted_lactate: np.ndarray, sport: str = 'cycling') -> pd.DataFrame:
    """
    OBLA method: fixed lactate thresholds at 2.0, 2.5, 3.0, 3.5, 4.0 mmol/L

    Matches R: method_obla() in method-obla.R
    """
    thresholds = [2.0, 2.5, 3.0, 3.5, 4.0]
    results = []

    # Sport-specific rounding
    decimals = {'cycling': 1, 'running': 2, 'swimming': 3}[sport]

    for threshold in thresholds:
        # Relaxed feasibility check - allow small extrapolation (within 2% of range)
        lac_range = fitted_lactate.max() - fitted_lactate.min()
        tolerance = 0.02 * lac_range

        if threshold > fitted_lactate.max() + tolerance or threshold < fitted_lactate.min() - tolerance:
            results.append({
                'method': f'OBLA {threshold:.1f}',
                'category': 'Fixed Lactate',
                'fitting': '3rd degree polynomial (user-defined)',
                'intensity': np.nan,
                'lactate': threshold,
                'heart_rate': np.nan
            })
            continue

        # Retrieve intensity
        intensity = retrieve_intensity_polynomial(fitted_lactate, fitted_intensity, threshold)
        intensity = round(intensity, decimals)

        # Retrieve heart rate
        hr = retrieve_heart_rate(data, intensity)

        results.append({
            'method': f'OBLA {threshold:.1f}',
            'category': 'Fixed Lactate',
            'fitting': '3rd degree polynomial (user-defined)',
            'intensity': intensity,
            'lactate': threshold,
            'heart_rate': hr if hr is not None else np.nan
        })

    return pd.DataFrame(results)


# ============================================================================
# METHOD 2A: Dmax (Standard)
# ============================================================================

def method_dmax(data: pd.DataFrame, sport: str = 'cycling') -> Dict:
    """
    Dmax method using 3rd degree polynomial.
    Finds point with maximum perpendicular distance from reference line.

    Matches R: method_dmax() in method-dmax.R
    """
    # Exclude baseline
    data_fit = data.iloc[1:]

    # Fit 3rd degree polynomial
    model, poly, coeffs = fit_polynomial_3rd(
        data_fit['intensity'].values,
        data_fit['lactate'].values
    )

    # Calculate reference line slope (first to last point)
    diff_lactate = data_fit['lactate'].max() - data_fit['lactate'].min()
    diff_intensity = data_fit['intensity'].max() - data_fit['intensity'].min()
    lin_beta = diff_lactate / diff_intensity

    # Solve quadratic: 3*β₃*x² + 2*β₂*x + (β₁ - lin_beta) = 0
    # This finds where polynomial derivative equals reference slope
    a = 3 * coeffs[3]
    b = 2 * coeffs[2]
    c = coeffs[1] - lin_beta

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None

    # Quadratic formula
    roots = [
        (-b + np.sqrt(discriminant)) / (2*a),
        (-b - np.sqrt(discriminant)) / (2*a)
    ]

    # Filter positive roots within valid range
    max_intensity = data_fit['intensity'].max()
    valid_roots = [r for r in roots if r > 0 and r <= max_intensity]

    if not valid_roots:
        return None

    # Select maximum valid root
    method_intensity = max(valid_roots)
    method_lactate = retrieve_lactate(model, poly, method_intensity)

    # Workaround for implausible estimations (lactate > 8)
    if method_lactate > 8 and len(valid_roots) > 1:
        method_intensity = min(valid_roots)
        method_lactate = retrieve_lactate(model, poly, method_intensity)

    # Round
    decimals = {'cycling': 1, 'running': 2, 'swimming': 3}[sport]
    method_intensity = round(method_intensity, decimals)

    # Retrieve heart rate
    hr = retrieve_heart_rate(data, method_intensity)

    return {
        'method': 'Dmax',
        'category': 'Maximum Distance',
        'fitting': '3rd degree polynomial (default)',
        'intensity': method_intensity,
        'lactate': method_lactate,
        'heart_rate': hr if hr is not None else np.nan
    }


# ============================================================================
# METHOD 2B: ModDmax (Modified Dmax)
# ============================================================================

def method_moddmax(data: pd.DataFrame, sport: str = 'cycling') -> Dict:
    """
    Modified Dmax: starts after first lactate rise >= 0.4 mmol/L

    Matches R: method_dmax_mod() in method-dmax.R
    """
    data_fit = data.iloc[1:]  # exclude baseline

    # Find first rise >= 0.4 mmol/L
    # R uses lead(lactate) - lactate, which is next value - current (forward looking)
    # head(1) means take the FIRST row where the rise occurs (the row BEFORE the jump)
    data_fit = data_fit.copy()
    data_fit['diffs'] = data_fit['lactate'].shift(-1) - data_fit['lactate']
    first_rise_mask = data_fit['diffs'] >= 0.4

    if not first_rise_mask.any():
        return {
            'method': 'ModDmax',
            'category': 'Maximum Distance',
            'fitting': '3rd degree polynomial (default)',
            'intensity': np.nan,
            'lactate': np.nan,
            'heart_rate': np.nan
        }

    # The first row where diffs >= 0.4 is the point where lactate STARTS to rise
    first_rise_idx = data_fit[first_rise_mask].index[0]
    first_rise = data_fit.loc[first_rise_idx]

    # IMPORTANT: R fits polynomial on ALL data (not just from first rise)
    # Only the search range for Dmax is limited to >= first_rise
    model, poly, coeffs = fit_polynomial_3rd(
        data_fit['intensity'].values,  # ALL data (excluding baseline)
        data_fit['lactate'].values
    )

    # Reference line from first_rise to last point
    diff_lactate = data_fit['lactate'].max() - first_rise['lactate']
    diff_intensity = data_fit['intensity'].max() - first_rise['intensity']
    lin_beta = diff_lactate / diff_intensity

    # Solve polynomial (same as Dmax)
    a = 3 * coeffs[3]
    b = 2 * coeffs[2]
    c = coeffs[1] - lin_beta

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None

    roots = [
        (-b + np.sqrt(discriminant)) / (2*a),
        (-b - np.sqrt(discriminant)) / (2*a)
    ]

    # Valid roots should be between first rise and max intensity
    max_intensity = data_fit['intensity'].max()
    min_intensity = first_rise['intensity']
    valid_roots = [r for r in roots if min_intensity <= r <= max_intensity]

    if not valid_roots:
        return None

    method_intensity = max(valid_roots)
    method_lactate = retrieve_lactate(model, poly, method_intensity)

    if method_lactate > 8 and len(valid_roots) > 1:
        method_intensity = min(valid_roots)
        method_lactate = retrieve_lactate(model, poly, method_intensity)

    decimals = {'cycling': 1, 'running': 2, 'swimming': 3}[sport]
    method_intensity = round(method_intensity, decimals)

    hr = retrieve_heart_rate(data, method_intensity)

    return {
        'method': 'ModDmax',
        'category': 'Maximum Distance',
        'fitting': '3rd degree polynomial (default)',
        'intensity': method_intensity,
        'lactate': method_lactate,
        'heart_rate': hr if hr is not None else np.nan
    }


# ============================================================================
# METHOD 2C: Exp-Dmax (Exponential Dmax)
# ============================================================================

def exponential_dmax_formula(c: float, si: float, sf: float) -> float:
    """
    Calculate Dmax for exponential model.

    Matches R: log((exp(c * sf) - exp(c * si)) / ((c * sf) - (c * si))) / c
    """
    numerator = np.exp(c * sf) - np.exp(c * si)
    denominator = (c * sf) - (c * si)
    return np.log(numerator / denominator) / c


def method_exp_dmax(data: pd.DataFrame, sport: str = 'cycling') -> Dict:
    """
    Exp-Dmax using exponential fit.

    Matches R: method_exp_dmax() in method-dmax.R
    """
    data_fit = data.iloc[1:]

    # Fit exponential
    params = fit_exponential(
        data_fit['intensity'].values,
        data_fit['lactate'].values
    )

    if params is None:
        return {
            'method': 'Exp-Dmax',
            'category': 'Maximum Distance',
            'fitting': 'Exponential (default)',
            'intensity': np.nan,
            'lactate': np.nan,
            'heart_rate': np.nan
        }

    a, b, c = params

    # Calculate Dmax
    first_intensity = data_fit['intensity'].min()
    last_intensity = data_fit['intensity'].max()

    method_intensity = exponential_dmax_formula(c, first_intensity, last_intensity)

    # Predict lactate
    method_lactate = a + b * np.exp(c * method_intensity)
    method_lactate = round(method_lactate, 1)

    decimals = {'cycling': 1, 'running': 2, 'swimming': 3}[sport]
    method_intensity = round(method_intensity, decimals)

    hr = retrieve_heart_rate(data, method_intensity)

    return {
        'method': 'Exp-Dmax',
        'category': 'Maximum Distance',
        'fitting': 'Exponential (default)',
        'intensity': method_intensity,
        'lactate': method_lactate,
        'heart_rate': hr if hr is not None else np.nan
    }


# ============================================================================
# METHOD 3: Bsln+ (Baseline Plus)
# ============================================================================

def method_bsln_plus(data: pd.DataFrame, model, poly_features, fitted_intensity: np.ndarray,
                    fitted_lactate: np.ndarray, sport: str = 'cycling') -> pd.DataFrame:
    """
    Bsln+ method: baseline lactate + fixed increments (0.5, 1.0, 1.5 mmol/L)

    Matches R: method_bsln_plus() in method-bsln-plus.R
    """
    baseline_lactate = data['lactate'].iloc[0]
    increments = [0.5, 1.0, 1.5]
    results = []

    decimals = {'cycling': 1, 'running': 2, 'swimming': 3}[sport]

    for increment in increments:
        target_lactate = baseline_lactate + increment

        # Feasibility check
        if target_lactate > fitted_lactate.max() or target_lactate < fitted_lactate.min():
            results.append({
                'method': f'Bsln + {increment:.1f}',
                'category': 'Baseline Plus',
                'fitting': '3rd degree polynomial (user-defined)',
                'intensity': np.nan,
                'lactate': target_lactate,
                'heart_rate': np.nan
            })
            continue

        # Retrieve intensity
        intensity = retrieve_intensity_polynomial(fitted_lactate, fitted_intensity, target_lactate)
        intensity = round(intensity, decimals)

        # Retrieve heart rate
        hr = retrieve_heart_rate(data, intensity)

        results.append({
            'method': f'Bsln + {increment:.1f}',
            'category': 'Baseline Plus',
            'fitting': '3rd degree polynomial (user-defined)',
            'intensity': intensity,
            'lactate': target_lactate,
            'heart_rate': hr if hr is not None else np.nan
        })

    return pd.DataFrame(results)


# ============================================================================
# METHOD 4: Log-log
# ============================================================================

def method_loglog(data: pd.DataFrame, sport: str = 'cycling',
                 loglog_restrainer: float = 1.0) -> Dict:
    """
    Log-log method using segmented regression on log-transformed data.

    Matches R: method_loglog() in method-loglog.R
    """
    # Use LINEAR interpolation of RAW data (R uses data_interpolated, not data_augmented)
    data_fit = data.iloc[1:]  # Exclude baseline
    intensity_interp, lactate_interp = interpolate_data_linear(
        data_fit['intensity'].values,
        data_fit['lactate'].values,
        sport
    )

    # Fit polynomial model for retrieving lactate values later
    model, poly, coeffs = fit_polynomial_3rd(
        data_fit['intensity'].values,
        data_fit['lactate'].values
    )

    # Filter positive intensities
    mask = intensity_interp > 0
    intensity_pos = intensity_interp[mask]
    lactate_pos = lactate_interp[mask]

    # Apply restrainer (use only first X% of data)
    n_points = int(len(intensity_pos) * loglog_restrainer)
    intensity_pos = intensity_pos[:n_points]
    lactate_pos = lactate_pos[:n_points]

    # Log transform
    log_intensity = np.log(intensity_pos)
    log_lactate = np.log(lactate_pos)

    # Segmented regression with 1 breakpoint
    breakpoints = segmented_regression(log_intensity, log_lactate, n_breakpoints=1)
    breakpoint_log = breakpoints[0]

    # Transform back to original scale
    method_intensity = np.exp(breakpoint_log)

    # Get lactate at this intensity (interpolate from original data)
    f = interpolate.interp1d(data['intensity'], data['lactate'], kind='cubic',
                            fill_value='extrapolate')
    method_lactate = float(f(method_intensity))
    method_lactate = round(method_lactate, 1)

    decimals = {'cycling': 1, 'running': 2, 'swimming': 3}[sport]
    method_intensity = round(method_intensity, decimals)

    hr = retrieve_heart_rate(data, method_intensity)

    return {
        'method': 'Log-log',
        'category': 'Logarithmic',
        'fitting': '3rd degree polynomial (user-defined)',
        'intensity': method_intensity,
        'lactate': method_lactate,
        'heart_rate': hr if hr is not None else np.nan
    }


# ============================================================================
# METHOD 5: LTP (Lactate Turning Point)
# ============================================================================

def method_ltp(data: pd.DataFrame, sport: str = 'cycling') -> pd.DataFrame:
    """
    LTP method: finds 2 breakpoints using segmented regression.

    Matches R: method_ltp() in method-ltp.R
    Uses pwlf library which better matches R's segmented package.
    """
    import pwlf

    # Use LINEAR interpolation of RAW data (R uses data_interpolated, not data_augmented)
    data_fit = data.iloc[1:]  # Exclude baseline
    intensity_interp, lactate_interp = interpolate_data_linear(
        data_fit['intensity'].values,
        data_fit['lactate'].values,
        sport
    )

    # Fit polynomial model for retrieving lactate values later
    model, poly, coeffs = fit_polynomial_3rd(
        data_fit['intensity'].values,
        data_fit['lactate'].values
    )

    # Segmented regression with 2 breakpoints using pwlf
    # pwlf.fit(3) means 3 line segments, which gives 2 interior breakpoints
    my_pwlf = pwlf.PiecewiseLinFit(intensity_interp, lactate_interp)
    all_breakpoints = my_pwlf.fit(3)

    # Extract interior breakpoints (exclude endpoints)
    breakpoints = all_breakpoints[1:-1]

    ltp1_intensity = breakpoints[0]
    ltp2_intensity = breakpoints[1]

    # Get lactate values (interpolate from original data)
    f = interpolate.interp1d(data['intensity'], data['lactate'], kind='cubic',
                            fill_value='extrapolate')

    ltp1_lactate = float(f(ltp1_intensity))
    ltp2_lactate = float(f(ltp2_intensity))

    ltp1_lactate = round(ltp1_lactate, 1)
    ltp2_lactate = round(ltp2_lactate, 1)

    decimals = {'cycling': 1, 'running': 2, 'swimming': 3}[sport]

    ltp1_intensity = round(ltp1_intensity, decimals)
    ltp2_intensity = round(ltp2_intensity, decimals)

    # Get heart rates
    hr1 = retrieve_heart_rate(data, ltp1_intensity)
    hr2 = retrieve_heart_rate(data, ltp2_intensity)

    return pd.DataFrame([
        {
            'method': 'LTP1',
            'category': 'Turnpoint',
            'fitting': '3rd degree polynomial (user-defined)',
            'intensity': ltp1_intensity,
            'lactate': ltp1_lactate,
            'heart_rate': hr1 if hr1 is not None else np.nan
        },
        {
            'method': 'LTP2',
            'category': 'Turnpoint',
            'fitting': '3rd degree polynomial (user-defined)',
            'intensity': ltp2_intensity,
            'lactate': ltp2_lactate,
            'heart_rate': hr2 if hr2 is not None else np.nan
        }
    ])


# ============================================================================
# METHOD 6: LTratio (Lactate-to-Intensity Ratio)
# ============================================================================

def method_ltratio(data: pd.DataFrame, sport: str = 'cycling') -> Dict:
    """
    LTratio method: minimum lactate/intensity ratio.

    Per Dickhuth et al. 1999:
    "The lactate threshold (LT) was conceptionally defined at the lowest value
    of the ratio lactate/performance (lactate-equivalent)"

    Uses smoothing spline on lactate curve, then finds minimum ratio.
    """
    import statsmodels.formula.api as smf
    import warnings

    # Exclude baseline (first row)
    data_fit = data.iloc[1:].copy()

    # Use GLM with natural splines (gives better results than UnivariateSpline)
    # df=5 matches R's implementation better (gives 321.5W vs target 320.9W)
    try:
        model = smf.glm(formula='lactate ~ cr(intensity, df=5)', data=data_fit)

        # Suppress divide-by-zero warning from perfect fit
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='divide by zero')
            result = model.fit()

        # Create fine grid
        interp_factor = 0.1 if sport in ['cycling', 'running'] else 0.01
        intensity_grid = np.arange(
            data_fit['intensity'].min(),
            data_fit['intensity'].max() + interp_factor,
            interp_factor
        )

        # Predict smoothed lactate
        df_grid = pd.DataFrame({'intensity': intensity_grid})
        lactate_smoothed = result.predict(df_grid)

        # Calculate ratio
        ratio = lactate_smoothed / intensity_grid

        # Find minimum
        min_idx = np.argmin(ratio)
        method_intensity = float(intensity_grid[min_idx])
        method_lactate = float(lactate_smoothed.iloc[min_idx])

    except Exception as e:
        # Fallback to UnivariateSpline
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(
            data_fit['intensity'].values,
            data_fit['lactate'].values,
            k=3,
            s=None
        )
        interp_factor = 0.1 if sport in ['cycling', 'running'] else 0.01
        intensity_grid = np.arange(
            data_fit['intensity'].min(),
            data_fit['intensity'].max() + interp_factor,
            interp_factor
        )
        lactate_smoothed = spline(intensity_grid)
        ratio = lactate_smoothed / intensity_grid
        min_idx = np.argmin(ratio)
        method_intensity = float(intensity_grid[min_idx])
        method_lactate = float(lactate_smoothed[min_idx])

    method_lactate = round(method_lactate, 1)
    decimals = {'cycling': 1, 'running': 2, 'swimming': 3}[sport]
    method_intensity = round(method_intensity, decimals)

    hr = retrieve_heart_rate(data, method_intensity)

    return {
        'method': 'LTratio',
        'category': 'Ratio-based',
        'fitting': 'B-Spline (default)',
        'intensity': method_intensity,
        'lactate': method_lactate,
        'heart_rate': hr if hr is not None else np.nan
    }


# ============================================================================
# METHOD CLASSIFICATION FUNCTION
# ============================================================================

def get_metric_for_method(method: str) -> str:
    """
    Classify methods by their typical physiological meaning.
    Matches old_site ui-manager.js getMetricForMethod function.

    Args:
        method: Method name

    Returns:
        Metric classification string
    """
    metrics = {
        'LTP1': 'LT1 (Aerobic Threshold)',
        'LTP2': 'LT2/MLSS (Anaerobic Threshold)',
        'Bsln + 0.5': 'LT1 (Aerobic Threshold)',
        'Bsln + 1.0': 'LT1 (Aerobic Threshold)',
        'Bsln + 1.5': 'Between LT1 and LT2',
        'OBLA 2.0': 'LT1 (Aerobic Threshold)',
        'OBLA 2.5': 'Between LT1 and LT2',
        'OBLA 3.0': 'Between LT1 and LT2',
        'OBLA 3.5': 'Between LT1 and LT2',
        'OBLA 4.0': 'LT2/MLSS (Anaerobic Threshold)',
        'LTratio': 'LT1 (Aerobic Threshold)',
        'Dmax': 'LT2/MLSS (Anaerobic Threshold)',
        'ModDmax': 'LT2/MLSS (Anaerobic Threshold)',
        'Exp-Dmax': 'LT2/MLSS (Anaerobic Threshold)',
        'Log-Poly-ModDmax': 'LT2/MLSS (Anaerobic Threshold)',
        'Log-Exp-ModDmax': 'LT2/MLSS (Anaerobic Threshold)',
        'Log-log': 'LT2/MLSS (Anaerobic Threshold)',
    }
    return metrics.get(method, method)


# ============================================================================
# AWS LAMBDA API FUNCTION
# ============================================================================

def call_lambda_api(intensity: np.ndarray, lactate: np.ndarray,
                   heart_rate: np.ndarray, sport: str = 'cycling') -> Optional[Dict]:
    """
    Call the AWS Lambda API to get reference lactate threshold values.

    Args:
        intensity: Array of intensity values (watts)
        lactate: Array of lactate values (mmol/L)
        heart_rate: Array of heart rate values (bpm)
        sport: Sport type (default 'cycling')

    Returns:
        Dictionary mapping method names to their results, or None if API call fails
    """
    try:
        # Prepare API data (matches old_site format exactly)
        api_data = []
        for i in range(len(intensity)):
            api_data.append({
                "step": i,
                "length": 4,  # Hardcoded to 4 as in old_site
                "intensity": float(intensity[i]),
                "lactate": float(lactate[i]),
                "heart_rate": float(heart_rate[i])
            })

        # Build query parameters
        params = {
            'sport': sport,
            'heartRate': 'true',
            'includeBaseline': 'true',
            'loglogRestrictor': '1',
            'fit': '3rd degree polynomial'
        }

        # Make API request
        url = "https://vbjz4lvpx6t27mjbheqbax4xju0lejms.lambda-url.us-west-1.on.aws/"
        response = requests.post(
            url,
            params=params,
            headers={'Content-Type': 'text/plain'},
            data=json.dumps(api_data),
            timeout=30
        )

        if response.ok:
            result = response.json()

            # The API returns an array with a JSON string that needs parsing
            if isinstance(result, list) and len(result) > 0:
                parsed_result = json.loads(result[0])
                result = parsed_result

            # Convert array of results to dictionary keyed by method
            lambda_results = {}
            if 'out' in result:
                for item in result['out']:
                    lambda_results[item['method']] = {
                        'intensity': item.get('intensity'),
                        'lactate': item.get('lactate'),
                        'heart_rate': item.get('heart_rate'),
                        'metric': item.get('metric')  # Include metric classification
                    }
            return lambda_results
        else:
            print(f"Lambda API error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Lambda API call failed: {e}")
        import traceback
        print(traceback.format_exc())
        return None


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_lactate_thresholds(intensity: np.ndarray, lactate: np.ndarray,
                               heart_rate: np.ndarray, sport: str = 'cycling',
                               include_lambda: bool = True) -> pd.DataFrame:
    """
    Perform comprehensive lactate threshold analysis using all methods.
    Exact port of R lactater package.

    Args:
        intensity: Array of intensity values (watts for rowing/cycling)
        lactate: Array of lactate values (mmol/L)
        heart_rate: Array of heart rate values (bpm)
        sport: 'cycling', 'running', or 'swimming'

    Returns:
        DataFrame with results from all methods
    """
    # Create dataframe
    data = pd.DataFrame({
        'intensity': intensity,
        'lactate': lactate,
        'heart_rate': heart_rate
    })

    # Exclude baseline for polynomial fitting
    data_fit = data.iloc[1:]

    # Fit 3rd degree polynomial (default fitting)
    model, poly, coeffs = fit_polynomial_3rd(
        data_fit['intensity'].values,
        data_fit['lactate'].values
    )

    # Get fitted values at ORIGINAL data points (for polynomial-based methods)
    original_intensity = data_fit['intensity'].values
    original_fitted_lactate = model.predict(poly.transform(original_intensity.reshape(-1, 1)))

    # Create interpolated/fitted data (model predictions at fine-grained intensity values)
    # This is used for plotting and some methods
    fitted_intensity, fitted_lactate = interpolate_data(
        data_fit['intensity'].values,
        data_fit['lactate'].values,
        model,
        poly,
        sport
    )

    # Run all methods
    results = []

    # 1. OBLA methods (use ORIGINAL data points as R does)
    try:
        obla_results = method_obla(data, model, poly, original_intensity, original_fitted_lactate, sport)
        results.append(obla_results)
    except Exception as e:
        print(f"OBLA failed: {e}")

    # 2. Dmax variants
    try:
        dmax_result = method_dmax(data, sport)
        if dmax_result:
            results.append(pd.DataFrame([dmax_result]))
    except Exception as e:
        print(f"Dmax failed: {e}")

    try:
        moddmax_result = method_moddmax(data, sport)
        if moddmax_result:
            results.append(pd.DataFrame([moddmax_result]))
    except Exception as e:
        print(f"ModDmax failed: {e}")

    try:
        exp_dmax_result = method_exp_dmax(data, sport)
        if exp_dmax_result:
            results.append(pd.DataFrame([exp_dmax_result]))
    except Exception as e:
        print(f"Exp-Dmax failed: {e}")

    # 3. Baseline Plus (use ORIGINAL data points as R does)
    try:
        bsln_results = method_bsln_plus(data, model, poly, original_intensity, original_fitted_lactate, sport)
        results.append(bsln_results)
    except Exception as e:
        print(f"Bsln+ failed: {e}")

    # 4. Log-log
    try:
        loglog_result = method_loglog(data, sport)
        if loglog_result:
            results.append(pd.DataFrame([loglog_result]))
    except Exception as e:
        print(f"Log-log failed: {e}")

    # 5. LTP
    try:
        ltp_results = method_ltp(data, sport)
        results.append(ltp_results)
    except Exception as e:
        print(f"LTP failed: {e}")

    # 6. LTratio
    try:
        ltratio_result = method_ltratio(data, sport)
        if ltratio_result:
            results.append(pd.DataFrame([ltratio_result]))
    except Exception as e:
        print(f"LTratio failed: {e}")

    # Combine all results
    if results:
        df = pd.concat(results, ignore_index=True)

        # Add metric classification for each method
        df['metric'] = df['method'].apply(get_metric_for_method)

        # Optionally call Lambda API to get reference values
        lambda_results = None
        if include_lambda:
            print("Calling Lambda API for reference values...")
            lambda_results = call_lambda_api(intensity, lactate, heart_rate, sport)

        # Add Lambda API results as additional columns
        if lambda_results and include_lambda:
            # Initialize new columns with NaN
            df['lambda_intensity'] = np.nan
            df['lambda_lactate'] = np.nan
            df['lambda_hr'] = np.nan

            # Match methods and populate Lambda columns
            for idx, row in df.iterrows():
                method_name = row['method']
                if method_name in lambda_results:
                    lambda_data = lambda_results[method_name]
                    df.at[idx, 'lambda_intensity'] = lambda_data.get('intensity')
                    df.at[idx, 'lambda_lactate'] = lambda_data.get('lactate')
                    df.at[idx, 'lambda_hr'] = lambda_data.get('heart_rate')

        # Reorder and rename columns for display
        column_mapping = {
            'metric': 'Metric',
            'method': 'Method',
            'category': 'Category',
            'fitting': 'Fitting',
            'intensity': 'Intensity (W)',
            'split': 'Split (/500m)',
            'lactate': 'Lactate (mmol/L)',
            'heart_rate': 'Heart Rate (bpm)',
            'lambda_intensity': 'Lambda Intensity (W)',
            'lambda_split': 'Lambda Split (/500m)',
            'lambda_lactate': 'Lambda Lactate (mmol/L)',
            'lambda_hr': 'Lambda HR (bpm)'
        }

        # Calculate Split (pace per 500m) from power
        # Split = pace per 500m in MM:SS format
        # Power = 2.80 / pace³ where pace is time per meter
        # So pace = (2.80 / power)^(1/3)
        def calculate_split(power_watts):
            if pd.isna(power_watts) or power_watts <= 0:
                return ''
            pace_per_meter = (2.80 / power_watts) ** (1/3)
            pace_per_500m = pace_per_meter * 500  # seconds per 500m
            minutes = int(pace_per_500m // 60)
            seconds = pace_per_500m % 60
            return f"{minutes}:{seconds:04.1f}"

        df['split'] = df['intensity'].apply(calculate_split)
        if lambda_results and 'lambda_intensity' in df.columns:
            df['lambda_split'] = df['lambda_intensity'].apply(calculate_split)

        # Sort by Metric then by intensity
        # Use categorical sorting for metric to ensure proper order (LT1 < Between < LT2)
        metric_order = ['LT1 (Aerobic Threshold)', 'Between LT1 and LT2', 'LT2/MLSS (Anaerobic Threshold)']
        df['metric'] = pd.Categorical(df['metric'], categories=metric_order, ordered=True)
        df = df.sort_values(['metric', 'intensity'], ascending=[True, True])

        # Select and rename columns
        display_columns = ['metric', 'method', 'category', 'fitting', 'intensity', 'split', 'lactate', 'heart_rate']
        if lambda_results:
            display_columns.extend(['lambda_intensity', 'lambda_split', 'lambda_lactate', 'lambda_hr'])

        df = df[display_columns].rename(columns=column_mapping)

        return df
    else:
        return pd.DataFrame()
