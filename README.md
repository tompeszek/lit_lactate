# Rowing Lactate Test Analyzer

A comprehensive web application for analyzing rowing lactate test results with 15 threshold detection methods, training zone calculations, and automatic power/split conversions.

## Features

### Lactate Test Analysis
- **15 Threshold Detection Methods**: OBLA (2.0-4.0), Dmax variants, LTP1/LTP2, Log-log, Baseline+, LTratio
- **Automatic Classification**: Methods categorized as LT1 (Aerobic), Between, or LT2/MLSS (Anaerobic)
- **Color-Coded Results**: Visual distinction between threshold types
- **Lambda API Integration**: Comparison with reference implementation

### Training Zones
- **7 Training Zones**: Z1-Z7 calculated from lactate thresholds and erg test results
- **Critical Power**: Computed from multiple erg test durations
- **Zone Metrics**: Power, pace, heart rate, and lactate ranges for each zone

### Data Input
- **Editable Tables**: Add/remove rows, edit inline
- **Power ↔ Split Conversion**: Automatic bidirectional conversion using Concept2 formula
- **Erg Test Results**: Input for zone calculation (1k, 2k, 5k, 6k, 60' tests)

## Quick Start

### Installation

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo>
cd lactate-analyzer
uv sync

# Run the app
uv run streamlit run app.py
```

App runs at `http://localhost:8501`

## Usage

### 1. Test Data Input
- Enter test steps with power (W) or split (m:ss.s format)
- Add heart rate, lactate, and RPE for each step
- Baseline (0W) step optional but recommended

### 2. Erg Test Results
- Input test results for zone calculation
- Supported: 1000m, 2000m, 5000m, 6000m, 60-minute tests
- Format times as `m:ss.s` (e.g., `6:24.3`)

### 3. Analysis Results
- **Lactate Threshold Analysis**: 15 methods with power, split, lactate, HR
- **Training Zones**: Power, pace, HR, and lactate ranges for Z1-Z7
- **Comparison**: Python vs Lambda API results (when available)

## Lactate Threshold Methods

### LT1 (Aerobic Threshold)
- **OBLA 2.0**: 2.0 mmol/L fixed concentration
- **Log-log**: Logarithmic transformation method
- **LTP1**: First lactate turnpoint (piecewise linear)
- **LTratio**: Lactate/intensity ratio method
- **Bsln + 0.5**: Baseline + 0.5 mmol/L
- **Bsln + 1.0**: Baseline + 1.0 mmol/L

### Between LT1 and LT2
- **OBLA 2.5**: 2.5 mmol/L
- **OBLA 3.0**: 3.0 mmol/L
- **OBLA 3.5**: 3.5 mmol/L
- **Bsln + 1.5**: Baseline + 1.5 mmol/L

### LT2/MLSS (Anaerobic Threshold)
- **OBLA 4.0**: 4.0 mmol/L (classic threshold)
- **Dmax**: Maximum distance from baseline-peak line
- **ModDmax**: Modified Dmax with exponential fit
- **Exp-Dmax**: Exponential Dmax variant
- **LTP2**: Second lactate turnpoint

## Power/Split Conversion

Uses Concept2 ergometer formula:

```
Power (W) = 2.80 / pace³
Pace (/500m) = ³√(2.80 / power) × 500
```

Where pace is in seconds per meter.

## Training Zones

Calculated from lactate thresholds and erg test data:

- **Z1**: Recovery (55-85% LT1 HR)
- **Z2**: Aerobic base (85-100% LT1)
- **Z3**: Tempo (LT1 to min(LT2, CP))
- **Z4**: Threshold (min(LT2, CP) to max(LT2, CP))
- **Z5**: VO2max (max(LT2, CP) to 92.5% max HR)
- **Z6**: Anaerobic (92.5% max HR to 1k power)
- **Z7**: ATP-PC (1k power to max power)

## Project Structure

```
lactate-analyzer/
├── app.py                 # Main Streamlit application
├── lactate_analysis.py    # Threshold detection methods
├── training_zones.py      # Zone calculation logic
├── pyproject.toml         # Dependencies
├── uv.lock                # Locked versions
├── Dockerfile             # Container configuration
└── README.md              # This file
```

## Deployment

### Railway

```bash
# Railway will auto-detect Dockerfile
railway up
```

### Docker

```bash
docker build -t lactate-analyzer .
docker run -p 8501:8501 lactate-analyzer
```

## Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **NumPy/SciPy**: Scientific computing
- **pandas**: Data manipulation
- **pwlf**: Piecewise linear fitting (LTP methods)
- **Plotly**: Interactive visualizations

### Data Format
- Power: Watts (W)
- Split: m:ss.s format (e.g., 1:45.2)
- Lactate: mmol/L
- Heart Rate: bpm

## Troubleshooting

### No results showing
- Minimum 3 exercise steps required (intensity > 0)
- Check for valid power/lactate/HR values
- Baseline step should be 0W or excluded

### LTP methods failing
- Requires smooth lactate progression
- Add more data points if curve is too irregular
- Check for data entry errors

### Zone calculation unavailable
- Requires both lactate analysis results AND erg test data
- Ensure erg times formatted correctly (m:ss.s)

## Credits

- Lactate methods based on [lactater R package](https://github.com/fmmattioni/lactater) by Felipe Mattioni Maturana
- Python implementation with improvements and optimizations
- Built with [Streamlit](https://streamlit.io/)
