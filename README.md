# Futures Basis Analysis Pipeline

This repository contains a research pipeline to analyze futures basis behavior using S&P 500, SOFR, and dividend futures data. The project preprocesses minute and daily resolution datasets, applies term structure modeling and interpolations, and produces spread visualizations and theoretical pricing based on D.E. Shaw-style financing and dividend models.

---

## 📦 Project Structure

```
.
├── main.py                     # Main entry point for data processing and visualization
├── preprocessing.py            # Preprocessing pipeline for various financial data sources
├── expiration_calendar.py      # Futures expiration calendar for ES, NQ, VIX, etc.
├── dividend_rate.py            # Dividend yield model interface
├── interest_rate.py            # Interest rate model including flat and term structures
├── interpolation.py            # Linear and spline interpolators
├── settings.py                 # Project-level config loader using python-decouple
├── utils.py                    # Utility functions including theoretical price and plotting
└── data_manual/                # Directory with all required raw data files (local, not tracked)
```

---

## 🔧 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/Kevin-finance/futures_basis.git
cd futures_basis
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your `.env` file

Create a `.env` file in the project root with the following variables (or modify the defaults in `settings.py`):

```env
MANUAL_DATA_DIR=data_manual
OUTPUT_DIR=_output
DATA_DIR=_data
START_DATE=2024-01-01
END_DATE=2024-12-31
PIPELINE_DEV_MODE=True
PIPELINE_THEME=pipeline
```

---

## 🚀 How to Run

Run the main pipeline script:

```bash
python main.py
```

It will:

- Load and preprocess interest rate, dividend, spot, and futures data
- Compute theoretical futures prices and spreads
- Generate several interactive Plotly visualizations
- Save results to:
  - `merged3.csv`
  - `plot_spreads_linear.html`
  - `plot_percentage_changes_linear.html`
  - `plot_twinx_linear.html`

---

## 📈 Dependencies

- [Polars](https://pola.rs/)
- [Plotly](https://plotly.com/)
- [exchange-calendars](https://pypi.org/project/exchange-calendars/)
- [python-decouple](https://github.com/HBNetwork/python-decouple)

---

## 📌 Notes

- Be sure the files in `data_manual/` match the expected filenames referenced in `main.py`.
- The script assumes minute-level futures and spot data, and daily-level interest/dividend data.
- Theoretical pricing formula is based on continuous compounding of risk-free rates and discounting of dividends.

---

## 📤 Outputs

All charts are exported as `.html` and can be opened directly in a browser. The final merged dataset is saved as `merged3.csv`.

---
