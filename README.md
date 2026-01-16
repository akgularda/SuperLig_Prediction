# Turkish Süper Lig 2025-26 Season Prediction

## 📁 Contents

### 📄 Files
- **`updated_2025_26_season_predictor.py`** - Enhanced CSR prediction system
- **`data_driven_predictor_2025_26.py`** - Historical data-driven predictor (RECOMMENDED)
- **`interactive_dashboard.py`** - Web dashboard for visualizing predictions
- **`launch_dashboard.py`** - Dashboard launcher script
- **`Start_Dashboard.bat`** - Windows batch file for easy dashboard startup
- **`dashboard_requirements.txt`** - Dashboard dependencies
- **`superlig_prediction_paper_corrected.pdf`** - Research paper with methodology
- **`tsl_dataset.csv`** - Historical match data (1958-2020, 18,079 matches)
- **`README.md`** - This file

## 🚀 How to Run

**Option 1 - Historical Data Predictor (RECOMMENDED):**
```bash
python data_driven_predictor_2025_26.py
```

**Option 2 - Enhanced CSR System:**
```bash
python updated_2025_26_season_predictor.py
```

**Option 3 - Interactive Web Dashboard:**
```bash
# Install dashboard dependencies first
pip install -r dashboard_requirements.txt

# Run dashboard (choose one method)
python launch_dashboard.py
# OR double-click Start_Dashboard.bat on Windows
# OR run directly: python interactive_dashboard.py
```

The dashboard will open in your web browser at `http://localhost:8050`

## 📊 What it Predicts

- **Championship probabilities** for all 18 teams
- **European qualification chances** (Top 5 spots)
- **Relegation risks** (Bottom 3 teams)
- **Season standings** with detailed statistics
- **Interactive visualizations** via web dashboard
- **Real-time match simulations** and league tables

## 🏆 Key Features

- ✅ **Correct 18 teams** for 2025-26 season
- ✅ **Enhanced CSR system** with manager effects
- ✅ **Transfer market impact** analysis
- ✅ **Stadium capacity** considerations
- ✅ **1000 Monte Carlo simulations**

## 📈 Expected Results

**Historical Data Predictor (More Realistic):**
- **Fenerbahçe**: ~50-55% championship chance
- **Galatasaray**: ~35-40% championship chance  
- **Beşiktaş**: ~8-12% championship chance
- **Trabzonspor**: ~0.5-1% championship chance
- **Başakşehir**: ~0-0.5% championship chance

**Enhanced CSR System:**
- **Galatasaray**: ~45-55% championship chance
- **Fenerbahçe**: ~35-45% championship chance  
- **Beşiktaş**: ~8-15% championship chance
- **Trabzonspor**: ~1-3% championship chance
- **Başakşehir**: ~0-2% championship chance

---
*Created: August 2025 | Turkish Football Analytics*

## Automated Website (GitHub Pages)

This repo includes a static commercial site in `site/` that reads the latest
prediction payload from `site/data/latest.json`.

### Build the site payload locally
```bash
pip install -r requirements.txt
python scripts/build_site.py --simulations 250 --output-dir site/data
```

### Preview the website locally
```bash
cd site
python -m http.server 8000
```

Then open `http://localhost:8000`.

### GitHub Actions
- `CI` verifies the data build on every push and PR.
- `Deploy Site` rebuilds and publishes the site to GitHub Pages on each push,
  on manual runs, and on a daily schedule.

### API endpoints (GitHub Pages)
The scheduled workflow generates JSON endpoints you can treat as an API:
- `/api/champion.json` - champion prediction + top contenders
- `/api/forecast.json` - full table with probabilities
- `/api/teams.json` - team metadata and key signings
- `/data/latest.json` - full payload (metadata + teams + predictions)

## Local Python App (Detailed)

The local app explains how probabilities are calculated, step by step, and
keeps the dataset refreshed automatically.

### Install
```bash
pip install -r app_requirements.txt
```

### Run
```bash
python -m streamlit run superlig_local_app.py
```

Or on Windows: double-click `Start_Local_App.bat`.

### Data updates
The app downloads the latest dataset from GitHub and shows the daily published
forecast from GitHub Pages.
