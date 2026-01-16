from __future__ import annotations

import contextlib
import io
import json
import math
import random
import hashlib
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from data_driven_predictor_2025_26 import DataDrivenSuperLigPredictor

REMOTE_DATASET_URL = (
    "https://raw.githubusercontent.com/akgularda/SuperLig_Prediction/main/tsl_dataset.csv"
)
REMOTE_FORECAST_URL = (
    "https://akgularda.github.io/SuperLig_Prediction/api/forecast.json"
)

DATA_DIR = Path("data_cache")
DATASET_PATH = DATA_DIR / "tsl_dataset.csv"
META_PATH = DATA_DIR / "dataset_meta.json"
AUTO_REFRESH_HOURS = 24


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _read_meta() -> dict:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_meta(meta: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> int:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "superlig-local-app"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    dest.write_bytes(data)
    return len(data)


def ensure_dataset(force: bool) -> tuple[Path, dict, str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta = _read_meta()
    dataset_exists = DATASET_PATH.exists()
    last_download = _parse_iso8601(meta.get("downloaded_at"))

    should_update = force
    if dataset_exists and not force and last_download:
        age = _utc_now() - last_download
        should_update = age >= timedelta(hours=AUTO_REFRESH_HOURS)

    if not dataset_exists:
        should_update = True

    status = "Using cached dataset."
    if should_update:
        try:
            size = _download(REMOTE_DATASET_URL, DATASET_PATH)
            meta = {
                "downloaded_at": _utc_now().isoformat().replace("+00:00", "Z"),
                "source_url": REMOTE_DATASET_URL,
                "size_bytes": size,
                "sha256": _sha256(DATASET_PATH),
            }
            _write_meta(meta)
            status = "Downloaded latest dataset."
        except (urllib.error.URLError, OSError) as exc:
            if dataset_exists:
                status = f"Update failed ({exc}). Using cached dataset."
            else:
                raise RuntimeError("Dataset download failed and no cache exists.") from exc

    if dataset_exists and "sha256" not in meta:
        meta["sha256"] = _sha256(DATASET_PATH)
        _write_meta(meta)
    return DATASET_PATH, meta, status


@st.cache_data(show_spinner=False)
def load_dataset(path: str, dataset_hash: str) -> pd.DataFrame:
    _ = dataset_hash
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_predictor(path: str, dataset_hash: str) -> tuple[DataDrivenSuperLigPredictor, str]:
    _ = dataset_hash
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        predictor = DataDrivenSuperLigPredictor(path)
    return predictor, buffer.getvalue()


@st.cache_data(show_spinner=False)
def fetch_remote_forecast(cache_bust: str | None) -> dict | None:
    _ = cache_bust
    request = urllib.request.Request(
        REMOTE_FORECAST_URL,
        headers={"User-Agent": "superlig-local-app"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload
    except (urllib.error.URLError, json.JSONDecodeError):
        return None


def build_metrics_table(predictor: DataDrivenSuperLigPredictor) -> pd.DataFrame:
    rows = []
    for team in predictor.team_names:
        metrics = predictor.historical_metrics[team]
        rows.append(
            {
                "team": team,
                "matches": metrics["total_matches"],
                "win_rate": round(metrics["win_rate"], 3),
                "avg_goals_scored": round(metrics["avg_goals_scored"], 2),
                "avg_goals_conceded": round(metrics["avg_goals_conceded"], 2),
                "home_advantage": round(metrics["home_advantage"], 3),
                "recent_form": round(metrics["recent_form"], 3),
                "big_match_performance": round(metrics["big_match_performance"], 3),
                "historical_strength": round(metrics["historical_strength"], 3),
            }
        )
    return pd.DataFrame(rows)


def csr_breakdown(predictor: DataDrivenSuperLigPredictor) -> pd.DataFrame:
    manager_boost = {
        10: 150,
        9: 100,
        8: 75,
        7: 50,
        6: 25,
        5: 10,
        4: 0,
        3: -10,
        2: -20,
        1: -30,
        0: -50,
    }
    financial_boost = {
        "A+": 100,
        "A": 75,
        "A-": 50,
        "B+": 25,
        "B": 0,
        "B-": -25,
        "C+": -50,
        "C": -75,
        "C-": -100,
    }
    europe_boost = {"High": 75, "Medium": 40, "Low": 10, "None": 0}
    academy_boost = {
        "A": 30,
        "A-": 20,
        "B+": 15,
        "B": 10,
        "B-": 5,
        "C+": 0,
        "C": -5,
        "C-": -10,
    }

    rows = []
    title_info = getattr(predictor, "team_title_info", {})
    for team in predictor.team_names:
        metrics = predictor.historical_metrics[team]
        data = predictor.teams_2025_26[team]

        base_csr = (
            metrics["win_rate"] * 1000
            + metrics["historical_strength"] * 500
            + metrics["recent_form"] * 300
            + metrics["big_match_performance"] * 200
            + (2.0 - metrics["avg_goals_conceded"]) * 100
            + metrics["avg_goals_scored"] * 50
        )
        manager = manager_boost.get(data["manager_experience"], 0)
        financial = financial_boost.get(data["financial_rating"], 0)
        market = min(200, data["market_value"] * 2)
        stadium = min(50, data["stadium_capacity"] / 1000)
        transfer = data["summer_transfers_net"] * 3
        europe = europe_boost.get(data["european_experience"], 0)
        academy = academy_boost.get(data["youth_academy"], 0)
        info = title_info.get(team, {"titles_last_window": 0, "years_since_title": 12})
        titles_boost = min(120, info["titles_last_window"] * 35)
        is_big = team in {"Galatasaray", "Fenerbahce", "Besiktas", "Trabzonspor"}
        drought_penalty = min(120, info["years_since_title"] * (10 if is_big else 5))

        total = (
            base_csr
            + manager
            + financial
            + market
            + stadium
            + transfer
            + europe
            + academy
            + titles_boost
            - drought_penalty
        )
        final_csr = max(1500, min(3000, total))

        rows.append(
            {
                "team": team,
                "base_csr": round(base_csr, 1),
                "manager_boost": manager,
                "financial_boost": financial,
                "market_boost": round(market, 1),
                "stadium_boost": round(stadium, 1),
                "transfer_boost": round(transfer, 1),
                "europe_boost": europe,
                "academy_boost": academy,
                "titles_boost": titles_boost,
                "drought_penalty": -drought_penalty,
                "final_csr": round(final_csr, 1),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("final_csr", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def match_probability_details(
    predictor: DataDrivenSuperLigPredictor, home_team: str, away_team: str
) -> dict:
    home_csr = predictor.teams_2025_26[home_team]["data_driven_csr"]
    away_csr = predictor.teams_2025_26[away_team]["data_driven_csr"]
    home_metrics = predictor.historical_metrics[home_team]
    away_metrics = predictor.historical_metrics[away_team]

    csr_diff = home_csr - away_csr
    base_home_prob = 1 / (1 + math.exp(-csr_diff / 400))
    adjusted_home_prob = base_home_prob + home_metrics["home_advantage"]
    draw_prob = max(0.15, min(0.35, 0.25 - abs(csr_diff) / 2000))
    adjusted_home_prob = max(0.15, min(0.75, adjusted_home_prob))
    away_prob = max(0.1, 1 - adjusted_home_prob - draw_prob)
    total = adjusted_home_prob + draw_prob + away_prob
    home_win = adjusted_home_prob / total
    draw = draw_prob / total
    away_win = away_prob / total

    return {
        "home_csr": home_csr,
        "away_csr": away_csr,
        "csr_diff": csr_diff,
        "base_home_prob": base_home_prob,
        "home_advantage": home_metrics["home_advantage"],
        "draw_prob": draw_prob,
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
    }


def reset_results(predictor: DataDrivenSuperLigPredictor) -> None:
    predictor.results = {
        team: {
            "points": [],
            "wins": [],
            "draws": [],
            "losses": [],
            "gf": [],
            "ga": [],
            "rank": [],
            "is_champion": 0,
            "in_europe": 0,
            "is_relegated": 0,
        }
        for team in predictor.team_names
    }


def build_simulation_table(predictor: DataDrivenSuperLigPredictor) -> pd.DataFrame:
    rows = []
    sims = max(1, int(predictor.simulations))
    for team in predictor.team_names:
        results = predictor.results[team]
        rows.append(
            {
                "team": team,
                "points": round(float(np.mean(results["points"])), 1),
                "wins": round(float(np.mean(results["wins"])), 1),
                "draws": round(float(np.mean(results["draws"])), 1),
                "losses": round(float(np.mean(results["losses"])), 1),
                "champion_pct": round(results["is_champion"] / sims * 100, 2),
                "europe_pct": round(results["in_europe"] / sims * 100, 2),
                "relegation_pct": round(results["is_relegated"] / sims * 100, 2),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["points", "champion_pct"], ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


st.set_page_config(page_title="Super Lig Prediction Lab", layout="wide")
st.title("Super Lig Prediction Lab")
st.caption("Local Python app with full calculation transparency and daily data refresh.")

with st.sidebar:
    st.header("Data update")
    force_update = st.button("Update dataset now")
    dataset_path, meta, status = ensure_dataset(force_update)
    st.write(status)
    if meta:
        st.write(f"Downloaded at: {meta.get('downloaded_at', '--')}")
        st.write(f"Size: {meta.get('size_bytes', 0)} bytes")
        st.write(f"SHA256: {meta.get('sha256', '--')}")
        st.write(f"Source: {meta.get('source_url', '--')}")

dataset_hash = meta.get("sha256") if meta else _sha256(dataset_path)
dataset_df = load_dataset(str(dataset_path), dataset_hash)
predictor, predictor_logs = load_predictor(str(dataset_path), dataset_hash)

st.subheader("Daily published results (auto updated)")
forecast_payload = fetch_remote_forecast(meta.get("downloaded_at"))
if forecast_payload:
    meta_info = forecast_payload.get("metadata", {})
    st.write(
        f"Generated at: {meta_info.get('generated_at', '--')} | "
        f"Simulations: {meta_info.get('simulations', '--')} | "
        f"Historical range: {meta_info.get('historical_range', '--')}"
    )
    daily_table = pd.DataFrame(forecast_payload.get("table", []))
    if not daily_table.empty:
        st.dataframe(daily_table.head(10), use_container_width=True)
else:
    st.warning("Daily API not reachable. Run the local simulation below.")

st.divider()
st.header("Calculation pipeline")

st.subheader("1) Data ingestion")
st.write(
    "The model loads the historical Super Lig dataset and builds per-team metrics "
    "from all matches in the file."
)
st.write(
    f"Rows: {len(dataset_df):,} | "
    f"Seasons: {dataset_df['Season'].min()} to {dataset_df['Season'].max()}"
)
st.dataframe(dataset_df.head(8), use_container_width=True)

st.subheader("2) Historical team metrics")
metrics_df = build_metrics_table(predictor)
st.dataframe(metrics_df, use_container_width=True)

st.subheader("3) CSR formula and components")
st.write("CSR blends history, current inputs, and context boosts.")
st.code(
    "base_csr = win_rate*1000 + historical_strength*500 + recent_form*300\n"
    "         + big_match_performance*200 + (2.0-avg_goals_conceded)*100\n"
    "         + avg_goals_scored*50\n"
    "final_csr = clamp(1500, 3000, base_csr + manager + financial + market\n"
    "         + stadium + transfer + europe + academy + titles - drought)",
    language="text",
)
csr_df = csr_breakdown(predictor)
st.dataframe(csr_df, use_container_width=True)

st.subheader("4) Match probability calculator")
teams = predictor.team_names
home_team = st.selectbox("Home team", teams, index=0)
away_team = st.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)
if home_team == away_team:
    st.warning("Select two different teams.")
else:
    details = match_probability_details(predictor, home_team, away_team)
    st.write(
        f"CSR diff: {details['csr_diff']} | "
        f"Home advantage: {details['home_advantage']:.3f} | "
        f"Draw base: {details['draw_prob']:.3f}"
    )
    probability_df = pd.DataFrame(
        [
            {"outcome": "Home win", "probability": details["home_win"]},
            {"outcome": "Draw", "probability": details["draw"]},
            {"outcome": "Away win", "probability": details["away_win"]},
        ]
    )
    st.bar_chart(probability_df.set_index("outcome"))

st.subheader("5) Season simulation details")
st.write(
    "The engine simulates a double round-robin season. With 18 teams, that is "
    "18 * 17 * 2 = 612 matches per season. Each match score is sampled with "
    "Poisson goal rates using the historical averages for each team."
)

st.subheader("6) Monte Carlo run (local)")
st.write(
    "Run local simulations to generate the probabilities shown in the tables. "
    "Higher simulation counts are slower but more stable."
)
simulations = st.number_input("Simulations", min_value=50, max_value=200000, value=1000, step=50)
seed = st.number_input("Random seed (optional)", min_value=0, max_value=999999, value=42, step=1)
run_button = st.button("Run local simulation")

if run_button:
    predictor.simulations = int(simulations)
    reset_results(predictor)
    random.seed(seed)
    np.random.seed(seed)
    with st.spinner("Running simulations..."):
        sim_buffer = io.StringIO()
        with contextlib.redirect_stdout(sim_buffer), contextlib.redirect_stderr(sim_buffer):
            predictor.run_data_driven_simulations()
        sim_logs = sim_buffer.getvalue()
    sim_table = build_simulation_table(predictor)
    st.success("Simulation complete.")
    st.dataframe(sim_table, use_container_width=True)
    st.bar_chart(sim_table.set_index("team")[["champion_pct"]].head(8))
    with st.expander("Simulation log"):
        st.text(sim_logs)

with st.expander("Model initialization log"):
    st.text(predictor_logs)
