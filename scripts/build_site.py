"""
Builds static site data from the data-driven predictor.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_driven_predictor_2025_26 import DataDrivenSuperLigPredictor


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _build_payload(predictor: DataDrivenSuperLigPredictor) -> dict:
    team_payload = {}
    for team, data in predictor.teams_2025_26.items():
        team_payload[team] = {
            "market_value_eur_m": float(data.get("market_value", 0.0)),
            "manager": data.get("manager", ""),
            "stadium_capacity": int(data.get("stadium_capacity", 0)),
            "summer_transfers_net_eur_m": float(data.get("summer_transfers_net", 0.0)),
            "key_signings": list(data.get("key_signings", [])),
            "financial_rating": data.get("financial_rating", ""),
            "youth_academy": data.get("youth_academy", ""),
            "european_experience": data.get("european_experience", ""),
        }

    table_rows = []
    sims = max(1, int(predictor.simulations))
    for team in predictor.team_names:
        results = predictor.results[team]
        avg_points = _mean(results["points"])
        avg_wins = _mean(results["wins"])
        avg_draws = _mean(results["draws"])
        avg_losses = _mean(results["losses"])
        avg_gf = _mean(results["gf"])
        avg_ga = _mean(results["ga"])
        avg_gd = avg_gf - avg_ga
        table_rows.append(
            {
                "team": team,
                "csr": int(predictor.teams_2025_26[team].get("data_driven_csr", 0)),
                "points": round(avg_points, 1),
                "wins": round(avg_wins, 1),
                "draws": round(avg_draws, 1),
                "losses": round(avg_losses, 1),
                "goals_for": round(avg_gf, 1),
                "goals_against": round(avg_ga, 1),
                "goal_difference": round(avg_gd, 1),
                "championship_probability": round(results["is_champion"] / sims * 100, 2),
                "europe_probability": round(results["in_europe"] / sims * 100, 2),
                "relegation_probability": round(results["is_relegated"] / sims * 100, 2),
            }
        )

    table_rows.sort(
        key=lambda row: (row["points"], row["goal_difference"], row["goals_for"]),
        reverse=True,
    )
    for rank, row in enumerate(table_rows, start=1):
        row["rank"] = rank

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "simulations": sims,
            "historical_matches": int(len(predictor.historical_data)),
            "historical_range": f"{predictor.historical_data['Season'].min()}-{predictor.historical_data['Season'].max()}",
            "teams_total": len(predictor.team_names),
            "source": "data_driven_predictor_2025_26.py",
        },
        "teams": team_payload,
        "predictions": {
            "table": table_rows,
        },
    }
    return payload


def _rank_contenders(table_rows: list[dict]) -> list[dict]:
    return sorted(
        table_rows,
        key=lambda row: (
            row["championship_probability"],
            row["points"],
            row["goal_difference"],
            row["goals_for"],
        ),
        reverse=True,
    )


def _build_api_payloads(payload: dict) -> tuple[dict, dict, dict]:
    table_rows = payload.get("predictions", {}).get("table", [])
    contenders = _rank_contenders(table_rows)
    champion = contenders[0] if contenders else {}

    champion_payload = {
        "metadata": payload.get("metadata", {}),
        "champion": champion,
        "top_contenders": contenders[:5],
    }
    forecast_payload = {
        "metadata": payload.get("metadata", {}),
        "table": table_rows,
    }
    teams_payload = {
        "metadata": payload.get("metadata", {}),
        "teams": payload.get("teams", {}),
    }
    return champion_payload, forecast_payload, teams_payload


def build_site_data(output_dir: Path, simulations: int, seed: int | None) -> Path:
    dataset_path = REPO_ROOT / "tsl_dataset.csv"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        predictor = DataDrivenSuperLigPredictor(str(dataset_path))
        predictor.simulations = simulations
        if seed is not None:
            np.random.seed(seed)
        predictor.run_data_driven_simulations()
    payload = _build_payload(predictor)

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.json"
    _write_json(latest_path, payload)

    api_dir = output_dir.parent / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    champion_payload, forecast_payload, teams_payload = _build_api_payloads(payload)
    _write_json(api_dir / "champion.json", champion_payload)
    _write_json(api_dir / "forecast.json", forecast_payload)
    _write_json(api_dir / "teams.json", teams_payload)
    return latest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static site data payload.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "site" / "data"),
        help="Directory for latest.json output",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=int(os.getenv("SIMULATIONS", "1000")),
        help="Number of Monte Carlo simulations",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    latest_path = build_site_data(output_dir, args.simulations, args.seed)
    print(f"Site data written to {latest_path}")


if __name__ == "__main__":
    main()
