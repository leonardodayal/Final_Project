"""Download public NFL player statistics and fantasy rankings."""

import argparse
import json
import warnings
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_STAT_COLUMNS = {"season", "week", "player_id", "fantasy_points_ppr"}


def collect_data(seasons: list[int], output_dir: Path) -> dict[str, int]:
    if not seasons:
        raise ValueError("Provide at least one NFL season.")

    import nflreadpy as nfl

    player_stats = nfl.load_player_stats(seasons=seasons, summary_level="week")
    missing_columns = REQUIRED_STAT_COLUMNS - set(player_stats.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Weekly player stats are missing required columns: {missing}")
    if player_stats.is_empty():
        raise RuntimeError("The data source returned no weekly player statistics.")

    player_ids = nfl.load_ff_playerids()
    if player_ids.is_empty():
        raise RuntimeError("The data source returned no fantasy player ID mappings.")

    rankings = nfl.load_ff_rankings(type="all")

    output_dir.mkdir(parents=True, exist_ok=True)
    season_label = "-".join(str(season) for season in seasons)
    player_stats.write_parquet(output_dir / f"weekly_player_stats_{season_label}.parquet")
    player_ids.write_parquet(output_dir / "fantasy_player_id_crosswalk.parquet")

    if rankings.is_empty():
        warnings.warn(
            "No historical fantasy rankings were returned; stats and ID mappings "
            "were saved without rankings.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        rankings.write_parquet(output_dir / "historical_fantasy_rankings.parquet")

    row_counts = {
        "weekly_player_stats": player_stats.height,
        "fantasy_player_id_crosswalk": player_ids.height,
        "historical_fantasy_rankings": rankings.height,
    }
    metadata = {
        "source": "nflverse and FantasyPros via nflreadpy",
        "nflreadpy_version": nfl.__version__,
        "seasons": seasons,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_counts": row_counts,
    }
    (output_dir / "collection_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return row_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="NFL season years to download, for example: --seasons 2021 2022 2023",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for generated Parquet files (default: data/raw)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    row_counts = collect_data(args.seasons, args.output_dir)
    for dataset, row_count in row_counts.items():
        print(f"{dataset}: {row_count:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
