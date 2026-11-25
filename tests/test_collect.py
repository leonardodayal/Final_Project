import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.collect import collect_data


class FakeFrame:
    def __init__(self, columns: list[str], height: int) -> None:
        self.columns = columns
        self.height = height

    def is_empty(self) -> bool:
        return self.height == 0

    def write_parquet(self, path: Path) -> None:
        path.write_text("test data", encoding="utf-8")


def test_collect_data_saves_stats_ids_and_rankings(tmp_path, monkeypatch) -> None:
    stats = FakeFrame(
        ["season", "week", "player_id", "fantasy_points_ppr"],
        12,
    )
    player_ids = FakeFrame(["player_id", "fantasypros_id"], 8)
    rankings = FakeFrame(["season", "week", "player_name", "rank"], 20)
    fake_nfl = SimpleNamespace(
        __version__="0.1.5",
        load_player_stats=lambda **kwargs: stats,
        load_ff_playerids=lambda: player_ids,
        load_ff_rankings=lambda **kwargs: rankings,
    )
    monkeypatch.setitem(sys.modules, "nflreadpy", fake_nfl)

    row_counts = collect_data([2024, 2025], tmp_path)

    assert row_counts == {
        "weekly_player_stats": 12,
        "fantasy_player_id_crosswalk": 8,
        "historical_fantasy_rankings": 20,
    }
    assert (tmp_path / "weekly_player_stats_2024-2025.parquet").exists()
    assert (tmp_path / "fantasy_player_id_crosswalk.parquet").exists()
    assert (tmp_path / "historical_fantasy_rankings.parquet").exists()
    assert (tmp_path / "collection_metadata.json").exists()


def test_collect_data_rejects_stats_without_ppr_scores(tmp_path, monkeypatch) -> None:
    stats = FakeFrame(["season", "week", "player_id"], 12)
    fake_nfl = SimpleNamespace(
        __version__="0.1.5",
        load_player_stats=lambda **kwargs: stats,
    )
    monkeypatch.setitem(sys.modules, "nflreadpy", fake_nfl)

    with pytest.raises(ValueError, match="fantasy_points_ppr"):
        collect_data([2025], tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_collect_data_warns_and_saves_when_rankings_are_empty(
    tmp_path,
    monkeypatch,
) -> None:
    stats = FakeFrame(
        ["season", "week", "player_id", "fantasy_points_ppr"],
        12,
    )
    player_ids = FakeFrame(["player_id", "fantasypros_id"], 8)
    rankings = FakeFrame(["season", "week", "player_name", "rank"], 0)
    fake_nfl = SimpleNamespace(
        __version__="0.1.5",
        load_player_stats=lambda **kwargs: stats,
        load_ff_playerids=lambda: player_ids,
        load_ff_rankings=lambda **kwargs: rankings,
    )
    monkeypatch.setitem(sys.modules, "nflreadpy", fake_nfl)

    with pytest.warns(RuntimeWarning, match="No historical fantasy rankings"):
        row_counts = collect_data([2025], tmp_path)

    assert row_counts["historical_fantasy_rankings"] == 0
    assert (tmp_path / "weekly_player_stats_2025.parquet").exists()
    assert not (tmp_path / "historical_fantasy_rankings.parquet").exists()
