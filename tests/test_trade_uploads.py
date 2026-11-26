from io import BytesIO, StringIO

import polars as pl
import pytest

from src.trades.uploads import read_csv_upload, read_forecast_upload


def test_read_forecast_upload_accepts_csv() -> None:
    content = b"player_id,week\nplayer-1,3\n"

    frame = read_forecast_upload("forecast.csv", content)

    assert frame.to_dict(as_series=False) == {
        "player_id": ["player-1"],
        "week": [3],
    }


def test_read_forecast_upload_accepts_parquet() -> None:
    source = pl.DataFrame({"player_id": ["player-1"], "week": [3]})
    buffer = BytesIO()
    source.write_parquet(buffer)

    frame = read_forecast_upload("forecast.parquet", buffer.getvalue())

    assert frame.equals(source)


def test_read_forecast_upload_rejects_other_extensions() -> None:
    with pytest.raises(ValueError, match=r"\.csv or \.parquet"):
        read_forecast_upload("forecast.xlsx", b"")


def test_read_csv_upload_decodes_utf8_and_removes_bom() -> None:
    stream = read_csv_upload(b"\xef\xbb\xbfteam,player_name\nA,Player\n")

    assert isinstance(stream, StringIO)
    assert stream.read() == "team,player_name\nA,Player\n"


def test_read_csv_upload_rejects_non_utf8() -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        read_csv_upload(b"\xff")
