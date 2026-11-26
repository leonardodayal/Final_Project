"""Read uploaded forecast and CSV files without writing them to disk."""

from io import BytesIO, StringIO

import polars as pl


def read_forecast_upload(filename: str, content: bytes) -> pl.DataFrame:
    """Read an uploaded weekly forecast in CSV or Parquet format."""
    suffix = filename.rpartition(".")[2].casefold()
    stream = BytesIO(content)
    if suffix == "csv":
        return pl.read_csv(stream)
    if suffix == "parquet":
        return pl.read_parquet(stream)
    raise ValueError("Forecast uploads must be a .csv or .parquet file.")


def read_csv_upload(content: bytes) -> StringIO:
    """Decode an uploaded UTF-8 CSV into an in-memory text stream."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV uploads must use UTF-8 encoding.") from error
    return StringIO(text)
