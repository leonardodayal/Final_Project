"""Streamlit interface for weekly fantasy projections and trade analysis."""

import hashlib
import json
from io import BytesIO
from pathlib import Path

import polars as pl
import streamlit as st

from src.trades.evaluate import evaluate_trade
from src.trades.forecast import FORECAST_SEASON, generate_current_forecast
from src.trades.manual import build_manual_csv_inputs
from src.trades.roster import BENCH_WEIGHT
from src.trades.uploads import read_csv_upload, read_forecast_upload
from src.trades.values import PROJECTION_COLUMN, aggregate_rest_of_season_values


def _signature(
    predictions: pl.DataFrame,
    bench_weight: float,
    roster_mode: str,
    roster_state: dict,
    roster_content: bytes,
    trade_content: bytes,
) -> str:
    buffer = BytesIO()
    predictions.write_parquet(buffer)
    digest = hashlib.sha256()
    for content in (
        buffer.getvalue(),
        json.dumps(
            [bench_weight, roster_mode, roster_state],
            sort_keys=True,
        ).encode("utf-8"),
        roster_content,
        trade_content,
    ):
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _player_labels(player_values: pl.DataFrame) -> dict[str, str]:
    rows = player_values.select("player_id", "player_name", "position").to_dicts()
    counts: dict[str, int] = {}
    for row in rows:
        base = f"{row['player_name']} · {row['position']}"
        counts[base] = counts.get(base, 0) + 1
    return {
        str(row["player_id"]): (
            f"{row['player_name']} · {row['position']}"
            if counts[f"{row['player_name']} · {row['position']}"] == 1
            else f"{row['player_name']} · {row['position']} · {row['player_id']}"
        )
        for row in rows
    }


def _show_report(report: dict, predictions: pl.DataFrame) -> None:
    verdicts = {
        "mutually_beneficial": "Both teams improve",
        "one_sided": "One team improves",
        "neither_team_improves": "Neither team improves",
        "neutral": "Trade is neutral",
    }
    if report["mutually_beneficial"]:
        st.success(verdicts[report["trade_classification"]])
    else:
        st.warning(verdicts[report["trade_classification"]])

    st.download_button(
        "Download trade report (JSON)",
        data=json.dumps(report, indent=2) + "\n",
        file_name=f"trade_evaluation_{report['season']}.json",
        mime="application/json",
    )
    teams = report["teams"]
    columns = st.columns(len(teams))
    for column, team in zip(columns, teams):
        column.metric(
            f"{team['team']} ROS lineup value",
            f"{team['after']['strength']:.1f}",
            f"{team['delta']:+.1f}",
            delta_color="normal",
        )
    st.caption(report["uncertainty_note"])

    impact_tab, projections_tab = st.tabs(
        ["Trade impact", "Weekly projections"]
    )
    with impact_tab:
        strength_chart = pl.DataFrame(
            [
                {
                    "Team": team["team"],
                    "Phase": phase,
                    "ROS lineup value": team[phase.lower()]["strength"],
                }
                for team in teams
                for phase in ("Before", "After")
            ]
        ).to_pandas()
        st.bar_chart(
            strength_chart,
            x="Team",
            y="ROS lineup value",
            color="Phase",
            stack=False,
        )
        for team in teams:
            with st.expander(f"{team['team']} lineup", expanded=True):
                before, after = st.columns(2)
                before.subheader("Before")
                before.dataframe(
                    team["before"]["starters"] + team["before"]["bench"],
                    hide_index=True,
                    use_container_width=True,
                )
                after.subheader("After")
                after.dataframe(
                    team["after"]["starters"] + team["after"]["bench"],
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    f"Bench value is weighted at {report['bench_weight']:.0%}."
                )
    with projections_tab:
        roster_names = {
            entry["player_name"]
            for team in teams
            for entry in team["before"]["starters"] + team["before"]["bench"]
        }
        weekly = (
            predictions.filter(pl.col("player_name").is_in(sorted(roster_names)))
            .select(
                "player_name",
                "position",
                "week",
                PROJECTION_COLUMN,
            )
            .sort(["week", "position", "player_name"])
            .rename(
                {
                    "player_name": "Player",
                    "position": "Position",
                    "week": "Week",
                    PROJECTION_COLUMN: "Projected PPR points",
                }
            )
        )
        st.dataframe(
            weekly.to_pandas(),
            hide_index=True,
            use_container_width=True,
        )


st.set_page_config(
    page_title="Gridiron Trade Lab",
    page_icon="🏈",
    layout="wide",
)

st.title("Gridiron Trade Lab")
st.markdown(
    "Build a weekly outlook, compare two rosters, and see how a trade changes "
    "each lineup."
)

with st.sidebar:
    st.header("League settings")
    bench_weight = st.slider(
        "Bench value weight",
        min_value=0.0,
        max_value=1.0,
        value=BENCH_WEIGHT,
        step=0.05,
        help="Lineup starters use full value; bench players use this fraction.",
    )
    st.download_button(
        "Roster CSV template",
        data=Path("data/roster_template.csv").read_bytes(),
        file_name="roster_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Trade CSV template",
        data=Path("data/trade_template.csv").read_bytes(),
        file_name="trade_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.subheader("1 · Weekly projections")
projection_source = st.radio(
    "Choose a projection source",
    ["Generate 2026 projections", "Upload existing projections"],
    horizontal=True,
)
predictions = None

if projection_source == "Generate 2026 projections":
    st.caption(
        "Uses the trained model and public nflverse data. The first run may "
        "download historical and current-season inputs."
    )
    refresh_public_data = st.checkbox(
        "Refresh public data before generating",
        value=False,
        help="Fetch the latest available public inputs instead of using cached data.",
    )
    if st.button("Generate weekly projections", type="primary"):
        with st.spinner("Preparing data and generating weekly projections…"):
            try:
                st.session_state["generated_predictions"] = (
                    generate_current_forecast(
                        refresh_public_data=refresh_public_data
                    )
                )
            except (
                FileNotFoundError,
                KeyError,
                OSError,
                RuntimeError,
                ValueError,
                pl.exceptions.PolarsError,
            ) as error:
                st.error(f"Could not generate projections: {error}")
    predictions = st.session_state.get("generated_predictions")
    if predictions is not None:
        weeks = sorted(int(week) for week in predictions["week"].unique())
        st.success(
            f"Loaded {FORECAST_SEASON} projections for weeks "
            f"{weeks[0]}–{weeks[-1]}."
        )
else:
    forecast_file = st.file_uploader(
        "Upload weekly forecast (.csv or .parquet)",
        type=["csv", "parquet"],
        key="forecast_upload",
    )
    if forecast_file is not None:
        try:
            predictions = read_forecast_upload(
                forecast_file.name,
                forecast_file.getvalue(),
            )
        except (ValueError, pl.exceptions.PolarsError) as error:
            st.error(f"Could not read forecast: {error}")

if predictions is not None:
    try:
        player_values = aggregate_rest_of_season_values(predictions)
    except (ValueError, pl.exceptions.PolarsError) as error:
        st.error(f"Forecast is not usable: {error}")
        predictions = None
    else:
        st.caption(
            f"{player_values.height:,} players · "
            f"{predictions['week'].n_unique()} forecast weeks · "
            f"{int(player_values['season'][0])} season"
        )

if predictions is not None:
    st.subheader("2 · Rosters and trade")
    roster_mode = st.radio(
        "How would you like to enter players?",
        ["Choose players manually", "Upload roster and trade CSVs"],
        horizontal=True,
    )
    roster_state: dict = {}
    roster_content = b""
    trade_content = b""
    roster_file = None
    trade_file = None

    if roster_mode == "Choose players manually":
        labels = _player_labels(player_values)
        player_ids = sorted(labels)
        team_a, team_b = st.columns(2)
        with team_a:
            team_a_name = st.text_input("First team", "Team A")
            team_a_players = st.multiselect(
                "First team's roster",
                player_ids,
                format_func=labels.__getitem__,
                key="team_a_players",
            )
            team_a_sends = st.multiselect(
                "Players this team sends",
                team_a_players,
                format_func=labels.__getitem__,
                key="team_a_sends",
            )
        with team_b:
            team_b_name = st.text_input("Second team", "Team B")
            team_b_players = st.multiselect(
                "Second team's roster",
                [player_id for player_id in player_ids if player_id not in team_a_players],
                format_func=labels.__getitem__,
                key="team_b_players",
            )
            team_b_sends = st.multiselect(
                "Players this team sends",
                team_b_players,
                format_func=labels.__getitem__,
                key="team_b_sends",
            )
        roster_state = {
            "team_a_name": team_a_name,
            "team_a_players": team_a_players,
            "team_a_sends": team_a_sends,
            "team_b_name": team_b_name,
            "team_b_players": team_b_players,
            "team_b_sends": team_b_sends,
        }
    else:
        roster_column, trade_column = st.columns(2)
        with roster_column:
            roster_file = st.file_uploader(
                "Upload roster CSV",
                type=["csv"],
                key="roster_upload",
            )
        with trade_column:
            trade_file = st.file_uploader(
                "Upload trade CSV",
                type=["csv"],
                key="trade_upload",
            )
        roster_content = roster_file.getvalue() if roster_file else b""
        trade_content = trade_file.getvalue() if trade_file else b""

    signature = _signature(
        predictions,
        bench_weight,
        roster_mode,
        roster_state,
        roster_content,
        trade_content,
    )
    if st.button("Analyze trade", type="primary"):
        try:
            if roster_mode == "Choose players manually":
                roster_input, trade_input = build_manual_csv_inputs(
                    player_values,
                    team_a_name,
                    team_a_players,
                    team_b_name,
                    team_b_players,
                    team_a_sends,
                    team_b_sends,
                )
            else:
                if roster_file is None or trade_file is None:
                    raise ValueError("Upload both the roster and trade CSV files.")
                roster_input = read_csv_upload(roster_content)
                trade_input = read_csv_upload(trade_content)
            report = evaluate_trade(
                predictions,
                roster_input,
                trade_input,
                bench_weight,
            )
        except (ValueError, pl.exceptions.PolarsError) as error:
            st.error(f"Could not analyze trade: {error}")
        else:
            st.session_state["trade_report"] = report
            st.session_state["trade_report_signature"] = signature

    if st.session_state.get("trade_report_signature") == signature:
        _show_report(st.session_state["trade_report"], predictions)
