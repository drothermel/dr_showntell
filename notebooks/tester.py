from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console

from dr_showntell.console_components import InfoBlock, SectionRule, TitlePanel
from dr_showntell.fancy_table import FancyTable, HeaderCell
from dr_showntell.table_formatter import load_table_config

console = Console()

DATA_DIR = Path.home() / "drotherm/repos/dr_wandb/data"


def load_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs_metadata = pd.read_parquet(DATA_DIR / "runs_metadata.parquet")
    runs_history = pd.read_parquet(DATA_DIR / "runs_history.parquet")

    console.print(f"Data loaded: {len(runs_metadata):,} metadata rows, {len(runs_history):,} history rows")
    console.print(f"Available columns: {list(runs_metadata.columns)}")

    return runs_metadata, runs_history


def test_console_components() -> None:
    console.print(TitlePanel("DR ShowNTell Library Test Suite"))
    console.print(SectionRule("Data Loading Complete", "bold green"))
    console.print(InfoBlock("Key Metrics: Total Runs 401 | Status: LOADED | Components: TESTING", "bold green"))


def test_fancy_table_single_header(df: pd.DataFrame) -> None:
    console.print(SectionRule("FancyTable - Single Header", "bold yellow"))

    sample_data = df.head(6)

    table = FancyTable(
        title="Training Results - Single Header",
        show_header=True,
        header_style="bold white on blue"
    )

    table.add_column("Run ID", style="dim")
    table.add_column("Run Name", style="cyan")
    table.add_column("State", style="green")
    table.add_column("Project", style="magenta")

    for _, row in sample_data.iterrows():
        table.add_row(
            str(row['run_id'])[:12] + "..." if len(str(row['run_id'])) > 12 else str(row['run_id']),
            str(row['run_name'])[:20] + "..." if len(str(row['run_name'])) > 20 else str(row['run_name']),
            str(row['state']),
            str(row['project'])
        )

    console.print(table)


def test_fancy_table_multi_header(df: pd.DataFrame) -> None:
    console.print(SectionRule("FancyTable - Multi-Level Headers", "bold magenta"))

    sample_data = df.head(8)

    table = FancyTable(
        title="Training Analysis - Multi-Level Headers",
        show_header=True,
        header_style="bold white on purple"
    )

    # First add columns without headers
    table.add_column(style="dim")
    table.add_column(style="cyan")
    table.add_column(style="green")
    table.add_column(style="yellow")

    # Then add the multi-level headers
    table.add_header_row("Experiment", "Experiment", "Status", "Status")
    table.add_header_row("Run ID", "Name", "State", "Project")

    for _, row in sample_data.iterrows():
        table.add_row(
            str(row['run_id'])[:10] + "..." if len(str(row['run_id'])) > 10 else str(row['run_id']),
            str(row['run_name'])[:15] + "..." if len(str(row['run_name'])) > 15 else str(row['run_name']),
            str(row['state']),
            str(row['project'])
        )

    console.print(table)


def test_configuration_system() -> None:
    console.print(SectionRule("Configuration System Test", "bold red"))

    config1 = load_table_config("wandb_analysis")
    config2 = load_table_config("sweep_performance")

    console.print(f"Loaded wandb_analysis config with {len(config1)} formatters")
    console.print(f"Loaded sweep_performance config with {len(config2)} formatters")

    sample_configs = dict(list(config1.items())[:3])
    console.print("Sample config entries:")
    for key, value in sample_configs.items():
        console.print(f"  {key}: {value}")


def test_fancy_table_advanced_formatting() -> None:
    console.print(SectionRule("FancyTable - Advanced Formatting", "bold cyan"))

    synthetic_data = [
        ["exp_001", "3.5e-4", "7B", "2.543", "-0.127"],
        ["exp_002", "1.0e-4", "7B", "2.401", "-0.269"],
        ["exp_003", "5.0e-5", "13B", "2.203", "-0.467"],
        ["exp_004", "2.0e-4", "13B", "2.334", "-0.336"],
        ["exp_005", "1.5e-4", "70B", "1.987", "-0.683"],
        ["exp_006", "8.0e-5", "70B", "1.943", "-0.727"],
    ]

    table = FancyTable(
        title="Synthetic Training Results - Advanced Formatting",
        show_header=True,
        header_style="bold white on red"
    )

    # First add columns without headers
    table.add_column(style="dim")
    table.add_column(style="cyan")
    table.add_column(style="green")
    table.add_column(style="red")
    table.add_column(style="yellow")

    # Then add the multi-level headers
    table.add_header_row("Training", "Hyperparams", "Hyperparams", "Results", "Results")
    table.add_header_row("Experiment", "Learn Rate", "Model Size", "Final Loss", "Loss Delta")

    for row in synthetic_data:
        table.add_row(*row)

    console.print(table)


def create_integrated_demo(metadata_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    console.print(SectionRule("Integrated Visual Demo", "bold white on blue"))

    console.print(TitlePanel("DR ShowNTell - Full Integration Test"))

    console.print(InfoBlock(f"Integration Results: Metadata {len(metadata_df):,} rows | History {len(history_df):,} rows | Status: PASSING", "bold cyan"))

    console.print(SectionRule("Demo Complete - All Components Working", "bold green"))


def main() -> None:
    try:
        console.print("Starting DR ShowNTell comprehensive test...")
        console.print()

        metadata_df, history_df = load_test_data()

        test_console_components()

        test_fancy_table_single_header(metadata_df)

        test_fancy_table_multi_header(metadata_df)

        test_fancy_table_advanced_formatting()

        test_configuration_system()

        create_integrated_demo(metadata_df, history_df)

        console.print("All tests completed successfully!")

    except Exception as e:
        console.print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()