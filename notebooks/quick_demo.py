# %%
from __future__ import annotations

import pandas as pd
from rich.console import Console
from rich.text import Text

from dr_showntell.fancy_table import FancyTable

console = Console()

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pretrain_df = pd.read_parquet("/Users/daniellerothermel/drotherm/repos/datadec/data/datadecide/mean_eval_melted.parquet")
    runs_df = pd.read_parquet("/Users/daniellerothermel/drotherm/repos/dr_wandb/data/runs_metadata.parquet")
    history_df = pd.read_parquet("/Users/daniellerothermel/drotherm/repos/dr_wandb/data/runs_history.parquet")
    return pretrain_df, runs_df, history_df


def display_dataframe_preview(df: pd.DataFrame, title: str, n_rows: int = 5, max_cell_width: int = 40) -> None:
    sample_data = df.head(n_rows)

    table = FancyTable(
        title=title,
        show_header=True,
        header_style="bold blue"
    )

    for col in sample_data.columns:
        table.add_column(str(col))

    for row_idx, (_, row) in enumerate(sample_data.iterrows()):
        formatted_row = []
        row_style = "dim" if row_idx % 2 == 1 else ""

        for col in sample_data.columns:
            cell_value = str(row[col]) if pd.notna(row[col]) else "N/A"

            if len(cell_value) > max_cell_width:
                wrapped_value = cell_value[:max_cell_width] + "..."
                formatted_row.append(wrapped_value)
            else:
                formatted_row.append(cell_value)

        table.add_row(*formatted_row, style=row_style)

    console.print(table)


def main() -> None:
    pretrain_df, runs_df, history_df = load_data()

    console.print(f"Loaded datasets: Pretrain ({len(pretrain_df):,} rows), Runs ({len(runs_df):,} rows), History ({len(history_df):,} rows)")
    console.print()

    display_dataframe_preview(pretrain_df, "Pretrain Dataset Preview", n_rows=5, max_cell_width=50)
    console.print()

    display_dataframe_preview(runs_df, "Runs Metadata Preview", n_rows=5, max_cell_width=50)
    console.print()

    display_dataframe_preview(history_df, "History Dataset Preview", n_rows=5, max_cell_width=50)

# %%
main()

# %%
if __name__ == "__main__":
    main()