#!/usr/bin/env python3

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import click
import pandas as pd
from rich.console import Console

from dr_plotter import FigureManager
from dr_plotter.configs import PlotConfig
from dr_showntell.console_components import dataframe_to_fancy_tables, SectionRule
import matplotlib.pyplot as plt

console = Console()

MAX_COLS = 10
FT_TOKENS_COL = 'ft_tok_real'
FT_HELLASWAG_COL = 'ft_hellaswag_acc_uncond'
PT_HELLASWAG_COL = 'pt_hellaswag_acc_uncond'
CKPT_PARAMS_COL = 'ckpt_params'
CKPT_DATA_COL = 'ckpt_data'
LR_COL = 'lr'

def load_combined_data() -> pd.DataFrame:
    data_path = Path("data/combined_plotting_data_matched.pkl")
    assert data_path.exists(), f"Data file not found: {data_path}"

    console.print(f"Loading combined data from: [cyan]{data_path}[/cyan]")
    with open(data_path, 'rb') as f:
        df = pickle.load(f)

    console.print(f"Loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def display_group_summary(df: pd.DataFrame) -> None:
    console.print("\n[bold blue]📊 Matched Group Summary[/bold blue]")

    group_summary = df[['matched_group_name', 'matched_group_id', 'matched_group_med_metric']].drop_duplicates()
    group_summary = group_summary.sort_values('matched_group_id')

    console.print(f"Found {len(group_summary)} unique matched groups")

    tables = dataframe_to_fancy_tables(
        group_summary,
        max_cols_per_table_split=MAX_COLS,
        title="Matched Groups Overview"
    )

    for table in tables:
        console.print(table)

def display_token_distribution_by_group(df: pd.DataFrame) -> None:
    console.print("\n[bold blue]📊 Fine-tuning Token Distribution by Group[/bold blue]")

    df_clean = df.dropna(subset=['ft_tok_real', 'matched_group_name'])
    df_clean = df_clean.copy()
    df_clean['ft_tokens_rounded'] = (df_clean['ft_tok_real'] / 10_000_000).round() * 10

    token_pivot = df_clean.groupby(['matched_group_name', 'ft_tokens_rounded']).size().unstack(fill_value=0)
    token_pivot = token_pivot.loc[df_clean.groupby('matched_group_name')['matched_group_id'].first().sort_values().index]

    token_pivot.columns = [f"{int(col)}M" for col in token_pivot.columns]
    token_pivot['Total'] = token_pivot.sum(axis=1)

    tables = dataframe_to_fancy_tables(
        token_pivot.reset_index(),
        max_cols_per_table_split=MAX_COLS,
        title="Fine-tuning Token Counts by Group (rounded to nearest 10M)"
    )

    for table in tables:
        console.print(table)

def display_checkpoint_distribution_by_group(df: pd.DataFrame) -> None:
    console.print("\n[bold blue]📊 Checkpoint Parameter Distribution by Group[/bold blue]")

    df_clean = df.dropna(subset=['ckpt_params', 'matched_group_name'])

    params_pivot = df_clean.groupby(['matched_group_name', 'ckpt_params']).size().unstack(fill_value=0)
    params_pivot = params_pivot.loc[df_clean.groupby('matched_group_name')['matched_group_id'].first().sort_values().index]

    def sort_model_sizes(col_name: str) -> float:
        if col_name.endswith('B'):
            return float(col_name[:-1]) * 1000
        elif col_name.endswith('M'):
            return float(col_name[:-1])
        else:
            return float('inf')

    sorted_cols = sorted(params_pivot.columns, key=sort_model_sizes)
    params_pivot = params_pivot[sorted_cols]
    params_pivot['Total'] = params_pivot.sum(axis=1)

    tables = dataframe_to_fancy_tables(
        params_pivot.reset_index(),
        max_cols_per_table_split=MAX_COLS,
        title="Checkpoint Parameter Counts by Group"
    )

    for table in tables:
        console.print(table)

def display_recipe_distribution_by_group(df: pd.DataFrame) -> None:
    console.print("\n[bold blue]📊 Checkpoint Recipe Distribution by Group[/bold blue]")

    df_clean = df.dropna(subset=['ckpt_data', 'matched_group_name'])

    recipe_pivot = df_clean.groupby(['matched_group_name', 'ckpt_data']).size().unstack(fill_value=0)
    recipe_pivot = recipe_pivot.loc[df_clean.groupby('matched_group_name')['matched_group_id'].first().sort_values().index]

    recipe_pivot['Total'] = recipe_pivot.sum(axis=1)

    tables = dataframe_to_fancy_tables(
        recipe_pivot.reset_index(),
        max_cols_per_table_split=MAX_COLS,
        title="Checkpoint Recipe Counts by Group"
    )

    for table in tables:
        console.print(table)

def display_learning_rate_distribution_by_group(df: pd.DataFrame) -> None:
    console.print("\n[bold blue]📊 Fine-tuning Learning Rate Distribution by Group[/bold blue]")

    df_clean = df.dropna(subset=['lr', 'matched_group_name'])

    lr_pivot = df_clean.groupby(['matched_group_name', 'lr']).size().unstack(fill_value=0)
    lr_pivot = lr_pivot.loc[df_clean.groupby('matched_group_name')['matched_group_id'].first().sort_values().index]

    sorted_lrs = sorted(lr_pivot.columns)
    lr_pivot = lr_pivot[sorted_lrs]
    lr_pivot['Total'] = lr_pivot.sum(axis=1)

    tables = dataframe_to_fancy_tables(
        lr_pivot.reset_index(),
        max_cols_per_table_split=MAX_COLS,
        title="Fine-tuning Learning Rate Counts by Group"
    )

    for table in tables:
        console.print(table)

def display_filtered_section(df: pd.DataFrame) -> None:
    console.print(SectionRule("Fixed LR=5e-05, FT Tokens ~ 100M", "bold blue"))

def display_filtered_recipe_distribution_by_group(df: pd.DataFrame) -> None:
    console.print("\n[bold green]📊 Checkpoint Recipe Distribution by Group (LR=5e-05, ~100M tokens)[/bold green]")

    target_lr = '5e-05'
    token_min = 71_000_000
    token_max = 111_000_000

    df_filtered = df[
        (df['lr'] == target_lr) &
        (df['ft_tok_real'].between(token_min, token_max)) &
        (df['matched_group_name'].notna()) &
        (df['ckpt_data'].notna())
    ]

    if df_filtered.empty:
        console.print("[yellow]No runs found matching the filter criteria[/yellow]")
        return

    console.print(f"Found {len(df_filtered)} runs matching filter criteria (LR={target_lr}, tokens {token_min/1e6:.0f}M-{token_max/1e6:.0f}M)")

    recipe_pivot = df_filtered.groupby(['matched_group_name', 'ckpt_data']).size().unstack(fill_value=0)

    group_order = df_filtered.groupby('matched_group_name')['matched_group_id'].first().sort_values()
    if not group_order.empty:
        recipe_pivot = recipe_pivot.loc[group_order.index]

    recipe_pivot['Total'] = recipe_pivot.sum(axis=1)

    tables = dataframe_to_fancy_tables(
        recipe_pivot.reset_index(),
        max_cols_per_table_split=MAX_COLS,
        title="Checkpoint Recipe Counts by Group (LR=5e-05, Filtered)"
    )

    for table in tables:
        console.print(table)

def display_second_filtered_section(df: pd.DataFrame) -> None:
    console.print(SectionRule("Fixed LR=5e-06, FT Tokens ~ 100M", "bold blue"))

def display_second_filtered_recipe_distribution_by_group(df: pd.DataFrame) -> None:
    console.print("\n[bold green]📊 Checkpoint Recipe Distribution by Group (LR=5e-06, ~100M tokens)[/bold green]")

    target_lr = '5e-06'
    token_min = 71_000_000
    token_max = 111_000_000

    df_filtered = df[
        (df['lr'] == target_lr) &
        (df['ft_tok_real'].between(token_min, token_max)) &
        (df['matched_group_name'].notna()) &
        (df['ckpt_data'].notna())
    ]

    if df_filtered.empty:
        console.print("[yellow]No runs found matching the filter criteria[/yellow]")
        return

    console.print(f"Found {len(df_filtered)} runs matching filter criteria (LR={target_lr}, tokens {token_min/1e6:.0f}M-{token_max/1e6:.0f}M)")

    recipe_pivot = df_filtered.groupby(['matched_group_name', 'ckpt_data']).size().unstack(fill_value=0)

    group_order = df_filtered.groupby('matched_group_name')['matched_group_id'].first().sort_values()
    if not group_order.empty:
        recipe_pivot = recipe_pivot.loc[group_order.index]

    recipe_pivot['Total'] = recipe_pivot.sum(axis=1)

    tables = dataframe_to_fancy_tables(
        recipe_pivot.reset_index(),
        max_cols_per_table_split=MAX_COLS,
        title="Checkpoint Recipe Counts by Group (LR=5e-06, Filtered)"
    )

    for table in tables:
        console.print(table)

def create_group_plots(group_data: pd.DataFrame, group_name: str) -> None:
    required_cols = [FT_TOKENS_COL, FT_HELLASWAG_COL, PT_HELLASWAG_COL]
    available_cols = [col for col in required_cols if col in group_data.columns]

    if len(available_cols) < 3:
        console.print(f"[yellow]Skipping plots for {group_name} - missing required columns[/yellow]")
        return

    plot_data = group_data[available_cols].dropna()

    if plot_data.empty:
        console.print(f"[yellow]No valid data for plotting in group {group_name}[/yellow]")
        return

    console.print(f"\n[bold magenta]📈 Plots for Group: {group_name}[/bold magenta]")

    with FigureManager(PlotConfig(layout={"rows": 1, "cols": 2, "figsize": (16, 6)})) as fm:
        fm.fig.suptitle(f"Performance Analysis - {group_name}", fontsize=14)

        fm.plot(
            "scatter", 0, 0, plot_data,
            x=FT_TOKENS_COL,
            y=FT_HELLASWAG_COL,
            title="FT Tokens vs HellaSwag Accuracy",
            alpha=0.7,
        )
        fm.axes[0].set_xlabel("Fine-tune Tokens")
        fm.axes[0].set_ylabel("HellaSwag Accuracy (Fine-tuned)")

        fm.plot(
            "scatter", 0, 1, plot_data,
            x=PT_HELLASWAG_COL,
            y=FT_HELLASWAG_COL,
            title="Pre-train vs Fine-tune HellaSwag",
            alpha=0.7,
            hue_by=FT_TOKENS_COL,
        )
        fm.axes[1].set_xlabel("HellaSwag Accuracy (Pre-trained)")
        fm.axes[1].set_ylabel("HellaSwag Accuracy (Fine-tuned)")

        fm.fig.tight_layout()

        plt.show()

def display_detailed_group_analysis(df: pd.DataFrame) -> None:
    console.print("\n[bold green]📊 Detailed Analysis by Group (LR=5e-06, ~100M tokens)[/bold green]")

    target_lr = '5e-06'
    token_min = 71_000_000
    token_max = 111_000_000

    df_filtered = df[
        (df['lr'] == target_lr) &
        (df['ft_tok_real'].between(token_min, token_max)) &
        (df['matched_group_name'].notna())
    ]

    if df_filtered.empty:
        console.print("[yellow]No runs found matching the filter criteria[/yellow]")
        return

    delta_acc_cols = [col for col in df_filtered.columns if col.startswith('delta_') and col.endswith('_acc_uncond')]

    for group_name in sorted(df_filtered['matched_group_name'].unique()):
        group_data = df_filtered[df_filtered['matched_group_name'] == group_name].copy()

        console.print(f"\n[bold cyan]Group: {group_name}[/bold cyan]")
        console.print(f"Runs in this group: {len(group_data)}")

        group_data['pt_metric_value'] = None
        for idx, row in group_data.iterrows():
            comparison_metric = row.get('comparison_metric')
            if pd.notna(comparison_metric):
                pt_metric_col = f"pt_{comparison_metric}"
                if pt_metric_col in group_data.columns:
                    group_data.loc[idx, 'pt_metric_value'] = row[pt_metric_col]

        display_cols = ['ckpt_params', 'ckpt_data', 'ckpt_steps', 'comparison_metric', 'pt_metric_value'] + delta_acc_cols
        available_cols = [col for col in display_cols if col in group_data.columns]

        display_df = group_data[available_cols].copy()

        for col in delta_acc_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)

        if 'pt_metric_value' in display_df.columns:
            display_df['pt_metric_value'] = pd.to_numeric(display_df['pt_metric_value'], errors='coerce').round(4)
            display_df = display_df.sort_values('pt_metric_value', ascending=True)

        tables = dataframe_to_fancy_tables(
            display_df,
            max_cols_per_table_split=7,
            title=f"Detailed Analysis - {group_name}"
        )

        for table in tables:
            console.print(table)

        create_group_plots(df[df["matched_group_name"] == group_name], group_name)

@click.command()
@click.option("--pause", type=int, default=0, help="Pause duration in seconds")
def main(pause: int, **kwargs: Any) -> None:
    console.print("[bold blue]🔬 Combined Data Presentation Tool[/bold blue]")

    df = load_combined_data()
    display_group_summary(df)
    display_token_distribution_by_group(df)
    display_checkpoint_distribution_by_group(df)
    display_recipe_distribution_by_group(df)
    display_learning_rate_distribution_by_group(df)

    display_filtered_section(df)
    display_filtered_recipe_distribution_by_group(df)

    display_second_filtered_section(df)
    display_second_filtered_recipe_distribution_by_group(df)
    display_detailed_group_analysis(df)

    if pause > 0:
        console.print(f"\n[dim]Pausing for {pause} seconds...[/dim]")
        import time
        time.sleep(pause)


if __name__ == "__main__":
    main()