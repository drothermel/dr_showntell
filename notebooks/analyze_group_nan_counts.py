from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from rich.console import Console

from dr_showntell.console_components import dataframe_to_fancy_tables

console = Console()


def load_matched_data() -> pd.DataFrame:
    pickle_files = sorted(Path('data').glob('combined_plotting_data_matched.pkl'))
    assert pickle_files, "No matched combined data found in data/ directory"

    latest_file = pickle_files[-1]
    console.print(f"Loading matched data from: [cyan]{latest_file}[/cyan]")

    with open(latest_file, 'rb') as f:
        matched_df = pickle.load(f)

    console.print(f"Loaded matched data: {matched_df.shape}")
    return matched_df


def analyze_group_structure(df: pd.DataFrame) -> None:
    console.print(f"\n[bold blue]🏗️ Analyzing group structure[/bold blue]")

    if 'matched_group_name' in df.columns:
        unique_groups = df['matched_group_name'].nunique()
        group_sizes = df['matched_group_name'].value_counts()

        console.print(f"Total number of matched groups: [cyan]{unique_groups}[/cyan]")
        console.print(f"Runs per group statistics:")
        console.print(f"  - Mean: {group_sizes.mean():.1f}")
        console.print(f"  - Median: {group_sizes.median():.1f}")
        console.print(f"  - Min: {group_sizes.min()}")
        console.print(f"  - Max: {group_sizes.max()}")
        console.print(f"  - Std: {group_sizes.std():.1f}")

        group_size_dist = group_sizes.value_counts().sort_index()
        console.print(f"\nGroup size distribution:")
        for size, count in group_size_dist.items():
            console.print(f"  - {count} groups have {size} runs each")

        # Enhanced group summary with token statistics
        if 'ft_tok_real' in df.columns:
            def categorize_tokens(tokens: float) -> str:
                if pd.isna(tokens):
                    return 'unknown'
                elif tokens < 5e6:  # < 5M
                    return '~1M'
                elif tokens < 50e6:  # < 50M
                    return '~10M'
                elif tokens < 1e9:  # < 1B
                    return '~100M'
                else:  # >= 1B
                    return '~2B'

            group_data = []
            for group_name in group_sizes.index:
                group_df = df[df['matched_group_name'] == group_name]
                tokens = group_df['ft_tok_real']

                # Token statistics
                min_tokens = tokens.min()
                avg_tokens = tokens.mean()
                max_tokens = tokens.max()

                # Token category counts
                token_categories = tokens.apply(categorize_tokens).value_counts()

                # Finetune epochs counts
                epochs_counts = {}
                if 'ft_epochs' in group_df.columns:
                    epochs_values = group_df['ft_epochs'].value_counts()
                    for epoch_val, count in epochs_values.items():
                        if pd.notna(epoch_val):
                            epochs_counts[f'epochs_{int(epoch_val)}'] = count
                        else:
                            epochs_counts['epochs_unknown'] = count

                group_data.append({
                    'group_name': group_name,
                    'run_count': len(group_df),
                    'min_ft_tokens': min_tokens,
                    'avg_ft_tokens': avg_tokens,
                    'max_ft_tokens': max_tokens,
                    'count_1M': token_categories.get('~1M', 0),
                    'count_10M': token_categories.get('~10M', 0),
                    'count_100M': token_categories.get('~100M', 0),
                    'count_2B': token_categories.get('~2B', 0),
                    'count_unknown': token_categories.get('unknown', 0),
                    **epochs_counts
                })

            group_summary_df = pd.DataFrame(group_data)
        else:
            console.print("[yellow]ft_tok_real column not found, using basic group summary[/yellow]")
            group_summary_df = pd.DataFrame({
                'group_name': group_sizes.index,
                'run_count': group_sizes.values
            }).reset_index(drop=True)

        tables = dataframe_to_fancy_tables(
            group_summary_df,
            max_cols_per_table_split=10,
            title="Runs per Group Summary with Token Statistics"
        )

        for table in tables:
            console.print(table)

    else:
        console.print("[yellow]No matched_group_name column found[/yellow]")


def analyze_group_nan_counts(df: pd.DataFrame) -> pd.DataFrame:
    console.print(f"\n[bold blue]🔍 Analyzing NaN counts in group_ columns[/bold blue]")

    group_columns = [col for col in df.columns if col.startswith('group_')]
    console.print(f"Found {len(group_columns)} group_ columns")

    if not group_columns:
        console.print("[yellow]No group_ columns found[/yellow]")
        return pd.DataFrame()

    analysis_data = []
    total_rows = len(df)

    for col in group_columns:
        nan_count = df[col].isna().sum()
        non_nan_count = total_rows - nan_count
        nan_percentage = (nan_count / total_rows) * 100

        analysis_data.append({
            'column': col,
            'total_rows': total_rows,
            'nan_count': nan_count,
            'non_nan_count': non_nan_count,
            'nan_percentage': nan_percentage
        })

    analysis_df = pd.DataFrame(analysis_data)
    analysis_df = analysis_df.sort_values('nan_percentage', ascending=False)

    return analysis_df


def display_analysis_results(analysis_df: pd.DataFrame) -> None:
    console.print(f"\n[bold blue]📊 Group Column NaN Analysis ({len(analysis_df)} columns)[/bold blue]")

    if analysis_df.empty:
        console.print("[yellow]No analysis data to display[/yellow]")
        return

    tables = dataframe_to_fancy_tables(
        analysis_df,
        max_cols_per_table_split=10,
        title="Group Column NaN Count Analysis"
    )

    for table in tables:
        console.print(table)

    completely_nan_cols = analysis_df[analysis_df['nan_percentage'] == 100.0]
    if not completely_nan_cols.empty:
        console.print(f"\n[red]⚠️  {len(completely_nan_cols)} columns are completely NaN:[/red]")
        for col in completely_nan_cols['column']:
            console.print(f"  - {col}")

    completely_valid_cols = analysis_df[analysis_df['nan_percentage'] == 0.0]
    if not completely_valid_cols.empty:
        console.print(f"\n[green]✓ {len(completely_valid_cols)} columns have no NaN values:[/green]")
        for col in completely_valid_cols['column']:
            console.print(f"  - {col}")

    partial_nan_cols = analysis_df[(analysis_df['nan_percentage'] > 0.0) & (analysis_df['nan_percentage'] < 100.0)]
    if not partial_nan_cols.empty:
        console.print(f"\n[yellow]⚠️  {len(partial_nan_cols)} columns have partial NaN values[/yellow]")


def main() -> None:
    console.print(f"[bold blue]🔄 Analyzing Matched Groups and NaN Counts[/bold blue]")

    matched_df = load_matched_data()
    analyze_group_structure(matched_df)
    analysis_df = analyze_group_nan_counts(matched_df)
    display_analysis_results(analysis_df)

    console.print(f"\n[green]✓ Analysis complete[/green]")


if __name__ == "__main__":
    main()