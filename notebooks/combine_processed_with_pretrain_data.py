from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from rich.console import Console

from dr_showntell.datadec_utils import load_data, filter_by_model_size, filter_by_recipe, filter_by_step, filter_by_run_id
from dr_showntell.fancy_table import FancyTable

console = Console()


def get_pretrained_data_counts(pretrain_df: pd.DataFrame, model_size: str | None, recipe: str | None, step: int | None = None) -> int:
    if not model_size or not recipe:
        return 0

    filtered_df = filter_by_model_size(pretrain_df, model_size)
    if filtered_df.empty:
        return 0

    filtered_df = filter_by_recipe(filtered_df, recipe)
    if filtered_df.empty:
        return 0

    if step is not None:
        filtered_df = filter_by_step(filtered_df, step)

    return len(filtered_df)


def load_datasets() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pickle_files = sorted(Path('data').glob('*_modernized_run_data.pkl'))
    assert pickle_files, "No modernized pickle files found in data/ directory"

    latest_file = pickle_files[-1]
    console.print(f"Loading processed data from: [cyan]{latest_file}[/cyan]")

    with open(latest_file, 'rb') as f:
        processed_data = pickle.load(f)

    pretrain_df, runs_df, history_df = load_data()
    console.print(f"Loaded raw data: pretrain({len(pretrain_df):,}), runs({len(runs_df):,}), history({len(history_df):,})")

    return processed_data, pretrain_df, runs_df, history_df


def analyze_run_data_availability(
    processed_runs: list[dict], pretrain_df: pd.DataFrame, runs_df: pd.DataFrame, history_df: pd.DataFrame
) -> list[dict]:
    console.print(f"Processing {len(processed_runs):,} runs...")
    analysis_results = []

    for i, run in enumerate(processed_runs):
        if i % 50 == 0:
            console.print(f"  Processing run {i+1}/{len(processed_runs)}...")

        run_id = run['run_id']
        comparison_size = run.get('comparison_model_size')
        comparison_recipe = run.get('comparison_model_recipe')
        initial_size = run.get('initial_checkpoint_size')
        initial_recipe = run.get('initial_checkpoint_recipe')
        initial_steps = run.get('initial_checkpoint_steps')

        history_rows = len(filter_by_run_id(history_df, run_id))

        runs_row = filter_by_run_id(runs_df, run_id)
        run_state = "N/A"
        if not runs_row.empty:
            run_state = runs_row.iloc[0].get('state', 'N/A')

        comparison_pretrain_rows = get_pretrained_data_counts(
            pretrain_df, comparison_size, comparison_recipe
        )

        initial_steps_int = None
        if initial_steps and initial_steps != 'main':
            try:
                initial_steps_int = int(initial_steps)
            except (ValueError, TypeError):
                pass

        initial_pretrain_rows = get_pretrained_data_counts(
            pretrain_df, initial_size, initial_recipe, initial_steps_int
        )

        analysis_results.append({
            'run_id': run_id,
            'run_type': run.get('_run_type', 'unknown'),
            'run_state': run_state,
            'comparison_model_size': comparison_size or 'N/A',
            'comparison_recipe': comparison_recipe or 'N/A',
            'initial_model_size': initial_size or 'N/A',
            'initial_recipe': initial_recipe or 'N/A',
            'initial_steps': initial_steps or 'N/A',
            'num_history_rows': history_rows,
            'num_pretrain_comparison': comparison_pretrain_rows,
            'num_pretrain_initial': initial_pretrain_rows
        })

    console.print(f"[green]✓ Analysis complete for {len(analysis_results)} runs[/green]")
    return analysis_results


def display_analysis_results(analysis_results: list[dict]) -> None:
    console.print(f"\n[bold blue]📊 Combined Data Analysis Results[/bold blue]")

    run_states = {}
    for r in analysis_results:
        state = r['run_state']
        run_states[state] = run_states.get(state, 0) + 1

    summary_stats = {
        'total_runs': len(analysis_results),
        'with_history': sum(1 for r in analysis_results if r['num_history_rows'] > 0),
        'with_comparison_pretrain': sum(1 for r in analysis_results if r['num_pretrain_comparison'] > 0),
        'with_initial_pretrain': sum(1 for r in analysis_results if r['num_pretrain_initial'] > 0),
        'with_both_pretrain': sum(1 for r in analysis_results if r['num_pretrain_comparison'] > 0 and r['num_pretrain_initial'] > 0),
        'run_states': run_states
    }

    console.print(f"[yellow]Summary Statistics:[/yellow]")
    console.print(f"  • Total runs analyzed: [cyan]{summary_stats['total_runs']:,}[/cyan]")
    console.print(f"  • Runs with history data: [green]{summary_stats['with_history']:,}[/green] ({summary_stats['with_history']/summary_stats['total_runs']*100:.1f}%)")
    console.print(f"  • Runs with comparison pretraining data: [blue]{summary_stats['with_comparison_pretrain']:,}[/blue] ({summary_stats['with_comparison_pretrain']/summary_stats['total_runs']*100:.1f}%)")
    console.print(f"  • Runs with initial checkpoint pretraining data: [magenta]{summary_stats['with_initial_pretrain']:,}[/magenta] ({summary_stats['with_initial_pretrain']/summary_stats['total_runs']*100:.1f}%)")
    console.print(f"  • Runs with both pretraining datasets: [yellow]{summary_stats['with_both_pretrain']:,}[/yellow] ({summary_stats['with_both_pretrain']/summary_stats['total_runs']*100:.1f}%)")

    console.print(f"\n[bold cyan]Run States Distribution:[/bold cyan]")
    for state, count in sorted(summary_stats['run_states'].items(), key=lambda x: x[1], reverse=True):
        console.print(f"  • {state}: [yellow]{count:,}[/yellow] ({count/summary_stats['total_runs']*100:.1f}%)")

    display_complete_results_table(analysis_results)
    display_run_type_breakdown(analysis_results)


def display_complete_results_table(analysis_results: list[dict]) -> None:
    console.print(f"\n[bold green]📋 Complete Results Table (all {len(analysis_results)} runs):[/bold green]")

    table = FancyTable(
        title="Combined Run and Pretraining Data Analysis - All Runs",
        show_header=True,
        header_style="bold blue"
    )

    table.add_column("Run Type", style="cyan")
    table.add_column("Run State", style="bright_yellow")
    table.add_column("Run ID", style="dim")
    table.add_column("Comp Size", style="yellow")
    table.add_column("Comp Recipe", style="green")
    table.add_column("Init Size", style="yellow")
    table.add_column("Init Recipe", style="green")
    table.add_column("Init Steps", style="magenta")
    table.add_column("Hist Rows", justify="right", style="blue")
    table.add_column("Comp PT Rows", justify="right", style="magenta")
    table.add_column("Init PT Rows", justify="right", style="red")

    for result in analysis_results:
        table.add_row(
            result['run_type'],
            result['run_state'],
            result['run_id'],
            result['comparison_model_size'],
            result['comparison_recipe'],
            result['initial_model_size'],
            result['initial_recipe'],
            result['initial_steps'],
            f"{result['num_history_rows']:,}",
            f"{result['num_pretrain_comparison']:,}" if result['num_pretrain_comparison'] > 0 else "0",
            f"{result['num_pretrain_initial']:,}" if result['num_pretrain_initial'] > 0 else "0"
        )

    console.print(table)


def display_run_type_breakdown(analysis_results: list[dict]) -> None:
    console.print(f"\n[bold blue]📈 Run Type Breakdown:[/bold blue]")

    type_analysis = {}
    for result in analysis_results:
        run_type = result['run_type']
        if run_type not in type_analysis:
            type_analysis[run_type] = {
                'count': 0,
                'with_history': 0,
                'with_comparison': 0,
                'with_initial': 0,
                'avg_history_rows': 0,
                'avg_comparison_rows': 0,
                'avg_initial_rows': 0
            }

        stats = type_analysis[run_type]
        stats['count'] += 1
        if result['num_history_rows'] > 0:
            stats['with_history'] += 1
        if result['num_pretrain_comparison'] > 0:
            stats['with_comparison'] += 1
        if result['num_pretrain_initial'] > 0:
            stats['with_initial'] += 1

        stats['avg_history_rows'] += result['num_history_rows']
        stats['avg_comparison_rows'] += result['num_pretrain_comparison']
        stats['avg_initial_rows'] += result['num_pretrain_initial']

    for run_type, stats in type_analysis.items():
        if stats['count'] > 0:
            stats['avg_history_rows'] /= stats['count']
            stats['avg_comparison_rows'] /= stats['count']
            stats['avg_initial_rows'] /= stats['count']

    type_table = FancyTable(
        title="Analysis by Run Type",
        show_header=True,
        header_style="bold green"
    )

    type_table.add_column("Run Type", style="cyan")
    type_table.add_column("Count", justify="right", style="yellow")
    type_table.add_column("w/ History", justify="right", style="blue")
    type_table.add_column("w/ Comp PT", justify="right", style="magenta")
    type_table.add_column("w/ Init PT", justify="right", style="red")
    type_table.add_column("Avg Hist", justify="right", style="dim")
    type_table.add_column("Avg Comp", justify="right", style="dim")
    type_table.add_column("Avg Init", justify="right", style="dim")

    for run_type in sorted(type_analysis.keys(), key=lambda x: type_analysis[x]['count'], reverse=True):
        stats = type_analysis[run_type]

        type_table.add_row(
            run_type,
            f"{stats['count']:,}",
            f"{stats['with_history']:,}",
            f"{stats['with_comparison']:,}",
            f"{stats['with_initial']:,}",
            f"{stats['avg_history_rows']:.1f}",
            f"{stats['avg_comparison_rows']:.0f}",
            f"{stats['avg_initial_rows']:.0f}"
        )

    console.print(type_table)


def save_combined_analysis(analysis_results: list[dict], extraction_timestamp: str) -> None:
    console.print(f"\n[bold blue]💾 Saving combined analysis results...[/bold blue]")

    combined_df = pd.DataFrame(analysis_results)
    output_file = f"data/combined_analysis_{extraction_timestamp}.parquet"
    combined_df.to_parquet(output_file, index=False)

    console.print(f"[green]✓ Saved combined analysis to: {output_file}[/green]")
    console.print(f"[dim]Columns: {list(combined_df.columns)}[/dim]")


def combine_processed_with_pretrain_data() -> None:
    console.print(f"[bold blue]🔄 Combining Processed Runs with Pretraining Data[/bold blue]")

    processed_data, pretrain_df, runs_df, history_df = load_datasets()
    processed_runs = processed_data['processed_runs']

    analysis_results = analyze_run_data_availability(processed_runs, pretrain_df, runs_df, history_df)
    display_analysis_results(analysis_results)
    save_combined_analysis(analysis_results, processed_data['metadata']['extraction_timestamp'])


if __name__ == "__main__":
    combine_processed_with_pretrain_data()