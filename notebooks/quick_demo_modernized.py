from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.text import Text

from dr_showntell.fancy_table import FancyTable
from dr_showntell.datadec_utils import load_data, size_match_pretrained_df, recipe_match_pretrained_df, step_match_pretrained_df, runids_match_df
from dr_showntell.run_id_parsing import parse_and_group_run_ids, convert_groups_to_dataframes, apply_processing, RECIPE_MAPPING

console = Console()


def get_pretrained_model_data(pretrain_df: pd.DataFrame, model_size: str, recipe: str, step: int | None = None) -> pd.DataFrame:
    size_match_df = size_match_pretrained_df(pretrain_df, model_size)
    if size_match_df.empty:
        console.print(f"[yellow]Warning: No data found for model size '{model_size}'[/yellow]")
        return pd.DataFrame()

    recipe_match_df = recipe_match_pretrained_df(size_match_df, recipe)
    if recipe_match_df.empty:
        console.print(f"[yellow]Warning: No data found for recipe '{recipe}' with size '{model_size}'[/yellow]")
        available_recipes = size_match_df['data'].unique()
        console.print(f"[dim]Available recipes for {model_size}: {list(available_recipes)}[/dim]")
        return pd.DataFrame()

    if step is not None:
        step_match_df = step_match_pretrained_df(recipe_match_df, step)
        if step_match_df.empty:
            console.print(f"[yellow]Warning: No data found for step {step}[/yellow]")
            available_steps = sorted(recipe_match_df['step'].unique())
            console.print(f"[dim]Available steps: {available_steps[:10]}{'...' if len(available_steps) > 10 else ''}[/dim]")
            return pd.DataFrame()
        return step_match_df
    return recipe_match_df


def display_run_type_analysis(grouped_data: dict[str, list[dict]], processed_dataframes: dict[str, pd.DataFrame]) -> None:
    console.print(f"\n[bold blue]📊 Run Type Distribution Analysis[/bold blue]")

    type_colors = {
        'matched': 'green',
        'simple_ft_vary_tokens': 'magenta',
        'simple_ft': 'blue',
        'dpo': 'bright_red',
        'old': 'bright_white',
        'reduce_type': 'white',
        'other': 'dim'
    }

    table = FancyTable(
        title="Run Type Summary",
        show_header=True,
        header_style="bold blue"
    )

    table.add_column("Run Type", style="bold")
    table.add_column("Count", justify="right", style="cyan")
    table.add_column("Has Parsed Data", justify="center", style="yellow")
    table.add_column("Token Validation", justify="center", style="magenta")

    for run_type in sorted(grouped_data.keys(), key=lambda x: len(grouped_data[x]), reverse=True):
        count = len(grouped_data[run_type])
        color = type_colors.get(run_type, 'white')

        has_processed = "✓" if run_type in processed_dataframes else "✗"

        token_validation = "N/A"
        if run_type in processed_dataframes:
            df = processed_dataframes[run_type]
            if 'num_finetune_tokens' in df.columns and 'num_finetuned_tokens_real' in df.columns:
                token_validation = "✓"
            elif 'num_finetune_tokens' in df.columns:
                token_validation = "Partial"

        table.add_row(
            f"[{color}]{run_type}[/{color}]",
            f"{count:,}",
            f"[{'green' if has_processed == '✓' else 'red'}]{has_processed}[/{'green' if has_processed == '✓' else 'red'}]",
            f"[{'green' if token_validation == '✓' else 'yellow' if token_validation == 'Partial' else 'dim'}]{token_validation}[/{'green' if token_validation == '✓' else 'yellow' if token_validation == 'Partial' else 'dim'}]"
        )

    console.print(table)


def display_token_validation_analysis(processed_dataframes: dict[str, pd.DataFrame]) -> None:
    console.print(f"\n[bold blue]🔍 Token Count Validation Analysis[/bold blue]")

    validation_results = []

    for run_type, df in processed_dataframes.items():
        if 'num_finetune_tokens' not in df.columns:
            continue

        for _, row in df.iterrows():
            parsed_tokens = row.get('num_finetune_tokens')
            real_tokens = row.get('num_finetuned_tokens_real')

            if pd.notna(parsed_tokens) and pd.notna(real_tokens):
                ratio = float(real_tokens) / float(parsed_tokens) if float(parsed_tokens) > 0 else 0
                validation_results.append({
                    'run_id': row['run_id'],
                    'run_type': run_type,
                    'parsed_tokens': parsed_tokens,
                    'real_tokens': real_tokens,
                    'ratio': ratio,
                    'accuracy': 'Excellent' if 0.8 <= ratio <= 1.2 else 'Good' if 0.5 <= ratio <= 2.0 else 'Poor'
                })

    if not validation_results:
        console.print("[yellow]No runs found with both parsed and real token counts[/yellow]")
        return

    console.print(f"Found [cyan]{len(validation_results)}[/cyan] runs with token validation data")

    accuracy_counts = {}
    for result in validation_results:
        accuracy = result['accuracy']
        accuracy_counts[accuracy] = accuracy_counts.get(accuracy, 0) + 1

    console.print(f"Accuracy Distribution:")
    for accuracy, count in accuracy_counts.items():
        color = 'green' if accuracy == 'Excellent' else 'yellow' if accuracy == 'Good' else 'red'
        console.print(f"  • [{color}]{accuracy}[/{color}]: {count} runs")

    table = FancyTable(
        title="Sample Token Validation Results",
        show_header=True,
        header_style="bold green"
    )

    table.add_column("Run Type", style="cyan")
    table.add_column("Parsed Tokens", justify="right", style="yellow")
    table.add_column("Real Tokens", justify="right", style="magenta")
    table.add_column("Ratio", justify="right", style="blue")
    table.add_column("Accuracy", style="bold")

    for result in sorted(validation_results, key=lambda x: abs(x['ratio'] - 1.0))[:10]:
        accuracy_color = 'green' if result['accuracy'] == 'Excellent' else 'yellow' if result['accuracy'] == 'Good' else 'red'

        parsed_millions = result['parsed_tokens'] / 1_000_000
        real_millions = result['real_tokens'] / 1_000_000

        table.add_row(
            result['run_type'],
            f"{parsed_millions:.1f}M",
            f"{real_millions:.1f}M",
            f"{result['ratio']:.2f}x",
            f"[{accuracy_color}]{result['accuracy']}[/{accuracy_color}]"
        )

    console.print(table)


def display_pattern_showcase(processed_dataframes: dict[str, pd.DataFrame]) -> None:
    console.print(f"\n[bold blue]🎯 Pattern Recognition Showcase[/bold blue]")

    pattern_examples = {}

    for run_type, df in processed_dataframes.items():
        if not df.empty and 'pattern_name' in df.columns:
            patterns = df['pattern_name'].value_counts()
            for pattern_name, count in patterns.items():
                if pattern_name not in pattern_examples:
                    pattern_examples[pattern_name] = []
                pattern_examples[pattern_name].append((run_type, count))

    if not pattern_examples:
        console.print("[yellow]No pattern information found in processed data[/yellow]")
        return

    table = FancyTable(
        title="Regex Pattern Recognition Summary",
        show_header=True,
        header_style="bold blue"
    )

    table.add_column("Pattern Name", style="cyan")
    table.add_column("Run Types", style="yellow")
    table.add_column("Total Matches", justify="right", style="magenta")

    for pattern_name in sorted(pattern_examples.keys()):
        type_counts = pattern_examples[pattern_name]
        run_types = ", ".join([f"{run_type}({count})" for run_type, count in type_counts])
        total_matches = sum(count for _, count in type_counts)

        table.add_row(
            pattern_name,
            run_types,
            f"{total_matches:,}"
        )

    console.print(table)


def display_comprehensive_run_example(processed_dataframes: dict[str, pd.DataFrame], pretrain_df: pd.DataFrame, runs_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    console.print(f"\n[bold blue]🔬 Comprehensive Run Analysis Example[/bold blue]")

    best_run_type = None
    best_run = None

    for run_type, df in processed_dataframes.items():
        if run_type in ['matched', 'simple_ft_vary_tokens', 'simple_ft']:
            for _, row in df.iterrows():
                if (pd.notna(row.get('num_finetune_tokens')) and
                    pd.notna(row.get('initial_checkpoint_size')) and
                    pd.notna(row.get('comparison_model_size'))):
                    best_run_type = run_type
                    best_run = row
                    break
            if best_run is not None:
                break

    if best_run is None:
        console.print("[yellow]No suitable run found for comprehensive analysis[/yellow]")
        return

    run_id = best_run['run_id']
    console.print(f"\n[bold green]Selected Run: {run_id}[/bold green]")
    console.print(f"Type: [{best_run_type}]{best_run_type}[/{best_run_type}]")

    console.print(f"\n[bold cyan]📋 Extracted Components:[/bold cyan]")

    component_table = FancyTable(show_header=True, header_style="bold green")
    component_table.add_column("Component", style="cyan")
    component_table.add_column("Value", style="yellow")

    key_components = [
        ('timestamp', 'Timestamp'),
        ('exp_name', 'Experiment Name'),
        ('comparison_model_size', 'Comparison Model Size'),
        ('comparison_model_recipe', 'Comparison Recipe'),
        ('num_finetune_tokens', 'Finetune Tokens (Parsed)'),
        ('num_finetune_epochs', 'Finetune Epochs'),
        ('initial_checkpoint_size', 'Initial Checkpoint Size'),
        ('initial_checkpoint_recipe', 'Checkpoint Recipe'),
        ('initial_checkpoint_steps', 'Checkpoint Steps'),
        ('lr', 'Learning Rate'),
        ('seed', 'Seed'),
        ('num_finetuned_tokens_real', 'Real Tokens (from logs)')
    ]

    for key, display_name in key_components:
        value = best_run.get(key, 'N/A')
        if pd.notna(value):
            if key in ['num_finetune_tokens', 'num_finetuned_tokens_real'] and isinstance(value, (int, float)):
                value = f"{value/1_000_000:.1f}M"
            component_table.add_row(display_name, str(value))

    console.print(component_table)

    if pd.notna(best_run.get('comparison_model_size')) and pd.notna(best_run.get('comparison_model_recipe')):
        console.print(f"\n[bold cyan]🔍 Pretrained Model Data Lookup:[/bold cyan]")

        size = best_run['comparison_model_size']
        recipe_raw = best_run['comparison_model_recipe']
        recipe_mapped = RECIPE_MAPPING.get(recipe_raw, recipe_raw)

        console.print(f"Looking for: [blue]{size}[/blue] + [magenta]{recipe_raw}[/magenta] → [green]{recipe_mapped}[/green]")

        pretrained_data = get_pretrained_model_data(pretrain_df, size, recipe_mapped)

        if not pretrained_data.empty:
            console.print(f"[green]✓ Found {len(pretrained_data)} pretrained model records[/green]")
            if 'metric' in pretrained_data.columns:
                console.print(f"  • Metrics: {pretrained_data['metric'].nunique()}")
            console.print(f"  • Steps: {pretrained_data['step'].nunique()}")
            console.print(f"  • Step range: {pretrained_data['step'].min()}-{pretrained_data['step'].max()}")
        else:
            console.print(f"[red]✗ No pretrained data found[/red]")

    run_row = runids_match_df(runs_df, run_id)
    history_data = runids_match_df(history_df, run_id)

    console.print(f"\n[bold cyan]📊 Data Availability:[/bold cyan]")
    console.print(f"  • Runs metadata: [{'green' if not run_row.empty else 'red'}]{'✓' if not run_row.empty else '✗'}[/{'green' if not run_row.empty else 'red'}]")
    console.print(f"  • History data: [{'green' if not history_data.empty else 'red'}]{'✓' if not history_data.empty else '✗'}[/{'green' if not history_data.empty else 'red'}] ({len(history_data)} records)")


def display_data_quality_insights(runs_df: pd.DataFrame, processed_dataframes: dict[str, pd.DataFrame]) -> None:
    console.print(f"\n[bold blue]📈 Data Quality Insights[/bold blue]")

    total_runs = len(runs_df)
    classified_runs = sum(len(df) for df in processed_dataframes.values())

    console.print(f"Total runs in dataset: [cyan]{total_runs:,}[/cyan]")
    console.print(f"Successfully classified: [green]{classified_runs:,}[/green] ({classified_runs/total_runs*100:.1f}%)")
    console.print(f"Unclassified: [yellow]{total_runs - classified_runs:,}[/yellow] ({(total_runs - classified_runs)/total_runs*100:.1f}%)")

    data_completeness = {}

    for run_type, df in processed_dataframes.items():
        if df.empty:
            continue

        completeness = {}
        for col in df.columns:
            if col != 'run_id':
                non_null_count = df[col].count()
                completeness[col] = non_null_count / len(df) * 100

        data_completeness[run_type] = completeness

    if data_completeness:
        console.print(f"\n[bold yellow]Data Completeness by Run Type:[/bold yellow]")

        for run_type in sorted(data_completeness.keys()):
            completeness = data_completeness[run_type]
            avg_completeness = sum(completeness.values()) / len(completeness)

            console.print(f"\n[cyan]{run_type}[/cyan] (avg: {avg_completeness:.1f}%):")

            for field in sorted(completeness.keys()):
                percentage = completeness[field]
                color = 'green' if percentage >= 90 else 'yellow' if percentage >= 50 else 'red'
                console.print(f"  • {field}: [{color}]{percentage:.1f}%[/{color}]")


def extract_and_pickle_modernized_data(runs_df: pd.DataFrame, history_df: pd.DataFrame, pretrain_df: pd.DataFrame) -> str:
    console.print(f"[bold blue]Extracting modernized comprehensive run data...[/bold blue]")

    grouped_data = parse_and_group_run_ids(runs_df)
    type_dataframes = convert_groups_to_dataframes(grouped_data)
    processed_dataframes = apply_processing(type_dataframes, runs_df=runs_df, history_df=history_df)

    all_processed_runs = []
    for run_type, df in processed_dataframes.items():
        for _, row in df.iterrows():
            run_data = row.to_dict()
            run_data['_run_type'] = run_type
            all_processed_runs.append(run_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_modernized_run_data.pkl"
    filepath = Path("data") / filename

    filepath.parent.mkdir(exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump({
            'processed_runs': all_processed_runs,
            'grouped_data': grouped_data,
            'processed_dataframes': processed_dataframes,
            'metadata': {
                'total_runs': len(runs_df),
                'classified_runs': len(all_processed_runs),
                'run_types': list(processed_dataframes.keys()),
                'extraction_timestamp': timestamp
            }
        }, f)

    console.print(f"[green]✓ Saved {len(all_processed_runs)} processed runs to: {filepath}[/green]")
    console.print(f"[dim]File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB[/dim]")

    return str(filepath)


def main() -> None:
    console.print(f"[bold green]🚀 DR ShowNTell Modernized Demo[/bold green]")
    console.print(f"[dim]Using centralized run ID parsing and validation system[/dim]\n")

    pretrain_df, runs_df, history_df = load_data()
    console.print(f"Loaded datasets: Pretrain ({len(pretrain_df):,} rows), Runs ({len(runs_df):,} rows), History ({len(history_df):,} rows)\n")

    console.print(f"[bold blue]🔄 Processing all runs with centralized parsing system...[/bold blue]")

    grouped_data = parse_and_group_run_ids(runs_df)
    console.print(f"[green]✓ Classified runs into {len(grouped_data)} types[/green]")

    type_dataframes = convert_groups_to_dataframes(grouped_data)
    console.print(f"[green]✓ Converted to structured dataframes[/green]")

    processed_dataframes = apply_processing(type_dataframes, runs_df=runs_df, history_df=history_df)
    console.print(f"[green]✓ Applied processing and validation[/green]")

    display_run_type_analysis(grouped_data, processed_dataframes)
    display_token_validation_analysis(processed_dataframes)
    display_pattern_showcase(processed_dataframes)
    display_comprehensive_run_example(processed_dataframes, pretrain_df, runs_df, history_df)
    display_data_quality_insights(runs_df, processed_dataframes)

    console.print(f"\n[bold blue]💾 Saving modernized data extraction...[/bold blue]")
    filepath = extract_and_pickle_modernized_data(runs_df, history_df, pretrain_df)
    console.print(f"Results saved to: [cyan]{filepath}[/cyan]")

    console.print(f"\n[bold green]🎉 Modernized demo complete![/bold green]")
    console.print(f"[dim]This demo showcases the library's centralized architecture and validation capabilities[/dim]")


if __name__ == "__main__":
    main()