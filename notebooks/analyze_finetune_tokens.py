from __future__ import annotations

import json
from typing import Any

import pandas as pd
from rich.console import Console

from dr_showntell.datadec_utils import load_data
from dr_showntell.fancy_table import FancyTable
from dr_showntell.run_id_parsing import parse_and_group_run_ids, convert_groups_to_dataframes, apply_processing

console = Console()


def extract_all_numeric_fields(json_str: str) -> dict[str, Any]:
    try:
        data = json.loads(json_str)
        numeric_fields = {}

        def extract_recursive(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, (int, float)) and abs(value) > 1000:
                        numeric_fields[new_key] = value
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value, new_key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_key = f"{prefix}[{i}]"
                    if isinstance(item, (int, float)) and abs(item) > 1000:
                        numeric_fields[new_key] = item
                    elif isinstance(item, (dict, list)):
                        extract_recursive(item, new_key)

        extract_recursive(data)
        return numeric_fields
    except (json.JSONDecodeError, TypeError):
        return {}


def get_max_total_tokens_from_history(history_df: pd.DataFrame, run_id: str) -> int | None:
    run_history = history_df[history_df['run_id'] == run_id]
    if run_history.empty or 'total_tokens' not in run_history.columns:
        return None

    total_tokens_col = run_history['total_tokens'].dropna()
    return int(total_tokens_col.max()) if not total_tokens_col.empty else None


def analyze_finetune_token_candidates() -> None:
    console.print("[bold green]Loading data and parsing run IDs...[/bold green]")
    pretrain_df, runs_df, history_df = load_data()

    grouped_data = parse_and_group_run_ids(runs_df)
    type_dataframes = convert_groups_to_dataframes(grouped_data)
    processed_dataframes = apply_processing(type_dataframes, runs_df=runs_df, history_df=history_df)

    runs_with_finetune_tokens = []
    for run_type, df in processed_dataframes.items():
        if 'num_finetune_tokens' in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row['num_finetune_tokens']):
                    runs_with_finetune_tokens.append({
                        'run_id': row['run_id'],
                        'run_type': run_type,
                        'num_finetune_tokens': row['num_finetune_tokens']
                    })

    console.print(f"Found [cyan]{len(runs_with_finetune_tokens)}[/cyan] runs with extracted num_finetune_tokens")

    if not runs_with_finetune_tokens:
        console.print("[red]No runs found with num_finetune_tokens extracted[/red]")
        return

    analysis_results = []

    for run_info in runs_with_finetune_tokens[:10]:  # Analyze first 10 for detailed comparison
        run_id = run_info['run_id']
        expected_tokens = run_info['num_finetune_tokens']

        run_row = runs_df[runs_df['run_id'] == run_id]
        if run_row.empty:
            continue

        row_data = run_row.iloc[0]

        candidates = {}

        for col in ['config', 'summary']:
            if col in row_data and pd.notna(row_data[col]):
                numeric_fields = extract_all_numeric_fields(str(row_data[col]))
                for field_name, value in numeric_fields.items():
                    candidates[f"{col}.{field_name}"] = value

        max_total_tokens = get_max_total_tokens_from_history(history_df, run_id)
        if max_total_tokens:
            candidates['history.max_total_tokens'] = max_total_tokens

        analysis_results.append({
            'run_id': run_id,
            'expected_tokens': expected_tokens,
            'candidates': candidates
        })

    console.print(f"\n[bold blue]Detailed Analysis of First {len(analysis_results)} Runs:[/bold blue]")

    for result in analysis_results:
        console.print(f"\n[green]Run ID: {result['run_id']}[/green]")
        console.print(f"Expected num_finetune_tokens: [yellow]{result['expected_tokens']}[/yellow]")

        if result['candidates']:
            table = FancyTable(
                title=f"Candidate Fields (Run: {result['run_id'][:20]}...)",
                show_header=True,
                header_style="bold blue"
            )

            table.add_column("Field", style="cyan")
            table.add_column("Value", justify="right", style="yellow")
            table.add_column("Ratio vs Expected", justify="right", style="magenta")

            expected_tokens_val = result['expected_tokens']
            if isinstance(expected_tokens_val, str):
                expected_val = float(expected_tokens_val.rstrip('MGTB'))
                if expected_tokens_val.endswith('M'):
                    expected_millions = expected_val
                elif expected_tokens_val.endswith('G'):
                    expected_millions = expected_val * 1000
                elif expected_tokens_val.endswith('T'):
                    expected_millions = expected_val * 1_000_000
                else:
                    expected_millions = expected_val / 1_000_000
            else:
                expected_millions = float(expected_tokens_val) / 1_000_000

            sorted_candidates = sorted(
                result['candidates'].items(),
                key=lambda x: abs(x[1] / 1_000_000 - expected_millions)
            )

            for field, value in sorted_candidates:
                value_millions = value / 1_000_000
                ratio = value_millions / expected_millions if expected_millions > 0 else 0

                table.add_row(
                    field,
                    f"{value:,} ({value_millions:.1f}M)",
                    f"{ratio:.2f}x"
                )

            console.print(table)
        else:
            console.print("[dim]No large numeric candidates found[/dim]")

    console.print(f"\n[bold yellow]Summary Analysis Across All {len(runs_with_finetune_tokens)} Runs:[/bold yellow]")

    field_frequency = {}
    field_ratios = {}

    for run_info in runs_with_finetune_tokens:
        run_id = run_info['run_id']
        expected_tokens = run_info['num_finetune_tokens']

        run_row = runs_df[runs_df['run_id'] == run_id]
        if run_row.empty:
            continue

        row_data = run_row.iloc[0]
        candidates = {}

        for col in ['config', 'summary']:
            if col in row_data and pd.notna(row_data[col]):
                numeric_fields = extract_all_numeric_fields(str(row_data[col]))
                for field_name, value in numeric_fields.items():
                    candidates[f"{col}.{field_name}"] = value

        max_total_tokens = get_max_total_tokens_from_history(history_df, run_id)
        if max_total_tokens:
            candidates['history.max_total_tokens'] = max_total_tokens

        if isinstance(expected_tokens, str):
            expected_val = float(expected_tokens.rstrip('MGTB'))
            if expected_tokens.endswith('M'):
                expected_millions = expected_val
            elif expected_tokens.endswith('G'):
                expected_millions = expected_val * 1000
            elif expected_tokens.endswith('T'):
                expected_millions = expected_val * 1_000_000
            else:
                expected_millions = expected_val / 1_000_000
        else:
            expected_millions = float(expected_tokens) / 1_000_000

        for field, value in candidates.items():
            if field not in field_frequency:
                field_frequency[field] = 0
                field_ratios[field] = []

            field_frequency[field] += 1
            value_millions = value / 1_000_000
            ratio = value_millions / expected_millions if expected_millions > 0 else 0
            field_ratios[field].append(ratio)

    summary_table = FancyTable(
        title="Field Analysis Summary",
        show_header=True,
        header_style="bold green"
    )

    summary_table.add_column("Field", style="cyan")
    summary_table.add_column("Frequency", justify="right", style="yellow")
    summary_table.add_column("Avg Ratio", justify="right", style="magenta")
    summary_table.add_column("Min Ratio", justify="right", style="dim")
    summary_table.add_column("Max Ratio", justify="right", style="dim")

    for field in sorted(field_frequency.keys(), key=lambda x: field_frequency[x], reverse=True):
        freq = field_frequency[field]
        ratios = field_ratios[field]
        avg_ratio = sum(ratios) / len(ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)

        summary_table.add_row(
            field,
            str(freq),
            f"{avg_ratio:.2f}x",
            f"{min_ratio:.2f}x",
            f"{max_ratio:.2f}x"
        )

    console.print(summary_table)


if __name__ == "__main__":
    analyze_finetune_token_candidates()