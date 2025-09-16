# %%
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.text import Text

from dr_showntell.fancy_table import FancyTable
from dr_showntell.datadec_utils import (
    parse_run_id_components, load_data, match_runids_pattern, size_match_pretrained_df, recipe_match_pretrained_df, step_match_pretrained_df, runids_match_df,
)
console = Console()

# Recipe mapping dictionaries

# For checkpoint models (short codes -> pretrain data format)
CHECKPOINT_RECIPE_MAPPING = {
    "d17": "Dolma1.7",
    "d16": "Dolma1.6++",
    "c4": "C4",
    "dclm": "DCLM-Baseline",
    "dclm_qc10p": "DCLM-Baseline (QC 10%)",
    "dclm_qc20p": "DCLM-Baseline (QC 20%)",
    "dclm_qc7p_fw2": "DCLM-Baseline (QC 7%, FW2)",
    "dclm_qc7p_fw3": "DCLM-Baseline (QC 7%, FW3)",
    "dclm_qcfw10p": "DCLM-Baseline (QC FW 10%)",
    "dclm_qcfw3p": "DCLM-Baseline (QC FW 3%)",
    "dclm25_dolma75": "DCLM-Baseline 25% / Dolma 75%",
    "dclm50_dolma50": "DCLM-Baseline 50% / Dolma 50%",
    "dclm75_dolma25": "DCLM-Baseline 75% / Dolma 25%",
    "falcon": "Falcon",
    "falcon_cc": "Falcon+CC",
    "falcon_cc_qc10p": "Falcon+CC (QC 10%)",
    "falcon_cc_qc20p": "Falcon+CC (QC 20%)",
    "falcon_cc_qcorig10p": "Falcon+CC (QC Orig 10%)",
    "falcon_cc_qctulu10p": "Falcon+CC (QC Tulu 10%)",
    "fineweb_edu": "FineWeb-Edu",
    "fineweb_pro": "FineWeb-Pro",
    # Add more as needed
}

# For comparison models (logical names -> pretrain data format)
COMPARISON_RECIPE_MAPPING = {
    "Dolma 1.7": "Dolma1.7",
    "Dolma 1.6": "Dolma1.6++",
    "C4": "C4",
    "DCLM": "DCLM-Baseline",
    "DCLM Baseline": "DCLM-Baseline",
    "Falcon": "Falcon",
    "FineWeb": "FineWeb-Edu",
}

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

def get_run_data_comprehensive(run_id: str, runs_df: pd.DataFrame, history_df: pd.DataFrame, pretrain_df: pd.DataFrame) -> dict:
    assert "run_id" in history_df.columns, "run_id column not found in history dataframe"
    components = parse_run_id_components(run_id)

    run_row = runids_match_df(runs_df, run_id)
    if run_row.empty:
        console.print(f"[red]Error: Run ID '{run_id}' not found in runs dataset[/red]")
        return {}

    summary_dict = {}
    if 'summary' in run_row.columns and not run_row['summary'].empty:
        try:
            import json
            summary_str = run_row['summary'].iloc[0]
            if isinstance(summary_str, str):
                summary_dict = json.loads(summary_str)
            elif isinstance(summary_str, dict):
                summary_dict = summary_str
        except (json.JSONDecodeError, AttributeError):
            console.print(f"[yellow]Warning: Could not parse summary for run {run_id}[/yellow]")

    history_data = runids_match_df(history_df, run_id)
    initial_recipe_code = components.get('initial_checkpoint_recipe', '')
    initial_recipe = CHECKPOINT_RECIPE_MAPPING.get(initial_recipe_code, initial_recipe_code)
    comparison_recipe_logical = components.get('comparison_model_recipe', '')
    comparison_recipe_mapped = COMPARISON_RECIPE_MAPPING.get(comparison_recipe_logical, comparison_recipe_logical)
    comparison_model_data = pd.DataFrame()
    if components.get('comparison_model_size') and comparison_recipe_mapped:
        comparison_model_data = get_pretrained_model_data(
            pretrain_df,
            components['comparison_model_size'],
            comparison_recipe_mapped
        )
    initial_checkpoint_data = pd.DataFrame()
    if components.get('initial_checkpoint_size') and initial_recipe:
        try:
            checkpoint_step = int(components.get('initial_checkpoint_steps', 0))
            initial_checkpoint_data = get_pretrained_model_data(
                pretrain_df,
                components['initial_checkpoint_size'],
                initial_recipe,
                checkpoint_step
            )
        except (ValueError, TypeError):
            console.print(f"[yellow]Warning: Could not parse checkpoint step for {run_id}[/yellow]")
    return {
        'run_id': run_id,
        'components': components,
        'summary': summary_dict,
        'history_data': history_data,
        'comparison_model_data': comparison_model_data,
        'initial_checkpoint_data': initial_checkpoint_data,
        'run_row': run_row.iloc[0] if not run_row.empty else {}
    }

def extract_run_ids_with_pattern(df: pd.DataFrame, pattern_match: str = "") -> list[str]:
    assert "run_id" in df.columns, "run_id column not found in dataframe"
    filtered_ids = match_runids_pattern(df, pattern_match)
    console.print(f"Found {len(filtered_ids)} run_ids matching pattern '{pattern_match}'")
    return filtered_ids

def safe_truncate(value: str | None, max_len: int) -> str:
    if value is None:
        return "N/A"
    return value[:max_len] + "..." if len(value) > max_len else value

def display_comprehensive_run_analysis(run_data: dict) -> None:
    if not run_data:
        return

    run_id = run_data['run_id']
    components = run_data['components']

    console.print(f"\n[bold blue]═══ Comprehensive Analysis for {run_id} ═══[/bold blue]")

    console.print(f"\n[bold green]📋 Experiment Components:[/bold green]")
    console.print(f"  📅 DateTime: [yellow]{components.get('datetime', 'N/A')}[/yellow]")
    console.print(f"  🧪 Experiment: [magenta]{components.get('exp_name', 'N/A')}[/magenta] ([green]{components.get('exp_type', 'N/A')}[/green])")
    console.print(f"  📊 Comparison Model: [blue]{components.get('comparison_model_size', 'N/A')} ({components.get('comparison_model_recipe', 'N/A')})[/blue]")
    console.print(f"  🎯 Finetune Config: [cyan]{components.get('num_finetune_tokens', 'N/A')} tokens × {components.get('num_finetune_epochs', 'N/A')} epochs[/cyan]")
    console.print(f"  🏁 Initial Checkpoint: [red]{components.get('initial_checkpoint_size', 'N/A')} ({components.get('initial_checkpoint_recipe', 'N/A')}) @ step {components.get('initial_checkpoint_steps', 'N/A')}[/red]")
    console.print(f"  ⚙️ LR: [bright_yellow]{components.get('lr', 'N/A')}[/bright_yellow], Seed: [dim]{components.get('seed', 'N/A')}[/dim]")

    console.print(f"\n[bold green]📈 Data Availability:[/bold green]")
    console.print(f"  • Summary data: [{'green' if run_data['summary'] else 'red'}]{'✓' if run_data['summary'] else '✗'}[/{'green' if run_data['summary'] else 'red'}] ({len(run_data['summary'])} keys)")
    console.print(f"  • History data: [{'green' if len(run_data['history_data']) > 0 else 'red'}]{'✓' if len(run_data['history_data']) > 0 else '✗'}[/{'green' if len(run_data['history_data']) > 0 else 'red'}] ({len(run_data['history_data'])} rows)")
    console.print(f"  • Comparison model data: [{'green' if len(run_data['comparison_model_data']) > 0 else 'red'}]{'✓' if len(run_data['comparison_model_data']) > 0 else '✗'}[/{'green' if len(run_data['comparison_model_data']) > 0 else 'red'}] ({len(run_data['comparison_model_data'])} rows)")
    console.print(f"  • Initial checkpoint data: [{'green' if len(run_data['initial_checkpoint_data']) > 0 else 'red'}]{'✓' if len(run_data['initial_checkpoint_data']) > 0 else '✗'}[/{'green' if len(run_data['initial_checkpoint_data']) > 0 else 'red'}] ({len(run_data['initial_checkpoint_data'])} rows)")

    if run_data['summary']:
        console.print(f"\n[bold green]📊 Sample Summary Data:[/bold green]")
        sample_keys = list(run_data['summary'].keys())[:5]
        for key in sample_keys:
            value = run_data['summary'][key]
            if isinstance(value, (int, float)):
                console.print(f"  • {key}: [cyan]{value}[/cyan]")
            else:
                value_str = str(value)[:30] + "..." if len(str(value)) > 30 else str(value)
                console.print(f"  • {key}: [dim]{value_str}[/dim]")



def display_run_id_analysis(df: pd.DataFrame, pattern: str = "", max_display: int = 10) -> None:
    console.print(f"\n[bold blue]Run ID Analysis[/bold blue]")

    run_ids = extract_run_ids_with_pattern(df, pattern)

    if not run_ids:
        console.print("[yellow]No matching run_ids found[/yellow]")
        return

    # Display first few run_ids
    display_count = min(len(run_ids), max_display)

    table = FancyTable(
        title=f"Run ID Analysis (showing {display_count} of {len(run_ids)})",
        show_header=True,
        header_style="bold green"
    )

    table.add_column("Run ID", style="cyan")
    table.add_column("DateTime", style="yellow")
    table.add_column("Exp Name", style="magenta")
    table.add_column("Type", style="green")
    table.add_column("Comp Size", style="blue")
    table.add_column("Init Size", style="red")
    table.add_column("Steps", style="dim")
    table.add_column("LR", style="bright_yellow")

    for i, run_id in enumerate(run_ids[:display_count]):
        components = parse_run_id_components(run_id)
        row_style = "dim" if i % 2 == 1 else ""

        table.add_row(
            safe_truncate(components['full_id'], 25),
            safe_truncate(components['datetime'], 16),
            safe_truncate(components['exp_name'], 15),
            components['exp_type'] or "N/A",
            components['comparison_model_size'] or "N/A",
            components['initial_checkpoint_size'] or "N/A",
            components['initial_checkpoint_steps'] or "N/A",
            components['lr'] or "N/A",
            style=row_style
        )

    console.print(table)


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


def extract_and_pickle_matched_run_data(runs_df: pd.DataFrame, history_df: pd.DataFrame, pretrain_df: pd.DataFrame, pattern: str = "") -> str:
    console.print(f"[bold blue]Extracting comprehensive run data for pattern: '{pattern}'[/bold blue]")

    all_run_ids = extract_run_ids_with_pattern(runs_df, pattern)

    run_ids_with_history = []
    for run_id in all_run_ids:
        if "_match_" not in run_id:
            continue
        history_data = runids_match_df(history_df, run_id)
        if len(history_data) > 0:
            run_ids_with_history.append(run_id)

    console.print(f"Found {len(run_ids_with_history)} run_ids with matching history data (out of {len(all_run_ids)} total matches)")

    comprehensive_data_list = []
    for i, run_id in enumerate(run_ids_with_history):
        if i % 50 == 0:
            console.print(f"Processing run {i+1}/{len(run_ids_with_history)}: {run_id[:50]}...")

        run_data = get_run_data_comprehensive(run_id, runs_df, history_df, pretrain_df)
        if run_data:
            comprehensive_data_list.append(run_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_matched_run_data.pkl"
    filepath = Path("data") / filename

    with open(filepath, "wb") as f:
        pickle.dump(comprehensive_data_list, f)

    console.print(f"[green]✓ Saved {len(comprehensive_data_list)} comprehensive run data entries to: {filepath}[/green]")
    console.print(f"[dim]File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB[/dim]")

    return str(filepath)


def main() -> None:
    pretrain_df, runs_df, history_df = load_data()

    console.print(f"Loaded datasets: Pretrain ({len(pretrain_df):,} rows), Runs ({len(runs_df):,} rows), History ({len(history_df):,} rows)")
    console.print()

    display_dataframe_preview(pretrain_df, "Pretrain Dataset Preview", n_rows=5, max_cell_width=50)
    console.print()

    display_dataframe_preview(runs_df, "Runs Metadata Preview", n_rows=5, max_cell_width=50)
    console.print()

    display_dataframe_preview(history_df, "History Dataset Preview", n_rows=5, max_cell_width=50)

    # Demonstrate run_id extraction and analysis
    display_run_id_analysis(runs_df, pattern="", max_display=8)

    # Example: Look for specific patterns
    console.print("\n[bold blue]Pattern-based Run ID Search Examples:[/bold blue]")

    finetune_runs = extract_run_ids_with_pattern(runs_df, "finetune")
    if finetune_runs:
        console.print(f"Sample finetune run IDs:")
        for run_id in finetune_runs[:3]:
            console.print(f"  • {run_id}")

    console.print()
    test_runs = extract_run_ids_with_pattern(runs_df, "test")
    if test_runs:
        console.print(f"Sample test run IDs:")
        for run_id in test_runs[:3]:
            console.print(f"  • {run_id}")

    # Show parsed components for a specific run
    if finetune_runs:
        console.print(f"\n[bold blue]Example Run ID Parsing:[/bold blue]")
        sample_run = finetune_runs[0]
        components = parse_run_id_components(sample_run)

        console.print(f"Run ID: [cyan]{sample_run}[/cyan]")
        for key, value in components.items():
            if value and key != 'full_id':
                console.print(f"  {key.replace('_', ' ').title()}: [yellow]{value}[/yellow]")

    # Test with your specific examples
    console.print(f"\n[bold blue]Testing with Specified Format Examples:[/bold blue]")

    test_examples = [
        "250904-222144_test_match_150M_finetune_10Mtx1_DD-d17-300M-17500-2_lr=5e-05",
        "250904-222144_test_match_150M_finetune_10Mtx1_DD-d17-530M-7500-2_lr=5e-05"
    ]

    for example in test_examples:
        console.print(f"\nParsing: [cyan]{example}[/cyan]")
        components = parse_run_id_components(example)

        # Display in organized format
        console.print(f"  📅 DateTime: [yellow]{components['datetime']}[/yellow]")
        console.print(f"  🧪 Exp Name: [magenta]{components['exp_name']}[/magenta]")
        console.print(f"  🔧 Exp Type: [green]{components['exp_type']}[/green]")
        console.print(f"  📊 Comparison Model: [blue]{components['comparison_model_size']} ({components['comparison_model_recipe']})[/blue]")
        console.print(f"  🎯 Finetune: [cyan]{components['num_finetune_tokens']} tokens, {components['num_finetune_epochs']} epochs[/cyan]")
        console.print(f"  🏁 Initial Checkpoint: [red]{components['initial_checkpoint_size']} ({components['initial_checkpoint_recipe']}), {components['initial_checkpoint_steps']} steps[/red]")
        console.print(f"  🌱 Seed: [dim]{components['seed']}[/dim]")
        console.print(f"  📈 Learning Rate: [bright_yellow]{components['lr']}[/bright_yellow]")

    # Demonstrate comprehensive run analysis
    console.print(f"\n[bold blue]Testing Comprehensive Run Data Extraction:[/bold blue]")

    # Test with the example format (won't find in actual data, but shows parsing)
    test_run_id = "250904-222144_test_match_150M_finetune_10Mtx1_DD-d17-300M-17500-2_lr=5e-05"
    console.print(f"\n[bold yellow]Testing with example format:[/bold yellow]")
    console.print(f"Run ID: [cyan]{test_run_id}[/cyan]")

    # Parse components and show what pretrained data we'd look for
    test_components = parse_run_id_components(test_run_id)

    # Show mappings
    comparison_logical = test_components['comparison_model_recipe']
    comparison_mapped = COMPARISON_RECIPE_MAPPING.get(comparison_logical, comparison_logical)

    checkpoint_code = test_components['initial_checkpoint_recipe']
    checkpoint_mapped = CHECKPOINT_RECIPE_MAPPING.get(checkpoint_code, checkpoint_code)

    console.print(f"Would look for comparison model: [blue]{test_components['comparison_model_size']} ({comparison_logical} → {comparison_mapped})[/blue]")
    console.print(f"Would look for initial checkpoint: [red]{test_components['initial_checkpoint_size']} ({checkpoint_code} → {checkpoint_mapped}) @ step {test_components['initial_checkpoint_steps']}[/red]")

    # Test with actual run from dataset
    if finetune_runs:
        actual_run = finetune_runs[0]
        console.print(f"\n[bold yellow]Testing with actual run from dataset:[/bold yellow]")
        run_data = get_run_data_comprehensive(actual_run, runs_df, history_df, pretrain_df)
        display_comprehensive_run_analysis(run_data)

    # Test pretrained model data extraction directly
    console.print(f"\n[bold blue]Testing Direct Pretrained Model Data Extraction:[/bold blue]")

    # Show available model sizes and recipes
    available_sizes = pretrain_df['params'].unique()[:5]
    available_recipes = pretrain_df['data'].unique()
    console.print(f"Available model sizes (first 5): [cyan]{list(available_sizes)}[/cyan]")
    console.print(f"Available recipes: [magenta]{list(available_recipes)}[/magenta]")

    # Test extraction with actual data
    if len(available_sizes) > 0 and len(available_recipes) > 0:
        test_size = available_sizes[0]
        test_recipe = available_recipes[0]
        console.print(f"\nTesting extraction for [cyan]{test_size}[/cyan] + [magenta]{test_recipe}[/magenta]:")

        sample_data = get_pretrained_model_data(pretrain_df, test_size, test_recipe)
        if not sample_data.empty:
            console.print(f"Found {len(sample_data)} rows")

            # Test with specific step
            available_steps = sorted(sample_data['step'].unique())
            if available_steps:
                test_step = available_steps[len(available_steps)//2]  # Pick middle step
                console.print(f"Testing with specific step {test_step}:")
                step_data = get_pretrained_model_data(pretrain_df, test_size, test_recipe, test_step)
                console.print(f"Found {len(step_data)} rows for specific step")

    # Analyze mismatches between run_ids and pretrained data
    console.print(f"\n[bold red]🔍 Analyzing Model/Recipe Mismatches:[/bold red]")

    # Get all unique model size + recipe combos from run_ids (using mapped values)
    run_combos = set()
    run_combos_raw = set()  # Keep track of raw values too
    for run_id in finetune_runs[:20]:  # Sample first 20 for analysis
        components = parse_run_id_components(run_id)
        comp_size = components.get('comparison_model_size', '')
        comp_recipe_raw = components.get('comparison_model_recipe', '')
        comp_recipe_mapped = COMPARISON_RECIPE_MAPPING.get(comp_recipe_raw, comp_recipe_raw)

        if comp_size and comp_recipe_mapped:
            run_combos.add((comp_size, comp_recipe_mapped))
            run_combos_raw.add((comp_size, comp_recipe_raw))

    # Get available combos from pretraining data
    pretrain_combos = set(zip(pretrain_df['params'], pretrain_df['data']))

    console.print(f"Found {len(run_combos)} unique (model_size, recipe) combos in run_ids (mapped)")
    console.print(f"Found {len(run_combos_raw)} unique combos in run_ids (raw)")
    console.print(f"Found {len(pretrain_combos)} unique combos in pretraining data")

    # Show mapping effect
    console.print(f"\n[bold cyan]🔄 Recipe Mapping Examples:[/bold cyan]")
    for size, recipe_raw in sorted(run_combos_raw):
        recipe_mapped = COMPARISON_RECIPE_MAPPING.get(recipe_raw, recipe_raw)
        if recipe_raw != recipe_mapped:
            console.print(f"  • [cyan]{size}[/cyan] + [dim]{recipe_raw}[/dim] → [magenta]{recipe_mapped}[/magenta]")

    # Find mismatches (after mapping)
    missing_combos = run_combos - pretrain_combos
    if missing_combos:
        console.print(f"\n[bold yellow]❌ Run combos NOT found in pretraining data (after mapping - {len(missing_combos)}):[/bold yellow]")
        for size, recipe in sorted(missing_combos):
            console.print(f"  • [cyan]{size}[/cyan] + [magenta]{recipe}[/magenta]")

            # Check what's available for that size
            size_matches = pretrain_df[pretrain_df['params'] == size]['data'].unique()
            if len(size_matches) > 0:
                console.print(f"    Available recipes for {size}: [dim]{list(size_matches)[:5]}{'...' if len(size_matches) > 5 else ''}[/dim]")
            else:
                console.print(f"    [red]No data found for model size {size} at all![/red]")

    # Find matching combos for comparison
    matching_combos = run_combos & pretrain_combos
    if matching_combos:
        console.print(f"\n[bold green]✅ Run combos that DO match pretraining data ({len(matching_combos)}):[/bold green]")
        for size, recipe in sorted(matching_combos):
            console.print(f"  • [cyan]{size}[/cyan] + [magenta]{recipe}[/magenta]")

    # Show example specific run analysis
    if missing_combos:
        console.print(f"\n[bold blue]🔬 Example Analysis of Non-Matching Runs:[/bold blue]")

        # Find a run with a missing combo
        for run_id in finetune_runs[:10]:
            components = parse_run_id_components(run_id)
            comp_size = components.get('comparison_model_size', '')
            comp_recipe_raw = components.get('comparison_model_recipe', '')
            comp_recipe_mapped = COMPARISON_RECIPE_MAPPING.get(comp_recipe_raw, comp_recipe_raw)

            if (comp_size, comp_recipe_mapped) in missing_combos:
                console.print(f"\nRun: [cyan]{run_id}[/cyan]")
                console.print(f"Raw: [dim]{comp_size} + {comp_recipe_raw}[/dim]")
                console.print(f"Mapped: [yellow]{comp_size} + {comp_recipe_mapped}[/yellow]")

                # Test the actual extraction
                result = get_pretrained_model_data(pretrain_df, comp_size, comp_recipe_mapped)
                console.print(f"Result: [{'red' if result.empty else 'green'}]{len(result)} rows found[/{'red' if result.empty else 'green'}]")
                break

    # Test if the mapping fixed our original issues
    console.print(f"\n[bold green]✅ Testing Original Problem Cases:[/bold green]")

    test_cases = [("150M", "Dolma 1.7"), ("60M", "Dolma 1.7")]
    for size, recipe_raw in test_cases:
        recipe_mapped = COMPARISON_RECIPE_MAPPING.get(recipe_raw, recipe_raw)
        console.print(f"Testing [cyan]{size}[/cyan] + [dim]{recipe_raw}[/dim] → [magenta]{recipe_mapped}[/magenta]:")

        result = get_pretrained_model_data(pretrain_df, size, recipe_mapped)
        status = f"[{'green' if not result.empty else 'red'}]{'✓' if not result.empty else '✗'}[/{'green' if not result.empty else 'red'}]"
        console.print(f"  {status} {len(result)} rows found")
        if not result.empty:
            metrics = result['metric'].nunique()
            steps = result['step'].nunique()
            console.print(f"    [dim]{metrics} unique metrics, {steps} unique steps[/dim]")

    console.print(f"\n[bold blue]Testing Comprehensive Data Extraction and Pickling:[/bold blue]")
    filepath = extract_and_pickle_matched_run_data(runs_df, history_df, pretrain_df, pattern="finetune")
    console.print(f"Results saved to: [cyan]{filepath}[/cyan]")

# %%
main()

# %%
if __name__ == "__main__":
    main()