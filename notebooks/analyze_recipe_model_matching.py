from __future__ import annotations

import pandas as pd
from rich.console import Console

from dr_showntell.datadec_utils import load_data, filter_by_model_size, filter_by_recipe
from dr_showntell.run_id_parsing import parse_and_group_run_ids, convert_groups_to_dataframes, apply_processing, RECIPE_MAPPING
from dr_showntell.fancy_table import FancyTable

console = Console()


def analyze_recipe_model_matching() -> None:
    console.print(f"[bold blue]🔍 Recipe-Model Size Matching Analysis[/bold blue]")

    pretrain_df, runs_df, history_df = load_data()

    grouped_data = parse_and_group_run_ids(runs_df)
    type_dataframes = convert_groups_to_dataframes(grouped_data)
    processed_dataframes = apply_processing(type_dataframes, runs_df=runs_df, history_df=history_df)

    available_pretrain_combos = set(zip(pretrain_df['params'], pretrain_df['data']))
    console.print(f"Available pretraining combinations: [cyan]{len(available_pretrain_combos):,}[/cyan]")

    run_combinations = set()
    run_data_by_combo = {}

    for run_type, df in processed_dataframes.items():
        if df.empty:
            continue

        for _, row in df.iterrows():
            comparison_size = row.get('comparison_model_size')
            comparison_recipe_raw = row.get('comparison_model_recipe')

            if pd.notna(comparison_size) and pd.notna(comparison_recipe_raw):
                comparison_recipe_mapped = RECIPE_MAPPING.get(comparison_recipe_raw, comparison_recipe_raw)

                combo = (comparison_size, comparison_recipe_mapped)
                run_combinations.add(combo)

                if combo not in run_data_by_combo:
                    run_data_by_combo[combo] = []

                run_data_by_combo[combo].append({
                    'run_id': row['run_id'],
                    'run_type': run_type,
                    'raw_recipe': comparison_recipe_raw,
                    'mapped_recipe': comparison_recipe_mapped
                })

    console.print(f"Unique run combinations: [yellow]{len(run_combinations):,}[/yellow]")

    matching_combos = run_combinations & available_pretrain_combos
    missing_combos = run_combinations - available_pretrain_combos

    console.print(f"\n[bold green]✅ Matching combinations: {len(matching_combos)} / {len(run_combinations)} ({len(matching_combos)/len(run_combinations)*100:.1f}%)[/bold green]")
    console.print(f"[bold red]❌ Missing combinations: {len(missing_combos)} / {len(run_combinations)} ({len(missing_combos)/len(run_combinations)*100:.1f}%)[/bold red]")

    if matching_combos:
        console.print(f"\n[bold green]✅ Successfully Matched Combinations:[/bold green]")

        match_table = FancyTable(
            title="Matching (Model Size, Recipe) Pairs",
            show_header=True,
            header_style="bold green"
        )

        match_table.add_column("Model Size", style="cyan")
        match_table.add_column("Recipe", style="yellow")
        match_table.add_column("Runs Using This", justify="right", style="magenta")
        match_table.add_column("Pretrain Records", justify="right", style="blue")

        for size, recipe in sorted(matching_combos):
            run_count = len(run_data_by_combo[(size, recipe)])

            pretrain_matches = filter_by_model_size(pretrain_df, size)
            pretrain_matches = filter_by_recipe(pretrain_matches, recipe)
            pretrain_count = len(pretrain_matches)

            match_table.add_row(
                size,
                recipe,
                f"{run_count:,}",
                f"{pretrain_count:,}"
            )

        console.print(match_table)

    if missing_combos:
        console.print(f"\n[bold red]❌ Missing Combinations (No Pretraining Data):[/bold red]")

        missing_table = FancyTable(
            title="Missing (Model Size, Recipe) Pairs",
            show_header=True,
            header_style="bold red"
        )

        missing_table.add_column("Model Size", style="cyan")
        missing_table.add_column("Recipe", style="yellow")
        missing_table.add_column("Runs Affected", justify="right", style="magenta")
        missing_table.add_column("Available Recipes for Size", style="dim")

        for size, recipe in sorted(missing_combos):
            run_count = len(run_data_by_combo[(size, recipe)])

            size_matches = filter_by_model_size(pretrain_df, size)
            available_recipes = sorted(size_matches['data'].unique()) if not size_matches.empty else []
            available_str = ", ".join(available_recipes[:3])
            if len(available_recipes) > 3:
                available_str += f" + {len(available_recipes) - 3} more"

            missing_table.add_row(
                size,
                recipe,
                f"{run_count:,}",
                available_str if available_recipes else "No data for this size"
            )

        console.print(missing_table)

    console.print(f"\n[bold blue]📊 Recipe Mapping Analysis:[/bold blue]")

    mapping_applied = {}
    for combo, runs in run_data_by_combo.items():
        for run_info in runs:
            raw = run_info['raw_recipe']
            mapped = run_info['mapped_recipe']
            if raw != mapped:
                if (raw, mapped) not in mapping_applied:
                    mapping_applied[(raw, mapped)] = 0
                mapping_applied[(raw, mapped)] += 1

    if mapping_applied:
        mapping_table = FancyTable(
            title="Recipe Mapping Applications",
            show_header=True,
            header_style="bold cyan"
        )

        mapping_table.add_column("Raw Recipe", style="dim")
        mapping_table.add_column("Mapped Recipe", style="yellow")
        mapping_table.add_column("Runs Affected", justify="right", style="magenta")

        for (raw, mapped), count in sorted(mapping_applied.items(), key=lambda x: x[1], reverse=True):
            mapping_table.add_row(raw, mapped, f"{count:,}")

        console.print(mapping_table)
    else:
        console.print("[dim]No recipe mapping was applied (all recipes were already in canonical form)[/dim]")

    runs_without_comparison_data = 0
    for run_type, df in processed_dataframes.items():
        for _, row in df.iterrows():
            comparison_size = row.get('comparison_model_size')
            comparison_recipe = row.get('comparison_model_recipe')

            if pd.isna(comparison_size) or pd.isna(comparison_recipe):
                runs_without_comparison_data += 1

    total_runs = sum(len(df) for df in processed_dataframes.values())
    runs_with_comparison = total_runs - runs_without_comparison_data

    console.print(f"\n[bold yellow]📈 Overall Summary:[/bold yellow]")
    console.print(f"Total processed runs: [cyan]{total_runs:,}[/cyan]")
    console.print(f"Runs with comparison model info: [green]{runs_with_comparison:,}[/green] ({runs_with_comparison/total_runs*100:.1f}%)")
    console.print(f"Runs missing comparison info: [red]{runs_without_comparison_data:,}[/red] ({runs_without_comparison_data/total_runs*100:.1f}%)")

    if runs_with_comparison > 0:
        matching_runs = sum(len(run_data_by_combo[combo]) for combo in matching_combos)
        missing_runs = sum(len(run_data_by_combo[combo]) for combo in missing_combos)

        console.print(f"\nOf runs with comparison model info:")
        console.print(f"  • Can match to pretraining data: [green]{matching_runs:,}[/green] ({matching_runs/runs_with_comparison*100:.1f}%)")
        console.print(f"  • Cannot match (missing pretraining): [red]{missing_runs:,}[/red] ({missing_runs/runs_with_comparison*100:.1f}%)")


if __name__ == "__main__":
    analyze_recipe_model_matching()