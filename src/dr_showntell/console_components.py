from typing import Optional

import pandas as pd
from rich.align import Align
from rich.box import DOUBLE_EDGE
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.panel import Panel
from rich.rule import Rule

from datadec.fancy_table import FancyTable, HeaderCell


class TitlePanel:
    def __init__(self, title: str):
        self.title = title

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        panel_content = Align.center(f"[bold]   ~ {self.title} ~   [/bold]")
        yield ""
        yield Panel(
            panel_content,
            expand=True,
            border_style="bold blue",
            box=DOUBLE_EDGE,
        )
        yield ""


class SectionRule:
    def __init__(self, title: str, style: str = "bold blue"):
        self.title = title.title()
        self.style = style
        if "blue" in style:
            self.title_style = "bold white on blue"
        elif "yellow" in style:
            self.title_style = "bold black on yellow"
        elif "green" in style:
            self.title_style = "bold white on green"
        elif "red" in style:
            self.title_style = "bold white on red"
        else:
            self.title_style = (
                f"bold white on {style.split()[-1] if ' ' in style else style}"
            )

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield ""
        yield Rule(
            f"[{self.title_style}] {self.title} [/{self.title_style}]", style=self.style
        )
        yield ""


class SectionTitlePanel:
    def __init__(self, label_text: str, *content):
        self.label_text = label_text
        self.content = content

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        panel_items = [f"    {self.label_text}    "]
        if self.content:
            panel_items.extend(["", *self.content])
        panel_content = Group(*panel_items)
        yield ""
        yield ""
        yield Panel.fit(panel_content, box=DOUBLE_EDGE)


class InfoBlock:
    def __init__(self, text: str, style: Optional[str] = None):
        self.text = text
        self.style = style

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if self.style:
            yield f"[{self.style}]{self.text}[/{self.style}]"
        else:
            yield self.text


def create_hyperparameter_sweep_table(
    data: pd.DataFrame,
    fixed_section: dict,
    swept_section: dict,
    optimization: str = "max",
    best_performance: Optional[dict] = None,
    highlight_threshold: float = 0.02,
) -> tuple[FancyTable, InfoBlock]:
    table = FancyTable(show_header=True, header_style="bold magenta")
    for col in data.columns:
        table.add_column()
    if best_performance and best_performance.get("enabled", True):
        table.add_column()
        table.add_column()
    header_row_1 = []
    fixed_cols = fixed_section["columns"]
    if fixed_cols:
        header_row_1.append(
            table.create_spanned_header(
                fixed_section["title"], len(fixed_cols), "bold blue"
            )
        )
    swept_cols = swept_section["columns"]
    if swept_cols:
        header_row_1.append(
            table.create_spanned_header(
                swept_section["title"], len(swept_cols), "bold green"
            )
        )
    if best_performance and best_performance.get("enabled", True):
        header_row_1.append(
            table.create_spanned_header(
                best_performance.get("title", "Best Perf"), 2, "bold green"
            )
        )
    table.add_header_row(*header_row_1)
    header_row_2 = []
    for col in fixed_cols:
        display_name = col.replace("_", " ").title()
        header_row_2.append(HeaderCell(display_name))
    transform_fn = swept_section.get("display_transform")
    for col in swept_cols:
        if transform_fn:
            display_name = transform_fn(col)
        else:
            display_name = col.replace("_", " ").title()
        header_row_2.append(HeaderCell(display_name))
    if best_performance and best_performance.get("enabled", True):
        col_names = best_performance.get("column_names", ["Param", "Value"])
        for name in col_names:
            header_row_2.append(HeaderCell(name))
    table.add_header_row(*header_row_2)
    for _, row in data.iterrows():
        best_col = None
        best_value = None
        if swept_cols:
            if optimization == "min":
                best_col = min(
                    swept_cols,
                    key=lambda x: float(row[x]) if pd.notna(row[x]) else float("inf"),
                )
            else:
                best_col = max(
                    swept_cols, key=lambda x: float(row[x]) if pd.notna(row[x]) else -1
                )
            best_value = float(row[best_col]) if pd.notna(row[best_col]) else None
        row_data = []
        all_cols = fixed_cols + swept_cols
        for col in all_cols:
            value = str(row.get(col, ""))
            if col in fixed_cols:
                value = f"[bold cyan]{value}[/bold cyan]"
            elif col in swept_cols and best_value is not None and pd.notna(row[col]):
                current_value = float(row[col])
                if optimization == "min":
                    threshold_value = best_value * (1 + highlight_threshold)
                    should_highlight = current_value <= threshold_value
                else:
                    threshold_value = best_value * (1 - highlight_threshold)
                    should_highlight = current_value >= threshold_value
                if should_highlight:
                    value = f"[bold]{value}[/bold]"
            row_data.append(value)
        if best_performance and best_performance.get("enabled", True) and best_col:
            transform_fn = swept_section.get("display_transform")
            if transform_fn:
                best_param_display = transform_fn(best_col)
            else:
                best_param_display = best_col
            row_data.append(f"[green]{best_param_display}[/green]")
            row_data.append(f"[green]{best_value}[/green]")
        elif best_performance and best_performance.get("enabled", True):
            row_data.extend(["", ""])
        table.add_row(*row_data)
    threshold_pct = int(highlight_threshold * 100)
    direction = "best" if optimization == "max" else "lowest"
    info_text = (
        f"Values within {threshold_pct}% of {direction} performance are highlighted"
    )
    info_block = InfoBlock(info_text, style="dim")
    return table, info_block


def create_counts_table(
    crosstab_df: pd.DataFrame,
    row_section_title: str,
    col_section_title: str,
    present_row_name: str = "Present",
    present_col_name: str = "Present",
    col_display_transform: Optional[callable] = None,
) -> FancyTable:
    data_columns = [col for col in crosstab_df.columns if col != present_col_name]
    table_data = []
    for row_name in crosstab_df.index:
        if row_name != present_row_name:
            row_data = {"row_label": str(row_name)}
            for col in data_columns:
                row_data[col] = int(crosstab_df.loc[row_name, col])
            row_data["present_total"] = int(crosstab_df.loc[row_name, present_col_name])
            table_data.append(row_data)
    present_row = {"row_label": present_row_name}
    for col in data_columns:
        present_row[col] = int(crosstab_df.loc[present_row_name, col])
    present_row["present_total"] = int(
        crosstab_df.loc[present_row_name, present_col_name]
    )
    table_data.append(present_row)
    table_df = pd.DataFrame(table_data)
    table = FancyTable(
        show_header=True, header_style="bold magenta", row_styles=["", "dim"]
    )
    for col in table_df.columns:
        table.add_column()
    header_row_1 = [
        table.create_spanned_header("", 1, ""),  # Empty for row label
        table.create_spanned_header(col_section_title, len(data_columns), "bold green"),
        table.create_spanned_header("", 1, ""),  # Empty for Present
    ]
    table.add_header_row(*header_row_1)
    header_row_2 = [HeaderCell(row_section_title)]
    for col in data_columns:
        if col_display_transform:
            display_name = col_display_transform(col)
        elif isinstance(col, (int, float)):
            display_name = f"{float(col):.2e}"
        else:
            display_name = str(col)
        header_row_2.append(HeaderCell(display_name))
    header_row_2.append(HeaderCell(present_col_name))
    table.add_header_row(*header_row_2)
    for _, row in table_df.iterrows():
        row_data = []
        row_label_value = str(row["row_label"])
        if present_row_name in row_label_value:
            row_data.append(f"[bold green]{row_label_value}[/bold green]")
        else:
            row_data.append(f"[bold cyan]{row_label_value}[/bold cyan]")
        for col in data_columns:
            value = str(row[col])
            if present_row_name in row_label_value:
                row_data.append(f"[bold green]{value}[/bold green]")
            else:
                row_data.append(value)
        present_value = str(row["present_total"])
        row_data.append(f"[bold green]{present_value}[/bold green]")
        table.add_row(*row_data)
    return table
