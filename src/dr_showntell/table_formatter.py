from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rich.table import Table
from tabulate import tabulate

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
TABLE_FORMATS_DIR = CONFIGS_DIR / "table_formats"

OUTPUT_FORMATS = {
    "console": "grid",
    "markdown": "pipe",
    "latex": "latex",
    "plain": "plain",
    "csv": "simple",
}

FORMATTER_TYPES: dict[str, Callable] = {
    "scientific": lambda x, precision=2: f"{x:.{precision}e}"
    if x is not None
    else "None",
    "decimal": lambda x, precision=3: f"{x:.{precision}f}" if x is not None else "None",
    "integer": lambda x: f"{x:,.0f}" if x is not None else "None",
    "comma": lambda x: f"{x:,}" if x is not None else "None",
    "truncate": lambda x, max_length=50: (
        str(x)[:max_length] + "..."
        if x is not None and len(str(x)) > max_length
        else (str(x) if x is not None else "None")
    ),
    "string": lambda x: str(x) if x is not None else "None",
}


def load_table_config(config_name: str) -> dict[str, dict]:
    config_file = TABLE_FORMATS_DIR / f"{config_name}.yaml"
    if config_file.exists():
        with config_file.open() as f:
            return yaml.safe_load(f)
    else:
        return {}


# Built-in config for coverage table (simple enough to keep inline)
COVERAGE_TABLE_CONFIG = {
    "index": {"header": "#", "formatter": "integer"},
    "column": {"header": "Column", "formatter": "truncate", "max_length": 35},
    "coverage": {"header": "Coverage %", "formatter": "decimal", "precision": 1},
}


def format_table(
    data: list[dict] | pd.DataFrame | list[list],
    headers: list[str] | None = None,
    output_format: str = "console",
    column_config: dict[str, dict] | None = None,
    title: str | None = None,
    table_style: str = "lines",
    disable_numparse: bool = True,
) -> str | Table:
    processed_data = _preprocess_data(data)
    column_names = _get_column_names(data)
    config = column_config or {}
    formatted_data = _apply_column_formatting(processed_data, config, column_names)
    final_headers = _resolve_headers(headers, column_names, config)

    if output_format == "console":
        return _create_rich_table(
            formatted_data, final_headers, config, column_names, title, table_style
        )
    else:
        tablefmt = OUTPUT_FORMATS.get(output_format, "grid")
        return tabulate(
            formatted_data,
            headers=final_headers,
            tablefmt=tablefmt,
            disable_numparse=disable_numparse,
        )


def _preprocess_data(data: list[dict] | pd.DataFrame | list[list]) -> list[list]:
    if isinstance(data, pd.DataFrame):
        return data.to_numpy().tolist()
    elif isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict):
            if len(data) == 0:
                return []
            keys = list(data[0].keys())
            return [[row.get(key) for key in keys] for row in data]
        else:
            return data
    return []


def _get_column_names(data: list[dict] | pd.DataFrame | list[list]) -> list[str]:
    if isinstance(data, pd.DataFrame):
        return list(data.columns)
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return list(data[0].keys())
    else:
        return []


def _apply_column_formatting(
    processed_data: list[list],
    config: dict[str, dict],
    column_names: list[str] | None = None,
) -> list[list]:
    if not processed_data:
        return processed_data
    formatted_data = []
    for row in processed_data:
        formatted_row = []
        for col_idx, value in enumerate(row):
            col_name = (
                column_names[col_idx]
                if column_names and col_idx < len(column_names)
                else None
            )
            formatted_row.append(_format_value(value, col_name, config))
        formatted_data.append(formatted_row)
    return formatted_data


def _format_value(
    value: Any,
    col_name: str | None,
    config: dict[str, dict],
    formatter_types: dict[str, Callable],
) -> str:
    if col_name and col_name in config:
        col_config = config[col_name]
        formatter_name = col_config.get("formatter", "string")
        formatter = formatter_types.get(formatter_name, formatter_types["string"])
        formatter_kwargs = {
            k: v for k, v in col_config.items() if k not in ["header", "formatter"]
        }
        try:
            return formatter(value, **formatter_kwargs)
        except (TypeError, ValueError):
            return str(value) if value is not None else "None"
    return str(value) if value is not None else "None"


def _resolve_headers(
    headers: list[str] | None, column_names: list[str], config: dict[str, dict]
) -> list[str]:
    if headers is not None:
        return headers
    if config and column_names:
        result_headers = []
        for col_name in column_names:
            if col_name in config:
                result_headers.append(config[col_name].get("header", col_name))
            else:
                result_headers.append(col_name)
        return result_headers
    if column_names:
        return column_names
    return []


def format_dynamics_table(
    dynamics_list: list[dict],
    columns: list[str] | None = None,
    output_format: str = "console",
    table_style: str = "lines",
    disable_numparse: bool = True,
) -> str:
    if not dynamics_list:
        return "No data to display"
    if columns:
        filtered_data = []
        for dynamics in dynamics_list:
            filtered_row = {col: dynamics.get(col) for col in columns}
            filtered_data.append(filtered_row)
        data_to_format = filtered_data
    else:
        data_to_format = dynamics_list
    wandb_config = load_table_config("wandb_analysis")
    return format_table(
        data=data_to_format,
        output_format=output_format,
        column_config=wandb_config,
        table_style=table_style,
        disable_numparse=disable_numparse,
    )


def format_coverage_table(
    df: pd.DataFrame,
    title: str = "Column Coverage",
    output_format: str = "console",
    table_style: str = "lines",
    disable_numparse: bool = True,
) -> str:
    coverage_data = []
    for i, col in enumerate(df.columns):
        coverage = df[col].notna().sum() / len(df) * 100
        coverage_data.append({"index": i + 1, "column": col, "coverage": coverage})
    result = f"{title} ({len(df.columns)} columns):\n"
    result += format_table(
        data=coverage_data,
        output_format=output_format,
        column_config=COVERAGE_TABLE_CONFIG,
        table_style=table_style,
        disable_numparse=disable_numparse,
    )
    return result


def _create_rich_table(
    formatted_data: list[list],
    headers: list[str],
    config: dict[str, dict],
    column_names: list[str],
    title: str | None = None,
    table_style: str = "lines",
) -> Table:
    if table_style == "zebra":
        table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta",
            row_styles=["", "dim"],
        )
    else:  # "lines" or default
        table = Table(
            title=title, show_header=True, header_style="bold magenta", show_lines=True
        )

    for i, header in enumerate(headers):
        col_name = _get_column_name_for_index(column_names, i)
        col_config = config.get(col_name, {})
        justify = _get_rich_justify(col_config)
        style = _get_rich_style(col_config)
        table.add_column(header, justify=justify, style=style)

    for row in formatted_data:
        table.add_row(*[str(cell) for cell in row])

    return table


def _get_column_name_for_index(column_names: list[str], index: int) -> str:
    return column_names[index] if index < len(column_names) else f"col_{index}"


def _get_rich_justify(col_config: dict) -> str:
    formatter = col_config.get("formatter", "string")
    return (
        "right"
        if formatter in ["scientific", "decimal", "integer", "comma"]
        else "left"
    )


def _get_rich_style(col_config: dict) -> str | None:
    formatter = col_config.get("formatter", "string")
    style_map = {
        "scientific": "yellow",
        "decimal": "green",
        "integer": "cyan",
        "comma": "cyan",
        "truncate": "dim",
    }
    return style_map.get(formatter)
