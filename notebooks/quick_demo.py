# %%
from __future__ import annotations

import pandas as pd
from rich.console import Console
from rich.text import Text

from dr_showntell.fancy_table import FancyTable

console = Console()

def load_data():
    pretrain_df = pd.read_parquet("/Users/daniellerothermel/drotherm/repos/datadec/data/datadecide/mean_eval_melted.parquet")
    runs_df = pd.read_parquet("/Users/daniellerothermel/drotherm/repos/dr_wandb/data/runs_metadata.parquet")
    history_df = pd.read_parquet("/Users/daniellerothermel/drotherm/repos/dr_wandb/data/runs_history.parquet")
    return pretrain_df, runs_df, history_df


def main():
    pretrain_df, runs_df, history_df = load_data()
    print("Pretrain DF:")
    print(pretrain_df.head())
    print()
    print("Runs DF:")
    print(runs_df.head())
    print()

# %%
main()

# %%
if __name__ == "__main__":
    main()