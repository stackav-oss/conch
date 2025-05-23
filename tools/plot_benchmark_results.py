# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Plot benchmarking results."""

from pathlib import Path

import click
import pandas as pd  # type: ignore[import-untyped]
from matplotlib import pyplot as plt


@click.command()
@click.option(
    "--results-directory",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=True, file_okay=False),
    help="Path to the benchmark results directory.",
)
@click.option(
    "--x-axis",
    required=True,
    type=str,
    help="Name of the column to use for the x-axis.",
)
@click.option(
    "--y-axis",
    required=True,
    type=str,
    default="runtime_ms",
    help="Name of the column to use for the y-axis.",
)
@click.option(
    "--triton-tag",
    required=True,
    type=str,
    default="Triton",
    help="The 'tag' column value for Triton results.",
)
@click.option(
    "--baseline-tag",
    required=True,
    type=str,
    default="Baseline",
    help="The 'tag' column value for baseline results.",
)
def main(results_directory: Path, x_axis: str, y_axis: str, triton_tag: str, baseline_tag: str) -> None:
    """Main function to plot benchmarking results."""
    dataframes = [pd.read_csv(file) for file in results_directory.iterdir() if file.suffix == ".csv"]
    if not dataframes:
        print(f"No CSV files found in the specified directory ({results_directory = }).")
        return

    if not all(all(df.columns == dataframes[0].columns) for df in dataframes):
        print("CSV files have different columns. Please ensure they are consistent.")
        return

    df = pd.concat(dataframes, ignore_index=True)

    triton_df = df[df["tag"] == triton_tag].sort_values(by=x_axis)
    baseline_df = df[df["tag"] == baseline_tag].sort_values(by=x_axis)

    columns_to_exclude = ["tag", "platform", x_axis, y_axis]
    metadata_columns = [col for col in triton_df.columns if col not in columns_to_exclude]

    triton_metadata = triton_df[metadata_columns].set_index(metadata_columns[0]).reset_index(drop=True)
    baseline_metadata = baseline_df[metadata_columns].set_index(metadata_columns[0]).reset_index(drop=True)

    if not all(triton_metadata == baseline_metadata):
        print("Metadata does not match between Triton and baseline results.")
        return

    metadata = triton_df[metadata_columns].iloc[0].to_dict()

    # Plot the results
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.suptitle(f"Benchmarking Results: {results_directory.name}")
    plt.title(f"Metadata: {metadata}")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    ax.plot(
        triton_df[x_axis],
        triton_df[y_axis],
        label=f"{triton_tag} (platform={triton_df['platform'].iloc[0]}, n={len(triton_df)})",
        color="blue",
    )

    ax.plot(
        baseline_df[x_axis],
        baseline_df[y_axis],
        label=f"{baseline_tag} (platform={baseline_df['platform'].iloc[0]}, n={len(baseline_df)})",
        color="red",
    )

    box = ax.get_position()
    ax.set_position((box.x0, box.y0, box.width * 0.85, box.height))
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    output_filename = results_directory / "plot.png"
    plt.savefig(output_filename)
    print(f"Plot saved to '{output_filename}'")


if __name__ == "__main__":
    main()
