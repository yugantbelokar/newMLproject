"""Explore the student performance dataset and save summary plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "dataset.csv"
PLOTS_DIR = PROJECT_DIR / "eda_plots"

FEATURE_COLUMNS = [
    "attendance",
    "internal_marks",
    "study_hours",
    "previous_performance",
    "assignment_score",
]
TARGET_COLUMN = "result"
RESULT_ORDER = ["Pass", "Fail"]
FEATURE_LABELS = {
    "attendance": "Attendance",
    "internal_marks": "Internal Marks",
    "study_hours": "Study Hours",
    "previous_performance": "Previous Performance",
    "assignment_score": "Assignment Score",
}


def save_plot(filename):
    """Save the current plot in the EDA output directory."""
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(DATASET_PATH)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    PLOTS_DIR.mkdir(exist_ok=True)

    print("=" * 40)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 40)
    print("\nDataset Shape:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumns:")
    print(", ".join(df.columns))
    print("\nData Types:")
    print(df.dtypes.to_string())
    print("\nFirst Few Records:")
    print(df.head().to_string(index=False))
    print("\nMissing Values:")
    print(df.isnull().sum().to_string())
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    print("\nStatistical Summary:")
    print(df.describe().to_string())

    sns.set_theme(style="whitegrid")

    for column in FEATURE_COLUMNS:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=column, bins=15, color="#4C72B0")
        plt.title(f"{FEATURE_LABELS[column]} Distribution")
        plt.xlabel(FEATURE_LABELS[column])
        plt.ylabel("Number of Students")
        save_plot(f"{column}_distribution.png")

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x=TARGET_COLUMN, order=RESULT_ORDER, color="#4C72B0")
    ax.bar_label(ax.containers[0])
    plt.title("Student Result Distribution")
    plt.xlabel("Result")
    plt.ylabel("Number of Students")
    save_plot("target_distribution.png")

    for column in FEATURE_COLUMNS:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=TARGET_COLUMN, y=column, order=RESULT_ORDER)
        plt.title(f"{FEATURE_LABELS[column]} by Result")
        plt.xlabel("Result")
        plt.ylabel(FEATURE_LABELS[column])
        save_plot(f"{column}_vs_result.png")

    correlation = df[FEATURE_COLUMNS].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
    )
    plt.title("Correlation Between Numerical Features")
    save_plot("correlation_heatmap.png")

    print("\nEDA completed successfully.")
    print(f"Plots saved in: {PLOTS_DIR.name}/")


if __name__ == "__main__":
    main()
