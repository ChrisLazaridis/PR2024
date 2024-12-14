import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def save_plot(fig, output_path) -> None:
    """
    Save plot to file, create or replace.
    :param fig: the figure to save
    :param output_path: the file path to save the figure
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist
    # if a plot with the same name exists, delete it
    if os.path.exists(output_path):
        os.remove(output_path)
    fig.savefig(output_path)  # Save the plo
    plt.close(fig)  # Close the figure


# noinspection PyShadowingNames
def create_relationship_plots(data_path, output_dir) -> None:
    """
    Create and save scatterplots for numeric relationships with improved visibility.
    :param data_path: the file path to the CSV dataset
    :param output_dir: the directory to save the scatterplots
    """
    df = pd.read_csv(data_path)

    # Pairs for plotting
    pairs = [("longitude", "latitude"),
             ("housing_median_age", "median_house_value"),
             ("median_income", "median_house_value"),
             ("total_rooms", "total_bedrooms"),
             ("population", "households"),
             ("longitude", "latitude", "median_income"),
             ("longitude", "latitude", "housing_median_age", "median_house_value")]

    for pair in pairs:
        if len(pair) == 2:
            # 2D scatterplot
            x, y = pair
            fig = plt.figure(figsize=(12, 8))
            sns.scatterplot(data=df, x=x, y=y, alpha=0.6, s=30)
            plt.title(f"{x} vs {y}", fontsize=14)
            plt.xlabel(x, fontsize=12)
            plt.ylabel(y, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            save_plot(fig, f"{output_dir}/{x}_{y}_scatterplot.png")
        elif len(pair) == 3:
            # 3-variable scatterplot with hue
            x, y, hue = pair
            fig = plt.figure(figsize=(12, 8))
            sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6, s=50, palette="viridis")
            plt.title(f"{x} vs {y} (hue: {hue})", fontsize=14)
            plt.xlabel(x, fontsize=12)
            plt.ylabel(y, fontsize=12)
            plt.legend(title=hue, fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            save_plot(fig, f"{output_dir}/{x}_{y}_{hue}_scatterplot.png")
        elif len(pair) == 4:
            # 4-variable scatterplot with hue and size
            x, y, hue, size = pair
            fig = plt.figure(figsize=(14, 10))
            sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, alpha=0.6, sizes=(20, 200), palette="coolwarm")
            plt.title(f"{x} vs {y} (hue: {hue}, size: {size})", fontsize=14)
            plt.xlabel(x, fontsize=12)
            plt.ylabel(y, fontsize=12)
            plt.legend(title=f"{hue} and {size}", fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            save_plot(fig, f"{output_dir}/{x}_{y}_{hue}_{size}_scatterplot.png")


data_path = "C:/Users/claza/PycharmProjects/PR2024/housing/housing.csv"
output_dir = "C:/Users/claza/PycharmProjects/PR2024/housing/relations"

create_relationship_plots(data_path, output_dir)
