import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_lists_same_length(data: dict):
    
    max_length = max(len(lst) for lst in data.values())
    
    for key, lst in data.items():
        last_element = lst[-1]
        while len(lst) < max_length:
            lst.append(last_element)


# Utility function. Convert a date time column to the float format.
def datetime_to_float(time_df, datetime_column, time_unit="second"):

    df = time_df.copy()

    time_rateo = 1

    if time_unit == "minute":
        time_rateo = 60
    elif time_unit == "hour":
        time_rateo = 3600
    elif time_unit == "day":
        time_rateo = 24 * 3600
    else:
        raise ValueError

    df["time_float"] = df[datetime_column] - df[datetime_column].min()
    df["time_float"] = df["time_float"].dt.total_seconds()
    df["time_float"] = df["time_float"] / time_rateo

    return df


# Plot the hyperparameter cmap
def plot_grid_search_heatmap(data,
                             plot_size=20,
                             palette="gist_earth",
                             reverse_cmap=True,
                             color_bar_name="Loss",
                             color_bar_shrink=0.26,
                             annot_size=None,
                             linewidths=.5,
                             linecolor="purple",
                             plot_title="Grid Search"):
    """
    Plot the heatmap for hyperparameters grid search.
    """

    # setting the dimensions of the plot
    _, ax = plt.subplots(figsize=(plot_size, plot_size))

    # setting the heatmap color map
    cmap = sns.color_palette(palette, as_cmap=True)
    if reverse_cmap:
        cmap = cmap.reversed()

    # seaborn heatmap
    ax = sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        robust=True,
        annot=True,
        square=True,
        cbar_kws={'label': color_bar_name, 'shrink': color_bar_shrink},
        annot_kws={"size": annot_size},
        linewidths=linewidths,
        linecolor=linecolor
    )

    # set y ticks horizontally
    plt.yticks(rotation=0)

    # set the plot title
    if plot_title:
        plt.title(plot_title)


def plot_dismantling_graph(dismantle_df, line_colors, line_markers):

    ax = dismantle_df.plot.line(logx='sym',
                                figsize=(14, 4),
                                grid=True,
                                color=line_colors,
                                style=line_markers,
                                linewidth=0.5,
                                markevery=0.012,
                                ms=6,
                                title='Remaining misinformation vs account removal')

    # Set axes titles
    ax.set_ylabel("Fraction of remaining misinformation")
    ax.set_xlabel("Rank position")
    
    plt.show()