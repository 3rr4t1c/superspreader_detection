import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_lists_same_length(data: dict):
    
    max_length = max(len(lst) for lst in data.values())
    
    for key, lst in data.items():
        last_element = lst[-1]
        while len(lst) < max_length:
            lst.append(last_element)


# # Utility function. Convert a date time column to the float format.
# def datetime_to_float(time_df, datetime_column, time_unit="second"):

#     df = time_df.copy()

#     time_rateo = 1

#     if time_unit == "minute":
#         time_rateo = 60
#     elif time_unit == "hour":
#         time_rateo = 3600
#     elif time_unit == "day":
#         time_rateo = 24 * 3600
#     else:
#         raise ValueError

#     df["time_float"] = df[datetime_column] - df[datetime_column].min()
#     df["time_float"] = df["time_float"].dt.total_seconds()
#     df["time_float"] = df["time_float"] / time_rateo

#     return df


def datetime_to_float(time_df, datetime_columns, time_unit="second"):
    """
    Convert datetime columns in a DataFrame to float format based on the specified time unit.

    Args:
        time_df (DataFrame): Input DataFrame.
        datetime_columns (str or list): Name or list of names of the datetime columns to convert.
        time_unit (str, optional): Unit of time to convert to. Options: "second", "minute", "hour", "day". Defaults to "second".

    Returns:
        DataFrame: DataFrame with datetime columns replaced by float columns while keeping the original column names and positions.
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df = time_df.copy()

    # Validate time_unit
    valid_time_units = ["second", "minute", "hour", "day"]
    if time_unit not in valid_time_units:
        raise ValueError("Invalid time_unit. Choose from: {}".format(", ".join(valid_time_units)))

    # Convert datetime_columns to list if a single column name is provided
    if isinstance(datetime_columns, str):
        datetime_columns = [datetime_columns]

    # Calculate time conversion rate based on time_unit
    if time_unit == "minute":
        time_rate = 60
    elif time_unit == "hour":
        time_rate = 3600
    elif time_unit == "day":
        time_rate = 24 * 3600
    else:
        time_rate = 1  # Default is seconds

    # Convert each datetime column to float and replace the original column
    for column in datetime_columns:
        # Calculate time difference from the minimum datetime value in seconds
        df[column] = (df[column] - df[column].min()).dt.total_seconds()
        # Convert to specified time unit
        df[column] /= time_rate

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
        robust=False,
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


def generate_splits(interval, split):
    """
    Usage:
    >>> generate_splits(213.0, 1.5)
    >>> [213.0, 142.0, 94.0, 62.0, 41.0, 27.0, 18.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0]
    """

    splits = [interval]
    while interval > 1:
        interval //= split
        splits.append(interval)

    return splits