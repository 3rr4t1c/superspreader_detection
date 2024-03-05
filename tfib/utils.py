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


def plot_heatmap(matrix,
                 row_axis_name,
                 col_axis_name,
                 colorbar_name,
                 x_ticks=None,
                 y_ticks=None,
                 palette='magma',
                 reverse_palette=False,
                 annot=True,
                 fmt=".4f",
                 annot_size=8,
                 figsize=(12, 8),
                 aspect='equal',
                 cbar_shrink=0.7):
    
    # Reverse the palette if specified
    if reverse_palette:
        palette = sns.color_palette(palette, as_cmap=True).reversed()
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, cmap=palette, annot=annot, fmt=fmt, 
                cbar_kws={'label': colorbar_name, 'shrink': cbar_shrink},
                annot_kws={"size": annot_size})

    # Rotate y-axis ticks horizontally for better readability
    plt.yticks(rotation=0)

    # Set labels for the axes
    plt.xlabel(col_axis_name)
    plt.ylabel(row_axis_name, rotation=90)  # Rotate y-axis label by 90 degrees
    
    # Set actual values for x and y ticks if provided
    if x_ticks is not None:
        plt.xticks(ticks=np.arange(len(x_ticks)) + 0.5, labels=x_ticks)
    if y_ticks is not None:
        plt.yticks(ticks=np.arange(len(y_ticks)) + 0.5, labels=y_ticks)

    # Set aspect ratio if specified
    plt.gca().set_aspect(aspect, adjustable='box')

    # Show the plot
    plt.show()


# Highlight the minimum
# # Find indices of minimum value
# min_row, min_col = np.unravel_index(np.argmin(matrix), matrix.shape)

# # Draw square around cell with lowest value
# cell_x = min_col
# cell_y = min_row
# cell_width = 1
# cell_height = 1
# rect = Rectangle((cell_x, cell_y), cell_width, cell_height, linewidth=2, edgecolor='red', facecolor='none')
# plt.gca().add_patch(rect)


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