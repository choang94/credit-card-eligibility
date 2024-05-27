"""A module containing functions for plotting."""

import matplotlib.pyplot as plt


def plot_category_frequency(df, column_name, plot_title):
    """Plot the frequency of a categorical column.

    Args:
        df (PandasDF): A pandas dataframe
        column_name (str): The name of the categorical column
        plot_title (str): The title of the plot

    Returns:
        A barplot in matplotlib
    """
    variable_dict = df[column_name].value_counts().to_dict()
    keys = [str(i) for i in variable_dict.keys()]
    values = variable_dict.values()

    plt.bar(x=keys, height=values)
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.title(plot_title)
