"""A module containing functions for plotting."""

import matplotlib.pyplot as plt


def get_labels_values_for_plot(df, column_name):
    """Get labels and valus of a categorical column.

    Args:
        df (PandasDF): A pandas dataframe
        column_name (str): The name of the categorical column

    Returns:
        A tuple of keys and values
    """
    variable_dict = df[column_name].value_counts().to_dict()
    keys = [str(i) for i in variable_dict.keys()]
    values = variable_dict.values()
    return keys, values


def plot_category_frequency(df, column_name, plot_title=None):
    """Plot the frequency of a categorical column.

    Args:
        df (PandasDF): A pandas dataframe
        column_name (str): The name of the categorical column
        plot_title (str): The title of the plot

    Returns:
        A barplot in matplotlib
    """
    keys, values = get_labels_values_for_plot(df, column_name)
    plt.figure(figsize=(7, 3))
    plt.bar(x=keys, height=values, width=0.4)
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    if plot_title:
        plt.title(plot_title)


def get_cut_bins_from_labels(labels):
    """Get a list of bins from the labels for pd.cut().

    Args:
        labels (list): A list of labels

    Returns:
        list: List of bins
    """
    cut_bins = [
        int(i.split('-')[0])
        for i in labels
    ] + [int(labels[-1].split('-')[1])]
    return cut_bins


class Labels:
    """Labels to transform continuous variable into categorical variable."""
    CATEGORICAL_LABELS = {
        'Age': ['18-24', '25-34', '35-44', '45-54', '55-200'],
        'Account_length': ['0-3', '4-9', '10-29', '30-49', '50-100'],
        'Total_income': [
            '0-10000', '10000-14999', '15000-24999', '25000-34999',
            '35000-49999', '50000-74999', '75000-99999',
            '100000-149999', '150000-199999', '200000-2000000',
        ],
        'Years_employed': ['0-3', '4-9', '10-24', '25-39', '40-59'],
    }
