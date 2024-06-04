"""Module containing functions that load the data."""


def get_data_matrix(data, target, excluded_features=None):
    """Get X, y data matrix for splitting and modeling.

    Args:
        data (PandasDF): The Pandas DF of the data
        target (str): The name of the target column
        excluded_features (list, Optional): A list of features that is
            excluded from the model

    Returns:
        tuple: 2 data matrix (X, y)
    """
    if excluded_features:
        columns_to_keep = list(
            set(data.columns) -
            set(excluded_features + [target]),
        )
    else:
        columns_to_keep = list(set(data.columns) - {target})
    X = data.loc[:, columns_to_keep]
    y = data[[target]]
    return X, y


def plot_target_variable(y_train, n_class, tick_labels):
    """Plot the target variable to examine the class balance.

    Args
        y_train (array): An array of the target variable
        n_class (int): Number of classes
        tick_labels (list): A list of the labels for target variable

    Returns:
        matplotlib histogram
    """
    import matplotlib.pyplot as plt

    plt.hist(y_train, bins=n_class, rwidth=0.8)
    plt.xticks(tick_labels)
