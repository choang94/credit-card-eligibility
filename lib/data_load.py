"""Module containing functions that load the data."""


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def split_train_validation_data(
    data,
    id_column,
    shuffle_data=True,
    validation_size=0.2,
    random_seed=13,
):
    """Split data into train and validation sets.

    Args:
        data (PandasDF): A pandas dataframe of the data
        id_column (str): The name of the identifier column
        shuffle_data (bool, Defaults to True): Flag for shuffling data
             before splitting
        validation_size (float, Defaults to 0.2): Size of the validation set
        random_seed (int): The version number if shuffling

    Returns:
        tuple: (train dataset, validation data set)
    """
    rows_count = data.shape[0]

    if validation_size < 0 or validation_size > 1:
        raise ValueError(f"""Invalid validation_size: {validation_size}.
            validation_size has to be between 0 and 1.""")

    train_count = int(rows_count * (1 - validation_size))

    if shuffle_data:
        train_df = shuffle(
            data, n_samples=train_count,
            random_state=random_seed,
        )
    else:
        train_df = data.loc[:train_count, :]

    validation_df = data[~data[id_column].isin(train_df[id_column].to_list())]
    return train_df, validation_df


def split_train_test(
    train_df,
    excluded_columns,
    target_column,
    test_size=0.2,
    random_state=23,
):
    """Split the training dataframe into train and test sets.

    Args:
        train_df (PandasDF): The training dataframe
        excluded_columns (list): A list of columns that is excluded
            from training the model
        target_column (str): the target column name
        test_size (float. Defaults to 0.2): The size of the test set
        random_state (int): The version number. Defaults to 23

    Returns:
        Tuple: (x_train, x_test, y_train, y_test)
    """
    columns_to_keep = list(set(train_df.columns) - set(excluded_columns))
    X = train_df.loc[:, columns_to_keep]
    Y = train_df[[target_column]]

    # split data
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, random_state=random_state, test_size=test_size,
    )
    return x_train, x_test, y_train, y_test


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
