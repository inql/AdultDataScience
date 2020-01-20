import pandas as pd


def get_one_hat_from_categories(dataframe):
    data_cols = dataframe.select_dtypes('category').columns

    data_cols_dropped = dataframe.drop(columns=data_cols, axis=1)
    one_hot = pd.get_dummies(dataframe[data_cols])

    return pd.concat([data_cols_dropped, one_hot], axis=1, join='inner')
