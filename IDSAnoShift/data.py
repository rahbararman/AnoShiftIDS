import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def get_data_splits(X, Y, train_size=0.8):
    """This function splits the whole dataset into train, test and validation sets.

    Args:
        X (pd.DataFrame): DataFrame containing feature values of data points of shape (num_samples,num_features)
        Y (pd.DataFrame): DataFrame containing labels for samples of shape (num_samples,)
        train_size (float, optional): size of the training set in (0,1), Defaults to 0.8.

    Returns:
        tuple: tuple containing X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    X_train, X_, Y_train, Y_ = train_test_split(X, Y, train_size=train_size, stratify=Y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_, Y_, train_size=0.5, stratify=Y_)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def rename_columns(df):
    """This function renames the columns of features DataFrame into cat_ and num_ styles.

    Args:
        df (pd.DataFrame): A DataFrame containing features as columns and samples as rows

    Returns:
        pd.DataFrame: A DataFrame with renamed columns.
    """
    categorical_cols = ["0", "1", "2", "3", "13"]
    numerical_cols = ["4", "5", "6", "7", "8", "9", "10", "11", "12"]
    new_names = []
    for col_name in df.columns.astype(str).values:
        if col_name in numerical_cols:
            df[col_name] = pd.to_numeric(df[col_name])
            new_names.append((col_name, "num_" + col_name))
        elif col_name in categorical_cols:
            new_names.append((col_name, "cat_" + col_name))
        else: # pragma: no cover, other data
            new_names.append((col_name, col_name))
    df.rename(columns=dict(new_names), inplace=True)
    return df


def preprocess(X_df, Y_df=None, label_col="18", enc=None):
    """This function preprocess the features and labels DataFrames by encoding categorical features and relabeling. If Y_df is not None, normal samples get label 1 and anomalies get label -1.

    Args:
        X_df (pd.DataFrame): DataFrame containing features with renamed columns.
        Y_df (pd.DataFrame, optional): DataFrame containing the labels for samples in X_df. Defaults to None.
        label_col (str, optional): Name of the label column in Y_df. Defaults to "18".
        enc (sklearn.preprocessing.OneHotEncoder, optional): Fitted OneHotEncoder, a new encoder will be fit to the data if none is given. Defaults to None.

    Returns:
        tuple: df_new, Y_df, enc, the encoded features, the labels, and the OneHotEncoder
    """
    X_df = X_df.reset_index(drop=True)
    if Y_df is not None:
        Y_df = Y_df.reset_index(drop=True)
    if not enc:
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit(X_df.loc[:, ["cat_" in i for i in X_df.columns]])
    num_cat_features = enc.transform(X_df.loc[:, ["cat_" in i for i in X_df.columns]]).toarray()
    df_catnum = pd.DataFrame(num_cat_features)
    df_catnum = df_catnum.add_prefix("catnum_")
    df_new = pd.concat([X_df, df_catnum], axis=1)
    if Y_df is not None:
        filter_clear = Y_df[label_col] == 1
        filter_infected = Y_df[label_col] < 0
        Y_df[label_col][filter_clear] = 1
        Y_df[label_col][filter_infected] = -1
    return df_new, Y_df, enc


def get_preprocessed_train(X_df, Y_df, label_col="18"):
    """This function prepares the dataset for training. Returns only normal samples.

    Args:
        X_df (pd.DataFrame): A DataFrame containing features for samples of shape (num_samples, num_features)
        Y_df (pd.DataFrame): A DataFrame containing labels for data samples of shape (n_samples,)
        label_col (str, optional): Name of the label column in Y_df. Defaults to "18"

    Returns:
        tuple: X_train_num, Y_train_clear, numerical_cols, ohe_enc: features and labels of normal samples together with names of the columns for training and the OneHotEncoder used in preprocessing
    """
    count_norm = X_df[Y_df[label_col] == 1].shape[0]
    count_anomaly = X_df[Y_df[label_col] != 1].shape[0]
    print("normal:", count_norm, "anomalies:", count_anomaly)
    X_df = rename_columns(X_df)
    X_df, Y_df, ohe_enc = preprocess(X_df, Y_df)
    # select numerical features
    numerical_cols = X_df.columns.to_numpy()[["num_" in i for i in X_df.columns]]
    X_train_clear = X_df[Y_df[label_col] == 1]
    Y_train_clear = Y_df[Y_df[label_col] == 1]
    X_train_num = X_train_clear[numerical_cols]
    return X_train_num, Y_train_clear, numerical_cols, ohe_enc


def get_test(X_df, enc, Y_df=None):
    """This function works similar to get_preprocessed_train, but prepares the test dataset for evaluation or inference purposes.

    Args:
        (pd.DataFrame): A DataFrame containing features for samples of shape (num_samples, num_features)
        enc (sklearn.preprocessing.OneHotEncoder): A fitted OneHotEncoder for transforming features
        Y_df (pd.DataFrame, optional): A DataFrame containing labels for data samples of shape (n_samples,), is none for inference purposes. Defaults to None.

    Returns:
        tuple: X_df, Y_df, preprocessed features and labels
    """
    X_df = rename_columns(X_df)
    X_df, Y_df, _ = preprocess(X_df, Y_df, enc=enc)
    numerical_cols = X_df.columns.to_numpy()[["num_" in i for i in X_df.columns]]
    X_df = X_df[numerical_cols]
    return X_df, Y_df
