import pandas as pd
from IPython.display import display
import missingno as msno
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)


def basic_check(df: pd.DataFrame, key_cols, label=None, figsize=(8, 8)):
    if label:
        print(f"\nbasic checks for {label}\n\n")
    print("shape:\n", df.shape)
    print("duplicates perc: ", df.duplicated(subset=key_cols, keep="first").mean())
    dtypes = df.dtypes.rename("dtype")
    nas = df.isnull().mean().rename("null perc")
    display(pd.concat([dtypes, nas], axis=1))
    msno.matrix(df, figsize=figsize)
    display(df.sample(2))


def cast_column_types(
    df: pd.DataFrame,
    categorical_cols=None,
    numeric_cols=None,
    timestamp_cols=None,
    timestamp_format=None,
    derived_col_prefix="derived",
):
    df = df.copy()
    if timestamp_cols:
        for col in timestamp_cols:
            cast_col = f"{derived_col_prefix}_{col}"
            df[cast_col] = pd.to_datetime(df[col], format=timestamp_format)
    if categorical_cols:
        df[categorical_cols] = df[categorical_cols].astype("category")
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].astype("float")
    return df
