from datetime import timedelta
import pandas as pd
from typing import List
import numpy as np


def train_val_test_split_by_time(df: pd.DataFrame, time_col: str, ratios: List[float]):
    if sum(ratios) != 1:
        raise ValueError("sum of ratios should be 1")
    start_time, end_time = df[time_col].min(), df[time_col].max()
    data_size = (end_time - start_time).total_seconds()
    split_times = [
        start_time + timedelta(seconds=int(data_size * v)) for v in np.cumsum(ratios)
    ]
    split_times = [start_time + timedelta(seconds=-1)] + split_times
    datasets = [
        df[(df[time_col] <= split_times[i]) & (df[time_col] > split_times[i - 1])]
        for i in range(1, len(split_times))
    ]
    print(
        "dataset time range: ",
        [(d[time_col].min(), d[time_col].max()) for d in datasets],
    )
    print("dataset sizes: ", [d.shape[0] for d in datasets])
    return datasets


def train_val_test_split(df: pd.DataFrame, ratios: List[float]):
    if sum(ratios) != 1:
        raise ValueError("sum of ratios should be 1")
    df_train = df.sample(frac=ratios[0])
    df_val = df.drop(df_train.index).sample(frac=ratios[1] / (ratios[1] + ratios[2]))
    df_test = df.drop(df_train.index.union(df_val.index))
    return df_train, df_val, df_test
