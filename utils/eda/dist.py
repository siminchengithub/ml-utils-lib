from os import stat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display
import sklearn.feature_selection as feature_selection
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing as preprocessing
from scipy import stats
from typing import List

plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True


def plot_datetime(s: pd.Series, figsize=(16, 3), level="h"):
    s = s.reset_index(drop=True)
    n_raw = s.isna().sum()
    s = s.dropna()
    s = s.astype(f"<M8[{level}]")
    c = s.value_counts().sort_index()
    if level in ["M", "Y"]:
        freq = "D"
    else:
        freq = level
    index = (
        pd.date_range(s.min(), s.max(), freq=freq)
        .to_series()
        .astype(f"<M8[{level}]")
        .drop_duplicates()
    )
    c_reindex = c.reindex(index, fill_value=0)
    fig = plt.figure(figsize=figsize)
    c_reindex.plot(kind="line")
    plt.title(
        f"time series plot of {s.name}, level={level}, dropped {n_raw} missing values\n"
        + f"original x axis size is {len(c)}, after filling missing datetime size is {len(index)}\n"
    )
    plt.show()


def plot_categorical(s: pd.Series, cap_n=25, figsize=(16, 3), label=None):
    s = s.reset_index(drop=True)
    n_na = s.isna().sum()
    s = s.dropna()
    c = s.value_counts()
    title = f"frequency count of {s.name}, {len(c)} unique values, {n_na} missing values"
    if label:
        title = title + f", data={label}"
    if len(c) > cap_n:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        c[:cap_n].plot(kind="bar", ax=axes[0])
        c[len(c) - cap_n :].plot(kind="bar", ax=axes[1])
        axes[0].set_title(f"top {cap_n}")
        axes[1].set_title(f"bottom {cap_n}")
        plt.suptitle(title)
        [ax.tick_params(axis="x", rotation=45) for ax in axes]
        plt.show()
    else:
        fig = plt.figure(figsize=figsize)
        c.plot(kind="bar", figsize=figsize)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.show()


def plot_numeric(
    s: pd.Series,
    quantile_threshold=0.975,
    bins=100,
    figsize=(16, 3),
    label=None,
):
    s = s.reset_index(drop=True)
    n_na = s.isna().sum()
    s = s.dropna()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    s.plot(kind="hist", bins=bins, ax=axes[0])
    cutoff = np.quantile(s, quantile_threshold)
    s[s <= cutoff].plot(kind="hist", bins=bins, ax=axes[1])
    axes[0].set_title("full data")
    axes[1].set_title(f"cutoff={cutoff}({quantile_threshold}q)")
    suptitle = f"histogram of {s.name}{f', data={label}' if label else ''}, dropped {n_na} missing values"
    plt.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_numeric_in_categorical(
    num_s: pd.Series,
    cat_s: pd.Series,
    quantile_threshold=0.975,
    kind="hist",
    discrete=False,
    n_neighbors=5,
    bins=100,
    figsize=(16, 6),
):
    num_s, cat_s = num_s.reset_index(drop=True), cat_s.reset_index(drop=True)
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    num_label, cat_label = num_s.name, cat_s.name
    print(num_label, cat_label)
    data = pd.DataFrame({num_label: num_s, cat_label: cat_s})
    cutoffs = {}
    nas = {}
    cat_vals = cat_s.unique()
    for c in cat_vals:
        s = data[data[cat_label] == c][num_label]
        n_na = s.isna().sum()
        nas[c] = n_na
        s = s.dropna()
        cutoff, s = quantile_filter(s, quantile_threshold)
        s.plot(kind="hist", bins=bins, alpha=0.5, ax=axes[0], label=c)
        data = data[
            (data[cat_label] != c)
            | ((data[cat_label] == c) & (data[num_label] <= cutoff))
        ]
        cutoffs[c] = round(cutoff, 5)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    sns.boxplot(y=cat_label,x=num_label, data=data, ax=axes[1])
    mi = feature_selection.mutual_info_classif(
        data[num_label].values.reshape(-1, 1),
        y=pd.factorize(data[cat_label].values)[0],
        discrete_features=discrete,
        n_neighbors=n_neighbors,
    )[0]
    f_p_value = feature_selection.f_classif(
        data[num_label].values.reshape(-1, 1),
        y=pd.factorize(data[cat_label].values)[0],
    )[1][0]
    plt.suptitle(
        f"plot of {num_label} per {cat_label}\n cutoffs({quantile_threshold}q): {cutoffs}\n dropped NAs: {nas}\n MI: {round(mi, 3)}, ANOVA F-test p-value: {round(f_p_value, 5)}"
    )
    plt.show()


def plot_categorical_in_categorical(
    r_s: pd.Series,
    p_s: pd.Series,
    max_per_cat=10,
    max_cat=10,
    n_neighbors=5,
    figsize=(16, 3),
):
    r_s, p_s = r_s.reset_index(drop=True), p_s.reset_index(drop=True)
    fig, axe = plt.subplots(1, 1, figsize=figsize)
    p_label, r_label = p_s.name, r_s.name
    print(p_label, r_label)
    p_s = p_s.fillna("MISSING")
    data = pd.DataFrame({p_label: p_s, r_label: r_s})
    g_cnt = data.groupby([p_label, r_label]).size().reset_index(name="count")
    t_cnt = g_cnt.groupby(g_cnt[r_label])["count"].sum().reset_index(name="group_count")
    t_cnt = t_cnt.sort_values("group_count", ascending=False).head(max_cat)
    g_cnt = pd.merge(g_cnt, t_cnt, on=r_label, how="inner")
    g_cnt["perc"] = g_cnt["count"] / g_cnt["group_count"]
    g_cnt["rank"] = g_cnt.groupby(r_label)["perc"].rank(method="first", ascending=False)
    g_cnt = g_cnt[g_cnt["rank"] <= max_per_cat].sort_values(
        "group_count", ascending=False
    )
    sns.barplot(x=r_label, y="perc", hue=p_label, data=g_cnt, ax=axe)
    mi = feature_selection.mutual_info_classif(
        pd.factorize(data[p_label].values)[0].reshape(-1, 1),
        pd.factorize(data[r_label].values)[0],
        discrete_features=True,
        n_neighbors=n_neighbors,
    )[0]
    chi2_p_value = stats.chi2_contingency(pd.crosstab(data[r_label], data[p_label]))[1]
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.xticks(rotation=45)
    plt.title(
        f"Percentage of {p_label} in {r_label}, showing at most {max_cat} {r_label} and at most {max_per_cat} {p_label} per {r_label}\n \
             MI: {round(mi, 3)}, chi2 p-value: {chi2_p_value}"
    )
    plt.show()
    display(t_cnt.set_index(r_label).loc[g_cnt[r_label].unique()].T)


def plot_numeric_in_numeric(
    p_s: pd.Series,
    r_s: pd.Series,
    quantile_threshold=0.975,
    discrete=False,
    n_neighbors=5,
    figsize=(6, 6),
):
    r_s, p_s = r_s.reset_index(drop=True), p_s.reset_index(drop=True)
    p_label, r_label = p_s.name, r_s.name
    print(p_label, r_label)
    n_row = p_s.shape[0]
    data = pd.DataFrame({p_label: p_s, r_label: r_s}).dropna()
    n_na = n_row - data.shape[0]
    cutoff_p, _ = quantile_filter(data[p_label], quantile_threshold)
    cutoff_r, _ = quantile_filter(data[r_label], quantile_threshold)
    data = data[(data[r_label] <= cutoff_r) & (data[p_label] <= cutoff_p)]
    fig, axe = plt.subplots(1, 1, figsize=figsize)
    sns.scatterplot(x=p_label, y=r_label, data=data, ax=axe, s=5)
    mi = feature_selection.mutual_info_regression(
        data[p_label].values.reshape(-1, 1),
        y=data[r_label].values,
        discrete_features=discrete,
        n_neighbors=n_neighbors,
    )[0]
    r = feature_selection.r_regression(
        data[p_label].values.reshape(-1, 1),
        y=data[r_label].values,
    )[0]
    f_p_value = feature_selection.f_regression(
        data[p_label].values.reshape(-1, 1),
        y=data[r_label].values,
    )[1][0]
    cutoffs = {p_label: cutoff_p, r_label: cutoff_r}
    axe.set_title(
        f"dropped {n_na} pairs with missing data, cutoffs({quantile_threshold}q): {cutoffs}\n MI: {round(mi, 3)}, r: {round(r, 3)}, F-test p-value: {round(f_p_value, 5)}"
    )
    plt.show()


def quantile_filter(s: pd.Series, threshold: float) -> pd.Series:
    cutoff = np.quantile(s, threshold)
    s = s[s <= cutoff]
    return cutoff, s


def plot_cumsum(s: pd.Series, figsize=(12, 6)):
    s = s / s.sum()
    cumsums = np.cumsum(s)
    fig, axe = plt.subplots(1, 1, figsize=figsize)
    cumsums.plot(kind="line")
    plt.show()


def product_unique(l: List):
    res = []
    for i in range(0, len(l) - 1):
        for j in range(i + 1, len(l)):
            res.append((l[i], l[j]))
    return res
