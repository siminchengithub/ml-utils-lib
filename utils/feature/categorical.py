import pandas as pd


def mean_encoding(cat_s: pd.Series, num_s: pd.Series):
    cat_s, num_s = cat_s.reset_index(drop=True), num_s.reset_index(drop=True)
    cat_label = cat_s.name
    num_label = num_s.name
    data = pd.DataFrame({cat_label: cat_s, num_label: num_s})
    codes = data.groupby(cat_label)[num_label].mean()
    return cat_s.map(codes), codes


def merge_rare(cat_s: pd.Series, min_freq=None, max_unique=None, rare_label: str = None):
    cat_s = cat_s.reset_index(drop=True).copy().fillna("MISSING")
    if min_freq and max_unique:
        raise ValueError("only one of min_freq and max_unique can be specified")
    if min_freq is None and max_unique is None:
        raise ValueError("one of min_freq and max_unique must be specified")
    rare_label = rare_label or "rare"
    cnts = cat_s.value_counts(ascending=False)
    if min_freq:
        rare_vals = cnts[cnts < min_freq].index
    else:
        rare_vals = cnts.iloc[max_unique:].index
    codes = {v: v if v not in rare_vals else rare_label for v in cat_s}
    cat_s[cat_s.isin(rare_vals)] = rare_label
    return cat_s, codes
