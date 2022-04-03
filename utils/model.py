from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from . import eda
from . import feature

from IPython.display import display

class Model(ABC):
    def __init__(self, num_features, cat_features, target_col):
        self.num_features = num_features
        self.cat_features = cat_features
        self.target_col = target_col
        self.model_num_features = self.num_features.copy()
        self.model_cat_features = self.cat_features.copy()

    @property
    def features(self):
        return self.num_features + self.cat_features

    @property
    def model_features(self):
        return self.model_num_features + self.model_cat_features

    @property
    def model_name(self):
        return self.__class__.__name__

    def train_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def predict_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @abstractmethod
    def feature_engineering_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def feature_engineering_apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class BaseRegressionModel(Model):
    def feature_engineering_fit(self, df: pd.DataFrame):
        num_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        cat_transformer = Pipeline(
            steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))]
        )
        feature_pipeline = ColumnTransformer(
            transformers=[
                ("num", num_transformer, self.num_features),
                ("cat", cat_transformer, self.cat_features),
            ]
        )
        self.feature_pipeline = feature_pipeline.fit(df)
        df = self.feature_pipeline.transform(df)
        return df

    def feature_engineering_apply(self, df: pd.DataFrame):
        df = self.feature_pipeline.transform(df)
        return df

    def predict(self, df: pd.DataFrame):
        df = self.predict_preprocess(df)
        X = self.feature_engineering_apply(df)
        predictions = self.model.predict(X)
        return predictions

    def train(self, df: pd.DataFrame, model=None):
        model = model or LinearRegression()
        print(f"Training...{'#'*10}")
        df = self.train_preprocess(df)
        X, y = df[self.model_features], df[self.target_col]
        X = self.feature_engineering_fit(X)
        model = model.fit(X, y)
        self.model = model
        self.evaluate(df)
        return self

    def evaluate(self, df: pd.DataFrame):
        X, y = df[set(df.columns) - set(self.target_col)], df[self.target_col]
        pred = self.predict(X)
        print(f"model: {self.model_name}")
        print("features: ", self.model_features)
        print("RMSE: ", mean_squared_error(y, pred, squared=False))
        print("MAE: ", mean_absolute_error(y, pred))
        print("\n")


class BaseClassificationModel(Model):
    def feature_engineering_fit(self, df: pd.DataFrame):
        num_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                ("scaler", StandardScaler())
            ]
        )
        cat_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent", fill_value="NA", missing_values=np.nan
                    ),
                ),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        feature_pipeline = ColumnTransformer(
            transformers=[
                ("num", num_transformer, self.num_features),
                ("cat", cat_transformer, self.cat_features),
            ]
        )
        self.feature_pipeline = feature_pipeline.fit(df)
        df = self.feature_pipeline.transform(df)
        return df

    def feature_engineering_apply(self, df: pd.DataFrame):
        df = self.feature_pipeline.transform(df)
        return df

    def predict(self, df: pd.DataFrame):
        df = self.predict_preprocess(df)
        X = self.feature_engineering_apply(df)
        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, df: pd.DataFrame):
        df = self.predict_preprocess(df)
        X = self.feature_engineering_apply(df)
        pred_proba = self.model.predict_proba(X)
        pred_proba = pd.DataFrame(pred_proba, columns=self.model.classes_)
        return pred_proba

    def train(self, df: pd.DataFrame, model=None):
        model = model or LogisticRegression(class_weight="balanced")
        print(f"Training...{'#'*10}")
        df = self.train_preprocess(df)
        X, y = df[self.model_features], df[self.target_col]
        X = self.feature_engineering_fit(X)
        model = model.fit(X, y)
        self.model = model
        self.evaluate(df, display_pred_distribution=True)
        return self

    def evaluate(self, df: pd.DataFrame, display_pred_distribution=False):
        X, y = df[set(df.columns) - set(self.target_col)], df[self.target_col]
        pred = self.predict(X)
        pred_prob = self.predict_proba(X)
        print(f"model: {self.model_name}")
        print("features: ", self.model_features)
        print(classification_report(y, pred))
        print("Confusion matrix: \n")
        display(pd.DataFrame(confusion_matrix(y, pred), columns=self.model.classes_))
        print(confusion_matrix(y, pred))
        # if y.unique().shape[0] < 3:
            # print("average precision score: ", average_precision_score(y, pred))
        print("roc-auc")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for c in self.model.classes_:
            print(c, roc_auc_score((y == c).astype(int), pred_prob[c]))
            x_axe, y_axe, _ = roc_curve((y == c).astype(int), pred_prob[c])
            pd.Series(index=x_axe, data=y_axe).plot(kind="line", ax=axes[0], label=c)
            x_axe, y_axe, _ = precision_recall_curve((y == c).astype(int), pred_prob[c])
            pd.Series(index=y_axe, data=x_axe).plot(kind="line", ax=axes[1], label=c)
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        axes[0].set_title("roc-auc per class, OVR")
        axes[1].set_xlabel("recall")
        axes[1].set_ylabel("precisioon")
        axes[1].set_title("precision-recall per class, OVR")
        plt.legend()
        plt.show()
        if display_pred_distribution:
            for c in self.model.classes_:
                eda.dist.plot_numeric_in_categorical(pred_prob[c], pd.Series((y == c).astype(str)), quantile_threshold=1, bins=20)
        print("\n")


class MeanEncodingClassifier(BaseClassificationModel):
    def __init__(self, num_features, cat_features, target_col, rare_encode_features=None, mean_encode_features=None, mean_encode_target_col=None):
        super().__init__(num_features, cat_features, target_col)
        self.mean_codes = {c: {} for c in mean_encode_features or self.cat_features}
        self.rare_codes = {c: None for c in rare_encode_features or self.cat_features}
        self.target_values = None
        self.mean_encode_target_col = self.target_col or mean_encode_target_col

    def train_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.model_num_features = self.num_features.copy()
        self.model_cat_features = self.cat_features.copy()
        df = df.copy()
        self.target_values = df[self.target_col].unique()
        for col in self.cat_features.copy():
            df[col], codes = feature.categorical.merge_rare(df[col], min_freq=3)
            self.rare_codes[col] = codes
            for t in self.target_values[:-1]:
                f_name = f"{col}_{t}_mean"
                df[f_name], codes = feature.categorical.mean_encoding(df[col].fillna("rare"), (df[self.mean_encode_target_col] == t).astype(int))
                self.model_num_features.append(f_name)
                self.mean_codes[col][t] = codes
            self.model_cat_features.remove(col)
        return df

    def predict_preprocess(self, df: pd.DataFrame):
        df = df.copy()
        for col in self.rare_codes:
            df[col] = df[col].map(self.rare_codes[col])
        for col in self.mean_codes:
            for t in self.target_values[:-1]:
                df[f"{col}_{t}_mean"] = df[col].fillna("rare").map(self.mean_codes[col][t])
        return df

    def feature_engineering_fit(self, df: pd.DataFrame):
        num_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        feature_pipeline = ColumnTransformer(
            transformers=[
                ("num", num_transformer, self.model_num_features),
            ]
        )
        self.feature_pipeline = feature_pipeline.fit(df)
        df = self.feature_pipeline.transform(df)
        return df