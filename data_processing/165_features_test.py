import ctypes
import gc
import os
import pickle
import random
import re
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import polars as pl
import sklearn
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer

## Sklearn package
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
pd.options.display.max_rows = 999
pd.options.display.max_colwidth = 99


class FeatureExtractor:
    def __init__(self, logs):
        self.logs = logs  # Training logs

    def _count_by_values(self, colname, used_cols):
        fts = self.logs.select(pl.col("id").unique(maintain_order=True))
        for i, col in enumerate(used_cols):
            tmp_logs = self.logs.group_by("id").agg(
                pl.col(colname).is_in([col]).sum().alias(f"{colname}_{i}_cnt")
            )
            fts = fts.join(tmp_logs, on="id", how="left")
        return fts

    # Create the features from statistics of activities, text changes, events
    def create_count_by_values_feats(self):
        activities = ["Input", "Remove/Cut", "Nonproduction", "Replace", "Paste"]
        events = [
            "q",
            "Space",
            "Backspace",
            "Shift",
            "ArrowRight",
            "Leftclick",
            "ArrowLeft",
            ".",
            ",",
            "ArrowDown",
            "ArrowUp",
            "Enter",
            "CapsLock",
            "'",
            "Delete",
            "Unidentified",
        ]
        text_changes = [
            "q",
            " ",
            ".",
            ",",
            "\n",
            "'",
            '"',
            "-",
            "?",
            ";",
            "=",
            "/",
            "\\",
            ":",
        ]
        # === Create the feature columns using count by values ===
        df = self._count_by_values("activity", activities)  # Create 'activity' column
        df = df.join(
            self._count_by_values("text_change", text_changes), on="id", how="left"
        )
        df = df.join(self._count_by_values("down_event", events), on="id", how="left")
        df = df.join(self._count_by_values("up_event", events), on="id", how="left")
        # print(df.collect().head())
        return df

    # Create the features
    def create_input_words_feats(self):
        # Filter no changes
        df = self.logs.filter(
            (~pl.col("text_change").str.contains("=>"))
            & (pl.col("text_change") != "NoChange")
        )
        # Aggregate the text changes by id
        df = df.group_by("id").agg(
            pl.col("text_change").str.concat("").str.extract_all(r"q+")
        )
        # creates only two columns ('id' and 'text_change')
        df = df.with_columns(
            input_word_count=pl.col("text_change").list.lengths(),
            input_word_length_mean=pl.col("text_change").apply(
                lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0)
            ),
            input_word_length_max=pl.col("text_change").apply(
                lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0)
            ),
            input_word_length_std=pl.col("text_change").apply(
                lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0)
            ),
            input_word_length_median=pl.col("text_change").apply(
                lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0)
            ),
            input_word_length_skew=pl.col("text_change").apply(
                lambda x: skew([len(i) for i in x] if len(x) > 0 else 0)
            ),
        )
        df = df.drop(
            "text_change"
        )  # Remove 'text_change' to avoid including duplicated `text_change` column
        return df

    # Create the statistical numeric features (e.g. sum, median, mean min, 0.5_quantile)
    def create_numeric_feats(self):
        num_cols = [
            "down_time",
            "up_time",
            "action_time",
            "cursor_position",
            "word_count",
        ]
        df = self.logs.group_by("id").agg(
            pl.sum("action_time").suffix("_sum"),
            pl.mean(num_cols).suffix("_mean"),
            pl.std(num_cols).suffix("_std"),
            pl.median(num_cols).suffix("_median"),
            pl.min(num_cols).suffix("_min"),
            pl.max(num_cols).suffix("_max"),
            pl.quantile(num_cols, 0.5).suffix("_quantile"),
        )
        return df

    def create_categorical_feats(self):
        df = self.logs.group_by("id").agg(
            pl.n_unique(["activity", "down_event", "up_event", "text_change"])
        )
        return df

    # Create the idle time features
    def create_idle_time_feats(self):
        df = self.logs.with_columns(
            pl.col("up_time").shift().over("id").alias("up_time_lagged")
        )
        df = df.with_columns(
            (abs(pl.col("down_time") - pl.col("up_time_lagged")) / 1000)
            .fill_null(0)
            .alias("time_diff")
        )
        df = df.filter(pl.col("activity").is_in(["Input", "Remove/Cut"]))
        df = df.group_by("id").agg(
            inter_key_largest_lantency=pl.max("time_diff"),
            inter_key_median_lantency=pl.median("time_diff"),
            mean_pause_time=pl.mean("time_diff"),
            std_pause_time=pl.std("time_diff"),
            total_pause_time=pl.sum("time_diff"),
            pauses_half_sec=pl.col("time_diff")
            .filter((pl.col("time_diff") > 0.5) & (pl.col("time_diff") < 1))
            .count(),
            pauses_1_sec=pl.col("time_diff")
            .filter((pl.col("time_diff") > 1) & (pl.col("time_diff") < 1.5))
            .count(),
            pauses_1_half_sec=pl.col("time_diff")
            .filter((pl.col("time_diff") > 1.5) & (pl.col("time_diff") < 2))
            .count(),
            pauses_2_sec=pl.col("time_diff")
            .filter((pl.col("time_diff") > 2) & (pl.col("time_diff") < 3))
            .count(),
            pauses_3_sec=pl.col("time_diff").filter(pl.col("time_diff") > 3).count(),
        )
        return df

    # Create p-bursts features using up and down time and activity
    def create_p_bursts_feats(self):
        df = self.logs.with_columns(
            pl.col("up_time").shift().over("id").alias("up_time_lagged")
        )
        df = df.with_columns(
            (abs(pl.col("down_time") - pl.col("up_time_lagged")) / 1000)
            .fill_null(0)
            .alias("time_diff")
        )
        df = df.filter(pl.col("activity").is_in(["Input", "Remove/Cut"]))
        df = df.with_columns(pl.col("time_diff") < 2)
        df = df.with_columns(
            pl.when(pl.col("time_diff") & pl.col("time_diff").is_last())
            .then(pl.count())
            .over(pl.col("time_diff").rle_id())
            .alias("P-bursts")
        )
        df = df.drop_nulls()
        df = df.group_by("id").agg(
            pl.mean("P-bursts").suffix("_mean"),
            pl.std("P-bursts").suffix("_std"),
            pl.count("P-bursts").suffix("_count"),
            pl.median("P-bursts").suffix("_median"),
            pl.max("P-bursts").suffix("_max"),
            pl.first("P-bursts").suffix("_first"),
            pl.last("P-bursts").suffix("_last"),
        )
        return df

    # Create r-burst features using activity
    def create_r_bursts_feats(self):
        df = self.logs.filter(pl.col("activity").is_in(["Input", "Remove/Cut"]))
        df = df.with_columns(pl.col("activity").is_in(["Remove/Cut"]))
        df = df.with_columns(
            pl.when(pl.col("activity") & pl.col("activity").is_last())
            .then(pl.count())
            .over(pl.col("activity").rle_id())
            .alias("R-bursts")
        )
        df = df.drop_nulls()
        df = df.group_by("id").agg(
            pl.mean("R-bursts").suffix("_mean"),
            pl.std("R-bursts").suffix("_std"),
            pl.median("R-bursts").suffix("_median"),
            pl.max("R-bursts").suffix("_max"),
            pl.first("R-bursts").suffix("_first"),
            pl.last("R-bursts").suffix("_last"),
        )
        return df

    # Main function creates all 122 features
    def create_feats(self):
        feats = self.create_count_by_values_feats()  # 52 columns in total
        #         print(f"< Count by values features > {len(feats.columns)}")
        feats = feats.join(
            self.create_input_words_feats(), on="id", how="left"
        )  # 58 columns
        #         print(f"< Input words stats features > {len(feats.columns)}")
        feats = feats.join(
            self.create_numeric_feats(), on="id", how="left"
        )  # 89 columns
        #         print(f"< Numerical features > {len(feats.columns)}")
        feats = feats.join(
            self.create_categorical_feats(), on="id", how="left"
        )  # 93 columns
        #         print(f"< Categorical features > {len(feats.columns)}")
        feats = feats.join(
            self.create_idle_time_feats(), on="id", how="left"
        )  # 103 columns
        #         print(f"< Idle time features > {len(feats.columns)}")
        feats = feats.join(
            self.create_p_bursts_feats(), on="id", how="left"
        )  # 110 columns
        #         print(f"< P-bursts features > {len(feats.columns)}")
        feats = feats.join(
            self.create_r_bursts_feats(), on="id", how="left"
        )  # 116 columns
        #         print(f"< R-bursts features > {len(feats.columns)}")
        return feats  # 116 features


if __name__ == "__main__":
    DATA_ROOT = (
        "/Users/kaiqu/kaggle-datasets/linking-writing-processes-to-writing-quality"
    )
    train_logs = pl.scan_csv(f"{DATA_ROOT}/train_logs.csv")
    feautre_extractor = FeatureExtractor(train_logs)
    train_features = feautre_extractor.create_feats().collect().to_pandas()
    print(train_features.head())
