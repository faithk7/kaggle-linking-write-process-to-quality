import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

sys.path.append("../")  # to make modules in the other directories visible

from data_processing.feature_engineering import *


def train(df, df_score):
    X = get_all_features(df)
    train_data = X.merge(df_score, on="id")
    y = train_data["score"]

    # print("X.shape", X.shape)
    # print("y.shape", y.shape)

    # make sure that we remove the id column after getting the desired targets
    X.drop(["id"], axis=1, inplace=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=CFG.random_state)

    oof_preds = np.zeros(X.shape[0])
    models = []

    for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
        print(f"Training fold {fold + 1}")

        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model = get_model()

        # TODO: a more elegant way to pass the paramters elegantly, currently it's a mess
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=CFG.early_stopping_rounds),
                lgb.callback.log_evaluation(period=CFG.logging_period),
            ],
        )
        oof_preds[valid_index] = model.predict(X_valid)

        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
            print(f"Created directory {MODEL_PATH}")
        else:
            print(f"Directory {MODEL_PATH} already exists")

        model_filename = os.path.join(MODEL_PATH, f"{CFG.model}_fold_{fold+1}.joblib")
        joblib.dump(model, model_filename)
        print(f"Model for fold {fold + 1} saved to {model_filename}")
        models.append(model)

    mse = mean_squared_error(y, oof_preds)
    print(f"Overall MSE: {mse}")

    return models


def get_model():
    if CFG.model == "lgbm":
        model = lgb.LGBMRegressor(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
            random_state=42,
            early_stopping_rounds=CFG.early_stopping_rounds,
            verbosity=1,
            num_iterations=1200,
        )
    # TODO: more models
    return model


def round_half_up(n):
    # Multiply by 2, round to nearest whole number, divide by 2 -> 四舍五入
    return np.round(n * 2) / 2


def infer(test_df, models):
    X_test = get_all_features(test_df)
    # make sure that we remove the id column after getting the desired targets
    test_id = X_test[["id"]]
    X_test.drop(["id"], axis=1, inplace=True)

    preds = np.zeros(X_test.shape[0])

    for model in models:
        preds += model.predict(X_test) / len(models)
    preds_df = pd.DataFrame(preds, columns=["score"])

    submission = pd.concat([test_id, preds_df], axis=1)
    submission.reset_index(drop=True, inplace=True)

    # adjust the range of submission to fall into bins of 0.5
    submission.loc[:, "score"] = submission["score"].clip(0.0, 6.0)
    submission.loc[:, "score"] = submission["score"].apply(round_half_up)
    submission.loc[:, "score"] = submission["score"].round(1)

    return submission


def load_pretrained_models():
    # iterate through mdoel path and load all models
    models = []
    for fold in range(CFG.n_splits):
        model_filename = os.path.join(MODEL_PATH, f"{CFG.model}_fold_{fold+1}.joblib")
        model = joblib.load(model_filename)
        models.append(model)

    return models


class CFG:
    # generic params
    if_train = True
    random_state = 42
    n_splits = 5
    model = "lgbm"  # ["lgbm", "ctb", "xgb"]

    # lgdb params
    early_stopping_rounds = 100
    logging_period = 100

    # ctb params

    # xgb params


if __name__ == "__main__":
    DATA_PATH = (
        "/Users/kaiqu/kaggle-datasets/linking-writing-processes-to-writing-quality/"
    )
    MODEL_PATH = "../SavedModels/"

    df = pd.read_csv(f"{DATA_PATH}train_logs.csv")
    df_score = pd.read_csv(f"{DATA_PATH}train_scores.csv")
    df_test = pd.read_csv(f"{DATA_PATH}test_logs.csv")

    if CFG.if_train:
        train(df, df_score)

    models = load_pretrained_models()
    # infer(df_test, models)
